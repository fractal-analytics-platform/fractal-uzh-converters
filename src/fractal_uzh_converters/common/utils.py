"""Common utilities for fractal UZH converters."""

import logging
from typing import Protocol, TypeVar

import polars
from ome_zarr_converters_tools import (
    AcquisitionOptions,
    AttributeType,
    ConverterOptions,
    TiledImage,
)
from pydantic import BaseModel, Field

logger = logging.getLogger("common_converters_compute_task")

STANDARD_ROWS_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class BaseAcquisitionModel(BaseModel):
    """Base model for acquisitions.

    Attributes:
        path: Path to the acquisition directory.
            Should contain MeasurementData.mlf and MeasurementDetail.mrf files.
        plate_name: Optional custom name for the plate. If not provided, the name will
            be the acquisition directory name.
        acquisition_id: Acquisition ID,
            used to identify the acquisition in case of multiple acquisitions.
        advanced: Advanced acquisition options.
    """

    path: str
    plate_name: str | None = None
    acquisition_id: int = Field(default=0, ge=0)
    advanced: AcquisitionOptions = Field(default_factory=AcquisitionOptions)

    @property
    def normalized_plate_name(self) -> str:
        """Get the normalized plate name."""
        if self.plate_name is not None:
            return self.plate_name
        name = self.path.rstrip("/").split("/")[-1]
        return name

    def get_condition_table(self) -> polars.DataFrame | None:
        """Get the path to the condition table if it exists."""
        if self.advanced.condition_table_path is not None:
            try:
                return polars.read_csv(self.advanced.condition_table_path)
            except Exception as e:
                raise ValueError(
                    "Failed to read condition table at "
                    f"{self.advanced.condition_table_path}: {e}"
                ) from e
        return None


AcquisitionModelType = TypeVar(
    "AcquisitionModelType", bound=BaseAcquisitionModel, contravariant=True
)


class ParserProtocol(Protocol[AcquisitionModelType]):
    """Protocol for acquisition metadata parser."""

    def __call__(
        self,
        *,
        acquisition_model: AcquisitionModelType,
        converter_options: ConverterOptions,
    ) -> list[TiledImage]:
        """Parse the acquisition metadata and return tiled images."""
        ...


def parse_acquisitions(
    *,
    parse_function: ParserProtocol[AcquisitionModelType],
    acquisitions: list[AcquisitionModelType],
    converter_options: ConverterOptions,
) -> list[TiledImage]:
    """Parse the acquisitions metadata and return tiled images.

    Args:
        parse_function (Callable): Function to parse the acquisition metadata
            and return tiled images.
        acquisitions (list[AcquisitionModelType]): List of acquisition models.
        converter_options (ConverterOptions): Converter options.

    Returns:
        list[TiledImage]: List of tiled images.
    """
    if not acquisitions:
        raise ValueError("Acquisitions list is empty.")

    # prepare the parallel list of zarr urls
    tiled_images = []
    for acq in acquisitions:
        _tiled_images = parse_function(
            acquisition_model=acq,
            converter_options=converter_options,
        )

        if not _tiled_images:
            logger.warning(f"No images found in {acq.path}")
            continue
        else:
            logger.info(f"Found {len(_tiled_images)} images in acquisition {acq.path}")
        tiled_images.extend(_tiled_images)

    if len(tiled_images) == 0:
        raise ValueError("No images found in any of the provided acquisitions.")
    logger.info(f"Total {len(tiled_images)} images found in all acquisitions.")
    return tiled_images


def get_attributes_from_condition_table(
    condition_table: polars.DataFrame | None,
    row: str,
    column: int,
    acquisition: int = 0,
) -> dict[str, AttributeType]:
    """Get the attributes from the condition table."""
    if condition_table is None:
        return {}
    columns = condition_table.columns
    columns_lower = [col.lower() for col in columns]
    if "row" not in columns_lower:
        raise ValueError("Condition table must contain a 'row' column.")
    row_col_name = columns[columns_lower.index("row")]

    if "column" in columns_lower:
        column_col_name = columns[columns_lower.index("column")]
    elif "col" in columns_lower:
        column_col_name = columns[columns_lower.index("col")]
    else:
        raise ValueError("Condition table must contain a 'column' or 'col' column.")

    filtered = condition_table.filter(
        (polars.col(row_col_name) == row) & (polars.col(column_col_name) == column)
    )
    if "acquisition" in columns_lower:
        acquisition_col_name = columns[columns_lower.index("acquisition")]
        filtered = filtered.filter(polars.col(acquisition_col_name) == acquisition)
    if filtered.is_empty():
        logger.warning(
            f"No matching entry found in condition table "
            f"for {row}{column} (acquisition {acquisition})"
        )
        return {}
    filtered_dict = filtered.to_dict(as_series=False)
    attributes = {}
    for key, value in filtered_dict.items():
        if key in ["row", "column", "acquisition"]:
            continue
        if all(isinstance(v, (str, type(None))) for v in value):
            formatted_value = [v if v is None else v.strip() for v in value]
            # Replace common placeholder values with None
            formatted_value = [
                None if v in ["", "Na", "NA", "N/A"] else v for v in formatted_value
            ]
            attributes[key] = formatted_value
        elif all(isinstance(v, (int, float, bool, type(None))) for v in value):
            attributes[key] = value
        else:
            types_found = {type(v).__name__ for v in value}
            raise ValueError(
                f"Condition table column '{key}' must contain either all strings"
                f", bools, or all numbers, but found types: {types_found}"
            )

    return attributes
