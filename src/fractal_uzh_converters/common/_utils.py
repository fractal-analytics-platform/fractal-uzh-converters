"""Common utilities for fractal UZH converters."""

import logging
from typing import Protocol, TypeVar

import polars
from fsspec.core import split_protocol
from ome_zarr_converters_tools import (
    AcquisitionOptions,
    AttributeType,
    ConverterOptions,
    ImageInPlate,
    TiledImage,
    join_url_paths,
)
from pydantic import BaseModel, Field

logger = logging.getLogger("common_converters_compute_task")

STANDARD_ROWS_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _path_basename(path: str) -> str:
    """Last path component, robust to fwd/back slashes and URL protocols.

    Uses ``fsspec`` rather than ``pathlib.Path`` so that remote URLs (e.g.
    ``s3://bucket/plate``) are not mangled, while still handling Windows
    backslash separators.
    """
    _, stripped = split_protocol(path)
    return stripped.replace("\\", "/").rstrip("/").split("/")[-1]


class BaseAcquisitionModel(BaseModel):
    """Shared base model for HCS and single-image acquisitions."""

    path: str
    """
    Path to the acquisition directory or file.
    """
    advanced: AcquisitionOptions = Field(default_factory=AcquisitionOptions)
    """
    Advanced acquisition options.
    """

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


class HCSBaseAcquisitionModel(BaseAcquisitionModel):
    """Base model for HCS (plate) acquisitions.

    Extends ``BaseAcquisitionModel`` with plate-specific fields: ``plate_name``
    and ``acquisition_id``.
    """

    plate_name: str | None = None
    """
    Optional custom name for the plate. If not provided, the name will be the
    acquisition directory name.
    """
    acquisition_id: int = Field(default=0, ge=0)
    """
    Acquisition ID, used to identify the acquisition in case of multiple acquisitions.
    """

    @property
    def normalized_plate_name(self) -> str:
        """Get the normalized plate name."""
        if self.plate_name is not None:
            return self.plate_name
        name = _path_basename(self.path)
        return name


class SingleBaseAcquisitionModel(BaseAcquisitionModel):
    """Base model for single-image acquisitions.

    Extends ``BaseAcquisitionModel`` with ``image_name`` for controlling the
    output OME-Zarr image name.
    """

    image_name: str | None = None
    """
    Optional custom name for the output OME-Zarr image. If not provided, the
    name will be derived from the acquisition directory or file name.
    """

    @property
    def normalized_image_name(self) -> str:
        """Get the normalized image name."""
        if self.image_name is not None:
            return self.image_name
        name = _path_basename(self.path)
        return name


AcquisitionModelType = TypeVar(
    "AcquisitionModelType", bound=BaseAcquisitionModel, contravariant=True
)


class ParserProtocol(Protocol[AcquisitionModelType]):
    """Protocol for acquisition metadata parser.

    Accepts any ``BaseAcquisitionModel``-derived type, including both HCS and
    single-image acquisition models.
    """

    def __call__(
        self,
        *,
        acquisition_model: AcquisitionModelType,
        converter_options: ConverterOptions,
    ) -> list[TiledImage]:
        """Parse the acquisition metadata and return tiled images."""
        ...


def parse_acquisitions_grouped(
    *,
    parse_function: ParserProtocol[AcquisitionModelType],
    acquisitions: list[AcquisitionModelType],
    converter_options: ConverterOptions,
) -> list[tuple[AcquisitionModelType, list[TiledImage]]]:
    """Parse the acquisitions metadata, keeping each image with its acquisition.

    Same work as `parse_acquisitions`, but without flattening. Callers that need
    to act on the raw acquisition directory *after* the images have been set up
    — copying the vendor metadata into the plate, say — need the association,
    and it appears nowhere on a `TiledImage`.

    Args:
        parse_function (Callable): Function to parse the acquisition metadata
            and return tiled images.
        acquisitions (list[AcquisitionModelType]): List of acquisition models.
        converter_options (ConverterOptions): Converter options.

    Returns:
        list[tuple[AcquisitionModelType, list[TiledImage]]]: One entry per
            acquisition that yielded images, in input order. Acquisitions that
            yielded none are dropped, so no entry has an empty image list.
    """
    if not acquisitions:
        raise ValueError("Acquisitions list is empty.")

    # prepare the parallel list of zarr urls
    grouped: list[tuple[AcquisitionModelType, list[TiledImage]]] = []
    total = 0
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
        grouped.append((acq, _tiled_images))
        total += len(_tiled_images)

    if total == 0:
        raise ValueError("No images found in any of the provided acquisitions.")
    logger.info(f"Total {total} images found in all acquisitions.")
    return grouped


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
    grouped = parse_acquisitions_grouped(
        parse_function=parse_function,
        acquisitions=acquisitions,
        converter_options=converter_options,
    )
    return [tiled_image for _, images in grouped for tiled_image in images]


def plate_urls_for_images(
    *, zarr_dir: str, tiled_images: list[TiledImage]
) -> list[str]:
    """Absolute URLs of the distinct plates `tiled_images` were written into.

    `ImageInPlate.plate_path()` is relative to `zarr_dir`, so it is joined here
    the same way the library's own plate setup does it. One acquisition can
    produce several plates — a Yokogawa acquisition yields one per projection
    algorithm — and each of them gets its own copy of the metadata.

    Args:
        zarr_dir: Directory the plates were written into.
        tiled_images: Images of a single acquisition.

    Returns:
        Sorted, deduplicated plate URLs. Empty for collections that are not
        plates, which carry no `plate_path`.
    """
    plate_paths = {
        image.collection.plate_path()
        for image in tiled_images
        if isinstance(image.collection, ImageInPlate)
    }
    return sorted(join_url_paths(zarr_dir, path) for path in plate_paths)


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
    skip_keys = {row_col_name, column_col_name}
    if "acquisition" in columns_lower:
        acquisition_col_name = columns[columns_lower.index("acquisition")]
        filtered = filtered.filter(polars.col(acquisition_col_name) == acquisition)
        skip_keys.add(acquisition_col_name)
    if filtered.is_empty():
        logger.warning(
            f"No matching entry found in condition table "
            f"for row:{row} / column:{column} / acquisition:{acquisition}"
        )
        return {}
    filtered_dict = filtered.to_dict(as_series=False)
    attributes = {}
    for key, value in filtered_dict.items():
        if key in skip_keys:
            continue
        if all(isinstance(v, str | type(None)) for v in value):
            formatted_value = [v if v is None else v.strip() for v in value]
            # Replace common placeholder values with None
            formatted_value = [
                None if v in ["", "Na", "NA", "N/A"] else v for v in formatted_value
            ]
            attributes[key] = formatted_value
        elif all(isinstance(v, int | float | bool | type(None)) for v in value):
            attributes[key] = value
        else:
            types_found = {type(v).__name__ for v in value}
            raise ValueError(
                f"Condition table column '{key}' must contain either all strings"
                f", bools, or all numbers, but found types: {types_found}"
            )

    return attributes
