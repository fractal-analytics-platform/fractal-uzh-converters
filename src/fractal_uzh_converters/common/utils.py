"""Common utilities for fractal UZH converters."""

import logging
from typing import Protocol, TypeVar

from ome_zarr_converters_tools import (
    AcquisitionOptions,
    ConverterOptions,
    TiledImage,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

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
