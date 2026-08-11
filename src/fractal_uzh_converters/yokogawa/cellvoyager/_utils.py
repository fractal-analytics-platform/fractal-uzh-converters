"""Acquisition model and parser entry point for Yokogawa CellVoyager data.

The parsing itself is shared with CQ3K — see `yokogawa/_parse.py`.
"""

from typing import Literal

from ome_zarr_converters_tools import (
    AcquisitionOptions,
    ConverterOptions,
    TiledImage,
)
from pydantic import Field

from fractal_uzh_converters.common import HCSBaseAcquisitionModel
from fractal_uzh_converters.yokogawa._parse import parse_yokogawa_metadata
from fractal_uzh_converters.yokogawa._z_processing import YokogawaZProcessingSelection

######################################################################
#
# Acquisition Input Model
#
######################################################################


class CellVoyagerAcquisitionModel(HCSBaseAcquisitionModel):
    """Acquisition details for the CellVoyager microscope data."""

    image_extension: Literal[".tif", ".png"] = ".tif"
    """
    File extension of the actual image files.
    The metadata (.mlf) always references '.tif', but the actual files
    may be '.png' or '.tif'. Select the extension matching your data.
    """
    z_processing: YokogawaZProcessingSelection | None = Field(
        default=None, title="Z Processing"
    )
    """
    Which Z-image processing outputs to convert, each written as its own plate.
    Leave unset to convert every one the acquisition contains.
    """
    advanced: AcquisitionOptions = Field(default_factory=AcquisitionOptions)
    """
    Advanced acquisition options.
    """


######################################################################
#
# Main metadata parsing function
#
######################################################################


def _replace_extension(filename: str, new_extension: str) -> str:
    """Replace the .tif extension in the metadata with the actual extension."""
    if filename.endswith(".tif"):
        return filename[: -len(".tif")] + new_extension
    return filename


def parse_cellvoyager_metadata(
    *,
    acquisition_model: CellVoyagerAcquisitionModel,
    converter_options: ConverterOptions,
) -> list[TiledImage]:
    """Parse CellVoyager metadata and return a list of TiledImages.

    Args:
        acquisition_model: Acquisition input model containing path and options.
        converter_options: Converter options for tile processing.

    Returns:
        List of TiledImage objects ready for conversion.
    """
    return parse_yokogawa_metadata(
        acquisition_model=acquisition_model,
        converter_options=converter_options,
        z_selection=acquisition_model.z_processing,
        filename_transform=lambda file_name: _replace_extension(
            file_name, acquisition_model.image_extension
        ),
    )
