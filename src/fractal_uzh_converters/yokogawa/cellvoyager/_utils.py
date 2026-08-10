"""Acquisition model and parser entry point for Yokogawa CellVoyager data.

The parsing itself is shared with CQ3K — see `yokogawa/_parse.py`.
"""

from typing import Literal

from ome_zarr_converters_tools import ConverterOptions, TiledImage

from fractal_uzh_converters.common import HCSBaseAcquisitionModel
from fractal_uzh_converters.yokogawa._parse import parse_yokogawa_metadata

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
        # The converter does not (yet) split a CellVoyager acquisition by
        # `bts:ZImageProcessing`: an acquisition carrying one lands in a single
        # unsuffixed plate, as it always has.
        split_z_processing=False,
        filename_transform=lambda file_name: _replace_extension(
            file_name, acquisition_model.image_extension
        ),
    )
