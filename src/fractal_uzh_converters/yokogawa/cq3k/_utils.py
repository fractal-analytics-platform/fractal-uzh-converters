"""Acquisition model and parser entry point for Yokogawa CQ3K data.

The parsing itself is shared with CellVoyager — see `yokogawa/_parse.py`.
"""

from ome_zarr_converters_tools import (
    AcquisitionOptions,
    ConverterOptions,
    TiledImage,
)
from pydantic import Field

from fractal_uzh_converters.common import HCSBaseAcquisitionModel
from fractal_uzh_converters.yokogawa._parse import parse_yokogawa_metadata
from fractal_uzh_converters.yokogawa._z_processing import ZProcessingSelection

######################################################################
#
# Acquisition Input Model
#
######################################################################


class CQ3KAcquisitionOptions(AcquisitionOptions):
    """Acquisition options for the CQ3K converter."""

    z_processing: ZProcessingSelection | None = Field(
        default=None, title="Z Processing"
    )
    """
    Which Z-image processing outputs to convert, each written as its own plate.
    Leave unset to convert every one the acquisition contains.
    """


class CQ3KAcquisitionModel(HCSBaseAcquisitionModel):
    """Acquisition details for the CQ3K microscope data."""

    advanced: CQ3KAcquisitionOptions = Field(default_factory=CQ3KAcquisitionOptions)
    """
    Advanced acquisition options.
    """


######################################################################
#
# Main metadata parsing function
#
######################################################################


def parse_cq3k_metadata(
    *,
    acquisition_model: CQ3KAcquisitionModel,
    converter_options: ConverterOptions,
) -> list[TiledImage]:
    """Parse CQ3K metadata and return a list of TiledImages.

    Args:
        acquisition_model: Acquisition input model containing path and options.
        converter_options: Converter options for tile processing.

    Returns:
        List of TiledImage objects ready for conversion.
    """
    return parse_yokogawa_metadata(
        acquisition_model=acquisition_model,
        converter_options=converter_options,
        z_selection=acquisition_model.advanced.z_processing,
    )
