"""Custom TIFF module for converting HCS and single-image TIFF data to OME-Zarr."""

from fractal_uzh_converters.custom_tiff.api import convert_hcs_tiff, convert_single_tiff
from fractal_uzh_converters.custom_tiff.convert_hcs_tiff_init_task import (
    convert_hcs_tiff_init_task,
)
from fractal_uzh_converters.custom_tiff.convert_single_tiff_init_task import (
    convert_single_tiff_init_task,
)

__all__ = [
    "convert_hcs_tiff",
    "convert_hcs_tiff_init_task",
    "convert_single_tiff",
    "convert_single_tiff_init_task",
]
