"""Plain TIFF module for converting HCS and single-image TIFF data to OME-Zarr."""

import fractal_uzh_converters.plain_tiff._setup  # noqa: F401
from fractal_uzh_converters.plain_tiff.convert_hcs_tiff_init_task import (
    convert_hcs_tiff_init_task,
)
from fractal_uzh_converters.plain_tiff.convert_single_tiff_task import (
    convert_single_tiff_init_task,
)

__all__ = [
    "convert_hcs_tiff_init_task",
    "convert_single_tiff_init_task",
]
