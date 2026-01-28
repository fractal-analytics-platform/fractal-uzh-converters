"""Olympus ScanR module for converting Olympus ScanR data to Fractal HCS format."""

from fractal_uzh_converters.operetta_compose.convert_operetta_compute_task import (
    convert_operetta_compute_task,
)
from fractal_uzh_converters.operetta_compose.convert_operetta_init_task import (
    convert_operetta_init_task,
)

__all__ = [
    "convert_operetta_compute_task",
    "convert_operetta_init_task",
]
