"""A collection of fractal tasks to convert HCS Plates to OME-Zarr"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fractal-uzh-converters")
except PackageNotFoundError:
    __version__ = "uninstalled"

from fractal_uzh_converters.custom_tiff import (
    HcsTiffAcquisitionModel,
    SingleTiffAcquisitionModel,
    convert_hcs_tiff,
    convert_single_tiff,
)
from fractal_uzh_converters.imagexpress_hcs import (
    MDAcquisitionOptions,
    MDImageXpressHCSaiAcquisitionModel,
    convert_imagexpress_hcs,
)
from fractal_uzh_converters.operetta import (
    OperettaAcquisitionModel,
    convert_operetta,
)
from fractal_uzh_converters.scanr import (
    ScanRAcquisitionModel,
    convert_scanr,
)
from fractal_uzh_converters.yokogawa import (
    CellVoyagerAcquisitionModel,
    CQ3KAcquisitionModel,
    convert_cellvoyager,
    convert_cq3k,
)

__all__ = [
    # Models
    "CQ3KAcquisitionModel",
    "CellVoyagerAcquisitionModel",
    "HcsTiffAcquisitionModel",
    "MDAcquisitionOptions",
    "MDImageXpressHCSaiAcquisitionModel",
    "OperettaAcquisitionModel",
    "ScanRAcquisitionModel",
    "SingleTiffAcquisitionModel",
    # Tasks
    "convert_cellvoyager",
    "convert_cq3k",
    "convert_hcs_tiff",
    "convert_imagexpress_hcs",
    "convert_operetta",
    "convert_scanr",
    "convert_single_tiff",
]
