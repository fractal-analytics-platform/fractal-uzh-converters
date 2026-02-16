"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import ConverterCompoundTask
from ome_zarr_converters_tools import converters_tools_models

AUTHORS = "Fractal Core Team"
DOCS_LINK = "https://fractal-analytics-platform.github.io/fractal-uzh-converters/stable"
INPUT_MODELS = [
    (
        "fractal_uzh_converters",
        "common/utils.py",
        "BaseAcquisitionModel",
    ),
    (
        "fractal_uzh_converters",
        "operetta/utils.py",
        "OperettaAcquisitionModel",
    ),
    (
        "fractal_uzh_converters",
        "olympus_scanr/utils.py",
        "ScanRAcquisitionModel",
    ),
    (
        "fractal_uzh_converters",
        "cq3k/utils.py",
        "CQ3KAcquisitionModel",
    ),
]
INPUT_MODELS += converters_tools_models()

TASK_LIST = [
    ConverterCompoundTask(
        name="Convert Olympus ScanR Plate to OME-Zarr",
        executable_init="olympus_scanr/convert_scanr_init_task.py",
        executable="common/image_in_plate_compute_task.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        modality="HCS",
        tags=[
            "Olympus",
            "ScanR",
            "Plate converter",
        ],
        docs_info="file:docs_info/scanr_task.md",
    ),
    ConverterCompoundTask(
        name="Convert Yokogawa CQ3K Plate to OME-Zarr",
        executable_init="cq3k/convert_cq3k_init_task.py",
        executable="common/image_in_plate_compute_task.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        modality="HCS",
        tags=[
            "Yokogawa",
            "CQ3K",
            "Plate converter",
        ],
        docs_info="file:docs_info/cq3k_task.md",
    ),
    ConverterCompoundTask(
        name="Convert Operetta Plate to OME-Zarr",
        executable_init="operetta/convert_operetta_init_task.py",
        executable="common/image_in_plate_compute_task.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        modality="HCS",
        tags=[
            "Operetta",
            "Plate converter",
        ],
        docs_info="file:docs_info/operetta_task.md",
    ),
]
