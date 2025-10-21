"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import ConverterCompoundTask

AUTHORS = "Fractal Core Team"
DOCS_LINK = "https://github.com/fractal-analytics-platform/fractal-uzh-converters"
INPUT_MODELS = [
    (
        "fractal_uzh_converters",
        "olympus_scanr/convert_scanr_init_task.py",
        "AcquisitionInputModel",
    ),
    (
        "ome_zarr_converters_tools",
        "_task_common_models.py",
        "AdvancedComputeOptions",
    ),
]

TASK_LIST = [
    ConverterCompoundTask(
        name="Convert Olympus ScanR Plate to OME-Zarr",
        executable_init="olympus_scanr/convert_scanr_init_task.py",
        executable="olympus_scanr/convert_scanr_compute_task.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 12000},
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
        executable="cq3k/convert_cq3k_compute_task.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 12000},
        category="Conversion",
        modality="HCS",
        tags=[
            "Yokogawa",
            "CQ3K",
            "Plate converter",
        ],
        docs_info="file:docs_info/cq3k_task.md",
    )
]
