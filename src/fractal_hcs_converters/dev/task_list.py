"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import ConverterCompoundTask

AUTHORS = "Lorenzo Cerrone"
INPUT_MODELS = [
    (
        "fractal_hcs_converters",
        "olympus_scanr/convert_scanr_init_task.py",
        "AcquisitionInputModel",
    ),
    (
        "fractal_hcs_converters",
        "olympus_scanr/convert_scanr_init_task.py",
        "AdvancedOptions",
    ),
]
DOCS_LINK = "https://github.com/fractal-analytics-platform/fractal-hcs-converters"

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
    )
]
