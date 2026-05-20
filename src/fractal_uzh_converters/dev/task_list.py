"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import ConverterCompoundTask

AUTHORS = "Fractal Core Team"
DOCS_LINK = "https://fractal-analytics-platform.github.io/fractal-uzh-converters/stable"


TASK_LIST = [
    ConverterCompoundTask(
        name="Convert Evident ScanR Plate to OME-Zarr",
        executable_init="evident_scanr/convert_scanr_init_task.py",
        executable="common/image_in_plate_compute_task.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        modality="HCS",
        tags=[
            "Evident",
            "ScanR",
            "Plate converter",
        ],
        docs_info="file:docs_info/scanr_task.md",
    ),
    ConverterCompoundTask(
        name="Convert Yokogawa CellVoyager Plate to OME-Zarr",
        executable_init="cellvoyager/convert_cellvoyager_init_task.py",
        executable="common/image_in_plate_compute_task.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        modality="HCS",
        tags=[
            "Yokogawa",
            "CellVoyager",
            "Plate converter",
        ],
        docs_info="file:docs_info/cellvoyager_task.md",
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
            "PerkinElmer",
            "Operetta",
            "Plate converter",
        ],
        docs_info="file:docs_info/operetta_task.md",
    ),
    ConverterCompoundTask(
        name="Convert MD ImageXpress HCS.ai Plate to OME-Zarr",
        executable_init="md_imagexpress_hcsai/convert_md_init_task.py",
        executable="common/image_in_plate_compute_task.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        modality="HCS",
        tags=[
            "Molecular Devices",
            "ImageXpress HCS.ai",
            "Plate converter",
        ],
        docs_info="file:docs_info/md_imagexpress_hcsai_task.md",
    ),
]
