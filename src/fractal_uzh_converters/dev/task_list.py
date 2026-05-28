"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import ConverterCompoundTask

AUTHORS = "Fractal Core Team"
DOCS_LINK = "https://fractal-analytics-platform.github.io/fractal-uzh-converters/stable"


TASK_LIST = [
    ConverterCompoundTask(
        name="Convert Evident ScanR Plate to OME-Zarr",
        executable_init="scanr/convert_scanr_init_task.py",
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
        name="Convert Revvity Operetta Plate to OME-Zarr",
        executable_init="operetta/convert_operetta_init_task.py",
        executable="common/image_in_plate_compute_task.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        modality="HCS",
        tags=[
            "Revvity",
            "Operetta",
            "Plate converter",
        ],
        docs_info="file:docs_info/operetta_task.md",
    ),
    ConverterCompoundTask(
        name="Convert MD ImageXpress HCS.ai Plate to OME-Zarr",
        executable_init="imagexpress_hcs/convert_imagexpress_hcs_init_task.py",
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
        docs_info="file:docs_info/imagexpress_hcs_task.md",
    ),
    ConverterCompoundTask(
        name="Convert Plain TIFF HCS Plate to OME-Zarr",
        executable_init="plain_tiff/convert_hcs_tiff_init_task.py",
        executable="common/image_in_plate_compute_task.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        modality="HCS",
        tags=[
            "Plain TIFF",
            "Plate converter",
        ],
    ),
    ConverterCompoundTask(
        name="Convert Plain TIFF Images to OME-Zarr",
        executable_init="plain_tiff/convert_single_tiff_task.py",
        executable="common/single_image_compute_task.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Conversion",
        tags=[
            "Plain TIFF",
            "Single Image Converter",
        ],
    ),
]
