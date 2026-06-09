"""Python API for MD ImageXpress HCS converters."""

from ome_zarr_converters_tools import (
    ConverterOptions,
    OverwriteMode,
    RunnerType,
    exec_compound_task,
)
from ome_zarr_converters_tools.fractal import ImageListUpdateDict

from fractal_uzh_converters.common.image_in_plate_compute_task import (
    image_in_plate_compute_task,
)
from fractal_uzh_converters.imagexpress_hcs.convert_imagexpress_hcs_init_task import (
    convert_imagexpress_hcs_init_task,
)
from fractal_uzh_converters.imagexpress_hcs.utils import (
    MDImageXpressHCSaiAcquisitionModel,
)


def convert_imagexpress_hcs(
    *,
    zarr_dir: str,
    acquisitions: list[MDImageXpressHCSaiAcquisitionModel],
    converter_options: ConverterOptions | None = None,
    overwrite: OverwriteMode = OverwriteMode.NO_OVERWRITE,
    runner: RunnerType | None = None,
) -> list[ImageListUpdateDict]:
    """Convert an MD ImageXpress HCS dataset to OME-Zarr.

    Args:
        zarr_dir (str): Directory to store the Zarr files.
        acquisitions (list[MDImageXpressHCSaiAcquisitionModel]): List of raw
            acquisitions to convert to OME-Zarr.
        converter_options (ConverterOptions | None): Advanced converter options.
        overwrite (OverwriteMode): Overwrite mode for existing data.
            - "No Overwrite": Do not overwrite existing data.
            - "Overwrite": Remove and replace existing data.
            - "Extend": Extend existing data without removing it.
            Default is "No Overwrite".
        runner (RunnerType | None): Execution strategy for compute tasks.
            Use SequentialRunner (default), ThreadedRunner, or MultiprocessingRunner.

    Returns:
        list[ImageListUpdateDict]: List of image list update dicts for the converted
            Zarr images.
    """
    converter_options = converter_options or ConverterOptions()
    init_task_kwargs = {
        "zarr_dir": zarr_dir,
        "acquisitions": acquisitions,
        "converter_options": converter_options,
        "overwrite": overwrite,
    }
    return exec_compound_task(
        init_task_fn=convert_imagexpress_hcs_init_task,
        compute_task_fn=image_in_plate_compute_task,
        init_task_kwargs=init_task_kwargs,
        runner=runner,
    )
