"""Convert MD ImageXpress HCS.ai datasets to OME-Zarr."""

import logging

from ome_zarr_converters_tools import (
    ConverterOptions,
    OverwriteMode,
    setup_images_for_conversion,
)
from pydantic import validate_call

from fractal_uzh_converters.common import (
    log_converter_warnings,
    parse_acquisitions,
)
from fractal_uzh_converters.imagexpress_hcs._utils import (
    MDImageXpressHCSaiAcquisitionModel,
    parse_md_metadata,
)

logger = logging.getLogger("convert_imagexpress_hcs_task")


default_converter_options = ConverterOptions()


@validate_call
def convert_imagexpress_hcs_init_task(
    *,
    # Fractal parameters
    zarr_dir: str,
    # Task parameters
    acquisitions: list[MDImageXpressHCSaiAcquisitionModel],
    converter_options: ConverterOptions = default_converter_options,
    overwrite: OverwriteMode = OverwriteMode.NO_OVERWRITE,
):
    """Initialize the task to convert a MD ImageXpress HCS.ai dataset to OME-Zarr.

    Args:
        zarr_dir (str): Directory to store the Zarr files.
        acquisitions (list[MDImageXpressHCSaiAcquisitionModel]): List of raw
        acquisitions to convert to OME-Zarr.
        converter_options (ConverterOptions): Advanced converter options.
        overwrite (OverwriteMode): Overwrite mode for existing data.
            - "No Overwrite": Do not overwrite existing data.
            - "Overwrite": Remove and replace existing data.
            - "Extend": Extend existing data without removing it.
            Default is "No Overwrite".
    """
    # Fractal captures the task's logging output, not its stderr, so the
    # converters' warnings are routed through the `py.warnings` logger.
    log_converter_warnings()

    tiled_images = parse_acquisitions(
        parse_function=parse_md_metadata,
        acquisitions=acquisitions,
        converter_options=converter_options,
    )

    parallelization_list = setup_images_for_conversion(
        tiled_images=tiled_images,
        zarr_dir=zarr_dir,
        converter_options=converter_options,
        collection_type="ImageInPlate",
        overwrite_mode=overwrite,
    )
    logger.info(
        f"Prepared parallelization list with {len(parallelization_list)} items."
    )
    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=convert_imagexpress_hcs_init_task)
