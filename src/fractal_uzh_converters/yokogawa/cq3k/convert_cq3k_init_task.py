"""Convert Yokogawa CQ3K acquisitions to OME-Zarr format."""

import logging

from ome_zarr_converters_tools import (
    ConverterOptions,
    OverwriteMode,
    setup_images_for_conversion,
)
from pydantic import validate_call

from fractal_uzh_converters.common import (
    parse_acquisitions_grouped,
    plate_urls_for_images,
)
from fractal_uzh_converters.yokogawa._source_metadata import copy_source_metadata
from fractal_uzh_converters.yokogawa.cq3k._utils import (
    CQ3KAcquisitionModel,
    parse_cq3k_metadata,
)

logger = logging.getLogger("convert_cq3k_task")

default_converter_options = ConverterOptions()


@validate_call
def convert_cq3k_init_task(
    *,
    # Fractal parameters
    zarr_dir: str,
    # Task parameters
    acquisitions: list[CQ3KAcquisitionModel],
    converter_options: ConverterOptions = default_converter_options,
    overwrite: OverwriteMode = OverwriteMode.NO_OVERWRITE,
):
    """Initialize the task to convert a CQ3K dataset to OME-Zarr.

    Args:
        zarr_dir (str): Directory to store the Zarr files.
        acquisitions (list[AcquisitionInputModel]): List of raw acquisitions to convert
            to OME-Zarr.
        converter_options (ConverterOptions): Advanced converter options.
        overwrite (OverwriteMode): Overwrite mode for existing data.
            - "No Overwrite": Do not overwrite existing data.
            - "Overwrite": Remove and replace existing data.
            - "Extend": Extend existing data without removing it.
            Default is "No Overwrite".
    """
    grouped = parse_acquisitions_grouped(
        parse_function=parse_cq3k_metadata,
        acquisitions=acquisitions,
        converter_options=converter_options,
    )
    tiled_images = [image for _, images in grouped for image in images]

    parallelization_list = setup_images_for_conversion(
        tiled_images=tiled_images,
        zarr_dir=zarr_dir,
        converter_options=converter_options,
        collection_type="ImageInPlate",
        overwrite_mode=overwrite,
    )

    # After the plates exist: under `OverwriteMode.OVERWRITE` the setup above
    # recreates them from scratch, so anything copied earlier would be wiped.
    for acquisition, images in grouped:
        copy_source_metadata(
            acquisition_dir=acquisition.path,
            plate_urls=plate_urls_for_images(zarr_dir=zarr_dir, tiled_images=images),
            acquisition_id=acquisition.acquisition_id,
        )

    logger.info(
        f"Prepared parallelization list with {len(parallelization_list)} items."
    )
    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=convert_cq3k_init_task)
