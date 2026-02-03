"""Generic compute task for plate and tiff based acquisitions."""

import logging
import time

from ome_zarr_converters_tools import (
    ConvertParallelInitArgs,
    DefaultImageLoader,
    ImageInPlate,
    ImageListUpdateDict,
    generic_compute_task,
)
from pydantic import validate_call

logger = logging.getLogger(__name__)


@validate_call
def image_in_plate_compute_task(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: ConvertParallelInitArgs,
) -> ImageListUpdateDict:
    """Create a single OME-Zarr image in a OME-Zarr plate.

    Args:
        zarr_url (str): URL to the OME-Zarr file.
        init_args (ConvertParallelInitArgs): Arguments for the compute task.
    """
    timer = time.time()
    img_list_update = generic_compute_task(
        zarr_url=zarr_url,
        init_args=init_args,
        collection_type=ImageInPlate,
        image_loader_type=DefaultImageLoader,
    )
    zarr_output = img_list_update["image_list_updates"][0]["zarr_url"]
    run_time = time.time() - timer
    logger.info(f"Succesfully converted: {zarr_output}, in {run_time:.2f}[s]")
    return img_list_update


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=image_in_plate_compute_task, logger_name=logger.name)
