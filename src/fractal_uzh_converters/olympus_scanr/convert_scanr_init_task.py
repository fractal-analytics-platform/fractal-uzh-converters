"""ScanR to OME-Zarr conversion task initialization."""

import logging
from pathlib import Path

from ome_zarr_converters_tools import (
    AdvancedComputeOptions,
    build_parallelization_list,
    initiate_ome_zarr_plates,
)
from pydantic import BaseModel, Field, validate_call

from fractal_uzh_converters.olympus_scanr.utils import parse_scanr_metadata

logger = logging.getLogger(__name__)


class AcquisitionInputModel(BaseModel):
    """Acquisition metadata.

    Attributes:
        path: Path to the acquisition directory.
            For scanr, this should include a 'data/' directory with the tiff files
            and a metadata.ome.xml file.
        plate_name: Optional custom name for the plate. If not provided, the name will
            be the acquisition directory name.
        acquisition_id: Acquisition ID,
            used to identify the acquisition in case of multiple acquisitions.
    """

    path: str
    plate_name: str | None = None
    acquisition_id: int = Field(default=0, ge=0)


class ConvertScanrInitArgs(BaseModel):
    """Arguments for the compute task."""

    tiled_image_pickled_path: str
    advanced_options: AdvancedComputeOptions = Field(
        default_factory=AdvancedComputeOptions
    )


@validate_call
def convert_scanr_init_task(
    *,
    # Fractal parameters
    zarr_dir: str,
    # Task parameters
    acquisitions: list[AcquisitionInputModel],
    overwrite: bool = False,
    advanced_options: AdvancedComputeOptions = AdvancedComputeOptions(),  # noqa: B008
):
    """Initialize the task to convert a ScanR dataset to OME-Zarr.

    Args:
        zarr_urls (list[str]): List of Zarr URLs.
        zarr_dir (str): Directory to store the Zarr files.
        acquisitions (list[AcquisitionInputModel]): List of raw acquisitions to convert
            to OME-Zarr.
        overwrite (bool): Overwrite existing Zarr files.
        advanced_options (AdvancedOptions): Advanced options for the conversion.
    """
    if not acquisitions:
        raise ValueError("No acquisitions provided.")

    zarr_dir_path = Path(zarr_dir)

    if not zarr_dir_path.exists():
        logger.info(f"Creating directory: {zarr_dir_path}")
        zarr_dir_path.mkdir(parents=True)

    # prepare the parallel list of zarr urls
    tiled_images = []
    for acq in acquisitions:
        acq_path = Path(acq.path)
        plate_name = acq_path.stem if acq.plate_name is None else acq.plate_name

        _tiled_images = parse_scanr_metadata(
            acq_path, acq_id=acq.acquisition_id, plate_name=plate_name
        )

        if not _tiled_images:
            logger.warning(f"No images found in {acq_path}")
            continue

        tiled_images.extend(list(_tiled_images.values()))

    if not tiled_images:
        raise ValueError("No images found in the acquisitions.")

    parallelization_list = build_parallelization_list(
        zarr_dir=zarr_dir_path,
        tiled_images=tiled_images,
        overwrite=overwrite,
        advanced_compute_options=advanced_options,
    )
    logger.info(f"Total {len(parallelization_list)} images to convert.")

    initiate_ome_zarr_plates(
        zarr_dir=zarr_dir_path,
        tiled_images=tiled_images,
        overwrite=overwrite,
    )
    logger.info(f"Initialized OME-Zarr Plate at: {zarr_dir_path}")
    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=convert_scanr_init_task, logger_name=logger.name)
