"""ScanR to OME-Zarr conversion task initialization."""

import logging
import pickle
from pathlib import Path
from typing import Literal, Optional

from fractal_converters_tools.omezarr_plate_writers import initiate_ome_zarr_plates
from pydantic import BaseModel, Field, validate_call

from fractal_hcs_converters.olympus_scanr.utils import parse_scanr_metadata

logger = logging.getLogger(__name__)


class AcquisitionInputModel(BaseModel):
    """Acquisition metadata."""

    path: str
    plate_name: str | None = None


class AdvancedOptions(BaseModel):
    """Advanced options for the conversion.

    Args:
        tiling_mode (Literal["auto", "grid", "free", "none"]): Specify the tiling mode.
            "auto" will automatically determine the tiling mode.
            "grid" if the input data is a grid, it will be tiled using snap-to-grid.
            "free" will remove any overlap between tiles using a snap-to-corner
            approach.
        swap_xy (bool): Swap x and y axes.
        invert_x (bool): Invert x axis.
        invert_y (bool): Invert y axis.

    """

    tiling_mode: Literal["auto", "grid", "free", "none"] = "auto"
    swap_xy: bool = False
    invert_x: bool = False
    invert_y: bool = False


class ConvertScanrInitArgs(BaseModel):
    """Arguments for the compute task."""

    tiled_image_pickled_path: str
    advanced_options: AdvancedOptions = Field(default_factory=AdvancedOptions)


@validate_call
def convert_scanr_init_task(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Task parameters
    list_acq: list[AcquisitionInputModel],
    plate_name: Optional[str] = None,
    overwrite: bool = False,
    advanced_options: AdvancedOptions = Field(default_factory=AdvancedOptions),  # noqa: B008
):
    """Initialize the task to convert a ScanR dataset to OME-Zarr.

    Args:
        zarr_urls (list[str]): List of Zarr URLs.
        zarr_dir (str): Directory to store the Zarr files.
        list_acq (list[AcquisitionInputModel]): List of acquisitions to convert.
        plate_name (Optional[str]): Name of the plate (e.g. experiment_2.zarr).
        overwrite (bool): Overwrite existing Zarr files.
        advanced_options (AdvancedOptions): Advanced options for the conversion.
    """
    if not list_acq:
        raise ValueError("No acquisitions provided.")

    zarr_dir = Path(zarr_dir)

    if not zarr_dir.exists():
        logger.info(f"Creating directory: {zarr_dir}")
        zarr_dir.mkdir(parents=True)

    if plate_name is None:
        plate_name = Path(list_acq[0].path).stem
        logger.info(
            f"No plate name provided. Using the first acquisition name {plate_name}"
        )

    # prepare the parallel list of zarr urls
    tiled_images, parallelization_list = [], []
    for acq in list_acq:
        acq_path = Path(acq.path)
        plate_name = acq.plate_name
        if plate_name is None:
            plate_name = acq_path.stem

        _tiled_images = parse_scanr_metadata(acq_path, acq_id=0, plate_name=plate_name)

        if not _tiled_images:
            logger.warning(f"No images found in {acq_path}")
            continue

        logger.info(f"Found {len(_tiled_images)} images in {acq_path})")
        for tile_id, tiled_image in _tiled_images.items():
            # pickle the tiled_image
            tile_id_pickle_path = (
                zarr_dir
                / f"_tmp_{tiled_image.path_builder.plate_path}"
                / f"{tile_id}.pickle"
            )
            tile_id_pickle_path.parent.mkdir(parents=True, exist_ok=True)

            with open(tile_id_pickle_path, "wb") as f:
                pickle.dump(tiled_image, f)

            parallelization_list.append(
                {
                    "zarr_url": str(zarr_dir),
                    "init_args": ConvertScanrInitArgs(
                        tiled_image_pickled_path=str(tile_id_pickle_path),
                        advanced_options=advanced_options,
                    ).model_dump(),
                }
            )
        tiled_images.extend(list(_tiled_images.values()))

    if not tiled_images:
        raise ValueError("No images found in the acquisitions.")

    logger.info(f"Total {len(parallelization_list)} images to convert.")

    initiate_ome_zarr_plates(
        store=zarr_dir,
        tiled_images=tiled_images,
        overwrite=overwrite,
    )
    logger.info(f"Initialized OME-Zarr Plate at: {zarr_dir}")
    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=convert_scanr_init_task, logger_name=logger.name)
