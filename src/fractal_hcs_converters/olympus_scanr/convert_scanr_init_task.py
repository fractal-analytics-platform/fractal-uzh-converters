"""ScanR to OME-Zarr conversion task initialization."""

import logging
import pickle
from pathlib import Path
from typing import Literal

from fractal_converters_tools.omezarr_plate_writers import initiate_ome_zarr_plates
from pydantic import BaseModel, Field, validate_call

from fractal_hcs_converters.olympus_scanr.utils import parse_scanr_metadata

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


class AdvancedOptions(BaseModel):
    """Advanced options for the conversion.

    Attributes:
        tiling_mode (Literal["auto", "grid", "free", "none"]): Specify the tiling mode.
            "auto" will automatically determine the tiling mode.
            "grid" if the input data is a grid, it will be tiled using snap-to-grid.
            "free" will remove any overlap between tiles using a snap-to-corner
            approach.
        swap_xy (bool): Swap x and y axes coordinates in the metadata. This is sometimes
            necessary to ensure correct image tiling and registration.
        invert_x (bool): Invert x axis coordinates in the metadata. This is
            sometimes necessary to ensure correct image tiling and registration.
        invert_y (bool): Invert y axis coordinates in the metadata. This is
            sometimes necessary to ensure correct image tiling and registration.

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
    zarr_dir: str,
    # Task parameters
    acquisitions: list[AcquisitionInputModel],
    overwrite: bool = False,
    advanced_options: AdvancedOptions = AdvancedOptions(),  # noqa: B008
):
    """Initialize the task to convert a ScanR dataset to OME-Zarr.

    Args:
        zarr_dir (str): Directory to store the Zarr files.
        acquisitions (list[AcquisitionInputModel]): List of raw acquisitions to convert
            to OME-Zarr.
        overwrite (bool): Overwrite existing Zarr files.
        advanced_options (AdvancedOptions): Advanced options for the conversion.
    """
    if not acquisitions:
        raise ValueError("No acquisitions provided.")

    zarr_dir = Path(zarr_dir)

    if not zarr_dir.exists():
        logger.info(f"Creating directory: {zarr_dir}")
        zarr_dir.mkdir(parents=True)

    # prepare the parallel list of zarr urls
    tiled_images, parallelization_list = [], []
    for acq in acquisitions:
        acq_path = Path(acq.path)
        plate_name = acq_path.stem if acq.plate_name is None else acq.plate_name

        _tiled_images = parse_scanr_metadata(
            acq_path, acq_id=acq.acquisition_id, plate_name=plate_name
        )

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
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=convert_scanr_init_task, logger_name=logger.name)
