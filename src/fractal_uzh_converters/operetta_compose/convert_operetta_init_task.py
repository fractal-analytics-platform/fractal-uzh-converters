"""Convert Operetta Compose datasets to OME-Zarr."""

import logging
from pathlib import Path

from ome_zarr_converters_tools import (
    AcquisitionOptions,
    ConverterOptions,
    OverwriteMode,
    TiledImage,
    setup_images_for_conversion,
)
from pydantic import BaseModel, Field, model_validator, validate_call

logger = logging.getLogger(__name__)


class OperettaAcquisitionModel(BaseModel):
    """Acquisition metadata.

    Attributes:
        path: Path to the acquisition directory.
            For operetta, this should include a 'data/' directory with the tiff files
            and a metadata.ome.xml file.
        plate_name: Optional custom name for the plate. If not provided, the name will
            be the acquisition directory name.
        acquisition_id: Acquisition ID,
            used to identify the acquisition in case of multiple acquisitions.
        advanced: Advanced acquisition options.
    """

    path: str
    plate_name: str = ""
    acquisition_id: int = Field(default=0, ge=0)
    advanced: AcquisitionOptions = Field(default_factory=AcquisitionOptions)

    @model_validator(mode="before")
    def set_default_plate_name(cls, values):
        """Set default plate name if not provided."""
        path = values.get("path")
        plate_name = values.get("plate_name")
        if plate_name == "" and path is not None:
            values["plate_name"] = Path(path).name
        return values


default_converter_options = ConverterOptions()


def parse_operetta_metadata(
    acquisition_model: OperettaAcquisitionModel,
    converter_options: ConverterOptions,
) -> list[TiledImage]:
    """Parse the Operetta acquisition metadata and return a list of TiledImage.

    Args:
        acquisition_model (OperettaAcquisitionModel): Acquisition metadata.
        converter_options (ConverterOptions): Converter options.

    Returns:
        list[TiledImage]: List of TiledImage objects.
    """
    return []


@validate_call
def convert_operetta_init_task(
    *,
    # Fractal parameters
    zarr_dir: str,
    # Task parameters
    acquisitions: list[OperettaAcquisitionModel],
    converter_options: ConverterOptions = default_converter_options,
    overwrite: OverwriteMode = OverwriteMode.NO_OVERWRITE,
):
    """Initialize the task to convert a Operetta dataset to OME-Zarr.

    Args:
        zarr_dir (str): Directory to store the Zarr files.
        acquisitions (list[OperettaAcquisitionModel]): List of raw acquisitions to
            convert to OME-Zarr.
        converter_options (ConverterOptions): Advanced converter options.
        overwrite (OverwriteMode): Overwrite mode for existing data.
            - "No Overwrite": Do not overwrite existing data.
            - "Overwrite": Remove and replace existing data.
            - "Extend": Extend existing data without removing it.
            Default is "No Overwrite".
    """
    if not acquisitions:
        raise ValueError("Acquisitions list is empty.")

    # prepare the parallel list of zarr urls
    tiled_images = []
    for acq in acquisitions:
        _tiled_images = parse_operetta_metadata(
            acquisition_model=acq, converter_options=converter_options
        )

        if not _tiled_images:
            logger.warning(f"No images found in {acq.path}")
            continue
        else:
            logger.info(f"Found {len(_tiled_images)} images in acquisition {acq.path}")
        tiled_images.extend(_tiled_images)

    if len(tiled_images) == 0:
        raise ValueError("No images found in any of the provided acquisitions.")
    logger.info(f"Total {len(tiled_images)} images found in all acquisitions.")

    parallelization_list = setup_images_for_conversion(
        tiled_images=tiled_images,
        zarr_dir=zarr_dir,
        converter_options=converter_options,
        collection_type="ImageInPlate",
        overwrite_mode=overwrite,
        ngff_version=converter_options.omezarr_options.ngff_version,
    )
    logger.info(
        f"Prepared parallelization list with {len(parallelization_list)} items."
    )
    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=convert_operetta_init_task, logger_name=logger.name)
