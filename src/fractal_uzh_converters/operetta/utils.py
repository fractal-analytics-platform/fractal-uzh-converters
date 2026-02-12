"""Utility functions for Operetta data."""

import logging
from typing import Any

import numpy as np
import xmltodict
from ome_zarr_converters_tools import (
    AcquisitionDetails,
    AttributeType,
    ChannelInfo,
    ConverterOptions,
    DataTypeEnum,
    DefaultImageLoader,
    ImageInPlate,
    Tile,
    TiledImage,
    default_axes_builder,
    join_url_paths,
    tiles_aggregation_pipeline,
)
from pydantic import BaseModel, Field, field_validator

from fractal_uzh_converters.common import (
    STANDARD_ROWS_NAMES,
    BaseAcquisitionModel,
    get_attributes_from_condition_table,
)

logger = logging.getLogger(__name__)


######################################################################
#
# Acquisition Input Model
#
######################################################################


class OperettaAcquisitionModel(BaseAcquisitionModel):
    """Acquisition details for the Operetta microscope data.

    Attributes:
        path: Path to the acquisition directory.
            For Operetta, this should be the base directory of the acquisition
            or the "{acquisition_dir}/Images" directory containing the tiff
            files and metadata.ome.xml file.
        plate_name: Optional custom name for the plate. If not provided, the name will
            be the acquisition directory name.
        acquisition_id: Acquisition ID,
            used to identify the acquisition in case of multiple acquisitions.
        advanced: Advanced acquisition options.
    """

    @field_validator("path", mode="before")
    def validate_path(cls, v) -> str:
        """Make the path more flexible.

        Allow:
         - path/to/acquisition/Images
         - path/to/acquisition/
        """
        v = v.rstrip("/")
        if v.endswith("/Images"):
            return v[: -len("/Images")]
        return v


######################################################################
#
# Pydantic models for parsing Operetta metadata
#
######################################################################


class MeasureWithUnit(BaseModel):
    """Measurement with unit."""

    unit: str = Field(..., alias="Unit")
    value: float = Field(..., alias="Value")

    def to_um(self) -> float:
        """Convert the measurement to micrometers."""
        if self.unit == "um":
            return self.value
        elif self.unit == "nm":
            return self.value / 1000.0
        elif self.unit == "m":
            return self.value * 1_000_000.0
        else:
            raise ValueError(f"Unknown unit: {self.unit}")


class OperettaImageMeta(BaseModel):
    """Image metadata for operetta XML records."""

    url: str = Field(..., alias="URL")
    row: str = Field(..., alias="Row")
    column: int = Field(..., alias="Col")
    field_id: int = Field(..., alias="FieldID")
    plane_id: int = Field(..., alias="PlaneID")
    channel_id: int = Field(..., alias="ChannelID")
    timepoint_id: int = Field(..., alias="TimepointID")
    channel_name: str = Field(..., alias="ChannelName")
    resolution_x: MeasureWithUnit = Field(..., alias="ImageResolutionX")
    resolution_y: MeasureWithUnit = Field(..., alias="ImageResolutionY")
    image_size_x: int = Field(..., alias="ImageSizeX")
    image_size_y: int = Field(..., alias="ImageSizeY")
    max_intensity: int = Field(..., alias="MaxIntensity")
    pos_x: MeasureWithUnit = Field(..., alias="PositionX")
    pos_y: MeasureWithUnit = Field(..., alias="PositionY")
    pos_z: MeasureWithUnit = Field(..., alias="PositionZ")
    abs_pos_z: MeasureWithUnit = Field(..., alias="AbsPositionZ")

    @field_validator("row", mode="before")
    def validate_row(cls, v) -> str:
        """Validate and convert row to letter if given as integer."""
        try:
            row = int(v)
            if 1 <= row <= 26:
                return STANDARD_ROWS_NAMES[row - 1]
        except (ValueError, TypeError):
            pass
        return v

    @property
    def well_id(self) -> str:
        """Get well ID in format 'A01'."""
        return f"{self.row}{self.column:02d}"

    @property
    def image_id(self) -> str:
        """Get unique image ID based on well and field."""
        return f"{self.well_id}_FOV{self.field_id}"


######################################################################
#
# XML parsing helpers
#
######################################################################


def _parse(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return xmltodict.parse(
            f.read(),
            process_namespaces=True,
            namespaces={"http://www.perkinelmer.com/PEHH/HarmonyV5": None},  # type: ignore
            attr_prefix="",
            cdata_key="Value",
        )


def _load_models(path: str) -> list[OperettaImageMeta]:
    """Load Operetta image metadata from XML file."""
    metadata_path = join_url_paths(path, "Images", "Index.idx.xml")
    xml_dict = _parse(metadata_path)
    images_data = xml_dict["EvaluationInputData"]["Images"]["Image"]
    if isinstance(images_data, dict):
        images_data = [images_data]
    images_meta = [
        OperettaImageMeta.model_validate(image_dict) for image_dict in images_data
    ]
    return images_meta


######################################################################
#
# Helper functions for building tiles (following ScanR pattern)
#
######################################################################


def _get_z_spacing(images: list[OperettaImageMeta]) -> float:
    """Calculate z spacing from image records."""
    z_positions = sorted({img.pos_z.to_um() for img in images})
    if len(z_positions) <= 1:
        return 1.0
    delta_z = np.diff(z_positions)
    if not np.allclose(delta_z, delta_z[0]):
        logger.warning("Z spacing is not constant, using mean value.")
    return float(np.mean(delta_z))


def _is_time_series(images: list[OperettaImageMeta]) -> bool:
    """Determine if the images represent a time series."""
    timepoints = {img.timepoint_id for img in images}
    return len(timepoints) > 1


def _channel_names(images: list[OperettaImageMeta]) -> list[str]:
    """Get unique channel names from image records."""
    channel_names = {}
    for img in images:
        if img.channel_id not in channel_names:
            channel_names[img.channel_id] = img.channel_name
    return [channel_names[ch_id] for ch_id in sorted(channel_names.keys())]


def _get_data_type(images: list[OperettaImageMeta]) -> DataTypeEnum:
    """Determine data type based on max intensity across all images."""
    if not images:
        return DataTypeEnum.UINT16
    max_intensity = max(img.max_intensity for img in images)
    if max_intensity <= 255:
        return DataTypeEnum.UINT8
    elif max_intensity <= 65535:
        return DataTypeEnum.UINT16
    else:
        return DataTypeEnum.UINT32


def build_acquisition_details(
    images: list[OperettaImageMeta],
    detail: OperettaImageMeta,
    acquisition_model: OperettaAcquisitionModel,
) -> AcquisitionDetails:
    """Build AcquisitionDetails from OperettaImageMeta."""
    pixelsize_x = detail.resolution_x.to_um()
    pixelsize_y = detail.resolution_y.to_um()
    channel_names = _channel_names(images)
    z_spacing = _get_z_spacing(images)
    t_spacing = 1
    data_type = _get_data_type(images)
    is_time_series = _is_time_series(images)

    if not np.isclose(pixelsize_x, pixelsize_y):
        logger.warning(
            f"Physical size x ({pixelsize_x}) and y ({pixelsize_y}) are not equal. "
            "Using x size for pixelsize."
        )
    axes = default_axes_builder(is_time_series=is_time_series)
    channels = [ChannelInfo(channel_label=ch_name) for ch_name in channel_names]
    acquisition_detail = AcquisitionDetails(
        pixelsize=pixelsize_x,
        z_spacing=z_spacing,
        t_spacing=t_spacing,
        channels=channels,
        axes=axes,
        start_x_coo="world",
        length_x_coo="pixel",
        start_y_coo="world",
        length_y_coo="pixel",
        start_z_coo="pixel",
        length_z_coo="pixel",
        start_t_coo="pixel",
        length_t_coo="pixel",
        data_type=data_type,
    )
    # Update with advanced options
    acquisition_detail = acquisition_model.advanced.update_acquisition_details(
        acquisition_details=acquisition_detail
    )
    return acquisition_detail


def _build_tiles(
    images: list[OperettaImageMeta],
    data_dir: str,
    acquisition_model: OperettaAcquisitionModel,
    row: str,
    column: int,
    fov_idx: int,
    attributes: dict[str, AttributeType],
) -> list[Tile]:
    """Build individual Tile objects for each image record."""
    image_0 = images[0]
    len_x = image_0.image_size_x
    len_y = image_0.image_size_y
    acquisition_details = build_acquisition_details(
        images=images,
        detail=image_0,
        acquisition_model=acquisition_model,
    )

    # Get plate name
    image_in_plate = ImageInPlate(
        plate_name=acquisition_model.normalized_plate_name,
        row=row,
        column=column,
        acquisition=acquisition_model.acquisition_id,
    )
    fov_name = f"FOV_{fov_idx}"
    tiles = []
    for img in images:
        tiff_path = join_url_paths(data_dir, "Images", img.url)
        # Operetta stage is in "standard" cartesian coordinates, but
        # for images we want to set the origin (as many viewers do) in the top-left
        # corner, so we need to invert the y position
        # This is equivalent to flipping the image along the y axis
        pos_x = img.pos_x.to_um()
        pos_y = -img.pos_y.to_um()

        _tile = Tile(
            fov_name=fov_name,
            start_x=pos_x,
            length_x=len_x,
            start_y=pos_y,
            length_y=len_y,
            start_z=img.plane_id - 1,  # Convert to 0-indexed
            length_z=1,
            start_c=img.channel_id - 1,  # Convert to 0-indexed
            length_c=1,
            start_t=img.timepoint_id,  # Already 0-indexed
            length_t=1,
            collection=image_in_plate,
            image_loader=DefaultImageLoader(file_path=tiff_path),
            acquisition_details=acquisition_details,
            attributes=attributes,
        )
        tiles.append(_tile)

    return tiles


######################################################################
#
# Main metadata parsing function
#
######################################################################


def parse_operetta_metadata(
    *,
    acquisition_model: OperettaAcquisitionModel,
    converter_options: ConverterOptions,
) -> list[TiledImage]:
    """Parse Operetta metadata and return a list of TiledImages.

    Args:
        acquisition_model: Acquisition input model containing path and options.
        converter_options: Converter options for tile processing.

    Returns:
        List of TiledImage objects ready for conversion.
    """
    acquisition_dir = acquisition_model.path
    data = _load_models(acquisition_dir)
    condition_table = acquisition_model.get_condition_table()

    if len(data) == 0:
        raise ValueError(f"No measurement records found in {acquisition_dir}")
    # Group images by z_type, well (row, column), and field of view
    plates_groups: dict[tuple[str, int, int], list[OperettaImageMeta]] = {}

    for image in data:
        row = image.row
        column = image.column
        fov_idx = image.field_id

        key = (row, column, fov_idx)

        if key not in plates_groups:
            plates_groups[key] = []
        plates_groups[key].append(image)

    # Build tiles for each group
    all_tiles = []
    for (row, column, fov_idx), images in plates_groups.items():
        attributes = get_attributes_from_condition_table(
            condition_table=condition_table,
            row=row,
            column=column,
            acquisition=acquisition_model.acquisition_id,
        )
        _tiles = _build_tiles(
            images=images,
            data_dir=acquisition_dir,
            acquisition_model=acquisition_model,
            row=row,
            column=column,
            fov_idx=fov_idx,
            attributes=attributes,
        )
        all_tiles.extend(_tiles)

    logger.info(f"Built {len(all_tiles)} tiles from {acquisition_dir}")

    # Use preprocessing pipeline to combine tiles into TiledImages
    tiled_images = tiles_aggregation_pipeline(
        tiles=all_tiles,
        converter_options=converter_options,
        filters=acquisition_model.advanced.filters,
        validators=None,
        resource=None,  # No resource context needed here
    )

    return tiled_images
