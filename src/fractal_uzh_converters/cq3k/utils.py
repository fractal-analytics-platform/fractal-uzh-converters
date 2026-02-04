"""Utility functions for Yokogawa CQ3K data."""

import logging
from typing import Annotated, Any, Literal

import numpy as np
import xmltodict
from ome_zarr_converters_tools import (
    AcquisitionDetails,
    ConverterOptions,
    DefaultImageLoader,
    ImageInPlate,
    Tile,
    TiledImage,
    join_url_paths,
    tiles_preprocessing_pipeline,
)
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_pascal

from fractal_uzh_converters.common import STANDARD_ROWS_NAMES, BaseAcquisitionModel

logger = logging.getLogger(__name__)


######################################################################
#
# Acquisition Input Model
#
######################################################################


class CQ3KAcquisitionModel(BaseAcquisitionModel):
    """Acquisition metadata for CQ3K data.

    Attributes:
        path: Path to the acquisition directory.
            Should contain MeasurementData.mlf and MeasurementDetail.mrf files.
        plate_name: Optional custom name for the plate. If not provided, the name will
            be the acquisition directory name.
        acquisition_id: Acquisition ID,
            used to identify the acquisition in case of multiple acquisitions.
        advanced: Advanced acquisition options.
    """

    pass


######################################################################
#
# Pydantic models for parsing CQ3K metadata
# are adapted from https://github.com/fmi-faim/cellvoyager-types
#
######################################################################


class Base(BaseModel):
    """Base model with common configuration."""

    model_config = ConfigDict(
        alias_generator=to_pascal,
        extra="forbid",
    )


class MeasurementRecordBase(Base):
    """Base class for measurement records."""

    time: str
    column: int
    row: int
    field_index: int
    time_point: int
    timeline_index: int
    x: float
    y: float
    value: str


class ImageMeasurementRecord(MeasurementRecordBase):
    """Image measurement record."""

    type: Literal["IMG"]
    tile_x_index: int | None = None
    tile_y_index: int | None = None
    z_index: int
    z_image_processing: str | None = None
    z_top: float | None = None
    z_bottom: float | None = None
    action_index: int
    action: str
    z: float
    ch: int
    partial_tile_index: int | None = None


class ErrorMeasurementRecord(MeasurementRecordBase):
    """Error measurement record."""

    type: Literal["ERR"]


class MeasurementData(Base):
    """Measurement data containing image and error records."""

    xmlns: Annotated[dict, Field(alias="xmlns")]
    version: Literal["1.0"]
    measurement_record: list[ImageMeasurementRecord | ErrorMeasurementRecord] | None = (
        None
    )


class MeasurementSamplePlate(Base):
    """Measurement sample plate details."""

    name: str
    well_plate_file_name: str
    well_plate_product_file_name: str


class MeasurementChannel(Base):
    """Measurement channel details."""

    ch: int
    horizontal_pixel_dimension: float
    vertical_pixel_dimension: float
    camera_number: int
    input_bit_depth: int
    input_level: int
    horizontal_pixels: int
    vertical_pixels: int
    filter_wheel_position: int
    filter_position: int
    shading_correction_source: str
    objective_magnification_ratio: float
    original_horizontal_pixels: int
    original_vertical_pixels: int


class MeasurementDetail(Base):
    """Measurement detail metadata."""

    xmlns: Annotated[dict, Field(alias="xmlns")]
    version: Literal["1.0"]
    operator_name: str
    title: str
    application: str
    begin_time: str
    end_time: str
    measurement_setting_file_name: str
    column_count: int
    row_count: int
    time_point_count: int
    field_count: int
    z_count: int
    target_system: str
    release_number: str
    status: str
    measurement_sample_plate: MeasurementSamplePlate
    measurement_channel: list[MeasurementChannel] | MeasurementChannel


######################################################################
#
# XML parsing helpers
#
######################################################################


def _parse(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return xmltodict.parse(
                f.read(),
                process_namespaces=True,
                namespaces={"http://www.yokogawa.co.jp/BTS/BTSSchema/1.0": None},  # type: ignore
                attr_prefix="",
                cdata_key="Value",
            )
    except FileNotFoundError as e:
        logger.error(f"File not found: {path}")
        raise e
    except Exception as e:
        logger.error(f"Error parsing XML file {path}: {e}")
        raise e


def _load_models(path: str) -> tuple[MeasurementData, MeasurementDetail]:
    mlf_path = join_url_paths(path, "MeasurementData.mlf")
    mrf_path = join_url_paths(path, "MeasurementDetail.mrf")
    mlf_dict = _parse(mlf_path)
    mrf_dict = _parse(mrf_path)
    mlf = MeasurementData(**mlf_dict["MeasurementData"])
    mrf = MeasurementDetail(**mrf_dict["MeasurementDetail"])
    return mlf, mrf


######################################################################
#
# Helper functions for building tiles (following ScanR pattern)
#
######################################################################


def _get_z_spacing(images: list[ImageMeasurementRecord]) -> float:
    """Calculate z spacing from image records."""
    z_positions = sorted({img.z for img in images})
    if len(z_positions) <= 1:
        return 1.0
    delta_z = np.diff(z_positions)
    if not np.allclose(delta_z, delta_z[0]):
        logger.warning("Z spacing is not constant, using mean value.")
    return float(np.mean(delta_z))


def build_acquisition_details(
    detail: MeasurementDetail,
    acquisition_model: CQ3KAcquisitionModel,
) -> AcquisitionDetails:
    """Build AcquisitionDetails from CQ3K metadata."""
    if isinstance(detail.measurement_channel, list):
        first_channel = detail.measurement_channel[0]
    else:
        first_channel = detail.measurement_channel

    pixelsize_x = first_channel.horizontal_pixel_dimension
    pixelsize_y = first_channel.vertical_pixel_dimension

    if not np.isclose(pixelsize_x, pixelsize_y):
        logger.warning(
            f"Physical size x ({pixelsize_x}) and y ({pixelsize_y}) are not equal. "
            "Using x size for pixelsize."
        )

    acquisition_detail = AcquisitionDetails(
        pixelsize=pixelsize_x,
        channel_names=None,
        wavelength_ids=None,
    )
    # Update with advanced options
    acquisition_detail = acquisition_model.advanced.update_acquisition_details(
        acquisition_details=acquisition_detail
    )
    return acquisition_detail


def _build_tiles(
    images: list[ImageMeasurementRecord],
    data_dir: str,
    detail: MeasurementDetail,
    acquisition_model: CQ3KAcquisitionModel,
    converter_options: ConverterOptions,
    row: str,
    column: int,
    fov_idx: int,
    z_type: str | None,
) -> list[Tile]:
    """Build individual Tile objects for each image record."""
    if isinstance(detail.measurement_channel, list):
        first_channel = detail.measurement_channel[0]
    else:
        first_channel = detail.measurement_channel

    len_x = first_channel.horizontal_pixels
    len_y = first_channel.vertical_pixels

    acquisition_details = build_acquisition_details(
        detail=detail,
        acquisition_model=acquisition_model,
    )

    # Get plate name, handling z_type suffix if needed
    plate_name = acquisition_model.normalized_plate_name
    if z_type is not None:
        plate_name = f"{plate_name}_{z_type}"

    image_in_plate = ImageInPlate(
        plate_name=plate_name,
        row=row,
        column=column,
        acquisition=acquisition_model.acquisition_id,
    )

    z_spacing = _get_z_spacing(images)
    fov_name = f"FOV_{fov_idx}"

    tiles = []
    for img in images:
        tiff_path = join_url_paths(data_dir, img.value)
        # CQ3k stage is in "standard" cartesian coordinates, but
        # for images we want to set the origin (as many viewers do) in the top-left
        # corner, so we need to invert the y position
        # This is equivalent to flipping the image along the y axis
        pos_x = img.x
        pos_y = -img.y

        _tile = Tile(
            fov_name=fov_name,
            start_x=pos_x,
            start_x_coo="world",
            length_x=len_x,
            length_x_coo="pixel",
            start_y=pos_y,
            start_y_coo="world",
            length_y=len_y,
            length_y_coo="pixel",
            start_z=img.z_index - 1,  # Convert to 0-indexed
            start_z_coo="pixel",
            length_z=1,
            length_z_coo="pixel",
            start_c=img.ch,
            length_c=1,
            start_t=img.time_point - 1,  # Convert to 0-indexed
            start_t_coo="pixel",
            length_t=1,
            length_t_coo="pixel",
            collection=image_in_plate,
            image_loader=DefaultImageLoader(file_path=tiff_path),
            pixelsize=acquisition_details.pixelsize,
            z_spacing=z_spacing,
            t_spacing=acquisition_details.t_spacing,
            channel_names=acquisition_details.channel_names,
            wavelength_ids=acquisition_details.wavelength_ids,
            colors=acquisition_details.colors,
            axes=acquisition_details.axes,
            data_type=acquisition_details.data_type,
            flip_x=converter_options.stage_correction.flip_x,
            flip_y=converter_options.stage_correction.flip_y,
            swap_xy=converter_options.stage_correction.swap_xy,
            attributes={},
        )
        tiles.append(_tile)

    return tiles


######################################################################
#
# Main metadata parsing function
#
######################################################################


def parse_cq3k_metadata(
    *,
    acquisition_model: CQ3KAcquisitionModel,
    converter_options: ConverterOptions,
) -> list[TiledImage]:
    """Parse CQ3K metadata and return a list of TiledImages.

    Args:
        acquisition_model: Acquisition input model containing path and options.
        converter_options: Converter options for tile processing.

    Returns:
        List of TiledImage objects ready for conversion.
    """
    acquisition_dir = acquisition_model.path
    data, detail = _load_models(path=acquisition_dir)

    if data.measurement_record is None:
        raise ValueError(f"No measurement records found in {acquisition_dir}")

    # Group images by z_type, well (row, column), and field of view
    plates_groups: dict[
        tuple[str | None, str, int, int], list[ImageMeasurementRecord]
    ] = {}

    for record in data.measurement_record:
        if not isinstance(record, ImageMeasurementRecord):
            continue

        z_type = record.z_image_processing
        row = STANDARD_ROWS_NAMES[record.row - 1]
        column = record.column
        fov_idx = record.field_index

        key = (z_type, row, column, fov_idx)

        if key not in plates_groups:
            plates_groups[key] = []
        plates_groups[key].append(record)

    # Build tiles for each group
    all_tiles = []
    for (z_type, row, column, fov_idx), images in plates_groups.items():
        _tiles = _build_tiles(
            images=images,
            data_dir=acquisition_dir,
            detail=detail,
            acquisition_model=acquisition_model,
            converter_options=converter_options,
            row=row,
            column=column,
            fov_idx=fov_idx,
            z_type=z_type,
        )
        all_tiles.extend(_tiles)

    logger.info(f"Built {len(all_tiles)} tiles from {acquisition_dir}")

    # Use preprocessing pipeline to combine tiles into TiledImages
    tiled_images = tiles_preprocessing_pipeline(
        tiles=all_tiles,
        converter_options=converter_options,
        filters=None,
        validators=None,
        resource=None,  # No resource context needed here
    )

    return tiled_images
