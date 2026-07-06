"""Utility functions for Yokogawa CellVoyager data."""

import logging
from typing import Annotated, Any, Literal

import numpy as np
import xmltodict
from ome_zarr_converters_tools import (
    AcquisitionDetails,
    AttributeType,
    ConverterOptions,
    DefaultImageLoader,
    ImageInPlate,
    Tile,
    TiledImage,
    default_axes_builder,
    filesystem_for_url,
    join_url_paths,
    tiles_aggregation_pipeline,
)
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_pascal

from fractal_uzh_converters.common import (
    STANDARD_ROWS_NAMES,
    HCSBaseAcquisitionModel,
    get_attributes_from_condition_table,
)

logger = logging.getLogger(__name__)


######################################################################
#
# Acquisition Input Model
#
######################################################################


class CellVoyagerAcquisitionModel(HCSBaseAcquisitionModel):
    """Acquisition details for the CellVoyager microscope data."""

    image_extension: Literal[".tif", ".png"] = ".tif"
    """
    File extension of the actual image files.
    The metadata (.mlf) always references '.tif', but the actual files
    may be '.png' or '.tif'. Select the extension matching your data.
    """


######################################################################
#
# Pydantic models for parsing CellVoyager metadata
#
######################################################################


class Base(BaseModel):
    """Base model with common configuration."""

    model_config = ConfigDict(
        alias_generator=to_pascal,
        extra="ignore",
    )


class MeasurementRecordBase(Base):
    """Base class for measurement records."""

    time: str
    column: int
    row: int
    time_point: int
    timeline_index: int
    x: float
    y: float
    value: str


class ImageMeasurementRecord(MeasurementRecordBase):
    """Image measurement record."""

    type: Literal["IMG"]
    field_index: int
    z_index: int
    action_index: int
    action: str
    z: float
    ch: int


class ErrorMeasurementRecord(MeasurementRecordBase):
    """Error measurement record."""

    type: Literal["ERR"]


class MeasurementData(Base):
    """Measurement data containing image and error records."""

    xmlns: Annotated[dict, Field(alias="xmlns")]
    version: Literal["1.0"]
    measurement_record: (
        list[ImageMeasurementRecord | ErrorMeasurementRecord]
        | ImageMeasurementRecord
        | ErrorMeasurementRecord
        | None
    ) = None


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
    measurement_sample_plate: MeasurementSamplePlate
    measurement_channel: list[MeasurementChannel] | MeasurementChannel


######################################################################
#
# XML parsing helpers
#
######################################################################


def _parse(path: str) -> dict[str, Any]:
    try:
        fs = filesystem_for_url(path)
        with fs.open(path, encoding="utf-8") as f:
            return xmltodict.parse(
                f.read(),
                process_namespaces=True,
                namespaces={"http://www.yokogawa.co.jp/BTS/BTSSchema/1.0": None},
                attr_prefix="",
                cdata_key="Value",
            )
    except FileNotFoundError as e:
        logger.error(f"File not found: {path}")
        raise e
    except Exception as e:
        logger.error(f"Error parsing XML file {path}: {e}")
        raise e


def _load_models(
    path: str,
) -> tuple[MeasurementData, MeasurementDetail]:
    mlf_path = join_url_paths(path, "MeasurementData.mlf")
    mrf_path = join_url_paths(path, "MeasurementDetail.mrf")
    mlf_dict = _parse(mlf_path)
    mrf_dict = _parse(mrf_path)
    mlf = MeasurementData(**mlf_dict["MeasurementData"])
    mrf = MeasurementDetail(**mrf_dict["MeasurementDetail"])
    return mlf, mrf


######################################################################
#
# Helper functions for building tiles
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


def _is_time_series(images: list[ImageMeasurementRecord]) -> bool:
    """Check if the images represent a time series."""
    time_points = {img.time_point for img in images}
    return len(time_points) > 1


def _replace_extension(filename: str, new_extension: str) -> str:
    """Replace the .tif extension in the metadata with the actual extension."""
    if filename.endswith(".tif"):
        return filename[: -len(".tif")] + new_extension
    return filename


def build_acquisition_details(
    images: list[ImageMeasurementRecord],
    detail: MeasurementDetail,
    acquisition_model: CellVoyagerAcquisitionModel,
) -> AcquisitionDetails:
    """Build AcquisitionDetails from CellVoyager metadata."""
    if isinstance(detail.measurement_channel, list):
        first_channel = detail.measurement_channel[0]
    else:
        first_channel = detail.measurement_channel

    pixelsize_x = first_channel.horizontal_pixel_dimension
    pixelsize_y = first_channel.vertical_pixel_dimension

    if not np.isclose(pixelsize_x, pixelsize_y):
        logger.warning(
            f"Physical size x ({pixelsize_x}) and y ({pixelsize_y}) are not "
            "equal. Using x size for pixelsize."
        )

    z_spacing = _get_z_spacing(images)
    is_time_series = _is_time_series(images)
    axes = default_axes_builder(is_time_series=is_time_series)

    acquisition_detail = AcquisitionDetails(
        pixelsize=pixelsize_x,
        z_spacing=z_spacing,
        t_spacing=1,
        channels=None,
        axes=axes,
        start_x_coo="world",
        length_x_coo="pixel",
        start_y_coo="world",
        length_y_coo="pixel",
        start_z_coo="pixel",
        length_z_coo="pixel",
        start_t_coo="pixel",
        length_t_coo="pixel",
    )
    acquisition_detail = acquisition_model.advanced.update_acquisition_details(
        acquisition_details=acquisition_detail
    )
    return acquisition_detail


def _build_tiles(
    images: list[ImageMeasurementRecord],
    data_dir: str,
    detail: MeasurementDetail,
    acquisition_model: CellVoyagerAcquisitionModel,
    row: str,
    column: int,
    fov_idx: int,
    attributes: dict[str, AttributeType],
) -> list[Tile]:
    """Build individual Tile objects for each image record."""
    if isinstance(detail.measurement_channel, list):
        first_channel = detail.measurement_channel[0]
    else:
        first_channel = detail.measurement_channel

    len_x = first_channel.horizontal_pixels
    len_y = first_channel.vertical_pixels

    acquisition_details = build_acquisition_details(
        images=images,
        detail=detail,
        acquisition_model=acquisition_model,
    )

    plate_name = acquisition_model.normalized_plate_name

    image_in_plate = ImageInPlate(
        plate_name=plate_name,
        row=row,
        column=column,
        acquisition=acquisition_model.acquisition_id,
    )

    fov_name = f"FOV_{fov_idx}"

    tiles = []
    for img in images:
        filename = _replace_extension(img.value, acquisition_model.image_extension)
        image_path = join_url_paths(data_dir, filename)

        # CellVoyager stage is in "standard" cartesian coordinates, but
        # for images we want to set the origin in the top-left corner,
        # so we invert the y position.
        pos_x = img.x
        pos_y = -img.y

        _tile = Tile(
            fov_name=fov_name,
            start_x=pos_x,
            length_x=len_x,
            start_y=pos_y,
            length_y=len_y,
            start_z=img.z_index - 1,
            length_z=1,
            start_c=img.ch,
            length_c=1,
            start_t=img.time_point - 1,
            length_t=1,
            collection=image_in_plate,
            image_loader=DefaultImageLoader(file_path=image_path),
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


def parse_cellvoyager_metadata(
    *,
    acquisition_model: CellVoyagerAcquisitionModel,
    converter_options: ConverterOptions,
) -> list[TiledImage]:
    """Parse CellVoyager metadata and return a list of TiledImages.

    Args:
        acquisition_model: Acquisition input model containing path and options.
        converter_options: Converter options for tile processing.

    Returns:
        List of TiledImage objects ready for conversion.
    """
    acquisition_dir = acquisition_model.path
    data, detail = _load_models(path=acquisition_dir)
    condition_table = acquisition_model.get_condition_table()

    if data.measurement_record is None:
        raise ValueError(f"No measurement records found in {acquisition_dir}")

    # Normalize single record to list (xmltodict returns dict for single elements)
    records = (
        data.measurement_record
        if isinstance(data.measurement_record, list)
        else [data.measurement_record]
    )

    # Group images by well (row, column) and field of view
    groups: dict[tuple[str, int, int], list[ImageMeasurementRecord]] = {}

    for record in records:
        if not isinstance(record, ImageMeasurementRecord):
            logger.warning(f"Skipping the Error record: {record}")
            continue

        row = STANDARD_ROWS_NAMES[record.row - 1]
        column = record.column
        fov_idx = record.field_index

        key = (row, column, fov_idx)

        if key not in groups:
            groups[key] = []
        groups[key].append(record)

    # Build tiles for each group
    all_tiles = []
    for (row, column, fov_idx), images in groups.items():
        attributes = get_attributes_from_condition_table(
            condition_table=condition_table,
            row=row,
            column=column,
            acquisition=acquisition_model.acquisition_id,
        )
        _tiles = _build_tiles(
            images=images,
            data_dir=acquisition_dir,
            detail=detail,
            acquisition_model=acquisition_model,
            row=row,
            column=column,
            fov_idx=fov_idx,
            attributes=attributes,
        )
        all_tiles.extend(_tiles)

    logger.info(f"Built {len(all_tiles)} tiles from {acquisition_dir}")

    tiled_images = tiles_aggregation_pipeline(
        tiles=all_tiles,
        converter_options=converter_options,
        filters=acquisition_model.advanced.filters,
        validators=None,
        resource=None,
    )

    return tiled_images
