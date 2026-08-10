"""Pydantic models for the Yokogawa `.mlf` and `.mrf` metadata files.

Shared by the CQ3K and the CellVoyager converters: the two instruments write the
same BTS schema, and CQ3K is a strict superset. The fields only a CQ3K `.mrf`
carries are optional here, so the same models parse a CellVoyager `.mrf` — none
of them is read anywhere.

Adapted from https://github.com/fmi-faim/cellvoyager-types
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_pascal


class Base(BaseModel):
    """Base model with common configuration."""

    model_config = ConfigDict(
        alias_generator=to_pascal,
        # `ignore`, not `forbid`: a firmware update adding a single `bts:` attribute
        # would otherwise be a hard parse failure. The `.mrf` of both instruments
        # already carries fields these models do not declare.
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
    """Image measurement record.

    The fields below `z_index` are written by CQ3K and absent from a CellVoyager
    `.mlf`, except `z_image_processing`, which a CellVoyager acquisition with a
    projection carries too.
    """

    type: Literal["IMG"]
    field_index: int
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
    """Error measurement record.

    An autofocus failure carries no `bts:FieldIndex`, which is why that field
    sits on `ImageMeasurementRecord` rather than the base (#41). The attributes
    an ERR record does carry beyond the base set — `bts:PartialTileIndex` — are
    never read, so `extra="ignore"` absorbs them and they stay unmodelled.
    """

    type: Literal["ERR"]


class MeasurementData(Base):
    """Measurement data containing image and error records."""

    xmlns: Annotated[dict, Field(alias="xmlns")]
    version: Literal["1.0"]
    # `xmltodict` collapses a lone `<bts:MeasurementRecord>` to a bare dict, so a
    # single-record acquisition has to be accepted here and normalised by the caller.
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
    """Measurement channel details.

    The last three fields are CQ3K-only and optional so that a CellVoyager
    `.mrf`, which declares none of them, still validates.
    """

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
    objective_magnification_ratio: float | None = None
    original_horizontal_pixels: int | None = None
    original_vertical_pixels: int | None = None


class MeasurementDetail(Base):
    """Measurement detail metadata.

    `status` is CQ3K-only, hence optional — see `MeasurementChannel`.
    """

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
    status: str | None = None
    measurement_sample_plate: MeasurementSamplePlate
    measurement_channel: list[MeasurementChannel] | MeasurementChannel
