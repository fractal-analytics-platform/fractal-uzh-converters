"""Utility functions for Yokogawa CQ3K data."""

import logging
from typing import Annotated, Any, Literal

import numpy as np
import xmltodict
from ome_zarr_converters_tools import (
    AcquisitionDetails,
    AcquisitionOptions,
    AttributeType,
    ChannelInfo,
    ConverterOptions,
    DefaultImageLoader,
    ImageInPlate,
    Tile,
    TiledImage,
    UserFacingModel,
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
    apply_channel_overrides,
    get_attributes_from_condition_table,
    max_acquired_channel,
    read_mes_channels,
    resolve_channels,
)

logger = logging.getLogger(__name__)


######################################################################
#
# Acquisition Input Model
#
######################################################################


class ZProcessingSelection(UserFacingModel):
    """Which Z-image processing outputs to convert."""

    z_slices: bool = Field(default=True, title="Z slices")
    """Convert the raw Z stack."""
    mip: bool = Field(default=False, title="MIP")
    """Convert the maximum intensity projection."""
    min_ip: bool = Field(default=False, title="MinIP")
    """Convert the minimum intensity projection."""
    sip: bool = Field(default=False, title="SIP")
    """Convert the sum intensity projection."""


class CQ3KAcquisitionOptions(AcquisitionOptions):
    """Acquisition options for the CQ3K converter."""

    z_processing: ZProcessingSelection | None = Field(
        default=None, title="Z Processing"
    )
    """
    Which Z-image processing outputs to convert, each written as its own plate.
    Leave unset to convert every one the acquisition contains.
    """


class CQ3KAcquisitionModel(HCSBaseAcquisitionModel):
    """Acquisition details for the CQ3K microscope data."""

    advanced: CQ3KAcquisitionOptions = Field(default_factory=CQ3KAcquisitionOptions)
    """
    Advanced acquisition options.
    """


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


def _measurement_channels(detail: MeasurementDetail) -> list[MeasurementChannel]:
    """The `.mrf` channel entries, normalised to a list.

    `xmltodict` collapses a lone `<bts:MeasurementChannel>` to a bare dict.
    """
    if isinstance(detail.measurement_channel, list):
        return detail.measurement_channel
    return [detail.measurement_channel]


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
    """Check whether at least one well holds more than one time point.

    `TimePoint` is a per-timeline counter, so distinct values across a plate do
    not make it a time series; only more than one within a single output image
    does. Evaluated plate-wide because `add_tile` requires uniform axes.
    """
    per_well: dict[tuple[int, int], set[int]] = {}
    for img in images:
        per_well.setdefault((img.row, img.column), set()).add(img.time_point)
    return any(len(time_points) > 1 for time_points in per_well.values())


#: `bts:ZImageProcessing` values, mapped to the conventional imaging abbreviations.
#: Anything else is used verbatim, with a warning.
_Z_PROCESSING_SUFFIXES = {"Maximum": "MIP", "Minimum": "MinIP", "Sum": "SIP"}

#: The token of the raw-slice plate, which carries no name suffix.
_Z_SLICES = "Z slices"

#: `ZProcessingSelection` field -> the token it selects. Also the set of tokens a
#: selection is able to name at all.
_Z_PROCESSING_TOKENS = {
    "z_slices": _Z_SLICES,
    "mip": "MIP",
    "min_ip": "MinIP",
    "sip": "SIP",
}


def _z_processing_token(z_type: str | None) -> str:
    """The selection token for one `bts:ZImageProcessing` value.

    Doubles as the plate name suffix. Resolve it once per distinct `z_type`: an
    unrecognised value warns here, and `_build_tiles` runs per field of view.
    """
    if z_type is None:
        return _Z_SLICES
    token = _Z_PROCESSING_SUFFIXES.get(z_type)
    if token is None:
        logger.warning(
            f"Unknown z image processing type '{z_type}'. Using it verbatim as the "
            "plate name suffix."
        )
        token = z_type
    return token


def _select_z_processing(
    *,
    plates_tokens: dict[str | None, str],
    selection: ZProcessingSelection | None,
    acquisition_dir: str,
) -> set[str | None]:
    """The `z_image_processing` groups to convert, given the user's selection.

    No selection keeps everything. Otherwise a selected kind the acquisition does not
    contain is a warning rather than an error, so that one selection can be applied to
    a batch of acquisitions; but a selection matching *nothing* raises, since the
    alternative is an empty output whose cause only surfaces much later, from the
    library, without naming the option responsible.

    Args:
        plates_tokens: `{z_image_processing: token}` for this acquisition.
        selection: The user's `advanced.z_processing`, or `None` for all of them.
        acquisition_dir: Named in the errors, since a batch fails one acquisition at
            a time.

    Returns:
        The `z_image_processing` values to keep.

    Raises:
        ValueError: If nothing is enabled, or if the selection matches none of the
            parsed plates. The two have different fixes, so they are reported apart.
    """
    if selection is None:
        return set(plates_tokens)

    selected = {
        token
        for field, token in _Z_PROCESSING_TOKENS.items()
        if getattr(selection, field)
    }
    if not selected:
        raise ValueError(
            "`advanced.z_processing` enables nothing, so there is nothing to convert "
            f"in {acquisition_dir}. Enable at least one output, or leave the option "
            "unset to convert everything the acquisition contains."
        )

    available = set(plates_tokens.values())
    kept = {z_type for z_type, token in plates_tokens.items() if token in selected}
    if not kept:
        raise ValueError(
            f"`advanced.z_processing` enables {sorted(selected)} but "
            f"{acquisition_dir} contains only {sorted(available)}. Clear the option "
            "to convert everything the acquisition contains."
        )

    missing = sorted(selected - available)
    if missing:
        logger.warning(
            f"`advanced.z_processing` enables {missing}, which {acquisition_dir} "
            "does not contain. Converting the rest of the selection."
        )

    # An unrecognised `bts:ZImageProcessing` value keeps its raw string as the token,
    # so no field of `ZProcessingSelection` can name it.
    unnameable = sorted(
        token
        for z_type, token in plates_tokens.items()
        if z_type not in kept and token not in _Z_PROCESSING_TOKENS.values()
    )
    if unnameable:
        logger.warning(
            f"Skipping the unrecognised z image processing type(s) {unnameable}, "
            "which `advanced.z_processing` cannot name. Use a `Path Regex Filter` on "
            "the plate name to select them."
        )
    return kept


def _plate_name(base_name: str, token: str) -> str:
    """Plate name for one `z_image_processing` group.

    Each projection algorithm is written as its own plate; the raw-slice plate
    carries no suffix and can coexist with them.
    """
    if token == _Z_SLICES:
        return base_name
    return f"{base_name}_{token}"


def build_acquisition_details(
    images: list[ImageMeasurementRecord],
    detail: MeasurementDetail,
    acquisition_model: CQ3KAcquisitionModel,
    channels: list[ChannelInfo],
    max_acquired_ch: int,
) -> AcquisitionDetails:
    """Build AcquisitionDetails from CQ3K metadata.

    Call this once per plate (i.e. per `z_image_processing` group) with all of
    that plate's image records, never per field of view: `TiledImage.add_tile`
    rejects tiles whose AcquisitionDetails differ, and several fields of view
    merge into a single output image.

    Args:
        images: The image records of this plate.
        detail: The parsed `.mrf`.
        acquisition_model: The acquisition input model.
        channels: One entry per instrument channel slot, from `resolve_channels`.
            Resolved once for the whole acquisition and shared by every plate, so
            that a channel keeps the same label across the raw and the projection
            plates.
        max_acquired_ch: Highest `bts:Ch` the acquisition actually acquired, used
            to reject an `advanced.channels` override that is too short.
    """
    first_channel = _measurement_channels(detail)[0]

    pixelsize_x = first_channel.horizontal_pixel_dimension
    pixelsize_y = first_channel.vertical_pixel_dimension

    if not np.isclose(pixelsize_x, pixelsize_y):
        logger.warning(
            f"Physical size x ({pixelsize_x}) and y ({pixelsize_y}) are not equal. "
            "Using x size for pixelsize."
        )

    z_spacing = _get_z_spacing(images)
    is_time_series = _is_time_series(images)
    axes = default_axes_builder(is_time_series=is_time_series)

    acquisition_detail = AcquisitionDetails(
        xy_pixel_size=pixelsize_x,
        z_spacing=z_spacing,
        t_spacing=1,
        channels=channels,
        axes=axes,
        start_x_space="world",
        length_x_space="pixel",
        start_y_space="world",
        length_y_space="pixel",
        start_z_space="pixel",
        length_z_space="pixel",
        start_t_space="pixel",
        length_t_space="pixel",
    )
    # Update with advanced options
    acquisition_detail = acquisition_model.advanced.update_acquisition_details(
        acquisition_details=acquisition_detail
    )
    # `update_acquisition_details` replaces `channels` wholesale with the user's
    # list, dropping the `.mes` colours and any slot the list is too short to
    # cover. Merge it back onto the resolved channels instead, so the list stays
    # indexed by `bts:Ch - 1`.
    acquisition_detail.channels = apply_channel_overrides(
        resolved=channels,
        overrides=acquisition_model.advanced.channels,
        max_acquired_ch=max_acquired_ch,
    )
    return acquisition_detail


def _build_tiles(
    images: list[ImageMeasurementRecord],
    data_dir: str,
    detail: MeasurementDetail,
    acquisition_model: CQ3KAcquisitionModel,
    acquisition_details: AcquisitionDetails,
    row: str,
    column: int,
    fov_idx: int,
    plate_name: str,
    attributes: dict[str, AttributeType],
) -> list[Tile]:
    """Build individual Tile objects for each image record."""
    first_channel = _measurement_channels(detail)[0]

    len_x = first_channel.horizontal_pixels
    len_y = first_channel.vertical_pixels

    image_in_plate = ImageInPlate(
        plate_name=plate_name,
        row=row,
        column=column,
        acquisition=acquisition_model.acquisition_id,
    )

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
            length_x=len_x,
            start_y=pos_y,
            length_y=len_y,
            start_z=img.z_index - 1,  # Convert to 0-indexed
            length_z=1,
            start_c=img.ch - 1,  # Convert to 0-indexed
            length_c=1,
            start_t=img.time_point - 1,  # Convert to 0-indexed
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
    condition_table = acquisition_model.get_condition_table()

    if data.measurement_record is None:
        raise ValueError(f"No measurement records found in {acquisition_dir}")

    # Group images by z_type, well (row, column), and field of view
    plates_groups: dict[
        tuple[str | None, str, int, int], list[ImageMeasurementRecord]
    ] = {}
    # ... and by z_type alone, since each z_type is written as its own plate
    plates_records: dict[str | None, list[ImageMeasurementRecord]] = {}

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

        if z_type not in plates_records:
            plates_records[z_type] = []
        plates_records[z_type].append(record)

    # Resolved once per distinct `z_type` rather than per (well, field of view)
    # group, so an unknown algorithm warns once per acquisition. The token is both
    # the `advanced.z_processing` value and the plate name suffix.
    plates_tokens = {z_type: _z_processing_token(z_type) for z_type in plates_records}
    kept = _select_z_processing(
        plates_tokens=plates_tokens,
        selection=acquisition_model.advanced.z_processing,
        acquisition_dir=acquisition_dir,
    )
    plates_records = {
        z_type: records for z_type, records in plates_records.items() if z_type in kept
    }
    plates_groups = {
        key: records for key, records in plates_groups.items() if key[0] in kept
    }
    # Flat, for the acquisition-wide channel resolution below. Built from the kept
    # records only, so that selecting a single plate also relaxes the required
    # `advanced.channels` length to that plate's highest `bts:Ch`.
    image_records = [
        record for records in plates_records.values() for record in records
    ]

    # Channel metadata comes from the `.mes` protocol file, whose basename is
    # recorded in the `.mrf`. Resolved once for the whole acquisition, not per
    # plate: the projection plates share one `.mes`, and a channel must keep the
    # same label across the raw and the projection plates. The resolved list
    # spans the full instrument channel range so that element `i` is
    # `bts:Ch i + 1`, matching `start_c = ch - 1`; slots this acquisition never
    # used are pruned per image at compute time.
    mes_channels = read_mes_channels(
        acquisition_dir=acquisition_dir,
        mes_file_name=detail.measurement_setting_file_name,
    )
    acquired = [(img.action_index, img.ch) for img in image_records]
    channels = resolve_channels(
        mes_channels=mes_channels,
        acquired=acquired,
        mrf_channel_count=len(_measurement_channels(detail)),
    )
    # Acquisition-wide as well, so that a too-short `advanced.channels` list
    # fails identically on every plate of the acquisition.
    max_acquired_ch = max_acquired_channel(acquired)

    # One AcquisitionDetails per plate: fields of view merge into a single output
    # image, and `TiledImage.add_tile` rejects tiles whose details disagree.
    plates_details = {
        z_type: build_acquisition_details(
            images=records,
            detail=detail,
            acquisition_model=acquisition_model,
            channels=channels,
            max_acquired_ch=max_acquired_ch,
        )
        for z_type, records in plates_records.items()
    }

    plates_names = {
        z_type: _plate_name(acquisition_model.normalized_plate_name, token)
        for z_type, token in plates_tokens.items()
        if z_type in kept
    }

    # Build tiles for each group
    all_tiles = []
    for (z_type, row, column, fov_idx), images in plates_groups.items():
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
            acquisition_details=plates_details[z_type],
            row=row,
            column=column,
            fov_idx=fov_idx,
            plate_name=plates_names[z_type],
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
