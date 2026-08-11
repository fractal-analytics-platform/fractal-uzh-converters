"""The shared Yokogawa `.mlf`/`.mrf` parser.

One implementation for both instruments: an acquisition is split into one plate per
`bts:ZImageProcessing` kind, and one carrying the attribute nowhere — the common
case — collapses to the single `z_type=None` plate.

The only per-instrument difference is `parse_yokogawa_metadata`'s
`filename_transform`, because a CellVoyager `.mlf` always names `.tif` even when
the files on disk are `.png`.
"""

import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from ome_zarr_converters_tools import (
    AcquisitionDetails,
    AttributeType,
    ChannelInfo,
    ConverterOptions,
    DefaultImageLoader,
    ImageInPlate,
    Tile,
    TiledImage,
    default_axes_builder,
    join_url_paths,
    tiles_aggregation_pipeline,
)

from fractal_uzh_converters.common import (
    STANDARD_ROWS_NAMES,
    GeometryWarning,
    HCSBaseAcquisitionModel,
    SourceMetadataWarning,
    get_attributes_from_condition_table,
)
from fractal_uzh_converters.yokogawa._channels import (
    ChannelGeometry,
    TimeIndex,
    apply_channel_overrides,
    build_time_index,
    max_acquired_channel,
    read_mes_channels,
    resolve_channels,
    warn_on_channel_geometry_mismatch,
)
from fractal_uzh_converters.yokogawa._records import (
    ImageMeasurementRecord,
    MeasurementChannel,
    MeasurementData,
    MeasurementDetail,
)
from fractal_uzh_converters.yokogawa._xml import parse_bts_xml
from fractal_uzh_converters.yokogawa._z_processing import (
    YokogawaZProcessingSelection,
    _plate_name,
    _select_z_processing,
    _z_processing_token,
)

logger = logging.getLogger(__name__)


######################################################################
#
# XML parsing helpers
#
######################################################################


def _parse(path: str) -> dict[str, Any]:
    try:
        return parse_bts_xml(path)
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
# Helper functions for building tiles
#
######################################################################


def _measurement_channels(detail: MeasurementDetail) -> list[MeasurementChannel]:
    """The `.mrf` channel entries, normalised to a list.

    `xmltodict` collapses a lone `<bts:MeasurementChannel>` to a bare dict.
    """
    if isinstance(detail.measurement_channel, list):
        return detail.measurement_channel
    return [detail.measurement_channel]


def _channel_geometries(detail: MeasurementDetail) -> list[ChannelGeometry]:
    """The geometry fields of every `.mrf` channel, for the consistency check."""
    return [
        ChannelGeometry(
            ch=channel.ch,
            xy_pixel_size=(
                channel.horizontal_pixel_dimension,
                channel.vertical_pixel_dimension,
            ),
            frame_size=(channel.horizontal_pixels, channel.vertical_pixels),
        )
        for channel in _measurement_channels(detail)
    ]


def _get_z_spacing(images: list[ImageMeasurementRecord]) -> float:
    """Calculate z spacing from image records."""
    z_positions = sorted({img.z for img in images})
    if len(z_positions) <= 1:
        return 1.0
    delta_z = np.diff(z_positions)
    if not np.allclose(delta_z, delta_z[0]):
        warnings.warn(
            "Z spacing is not constant, using mean value.",
            GeometryWarning,
            stacklevel=2,
        )
    return float(np.mean(delta_z))


def build_acquisition_details(
    images: list[ImageMeasurementRecord],
    detail: MeasurementDetail,
    acquisition_model: HCSBaseAcquisitionModel,
    channels: list[ChannelInfo],
    max_acquired_ch: int,
    time_index: TimeIndex,
) -> AcquisitionDetails:
    """Build AcquisitionDetails from the Yokogawa metadata.

    Call once per plate with all of that plate's image records, never per field of
    view: `TiledImage.add_tile` rejects tiles whose details differ, and several
    fields of view merge into one output image.

    Args:
        images: The image records of this plate.
        detail: The parsed `.mrf`.
        acquisition_model: The acquisition input model.
        channels: One entry per instrument channel slot, from `resolve_channels`.
            Shared by every plate of the acquisition.
        max_acquired_ch: Highest `bts:Ch` actually acquired, used to reject an
            `advanced.channels` override that is too short.
        time_index: This plate's time axis, from `build_time_index`.
    """
    first_channel = _measurement_channels(detail)[0]

    pixelsize_x = first_channel.horizontal_pixel_dimension
    pixelsize_y = first_channel.vertical_pixel_dimension

    if not np.isclose(pixelsize_x, pixelsize_y):
        warnings.warn(
            f"Physical size x ({pixelsize_x}) and y ({pixelsize_y}) are not equal. "
            "Using x size for pixelsize.",
            GeometryWarning,
            stacklevel=2,
        )

    z_spacing = _get_z_spacing(images)
    axes = default_axes_builder(is_time_series=time_index.is_time_series)

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
    # `update_acquisition_details` replaced `channels` wholesale; merge the user's
    # list back onto the resolved ones so it stays indexed by `bts:Ch - 1`.
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
    acquisition_model: HCSBaseAcquisitionModel,
    acquisition_details: AcquisitionDetails,
    time_index: TimeIndex,
    row: str,
    column: int,
    fov_idx: int,
    plate_name: str,
    attributes: dict[str, AttributeType],
    filename_transform: Callable[[str], str],
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
        image_path = join_url_paths(data_dir, filename_transform(img.value))
        # The Yokogawa stage is in "standard" cartesian coordinates, but
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
            # Not `time_point - 1`: `bts:TimePoint` counts timelines, not frames.
            start_t=time_index.start_t(
                row=img.row, column=img.column, time_point=img.time_point
            ),
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


def parse_yokogawa_metadata(
    *,
    acquisition_model: HCSBaseAcquisitionModel,
    converter_options: ConverterOptions,
    z_selection: YokogawaZProcessingSelection | None = None,
    filename_transform: Callable[[str], str] = lambda file_name: file_name,
) -> list[TiledImage]:
    """Parse a Yokogawa acquisition and return a list of TiledImages.

    Args:
        acquisition_model: Acquisition input model containing path and options.
        converter_options: Converter options for tile processing.
        z_selection: Which `bts:ZImageProcessing` kinds to convert, or `None` for
            all of them.
        filename_transform: Applied to `bts:MeasurementRecord`'s file name before
            it is resolved against the acquisition directory.

    Returns:
        List of TiledImage objects ready for conversion.
    """
    acquisition_dir = acquisition_model.path
    data, detail = _load_models(path=acquisition_dir)
    condition_table = acquisition_model.get_condition_table()

    # Once per acquisition — the two functions below run per plate and per field of
    # view, so neither is the right place to warn from.
    warn_on_channel_geometry_mismatch(
        geometries=_channel_geometries(detail),
        acquisition_dir=acquisition_dir,
    )

    if data.measurement_record is None:
        raise ValueError(f"No measurement records found in {acquisition_dir}")

    # Normalize single record to list (xmltodict returns dict for single elements)
    records = (
        data.measurement_record
        if isinstance(data.measurement_record, list)
        else [data.measurement_record]
    )

    # `bts:Type="ERR"` records — autofocus failures — describe no image. Dropped in
    # one pass so a plate-wide focus failure warns once, not hundreds of times.
    all_image_records = [r for r in records if isinstance(r, ImageMeasurementRecord)]
    skipped = len(records) - len(all_image_records)
    if skipped:
        warnings.warn(
            f"Skipping {skipped} error record(s) (bts:Type='ERR') in "
            f"{acquisition_dir}.",
            SourceMetadataWarning,
            stacklevel=2,
        )
    if not all_image_records:
        raise ValueError(
            f"No image measurement records found in {acquisition_dir}: "
            f"all {skipped} record(s) are errors."
        )

    # Group images by z_type, well (row, column), and field of view
    plates_groups: dict[
        tuple[str | None, str, int, int], list[ImageMeasurementRecord]
    ] = {}
    # ... and by z_type alone, since each z_type is written as its own plate
    plates_records: dict[str | None, list[ImageMeasurementRecord]] = {}

    for record in all_image_records:
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

    # Once per distinct `z_type`, so an unknown algorithm warns once per
    # acquisition. The token is both the `z_processing` value and the name suffix.
    plates_tokens = {z_type: _z_processing_token(z_type) for z_type in plates_records}
    kept = _select_z_processing(
        plates_tokens=plates_tokens,
        selection=z_selection,
        acquisition_dir=acquisition_dir,
    )
    plates_records = {
        z_type: records for z_type, records in plates_records.items() if z_type in kept
    }
    plates_groups = {
        key: records for key, records in plates_groups.items() if key[0] in kept
    }
    # Kept records only, so selecting a single plate also relaxes the required
    # `advanced.channels` length to that plate's highest `bts:Ch`.
    image_records = [
        record for records in plates_records.values() for record in records
    ]

    # Once for the whole acquisition, not per plate: the projection plates share one
    # `.mes` and a channel must keep the same label across all of them.
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
    # Acquisition-wide too, so a too-short `advanced.channels` list fails
    # identically on every plate.
    max_acquired_ch = max_acquired_channel(acquired)

    # Per plate: each is its own collection, so a projection acquired at fewer time
    # points than the raw slices gets its own dense time axis.
    plates_time_indices = {
        z_type: build_time_index(
            (record.row, record.column, record.time_point) for record in records
        )
        for z_type, records in plates_records.items()
    }

    # One AcquisitionDetails per plate: fields of view merge into a single output
    # image, and `TiledImage.add_tile` rejects tiles whose details disagree.
    plates_details = {
        z_type: build_acquisition_details(
            images=records,
            detail=detail,
            acquisition_model=acquisition_model,
            channels=channels,
            max_acquired_ch=max_acquired_ch,
            time_index=plates_time_indices[z_type],
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
            time_index=plates_time_indices[z_type],
            row=row,
            column=column,
            fov_idx=fov_idx,
            plate_name=plates_names[z_type],
            attributes=attributes,
            filename_transform=filename_transform,
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
