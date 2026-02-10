"""Utility functions for Olympus ScanR data."""

import logging
import re
from typing import Literal, NamedTuple

import numpy as np
import polars
from ome_types import from_xml
from ome_zarr_converters_tools import (
    AcquisitionDetails,
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
from pydantic import field_validator

from fractal_uzh_converters.common import (
    STANDARD_ROWS_NAMES,
    BaseAcquisitionModel,
    get_attributes_from_condition_table,
)

AVAILABLE_PLATE_LAYOUTS = Literal["24-well", "48-well", "96-well", "384-well"]
STANDARD_PLATES_LAYOUTS: dict[AVAILABLE_PLATE_LAYOUTS, dict[str, int]] = {
    "24-well": {
        "rows": 4,
        "columns": 6,
    },
    "48-well": {
        "rows": 6,
        "columns": 8,
    },
    "96-well": {
        "rows": 8,
        "columns": 12,
    },
    "384-well": {
        "rows": 16,
        "columns": 24,
    },
}

logger = logging.getLogger(__name__)


class ScanRAcquisitionModel(BaseAcquisitionModel):
    """Acquisition details for the Olympus ScanR microscope data.

    Attributes:
        path: Path to the acquisition directory.
            For scanr, this should be the base directory of the acquisition
            or the "{acquisition_dir}/data" directory containing the metadata.ome.xml
            file and the "data" directory with the tiff files.
        plate_name: Optional custom name for the plate. If not provided, the name will
            be the acquisition directory name.
        acquisition_id: Acquisition ID,
            used to identify the acquisition in case of multiple acquisitions.
        layout: Plate layout type.
        advanced: Advanced acquisition options.
    """

    layout: AVAILABLE_PLATE_LAYOUTS = "96-well"

    @field_validator("path", mode="before")
    def validate_path(cls, v):
        """Make the path more flexible.

        Allow:
         - path/to/acquisition/data
         - path/to/acquisition/
        """
        v = v.rstrip("/")
        if v.endswith("/data"):
            return v[: -len("/data")]
        return v


def _wellid_to_row_column(
    well_id: int, layout: AVAILABLE_PLATE_LAYOUTS
) -> tuple[str, int]:
    """Get row and column from well id."""
    layout_dict = STANDARD_PLATES_LAYOUTS[layout]
    num_columns = layout_dict["columns"]
    well_id -= 1

    row = well_id // num_columns
    column = well_id % num_columns + 1

    if row >= layout_dict["rows"]:
        raise ValueError(f"Well id {well_id} is out of bounds for layout {layout}.")

    row_str = STANDARD_ROWS_NAMES[row]
    return row_str, column


def _extract_well_position_id(
    s: str, layout: AVAILABLE_PLATE_LAYOUTS
) -> tuple[tuple[str, int], int]:
    """Extract Well and Position information from a string."""
    pattern = r"W(\d+)P(\d+)"
    match = re.search(pattern, s)
    if match:
        w_id, p = match.groups()
        w_id, p = int(w_id), int(p)
        row, col = _wellid_to_row_column(w_id, layout)
        return (row, col), p
    else:
        raise ValueError(
            f"Could not extract Well and Position information from string: {s}"
        )


def _get_channel_names(image) -> list[str] | None:
    try:
        parsed_channels = [channel.name for channel in image.pixels.channels]
        if all(name is not None for name in parsed_channels):
            return parsed_channels  # type: ignore[return-value]
        else:
            return None
    except Exception as e:
        logger.warning(f"Could not parse channel names: {e}")
        return None


def _get_z_spacing(image) -> float:
    positions_z = []
    for plane in image.pixels.planes:
        if plane.the_t == 0 and plane.the_c == 0:
            positions_z.append(plane.position_z)

    if len(positions_z) == 0 or len(positions_z) == 1:
        return 1

    delta_z = np.diff(positions_z)
    if not np.allclose(delta_z, delta_z[0]):
        raise ValueError("Z spacing is not constant.")

    return delta_z[0]


def _mean_z_spacing(list_images) -> float:
    z_spacings = []
    for image in list_images:
        z_spacing = _get_z_spacing(image)
        z_spacings.append(z_spacing)
    mean_z_spacing = float(np.mean(z_spacings))
    return mean_z_spacing


def _is_time_series(image) -> bool:
    """Check if the images represent a time series."""
    time_points = {plane.the_t for plane in image.pixels.planes}
    return len(time_points) > 1


def build_acquisition_details(
    image_meta,
    acquisition_model: ScanRAcquisitionModel,
    is_time_series: bool,
    z_spacing: float,
) -> AcquisitionDetails:
    """Build AcquisitionDetails from AcquisitionInputModel."""
    pixelsize_x = image_meta.pixels.physical_size_x or 1
    pixelsize_y = image_meta.pixels.physical_size_y or 1
    if not np.isclose(pixelsize_x, pixelsize_y):
        logger.warning(
            f"Physical size x ({pixelsize_x}) and y ({pixelsize_y}) are not equal. "
            "Using x size for pixelsize."
        )

    channel_names = _get_channel_names(image_meta)
    channels = None
    if channel_names is not None:
        channels = [ChannelInfo(channel_label=name) for name in channel_names]

    axes = default_axes_builder(is_time_series=is_time_series)
    acquisition_detail = AcquisitionDetails(
        pixelsize=pixelsize_x,
        z_spacing=z_spacing,
        t_spacing=1,
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
    )
    # Update with advanced options
    acquisition_detail = acquisition_model.advanced.update_acquisition_details(
        acquisition_details=acquisition_detail
    )
    return acquisition_detail


class PlaneInfo(NamedTuple):
    """Information about a single plane in the image."""

    x: float
    y: float
    z: float
    c: int
    t: int
    tiff_path: str


def _match_tiff_to_plane(tiff_data_block: list, planes: list) -> list:
    tiff_blocks_dict = {}
    for tiff in tiff_data_block:
        key = (tiff.first_t, tiff.first_c, tiff.first_z)
        tiff_blocks_dict[key] = tiff
    planes_dict = {}
    for plane in planes:
        key = (plane.the_t, plane.the_c, plane.the_z)
        planes_dict[key] = plane
    matched = []
    for key in planes_dict:
        assert key in tiff_blocks_dict, f"Could not find matching TIFF for plane {key}"
        matched.append(
            PlaneInfo(
                x=planes_dict[key].position_x or 0.0,
                y=planes_dict[key].position_y or 0.0,
                z=planes_dict[key].the_z,
                c=planes_dict[key].the_c,
                t=planes_dict[key].the_t,
                tiff_path=tiff_blocks_dict[key].uuid.file_name,
            )
        )
    return matched


def _build_tiles(
    image_meta,
    acquisition_model: ScanRAcquisitionModel,
    mean_z_spacing: float,
    condition_table: polars.DataFrame | None,
) -> list[Tile]:
    """Build individual Tile objects for each image record."""
    (row, column), pos_id = _extract_well_position_id(
        image_meta.id, layout=acquisition_model.layout
    )
    image_in_plate = ImageInPlate(
        plate_name=acquisition_model.normalized_plate_name,
        row=row,
        column=column,
        acquisition=acquisition_model.acquisition_id,
    )
    is_time_series = _is_time_series(image_meta)
    acquisition_details = build_acquisition_details(
        image_meta=image_meta,
        acquisition_model=acquisition_model,
        is_time_series=is_time_series,
        z_spacing=mean_z_spacing,
    )
    tiles = []
    len_x = image_meta.pixels.size_x
    len_y = image_meta.pixels.size_y
    pixels = image_meta.pixels

    matched_planes = _match_tiff_to_plane(
        tiff_data_block=pixels.tiff_data_blocks,
        planes=pixels.planes,
    )
    base_tiff_dir = join_url_paths(acquisition_model.path, "data")
    attributes = get_attributes_from_condition_table(
        condition_table=condition_table,
        row=row,
        column=column,
        acquisition=acquisition_model.acquisition_id,
    )

    for plane_info in matched_planes:
        tiff_path = join_url_paths(base_tiff_dir, plane_info.tiff_path)
        # ScanR stage is in "standard" cartesian coordinates, but
        # for images we want to set the origin (as many viewers do) in the top-left
        # corner, so we need to invert the y position
        # This is equivalent to flipping the image along the y axis
        pos_x = plane_info.x or 0.0
        pos_y = -(plane_info.y or 0.0)
        _tile = Tile(
            fov_name=f"FOV_{pos_id}",
            start_x=pos_x,
            length_x=len_x,
            start_y=pos_y,
            length_y=len_y,
            start_z=plane_info.z,
            length_z=1,
            start_c=plane_info.c,
            length_c=1,
            start_t=plane_info.t,
            length_t=1,
            collection=image_in_plate,
            image_loader=DefaultImageLoader(file_path=tiff_path),
            acquisition_details=acquisition_details,
            attributes=attributes,
        )
        tiles.append(_tile)
    return tiles


def parse_scanr_metadata(
    *,
    acquisition_model: ScanRAcquisitionModel,
    converter_options: ConverterOptions,
) -> list[TiledImage]:
    """Parse ScanR metadata and return a dictionary of TiledImages."""
    acquisition_dir = acquisition_model.path
    metadata_path = join_url_paths(acquisition_dir, "data", "metadata.ome.xml")
    try:
        meta = from_xml(metadata_path)
    except Exception as e:
        raise ValueError(
            f"Could not parse OME-XML metadata file: {metadata_path}"
        ) from e

    if len(meta.images) == 0:
        raise ValueError(f"No images found in metadata file: {metadata_path}")

    condition_table = acquisition_model.get_condition_table()
    mean_z_spacing = _mean_z_spacing(meta.images)
    tiles = []
    for image in meta.images:
        _tiles = _build_tiles(
            image_meta=image,
            acquisition_model=acquisition_model,
            mean_z_spacing=mean_z_spacing,
            condition_table=condition_table,
        )
        tiles.extend(_tiles)
    tiled_images = tiles_aggregation_pipeline(
        tiles=tiles,
        converter_options=converter_options,
        filters=acquisition_model.advanced.filters,
        validators=None,
        resource=None,  # No resource context needed here
    )
    return tiled_images
