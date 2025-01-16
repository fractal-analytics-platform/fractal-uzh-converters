"""Utility functions for Olympus ScanR data."""

import re
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
from fractal_converters_tools.microplate_utils import get_row_column
from fractal_converters_tools.tile import OriginDict, Point, Tile
from fractal_converters_tools.tiled_image import TiledImage
from ome_types import from_xml
from tifffile import imread

logger = getLogger(__name__)


class TiffLoader:
    """Load a full tile from a list of tiff images."""

    def __init__(self, image: Any, data_dir: Path, shapes: dict[str, int]):
        """Initialize the TiffLoader."""
        self.image = image
        self.data_dir = data_dir / "data"
        self.shapes = shapes

    def __call__(self) -> np.ndarray:
        """Return the full tile."""
        first_acq = self.image.pixels.tiff_data_blocks[0]
        im = imread(self.data_dir / first_acq.uuid.file_name)

        if im.ndim != 2:
            raise ValueError("Only 2D tiff files are currently supported.")

        shape_y, shape_x = im.shape
        if shape_y != self.shapes["y"] or shape_x != self.shapes["x"]:
            raise ValueError(
                "Tiff file shape does not match the expected "
                "shape from the OME metadata."
            )

        tile_shape = (
            self.shapes["t"],
            self.shapes["c"],
            self.shapes["z"],
            self.shapes["y"],
            self.shapes["x"],
        )
        full_tile = np.zeros(shape=tile_shape, dtype=im.dtype)
        full_tile[first_acq.first_t, first_acq.first_z, first_acq.first_c] = im

        for tif in self.image.pixels.tiff_data_blocks[1:]:
            try:
                im = imread(self.data_dir / tif.uuid.file_name)
            except FileNotFoundError:
                logger.warning(
                    f"Tiff tile not found: {self.data_dir / tif.uuid.file_name}"
                )
                continue
            except Exception as e:
                logger.error(f"Error loading tiff tile: {e}")
                continue

            full_tile[tif.first_t, tif.first_c, tif.first_z] = im

        return full_tile


def _get_channel_names(image) -> list[str]:
    return [channel.name for channel in image.pixels.channels]


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


def _get_tiles_shapes(image) -> dict[str, int]:
    return {
        "t": image.pixels.size_t,
        "c": image.pixels.size_c,
        "z": image.pixels.size_z,
        "y": image.pixels.size_y,
        "x": image.pixels.size_x,
    }


def tile_from_ome_image(
    image, data_dir: Path, scale_xy: float | None = None, scale_z: float | None = None
) -> Tile:
    """Create a Tile object from an OME image."""
    size_z, size_c, size_t = (
        image.pixels.size_z,
        image.pixels.size_c,
        image.pixels.size_t,
    )

    channel_names = _get_channel_names(image)

    physical_size_x = scale_xy or image.pixels.physical_size_x or 1
    physical_size_y = scale_xy or image.pixels.physical_size_y or 1
    physical_size_z = scale_z or _get_z_spacing(image)

    if physical_size_x != physical_size_y:
        raise ValueError("Physical size x and y are not equal. This is not supported.")

    length_x_physical = physical_size_x * image.pixels.size_x
    length_y_physical = physical_size_y * image.pixels.size_y

    # find top_l point
    for plane in image.pixels.planes:
        # find top_l point
        if plane.the_z == 0 and plane.the_c == 0 and plane.the_t == 0:
            x = plane.position_x or 0
            y = plane.position_y or 0
            top_l = Point(x=x, y=y, z=0, t=0, c=0)
            top_l_z_real = plane.position_z or 0
            top_l_t_real = plane.delta_t or 0

        if (
            plane.the_z == size_z - 1
            and plane.the_c == size_c - 1
            and plane.the_t == size_t - 1
        ):
            x = plane.position_x or 0
            y = plane.position_y or 0
            x += length_x_physical
            y += length_y_physical
            z = size_z * physical_size_z
            bot_r = Point(x=x, y=y, z=z, t=size_t, c=size_c)

    tiff_loader = TiffLoader(
        image=image, data_dir=data_dir, shapes=_get_tiles_shapes(image)
    )

    origin = OriginDict(
        x_micrometer_original=top_l.x,
        y_micrometer_original=top_l.y,
        z_micrometer_original=top_l_z_real,
        t_original=top_l_t_real,
    )

    tile = Tile.from_points(
        top_l,
        bot_r,
        origin=origin,
        data_loader=tiff_loader,
        channel_names=channel_names,
        xy_scale=physical_size_x,
        z_scale=physical_size_z,
    )
    return tile


def extract_well_position_id(s: str) -> tuple[int, int]:
    """Extract Well and Position information from a string."""
    pattern = r"W(\d+)P(\d+)"
    match = re.search(pattern, s)
    if match:
        w, p = match.groups()
        w, p = int(w), int(p)
        return w, p
    else:
        raise ValueError(
            f"Could not extract Well and Position information from string: {s}"
        )


def parse_scanr_metadata(
    data_dir: Path,
    acq_id: int,
    plate_layout: str = "96-well",
    channel_names: list[str] | None = None,
    num_levels: int = 1,
) -> dict[str, TiledImage]:
    """Parse ScanR metadata and return a dictionary of TiledImages."""
    metadata_path = data_dir / "data" / "metadata.ome.xml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    meta = from_xml(metadata_path)

    tiled_images = {}
    for image in meta.images:
        well_id, pos_id = extract_well_position_id(image.id)
        well_acq_id = f"{well_id}_{acq_id}"

        tile = tile_from_ome_image(image, data_dir)

        if well_acq_id not in tiled_images:
            row, column = get_row_column(well_id, plate_layout)
            name = image.name if image.name else f"{well_id}/{pos_id}"
            tiled_images[well_acq_id] = TiledImage(
                name=name,
                row=row,
                column=column,
                acquisition_id=acq_id,
                tiles=[tile],
                channel_names=channel_names,
                num_levels=num_levels,
            )
        else:
            tiled_images[well_acq_id].tiles.append(tile)
    return tiled_images
