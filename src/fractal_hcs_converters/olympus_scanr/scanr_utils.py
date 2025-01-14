import re
from pathlib import Path
from typing import Any

import numpy as np
from fractal_converters_tools.microplate_utils import get_row_column
from fractal_converters_tools.tile import OriginDict, Point, Tile
from fractal_converters_tools.tiled_image import TiledImage
from ome_types import from_xml
from tifffile import imread


class TiffLoader:
    def __init__(self, image: Any, data_dir: Path):
        self.image = image
        self.data_dir = data_dir

    def __call__(self) -> np.ndarray:
        """Return the full tile."""
        shape_t = len(
            np.unique([tif.first_t for tif in self.image.pixels.tiff_data_blocks])
        )
        shape_z = len(
            np.unique([tif.first_z for tif in self.image.pixels.tiff_data_blocks])
        )
        shape_c = len(
            np.unique([tif.first_c for tif in self.image.pixels.tiff_data_blocks])
        )

        first_acq = self.image.pixels.tiff_data_blocks[0]
        im = imread(self.data_dir / first_acq.uuid.file_name)

        if im.ndim != 2:
            raise ValueError("Only 2D tiff files are currently supported.")

        shape_y, shape_x = im.shape

        full_tile = np.zeros(
            (shape_t, shape_c, shape_z, shape_y, shape_x), dtype=im.dtype
        )
        full_tile[first_acq.first_t, first_acq.first_z, first_acq.first_c] = im

        for tif in self.image.pixels.tiff_data_blocks[1:]:
            im = imread(self.data_dir / tif.uuid.file_name)
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


def tile_from_ome_image(image, 
                        data_dir: Path,
                        scale_xy: float | None = None,
                        scale_z: float | None = None) -> Tile:
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

    tiff_loader = TiffLoader(image=image, data_dir=data_dir)

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

def parse_scanr_metadata(data_dir: Path,
                         acq_id: int,
                         plate_layout: str = "96-well",
                         channel_names: list[str] | None = None,
                         num_levels: int = 1,
                         ) -> dict[str, TiledImage]:
    """Parse ScanR metadata and return a dictionary of TiledImages."""
    metadata_path = data_dir / "metadata.ome.xml"
    meta = from_xml(metadata_path)
    
    tiled_images = {}
    for image in meta.images:
        well_id, pos_id = extract_well_position_id(image.id)
        well_acq_id = f"{well_id}_{acq_id}"

        tile = tile_from_ome_image(image)

        if well_acq_id not in tiled_images:
            row, column = get_row_column(well_id, plate_layout)
            name = image.name if image.name else f"{well_id}_{pos_id}"
            tiled_images[well_acq_id] = TiledImage(
                name=name,
                row=row,
                column=column,
                acquisition_id=acq_id,
                tiles=[tile],
                channel_names=channel_names,
                num_levels=num_levels
            )
        else:
            tiled_images[well_acq_id].tiles.append(tile)
    return tiled_images