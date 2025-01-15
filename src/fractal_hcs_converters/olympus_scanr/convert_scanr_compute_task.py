"""ScanR to OME-Zarr conversion task compute."""

import logging
from pathlib import Path
from typing import Literal

from fractal_converters_tools.stitching import standard_stitching_pipe
from fractal_converters_tools.tiled_image import TiledImage
from ngio import NgffImage
from ngio.core.roi import RasterCooROI, WorldCooROI
from ngio.core.utils import PixelSize, create_empty_ome_zarr_image
from pydantic import validate_call

from fractal_hcs_converters.olympus_scanr.convert_scanr_init_task import (
    ConvertScanrInitArgs,
)
from fractal_hcs_converters.olympus_scanr.utils import parse_scanr_metadata

logger = logging.getLogger(__name__)


def build_tiled_image(
    zarr_dir: Path,
    tiled_image: TiledImage,
    mode: Literal["auto", "grid", "free", "none"] = "auto",
    swap_xy: bool = False,
    invert_x: bool = False,
    invert_y: bool = False,
) -> tuple[str, bool, str]:
    """Build a tiled ome-zarr image from a TiledImage object."""
    tiles = tiled_image.tiles
    tiles = standard_stitching_pipe(
        tiles, mode=mode, swap_xy=swap_xy, invert_x=invert_x, invert_y=invert_y
    )
    shape_x = max(int(tile.bot_r.x) for tile in tiles)
    shape_y = max(int(tile.bot_r.y) for tile in tiles)

    tile_shape = tiles[0].shape()
    shape_t = tile_shape["t"]
    shape_c = tile_shape["c"]
    shape_z = tile_shape["z"]

    on_disk_shape = (shape_t, shape_c, shape_z, shape_y, shape_x)
    on_disk_axis = ("t", "c", "z", "y", "x")
    sample_tile = tiles[0]
    sample_tile_data = sample_tile.load()
    tile_shape = sample_tile_data.shape
    chunk_shape = (1, 1, 1, sample_tile_data.shape[-2], sample_tile_data.shape[-1])
    
    if shape_t == 1:
        chunk_shape = (1, 1, sample_tile_data.shape[-2], sample_tile_data.shape[-1])
        on_disk_axis = ("c", "z", "y", "x")
        on_disk_shape = (shape_c, shape_z, shape_y, shape_x)
        
    tile_dtype = sample_tile_data.dtype
    
    tile_pixel_sizes = PixelSize(
        x=sample_tile.xy_scale,
        y=sample_tile.xy_scale,
        z=sample_tile.z_scale,
    )
    logger.info(f"Building tiled image with shape {on_disk_shape}.")
    logger.info(f"Chunk shape: {chunk_shape}")
    logger.info(f"Tiles shape: {tile_shape}")
    logger.info(f"Tile dtype: {sample_tile_data.dtype}")
    logger.info(f"Tile pixel sizes: {tile_pixel_sizes}")

    new_zarr_url = str(zarr_dir / tiled_image.acquisition_path)

    create_empty_ome_zarr_image(
        store=new_zarr_url,
        on_disk_shape=on_disk_shape,
        on_disk_axis=on_disk_axis,
        chunks=chunk_shape,
        dtype=tile_dtype,
        pixel_sizes=tile_pixel_sizes,
        channel_labels=tiled_image.channel_names,
        channel_wavelengths=tiled_image.channel_names
    )
    logger.info(f"Created empty OME-Zarr image at {new_zarr_url}.")

    ngff_image = NgffImage(store=new_zarr_url)
    well_roi_table = ngff_image.tables.new("well_ROI_table", table_type="roi_table")
    well_roi = WorldCooROI(
        x=0,
        y=0,
        z=0,
        x_length=shape_x * tile_pixel_sizes.x,
        y_length=shape_y * tile_pixel_sizes.y,
        z_length=shape_z * tile_pixel_sizes.z,
        unit="micrometer",
        infos={"FieldIndex": "Well"},
    )
    well_roi_table.set_rois([well_roi])
    well_roi_table.consolidate()
    logger.info("Created well ROI.")

    image = ngff_image.get_image()
    fov_roi_table = ngff_image.tables.new("FOV_ROI_table", table_type="roi_table")
    _fov_rois = []
    for i, tile in enumerate(tiles):
        # Create the ROI for the tile
        roi = RasterCooROI(
            x=int(tile.top_l.x),
            y=int(tile.top_l.y),
            z=int(tile.top_l.z),
            x_length=int(tile.diag.x),
            y_length=int(tile.diag.y),
            z_length=int(tile.diag.z),
            original_roi=well_roi,
        ).to_world_coo_roi(pixel_size=tile_pixel_sizes)
        roi.infos = {"FieldIndex": f"FOV_{i}", **tile.origin._asdict()}
        _fov_rois.append(roi)

        # Load the whole tile and set the data in the image
        tile_data = tile.load()
        if shape_t == 1:
            tile_data = tile_data[0]
        image.set_array_from_roi(tile_data, roi)

    logger.info("Image data at high resolution set.")
    image.consolidate(order=1)
    logger.info("Image data consolidated.")
    ngff_image.update_omero_window(start_percentile=1, end_percentile=99.9)
    fov_roi_table.set_rois(_fov_rois)
    fov_roi_table.consolidate()
    logger.info("Created FOV ROIs.")
    return new_zarr_url, image.is_3d, f"{tiled_image.row}{tiled_image.column}"


@validate_call
def convert_scanr_compute_task(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: ConvertScanrInitArgs,
):
    """Initialize the task to convert a LIF plate to OME-Zarr.

    Args:
        zarr_url (str): URL to the OME-Zarr file.
        init_args (ConvertScanrInitArgs): Arguments for the initialization task.
    """
    acq = init_args.acquision
    acq_path = Path(acq.path)
    tiled_images = parse_scanr_metadata(acq_path, acq_id=acq.acquisition_id)
    tiled_image = tiled_images[init_args.tiled_image_id]
    logger.info(f"Converting {acq_path} to OME-Zarr.")
    new_zarr_url, is_3d, well = build_tiled_image(
        zarr_dir=Path(zarr_url),
        tiled_image=tiled_image,
        mode=init_args.advanced_options.tiling_mode,
        swap_xy=init_args.advanced_options.swap_xy,
        invert_x=init_args.advanced_options.invert_x,
        invert_y=init_args.advanced_options.invert_y,
    )

    p_types = {"is_3D": is_3d}
    attributes = {"well": well, "plate": init_args.plate_name}

    return {
        "image_list_updates": [
            {"zarr_url": new_zarr_url, "types": p_types, "attributes": attributes}
        ]
    }


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=convert_scanr_compute_task, logger_name=logger.name)
