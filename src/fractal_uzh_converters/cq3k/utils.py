"""Utility functions for Yokogawa CQ3K data."""

from logging import getLogger
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import xmltodict
from ngio import PixelSize
from ome_zarr_converters_tools import PlatePathBuilder, Point, Tile, TiledImage, Vector
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_pascal
from tifffile import imread

logger = getLogger(__name__)


class CQ3KTiffLoader:
    """Load a full tile from a list of tiff images."""

    def __init__(
        self,
        image: list["ImageMeasurementRecord"],
        data_dir: Path,
        shapes: dict[str, int],
    ):
        """Initialize the TiffLoader."""
        self.image = image
        self.data_dir = data_dir
        self.shapes = shapes

    @property
    def dtype(self) -> "str":
        """Return the dtype of the tiff files."""
        first_acq = self.image[0].value
        im = imread(self.data_dir / first_acq)
        return str(im.dtype)

    def load(self) -> np.ndarray:
        """Return the full tile."""
        first_acq = self.image[0]
        im = imread(self.data_dir / first_acq.value)

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
        full_tile = np.zeros(shape=tile_shape, dtype=self.dtype)
        full_tile[first_acq.time_point - 1, first_acq.ch - 1, first_acq.z_index - 1] = (
            im
        )

        for tif in self.image[1:]:
            try:
                im = imread(self.data_dir / tif.value)
            except FileNotFoundError:
                logger.warning(f"Tiff tile not found: {self.data_dir / tif.value}")
                continue
            except Exception as e:
                logger.error(f"Error loading tiff tile: {e}")
                continue

            full_tile[tif.time_point - 1, tif.ch - 1, tif.z_index - 1] = im

        return full_tile


######################################################################
#
# Pydantic models for parsing CQ3K metadata
# are adapted from https://github.com/fmi-faim/cellvoyager-types
#
######################################################################


class Base(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_pascal,
        extra="forbid",
    )


class MeasurementRecordBase(Base):
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
    type: Literal["ERR"]


class MeasurementData(Base):
    xmlns: Annotated[dict, Field(alias="xmlns")]
    version: Literal["1.0"]
    measurement_record: list[ImageMeasurementRecord | ErrorMeasurementRecord] | None = (
        None
    )


class MeasurementSamplePlate(Base):
    name: str
    well_plate_file_name: str
    well_plate_product_file_name: str


class MeasurementChannel(Base):
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
    measurement_channel: list[MeasurementChannel]


def _parse(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return xmltodict.parse(
            f.read(),
            process_namespaces=True,
            namespaces={"http://www.yokogawa.co.jp/BTS/BTSSchema/1.0": None},
            attr_prefix="",
            cdata_key="Value",
        )


def _load_models(path: Path) -> tuple[MeasurementData, MeasurementDetail]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    if not path.is_dir():
        raise ValueError(
            f"{path} is not a directory. Please provide a directory path containing the"
            "MeasurementData.mlf, and MeasurementDetail.mrf files."
        )
    mlf_dict = _parse(path / "MeasurementData.mlf")
    mrf_dict = _parse(path / "MeasurementDetail.mrf")
    mlf = MeasurementData(**mlf_dict["MeasurementData"])
    mrf = MeasurementDetail(**mrf_dict["MeasurementDetail"])
    return mlf, mrf


def parse_cq3k_metadata(
    data_dir: Path,
    acq_id: int,
    plate_name: str,
    channel_names: list[str] | None = None,
    channel_wavelengths: list[str] | None = None,
) -> list[TiledImage]:
    """Parse CQ3K metadata and return a list of TiledImages."""
    data, detail = _load_models(data_dir)

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    plates_groups = {}

    assert data.measurement_record is not None
    z_type_keys = set()
    for img in data.measurement_record:
        assert isinstance(img, ImageMeasurementRecord)
        z_type = img.z_image_processing
        z_type_keys.add(z_type)

        row = alphabet[img.row - 1]
        key = (z_type, row, img.column)

        if key not in plates_groups:
            plates_groups[key] = []
        plates_groups[key].append(img)

    _channel = detail.measurement_channel[0]
    pixel_size_x, pixel_size_y = (
        _channel.horizontal_pixel_dimension,
        _channel.vertical_pixel_dimension,
    )
    shape_x, shape_y = (_channel.horizontal_pixels, _channel.vertical_pixels)

    tiled_images = []
    for z_type in z_type_keys:
        # Create TiledImages for each well in the plate
        plates_groups_z_type = {
            (key[1], key[2]): value
            for key, value in plates_groups.items()
            if key[0] == z_type
        }
        if z_type is not None:
            plate_name_z_type = f"{plate_name}_{z_type}"
        else:
            plate_name_z_type = plate_name

        for (row, column), images in plates_groups_z_type.items():
            tiled_image = TiledImage(
                name=f"Acq{acq_id}_Well{row}{column}",
                path_builder=PlatePathBuilder(
                    plate_name=plate_name_z_type,
                    row=row,
                    column=column,
                    acquisition_id=acq_id,
                ),
                channel_names=channel_names,
                wavelength_ids=channel_wavelengths,
            )

            fov_index = {img.field_index for img in images}

            for fov_idx in fov_index:
                imgs = [img for img in images if img.field_index == fov_idx]
                z = np.unique([img.z for img in imgs])
                if len(z) == 1:
                    z_spacing = 1.0
                else:
                    # Add check for uniform spacing
                    z_spacing = float(z[1] - z[0])

                pixel_size = PixelSize(
                    x=pixel_size_x,
                    y=pixel_size_y,
                    z=z_spacing,
                    t=1,
                    space_unit="micrometer",
                )

                t = [img.time_point for img in imgs]
                c = [img.ch for img in imgs]
                z = [img.z_index for img in imgs]
                x = [img.x for img in imgs]
                y = [img.y for img in imgs]
                top_l = Point(x=min(x), y=min(y), z=0, c=0, t=0)
                diag = Vector(
                    x=shape_x * pixel_size.x,
                    y=shape_y * pixel_size.y,
                    z=len(set(z)),
                    c=len(set(c)),
                    t=len(set(t)),
                )
                shape_tile = (
                    len(set(t)),
                    len(set(c)),
                    len(set(z)),
                    shape_y,
                    shape_x,
                )
                tile = Tile(
                    top_l=top_l,
                    diag=diag,
                    pixel_size=pixel_size,
                    shape=shape_tile,
                    data_loader=CQ3KTiffLoader(
                        image=imgs,
                        data_dir=data_dir,
                        shapes={
                            "t": shape_tile[0],
                            "c": shape_tile[1],
                            "z": shape_tile[2],
                            "y": shape_tile[3],
                            "x": shape_tile[4],
                        },
                    ),
                )
                tiled_image.add_tile(tile)

            tiled_images.append(tiled_image)
    return tiled_images
