from pathlib import Path

import numpy as np
import yaml
from ngio import open_ome_zarr_plate
from pydantic import BaseModel, Field, model_validator


class ImageAssertionModel(BaseModel):
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    pixelsize: tuple[float, ...]
    types: dict[str, bool] = Field(default_factory=dict)
    attributes: dict[str, str | int | float] = Field(default_factory=dict)
    tables: list[str] = Field(default_factory=list)


def deep_merge(a, b):
    result = a.copy()
    for key, value in b.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class PlateAssertionModel(BaseModel):
    wells: list[str]
    images: dict[str, ImageAssertionModel]

    @model_validator(mode="before")
    def validate_images(cls, values):
        common_assertions = values.pop("images_common", {})
        images = values.get("images", {})
        updated_image_assertions = {}
        for image_path, image_assertions in images.items():
            for key in image_assertions.keys():
                if key in common_assertions:
                    image_assertions = deep_merge(image_assertions, common_assertions)
            updated_image_assertions[image_path] = image_assertions
        values["images"] = updated_image_assertions
        return values


class MultiPlateAssertionModel(BaseModel):
    plates: dict[str, PlateAssertionModel]

    @property
    def expected_parallelization_list_length(self) -> int:
        return sum(
            len(plate_assertions.images) for plate_assertions in self.plates.values()
        )

    def aggregated_types(self) -> dict[str, bool]:
        aggregated_types = {}
        for plate_path, plate_assertions in self.plates.items():
            for image_path, image_assertions in plate_assertions.images.items():
                path = f"{plate_path}/{image_path}"
                aggregated_types[path] = image_assertions.types
        return aggregated_types

    def aggregated_attributes(self) -> dict[str, dict[str, str | int | float]]:
        aggregated_attributes = {}
        for plate_path, plate_assertions in self.plates.items():
            for image_path, image_assertions in plate_assertions.images.items():
                path = f"{plate_path}/{image_path}"
                aggregated_attributes[path] = image_assertions.attributes
        return aggregated_attributes


class ConversionSettingsModel(BaseModel):
    converter: str
    init_task_kwargs: dict


class TestConfigModel(BaseModel):
    conversion_settings: ConversionSettingsModel
    multi_plate_assertions: MultiPlateAssertionModel


def load_yaml_assertions(yaml_path: Path) -> TestConfigModel:
    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)
    return TestConfigModel(**yaml_data)


def plate_after_init_checks(
    *,
    init_output: dict,
    multi_plate_assertions: MultiPlateAssertionModel,
    zarr_dir: Path,
):
    parallelization_list = len(init_output["parallelization_list"])
    expected_length = multi_plate_assertions.expected_parallelization_list_length
    assert parallelization_list == expected_length
    for plate_path, plate_assertions in multi_plate_assertions.plates.items():
        plate_path = zarr_dir / plate_path
        ome_zarr_plate = open_ome_zarr_plate(plate_path)
        ome_zarr_plate.get_wells()
        wells = ome_zarr_plate.get_wells().keys()
        assert set(wells) == set(plate_assertions.wells)


def image_list_updates_checks(
    *,
    image_list_updates: list[dict],
    multi_plate_assertions: MultiPlateAssertionModel,
    zarr_dir: Path,
):
    aggregated_types = multi_plate_assertions.aggregated_types()
    aggregated_attributes = multi_plate_assertions.aggregated_attributes()
    for updates in image_list_updates:
        assert "image_list_updates" in updates
        assert len(updates["image_list_updates"]) == 1
        types_updates = updates["image_list_updates"][0]["types"]
        attribute_updates = updates["image_list_updates"][0]["attributes"]
        zarr_url = Path(updates["image_list_updates"][0]["zarr_url"])
        assert zarr_url.exists()
        image_path = zarr_url.relative_to(zarr_dir).as_posix()
        assert image_path in aggregated_types
        assert zarr_url.exists()
        assert types_updates == aggregated_types[image_path]
        assert attribute_updates == aggregated_attributes[image_path]


def post_compute_checks(
    *, multi_plate_assertions: MultiPlateAssertionModel, zarr_dir: Path
):
    for plate_path, plate_assertions in multi_plate_assertions.plates.items():
        plate_path = zarr_dir / plate_path
        ome_zarr_plate = open_ome_zarr_plate(plate_path)
        images = ome_zarr_plate.get_images()
        for image_path, ome_zarr_image in images.items():
            assert image_path in plate_assertions.images
            image_assertions = plate_assertions.images[image_path]
            image = ome_zarr_image.get_image()
            assert image.axes == image_assertions.axes
            assert image.shape == image_assertions.shape
            assert np.allclose(image.pixel_size.tzyx, image_assertions.pixelsize)
            assert set(ome_zarr_image.list_tables()) == set(image_assertions.tables)
