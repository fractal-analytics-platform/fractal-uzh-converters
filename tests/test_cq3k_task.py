from pathlib import Path

import pytest

from fractal_uzh_converters.common import image_in_plate_compute_task
from fractal_uzh_converters.cq3k.convert_cq3k_init_task import (
    convert_cq3k_init_task,
)

from .utils import (
    image_list_updates_checks,
    load_yaml_assertions,
    plate_after_init_checks,
    post_compute_checks,
)


@pytest.mark.parametrize(
    "test_config_path",
    ["data/CQ3K/configs/2w1p1c1z1t_mip.yaml"],
)
def test_cq3k(tmp_path: Path, test_config_path: str):
    """Test the CQ3K converter using config files."""
    zarr_dir = tmp_path / "test_zarr_dir"
    _test_config_path = Path(__file__).parent / test_config_path
    config = load_yaml_assertions(_test_config_path)
    output = convert_cq3k_init_task(
        zarr_dir=str(zarr_dir), **config.conversion_settings.init_task_kwargs
    )
    plate_after_init_checks(
        init_output=output,
        multi_plate_assertions=config.multi_plate_assertions,
        zarr_dir=zarr_dir,
    )
    updates_list = []
    for p in output["parallelization_list"]:
        update = image_in_plate_compute_task(**p)
        updates_list.append(update)

    image_list_updates_checks(
        image_list_updates=updates_list,
        multi_plate_assertions=config.multi_plate_assertions,
        zarr_dir=zarr_dir,
    )
    post_compute_checks(
        multi_plate_assertions=config.multi_plate_assertions, zarr_dir=zarr_dir
    )
