import shutil
from pathlib import Path

import pytest
from devtools import debug
from fractal_tasks_core.channels import ChannelInputModel

from fractal_hcs_converters.thresholding_label_task import thresholding_label_task


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    source_dir = (Path(__file__).parent / "data/ngff_example/my_image").as_posix()
    dest_dir = (tmp_path / "my_image").as_posix()
    debug(source_dir, dest_dir)
    shutil.copytree(source_dir, dest_dir)
    return dest_dir


def test_thresholding_label_task(test_data_dir):
    thresholding_label_task(
        zarr_url=test_data_dir,
        threshold=180,
        channel=ChannelInputModel(label="DAPI")
    )
