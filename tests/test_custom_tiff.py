"""Tests for the custom TIFF converters (HCS plate and single image)."""

from pathlib import Path

import pytest

from fractal_uzh_converters.custom_tiff import (
    convert_hcs_tiff_init_task,
    convert_single_tiff_init_task,
)
from tests.utils import DATA_DIR, run_converter_test, run_single_image_converter_test

RAW_DIR = DATA_DIR / "CustomTiff" / "raw"
SNAPSHOT_DIR = DATA_DIR / "CustomTiff" / "snapshots"

_HCS_DATASETS = ["hcs_1w2p2c1z1t"]
_SINGLE_DATASETS = ["img_2p2c1z1t"]


@pytest.mark.parametrize(
    "init_task_kwargs, snapshot_name",
    [
        pytest.param(
            {"acquisitions": [{"path": str(RAW_DIR / name)}]},
            name,
        )
        for name in _HCS_DATASETS
    ],
)
def test_hcs_tiff(
    tmp_path: Path,
    init_task_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
    converter_options,
):
    run_converter_test(
        tmp_path=tmp_path,
        init_task_fn=convert_hcs_tiff_init_task,
        init_task_kwargs=init_task_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.yaml",
        update_snapshots=update_snapshots,
        converter_options=converter_options,
    )


@pytest.mark.parametrize(
    "init_task_kwargs, snapshot_name",
    [
        pytest.param(
            {"acquisitions": [{"path": str(RAW_DIR / name)}]},
            name,
        )
        for name in _SINGLE_DATASETS
    ],
)
def test_single_tiff(
    tmp_path: Path,
    init_task_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
    converter_options,
):
    run_single_image_converter_test(
        tmp_path=tmp_path,
        init_task_fn=convert_single_tiff_init_task,
        init_task_kwargs=init_task_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.yaml",
        update_snapshots=update_snapshots,
        converter_options=converter_options,
    )
