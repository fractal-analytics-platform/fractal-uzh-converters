from pathlib import Path

import pytest

from fractal_uzh_converters.scanr.convert_scanr_init_task import (
    convert_scanr_init_task,
)

from .utils import run_converter_test

EXTENDED_DATA_DIR = Path(__file__).parent / "data-extended"
SNAPSHOT_DIR = EXTENDED_DATA_DIR / "Evident-scanR" / "snapshots"
RAW_DIR = EXTENDED_DATA_DIR / "Evident-scanR" / "raw"

_DATASETS = [
    "hcs_1w1p1c1z1t",
    "hcs_2w4p1c1z1t_1000spacing",
    "hcs_2w4p1c1z1t_10overlap",
    "hcs_2w4p1c1z1t_nospacing",
    "hcs_2w4p4c1z1t_dual",
    "hcs_2w4p4c1z1t_seq",
    "hcs_2w4p4c5z1t_seq",
    "hcs_2w4p4c5z4t_dual",
    "hcs_2w4p4c5z3t_seq",
]


@pytest.mark.extended
@pytest.mark.parametrize(
    "init_task_kwargs, snapshot_name",
    [
        (
            {
                "acquisitions": [
                    {"path": str(RAW_DIR / name), "acquisition_id": 0}
                ]
            },
            name,
        )
        for name in _DATASETS
    ],
)
def test_scanr_extended(
    tmp_path: Path,
    init_task_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
):
    run_converter_test(
        tmp_path=tmp_path,
        init_task_fn=convert_scanr_init_task,
        init_task_kwargs=init_task_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.yaml",
        update_snapshots=update_snapshots,
    )
