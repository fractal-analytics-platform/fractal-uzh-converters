from pathlib import Path

import pytest

from fractal_uzh_converters.evident_scanr.convert_scanr_init_task import (
    convert_scanr_init_task,
)

from .utils import DATA_DIR, run_converter_test

SNAPSHOT_DIR = DATA_DIR / "EvidentScanR" / "snapshots"


@pytest.mark.parametrize(
    "init_task_kwargs, snapshot_name",
    [
        (
            {
                "acquisitions": [
                    {
                        "path": f"{DATA_DIR}/EvidentScanR"
                        "/EvidentScanR_reference_acquisitions"
                        "/1w1p1c1z1t",
                        "acquisition_id": 0,
                    }
                ]
            },
            "1w1p1c1z1t",
        ),
    ],
)
def test_scanr(
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
