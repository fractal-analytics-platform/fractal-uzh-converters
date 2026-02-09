from pathlib import Path

import pytest

from fractal_uzh_converters.cq3k.convert_cq3k_init_task import (
    convert_cq3k_init_task,
)

from .utils import run_converter_test

SNAPSHOT_DIR = Path(__file__).parent / "data" / "CQ3K" / "snapshots"


@pytest.mark.parametrize(
    "init_task_kwargs, snapshot_name",
    [
        (
            {
                "acquisitions": [
                    {
                        "path": "tests/data/CQ3K"
                        "/CQ3K_reference_acquisitions"
                        "/2w1p1c1z1t_mip",
                        "acquisition_id": 0,
                    }
                ]
            },
            "2w1p1c1z1t_mip",
        ),
    ],
)
def test_cq3k(
    tmp_path: Path,
    init_task_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
):
    run_converter_test(
        tmp_path=tmp_path,
        init_task_fn=convert_cq3k_init_task,
        init_task_kwargs=init_task_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.yaml",
        update_snapshots=update_snapshots,
    )
