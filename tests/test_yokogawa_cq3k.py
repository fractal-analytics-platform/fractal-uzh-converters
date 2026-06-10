from pathlib import Path

import pytest

from fractal_uzh_converters.cq3k import convert_cq3k

from .utils import DATA_DIR, run_converter_test

SNAPSHOT_DIR = DATA_DIR / "Yokogawa-CQ3K" / "snapshots"
RAW_DIR = DATA_DIR / "Yokogawa-CQ3K" / "raw"


@pytest.mark.parametrize(
    "init_task_kwargs, snapshot_name",
    [
        (
            {
                "acquisitions": [
                    {
                        "path": str(RAW_DIR / "hcs_2w1p1c1z1t_mip"),
                        "acquisition_id": 0,
                    }
                ]
            },
            "hcs_2w1p1c1z1t_mip",
        ),
    ],
)
def test_cq3k(
    tmp_path: Path,
    init_task_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
    converter_options,
):
    run_converter_test(
        tmp_path=tmp_path,
        api_fn=convert_cq3k,
        api_kwargs=init_task_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.yaml",
        update_snapshots=update_snapshots,
        converter_options=converter_options,
    )
