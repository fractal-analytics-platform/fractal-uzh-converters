from pathlib import Path

import pytest

from fractal_uzh_converters.cellvoyager import convert_cellvoyager

from .utils import DATA_DIR, run_converter_test

SNAPSHOT_DIR = DATA_DIR / "Yokogawa-CellVoyager" / "snapshots"
RAW_DIR = DATA_DIR / "Yokogawa-CellVoyager" / "raw"


@pytest.mark.parametrize(
    "init_task_kwargs, snapshot_name",
    [
        (
            {
                "acquisitions": [
                    {
                        "path": str(RAW_DIR / "hcs_1w1p1c1z1t"),
                        "acquisition_id": 0,
                        "image_extension": ".png",
                    }
                ]
            },
            "hcs_1w1p1c1z1t",
        ),
    ],
)
def test_cellvoyager(
    tmp_path: Path,
    init_task_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
    converter_options,
):
    run_converter_test(
        tmp_path=tmp_path,
        api_fn=convert_cellvoyager,
        api_kwargs=init_task_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.yaml",
        update_snapshots=update_snapshots,
        converter_options=converter_options,
    )
