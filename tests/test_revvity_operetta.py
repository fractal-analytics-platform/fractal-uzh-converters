from pathlib import Path

import pytest

from fractal_uzh_converters.operetta.convert_operetta_init_task import (
    convert_operetta_init_task,
)

from .utils import DATA_DIR, run_converter_test

SNAPSHOT_DIR = DATA_DIR / "Revvity-Operetta" / "snapshots"
RAW_DIR = DATA_DIR / "Revvity-Operetta" / "raw"


@pytest.mark.parametrize(
    "init_task_kwargs, snapshot_name",
    [
        (
            {
                "acquisitions": [
                    {
                        "path": str(RAW_DIR / "hcs_1w1p1c1z1t"),
                        "acquisition_id": 0,
                    }
                ]
            },
            "hcs_1w1p1c1z1t",
        ),
        (
            {
                "acquisitions": [
                    {
                        "path": str(RAW_DIR / "hcs_1w1p1c1z1t"),
                        "acquisition_id": 0,
                        "advanced": {
                            "condition_table_path": str(
                                RAW_DIR / "hcs_1w1p1c1z1t" / "condition_table.csv"
                            )
                        },
                    }
                ]
            },
            "hcs_1w1p1c1z1t_with_condition_table",
        ),
    ],
)
def test_operetta(
    tmp_path: Path,
    init_task_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
    converter_options,
):
    run_converter_test(
        tmp_path=tmp_path,
        init_task_fn=convert_operetta_init_task,
        init_task_kwargs=init_task_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.yaml",
        update_snapshots=update_snapshots,
        converter_options=converter_options,
    )
