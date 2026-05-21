from pathlib import Path

import pytest

from fractal_uzh_converters.imagexpress_hcs.convert_imagexpress_hcs_init_task import (
    convert_imagexpress_hcs_init_task,
)

from .utils import DATA_DIR, run_converter_test

SNAPSHOT_DIR = DATA_DIR / "MolecularDevices-ImageXpressHCSai" / "snapshots"
RAW_DIR = DATA_DIR / "MolecularDevices-ImageXpressHCSai" / "raw"


@pytest.mark.parametrize(
    "init_task_kwargs, snapshot_name",
    [
        (
            {
                "acquisitions": [
                    {
                        "path": str(RAW_DIR / "hcs_1w1s1t1c1z_binning4x4"),
                        "acquisition_id": 0,
                    }
                ]
            },
            "hcs_1w1s1t1c1z_binning4x4",
        ),
    ],
)
def test_imagexpress_hcs(
    tmp_path: Path,
    init_task_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
):
    run_converter_test(
        tmp_path=tmp_path,
        init_task_fn=convert_imagexpress_hcs_init_task,
        init_task_kwargs=init_task_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.yaml",
        update_snapshots=update_snapshots,
    )
