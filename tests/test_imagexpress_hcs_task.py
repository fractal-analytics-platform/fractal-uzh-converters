from pathlib import Path

import pytest

from fractal_uzh_converters.imagexpress_hcs.convert_imagexpress_hcs_init_task import (
    convert_imagexpress_hcs_init_task,
)

from .utils import DATA_DIR, run_converter_test

SNAPSHOT_DIR = DATA_DIR / "MDImageXpressHCSAI" / "snapshots"


@pytest.mark.parametrize(
    "init_task_kwargs, snapshot_name",
    [
        (
            {
                "acquisitions": [
                    {
                        "path": f"{DATA_DIR}/MDImageXpressHCSAI"
                        "/MD_reference_acquisitions"
                        "/1w1s1t1c1z_binning4x4"
                        "/test_data_20260302_135747",
                        "acquisition_id": 0,
                        "plate_name": "Plate",
                    }
                ]
            },
            "1w1s1t1c1z_binning4x4",
        ),
    ],
)
def test_md_imagexpress(
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
