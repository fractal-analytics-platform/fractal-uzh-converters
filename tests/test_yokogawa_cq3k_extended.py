from pathlib import Path

import pytest

from fractal_uzh_converters.cq3k.convert_cq3k_init_task import (
    convert_cq3k_init_task,
)

from .utils import run_converter_test

EXTENDED_DATA_DIR = Path(__file__).parent / "data-extended"
SNAPSHOT_DIR = EXTENDED_DATA_DIR / "Yokogawa-CQ3K" / "snapshots"
RAW_DIR = EXTENDED_DATA_DIR / "Yokogawa-CQ3K" / "raw"

_DATASETS = [
    "hcs_1w2p1c1z1t_TS_FP_SF_Centered",
    "hcs_1w2p1c1z1t_TS_FP_SF_Grid",
    "hcs_1w2p1c1z1t_TS_SP_Centered",
    "hcs_1w2p1c1z1t_TS_SP_Grid",
    "hcs_2w1p1c1z1t_Channel_MIP_Only",
    "hcs_2w1p1c1z1t_Channel_MIP_SUM",
    "hcs_2w1p1c3z1t_Channel_MIP_Slice",
    "hcs_2w1p1c3z1t_Channel_MIP_Slice_2bin",
    "hcs_2w1p1c3z1t_Channel_MIP_SUM_Slice",
    "hcs_3w1p1c1z1t_Channel_MIP",
    "hcs_3w2p8c1z1t_Channel",
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
def test_cq3k_extended(
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
