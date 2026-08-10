from pathlib import Path

import pytest
from ome_zarr_converters_tools.testing import run_converter_test

from fractal_uzh_converters.cq3k import convert_cq3k

EXTENDED_DATA_DIR = Path(__file__).parent / "data-extended"
SNAPSHOT_DIR = EXTENDED_DATA_DIR / "Yokogawa-CQ3K" / "snapshots"
RAW_DIR = EXTENDED_DATA_DIR / "Yokogawa-CQ3K" / "raw"

_DATASETS = [
    "20251201T133446_Channel_Well_test",
    "20251201T134617_Channel_WellTestC5F9_MIP_SUM",
    "20251201T134728_Channel_WellTestC5F9_MIP_SUM_Slice",
    "20251201T134840_Channel_WellTestC5F9_MIP_Only",
    "20251201T134956_Channel_WellTestC5F9_MIP_Slice",
    "20251201T135103_Channel_WellTestC5F9_MIP_Slice_2bin",
    "20251201T135346_Channel_WellTestC5H12M19_MIP",
    "20251201T140917_BVC_TS_Test_FP",
    # Single-record `.mlf` — the regression case for the widened
    # `MeasurementData.measurement_record` type.
    "20251201T140949_BVC_TS_Test_SP_grid",
    "20251201T141350_BVC_TS_Test_SP_centered",
    "20260617T083657_2026-06-17_mNgn3_dox_fractal_test",
]


@pytest.mark.extended
@pytest.mark.parametrize(
    "init_task_kwargs, snapshot_name",
    [
        (
            {"acquisitions": [{"path": str(RAW_DIR / name), "acquisition_id": 0}]},
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
    converter_options,
):
    run_converter_test(
        tmp_path=tmp_path,
        api_fn=convert_cq3k,
        api_kwargs=init_task_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.json",
        update_snapshots=update_snapshots,
        converter_options=converter_options,
    )
