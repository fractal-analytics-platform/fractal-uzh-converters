from pathlib import Path

import pytest
from ome_zarr_converters_tools.testing import run_converter_test

from fractal_uzh_converters.yokogawa.cq3k import convert_cq3k

EXTENDED_DATA_DIR = Path(__file__).parent / "data-extended"
SNAPSHOT_DIR = EXTENDED_DATA_DIR / "Yokogawa-CQ3K" / "snapshots"
RAW_DIR = EXTENDED_DATA_DIR / "Yokogawa-CQ3K" / "raw"

_DATASETS = [
    # Single-record `.mlf` — the regression case for the widened
    # `MeasurementData.measurement_record` type.
    "hcs_1w1p1c1z1t_SearchFirst_SP_Grid",
    "hcs_1w2p1c1z1t_SearchFirst_FP",
    "hcs_1w2p1c1z1t_SearchFirst_SP_Centered",
    "hcs_2w1p1c1z1t_MIP_Only",
    "hcs_2w1p1c1z1t_MIP_SUM",
    "hcs_2w1p1c33z1t_MIP_SUM_Slice",
    "hcs_2w1p1c33z1t_MIP_Slice",
    "hcs_2w1p1c33z1t_MIP_Slice_2bin",
    "hcs_2w4p2c10z1t_Slices_MIP_MinIP",
    "hcs_3w1p1c1z1t_MIP_384Well",
    "hcs_3w2p4c1z1t_Channels_MIP_MinIP",
]


@pytest.mark.extended
@pytest.mark.parametrize(
    "api_kwargs, snapshot_name",
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
    api_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
    converter_options,
):
    run_converter_test(
        tmp_path=tmp_path,
        api_fn=convert_cq3k,
        api_kwargs=api_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.json",
        update_snapshots=update_snapshots,
        converter_options=converter_options,
    )
