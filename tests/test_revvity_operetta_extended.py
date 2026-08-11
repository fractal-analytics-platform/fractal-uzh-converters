from pathlib import Path

import pytest
from ome_zarr_converters_tools.testing import run_converter_test

from fractal_uzh_converters.operetta import convert_operetta

EXTENDED_DATA_DIR = Path(__file__).parent / "data-extended"
SNAPSHOT_DIR = EXTENDED_DATA_DIR / "Revvity-Operetta" / "snapshots"
RAW_DIR = EXTENDED_DATA_DIR / "Revvity-Operetta" / "raw"

_DATASETS: list[str] = [
    "hcs_1w1p1c1z1t",
    "hcs_1w1p1c5z1t_zStack",
    "hcs_1w1p3c1z1t_MultiChannel",
    "hcs_1w4p1c1z1t_MultipleFOV",
    "hcs_1w4p1c1z1t_15Overlap",
    "hcs_1w9p1c1z1t_15Overlap_20x",
    "hcs_1w4p1c5z1t_MultipleFOV_zStack",
    "hcs_1w1p1c1z1t_Multiplex1",
    "hcs_1w1p1c1z1t_Multiplex1_1",
    "hcs_1w1p1c1z1t_Multiplex2",
    "hcs_1w1p1c1z1t_Multiplex2_1",
    "hcs_1w25p1c1z1t_LotsOfFOV_20x",
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
def test_operetta_extended(
    tmp_path: Path,
    api_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
    converter_options,
):
    run_converter_test(
        tmp_path=tmp_path,
        api_fn=convert_operetta,
        api_kwargs=api_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.json",
        update_snapshots=update_snapshots,
        converter_options=converter_options,
    )
