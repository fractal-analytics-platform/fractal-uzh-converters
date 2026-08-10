from pathlib import Path

import pytest
from ome_zarr_converters_tools.testing import run_converter_test

from fractal_uzh_converters.yokogawa.cellvoyager import convert_cellvoyager

EXTENDED_DATA_DIR = Path(__file__).parent / "data-extended"
SNAPSHOT_DIR = EXTENDED_DATA_DIR / "Yokogawa-CellVoyager" / "snapshots"
RAW_DIR = EXTENDED_DATA_DIR / "Yokogawa-CellVoyager" / "raw"

_DATASETS = [
    "hcs_2w1p3c9z1t_DuplicateTargets",
    "hcs_2w1p3c9z1t_SequentialChannels",
    "hcs_2w1p4c9z1t_SimultaneousChannels",
    "hcs_2w2p1c9z1t_TimelinesPerWellChannels",
    "hcs_2w2p3c9z1t_PartialTile",
    "hcs_2w2p3c9z1t_PartialTile_DuplicateTargets",
    "hcs_2w4p1c1z1t_SearchFirst_Seed",
    "hcs_2w20p3c9z1t_SearchFirst",
    "hcs_3w2p2c9z1t_TimelinesSharedChannel",
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
def test_cellvoyager_extended(
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
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.json",
        update_snapshots=update_snapshots,
        converter_options=converter_options,
    )
