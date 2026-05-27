from pathlib import Path

import pytest

from fractal_uzh_converters.imagexpress_hcs.convert_imagexpress_hcs_init_task import (
    convert_imagexpress_hcs_init_task,
)

from .utils import run_converter_test

EXTENDED_DATA_DIR = Path(__file__).parent / "data-extended"
SNAPSHOT_DIR = EXTENDED_DATA_DIR / "MolecularDevices-ImageXpressHCSai" / "snapshots"
RAW_DIR = EXTENDED_DATA_DIR / "MolecularDevices-ImageXpressHCSai" / "raw"

_DATASETS: list[str] = [
    "hcs_1w1p1c1z1t",
    "hcs_1w1p1c1z1t_4bin",
    "hcs_1w1p2c4z3t_4bin",
    "hcs_2w4p2c1z1t_4bin",
    "hcs_2w4p2c4z4t_4bin",
    "hcs_1w4p1c1z1t_4bin_Overlap",
    "hcs_1w4p1c1z1t_4bin_Overlap_MontageNone",
    "hcs_1w4p1c1z1t_4bin_Overlap_MontageStitch",
    "hcs_1w4p1c1z1t_4bin_Tiled",
    "hcs_1w4p1c1z1t_4bin_Tiled_FOV576x288",
    "hcs_1w4p1c1z1t_4bin_Tiled_MontageNone",
    "hcs_1w4p1c1z1t_4bin_Tiled_MontageTile",
    "hcs_1w6p1c4z1t_4bin_Tiled_MontageStitch_Projection",
    "hcs_1w6p1c4z1t_4bin_Tiled_MontageStitch_ZStack",
    "hcs_1w1p2c1z1t_4bin_MIP",
    "hcs_1w1p2c4z1t_4bin_Mixed_Stack_MIP",
    "hcs_1w1p2c4z1t_4bin_Mixed_Stack_SinglePlane",
    "hcs_1w1p2c4z1t_4bin_Mixed_StackOnly_MIP",
    "hcs_1w1p2c1z1t_4bin_SinglePlane",
    "hcs_1w1p2c4z1t_4bin_Stack_MIP",
    "hcs_1w1p2c4z1t_4bin_Stack",
    "hcs_1w1p2c1z3t_4bin",
    "hcs_1w1p2c1z3t_4bin_Mix_EveryTP_1stTP",
    "hcs_2w4p2c4z4t_4bin_Cancelled",
    "hcs_1w2p1c1z1t_4bin_1Region",
    "hcs_1w3p1c1z1t_4bin_2Regions",
    "hcs_1w4p1c1z1t_4bin_2Regions_Stitch",
    "hcs_2w2p1c1z1t_4bin_DiffRgnsPerWell",
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
def test_imagexpress_hcs_extended(
    tmp_path: Path,
    init_task_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
    converter_options,
):
    run_converter_test(
        tmp_path=tmp_path,
        init_task_fn=convert_imagexpress_hcs_init_task,
        init_task_kwargs=init_task_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.yaml",
        update_snapshots=update_snapshots,
        converter_options=converter_options,
    )
