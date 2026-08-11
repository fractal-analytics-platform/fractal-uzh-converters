from pathlib import Path

import pytest
from ome_zarr_converters_tools.testing import run_converter_test

from fractal_uzh_converters.scanr import convert_scanr

from .utils import DATA_DIR

SNAPSHOT_DIR = DATA_DIR / "Evident-scanR" / "snapshots"
RAW_DIR = DATA_DIR / "Evident-scanR" / "raw"


@pytest.mark.parametrize(
    "api_kwargs, snapshot_name",
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
    ],
)
def test_scanr(
    tmp_path: Path,
    api_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
    converter_options,
):
    run_converter_test(
        tmp_path=tmp_path,
        api_fn=convert_scanr,
        api_kwargs=api_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.json",
        update_snapshots=update_snapshots,
        converter_options=converter_options,
    )
