import sys
from pathlib import Path

import pytest
from ome_zarr_converters_tools.testing import run_converter_test

from fractal_uzh_converters.scanr import convert_scanr
from fractal_uzh_converters.scanr._utils import ScanRAcquisitionModel

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


def test_ScanRAcquisitionModel_field_validator():
    x = ScanRAcquisitionModel(path="/some/path/data")
    assert x.path == "/some/path"

    x = ScanRAcquisitionModel(path="C:\\some\\path\\data\\hcs_123\\data")
    if sys.platform == "win32":
        assert x.path == "C:\\some\\path\\data\\hcs_123"
    else:
        # Outside Windows "\" is not a separator, so the path is left untouched
        assert x.path == "C:\\some\\path\\data\\hcs_123\\data"
