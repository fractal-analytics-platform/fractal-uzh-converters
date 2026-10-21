from pathlib import Path

import pytest
from ome_zarr_converters_tools.testing import run_converter_test
from pydantic import ValidationError

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


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/some/path", "/some/path"),
        ("/some/path/", "/some/path"),
        ("/some/path/data", "/some/path"),
        ("/some/path/data/", "/some/path"),
        # Only the trailing `data` component is stripped
        ("/some/data/hcs_123/data", "/some/data/hcs_123"),
        ("/some/path/data_x", "/some/path/data_x"),
        ("s3://bucket/acq/data", "s3://bucket/acq"),
        # Windows paths are handled the same on every host
        ("C:\\some\\path\\data\\hcs_123\\data", "C:/some/path/data/hcs_123"),
        ("C:\\some\\path\\data\\", "C:/some/path"),
        ("\\\\server\\share\\acq\\data", "//server/share/acq"),
    ],
)
def test_acquisition_model_path_validator(path: str, expected: str):
    model = ScanRAcquisitionModel(path=path, acquisition_id=0)
    assert model.path == expected


def test_acquisition_model_path_validator_bare_data_raises():
    # A bare `data` has no parent to fall back to
    with pytest.raises(ValidationError, match="No parent directory"):
        ScanRAcquisitionModel(path="data", acquisition_id=0)
