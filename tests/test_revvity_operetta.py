from pathlib import Path

import pytest
from ome_zarr_converters_tools.testing import run_converter_test

from fractal_uzh_converters.operetta import convert_operetta
from fractal_uzh_converters.operetta._utils import OperettaAcquisitionModel

from .utils import DATA_DIR

SNAPSHOT_DIR = DATA_DIR / "Revvity-Operetta" / "snapshots"
RAW_DIR = DATA_DIR / "Revvity-Operetta" / "raw"


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
        (
            {
                "acquisitions": [
                    {
                        "path": str(RAW_DIR / "hcs_1w1p1c1z1t"),
                        "acquisition_id": 0,
                        "advanced": {
                            "condition_table_path": str(
                                RAW_DIR / "hcs_1w1p1c1z1t" / "condition_table.csv"
                            )
                        },
                    }
                ]
            },
            "hcs_1w1p1c1z1t_with_condition_table",
        ),
    ],
)
def test_operetta(
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


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/some/acq", "/some/acq"),
        ("/some/acq/", "/some/acq"),
        ("/some/acq/Images", "/some/acq"),
        ("/some/acq/Images/", "/some/acq"),
        # Only a whole trailing `Images` component is stripped
        ("/some/acq/MyImages", "/some/acq/MyImages"),
        ("s3://bucket/acq/Images", "s3://bucket/acq"),
        # Windows paths are handled the same on every host
        ("C:\\some\\acq\\Images", "C:/some/acq"),
        ("C:\\some\\acq\\Images\\", "C:/some/acq"),
        ("\\\\server\\share\\acq\\Images", "//server/share/acq"),
    ],
)
def test_acquisition_model_path_validator(path: str, expected: str):
    model = OperettaAcquisitionModel(path=path, acquisition_id=0)
    assert model.path == expected
