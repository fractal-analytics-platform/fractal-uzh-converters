from pathlib import Path

import pytest
from ome_zarr_converters_tools.testing import run_converter_test

from fractal_uzh_converters.imagexpress_hcs import convert_imagexpress_hcs

from .utils import DATA_DIR

SNAPSHOT_DIR = DATA_DIR / "MolecularDevices-ImageXpressHCSai" / "snapshots"
RAW_DIR = DATA_DIR / "MolecularDevices-ImageXpressHCSai" / "raw"


@pytest.mark.parametrize(
    "api_kwargs, snapshot_name",
    [
        (
            {
                "acquisitions": [
                    {
                        "path": str(RAW_DIR / "hcs_1w1s1t1c1z_binning4x4"),
                        "acquisition_id": 0,
                    }
                ]
            },
            "hcs_1w1s1t1c1z_binning4x4",
        ),
    ],
)
def test_imagexpress_hcs(
    tmp_path: Path,
    api_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
    converter_options,
):
    run_converter_test(
        tmp_path=tmp_path,
        api_fn=convert_imagexpress_hcs,
        api_kwargs=api_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.json",
        update_snapshots=update_snapshots,
        converter_options=converter_options,
    )
