from pathlib import Path

import pytest
from ngio import open_ome_zarr_container
from ome_zarr_converters_tools import OverwriteMode

from fractal_uzh_converters.common import image_in_plate_compute_task
from fractal_uzh_converters.olympus_scanr.convert_scanr_init_task import (
    ScanRAcquisitionModel,
    convert_scanr_init_task,
)


def test_1w_1p_1c_1z_1t(tmp_path):
    """Test the base workflow of the ScanR converter."""
    zarr_dir = tmp_path / "test_zarr_dir"
    test_data = Path(__file__).parent / "data" / "scanr" / "1w_1p_1c_1z_1t"

    p_list = convert_scanr_init_task(
        zarr_dir=str(zarr_dir),
        acquisitions=[
            ScanRAcquisitionModel(
                path=str(test_data),
                acquisition_id=0,
            ),
        ],
        overwrite=OverwriteMode.NO_OVERWRITE,
    )
    assert len(p_list["parallelization_list"]) == 1
    for p in p_list["parallelization_list"]:
        results = image_in_plate_compute_task(**p)
        assert "image_list_updates" in results
        updates = results["image_list_updates"]
        assert len(updates) == 1

        assert not updates[0]["types"]["is_3D"]
        assert updates[0]["attributes"]["well"] == "B02"
        assert updates[0]["attributes"]["plate"] == "1w_1p_1c_1z_1t.zarr"

        zarr_url = Path(updates[0]["zarr_url"])
        assert zarr_url.exists()

        ngff_image = open_ome_zarr_container(zarr_url)
        assert ngff_image.levels == 5
        image = ngff_image.get_image()
        assert image.shape == (1, 1, 2048, 2048)
        assert image.pixel_size.x == image.pixel_size.y
        assert abs(image.pixel_size.x - 0.325) < 1e-6
        # FOV_ROI_table is only created for multi-FOV acquisitions
        assert set(ngff_image.list_tables()) == {"well_ROI_table"}

    with pytest.raises(FileExistsError):
        p_list = convert_scanr_init_task(
            zarr_dir=str(zarr_dir),
            acquisitions=[
                ScanRAcquisitionModel(
                    path=str(test_data),
                    acquisition_id=0,
                ),
            ],
            overwrite=OverwriteMode.NO_OVERWRITE,
        )
