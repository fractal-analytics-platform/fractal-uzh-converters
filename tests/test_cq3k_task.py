from pathlib import Path

import pytest
from ngio import open_ome_zarr_container
from ome_zarr_converters_tools import OverwriteMode

from fractal_uzh_converters.common import image_in_plate_compute_task
from fractal_uzh_converters.cq3k.convert_cq3k_init_task import (
    CQ3KAcquisitionModel,
    convert_cq3k_init_task,
)


def test_MIP_only(tmp_path):
    """Test the base workflow of the CQ3K converter."""
    zarr_dir = tmp_path / "test_zarr_dir"
    plate_name = "20251020T100401_BVC_Channel_WellTestC5F9_MIP_Only"
    test_data = (
        Path(__file__).parent
        / "data"
        / "extended_test_data"
        / "CQ3K_reference_acquisitions"
        / plate_name
    )

    p_list = convert_cq3k_init_task(
        zarr_dir=str(zarr_dir),
        acquisitions=[
            CQ3KAcquisitionModel(
                path=str(test_data),
                acquisition_id=0,
            ),
        ],
        overwrite=OverwriteMode.NO_OVERWRITE,
    )
    assert len(p_list["parallelization_list"]) == 2
    for p in p_list["parallelization_list"]:
        results = image_in_plate_compute_task(**p)
        assert "image_list_updates" in results
        updates = results["image_list_updates"]
        assert len(updates) == 1

        assert not updates[0]["types"]["is_3D"]
        assert updates[0]["attributes"]["well"] == "C05"
        assert updates[0]["attributes"]["plate"] == f"{plate_name}_Maximum.zarr"

        zarr_url = Path(updates[0]["zarr_url"])
        assert zarr_url.exists()

        ngff_image = open_ome_zarr_container(zarr_url)
        assert ngff_image.levels == 5
        image = ngff_image.get_image()
        assert image.shape == (1, 1, 2000, 2000)
        assert image.pixel_size.x == image.pixel_size.y
        assert abs(image.pixel_size.x - 0.3242218675179569) < 1e-6
        # FOV_ROI_table is only created for multi-FOV acquisitions
        assert set(ngff_image.list_tables()) == {"well_ROI_table"}
        break  # Only need to check one of the two images

    with pytest.raises(FileExistsError):
        p_list = convert_cq3k_init_task(
            zarr_dir=str(zarr_dir),
            acquisitions=[
                CQ3KAcquisitionModel(
                    path=str(test_data),
                    acquisition_id=0,
                ),
            ],
            overwrite=OverwriteMode.NO_OVERWRITE,
        )
