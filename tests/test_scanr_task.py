from pathlib import Path

import pytest
from ngio import NgffImage

from fractal_hcs_converters.olympus_scanr.convert_scanr_compute_task import (
    convert_scanr_compute_task,
)
from fractal_hcs_converters.olympus_scanr.convert_scanr_init_task import (
    AcquisitionInputModel,
    convert_scanr_init_task,
)


def test_base_workflow(tmp_path):
    """Base workflow test.

    TODO: Extend this test to include more complex scenarios.
    TODO: Add non-happy-path tests.
    """
    zarr_dir = tmp_path / "test_zarr_dir"

    test_data = Path(__file__).parent / "data" / "scanr" / "1w_1p_1c_1z_1t"

    p_list = convert_scanr_init_task(
        zarr_urls=[],
        zarr_dir=str(zarr_dir),
        acquisitions=[
            AcquisitionInputModel(
                path=str(test_data),
                acquisition_id=0,
            ),
        ],
        overwrite=True,
    )

    assert len(p_list["parallelization_list"]) == 1

    for p in p_list["parallelization_list"]:
        results = convert_scanr_compute_task(**p)
        assert "image_list_updates" in results
        updates = results["image_list_updates"]
        assert len(updates) == 1

        assert not updates[0]["types"]["is_3D"]
        assert updates[0]["attributes"]["well"] == "B2"
        assert updates[0]["attributes"]["plate"] == "1w_1p_1c_1z_1t.zarr"

        zarr_url = Path(updates[0]["zarr_url"])
        assert zarr_url.exists()

        ngff_image = NgffImage(zarr_url)
        assert ngff_image.num_levels == 5
        image = ngff_image.get_image()
        assert image.shape == (1, 1, 2048, 2048)
        assert image.pixel_size.x == image.pixel_size.y
        assert abs(image.pixel_size.x - 0.325) < 1e-6
        assert set(ngff_image.tables.list()) == {"well_ROI_table", "FOV_ROI_table"}

    with pytest.raises(FileExistsError):
        p_list = convert_scanr_init_task(
            zarr_urls=[],
            zarr_dir=str(zarr_dir),
            acquisitions=[
                AcquisitionInputModel(
                    path=str(test_data),
                    acquisition_id=0,
                ),
            ],
            overwrite=False,
        )
