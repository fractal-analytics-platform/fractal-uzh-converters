from pathlib import Path

import pytest
from ngio import open_ome_zarr_container
from ome_zarr_converters_tools import OverwriteMode

from fractal_uzh_converters.common import image_in_plate_compute_task
from fractal_uzh_converters.operetta.convert_operetta_init_task import (
    OperettaAcquisitionModel,
    convert_operetta_init_task,
)


def test_SingleWell_MultipleFOV_SingleCh(tmp_path):
    """Test the base workflow of the Operetta converter."""
    zarr_dir = tmp_path / "test_zarr_dir"
    test_data = (
        Path(__file__).parent
        / "data"
        / "extended_test_data"
        / "Operetta_reference_acquisitions"
        / "SingleWell-MultipleFOV-SingleCh"
    )

    p_list = convert_operetta_init_task(
        zarr_dir=str(zarr_dir),
        acquisitions=[
            OperettaAcquisitionModel(
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
        assert updates[0]["attributes"]["well"] == "C11"
        assert (
            updates[0]["attributes"]["plate"] == "SingleWell-MultipleFOV-SingleCh.zarr"
        )

        zarr_url = Path(updates[0]["zarr_url"])
        assert zarr_url.exists()

        ngff_image = open_ome_zarr_container(zarr_url)
        assert ngff_image.levels == 5
        image = ngff_image.get_image()
        assert image.shape == (1, 1, 4320, 4320)
        assert image.pixel_size.x == image.pixel_size.y
        assert abs(image.pixel_size.x - 0.5979760809567617) < 1e-6
        # FOV_ROI_table is only created for multi-FOV acquisitions
        assert set(ngff_image.list_tables()) == {"well_ROI_table", "FOV_ROI_table"}

    with pytest.raises(FileExistsError):
        p_list = convert_operetta_init_task(
            zarr_dir=str(zarr_dir),
            acquisitions=[
                OperettaAcquisitionModel(
                    path=str(test_data),
                    acquisition_id=0,
                ),
            ],
            overwrite=OverwriteMode.NO_OVERWRITE,
        )
