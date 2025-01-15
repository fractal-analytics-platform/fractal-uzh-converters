from pathlib import Path

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
    list_acq=[
        AcquisitionInputModel(
            path= str(test_data),
            acquisition_id=0,
        ),
    ],
    overwrite=True)
    for p in p_list["parallelization_list"]:
        convert_scanr_compute_task(**p)
