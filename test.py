# Set global logger level to DEBUG for more verbose output
import logging
from pathlib import Path

from ome_zarr_converters_tools.models import (
    BackendType,
    ConverterOptions,
    OmeZarrOptions,
    StageCorrections,
    TilingMode,
    WriterMode,
)

from fractal_uzh_converters.cq3k import (
    convert_cq3k_compute_task,
    convert_cq3k_init_task,
)
from fractal_uzh_converters.cq3k.convert_cq3k_init_task import OverwriteMode
from fractal_uzh_converters.cq3k.utils import CQ3KAcquisitionModel

# Add time stamp to log messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# path = Path("./tests/data/scanr/1w_1p_1c_1z_1t")
# path = Path("/Users/locerr/data/ZMB_converters/testData_Olympus/testData_ScanR/2w_4p_4c_seq_5z_1t__001")
path = Path(
    "/Users/locerr/data/cq3k_test_data/20250718T090423_250718-CV-multiwell_test"
)

model = CQ3KAcquisitionModel(path=str(path.as_posix()))

conv_options = ConverterOptions(
    tiling_mode=TilingMode.INPLACE,
    stage_correction=StageCorrections(
        flip_x=False,
        flip_y=True,
    ),
    omezarr_options=OmeZarrOptions(table_backend=BackendType.CSV),
    writer_mode=WriterMode.BY_FOV,
)
zarr_dir = str(Path("./test_cq3k_output").resolve())
par_list = convert_cq3k_init_task(
    zarr_dir=zarr_dir,
    acquisitions=[model],
    overwrite=OverwriteMode.OVERWRITE,
    converter_options=conv_options,
)


for par in par_list[0]["parallelization_list"]:
    convert_cq3k_compute_task(
        zarr_url=par["zarr_url"],
        init_args=par["init_args"],
    )
