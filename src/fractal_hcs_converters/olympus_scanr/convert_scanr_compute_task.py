"""ScanR to OME-Zarr conversion task compute."""
from pydantic import Field, validate_call
import logging

logger = logging.getLogger(__name__)

@validate_call
def convert_scanr_compute_task(
    *,
    # Fractal parameters
    zarr_url: str,
    init_args: dict,
    # Task parameters
):
    """Initialize the task to convert a LIF plate to OME-Zarr.
    
    Parameters
    ----------
    """
    ...


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(convert_scanr_compute_task, logger_name=logger.name)
