"""ScanR to OME-Zarr conversion task initialization."""
from pydantic import Field, validate_call
import logging

logger = logging.getLogger(__name__)

@validate_call
def convert_scanr_init_task(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Task parameters
):
    """Initialize the task to convert a LIF plate to OME-Zarr.
    
    Parameters
    ----------
    """
    ...


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(convert_scanr_init_task, logger_name=logger.name)
