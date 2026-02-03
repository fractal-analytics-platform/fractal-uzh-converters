"""Common utilities for fractal UZH converters."""

from pathlib import Path

from ome_zarr_converters_tools import AcquisitionOptions
from pydantic import BaseModel, Field, model_validator

STANDARD_ROWS_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class BaseAcquisitionModel(BaseModel):
    """Base model for acquisitions.

    Attributes:
        path: Path to the acquisition directory.
            Should contain MeasurementData.mlf and MeasurementDetail.mrf files.
        plate_name: Optional custom name for the plate. If not provided, the name will
            be the acquisition directory name.
        acquisition_id: Acquisition ID,
            used to identify the acquisition in case of multiple acquisitions.
        advanced: Advanced acquisition options.
    """

    path: str
    plate_name: str = ""
    acquisition_id: int = Field(default=0, ge=0)
    advanced: AcquisitionOptions = Field(default_factory=AcquisitionOptions)

    @model_validator(mode="before")
    def set_default_plate_name(cls, values):
        """Set default plate name if not provided."""
        path = values.get("path")
        plate_name = values.get("plate_name")
        if plate_name == "" and path is not None:
            values["plate_name"] = Path(path).name
        return values
