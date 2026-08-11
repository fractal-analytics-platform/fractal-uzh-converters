"""The shape of a Z-image-processing selection, shared across converters.

Instruments differ in what they can compute on board, but they all answer the same
user question: which of the outputs an acquisition holds should be converted? Each
converter subclasses the model below with only the modes its instrument produces,
so the first switch reads `Raw` everywhere and the rest is instrument-specific.

The selection is a top-level acquisition field rather than an advanced option: it
changes what comes out of the conversion, not how it is written.
"""

from ome_zarr_converters_tools import UserFacingModel
from pydantic import ConfigDict, Field


class ZProcessingSelection(UserFacingModel):
    """Which Z-image processing outputs to convert."""

    model_config = ConfigDict(title="Z Processing Selection")

    raw: bool = Field(default=True, title="Raw")
    """Convert the unprocessed images."""
