"""Splitting a Yokogawa acquisition into one plate per Z-image processing kind.

A `.mlf` can interleave raw Z slices with projections computed on the
instrument, tagged by `bts:ZImageProcessing`. They share the well, field of view
and channel numbering, so converting them into one plate would overlap their
tiles; each kind is written as its own plate instead, suffixed by the
abbreviation of its algorithm.
"""

import logging

from ome_zarr_converters_tools import UserFacingModel
from pydantic import Field

logger = logging.getLogger(__name__)

#: `bts:ZImageProcessing` values, mapped to the conventional imaging abbreviations.
#: Anything else is used verbatim, with a warning.
_Z_PROCESSING_SUFFIXES = {"Maximum": "MIP", "Minimum": "MinIP", "Sum": "SIP"}

#: The token of the raw-slice plate, which carries no name suffix.
_Z_SLICES = "Z slices"


class ZProcessingSelection(UserFacingModel):
    """Which Z-image processing outputs to convert."""

    z_slices: bool = Field(default=True, title="Z slices")
    """Convert the raw Z stack."""
    mip: bool = Field(default=False, title="MIP")
    """Convert the maximum intensity projection."""
    min_ip: bool = Field(default=False, title="MinIP")
    """Convert the minimum intensity projection."""
    sip: bool = Field(default=False, title="SIP")
    """Convert the sum intensity projection."""


#: `ZProcessingSelection` field -> the token it selects. Also the set of tokens a
#: selection is able to name at all.
_Z_PROCESSING_TOKENS = {
    "z_slices": _Z_SLICES,
    "mip": "MIP",
    "min_ip": "MinIP",
    "sip": "SIP",
}


def _z_processing_token(z_type: str | None) -> str:
    """The selection token for one `bts:ZImageProcessing` value.

    Doubles as the plate name suffix. Resolve it once per distinct `z_type`: an
    unrecognised value warns here, and `_build_tiles` runs per field of view.
    """
    if z_type is None:
        return _Z_SLICES
    token = _Z_PROCESSING_SUFFIXES.get(z_type)
    if token is None:
        logger.warning(
            f"Unknown z image processing type '{z_type}'. Using it verbatim as the "
            "plate name suffix."
        )
        token = z_type
    return token


def _select_z_processing(
    *,
    plates_tokens: dict[str | None, str],
    selection: ZProcessingSelection | None,
    acquisition_dir: str,
) -> set[str | None]:
    """The `z_image_processing` groups to convert, given the user's selection.

    No selection keeps everything. Otherwise a selected kind the acquisition does not
    contain is a warning rather than an error, so that one selection can be applied to
    a batch of acquisitions; but a selection matching *nothing* raises, since the
    alternative is an empty output whose cause only surfaces much later, from the
    library, without naming the option responsible.

    Args:
        plates_tokens: `{z_image_processing: token}` for this acquisition.
        selection: The user's `advanced.z_processing`, or `None` for all of them.
        acquisition_dir: Named in the errors, since a batch fails one acquisition at
            a time.

    Returns:
        The `z_image_processing` values to keep.

    Raises:
        ValueError: If nothing is enabled, or if the selection matches none of the
            parsed plates. The two have different fixes, so they are reported apart.
    """
    if selection is None:
        return set(plates_tokens)

    selected = {
        token
        for field, token in _Z_PROCESSING_TOKENS.items()
        if getattr(selection, field)
    }
    if not selected:
        raise ValueError(
            "`advanced.z_processing` enables nothing, so there is nothing to convert "
            f"in {acquisition_dir}. Enable at least one output, or leave the option "
            "unset to convert everything the acquisition contains."
        )

    available = set(plates_tokens.values())
    kept = {z_type for z_type, token in plates_tokens.items() if token in selected}
    if not kept:
        raise ValueError(
            f"`advanced.z_processing` enables {sorted(selected)} but "
            f"{acquisition_dir} contains only {sorted(available)}. Clear the option "
            "to convert everything the acquisition contains."
        )

    missing = sorted(selected - available)
    if missing:
        logger.warning(
            f"`advanced.z_processing` enables {missing}, which {acquisition_dir} "
            "does not contain. Converting the rest of the selection."
        )

    # An unrecognised `bts:ZImageProcessing` value keeps its raw string as the token,
    # so no field of `ZProcessingSelection` can name it.
    unnameable = sorted(
        token
        for z_type, token in plates_tokens.items()
        if z_type not in kept and token not in _Z_PROCESSING_TOKENS.values()
    )
    if unnameable:
        logger.warning(
            f"Skipping the unrecognised z image processing type(s) {unnameable}, "
            "which `advanced.z_processing` cannot name. Use a `Path Regex Filter` on "
            "the plate name to select them."
        )
    return kept


def _plate_name(base_name: str, token: str) -> str:
    """Plate name for one `z_image_processing` group.

    Each projection algorithm is written as its own plate; the raw-slice plate
    carries no suffix and can coexist with them.
    """
    if token == _Z_SLICES:
        return base_name
    return f"{base_name}_{token}"
