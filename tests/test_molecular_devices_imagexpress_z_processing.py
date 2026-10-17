"""ImageXpress `Z Processing`: which source directory an acquisition is read from.

MD does not tag a record as a projection — it writes the projections of a Z stack
into `experiment/` and the stack itself into `experiment_z_stack/`. So, unlike
Yokogawa, the selection picks a directory rather than splitting one acquisition
into several plates, and only one of `Raw` and `MIP` can be converted per run.

Cases needing two source directories live in the extended store; the in-repo
dataset carries only `experiment/`, so it covers the validation rules.
"""

from pathlib import Path

import pytest
from ome_zarr_converters_tools import ConverterOptions

from fractal_uzh_converters.imagexpress_hcs._utils import (
    MDImageXpressHCSaiAcquisitionModel,
    parse_md_metadata,
)

from .utils import DATA_DIR, EXTENDED_DATA_DIR

RAW_DIR = DATA_DIR / "MolecularDevices-ImageXpressHCSai" / "raw"
EXTENDED_RAW_DIR = EXTENDED_DATA_DIR / "MolecularDevices-ImageXpressHCSai" / "raw"

#: In-repo, `experiment/` only.
_FIXTURE = "hcs_1w1s1t1c1z_binning4x4"
#: Extended, `experiment/` *and* `experiment_z_stack/` — the raw/MIP choice is
#: only meaningful where both exist.
_STACK_AND_MIP = "hcs_1w1p2c4z1t_4bin_Stack_MIP"
#: Extended, `experiment_z_stack/` only.
_STACK_ONLY = "hcs_1w1p2c4z1t_4bin_Stack"
#: Extended, all three directories, montage holding a Z stack.
_MONTAGE_ZSTACK = "hcs_1w6p1c4z1t_4bin_Tiled_MontageStitch_ZStack"


def _parse(
    name: str,
    converter_options: ConverterOptions,
    root: Path | None = None,
    z_processing=None,
    **advanced,
):
    """Parse an acquisition without converting it.

    `root` defaults to the extended store.
    """
    return parse_md_metadata(
        acquisition_model=MDImageXpressHCSaiAcquisitionModel(
            path=str((root or EXTENDED_RAW_DIR) / name),
            acquisition_id=0,
            z_processing=z_processing,
            advanced=advanced,
        ),
        converter_options=converter_options,
    )


def _z_extent(tiled_images) -> float:
    """How deep the converted images are — 1 plane per channel for a projection."""
    return max(
        region.roi.get("z").start + region.roi.get("z").length
        for tiled_image in tiled_images
        for region in tiled_image.regions
    )


######################################################################
#
# The selection rules, which need no particular acquisition
#
######################################################################


def test_unset_and_raw_agree(converter_options):
    """`Raw` is the default, so spelling it out cannot change the conversion."""
    unset = _parse(_FIXTURE, converter_options, root=RAW_DIR)
    raw = _parse(_FIXTURE, converter_options, root=RAW_DIR, z_processing={"raw": True})

    assert _z_extent(unset) == _z_extent(raw)
    assert len(unset) == len(raw)


def test_enabling_nothing_raises(converter_options):
    """Unticking both is a mistake worth naming, not an empty conversion."""
    with pytest.raises(ValueError, match="enables nothing"):
        _parse(_FIXTURE, converter_options, root=RAW_DIR, z_processing={"raw": False})


def test_enabling_both_raises(converter_options):
    """One acquisition, one source directory, one unsuffixed plate.

    Yokogawa can convert several kinds at once because each becomes its own
    suffixed plate; here they would collide, so the conflict is refused up front
    rather than resolved silently.
    """
    with pytest.raises(ValueError, match="single source directory"):
        _parse(
            _FIXTURE,
            converter_options,
            root=RAW_DIR,
            z_processing={"raw": True, "mip": True},
        )


######################################################################
#
# The selection picks the source directory
#
######################################################################


@pytest.mark.extended
def test_mip_reads_the_projections_next_to_the_stack(converter_options):
    """`Stack_MIP` holds both, so the two selections must not agree."""
    raw = _parse(_STACK_AND_MIP, converter_options)
    mip = _parse(
        _STACK_AND_MIP, converter_options, z_processing={"raw": False, "mip": True}
    )

    assert _z_extent(raw) == 16
    assert _z_extent(mip) == 4


@pytest.mark.extended
def test_mip_without_projections_raises(converter_options):
    """`Stack` was acquired without projections; there is nothing to fall back to."""
    with pytest.raises(FileNotFoundError, match="'experiment' folder"):
        _parse(_STACK_ONLY, converter_options, z_processing={"raw": False, "mip": True})


@pytest.mark.extended
def test_mip_with_a_z_stack_montage_raises(converter_options):
    """`Convert Montages` wins the directory choice, and then contradicts `MIP`."""
    with pytest.raises(ValueError, match="montage data is a z-stack"):
        _parse(
            _MONTAGE_ZSTACK,
            converter_options,
            z_processing={"raw": False, "mip": True},
            convert_montages=True,
        )
