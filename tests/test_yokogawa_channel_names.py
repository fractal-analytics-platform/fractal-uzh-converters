"""Channel naming for the Yokogawa converters, resolved from the `.mes` file.

The `.mes` protocol file carries the human-readable `bts:Target` of each
instrument channel; the `.mlf`/`.mrf` pair does not. Without it every output was
labelled `channel_0…channel_N` (issue #27).

`.mes` parsing and channel resolution are unit-tested in
`test_yokogawa_mes_channels.py`; what is under test here is the *wiring* — that
labels survive the conversion, that the acquisition-wide list is pruned per image,
and that an `advanced.channels` override lands on the right slots.
"""

import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from fractal_uzh_converters.common import ChannelMetadataWarning
from fractal_uzh_converters.yokogawa.cellvoyager import convert_cellvoyager
from fractal_uzh_converters.yokogawa.cq3k import convert_cq3k
from fractal_uzh_converters.yokogawa.cq3k._utils import (
    CQ3KAcquisitionModel,
    parse_cq3k_metadata,
)

from .utils import DATA_DIR, EXTENDED_DATA_DIR, channel_metadata

CQ3K_RAW_DIR = DATA_DIR / "Yokogawa-CQ3K" / "raw"
CELLVOYAGER_RAW_DIR = DATA_DIR / "Yokogawa-CellVoyager" / "raw"
CELLVOYAGER_EXTENDED_RAW_DIR = EXTENDED_DATA_DIR / "Yokogawa-CellVoyager" / "raw"

# The single `<bts:Channel>` of the in-repo CQ3K fixture's MeasurementProtocol.mes.
_CQ3K_FIXTURE_TARGET = "Ch1_ConfocalFluorescence_405nm/100mW_BP447/60"


def _convert_cellvoyager(tmp_path: Path, converter_options, name: str, **advanced):
    """Convert an extended CellVoyager acquisition, return its channel metadata."""
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir(parents=True)
    image_list_updates = convert_cellvoyager(
        zarr_dir=str(zarr_dir),
        acquisitions=[
            {
                "path": str(CELLVOYAGER_EXTENDED_RAW_DIR / name),
                "acquisition_id": 0,
                "advanced": advanced,
            }
        ],
        converter_options=converter_options,
    )
    return channel_metadata(zarr_dir, image_list_updates)


######################################################################
#
# In-repo fixtures
#
######################################################################


def test_cq3k_labels_come_from_the_mes(tmp_path: Path, converter_options):
    """The `bts:Target` becomes the label and the ARGB colour loses its alpha."""
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()

    image_list_updates = convert_cq3k(
        zarr_dir=str(zarr_dir),
        acquisitions=[
            {
                "path": str(CQ3K_RAW_DIR / "hcs_2w1p1c1z1t_mip"),
                "acquisition_id": 0,
            }
        ],
        converter_options=converter_options,
    )

    channels = channel_metadata(zarr_dir, image_list_updates)
    assert channels
    for labels, wavelength_ids in channels.values():
        assert labels == [_CQ3K_FIXTURE_TARGET]
        assert wavelength_ids == ["A01_C01"]


def test_cellvoyager_without_a_mes_falls_back_to_the_wavelength_id(
    tmp_path: Path, converter_options
):
    """A `.mes` named in the `.mrf` but never shipped is a warning, not an error."""
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()

    with pytest.warns(ChannelMetadataWarning, match="does not exist"):
        image_list_updates = convert_cellvoyager(
            zarr_dir=str(zarr_dir),
            acquisitions=[
                {
                    "path": str(CELLVOYAGER_RAW_DIR / "hcs_1w1p1c1z1t"),
                    "acquisition_id": 0,
                    "image_extension": ".png",
                }
            ],
            converter_options=converter_options,
        )

    channels = channel_metadata(zarr_dir, image_list_updates)
    assert channels
    for labels, wavelength_ids in channels.values():
        assert labels == wavelength_ids == ["A01_C01"]


######################################################################
#
# Extended datasets: the cases the naming rules exist for
#
######################################################################


@pytest.mark.extended
def test_duplicate_targets_are_deduplicated(tmp_path: Path, converter_options):
    """`DuplicateTargets` names Ch2-Ch5 all `test`; the suffixes must be stable.

    Deduplication runs over the whole instrument channel list in `bts:Ch` order,
    before per-image pruning, so a well's names cannot depend on which other
    wells were acquired. Both wells acquire Ch2-Ch4 out of the five declared.
    """
    channels = _convert_cellvoyager(
        tmp_path, converter_options, "hcs_2w1p3c9z1t_DuplicateTargets"
    )

    assert channels
    for labels, _ in channels.values():
        assert labels == ["test", "test_1", "test_2"]


@pytest.mark.extended
def test_per_well_channel_subsets_are_pruned(tmp_path: Path, converter_options):
    """`TimelinesSharedChannel` acquires a different channel set per well.

    The converter indexes channels over the full instrument range and the
    library's `reindex_channels` compacts each image to what it actually
    acquired. Ch6 is a second `488` line, deduplicated to `488-2`.
    """
    channels = _convert_cellvoyager(
        tmp_path, converter_options, "hcs_3w2p2c9z1t_TimelinesSharedChannel"
    )

    labels_by_well = {path: labels for path, (labels, _) in channels.items()}
    assert labels_by_well == {
        "C/02/0": ["640"],
        "C/03/0": ["405", "488-2"],
        "C/04/0": ["488", "488-2"],
    }


@pytest.mark.extended
def test_wavelength_ids_carry_the_action_index(tmp_path: Path, converter_options):
    """`SimultaneousChannels` runs 4 channels over 3 actions, A01 acquiring Ch1+Ch4."""
    channels = _convert_cellvoyager(
        tmp_path, converter_options, "hcs_2w1p4c9z1t_SimultaneousChannels"
    )

    assert channels
    for _, wavelength_ids in channels.values():
        assert wavelength_ids == ["A01_C01", "A03_C02", "A02_C03", "A01_C04"]


######################################################################
#
# `advanced.channels` entries are keyed by `bts:Ch`
#
######################################################################


def test_override_for_an_undeclared_channel_raises(converter_options):
    """The in-repo CQ3K protocol declares Ch1 only; `ch=2` is a typo, not a gap."""
    acquisition_model = CQ3KAcquisitionModel(
        path=str(CQ3K_RAW_DIR / "hcs_2w1p1c1z1t_mip"),
        acquisition_id=0,
        advanced={"channels": [{"key": 2, "channel_label": "GFP"}]},
    )

    with pytest.raises(ValueError, match=r"Ch1\.\.Ch1"):
        parse_cq3k_metadata(
            acquisition_model=acquisition_model,
            converter_options=converter_options,
        )


def test_partial_override_is_applied_and_warns(tmp_path: Path, converter_options):
    """The in-repo CQ3K fixture acquires Ch1 only, so a colour-only entry for it
    is complete; the warning fires only when an acquired channel has no entry."""
    acquisition_model = CQ3KAcquisitionModel(
        path=str(CQ3K_RAW_DIR / "hcs_2w1p1c1z1t_mip"),
        acquisition_id=0,
        advanced={
            "channels": [{"key": 1, "color": "Red (#FF0000)", "channel_label": ""}]
        },
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", ChannelMetadataWarning)
        tiled_images = parse_cq3k_metadata(
            acquisition_model=acquisition_model,
            converter_options=converter_options,
        )

    channels = tiled_images[0].channels
    assert [c.channel_label for c in channels] == [_CQ3K_FIXTURE_TARGET]
    assert [c.color for c in channels] == ["#FF0000"]


def test_override_without_a_channel_number_is_rejected():
    """The positional form is gone: an entry must say which channel it names."""
    with pytest.raises(ValidationError, match="key"):
        CQ3KAcquisitionModel(
            path=str(CQ3K_RAW_DIR / "hcs_2w1p1c1z1t_mip"),
            advanced={"channels": [{"channel_label": "DAPI"}]},
        )


# `TimelinesPerWellChannels` acquires only Ch1 (well C03) and Ch4 (well C02) of
# the five channels its `.mrf` declares -- the case where a dense,
# acquired-channels-only list would silently mislabel both wells, and where a
# keyed list needs no entry for Ch2, Ch3 and Ch5.
_ILL_QC = "hcs_2w2p1c9z1t_TimelinesPerWellChannels"


@pytest.mark.extended
def test_override_is_keyed_by_channel_number(tmp_path: Path, converter_options):
    """Two entries, in reverse order, reach Ch4 and Ch1 through their `key`."""
    channels = _convert_cellvoyager(
        tmp_path,
        converter_options,
        _ILL_QC,
        channels=[
            {"key": 4, "channel_label": "user_ch4", "wavelength_id": "W04"},
            {"key": 1, "channel_label": "user_ch1", "wavelength_id": "W01"},
        ],
    )

    assert channels == {
        "C/02/0": (["user_ch4"], ["W04"]),  # acquires Ch4
        "C/03/0": (["user_ch1"], ["W01"]),  # acquires Ch1
    }


@pytest.mark.extended
def test_partial_override_warns_about_the_unlisted_acquired_channel(
    tmp_path: Path, converter_options
):
    """Naming Ch1 but not Ch4 is accepted, and the warning names Ch4 only."""
    with pytest.warns(ChannelMetadataWarning, match=r"acquired channel\(s\) Ch4;"):
        channels = _convert_cellvoyager(
            tmp_path,
            converter_options,
            _ILL_QC,
            channels=[{"key": 1, "channel_label": "user_ch1"}],
        )

    assert channels["C/03/0"][0] == ["user_ch1"]
    assert channels["C/02/0"][0] != ["user_ch1"]


@pytest.mark.extended
def test_override_for_a_never_acquired_channel_is_accepted(
    tmp_path: Path, converter_options
):
    """Ch2 is declared by the `.mrf` but acquired by neither well.

    The entry lands on a slot `reindex_channels` prunes, so the output is the
    same as without any override; it must not be rejected as a typo.
    """
    reference = _convert_cellvoyager(tmp_path / "reference", converter_options, _ILL_QC)
    overridden = _convert_cellvoyager(
        tmp_path / "overridden",
        converter_options,
        _ILL_QC,
        channels=[{"key": 2, "channel_label": "unused"}],
    )

    assert overridden == reference
    assert all("unused" not in labels for labels, _ in overridden.values())
