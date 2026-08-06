"""Channel naming for the Yokogawa converters, resolved from the `.mes` file.

The `.mes` protocol file carries the human-readable `bts:Target` of each
instrument channel; the `.mlf`/`.mrf` pair does not. Without it every output was
labelled `channel_0…channel_N` (issue #27).

Wiring this in rewrites every Yokogawa snapshot, so the cases that matter are
asserted here explicitly rather than left to the snapshots. `.mes` parsing and
channel resolution themselves are unit-tested in `test_yokogawa_mes_channels.py`;
what is under test here is the *wiring*: that the labels survive the conversion,
that the acquisition-wide channel list is pruned per image, and that an
`advanced.channels` override lands on the right slots.
"""

from pathlib import Path

import pytest

from fractal_uzh_converters.cellvoyager import convert_cellvoyager
from fractal_uzh_converters.cellvoyager._utils import (
    CellVoyagerAcquisitionModel,
    parse_cellvoyager_metadata,
)
from fractal_uzh_converters.cq3k import convert_cq3k

from .utils import DATA_DIR, EXTENDED_DATA_DIR, channel_metadata

CQ3K_RAW_DIR = DATA_DIR / "Yokogawa-CQ3K" / "raw"
CELLVOYAGER_RAW_DIR = DATA_DIR / "Yokogawa-CellVoyager" / "raw"
CELLVOYAGER_EXTENDED_RAW_DIR = EXTENDED_DATA_DIR / "Yokogawa-CellVoyager" / "raw"

# The single `<bts:Channel>` of the in-repo CQ3K fixture's MeasurementProtocol.mes.
_CQ3K_FIXTURE_TARGET = "Ch1_ConfocalFluorescence_405nm/100mW_BP447/60"


def _convert_cellvoyager(tmp_path: Path, converter_options, name: str, **advanced):
    """Convert an extended CellVoyager acquisition, return its channel metadata."""
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()
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
    tmp_path: Path, converter_options, caplog
):
    """A `.mes` named in the `.mrf` but never shipped is a warning, not an error."""
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()

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
    assert "does not exist" in caplog.text


######################################################################
#
# Extended datasets: the cases the naming rules exist for
#
######################################################################


@pytest.mark.extended
def test_duplicate_targets_are_deduplicated(tmp_path: Path, converter_options):
    """`same-target` names Ch2-Ch5 all `test`; the suffixes must be stable.

    Deduplication runs over the whole instrument channel list in `bts:Ch` order,
    before per-image pruning, so a well's names cannot depend on which other
    wells were acquired. Both wells acquire Ch2-Ch4 out of the five declared.
    """
    channels = _convert_cellvoyager(tmp_path, converter_options, "same-target")

    assert channels
    for labels, _ in channels.values():
        assert labels == ["test", "test_1", "test_2"]


@pytest.mark.extended
def test_per_well_channel_subsets_are_pruned(tmp_path: Path, converter_options):
    """`time-lines-test` acquires a different channel set per well.

    The converter indexes channels over the full instrument range and the
    library's `reindex_channels` compacts each image to what it actually
    acquired. Ch6 is a second `488` line, deduplicated to `488-2`.
    """
    channels = _convert_cellvoyager(tmp_path, converter_options, "time-lines-test")

    labels_by_well = {path: labels for path, (labels, _) in channels.items()}
    assert labels_by_well == {
        "C/02/0": ["640"],
        "C/03/0": ["405", "488-2"],
        "C/04/0": ["488", "488-2"],
    }


@pytest.mark.extended
def test_wavelength_ids_carry_the_action_index(tmp_path: Path, converter_options):
    """`2ch-sim` runs 4 channels over 3 actions, A01 acquiring Ch1 and Ch4."""
    channels = _convert_cellvoyager(tmp_path, converter_options, "2ch-sim")

    assert channels
    for _, wavelength_ids in channels.values():
        assert wavelength_ids == ["A01_C01", "A03_C02", "A02_C03", "A01_C04"]


######################################################################
#
# `advanced.channels` overrides map positionally: element 0 is Ch1
#
######################################################################

# `time-lines-ill-qc` acquires only Ch1 (well C03) and Ch4 (well C02) of the five
# channels its `.mrf` declares -- the case where a dense, acquired-channels-only
# override would silently mislabel both wells.
_ILL_QC = "time-lines-ill-qc"


@pytest.mark.extended
def test_override_shorter_than_the_acquired_range_raises(converter_options):
    """The converter's own error names the required length.

    The generic `Tile` validator would instead report the offending channel
    index against the override's length, which reads as an off-by-one.
    """
    acquisition_model = CellVoyagerAcquisitionModel(
        path=str(CELLVOYAGER_EXTENDED_RAW_DIR / _ILL_QC),
        acquisition_id=0,
        advanced={
            "channels": [
                {"channel_label": "first"},
                {"channel_label": "second"},
            ]
        },
    )

    with pytest.raises(ValueError, match="at least 4 entries"):
        parse_cellvoyager_metadata(
            acquisition_model=acquisition_model,
            converter_options=converter_options,
        )


@pytest.mark.extended
def test_override_is_indexed_by_channel_not_by_position(
    tmp_path: Path, converter_options
):
    """A 4-entry override reaches Ch4 through element 3, not element 1."""
    channels = _convert_cellvoyager(
        tmp_path,
        converter_options,
        _ILL_QC,
        channels=[
            {"channel_label": f"user_ch{i + 1}", "wavelength_id": f"W{i + 1:02d}"}
            for i in range(4)
        ],
    )

    assert channels == {
        "C/02/0": (["user_ch4"], ["W04"]),  # acquires Ch4 -> element 3
        "C/03/0": (["user_ch1"], ["W01"]),  # acquires Ch1 -> element 0
    }
