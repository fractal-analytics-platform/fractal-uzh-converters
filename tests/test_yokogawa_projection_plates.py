"""Yokogawa projection plates: one plate per `bts:ZImageProcessing` algorithm.

The suffix used to be the raw Yokogawa string (`_Maximum`, `_Minimum`, `_Sum`);
it is now the conventional imaging abbreviation (`_MIP`, `_MinIP`, `_SIP`), with
anything unrecognised passed through verbatim (issue #45).

Renaming the plates rewrites every CQ3K snapshot, so the cases that matter are
asserted here explicitly. Where only the plate *names* are under test the
assertions run off `parse_cq3k_metadata`, which reads no image data.

The split applies to both instruments, but the only CellVoyager acquisition with a
`bts:ZImageProcessing` is in the gitignored extended store — so the CellVoyager
cases rewrite the in-repo fixture's `.mlf` in `tmp_path`, to give CI coverage.
"""

import shutil
from pathlib import Path

import pytest

from fractal_uzh_converters.common import ConverterWarning
from fractal_uzh_converters.yokogawa._z_processing import (
    _plate_name,
    _z_processing_token,
)
from fractal_uzh_converters.yokogawa.cellvoyager._utils import (
    CellVoyagerAcquisitionModel,
    parse_cellvoyager_metadata,
)
from fractal_uzh_converters.yokogawa.cq3k import convert_cq3k
from fractal_uzh_converters.yokogawa.cq3k._utils import (
    CQ3KAcquisitionModel,
    parse_cq3k_metadata,
)

from .utils import DATA_DIR, EXTENDED_DATA_DIR, plate_channel_metadata

CQ3K_RAW_DIR = DATA_DIR / "Yokogawa-CQ3K" / "raw"
CQ3K_EXTENDED_RAW_DIR = EXTENDED_DATA_DIR / "Yokogawa-CQ3K" / "raw"
CELLVOYAGER_FIXTURE = DATA_DIR / "Yokogawa-CellVoyager" / "raw" / "hcs_1w1p1c1z1t"


def _parse(name: str, converter_options, root=None, z_processing=None, **advanced):
    """Parse an acquisition without converting it.

    `root` defaults to the extended store. `z_processing` is a top-level field;
    the remaining kwargs are advanced options.
    """
    return parse_cq3k_metadata(
        acquisition_model=CQ3KAcquisitionModel(
            path=str((root or CQ3K_EXTENDED_RAW_DIR) / name),
            acquisition_id=0,
            z_processing=z_processing,
            advanced=advanced,
        ),
        converter_options=converter_options,
    )


def _plate_names(name: str, converter_options, **kwargs) -> set[str]:
    """Plate paths an extended acquisition would write, without converting it."""
    tiled_images = _parse(name, converter_options, **kwargs)
    return {tiled_image.collection.plate_path() for tiled_image in tiled_images}


def _acquired_channels(name: str, converter_options) -> dict[str, set[int]]:
    """`{plate_path: {0-based channel index}}` over an acquisition's tile regions."""
    tiled_images = _parse(name, converter_options)
    per_plate: dict[str, set[int]] = {}
    for tiled_image in tiled_images:
        plate = tiled_image.collection.plate_path()
        for region in tiled_image.regions:
            per_plate.setdefault(plate, set()).add(region.roi.get("c").start)
    return per_plate


######################################################################
#
# The suffix mapping itself
#
######################################################################


@pytest.mark.parametrize(
    ("z_type", "token", "expected"),
    [
        (None, "Raw", "plate"),
        ("Maximum", "MIP", "plate_MIP"),
        ("Minimum", "MinIP", "plate_MinIP"),
        ("Sum", "SIP", "plate_SIP"),
    ],
)
def test_known_algorithms_map_to_their_abbreviation(z_type, token, expected):
    """The token is both the `z_processing` value and the plate name suffix."""
    assert _z_processing_token(z_type) == token
    assert _plate_name("plate", token) == expected


def test_an_unknown_algorithm_is_used_verbatim_with_a_warning():
    """A firmware update adding an algorithm must not break the conversion.

    No acquisition in either test store carries a value outside the three known
    ones, so this is the only coverage the fallback gets.
    """
    with pytest.warns(ConverterWarning, match="Median"):
        assert _z_processing_token("Median") == "Median"
    assert _plate_name("plate", "Median") == "plate_Median"


######################################################################
#
# In-repo fixture
#
######################################################################


def test_cq3k_fixture_writes_a_mip_plate(tmp_path: Path, converter_options):
    """The fixture is projection-only: one plate, suffixed."""
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

    plates = plate_channel_metadata(zarr_dir, image_list_updates)
    assert set(plates) == {"hcs_2w1p1c1z1t_mip_MIP.zarr"}


######################################################################
#
# Extended datasets: one acquisition, several plates
#
######################################################################

_SLICES_MIP_MINIP = "hcs_2w4p2c10z1t_Slices_MIP_MinIP"
_CHANNELS_MIP_MINIP = "hcs_3w2p4c1z1t_Channels_MIP_MinIP"


@pytest.mark.extended
def test_slices_and_both_projection_kinds_split_into_three_plates(converter_options):
    """It acquires Ch1+Ch2 as slices, Ch3 min-projected and Ch4 max-projected.

    Each algorithm carries its own `bts:Ch` numbers — the root cause of #45 —
    so the per-plate channel sets are disjoint.
    """
    assert _acquired_channels(_SLICES_MIP_MINIP, converter_options) == {
        f"{_SLICES_MIP_MINIP}.zarr": {0, 1},
        f"{_SLICES_MIP_MINIP}_MinIP.zarr": {2},
        f"{_SLICES_MIP_MINIP}_MIP.zarr": {3},
    }


@pytest.mark.extended
def test_min_projected_brightfield_lands_in_its_own_plate(
    tmp_path: Path, converter_options
):
    """The canonical #45 case: one acquisition, two plates.

    Four max-projected fluorescence channels go to `_MIP`, the min-projected
    brightfield to `_MinIP`. This one is converted rather than only parsed,
    because the per-image channel pruning happens at compute time.
    """
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()

    image_list_updates = convert_cq3k(
        zarr_dir=str(zarr_dir),
        acquisitions=[
            {
                "path": str(CQ3K_EXTENDED_RAW_DIR / _CHANNELS_MIP_MINIP),
                "acquisition_id": 0,
            }
        ],
        converter_options=converter_options,
    )

    plates = plate_channel_metadata(zarr_dir, image_list_updates)
    assert set(plates) == {
        f"{_CHANNELS_MIP_MINIP}_MIP.zarr",
        f"{_CHANNELS_MIP_MINIP}_MinIP.zarr",
    }
    assert plates[f"{_CHANNELS_MIP_MINIP}_MIP.zarr"]
    for labels, _ in plates[f"{_CHANNELS_MIP_MINIP}_MIP.zarr"].values():
        assert len(labels) == 4
    assert plates[f"{_CHANNELS_MIP_MINIP}_MinIP.zarr"]
    for labels, _ in plates[f"{_CHANNELS_MIP_MINIP}_MinIP.zarr"].values():
        assert len(labels) == 1

    # The algorithm belongs in the plate name only: labels come from the `.mes`
    # and must stay comparable across the plates of an acquisition.
    for plate in plates.values():
        for labels, _ in plate.values():
            assert not any(
                suffix in label
                for label in labels
                for suffix in ("MIP", "MinIP", "SIP", "Maximum", "Minimum")
            )


@pytest.mark.extended
def test_sum_projections_are_suffixed_sip(converter_options):
    """`Sum` is the third algorithm, alongside a raw-slice plate."""
    name = "hcs_2w1p1c33z1t_MIP_SUM_Slice"

    assert _plate_names(name, converter_options) == {
        f"{name}.zarr",
        f"{name}_MIP.zarr",
        f"{name}_SIP.zarr",
    }


######################################################################
#
# `z_processing` selects which of those plates are written
#
######################################################################


@pytest.mark.extended
@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        # Unset is the default and converts everything.
        (None, {"", "_MIP", "_MinIP"}),
        # `raw` is the only field defaulting to True, so `{}` is slices-only
        # and `{"mip": True}` adds the MIP rather than replacing the slices.
        ({}, {""}),
        ({"mip": True}, {"", "_MIP"}),
        ({"raw": False, "mip": True}, {"_MIP"}),
        ({"raw": False, "mip": True, "min_ip": True}, {"_MIP", "_MinIP"}),
    ],
)
def test_selection_picks_the_plate_subset(selection, expected, converter_options):
    """`Slices_MIP_MinIP` is the only acquisition holding slices *and* two kinds."""
    assert _plate_names(
        _SLICES_MIP_MINIP, converter_options, z_processing=selection
    ) == {f"{_SLICES_MIP_MINIP}{suffix}.zarr" for suffix in expected}


@pytest.mark.extended
def test_a_selected_kind_the_acquisition_lacks_only_warns(converter_options):
    """One selection has to stay usable across a batch of mixed acquisitions."""
    with pytest.warns(ConverterWarning, match="SIP"):
        plates = _plate_names(
            _SLICES_MIP_MINIP,
            converter_options,
            z_processing={"raw": False, "mip": True, "sip": True},
        )

    assert plates == {f"{_SLICES_MIP_MINIP}_MIP.zarr"}


def test_enabling_nothing_raises(converter_options):
    """Unticking everything is a mistake worth naming, not an empty conversion.

    Distinct from the no-match error below: the fix is to enable something, not to
    pick a kind the acquisition actually has.
    """
    with pytest.raises(ValueError, match="enables nothing"):
        _parse(
            "hcs_2w1p1c1z1t_mip",
            converter_options,
            root=CQ3K_RAW_DIR,
            z_processing={"raw": False},
        )


def test_a_selection_matching_nothing_raises(converter_options):
    """The in-repo fixture is projection-only, so `Raw` matches none of it.

    Without this the failure would surface later, from the library, as "No tiles
    provided to build TiledImage" — which never names the option responsible.
    """
    with pytest.raises(ValueError, match="z_processing"):
        _parse(
            "hcs_2w1p1c1z1t_mip",
            converter_options,
            root=CQ3K_RAW_DIR,
            z_processing={},
        )


def test_the_error_names_what_the_acquisition_contains(converter_options):
    """A user cannot fix the option without knowing what is actually there."""
    with pytest.raises(ValueError, match="MIP"):
        _parse(
            "hcs_2w1p1c1z1t_mip",
            converter_options,
            root=CQ3K_RAW_DIR,
            z_processing={"raw": False, "min_ip": True},
        )


@pytest.mark.extended
def test_selection_relaxes_the_required_channel_override_length(converter_options):
    """Channel resolution follows the selection, not the whole acquisition.

    `Channel_Well_test` min-projects Ch1 and max-projects Ch2-Ch5, so a full
    acquisition needs a 5-entry `advanced.channels`. Selecting only the MinIP
    plate leaves Ch1 as the highest acquired channel, and one entry is enough.
    """
    one_entry = [{"channel_label": "brightfield"}]

    tiled_images = _parse(
        _CHANNELS_MIP_MINIP,
        converter_options,
        z_processing={"raw": False, "min_ip": True},
        channels=one_entry,
    )
    assert tiled_images
    for tiled_image in tiled_images:
        assert tiled_image.channels[0].channel_label == "brightfield"

    with pytest.raises(ValueError, match="at least 5 entries"):
        _parse(_CHANNELS_MIP_MINIP, converter_options, channels=one_entry)


@pytest.mark.extended
def test_a_selection_converts_only_the_selected_plate(
    tmp_path: Path, converter_options
):
    """End to end: the unselected plates never reach the store."""
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()

    image_list_updates = convert_cq3k(
        zarr_dir=str(zarr_dir),
        acquisitions=[
            {
                "path": str(CQ3K_EXTENDED_RAW_DIR / _CHANNELS_MIP_MINIP),
                "acquisition_id": 0,
                "z_processing": {"raw": False, "min_ip": True},
            }
        ],
        converter_options=converter_options,
    )

    plates = plate_channel_metadata(zarr_dir, image_list_updates)
    assert set(plates) == {f"{_CHANNELS_MIP_MINIP}_MinIP.zarr"}
    assert not (zarr_dir / f"{_CHANNELS_MIP_MINIP}_MIP.zarr").exists()


######################################################################
#
# CellVoyager: the same split, on a fixture rewritten in `tmp_path`
#
######################################################################


def _cellvoyager_plate_names(
    acquisition_dir: Path, converter_options, z_processing=None, **advanced
):
    """Plate paths a CellVoyager acquisition would write, without converting it."""
    tiled_images = parse_cellvoyager_metadata(
        acquisition_model=CellVoyagerAcquisitionModel(
            path=str(acquisition_dir),
            acquisition_id=0,
            # The in-repo CellVoyager fixture ships `.png`, not `.tif`.
            image_extension=".png",
            z_processing=z_processing,
            advanced=advanced,
        ),
        converter_options=converter_options,
    )
    return {tiled_image.collection.plate_path() for tiled_image in tiled_images}


def _projected_fixture(tmp_path: Path, algorithm: str = "Maximum") -> Path:
    """The in-repo CellVoyager fixture, its one record tagged as a projection.

    No CellVoyager acquisition in the in-repo store carries a
    `bts:ZImageProcessing`, so the attribute is injected here rather than shipped
    as a second fixture.
    """
    acquisition_dir = tmp_path / CELLVOYAGER_FIXTURE.name
    shutil.copytree(CELLVOYAGER_FIXTURE, acquisition_dir)
    mlf = acquisition_dir / "MeasurementData.mlf"
    text = mlf.read_text(encoding="utf-8")
    marker = '<bts:MeasurementRecord bts:Type="IMG"'
    assert text.count(marker) == 1, "fixture changed; this helper tags one record"
    mlf.write_text(
        text.replace(marker, f'{marker} bts:ZImageProcessing="{algorithm}"'),
        encoding="utf-8",
    )
    return acquisition_dir


def test_cellvoyager_without_the_attribute_writes_the_unsuffixed_plate(
    converter_options,
):
    """The baseline the case below is a departure from, and the common case."""
    assert _cellvoyager_plate_names(CELLVOYAGER_FIXTURE, converter_options) == {
        f"{CELLVOYAGER_FIXTURE.name}.zarr"
    }


def test_cellvoyager_projections_land_in_their_own_plate(
    tmp_path: Path, converter_options
):
    """CellVoyager used to drop `bts:ZImageProcessing` and write one plate.

    Projection and raw records share the well, field of view and channel
    numbering, so a single plate meant overlapping tiles.
    """
    acquisition_dir = _projected_fixture(tmp_path)

    assert _cellvoyager_plate_names(acquisition_dir, converter_options) == {
        f"{CELLVOYAGER_FIXTURE.name}_MIP.zarr"
    }


def test_cellvoyager_selection_picks_the_plate_subset(
    tmp_path: Path, converter_options
):
    """`z_processing` reaches the CellVoyager parser, as it does CQ3K's."""
    acquisition_dir = _projected_fixture(tmp_path)

    assert _cellvoyager_plate_names(
        acquisition_dir,
        converter_options,
        z_processing={"raw": False, "mip": True},
    ) == {f"{CELLVOYAGER_FIXTURE.name}_MIP.zarr"}


def test_cellvoyager_a_selection_matching_nothing_raises(
    tmp_path: Path, converter_options
):
    """The rewritten fixture is projection-only, so `Raw` matches none of it."""
    acquisition_dir = _projected_fixture(tmp_path)

    with pytest.raises(ValueError, match="z_processing"):
        _cellvoyager_plate_names(acquisition_dir, converter_options, z_processing={})
