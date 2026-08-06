"""Regression tests for the 0-based Yokogawa channel index.

`start_c` used to be the raw 1-based `bts:Ch` number while `start_z`/`start_t`
were converted to 0-based. The off-by-one was masked by the library default
`reindex_channels=True`, but any user filling `advanced.channels` hit a
``Tile 'FOV_1' references channel index N`` validation error.

The second half of these tests covers the related fix: `AcquisitionDetails` is
now built once per plate instead of once per field of view, because
`TiledImage.add_tile` rejects tiles whose `AcquisitionDetails` disagree and
several fields of view merge into a single output image.
"""

from pathlib import Path

import pytest

from fractal_uzh_converters.cellvoyager import _utils as cellvoyager_utils
from fractal_uzh_converters.cellvoyager import convert_cellvoyager
from fractal_uzh_converters.cellvoyager._utils import (
    CellVoyagerAcquisitionModel,
    parse_cellvoyager_metadata,
)
from fractal_uzh_converters.cq3k import _utils as cq3k_utils
from fractal_uzh_converters.cq3k import convert_cq3k
from fractal_uzh_converters.cq3k._utils import (
    CQ3KAcquisitionModel,
    parse_cq3k_metadata,
)

from .utils import DATA_DIR, EXTENDED_DATA_DIR, channel_metadata

CQ3K_RAW_DIR = DATA_DIR / "Yokogawa-CQ3K" / "raw"
CELLVOYAGER_RAW_DIR = DATA_DIR / "Yokogawa-CellVoyager" / "raw"
CQ3K_EXTENDED_RAW_DIR = EXTENDED_DATA_DIR / "Yokogawa-CQ3K" / "raw"
CELLVOYAGER_EXTENDED_RAW_DIR = EXTENDED_DATA_DIR / "Yokogawa-CellVoyager" / "raw"

# Both in-repo fixtures acquire a single channel (Ch1), so a one-entry override
# is full length and maps to `start_c == 0`.
_SINGLE_CHANNEL_OVERRIDE = [{"channel_label": "DAPI", "wavelength_id": "A01_C01"}]


def _count_acquisition_details_calls(monkeypatch, module) -> list:
    """Record every `build_acquisition_details` call made by `module`."""
    calls: list = []
    original = module.build_acquisition_details

    def _spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(module, "build_acquisition_details", _spy)
    return calls


def test_cq3k_channel_override(tmp_path: Path, converter_options):
    """A full-length `advanced.channels` list is applied instead of raising."""
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()

    image_list_updates = convert_cq3k(
        zarr_dir=str(zarr_dir),
        acquisitions=[
            {
                "path": str(CQ3K_RAW_DIR / "hcs_2w1p1c1z1t_mip"),
                "acquisition_id": 0,
                "advanced": {"channels": _SINGLE_CHANNEL_OVERRIDE},
            }
        ],
        converter_options=converter_options,
    )

    channels = channel_metadata(zarr_dir, image_list_updates)
    assert channels
    for labels, wavelength_ids in channels.values():
        assert labels == ["DAPI"]
        assert wavelength_ids == ["A01_C01"]


def test_cellvoyager_channel_override(tmp_path: Path, converter_options):
    """A full-length `advanced.channels` list is applied instead of raising."""
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()

    image_list_updates = convert_cellvoyager(
        zarr_dir=str(zarr_dir),
        acquisitions=[
            {
                "path": str(CELLVOYAGER_RAW_DIR / "hcs_1w1p1c1z1t"),
                "acquisition_id": 0,
                "image_extension": ".png",
                "advanced": {"channels": _SINGLE_CHANNEL_OVERRIDE},
            }
        ],
        converter_options=converter_options,
    )

    channels = channel_metadata(zarr_dir, image_list_updates)
    assert channels
    for labels, wavelength_ids in channels.values():
        assert labels == ["DAPI"]
        assert wavelength_ids == ["A01_C01"]


def test_cq3k_builds_one_acquisition_details_per_plate(monkeypatch, converter_options):
    """AcquisitionDetails is built per plate, not per (well, field of view)."""
    calls = _count_acquisition_details_calls(monkeypatch, cq3k_utils)

    parse_cq3k_metadata(
        acquisition_model=CQ3KAcquisitionModel(
            path=str(CQ3K_RAW_DIR / "hcs_2w1p1c1z1t_mip"),
            acquisition_id=0,
        ),
        converter_options=converter_options,
    )

    # Two (well, field of view) groups, but a single `Maximum` plate.
    assert len(calls) == 1


# Neither in-repo fixture has more than one field of view per well, so the
# `add_tile` merge path needs the extended datasets. Both `.mrf` files declare
# five channels; the acquired subsets are Ch1 / Ch2-5 (CQ3K) and Ch2-4
# (CellVoyager), all of which must index into a five-entry override.
_FIVE_CHANNEL_OVERRIDE = [
    {"channel_label": f"channel_{i}", "wavelength_id": f"A01_C{i + 1:02d}"}
    for i in range(5)
]


@pytest.mark.extended
def test_cq3k_acquisition_details_are_plate_scoped(monkeypatch, converter_options):
    """Multi-field wells merge without an AcquisitionDetails mismatch."""
    calls = _count_acquisition_details_calls(monkeypatch, cq3k_utils)
    acquisition_model = CQ3KAcquisitionModel(
        path=str(CQ3K_EXTENDED_RAW_DIR / "20251201T133446_Channel_Well_test"),
        acquisition_id=0,
        advanced={"channels": _FIVE_CHANNEL_OVERRIDE},
    )
    tiled_images = parse_cq3k_metadata(
        acquisition_model=acquisition_model,
        converter_options=converter_options,
    )

    # Ten (well, field of view) groups over two plates: `Maximum` and `Minimum`.
    # Per plate rather than per acquisition, since the two carry different z
    # spacings.
    assert len(calls) == 2
    assert tiled_images
    assert {tuple(tiled_image.axes) for tiled_image in tiled_images} == {
        ("c", "z", "y", "x")
    }


@pytest.mark.extended
def test_cellvoyager_acquisition_details_are_acquisition_scoped(
    monkeypatch, converter_options
):
    """Multi-field wells merge without an AcquisitionDetails mismatch."""
    calls = _count_acquisition_details_calls(monkeypatch, cellvoyager_utils)
    acquisition_model = CellVoyagerAcquisitionModel(
        path=str(CELLVOYAGER_EXTENDED_RAW_DIR / "partial-tile_unique-targets"),
        acquisition_id=0,
        advanced={"channels": _FIVE_CHANNEL_OVERRIDE},
    )
    tiled_images = parse_cellvoyager_metadata(
        acquisition_model=acquisition_model,
        converter_options=converter_options,
    )

    # Four (well, field of view) groups, but a single plate.
    assert len(calls) == 1
    assert tiled_images
    assert {tuple(tiled_image.axes) for tiled_image in tiled_images} == {
        ("c", "z", "y", "x")
    }
