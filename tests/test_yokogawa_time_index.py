"""Time index normalisation for the Yokogawa converters.

`bts:TimePoint` counts *timelines*, not frames: wells acquired by three timelines
carry `TimePoint` 1, 2 and 3 although each holds a single time point. On a `czyx`
image that is inert (`Tile.to_roi` drops `start_t`), which is why no snapshot
moves; it bites as soon as a well genuinely holds several time points, since the
raw counter leaves the unused indices as empty frames and there is no `reindex_t`
to compact them.

No acquisition in either reference store has more than one time point per well, so
the multi-time-point cases are hand-built by rewriting an in-repo fixture's `.mlf`
in `tmp_path`. The metadata is under test, not the pixels.
"""

import re
import shutil
from pathlib import Path

import pytest
from ome_zarr_converters_tools.testing import build_snapshot

from fractal_uzh_converters.yokogawa import _parse as yokogawa_parse
from fractal_uzh_converters.yokogawa._channels import build_time_index
from fractal_uzh_converters.yokogawa.cellvoyager import convert_cellvoyager
from fractal_uzh_converters.yokogawa.cq3k import convert_cq3k

from .utils import DATA_DIR

CQ3K_FIXTURE = DATA_DIR / "Yokogawa-CQ3K" / "raw" / "hcs_2w1p1c1z1t_mip"
CELLVOYAGER_FIXTURE = DATA_DIR / "Yokogawa-CellVoyager" / "raw" / "hcs_1w1p1c1z1t"


######################################################################
#
# The dense mapping itself
#
######################################################################


def test_one_time_point_per_well_is_not_a_time_series():
    """The real-data case: `TimePoint` counting timelines, not frames.

    `TimelinesSharedChannel` acquires three wells with three timelines, so its
    records carry `TimePoint` 1, 2 and 3 while every well holds a single time point. The
    plate must stay `czyx`, and every record must land on frame 0.
    """
    time_index = build_time_index([(3, 2, 1), (3, 3, 2), (3, 4, 3)])

    assert not time_index.is_time_series
    assert time_index.start_t(row=3, column=2, time_point=1) == 0
    assert time_index.start_t(row=3, column=3, time_point=2) == 0
    assert time_index.start_t(row=3, column=4, time_point=3) == 0


def test_sparse_time_points_are_compacted():
    """A well's distinct `TimePoint` values map onto 0..n-1, gaps closed."""
    time_index = build_time_index([(3, 2, 7), (3, 2, 1), (3, 2, 3), (3, 2, 7)])

    assert time_index.is_time_series
    assert time_index.dense_indices == {(3, 2): {1: 0, 3: 1, 7: 2}}


def test_time_series_is_decided_across_the_whole_plate():
    """One multi-frame well makes the plate a time series; the others still start at 0.

    `TiledImage.add_tile` requires the tiles of an image to agree on `axes`, so
    the `t` axis cannot be decided per well — but the *indices* are per well, so
    a single-frame well keeps one frame instead of inheriting the other's range.
    """
    time_index = build_time_index([(3, 2, 1), (3, 2, 3), (3, 5, 9)])

    assert time_index.is_time_series
    assert time_index.dense_indices == {(3, 2): {1: 0, 3: 1}, (3, 5): {9: 0}}


def test_no_records_is_not_a_time_series():
    """An empty plate has no time axis and no wells to index."""
    time_index = build_time_index([])

    assert not time_index.is_time_series
    assert time_index.dense_indices == {}


######################################################################
#
# Wiring: the inert case the reference data actually contains
#
######################################################################


def _capture_tiles(monkeypatch) -> list:
    """Collect every Tile the shared parser's `_build_tiles` returns.

    The tiles are the only place `start_t` survives on a `czyx` plate: `to_roi`
    drops it, so it cannot be observed on the TiledImage or in a snapshot.

    Patched on `yokogawa/_parse.py`, which both converters call into — patching
    a re-export on either instrument package would not intercept.
    """
    tiles: list = []
    original = yokogawa_parse._build_tiles

    def _spy(**kwargs):
        built = original(**kwargs)
        tiles.extend(built)
        return built

    monkeypatch.setattr(yokogawa_parse, "_build_tiles", _spy)
    return tiles


@pytest.mark.parametrize(
    "fixture, api_fn, acquisition_kwargs",
    [
        pytest.param(CQ3K_FIXTURE, convert_cq3k, {}, id="cq3k"),
        pytest.param(
            CELLVOYAGER_FIXTURE,
            convert_cellvoyager,
            # The in-repo CellVoyager fixture ships `.png`, not `.tif`.
            {"image_extension": ".png"},
            id="cellvoyager",
        ),
    ],
)
def test_single_time_point_starts_at_frame_zero(
    tmp_path: Path,
    monkeypatch,
    converter_options,
    fixture: Path,
    api_fn,
    acquisition_kwargs: dict,
):
    """A lone time point lands on frame 0 whatever `bts:TimePoint` says.

    The in-repo CellVoyager fixture records `bts:TimePoint="2"`, which used to
    become `start_t=1` on a `czyx` image.
    """
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()
    tiles = _capture_tiles(monkeypatch)

    api_fn(
        zarr_dir=str(zarr_dir),
        acquisitions=[
            {"path": str(fixture), "acquisition_id": 0, **acquisition_kwargs}
        ],
        converter_options=converter_options,
    )

    assert tiles
    assert {tile.start_t for tile in tiles} == {0}


######################################################################
#
# Wiring: a genuine multi-time-point acquisition
#
######################################################################


def _set_attributes(record: str, **attributes: int) -> str:
    """Overwrite `bts:` attributes of one `<bts:MeasurementRecord>` line."""
    for name, value in attributes.items():
        record, count = re.subn(rf'bts:{name}="[^"]*"', f'bts:{name}="{value}"', record)
        assert count == 1, f"expected exactly one bts:{name} in the record"
    return record


def _rewrite_time_points(
    source: Path, tmp_path: Path, wells: dict[tuple[int, int], list[int]]
) -> Path:
    """Copy an acquisition and give each well the requested `bts:TimePoint`s.

    All records are cloned from the fixture's first one, so they share its image
    file, position, z index and channel and differ only in well and time point.
    """
    destination = tmp_path / source.name
    shutil.copytree(source, destination)

    mlf = destination / "MeasurementData.mlf"
    lines = mlf.read_text(encoding="utf-8").splitlines(keepends=True)
    indices = [i for i, line in enumerate(lines) if "<bts:MeasurementRecord" in line]
    assert indices, "fixture has no <bts:MeasurementRecord>"

    records = [
        _set_attributes(lines[indices[0]], Row=row, Column=column, TimePoint=time_point)
        for (row, column), time_points in wells.items()
        for time_point in time_points
    ]
    mlf.write_text(
        "".join(lines[: indices[0]] + records + lines[indices[-1] + 1 :]),
        encoding="utf-8",
    )
    return destination


def _image_shapes(
    zarr_dir: Path, image_list_updates: list[dict]
) -> dict[str, tuple[tuple[str, ...], tuple[int, ...]]]:
    """Return `{image_path: (axes, shape)}` across every plate of a run."""
    snapshot = build_snapshot(
        zarr_dir=zarr_dir,
        image_list_updates=image_list_updates,
        output_type="plate",
    )
    return {
        image_path: (tuple(image.axes), tuple(image.shape))
        for plate in snapshot.plates.values()
        for image_path, image in plate.images.items()
    }


# Well B03 is imaged at `bts:TimePoint` 1 and 3 — two frames with a gap — while
# C05 is imaged once, at 7. Taken literally that is a 3-frame and a 7-frame
# image, six of whose frames are empty.
_SPARSE_WELLS = {(2, 3): [1, 3], (3, 5): [7]}


@pytest.mark.parametrize(
    "fixture, api_fn, acquisition_kwargs",
    [
        pytest.param(CQ3K_FIXTURE, convert_cq3k, {}, id="cq3k"),
        pytest.param(
            CELLVOYAGER_FIXTURE,
            convert_cellvoyager,
            {"image_extension": ".png"},
            id="cellvoyager",
        ),
    ],
)
def test_sparse_time_points_convert_to_dense_frames(
    tmp_path: Path,
    converter_options,
    fixture: Path,
    api_fn,
    acquisition_kwargs: dict,
):
    """Sparse per-well `TimePoint`s become consecutive frames, per well."""
    acquisition_dir = _rewrite_time_points(fixture, tmp_path, _SPARSE_WELLS)
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()

    image_list_updates = api_fn(
        zarr_dir=str(zarr_dir),
        acquisitions=[
            {
                "path": str(acquisition_dir),
                "acquisition_id": 0,
                **acquisition_kwargs,
            }
        ],
        converter_options=converter_options,
    )

    images = _image_shapes(zarr_dir, image_list_updates)
    assert sorted(images) == ["B/03/0", "C/05/0"]
    for axes, _ in images.values():
        # One well holding two time points makes the whole plate a time series.
        assert axes == ("t", "c", "z", "y", "x")
    assert images["B/03/0"][1][0] == 2, "TimePoint 1 and 3 are frames 0 and 1"
    assert images["C/05/0"][1][0] == 1, "a lone TimePoint 7 is frame 0"
