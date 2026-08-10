"""Robustness of the Yokogawa `.mlf`/`.mrf` parsers against real-world variation.

Three independent defects, each with a concrete failing input:

1. `xmltodict` collapses a lone `<bts:MeasurementRecord>` to a bare dict, so the
   CQ3K parser rejected any acquisition holding exactly one record —
   `20251201T140949_BVC_TS_Test_SP_grid` in the extended store.
2. The CQ3K models used `extra="forbid"`, making a single new `bts:` attribute
   from a firmware update a hard parse failure.
3. Both converters take pixel size and frame size from the *first* `.mrf`
   channel and apply them to all of them, silently.

The fixtures are derived from the in-repo CQ3K and CellVoyager acquisitions by
rewriting their XML in `tmp_path`, so all three are covered by the fast suite.
"""

import logging
import re
import shutil
from pathlib import Path

import pytest

from fractal_uzh_converters.cellvoyager._utils import (
    CellVoyagerAcquisitionModel,
    parse_cellvoyager_metadata,
)
from fractal_uzh_converters.common import (
    ChannelGeometry,
    warn_on_channel_geometry_mismatch,
)
from fractal_uzh_converters.cq3k._utils import (
    CQ3KAcquisitionModel,
    parse_cq3k_metadata,
)

from .utils import DATA_DIR

CQ3K_FIXTURE = DATA_DIR / "Yokogawa-CQ3K" / "raw" / "hcs_2w1p1c1z1t_mip"
CELLVOYAGER_FIXTURE = DATA_DIR / "Yokogawa-CellVoyager" / "raw" / "hcs_1w1p1c1z1t"


def _copy_acquisition(source: Path, tmp_path: Path) -> Path:
    """Copy an acquisition into `tmp_path` so its XML can be rewritten."""
    destination = tmp_path / source.name
    shutil.copytree(source, destination)
    return destination


######################################################################
#
# 1. Single-record `.mlf`
#
######################################################################


def test_single_record_mlf_parses(tmp_path: Path, converter_options):
    """One `<bts:MeasurementRecord>` must parse as readily as several.

    `xmltodict` returns a bare dict rather than a one-element list for it, which
    used to fail validation against `list[...] | None`.
    """
    acquisition_dir = _copy_acquisition(CQ3K_FIXTURE, tmp_path)
    mlf = acquisition_dir / "MeasurementData.mlf"
    lines = mlf.read_text(encoding="utf-8").splitlines(keepends=True)
    record_indices = [
        i for i, line in enumerate(lines) if "<bts:MeasurementRecord" in line
    ]
    assert len(record_indices) == 2, "fixture changed; this test trims 2 records to 1"
    # Drop all but the first record.
    del lines[record_indices[1]]
    mlf.write_text("".join(lines), encoding="utf-8")

    tiled_images = parse_cq3k_metadata(
        acquisition_model=CQ3KAcquisitionModel(
            path=str(acquisition_dir), acquisition_id=0
        ),
        converter_options=converter_options,
    )

    assert len(tiled_images) == 1
    assert sum(len(image.regions) for image in tiled_images) == 1


######################################################################
#
# 2. Unknown `bts:` attributes
#
######################################################################


@pytest.mark.parametrize("filename", ["MeasurementData.mlf", "MeasurementDetail.mrf"])
def test_unknown_bts_attribute_is_ignored(
    tmp_path: Path, converter_options, filename: str
):
    """A `bts:` attribute the models do not declare must not fail the parse.

    A firmware update adding one attribute would otherwise make every existing
    acquisition unconvertible.
    """
    acquisition_dir = _copy_acquisition(CQ3K_FIXTURE, tmp_path)
    target = acquisition_dir / filename
    # Add the attribute to every opening `bts:` tag, so the record, the detail and
    # the channel models all see one they do not declare. Closing tags (`</bts:`)
    # do not match.
    text = re.sub(
        r"(<bts:\w+)",
        r'\1 bts:SomeFutureAttribute="1"',
        target.read_text(encoding="utf-8"),
    )
    target.write_text(text, encoding="utf-8")

    tiled_images = parse_cq3k_metadata(
        acquisition_model=CQ3KAcquisitionModel(
            path=str(acquisition_dir), acquisition_id=0
        ),
        converter_options=converter_options,
    )

    assert tiled_images


######################################################################
#
# 3. Per-channel geometry disagreement
#
######################################################################


def _geometry(ch: int, pixel_size: float = 0.5, frame: int = 2000) -> ChannelGeometry:
    return ChannelGeometry(
        ch=ch, xy_pixel_size=(pixel_size, pixel_size), frame_size=(frame, frame)
    )


@pytest.mark.parametrize(
    "geometries",
    [
        pytest.param([], id="no-channels"),
        pytest.param([_geometry(1)], id="one-channel"),
        pytest.param([_geometry(1), _geometry(2)], id="agreeing"),
    ],
)
def test_agreeing_channel_geometry_is_silent(caplog, geometries):
    """Nothing to report when the channels agree, or there is nothing to compare."""
    with caplog.at_level(logging.WARNING):
        warn_on_channel_geometry_mismatch(
            geometries=geometries, acquisition_dir="/some/acquisition"
        )
    assert caplog.records == []


@pytest.mark.parametrize(
    "odd_one_out, expected",
    [
        pytest.param(_geometry(2, pixel_size=0.25), "0.25", id="pixel-size"),
        pytest.param(_geometry(2, frame=1000), "1000", id="frame-size"),
    ],
)
def test_disagreeing_channel_geometry_warns(caplog, odd_one_out, expected: str):
    """A channel whose geometry differs from channel 1's is reported, once."""
    with caplog.at_level(logging.WARNING):
        warn_on_channel_geometry_mismatch(
            geometries=[_geometry(1), odd_one_out],
            acquisition_dir="/some/acquisition",
        )

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "bts:Ch 2" in message
    assert expected in message
    assert "/some/acquisition" in message


@pytest.mark.parametrize(
    "fixture, model_cls, parse_fn, model_kwargs",
    [
        pytest.param(
            CQ3K_FIXTURE, CQ3KAcquisitionModel, parse_cq3k_metadata, {}, id="cq3k"
        ),
        pytest.param(
            CELLVOYAGER_FIXTURE,
            CellVoyagerAcquisitionModel,
            parse_cellvoyager_metadata,
            # The in-repo CellVoyager fixture ships `.png`, not `.tif`.
            {"image_extension": ".png"},
            id="cellvoyager",
        ),
    ],
)
def test_geometry_warning_does_not_change_the_applied_pixel_size(
    tmp_path: Path,
    caplog,
    converter_options,
    fixture: Path,
    model_cls,
    parse_fn,
    model_kwargs: dict,
):
    """The warning reports a suspicion; channel 1's geometry is still applied.

    Both converters read pixel size and frame size from the first `.mrf` channel.
    This fix warns about a disagreement, it does not attempt to resolve one.
    """
    acquisition_dir = _copy_acquisition(fixture, tmp_path)
    mrf = acquisition_dir / "MeasurementDetail.mrf"
    text = mrf.read_text(encoding="utf-8")
    match = re.search(r"<bts:MeasurementChannel\b.*?/>", text, flags=re.DOTALL)
    assert match is not None, "fixture has no <bts:MeasurementChannel>"
    first_channel = match.group()
    # Append a second channel that declares a ten-times-larger pixel size. `bts:Ch`
    # 99 keeps it out of the acquired range, so only the geometry check sees it.
    extra_channel = (
        re.sub(r'bts:Ch="\d+"', 'bts:Ch="99"', first_channel)
        .replace('bts:HorizontalPixelDimension="', 'bts:HorizontalPixelDimension="9')
        .replace('bts:VerticalPixelDimension="', 'bts:VerticalPixelDimension="9')
    )
    mrf.write_text(text.replace(first_channel, first_channel + extra_channel), "utf-8")

    with caplog.at_level(logging.WARNING):
        tiled_images = parse_fn(
            acquisition_model=model_cls(
                path=str(acquisition_dir), acquisition_id=0, **model_kwargs
            ),
            converter_options=converter_options,
        )

    warnings = [r.message for r in caplog.records if "bts:Ch 99" in r.message]
    assert len(warnings) == 1, "expected exactly one warning, per acquisition"

    # Unchanged behaviour: every image still carries channel 1's pixel size. The
    # injected channel 99 declares a ten-times-larger one.
    applied = {image.xy_pixel_size for image in tiled_images}
    assert len(applied) == 1
    assert applied.pop() < 1.0, "channel 1's sub-micron pixel size, not channel 99's"
