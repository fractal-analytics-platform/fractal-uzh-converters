"""Robustness of the Yokogawa `.mlf`/`.mrf` parsers against real-world variation.

Four independent defects, each with a concrete failing input:

1. `xmltodict` collapses a lone `<bts:MeasurementRecord>` to a bare dict, so the
   CQ3K parser rejected any acquisition holding exactly one record —
   `hcs_1w1p1c1z1t_SearchFirst_SP_Grid` in the extended store.
2. The CQ3K models used `extra="forbid"`, making a single new `bts:` attribute
   from a firmware update a hard parse failure.
3. Both converters take pixel size and frame size from the *first* `.mrf`
   channel and apply them to all of them, silently.
4. `bts:Type="ERR"` records carry no `bts:FieldIndex`, which was required on the
   shared record base — one autofocus failure made the whole `.mlf` unparsable.

The fixtures are derived from the in-repo CQ3K and CellVoyager acquisitions by
rewriting their XML in `tmp_path`, so all four are covered by the fast suite.
"""

import re
import shutil
import warnings
from pathlib import Path

import pytest

from fractal_uzh_converters.common import GeometryWarning, SourceMetadataWarning
from fractal_uzh_converters.yokogawa._channels import (
    ChannelGeometry,
    warn_on_channel_geometry_mismatch,
)
from fractal_uzh_converters.yokogawa.cellvoyager._utils import (
    CellVoyagerAcquisitionModel,
    parse_cellvoyager_metadata,
)
from fractal_uzh_converters.yokogawa.cq3k._utils import (
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


#: The two in-repo Yokogawa acquisitions, with what each parser needs to read one.
BOTH_CONVERTERS = pytest.mark.parametrize(
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
def test_agreeing_channel_geometry_is_silent(geometries):
    """Nothing to report when the channels agree, or there is nothing to compare.

    `pytest.warns` cannot assert an absence, so the recording context is opened by
    hand. `simplefilter("always")` overrides the suite-wide
    `ignore::ConverterWarning`, which would otherwise make this pass vacuously.
    """
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        warn_on_channel_geometry_mismatch(
            geometries=geometries, acquisition_dir="/some/acquisition"
        )
    assert raised == []


@pytest.mark.parametrize(
    "odd_one_out, expected",
    [
        pytest.param(_geometry(2, pixel_size=0.25), "0.25", id="pixel-size"),
        pytest.param(_geometry(2, frame=1000), "1000", id="frame-size"),
    ],
)
def test_disagreeing_channel_geometry_warns(odd_one_out, expected: str):
    """A channel whose geometry differs from channel 1's is reported, once."""
    with pytest.warns(GeometryWarning) as raised:
        warn_on_channel_geometry_mismatch(
            geometries=[_geometry(1), odd_one_out],
            acquisition_dir="/some/acquisition",
        )

    assert len(raised) == 1
    message = str(raised[0].message)
    assert "bts:Ch 2" in message
    assert expected in message
    assert "/some/acquisition" in message


@BOTH_CONVERTERS
def test_geometry_warning_does_not_change_the_applied_pixel_size(
    tmp_path: Path,
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

    with pytest.warns(GeometryWarning) as raised:
        tiled_images = parse_fn(
            acquisition_model=model_cls(
                path=str(acquisition_dir), acquisition_id=0, **model_kwargs
            ),
            converter_options=converter_options,
        )

    reported = [str(w.message) for w in raised if "bts:Ch 99" in str(w.message)]
    assert len(reported) == 1, "expected exactly one warning, per acquisition"

    # Unchanged behaviour: every image still carries channel 1's pixel size. The
    # injected channel 99 declares a ten-times-larger one.
    applied = {image.xy_pixel_size for image in tiled_images}
    assert len(applied) == 1
    assert applied.pop() < 1.0, "channel 1's sub-micron pixel size, not channel 99's"


######################################################################
#
# 4. `bts:Type="ERR"` measurement records
#
######################################################################

# Verbatim from issue #41, with its data already mangled by the reporter. Note
# what an autofocus failure does *not* carry: no `bts:FieldIndex`, and no
# `bts:ZIndex`/`bts:Ch`/`bts:Action` either.
ERROR_RECORDS = (
    '<bts:MeasurementRecord bts:Type="ERR" bts:Time="2020-01-01T11:00:00.000+01:00"'
    ' bts:Column="14" bts:Row="3" bts:TimePoint="1" bts:PartialTileIndex="1"'
    ' bts:TimelineIndex="1" bts:X="-20.0" bts:Y="25.0">'
    "AF Error at C14 No. 0 (SAMPLE_PLATE_000000_000000)</bts:MeasurementRecord>\n"
    '<bts:MeasurementRecord bts:Type="ERR" bts:Time="2020-01-01T12:00:00.000+01:00"'
    ' bts:Column="9" bts:Row="14" bts:TimePoint="1" bts:PartialTileIndex="1"'
    ' bts:TimelineIndex="1" bts:X="-20.0" bts:Y="25.0">'
    "AF Error at N09 No. 0 (SAMPLE_PLATE_000000_000000)</bts:MeasurementRecord>\n"
)


def _add_error_records(acquisition_dir: Path) -> None:
    """Append the two ERR records of issue #41 to a copied `.mlf`."""
    mlf = acquisition_dir / "MeasurementData.mlf"
    text = mlf.read_text(encoding="utf-8")
    closing = "</bts:MeasurementData>"
    assert closing in text, "fixture has no closing <bts:MeasurementData>"
    mlf.write_text(text.replace(closing, ERROR_RECORDS + closing), encoding="utf-8")


def _drop_image_records(acquisition_dir: Path) -> None:
    """Remove every `<bts:MeasurementRecord>` line from a copied `.mlf`."""
    mlf = acquisition_dir / "MeasurementData.mlf"
    lines = mlf.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if "<bts:MeasurementRecord" not in line]
    assert len(kept) < len(lines), "fixture has no <bts:MeasurementRecord>"
    mlf.write_text("".join(kept), encoding="utf-8")


@BOTH_CONVERTERS
def test_error_records_are_skipped(
    tmp_path: Path,
    converter_options,
    fixture: Path,
    model_cls,
    parse_fn,
    model_kwargs: dict,
):
    """An ERR record must be dropped, not fail the whole acquisition (#41).

    `bts:FieldIndex` used to be required on the shared record base, so a single
    autofocus failure made the `.mlf` unparsable in both converters.
    """

    def parse(acquisition_dir: Path):
        return parse_fn(
            acquisition_model=model_cls(
                path=str(acquisition_dir), acquisition_id=0, **model_kwargs
            ),
            converter_options=converter_options,
        )

    # The unmodified fixture is only read, never written.
    baseline = parse(fixture)
    assert baseline, "fixture parses to nothing; there is nothing to compare against"

    acquisition_dir = _copy_acquisition(fixture, tmp_path)
    _add_error_records(acquisition_dir)
    tiled_images = parse(acquisition_dir)

    # ERR records describe no image, so the output must be untouched by them.
    assert len(tiled_images) == len(baseline)
    assert sum(len(image.regions) for image in tiled_images) == sum(
        len(image.regions) for image in baseline
    )


@BOTH_CONVERTERS
def test_error_records_warn_once_per_acquisition(
    tmp_path: Path,
    converter_options,
    fixture: Path,
    model_cls,
    parse_fn,
    model_kwargs: dict,
):
    """Two skipped records, one warning — not one warning per record.

    A plate-wide autofocus failure produces hundreds of ERR records, so the
    count is reported once for the acquisition.
    """
    acquisition_dir = _copy_acquisition(fixture, tmp_path)
    _add_error_records(acquisition_dir)

    with pytest.warns(SourceMetadataWarning) as raised:
        parse_fn(
            acquisition_model=model_cls(
                path=str(acquisition_dir), acquisition_id=0, **model_kwargs
            ),
            converter_options=converter_options,
        )

    reported = [str(w.message) for w in raised if "bts:Type='ERR'" in str(w.message)]
    assert len(reported) == 1, "expected exactly one warning, per acquisition"
    assert "2" in reported[0], "the warning must carry the count"
    assert str(acquisition_dir) in reported[0]


@BOTH_CONVERTERS
def test_all_error_mlf_raises(
    tmp_path: Path,
    converter_options,
    fixture: Path,
    model_cls,
    parse_fn,
    model_kwargs: dict,
):
    """An `.mlf` of nothing but ERR records fails with a message that says so.

    It holds records, so the empty-`.mlf` check does not catch it; without an
    explicit guard it would fall through to an obscure failure downstream.
    """
    acquisition_dir = _copy_acquisition(fixture, tmp_path)
    _drop_image_records(acquisition_dir)
    _add_error_records(acquisition_dir)

    with pytest.raises(ValueError, match="No image measurement records"):
        parse_fn(
            acquisition_model=model_cls(
                path=str(acquisition_dir), acquisition_id=0, **model_kwargs
            ),
            converter_options=converter_options,
        )
