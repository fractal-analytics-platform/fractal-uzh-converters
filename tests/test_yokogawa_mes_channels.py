"""Tests for the Yokogawa `.mes` reader and channel resolution.

`common/_yokogawa.py` is not wired into either converter yet — these cover it
directly, so the behaviour is pinned before the wiring rewrites every Yokogawa
snapshot.

The `.mes` file supplies the human-readable channel targets that the `.mlf`/`.mrf`
pair lacks. The resolved list spans the full instrument channel range, not the
acquired subset: pruning to what a given image actually acquired happens at
compute time in the library's `reindex_channels`, so the per-well expectations
below simulate it with `[full[ch - 1] for ch in sorted(acquired)]` rather than
asserting an end-to-end result.
"""

from pathlib import Path

import pytest
from ome_zarr_converters_tools import ChannelInfo, ChannelInfoUI

from fractal_uzh_converters.common._yokogawa import (
    MesChannel,
    _actions_by_channel,
    _dedup_labels,
    _strip_argb_alpha,
    _wavelength_id,
    apply_channel_overrides,
    max_acquired_channel,
    parse_mes,
    read_mes_channels,
    resolve_channels,
)

from .utils import DATA_DIR

EXTENDED_DATA_DIR = Path(__file__).parent / "data-extended"

CQ3K_RAW_DIR = DATA_DIR / "Yokogawa-CQ3K" / "raw"
CELLVOYAGER_RAW_DIR = DATA_DIR / "Yokogawa-CellVoyager" / "raw"
CQ3K_EXTENDED_RAW_DIR = EXTENDED_DATA_DIR / "Yokogawa-CQ3K" / "raw"
CELLVOYAGER_EXTENDED_RAW_DIR = EXTENDED_DATA_DIR / "Yokogawa-CellVoyager" / "raw"

CQ3K_FIXTURE = CQ3K_RAW_DIR / "hcs_2w1p1c1z1t_mip"
CELLVOYAGER_FIXTURE = CELLVOYAGER_RAW_DIR / "hcs_1w1p1c1z1t"

# The one channel of the in-repo CQ3K fixture.
_CQ3K_FIXTURE_TARGET = "Ch1_ConfocalFluorescence_405nm/100mW_BP447/60"

# The `.mes` this `.mrf` names was never shipped with the CV7000 demo fixture.
_CELLVOYAGER_FIXTURE_MES = "20200812-Joel-CardiomyocyteDifferentiation14-Cycle1.mes"

_MES_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<bts:MeasurementSetting xmlns:bts="http://www.yokogawa.co.jp/BTS/BTSSchema/1.0"
                        bts:Version="1.0" bts:ProductID="10">
{body}
</bts:MeasurementSetting>
"""

_CHANNEL_LIST = """  <bts:ChannelList>
{channels}
  </bts:ChannelList>"""


def _write_mes(tmp_path: Path, body: str, name: str = "Protocol.mes") -> str:
    """Write a synthetic `.mes` and return its URL."""
    path = tmp_path / name
    path.write_text(_MES_TEMPLATE.format(body=body), encoding="utf-8")
    return str(path)


def _channel_list(*channels: str) -> str:
    return _CHANNEL_LIST.format(channels="\n".join(f"    {c}" for c in channels))


# ---------------------------------------------------------------------------
# Reading the `.mes`
# ---------------------------------------------------------------------------


class TestReadMesChannels:
    """Discovery and parsing of the `.mes` channel list."""

    def test_in_repo_cq3k_fixture(self):
        """A single `<bts:Channel>` still yields a one-element list.

        `xmltodict` collapses a lone child element to a bare dict rather than a
        one-element list; this fixture is the fast suite's only cover of that.
        """
        channels = read_mes_channels(
            acquisition_dir=str(CQ3K_FIXTURE),
            mes_file_name="MeasurementProtocol.mes",
        )

        assert channels == [
            MesChannel(ch=1, target=_CQ3K_FIXTURE_TARGET, color="#FF00FFFF")
        ]

    def test_discovery_goes_through_the_mrf(self):
        """The `.mes` name comes from the `.mrf`, never from a `*.mes` glob.

        Sibling acquisitions can share a directory, and two acquisitions can
        carry the same `.mes` basename with different content.
        """
        from fractal_uzh_converters.cq3k._utils import _load_models

        _, mrf = _load_models(str(CQ3K_FIXTURE))
        assert mrf.measurement_setting_file_name == "MeasurementProtocol.mes"

        channels = read_mes_channels(
            acquisition_dir=str(CQ3K_FIXTURE),
            mes_file_name=mrf.measurement_setting_file_name,
        )
        assert channels is not None
        assert [channel.ch for channel in channels] == [1]

    def test_missing_mes_is_not_an_error(self, caplog):
        """A `.mes` named in the `.mrf` but absent falls back, it does not raise."""
        with caplog.at_level("WARNING"):
            channels = read_mes_channels(
                acquisition_dir=str(CELLVOYAGER_FIXTURE),
                mes_file_name=_CELLVOYAGER_FIXTURE_MES,
            )

        assert channels is None
        assert "does not exist" in caplog.text

    def test_unnamed_mes(self, caplog):
        with caplog.at_level("WARNING"):
            assert (
                read_mes_channels(acquisition_dir=str(CQ3K_FIXTURE), mes_file_name=None)
                is None
            )
        assert "No `.mes` file name" in caplog.text

    def test_channels_are_sorted_by_ch(self, tmp_path: Path):
        url = _write_mes(
            tmp_path,
            _channel_list(
                '<bts:Channel bts:Ch="3" bts:Target="c" />',
                '<bts:Channel bts:Ch="1" bts:Target="a" />',
                '<bts:Channel bts:Ch="2" bts:Target="b" />',
            ),
        )
        channels = read_mes_channels(
            acquisition_dir=str(tmp_path), mes_file_name=Path(url).name
        )
        assert channels is not None
        assert [channel.target for channel in channels] == ["a", "b", "c"]

    def test_unknown_attributes_are_ignored(self, tmp_path: Path):
        """CV8000 adds `CSUID`/`PinholeDiameter`/`InputLevel`/`Fluorophore`."""
        url = _write_mes(
            tmp_path,
            _channel_list(
                '<bts:Channel bts:Ch="1" bts:Target="405" bts:CSUID="2" '
                'bts:PinholeDiameter="50" bts:InputLevel="10000" '
                'bts:Fluorophore="" bts:Whatever="new in a firmware update">'
                "<bts:LightSourceName>405nm</bts:LightSourceName>"
                "<bts:LightSourceName>640nm</bts:LightSourceName>"
                "</bts:Channel>",
            ),
        )
        channels = read_mes_channels(
            acquisition_dir=str(tmp_path), mes_file_name=Path(url).name
        )
        assert channels == [MesChannel(ch=1, target="405", color=None)]

    def test_no_channel_list(self, tmp_path: Path, caplog):
        url = _write_mes(tmp_path, "  <bts:LightSourceList />")
        with caplog.at_level("WARNING"):
            assert (
                read_mes_channels(
                    acquisition_dir=str(tmp_path), mes_file_name=Path(url).name
                )
                is None
            )
        assert "no channel list" in caplog.text

    def test_malformed_xml_raises(self, tmp_path: Path):
        path = tmp_path / "broken.mes"
        path.write_text("<bts:MeasurementSetting>", encoding="utf-8")
        with pytest.raises(Exception, match="."):
            parse_mes(str(path))

    def test_missing_version_attribute_is_accepted(self, tmp_path: Path):
        """The CV8000 `sf-test_071026_133615` protocol has no `bts:Version`."""
        path = tmp_path / "no_version.mes"
        path.write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            "<bts:MeasurementSetting "
            'xmlns:bts="http://www.yokogawa.co.jp/BTS/BTSSchema/1.0">\n'
            '  <bts:ChannelList><bts:Channel bts:Ch="1" bts:Target="405" />'
            "</bts:ChannelList>\n"
            "</bts:MeasurementSetting>\n",
            encoding="utf-8",
        )
        setting = parse_mes(str(path))
        assert setting.channel_list is not None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPureHelpers:
    """The naming primitives, with no filesystem involved."""

    @pytest.mark.parametrize(
        "labels, expected",
        [
            # The `same-target` / `partial-tile_duplicate-targets` channel list.
            (
                ["1", "test", "test", "test", "test"],
                ["1", "test", "test_1", "test_2", "test_3"],
            ),
            # A suffix that would collide with a label already in the list.
            (["a", "a_1", "a", "a"], ["a", "a_1", "a_2", "a_3"]),
            (["405", "488", "561"], ["405", "488", "561"]),
            ([], []),
        ],
    )
    def test_dedup_labels(self, labels, expected):
        assert _dedup_labels(labels) == expected

    @pytest.mark.parametrize(
        "color, expected",
        [
            ("#FF002FFF", "#002FFF"),  # ARGB
            ("#FFFFFFFF", "#FFFFFF"),
            ("#002FFF", "#002FFF"),  # already RGB
            ("#abc", "#abc"),
            (None, None),
            ("nonsense", None),
            ("#12345", None),
            ("#GGGGGG", None),
        ],
    )
    def test_strip_argb_alpha(self, color, expected):
        assert _strip_argb_alpha(color, ch=1) == expected

    @pytest.mark.parametrize(
        "action_index, ch, expected",
        [(1, 1, "A01_C01"), (3, 12, "A03_C12"), (10, 10, "A10_C10")],
    )
    def test_wavelength_id(self, action_index, ch, expected):
        assert _wavelength_id(action_index, ch) == expected

    def test_unacquired_slot_placeholder(self):
        """Slots the acquisition never used get action `00`."""
        assert _wavelength_id(0, 5) == "A00_C05"

    def test_actions_by_channel(self):
        assert _actions_by_channel([(2, 1), (2, 1), (1, 4)]) == {1: 2, 4: 1}

    def test_actions_by_channel_warns_on_ambiguity(self, caplog):
        """`(ActionIndex, Ch)` is 1:1 on `Ch` in all known data."""
        with caplog.at_level("WARNING"):
            assert _actions_by_channel([(3, 1), (1, 1)]) == {1: 1}
        assert "more than one action" in caplog.text

    @pytest.mark.parametrize(
        "acquired, expected", [([], 0), ([(1, 2), (1, 4), (2, 3)], 4)]
    )
    def test_max_acquired_channel(self, acquired, expected):
        assert max_acquired_channel(acquired) == expected


# ---------------------------------------------------------------------------
# Channel resolution
# ---------------------------------------------------------------------------


class TestResolveChannels:
    """The six naming rules, over the in-repo fixtures and synthetic input."""

    def test_in_repo_cq3k_fixture(self):
        """Rules 1, 2 and 6 end to end."""
        channels = read_mes_channels(
            acquisition_dir=str(CQ3K_FIXTURE),
            mes_file_name="MeasurementProtocol.mes",
        )
        resolved = resolve_channels(
            mes_channels=channels, acquired=[(1, 1)], mrf_channel_count=1
        )

        assert resolved == [
            ChannelInfo(
                channel_label=_CQ3K_FIXTURE_TARGET,
                wavelength_id="A01_C01",
                color="#00FFFF",  # `#FF00FFFF` with the ARGB alpha stripped
            )
        ]

    def test_no_mes_falls_back_to_the_wavelength_id(self):
        """Rule 3, as exercised by the in-repo CV7000 fixture."""
        resolved = resolve_channels(
            mes_channels=None, acquired=[(1, 1)], mrf_channel_count=1
        )

        assert len(resolved) == 1
        assert resolved[0].channel_label == "A01_C01"
        assert resolved[0].wavelength_id == "A01_C01"
        # The colour is the library's own heuristic here, not our contract.

    def test_empty_target_falls_back_to_the_wavelength_id(self):
        resolved = resolve_channels(
            mes_channels=[MesChannel(ch=1, target="   ")],
            acquired=[(2, 1)],
            mrf_channel_count=1,
        )
        assert resolved[0].channel_label == "A02_C01"

    def test_spans_the_full_instrument_range(self):
        """Unacquired slots are kept as placeholders, not dropped.

        Keeping index `i` at `bts:Ch == i + 1` is what makes the converters'
        `start_c = ch - 1` and a positional user override line up.
        """
        mes_channels = [
            MesChannel(ch=ch, target=target)
            for ch, target in enumerate(["405", "488", "561", "640", "BF"], start=1)
        ]
        resolved = resolve_channels(
            mes_channels=mes_channels,
            acquired=[(3, 2), (2, 3), (1, 4)],
            mrf_channel_count=5,
        )

        assert [c.channel_label for c in resolved] == ["405", "488", "561", "640", "BF"]
        assert [c.wavelength_id for c in resolved] == [
            "A00_C01",
            "A03_C02",
            "A02_C03",
            "A01_C04",
            "A00_C05",
        ]

    def test_dedup_covers_unacquired_channels_too(self):
        """Rule 5: a well's names must not depend on what other wells acquired."""
        mes_channels = [
            MesChannel(ch=ch, target=target)
            for ch, target in enumerate(["1", "test", "test", "test", "test"], start=1)
        ]
        resolved = resolve_channels(
            mes_channels=mes_channels,
            acquired=[(3, 2), (2, 3), (1, 4)],
            mrf_channel_count=5,
        )

        assert [c.channel_label for c in resolved] == [
            "1",
            "test",
            "test_1",
            "test_2",
            "test_3",
        ]
        # What the acquired subset {2, 3, 4} prunes down to.
        assert [resolved[ch - 1].channel_label for ch in (2, 3, 4)] == [
            "test",
            "test_1",
            "test_2",
        ]

    def test_sparse_channel_numbers(self):
        """A gap in the `.mes` `bts:Ch` numbering yields placeholder slots."""
        mes_channels = [MesChannel(ch=ch, target=f"t{ch}") for ch in (1, 2, 5)]
        resolved = resolve_channels(
            mes_channels=mes_channels, acquired=[(1, 1)], mrf_channel_count=3
        )

        assert [c.channel_label for c in resolved] == [
            "t1",
            "t2",
            "A00_C03",
            "A00_C04",
            "t5",
        ]

    def test_duplicate_ch_keeps_the_first(self, caplog):
        with caplog.at_level("WARNING"):
            resolved = resolve_channels(
                mes_channels=[
                    MesChannel(ch=1, target="first"),
                    MesChannel(ch=1, target="second"),
                ],
                acquired=[(1, 1)],
                mrf_channel_count=1,
            )
        assert [c.channel_label for c in resolved] == ["first"]
        assert "more than once" in caplog.text

    def test_mrf_channel_count_widens_the_range(self):
        resolved = resolve_channels(
            mes_channels=None, acquired=[(1, 1)], mrf_channel_count=3
        )
        assert len(resolved) == 3

    def test_no_channels_anywhere_raises(self):
        with pytest.raises(ValueError, match="number of channels"):
            resolve_channels(mes_channels=None, acquired=[], mrf_channel_count=0)

    def test_timeline_color_is_never_read(self, tmp_path: Path):
        """`bts:Color` also lives on `<bts:Timeline>`; it must not leak through."""
        body = (
            "  <bts:Timelapse>\n"
            '    <bts:Timeline bts:Color="#ff44ee66" />\n'
            "  </bts:Timelapse>\n"
            + _channel_list('<bts:Channel bts:Ch="1" bts:Target="405" />')
        )
        url = _write_mes(tmp_path, body)
        channels = read_mes_channels(
            acquisition_dir=str(tmp_path), mes_file_name=Path(url).name
        )
        resolved = resolve_channels(
            mes_channels=channels, acquired=[(1, 1)], mrf_channel_count=1
        )
        assert resolved[0].color != "#44ee66"

    def test_reads_through_fsspec(self, tmp_path: Path, monkeypatch):
        """Reads go through `filesystem_for_url`, so `s3://` inputs keep working.

        The library only resolves local and `s3://` URLs, so this asserts the
        call rather than exercising a second backend.
        """
        from fractal_uzh_converters.common import _yokogawa

        urls = []
        original = _yokogawa.filesystem_for_url

        def _spy(url, *args, **kwargs):
            urls.append(url)
            return original(url, *args, **kwargs)

        monkeypatch.setattr(_yokogawa, "filesystem_for_url", _spy)

        url = _write_mes(
            tmp_path,
            _channel_list(
                '<bts:Channel bts:Ch="1" bts:Target="405" bts:Color="#FF002FFF" />'
            ),
        )
        channels = read_mes_channels(
            acquisition_dir=str(tmp_path), mes_file_name=Path(url).name
        )

        assert urls == [url]
        assert channels == [MesChannel(ch=1, target="405", color="#FF002FFF")]


# ---------------------------------------------------------------------------
# Rule 4 — user overrides
# ---------------------------------------------------------------------------


def _resolved(n: int = 5) -> list[ChannelInfo]:
    return resolve_channels(
        mes_channels=[
            MesChannel(ch=ch, target=f"t{ch}", color="#FF0000FF")
            for ch in range(1, n + 1)
        ],
        acquired=[(1, 2), (2, 4)],
        mrf_channel_count=n,
    )


class TestApplyChannelOverrides:
    """`advanced.channels` maps positionally onto the instrument range."""

    def test_no_override_is_the_identity(self):
        resolved = _resolved()
        assert (
            apply_channel_overrides(
                resolved=resolved, overrides=None, max_acquired_ch=4
            )
            == resolved
        )
        assert (
            apply_channel_overrides(resolved=resolved, overrides=[], max_acquired_ch=4)
            == resolved
        )

    def test_too_short_raises_a_converter_level_error(self):
        """The generic `Tile` validator's message is misleading here."""
        overrides = [ChannelInfoUI(channel_label=f"c{i}") for i in range(2)]
        with pytest.raises(ValueError, match="at least 4"):
            apply_channel_overrides(
                resolved=_resolved(), overrides=overrides, max_acquired_ch=4
            )

    def test_short_list_is_padded_from_the_resolved_channels(self):
        """Four entries on a five-channel instrument: element 0 is Ch1."""
        overrides = [ChannelInfoUI(channel_label=f"c{i}") for i in range(4)]
        merged = apply_channel_overrides(
            resolved=_resolved(), overrides=overrides, max_acquired_ch=4
        )

        assert [c.channel_label for c in merged] == ["c0", "c1", "c2", "c3", "t5"]

    def test_extra_entries_are_discarded(self):
        overrides = [ChannelInfoUI(channel_label=f"c{i}") for i in range(7)]
        merged = apply_channel_overrides(
            resolved=_resolved(), overrides=overrides, max_acquired_ch=4
        )
        assert len(merged) == 5

    def test_unset_wavelength_id_is_reinjected(self):
        overrides = [ChannelInfoUI(channel_label=f"c{i}") for i in range(4)]
        merged = apply_channel_overrides(
            resolved=_resolved(), overrides=overrides, max_acquired_ch=4
        )
        assert [c.wavelength_id for c in merged[:4]] == [
            "A00_C01",
            "A01_C02",
            "A00_C03",
            "A02_C04",
        ]

    def test_user_wavelength_id_wins(self):
        overrides = [ChannelInfoUI(channel_label="c0", wavelength_id="custom")]
        overrides += [ChannelInfoUI(channel_label=f"c{i}") for i in range(1, 4)]
        merged = apply_channel_overrides(
            resolved=_resolved(), overrides=overrides, max_acquired_ch=4
        )
        assert merged[0].wavelength_id == "custom"

    def test_auto_color_keeps_the_mes_color(self):
        """`ColorMenu.Auto` means "no preference", not "discard the protocol"."""
        overrides = [ChannelInfoUI(channel_label=f"c{i}") for i in range(4)]
        merged = apply_channel_overrides(
            resolved=_resolved(), overrides=overrides, max_acquired_ch=4
        )
        assert [c.color for c in merged] == ["#0000FF"] * 5

    def test_explicit_color_wins(self):
        # `ColorMenu` members are display strings, as picked in the Fractal UI.
        overrides = [ChannelInfoUI(channel_label="c0", color="Red (#FF0000)")]
        overrides += [ChannelInfoUI(channel_label=f"c{i}") for i in range(1, 4)]
        merged = apply_channel_overrides(
            resolved=_resolved(), overrides=overrides, max_acquired_ch=4
        )
        assert merged[0].color != "#0000FF"
        assert merged[1].color == "#0000FF"

    def test_duplicate_labels_warn_but_are_not_renamed(self, caplog):
        """Renaming a label the user typed on purpose would be worse."""
        overrides = [ChannelInfoUI(channel_label="same") for _ in range(4)]
        with caplog.at_level("WARNING"):
            merged = apply_channel_overrides(
                resolved=_resolved(), overrides=overrides, max_acquired_ch=4
            )

        assert [c.channel_label for c in merged[:4]] == ["same"] * 4
        assert "duplicate channel labels" in caplog.text


# ---------------------------------------------------------------------------
# Extended: the real `.mes` files
# ---------------------------------------------------------------------------


def _resolve_acquisition(kind: str, acquisition_dir: Path) -> list[ChannelInfo]:
    """Resolve one acquisition's channels straight from its own metadata."""
    if kind == "cq3k":
        from fractal_uzh_converters.cq3k._utils import _load_models
    else:
        from fractal_uzh_converters.cellvoyager._utils import _load_models

    mlf, mrf = _load_models(str(acquisition_dir))
    records = mlf.measurement_record or []
    if not isinstance(records, list):
        records = [records]
    images = [record for record in records if record.type == "IMG"]

    mrf_channels = mrf.measurement_channel
    if not isinstance(mrf_channels, list):
        mrf_channels = [mrf_channels]

    return resolve_channels(
        mes_channels=read_mes_channels(
            acquisition_dir=str(acquisition_dir),
            mes_file_name=mrf.measurement_setting_file_name,
        ),
        acquired=[(record.action_index, record.ch) for record in images],
        mrf_channel_count=len(mrf_channels),
    )


# `20251201T140949_BVC_TS_Test_SP_grid` is excluded: its single-record `.mlf`
# cannot be parsed at all today. That is a separate, already-identified defect.
_CQ3K_ACQUISITIONS = [
    "20251201T133446_Channel_Well_test",
    "20251201T134617_Channel_WellTestC5F9_MIP_SUM",
    "20251201T134728_Channel_WellTestC5F9_MIP_SUM_Slice",
    "20251201T134840_Channel_WellTestC5F9_MIP_Only",
    "20251201T134956_Channel_WellTestC5F9_MIP_Slice",
    "20251201T135103_Channel_WellTestC5F9_MIP_Slice_2bin",
    "20251201T135346_Channel_WellTestC5H12M19_MIP",
    "20251201T140917_BVC_TS_Test_FP",
    "20251201T141350_BVC_TS_Test_SP_centered",
    "20260617T083657_2026-06-17_mNgn3_dox_fractal_test",
]

_CELLVOYAGER_ACQUISITIONS = [
    "2ch-sim",
    "no-sim-ch",
    "partial-tile_duplicate-targets",
    "partial-tile_unique-targets",
    "same-target",
    "sf-test_071026_133540",
    "sf-test_071026_133615",
    "time-lines-ill-qc",
    "time-lines-test",
]

_ALL_ACQUISITIONS = [("cq3k", name) for name in _CQ3K_ACQUISITIONS] + [
    ("cellvoyager", name) for name in _CELLVOYAGER_ACQUISITIONS
]


@pytest.mark.extended
@pytest.mark.parametrize("kind, name", _ALL_ACQUISITIONS)
def test_resolution_invariants_hold_on_every_acquisition(kind, name):
    """Labels and wavelength ids are unique and colours are well formed."""
    raw_dir = CQ3K_EXTENDED_RAW_DIR if kind == "cq3k" else CELLVOYAGER_EXTENDED_RAW_DIR
    resolved = _resolve_acquisition(kind, raw_dir / name)

    labels = [channel.channel_label for channel in resolved]
    wavelength_ids = [channel.wavelength_id for channel in resolved]

    assert resolved
    assert len(set(labels)) == len(labels)
    assert len(set(wavelength_ids)) == len(wavelength_ids)
    for channel in resolved:
        assert channel.color is not None
        assert len(channel.color) in (4, 7)


@pytest.mark.extended
@pytest.mark.parametrize(
    "kind, name, labels, wavelength_ids",
    [
        (
            "cellvoyager",
            "2ch-sim",
            ["405", "488", "561", "640", "BF"],
            # Ch1 and Ch4 share action 1: one action, two channels at once.
            ["A01_C01", "A03_C02", "A02_C03", "A01_C04", "A00_C05"],
        ),
        (
            "cellvoyager",
            "same-target",
            ["1", "test", "test_1", "test_2", "test_3"],
            ["A00_C01", "A03_C02", "A02_C03", "A01_C04", "A00_C05"],
        ),
        (
            "cellvoyager",
            "time-lines-test",
            ["405", "488", "561", "640", "BF", "488-2"],
            ["A01_C01", "A01_C02", "A00_C03", "A01_C04", "A00_C05", "A02_C06"],
        ),
        (
            "cellvoyager",
            "no-sim-ch",
            ["405", "488", "561", "640", "BF"],
            ["A00_C01", "A03_C02", "A02_C03", "A01_C04", "A00_C05"],
        ),
        (
            "cq3k",
            "20260617T083657_2026-06-17_mNgn3_dox_fractal_test",
            [
                "Ch1_BrightField(Confocal Path)_Lamp_Through",
                "Ch2_EpiFluorescence_561nm/150mW_BP599/44",
                "Ch3_BrightField(Confocal Path)_Lamp_Through",
                "Ch4_EpiFluorescence_561nm/150mW_BP599/44",
            ],
            ["A01_C01", "A03_C02", "A02_C03", "A04_C04"],
        ),
    ],
)
def test_resolved_channels_match_the_protocol(kind, name, labels, wavelength_ids):
    """Exact expectations, so a regression cannot hide in a snapshot diff."""
    raw_dir = CQ3K_EXTENDED_RAW_DIR if kind == "cq3k" else CELLVOYAGER_EXTENDED_RAW_DIR
    resolved = _resolve_acquisition(kind, raw_dir / name)

    assert [channel.channel_label for channel in resolved] == labels
    assert [channel.wavelength_id for channel in resolved] == wavelength_ids


@pytest.mark.extended
def test_cq3k_colors_come_from_the_protocol():
    """CQ3K `bts:Color` values survive as `#RRGGBB`, alpha stripped."""
    resolved = _resolve_acquisition(
        "cq3k", CQ3K_EXTENDED_RAW_DIR / "20251201T133446_Channel_Well_test"
    )

    assert [channel.color for channel in resolved] == [
        "#FFFFFF",
        "#00FFFF",
        "#19FF06",
        "#FF8001",
        "#ED0900",
    ]


@pytest.mark.extended
@pytest.mark.parametrize(
    "well_channels, expected",
    [
        # `time-lines-test` acquires a different channel set per well; each still
        # gets the labels its channel numbers point at.
        ((4,), ["640"]),
        ((1, 6), ["405", "488-2"]),
        ((2, 6), ["488", "488-2"]),
    ],
)
def test_per_well_channel_subsets(well_channels, expected):
    """What `reindex_channels` will prune each well down to at compute time."""
    resolved = _resolve_acquisition(
        "cellvoyager", CELLVOYAGER_EXTENDED_RAW_DIR / "time-lines-test"
    )
    assert [resolved[ch - 1].channel_label for ch in well_channels] == expected


@pytest.mark.extended
def test_same_mes_basename_resolves_per_acquisition():
    """Both `partial-tile_*` acquisitions name the same `.mes`, with different content.

    This is what a `*.mes` glob would get wrong.
    """
    duplicate = _resolve_acquisition(
        "cellvoyager", CELLVOYAGER_EXTENDED_RAW_DIR / "partial-tile_duplicate-targets"
    )
    unique = _resolve_acquisition(
        "cellvoyager", CELLVOYAGER_EXTENDED_RAW_DIR / "partial-tile_unique-targets"
    )

    assert [c.channel_label for c in duplicate] == [
        "1",
        "test",
        "test_1",
        "test_2",
        "test_3",
    ]
    assert [c.channel_label for c in unique] == ["405", "488", "561", "640", "BF"]
