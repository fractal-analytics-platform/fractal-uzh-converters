"""Channel labels and wavelength ids are stripped at every construction site.

ngio validates channel labels only for *uniqueness*
(`ngio_specs/_channels.py`), so `"DAPI"` and `"DAPI "` are two valid, distinct
channels — and a downstream `get_channel_idx(channel_label="DAPI")` then
silently misses the padded one. Nothing in `ome-zarr-converters-tools` or ngio
normalises them, so each converter must.

No reference dataset in either store carries stray whitespace — the ScanR
OME names, the Yokogawa `.mes` targets and the Operetta `ChannelName`s are all
clean — so this behaviour is invisible to the snapshots and is covered here with
hand-built inputs instead.

A string that is *only* whitespace must not become `""`: it falls through to
whatever that converter already does for a nameless channel.
"""

from types import SimpleNamespace

import pytest
from ome_zarr_converters_tools import ChannelInfo, ChannelInfoUI

from fractal_uzh_converters.common import ChannelMetadataWarning, clean_channel_string
from fractal_uzh_converters.custom_tiff._utils import (
    _build_acquisition_details as custom_tiff_details,
)
from fractal_uzh_converters.imagexpress_hcs._utils import (
    MDImageXpressHCSaiAcquisitionModel,
)
from fractal_uzh_converters.imagexpress_hcs._utils import (
    _build_acquisition_details as imagexpress_details,
)
from fractal_uzh_converters.operetta._utils import _channel_names
from fractal_uzh_converters.scanr._utils import _get_channel_names
from fractal_uzh_converters.yokogawa._channels import (
    MesChannel,
    apply_channel_overrides,
    resolve_channels,
)

# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------


class TestCleanChannelString:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("DAPI ", "DAPI"),
            (" DAPI", "DAPI"),
            ("\tDAPI\n", "DAPI"),
            ("DAPI", "DAPI"),
            # Internal whitespace is part of the vendor's identifier.
            ("GFP 488", "GFP 488"),
            ("", None),
            ("   ", None),
            ("\t\n ", None),
            (None, None),
        ],
    )
    def test_strips_only(self, raw, expected):
        assert clean_channel_string(raw) == expected


# ---------------------------------------------------------------------------
# Yokogawa — the `.mes` target and the user's `advanced.channels`
# ---------------------------------------------------------------------------


class TestYokogawa:
    def test_mes_target_is_stripped(self):
        resolved = resolve_channels(
            mes_channels=[MesChannel(ch=1, target=" DAPI ", color=None)],
            acquired=[(1, 1)],
            mrf_channel_count=1,
        )
        assert [c.channel_label for c in resolved] == ["DAPI"]

    def test_blank_mes_target_falls_back_to_wavelength_id(self):
        resolved = resolve_channels(
            mes_channels=[MesChannel(ch=1, target="   ", color=None)],
            acquired=[(1, 1)],
            mrf_channel_count=1,
        )
        assert [c.channel_label for c in resolved] == ["A01_C01"]

    def test_override_label_and_wavelength_id_are_stripped(self):
        """The one Yokogawa path where an untrimmed label reaches the output."""
        resolved = resolve_channels(
            mes_channels=[MesChannel(ch=1, target="405", color=None)],
            acquired=[(1, 1)],
            mrf_channel_count=1,
        )
        merged = apply_channel_overrides(
            resolved=resolved,
            overrides=[
                ChannelInfoUI(channel_label=" DAPI ", wavelength_id=" A01_C01 ")
            ],
            max_acquired_ch=1,
        )
        assert merged[0].channel_label == "DAPI"
        assert merged[0].wavelength_id == "A01_C01"

    def test_blank_override_keeps_the_mes_label(self):
        resolved = resolve_channels(
            mes_channels=[MesChannel(ch=1, target="405", color=None)],
            acquired=[(1, 1)],
            mrf_channel_count=1,
        )
        merged = apply_channel_overrides(
            resolved=resolved,
            overrides=[ChannelInfoUI(channel_label="  ", wavelength_id="  ")],
            max_acquired_ch=1,
        )
        assert merged[0].channel_label == "405"
        assert merged[0].wavelength_id == "A01_C01"

    def test_padded_duplicates_are_reported(self):
        """`"DAPI"` vs `"DAPI "` is a duplicate the unstripped check missed."""
        resolved = resolve_channels(
            mes_channels=[
                MesChannel(ch=1, target="405", color=None),
                MesChannel(ch=2, target="488", color=None),
            ],
            acquired=[(1, 1), (1, 2)],
            mrf_channel_count=2,
        )
        with pytest.warns(ChannelMetadataWarning, match="duplicate channel labels"):
            merged = apply_channel_overrides(
                resolved=resolved,
                overrides=[
                    ChannelInfoUI(channel_label="DAPI"),
                    ChannelInfoUI(channel_label="DAPI "),
                ],
                max_acquired_ch=2,
            )

        assert [c.channel_label for c in merged] == ["DAPI", "DAPI"]


# ---------------------------------------------------------------------------
# Operetta — `ChannelName` from the XML
# ---------------------------------------------------------------------------


def _operetta_image(channel_id: int, channel_name: str) -> SimpleNamespace:
    """The two attributes `_channel_names` reads."""
    return SimpleNamespace(channel_id=channel_id, channel_name=channel_name)


class TestOperetta:
    def test_channel_name_is_stripped(self):
        names = _channel_names([_operetta_image(1, " HOECHST 33342 ")])
        assert names == ["HOECHST 33342"]

    def test_blank_channel_name_becomes_none(self):
        """The caller then falls back to the wavelength id."""
        assert _channel_names([_operetta_image(1, "   ")]) == [None]


# ---------------------------------------------------------------------------
# ScanR — `Channel/@Name` from the OME-XML
# ---------------------------------------------------------------------------


def _scanr_image(*names: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        pixels=SimpleNamespace(channels=[SimpleNamespace(name=n) for n in names])
    )


class TestScanR:
    def test_names_are_stripped(self):
        assert _get_channel_names(_scanr_image(" DAPI ", "GFP\t")) == ["DAPI", "GFP"]

    def test_blank_name_drops_the_list(self):
        """Same fallback a missing name already had: the library names them."""
        assert _get_channel_names(_scanr_image("DAPI", "  ")) is None


# ---------------------------------------------------------------------------
# ImageXpress — `EmissionFilter.Name` from the MD JSON
# ---------------------------------------------------------------------------


def _imagexpress_meta(*names: str) -> SimpleNamespace:
    """The attributes `_build_acquisition_details` reads off the experiment."""
    return SimpleNamespace(
        channels=[
            SimpleNamespace(
                index=index,
                emission_filter=SimpleNamespace(name=name, wavelength=447.0 + index),
            )
            for index, name in enumerate(names)
        ],
        pixel_size_x=0.325,
        z_step_um=1.0,
        is_time_series=False,
    )


_IMAGEXPRESS_MODEL = MDImageXpressHCSaiAcquisitionModel(path="unused")


class TestImageXpress:
    def test_filter_name_is_stripped(self):
        details = imagexpress_details(_imagexpress_meta(" DAPI "), _IMAGEXPRESS_MODEL)
        assert [c.channel_label for c in details.channels] == ["DAPI"]

    def test_blank_filter_name_falls_back_to_wavelength_id(self):
        details = imagexpress_details(_imagexpress_meta("   "), _IMAGEXPRESS_MODEL)
        assert [c.channel_label for c in details.channels] == ["447"]


# ---------------------------------------------------------------------------
# Custom TIFF — the user's own `acquisition_details.toml`
# ---------------------------------------------------------------------------


class TestCustomTiff:
    def test_names_and_wavelengths_are_stripped(self):
        details = custom_tiff_details(
            {
                "channel_names": [" DAPI ", "GFP"],
                "wavelengths": ["405 ", " 488"],
                "xy_pixel_size": 0.325,
            }
        )
        assert [c.channel_label for c in details.channels] == ["DAPI", "GFP"]
        assert [c.wavelength_id for c in details.channels] == ["405", "488"]

    def test_blank_name_drops_the_list(self):
        """Rather than leaving one channel of the user's plate nameless."""
        with pytest.warns(ChannelMetadataWarning, match="blank entry"):
            details = custom_tiff_details(
                {"channel_names": ["DAPI", "  "], "xy_pixel_size": 0.325}
            )

        assert details.channels is None


# ---------------------------------------------------------------------------
# The property that motivates all of the above
# ---------------------------------------------------------------------------


def test_padded_label_is_a_distinct_channel_without_stripping():
    """Why this matters: ngio would accept both as separate valid channels."""
    assert ChannelInfo(channel_label="DAPI ").channel_label != "DAPI"
