"""Channel metadata for the Yokogawa BTS schema, shared by both instruments.

Covers the `.mes` channel list and the naming rules that turn it into `ChannelInfo`,
the user-facing `advanced.channels` override models keyed by `bts:Ch`, the `.mrf`
channel-geometry check and the `.mlf` time-index normalisation. The `.mlf`/`.mrf`
models live in `_records.py`, the parser in `_parse.py`.

The `.mes` carries the human-readable channel targets the `.mlf`/`.mrf` pair lacks.
"""

import logging
import warnings
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
from ome_zarr_converters_tools import (
    AcquisitionDetails,
    AcquisitionOptions,
    ChannelInfo,
    UserFacingModel,
    join_url_paths,
)

# `ColorMenu` is the enum behind `ChannelInfoUI.color`; the library does not
# re-export it, and `YokogawaChannelInfoUI` must declare the same field.
from ome_zarr_converters_tools.fractal._models import ColorMenu
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_pascal

from fractal_uzh_converters.common import (
    ChannelMetadataWarning,
    GeometryWarning,
    clean_channel_string,
)
from fractal_uzh_converters.yokogawa._xml import parse_bts_xml

logger = logging.getLogger(__name__)

######################################################################
#
# Pydantic models for parsing the `.mes` MeasurementSetting
#
######################################################################


class _MesBase(BaseModel):
    """Base model for `.mes` elements.

    `extra="ignore"`: `<bts:Channel>` carries a different attribute set per
    instrument and firmware. `populate_by_name` lets these be built from Python
    too, not only from the PascalCase keys `xmltodict` produces.
    """

    model_config = ConfigDict(
        alias_generator=to_pascal,
        extra="ignore",
        populate_by_name=True,
    )


class MesChannel(_MesBase):
    """One `<bts:Channel>` entry of a `.mes` `<bts:ChannelList>`."""

    ch: int
    target: str | None = None
    color: str | None = None


class MesChannelList(_MesBase):
    """`<bts:ChannelList>`.

    `xmltodict` collapses a single `<bts:Channel>` to a bare dict rather than a
    one-element list, hence the union.
    """

    channel: list[MesChannel] | MesChannel = Field(default_factory=list)


class MeasurementSetting(_MesBase):
    """Root `<bts:MeasurementSetting>` of a `.mes` file.

    Only the channel list is modelled. Leaving `<bts:Timelapse>` out also makes it
    impossible to mistake a `<bts:Timeline>`'s `bts:Color` for a channel colour.
    `bts:Version` is not pinned here as it is on the `.mlf`/`.mrf` roots: some
    CV8000 `.mes` files carry no such attribute.
    """

    channel_list: MesChannelList | None = None


######################################################################
#
# XML parsing helpers
#
######################################################################


def parse_mes(mes_url: str) -> MeasurementSetting:
    """Parse the `.mes` MeasurementSetting document at `mes_url`.

    Args:
        mes_url: URL of the `.mes` file.

    Returns:
        The parsed MeasurementSetting.

    Raises:
        FileNotFoundError: If `mes_url` does not exist.
        Exception: If the document is malformed or does not match the schema.
    """
    try:
        parsed = parse_bts_xml(mes_url)
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error parsing XML file {mes_url}: {e}")
        raise
    return MeasurementSetting(**parsed["MeasurementSetting"])


def read_mes_channels(
    *, acquisition_dir: str, mes_file_name: str | None
) -> list[MesChannel] | None:
    """Read the `.mes` channel list of an acquisition, sorted by `bts:Ch`.

    The `.mes` basename varies per acquisition, so it comes from the `.mrf` rather
    than from a `*.mes` glob: sibling acquisitions can share a directory, and two
    can carry the same basename with different content.

    Args:
        acquisition_dir: URL of the acquisition directory.
        mes_file_name: The `.mrf` `MeasurementSettingFileName` value.

    Returns:
        The `<bts:Channel>` entries sorted by `bts:Ch`, or `None` when the `.mes` is
        unnamed, absent, or carries no channel list. A missing file is normal — a
        `.mrf` routinely names a `.mes` that was never shipped with it.
    """
    if not mes_file_name:
        warnings.warn(
            "No `.mes` file name recorded in the `.mrf`; channel labels will fall "
            "back to their wavelength ids.",
            ChannelMetadataWarning,
            stacklevel=2,
        )
        return None

    mes_url = join_url_paths(acquisition_dir, mes_file_name)
    try:
        mes = parse_mes(mes_url)
    except FileNotFoundError:
        warnings.warn(
            f"`.mes` file {mes_url} is named in the `.mrf` but does not exist; "
            "channel labels will fall back to their wavelength ids.",
            ChannelMetadataWarning,
            stacklevel=2,
        )
        return None

    if mes.channel_list is None:
        warnings.warn(
            f"`.mes` file {mes_url} has no channel list.",
            ChannelMetadataWarning,
            stacklevel=2,
        )
        return None

    channels = mes.channel_list.channel
    if isinstance(channels, MesChannel):
        channels = [channels]
    if not channels:
        warnings.warn(
            f"`.mes` file {mes_url} has an empty channel list.",
            ChannelMetadataWarning,
            stacklevel=2,
        )
        return None
    return sorted(channels, key=lambda channel: channel.ch)


######################################################################
#
# Channel resolution
#
######################################################################


def _wavelength_id(action_index: int, ch: int) -> str:
    """Format a wavelength id as `A{action:02d}_C{ch:02d}`.

    `action_index=0` marks a channel the instrument offers but the acquisition
    never used; there is no action to name for it.
    """
    return f"A{action_index:02d}_C{ch:02d}"


def _actions_by_channel(acquired: list[tuple[int, int]]) -> dict[int, int]:
    """Map each acquired `bts:Ch` to its `bts:ActionIndex`.

    One action can carry several channels, but no known data has a channel under
    two actions. If one ever does, take the lowest and warn.
    """
    actions: dict[int, set[int]] = {}
    for action_index, ch in acquired:
        actions.setdefault(ch, set()).add(action_index)

    resolved = {}
    for ch, ch_actions in actions.items():
        if len(ch_actions) > 1:
            warnings.warn(
                f"Channel Ch{ch} is acquired by more than one action "
                f"({sorted(ch_actions)}); using the lowest for its wavelength id.",
                ChannelMetadataWarning,
                stacklevel=2,
            )
        resolved[ch] = min(ch_actions)
    return resolved


def _strip_argb_alpha(color: str | None, *, ch: int) -> str | None:
    """Convert a `.mes` `bts:Color` to a `ChannelInfo`-compatible hex string.

    `.mes` colours are ARGB (`#AARRGGBB`); `ChannelInfo.color` takes 3 or 6 hex
    digits. Anything unrecognised returns `None`, letting `ChannelInfo` pick one.
    """
    if color is None:
        return None

    value = color.strip()
    if not value.startswith("#"):
        warnings.warn(
            f"Ignoring malformed `.mes` color {color!r} for Ch{ch}.",
            ChannelMetadataWarning,
            stacklevel=2,
        )
        return None

    digits = value[1:]
    if not all(c in "0123456789abcdefABCDEF" for c in digits):
        warnings.warn(
            f"Ignoring malformed `.mes` color {color!r} for Ch{ch}.",
            ChannelMetadataWarning,
            stacklevel=2,
        )
        return None

    if len(digits) == 8:  # ARGB
        return f"#{digits[2:]}"
    if len(digits) in (3, 6):
        return f"#{digits}"

    warnings.warn(
        f"Ignoring malformed `.mes` color {color!r} for Ch{ch}.",
        ChannelMetadataWarning,
        stacklevel=2,
    )
    return None


def _dedup_labels(labels: list[str]) -> list[str]:
    """Suffix duplicate labels as `label`, `label_1`, `label_2`, ….

    Must run over the whole instrument channel list, in `bts:Ch` order, *before*
    any per-image pruning — otherwise a well's channel names would depend on which
    other wells were acquired.
    """
    taken = set(labels)
    seen: set[str] = set()
    deduped = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            deduped.append(label)
            continue
        index = 1
        while f"{label}_{index}" in taken or f"{label}_{index}" in seen:
            index += 1
        candidate = f"{label}_{index}"
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def max_acquired_channel(acquired: list[tuple[int, int]]) -> int:
    """Return the highest acquired `bts:Ch`, or 0 when nothing was acquired."""
    return max((ch for _, ch in acquired), default=0)


def resolve_channels(
    *,
    mes_channels: list[MesChannel] | None,
    acquired: list[tuple[int, int]],
    mrf_channel_count: int,
) -> list[ChannelInfo]:
    """Build the channel metadata for a whole acquisition.

    The returned list spans the **full instrument channel range**: element `i` is
    `bts:Ch == i + 1`, matching the converters' `start_c = ch - 1`. Never-acquired
    slots get a placeholder and are pruned per image at compute time by the
    library's `reindex_channels`.

    Call once per acquisition, not per plate — the projection plates share one
    `.mes` and a channel must keep the same label across all of them.

    Args:
        mes_channels: The `.mes` channel list, or `None` when there is no `.mes`.
        acquired: `(action_index, ch)` pairs from the `.mlf` image records.
            Duplicates are expected; order is irrelevant.
        mrf_channel_count: Number of `<bts:MeasurementChannel>` entries in the `.mrf`.

    Returns:
        One `ChannelInfo` per instrument channel slot, in `bts:Ch` order.

    Raises:
        ValueError: If no source declares a single channel.
    """
    actions = _actions_by_channel(acquired)

    by_ch: dict[int, MesChannel] = {}
    for channel in mes_channels or []:
        if channel.ch in by_ch:
            warnings.warn(
                f"`.mes` channel list declares Ch{channel.ch} more than once; "
                "keeping the first entry.",
                ChannelMetadataWarning,
                stacklevel=2,
            )
            continue
        by_ch[channel.ch] = channel

    n_channels = max(
        len(by_ch),
        max(by_ch, default=0),
        mrf_channel_count,
        max_acquired_channel(acquired),
    )
    if n_channels < 1:
        raise ValueError(
            "Could not determine the number of channels: the `.mes`, the `.mrf` "
            "and the `.mlf` records all declare none."
        )

    wavelength_ids = [
        _wavelength_id(actions.get(ch, 0), ch) for ch in range(1, n_channels + 1)
    ]
    raw_labels = []
    colors = []
    for ch, wavelength_id in enumerate(wavelength_ids, start=1):
        channel = by_ch.get(ch)
        target = clean_channel_string(channel.target) if channel is not None else None
        raw_labels.append(target or wavelength_id)
        colors.append(
            _strip_argb_alpha(channel.color, ch=ch) if channel is not None else None
        )

    return [
        ChannelInfo(channel_label=label, wavelength_id=wavelength_id, color=color)
        for label, wavelength_id, color in zip(
            _dedup_labels(raw_labels), wavelength_ids, colors, strict=True
        )
    ]


######################################################################
#
# User-facing override models
#
######################################################################


class YokogawaChannelInfoUI(UserFacingModel):
    """Override one instrument channel, identified by its `bts:Ch` number.

    Not a subclass of the library's `ChannelInfoUI`: Pydantic keeps inherited
    fields first, and `key` must render first in the Fractal form. The three
    fields below are copied from `ChannelInfoUI` verbatim.
    """

    key: int = Field(ge=1, title="Channel Key")
    """
    Number of the instrument channel this entry applies to, counted from 1.
    To find it, look at the `bts:Ch` attribute next to each channel name in the
    acquisition's `.mes` file, or at the `bts:Ch` attribute of each image record in
    the `.mlf` file that lists all the TIFFs, or at the `C` suffix of the TIFF
    file names (`...C1.tif` or `...C01.tif` is channel 1).
    """
    channel_label: str
    """Name of the channel, e.g. `DAPI` or `GFP`."""
    wavelength_id: str | None = Field(default=None, title="Wavelength ID")
    """
    The wavelength ID of the channel.
    Some tasks can use it instead of the channel name, e.g. to apply
    illumination correction per wavelength in multiplexed acquisitions.
    """
    color: ColorMenu = ColorMenu.Auto
    """Display color of the channel, e.g. for visualization purposes."""


class YokogawaAcquisitionOptions(AcquisitionOptions):
    """Acquisition options whose channel overrides are keyed by `bts:Ch`."""

    channels: list[YokogawaChannelInfoUI] | None = None
    """
    Override the channel metadata read from the `.mes` file. List only the
    channels you want to change, in any order, each named by `Channel Key`;
    every other channel keeps the label, wavelength id and color parsed from
    the `.mes`. Inside an entry, a blank label or wavelength id and the `Auto`
    color also keep the `.mes` value. If the acquisition has no `.mes`, the
    defaults are the wavelength ids `A{action}_C{channel}`, so this is where to
    give such channels readable names.
    """

    @field_validator("channels")
    @classmethod
    def _unique_channel_numbers(
        cls, channels: list[YokogawaChannelInfoUI] | None
    ) -> list[YokogawaChannelInfoUI] | None:
        """Reject a `key` listed twice: the entries would silently overwrite."""
        if channels is None:
            return channels
        numbers = [channel.key for channel in channels]
        duplicates = sorted({ch for ch in numbers if numbers.count(ch) > 1})
        if duplicates:
            raise ValueError(
                f"`channels` lists bts:Ch {duplicates} more than once; each channel "
                "number may appear at most once."
            )
        return channels

    def update_acquisition_details(
        self,
        acquisition_details: AcquisitionDetails,
    ) -> AcquisitionDetails:
        """Apply every option; `channels` is merged by `key` rather than replaced.

        The base implementation replaces `channels` wholesale (and flattens an
        `Auto` colour to a guess), so the merge starts from the *input* details,
        where the `.mes`-resolved list is still intact.
        """
        updated = super().update_acquisition_details(
            acquisition_details=acquisition_details
        )
        if self.channels is None:
            return updated
        # Not `if not self.channels`: an empty list made the base class blank the
        # channels out, and the merge below restores them.
        resolved = acquisition_details.channels
        if resolved is None:
            raise ValueError(
                "`channels` overrides are matched by channel number, which needs "
                "the channel list resolved from the acquisition metadata."
            )
        updated.channels = apply_channel_overrides(
            resolved=resolved, overrides=self.channels
        )
        return updated


def warn_on_unlisted_acquired_channels(
    *,
    overrides: list[YokogawaChannelInfoUI] | None,
    acquired: list[tuple[int, int]],
) -> None:
    """Warn when `advanced.channels` leaves an acquired channel without an entry.

    A partial list is allowed — that is the point of keyed entries — but a user
    who meant to name every channel and forgot one would otherwise only notice
    in the converted plate, where a `.mes` target or an `A01_C03` placeholder
    sits next to their own labels. Channels the acquisition never used are not
    reported: they are pruned from the output anyway.

    Call once per acquisition, with the acquired pairs of the kept plates only,
    so a plate deselected through `z_processing` does not count.

    Args:
        overrides: The user's `advanced.channels`, or `None`.
        acquired: `(action_index, ch)` pairs from the `.mlf` image records.
    """
    if not overrides:
        return
    listed = {override.key for override in overrides}
    unlisted = sorted({ch for _, ch in acquired} - listed)
    if unlisted:
        names = ", ".join(f"Ch{ch}" for ch in unlisted)
        warnings.warn(
            f"`advanced.channels` has no entry for acquired channel(s) {names}; "
            "they keep the metadata read from the acquisition.",
            ChannelMetadataWarning,
            stacklevel=2,
        )


def apply_channel_overrides(
    *,
    resolved: list[ChannelInfo],
    overrides: list[YokogawaChannelInfoUI] | None,
) -> list[ChannelInfo]:
    """Merge user `advanced.channels` entries onto the resolved channels.

    Each entry's `key` is a `bts:Ch` number and lands on `resolved[key - 1]`;
    slots without an entry are untouched, so the result is always as long as
    `resolved` and `start_c = ch - 1` keeps addressing the right entry. An entry
    for a channel the acquisition never used is accepted: it lands on a slot
    `reindex_channels` prunes at compute time. Duplicate keys are rejected
    upstream by `YokogawaAcquisitionOptions`.

    Args:
        resolved: Output of `resolve_channels` for this acquisition.
        overrides: The user's `advanced.channels`, or `None`.

    Returns:
        The merged channel list, the same length as `resolved`.

    Raises:
        ValueError: If an entry names a channel outside `1..len(resolved)`.
    """
    if not overrides:
        return list(resolved)

    merged = list(resolved)
    for override in overrides:
        if override.key > len(resolved):
            declared = ", ".join(
                f"Ch{ch} {channel.channel_label!r}"
                for ch, channel in enumerate(resolved, start=1)
            )
            raise ValueError(
                f"`advanced.channels` key {override.key} names bts:Ch "
                f"{override.key}, but this acquisition declares "
                f"Ch1..Ch{len(resolved)}: {declared}."
            )
        index = override.key - 1
        fallback = resolved[index]
        # `to_hexstr()` returns None exactly for `ColorMenu.Auto`, i.e. when the
        # user expressed no preference — so the `.mes` colour stands.
        color = override.color.to_hexstr()
        # A label the user left blank (or padded into blankness) keeps the
        # `.mes`-derived one, exactly as an omitted wavelength id does.
        merged[index] = ChannelInfo(
            channel_label=(
                clean_channel_string(override.channel_label) or fallback.channel_label
            ),
            wavelength_id=(
                clean_channel_string(override.wavelength_id) or fallback.wavelength_id
            ),
            color=fallback.color if color is None else color,
        )

    labels = [channel.channel_label for channel in merged]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        warnings.warn(
            f"`advanced.channels` produces duplicate channel labels {duplicates}. "
            "Channel labels must be unique within an image; the conversion will "
            "fail if more than one of them is acquired.",
            ChannelMetadataWarning,
            stacklevel=2,
        )
    return merged


######################################################################
#
# `.mlf` time index
#
######################################################################


class TimeIndex(NamedTuple):
    """The time axis of one output plate, derived from `bts:TimePoint`.

    `bts:TimePoint` counts *timelines*, not frames: three wells acquired by three
    timelines each carry a different `TimePoint` while each holds a single time
    point. Used raw it mislabels those wells as frames 0, 1 and 2, and leaves gaps
    as empty frames — the pipeline has no `reindex_t` to compact them the way
    `reindex_channels` compacts channels.
    """

    is_time_series: bool
    """Whether any well of the plate holds more than one time point.

    Plate-wide rather than per well: `TiledImage.add_tile` requires an image's tiles
    to agree on `axes`.
    """

    dense_indices: dict[tuple[int, int], dict[int, int]]
    """`{(bts:Row, bts:Column): {bts:TimePoint: 0-based time index}}`."""

    def start_t(self, *, row: int, column: int, time_point: int) -> int:
        """The 0-based time index of one `.mlf` image record."""
        return self.dense_indices[(row, column)][time_point]


def build_time_index(records: Iterable[tuple[int, int, int]]) -> TimeIndex:
    """Map each well's `bts:TimePoint` values onto a dense 0-based range.

    Call once per output plate with every image record of that plate: the mapping
    is per well, but `is_time_series` is decided across all of them.

    Args:
        records: `(bts:Row, bts:Column, bts:TimePoint)` of every image record of
            the plate. Duplicates are expected; order is irrelevant.

    Returns:
        The plate's `TimeIndex`.
    """
    per_well: dict[tuple[int, int], set[int]] = {}
    for row, column, time_point in records:
        per_well.setdefault((row, column), set()).add(time_point)

    return TimeIndex(
        is_time_series=any(len(time_points) > 1 for time_points in per_well.values()),
        dense_indices={
            well: {time_point: index for index, time_point in enumerate(sorted(points))}
            for well, points in per_well.items()
        },
    )


######################################################################
#
# `.mrf` channel geometry
#
######################################################################


class ChannelGeometry(NamedTuple):
    """The per-channel geometry fields of one `.mrf` `<bts:MeasurementChannel>`."""

    ch: int
    xy_pixel_size: tuple[float, float]
    frame_size: tuple[int, int]


def warn_on_channel_geometry_mismatch(
    *,
    geometries: list[ChannelGeometry],
    acquisition_dir: str,
) -> None:
    """Warn when the `.mrf` channels disagree on pixel size or frame size.

    Both converters apply the *first* channel's geometry to all of them, which is
    right for all data seen so far but would silently misplace a binned or
    differently-cropped channel. This reports the suspicion; it changes nothing.

    Call once per acquisition, not per plate or per field of view.

    Args:
        geometries: One entry per `.mrf` channel, in file order.
        acquisition_dir: Named in the warning, since a batch is converted one
            acquisition at a time.
    """
    if len(geometries) < 2:
        return

    reference = geometries[0]
    for other in geometries[1:]:
        pixel_size_differs = not np.allclose(
            other.xy_pixel_size, reference.xy_pixel_size
        )
        frame_size_differs = other.frame_size != reference.frame_size
        if pixel_size_differs or frame_size_differs:
            warnings.warn(
                f"In {acquisition_dir}, bts:Ch {other.ch} declares pixel size "
                f"{other.xy_pixel_size} and frame size {other.frame_size}, but "
                f"bts:Ch {reference.ch} declares {reference.xy_pixel_size} and "
                f"{reference.frame_size}. Channel {reference.ch}'s geometry is "
                "applied to every channel, so this image may be misplaced.",
                GeometryWarning,
                stacklevel=2,
            )
