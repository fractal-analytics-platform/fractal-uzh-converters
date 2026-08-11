"""Channel metadata for the Yokogawa BTS schema.

Both Yokogawa instruments (`cellvoyager/` for CV7k/CV8k, `cq3k/` for
CQ3000/CQ3K) read the same `.mlf`/`.mrf`/`.mes` XML schema. This module covers
the `.mes` (MeasurementProtocol) channel list, the channel-naming rules that
turn it into `ChannelInfo` metadata, the `.mrf` channel-geometry consistency
check and the `.mlf` time-index normalisation; the `.mlf`/`.mrf` models
themselves live in `_records.py` and the parser in `_parse.py`.

The `.mes` file carries the acquisition protocol, including the human-readable
channel targets that the `.mlf`/`.mrf` pair lacks. Without it every output is
labelled `channel_0…channel_N`.
"""

import logging
import warnings
from collections.abc import Iterable
from typing import Any, NamedTuple

import numpy as np
import xmltodict
from ome_zarr_converters_tools import (
    ChannelInfo,
    ChannelInfoUI,
    filesystem_for_url,
    join_url_paths,
)
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_pascal

from fractal_uzh_converters.common import (
    ChannelMetadataWarning,
    GeometryWarning,
    clean_channel_string,
)

logger = logging.getLogger(__name__)

BTS_NAMESPACE = "http://www.yokogawa.co.jp/BTS/BTSSchema/1.0"

######################################################################
#
# Pydantic models for parsing the `.mes` MeasurementSetting
#
######################################################################


class _MesBase(BaseModel):
    """Base model for `.mes` elements.

    Unlike the `.mlf`/`.mrf` models this uses `extra="ignore"`: `<bts:Channel>`
    carries 16 attributes on CQ3000 and 20 on CV8000 (`CSUID`, `PinholeDiameter`,
    `InputLevel`, `Fluorophore`), in varying order, and firmware updates add more.

    `populate_by_name` lets these be built from Python too, not only from the
    PascalCase keys `xmltodict` produces.
    """

    model_config = ConfigDict(
        alias_generator=to_pascal,
        extra="ignore",
        populate_by_name=True,
    )


class MesChannel(_MesBase):
    """One `<bts:Channel>` entry of a `.mes` `<bts:ChannelList>`.

    `<bts:Channel>` also holds one or more `<bts:LightSourceName>` *child*
    elements. `xmltodict` is called with `attr_prefix=""`, so those land in the
    same flat dict as the attributes, as a `str` or a `list[str]` depending on
    count. Not modelling them keeps that polymorphism out of the way.
    """

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

    Only the channel list is modelled. Dropping `<bts:Timelapse>` is not just
    economy: its `<bts:Timeline>` elements carry a `bts:Color` attribute of their
    own, and not modelling them makes it structurally impossible to mistake a
    timeline colour for a channel colour.

    `bts:Version` is deliberately absent from this model — unlike the `.mlf`/`.mrf`
    roots, which pin `version: Literal["1.0"]`. The CV8000 acquisition
    `hcs_2w20p3c9z1t_SearchFirst` ships a `.mes` with no `bts:Version` attribute at all.
    """

    channel_list: MesChannelList | None = None


######################################################################
#
# XML parsing helpers
#
######################################################################


# NOTE: this mirrors the `_parse` helpers in `cq3k/_utils.py` and
# `cellvoyager/_utils.py`. It is kept separate on purpose: reading a `.mes` must
# distinguish "not there" (a normal input — see `read_mes_channels`) from a real
# parse failure, whereas those two log an error and re-raise either way.
def _parse_bts_xml(url: str) -> dict[str, Any]:
    """Parse a Yokogawa BTS XML document, stripping the `bts:` namespace."""
    fs = filesystem_for_url(url)
    with fs.open(url, encoding="utf-8") as f:
        return xmltodict.parse(
            f.read(),
            process_namespaces=True,
            namespaces={BTS_NAMESPACE: None},
            attr_prefix="",
            cdata_key="Value",
        )


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
        parsed = _parse_bts_xml(mes_url)
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

    The `.mes` basename is not fixed (`MeasurementProtocol.mes` on CQ3000,
    `20260710_ref_data_morech.mes` on CV8000), so it must come from the `.mrf`
    `MeasurementSettingFileName` attribute and be resolved relative to the
    acquisition directory. Globbing `*.mes` is wrong: sibling acquisitions can
    share a directory, and two acquisitions can carry the same `.mes` basename
    with different content.

    Args:
        acquisition_dir: URL of the acquisition directory.
        mes_file_name: The `.mrf` `MeasurementSettingFileName` value.

    Returns:
        The `<bts:Channel>` entries sorted by `bts:Ch`, or `None` when the `.mes`
        is unnamed, absent, or carries no channel list. A missing file is normal:
        the in-repo CV7000 fixture names a `.mes` that was never shipped with it.
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

    `(ActionIndex, Ch)` is 1:1 on `Ch` in all known data — one action can carry
    several channels, but no channel appears under two actions. Should that ever
    break, take the lowest action and warn rather than picking arbitrarily.
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

    `.mes` colours are ARGB (`#AARRGGBB`) while `ChannelInfo.color` accepts only
    3 or 6 hex digits, so the alpha is dropped. Anything unrecognised returns
    `None`, which lets `ChannelInfo`'s own validator pick a colour instead.
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

    Operators routinely give several channels the same `bts:Target`. Deduplication
    must run over the whole instrument channel list, in `bts:Ch` order, *before*
    any per-image pruning: deduplicating over only the acquired channels would
    make a well's channel names depend on which other wells were acquired.
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

    The returned list spans the **full instrument channel range**, not just the
    acquired channels: element `i` is `bts:Ch == i + 1`, matching the converters'
    `start_c = ch - 1`. Slots that were never acquired get a placeholder and are
    pruned per image at compute time by the library's `reindex_channels`.

    Call this once per acquisition, never once per output plate. The CQ3K
    converter splits an acquisition into several plates by projection algorithm,
    but they share one `.mes` and a channel must keep the same label across the
    raw and the projection plates.

    Args:
        mes_channels: The `.mes` channel list, or `None` when there is no `.mes`.
        acquired: `(action_index, ch)` pairs from the `.mlf` image records.
            Duplicates are expected; order is irrelevant.
        mrf_channel_count: Number of `<bts:MeasurementChannel>` entries in the
            `.mrf`.

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


def apply_channel_overrides(
    *,
    resolved: list[ChannelInfo],
    overrides: list[ChannelInfoUI] | None,
    max_acquired_ch: int,
) -> list[ChannelInfo]:
    """Merge a user `advanced.channels` list onto the resolved channels.

    The override maps **positionally onto the instrument channel range**: element
    0 is `bts:Ch` 1. A short list overrides its leading slots and the rest keep
    their `.mes`-derived metadata, so the result is always as long as `resolved` —
    which is what keeps `start_c = ch - 1` addressing the right entry.

    Call this *after* `AcquisitionOptions.update_acquisition_details`, and pass it
    the raw `ChannelInfoUI` list rather than that method's output. The library
    replaces `AcquisitionDetails.channels` wholesale, and it flattens an unset
    colour (`ColorMenu.Auto`) to `None` before `ChannelInfo`'s validator fills in a
    guess — after which a deliberate colour is indistinguishable from a default,
    and the `.mes` colours are gone.

    Args:
        resolved: Output of `resolve_channels` for this acquisition.
        overrides: The user's `advanced.channels`, or `None`.
        max_acquired_ch: Highest `bts:Ch` the acquisition actually acquired.

    Returns:
        The merged channel list, the same length as `resolved`.

    Raises:
        ValueError: If `overrides` is too short to cover every acquired channel.
    """
    if not overrides:
        return list(resolved)

    if len(overrides) < max_acquired_ch:
        raise ValueError(
            f"`advanced.channels` has {len(overrides)} entries but this "
            f"acquisition acquires up to bts:Ch {max_acquired_ch}. Supply at "
            f"least {max_acquired_ch} entries; element 0 maps to Ch1. Entries "
            "for channels the acquisition does not use are discarded."
        )

    merged = list(resolved)
    for index, override in enumerate(overrides[: len(resolved)]):
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

    `bts:TimePoint` is a per-*timeline* counter, not a plate-wide time index. In
    the CV8000 `TimelinesSharedChannel` acquisition three wells are acquired by
    three timelines and carry `TimePoint` 1, 2 and 3, yet each holds a *single* time
    point. Used raw it both mislabels those wells as frames 0, 1 and 2 and, for a
    well whose time points are sparse, leaves the missing indices as empty
    frames — there is no `reindex_t` in the registration pipeline to compact them
    the way `reindex_channels` compacts channels.

    Both fields come from the same per-well grouping, so the `t` axis exists
    exactly when some well fills more than one index.
    """

    is_time_series: bool
    """Whether any well of the plate holds more than one time point.

    Decided plate-wide rather than per well: `TiledImage.add_tile` requires the
    tiles of an image to agree on `axes`, and a plate whose wells disagreed would
    be unreadable as a single collection.
    """

    dense_indices: dict[tuple[int, int], dict[int, int]]
    """`{(bts:Row, bts:Column): {bts:TimePoint: 0-based time index}}`."""

    def start_t(self, *, row: int, column: int, time_point: int) -> int:
        """The 0-based time index of one `.mlf` image record."""
        return self.dense_indices[(row, column)][time_point]


def build_time_index(records: Iterable[tuple[int, int, int]]) -> TimeIndex:
    """Map each well's `bts:TimePoint` values onto a dense 0-based range.

    Call this once per output plate, with every image record of that plate: the
    mapping is per well because each well is a separate output image, while
    `is_time_series` has to be decided across all of them (see `TimeIndex`).

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

    Both converters read pixel size and frame size from the *first*
    `<bts:MeasurementChannel>` and apply them to every channel. That is correct for
    all data seen so far, but a binned or differently-cropped channel would be
    silently misplaced. Warn rather than raise: the geometry of channel 1 is still
    what gets applied, so this reports a suspicion, it does not change behaviour.

    Call once per acquisition, not per plate or per field of view.

    Args:
        geometries: One entry per `.mrf` channel, in file order.
        acquisition_dir: Named in the warning, since conversion runs a batch of
            acquisitions one at a time.
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
