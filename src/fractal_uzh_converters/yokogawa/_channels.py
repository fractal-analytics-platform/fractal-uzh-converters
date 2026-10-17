"""Channel metadata for the Yokogawa BTS schema, shared by both instruments.

Covers the `.mes` channel list and the naming rules that turn it into `ChannelInfo`,
the `.mrf` channel-geometry check and the `.mlf` time-index normalisation. The
`.mlf`/`.mrf` models live in `_records.py`, the parser in `_parse.py`.

The `.mes` carries the human-readable channel targets the `.mlf`/`.mrf` pair lacks.
"""

import logging
import warnings
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
from ome_zarr_converters_tools import (
    ChannelInfo,
    ChannelInfoUI,
    join_url_paths,
)
from pydantic import BaseModel, ConfigDict, Field
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


def apply_channel_overrides(
    *,
    resolved: list[ChannelInfo],
    overrides: list[ChannelInfoUI] | None,
    max_acquired_ch: int,
) -> list[ChannelInfo]:
    """Merge a user `advanced.channels` list onto the resolved channels.

    The override maps **positionally onto the instrument channel range**: element 0
    is `bts:Ch` 1. A short list overrides its leading slots only, so the result is
    always as long as `resolved` and `start_c = ch - 1` keeps addressing the right
    entry.

    Call *after* `AcquisitionOptions.update_acquisition_details`, passing the raw
    `ChannelInfoUI` list rather than that method's output: it replaces `channels`
    wholesale and flattens an unset colour to a guess, after which the `.mes`
    colours are unrecoverable.

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
