"""Copy the plate-level Yokogawa vendor files into the converted plate.

An acquisition carries five plate-level metadata files: the `.mlf` record list
and the `.mrf` acquisition detail under fixed names, plus a `.mes` protocol, a
`.wpi` plate definition and a `.wpp` plate product file, whose names are recorded
inside the `.mrf`. The converters model only the fraction of them they need, so
they are copied verbatim into `<plate>.zarr/metadata/` and nothing is lost.

Nothing here raises: a metadata copy must never fail a conversion that has
already written its images.
"""

import logging

from ome_zarr_converters_tools import (
    ImageInPlate,
    TiledImage,
    filesystem_for_url,
    join_url_paths,
)
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_pascal

from fractal_uzh_converters.common._yokogawa import _parse_bts_xml

logger = logging.getLogger(__name__)

#: Plate-level files whose basename is fixed by the BTS schema.
_FIXED_FILE_NAMES = ("MeasurementData.mlf", "MeasurementDetail.mrf")

#: Subdirectory of the plate the vendor files are copied into.
METADATA_DIR_NAME = "metadata"

#: How many `_acq{id}_{k}` variants to try before giving up on a name collision.
_MAX_COLLISION_VARIANTS = 10


######################################################################
#
# Reading the variable file names out of the `.mrf`
#
######################################################################


class _MrfNamesBase(BaseModel):
    """Base for the trimmed `.mrf` view used to discover file names.

    Deliberately *not* the converters' `MeasurementDetail`: those pin a dozen
    required fields and `cq3k` still uses `extra="forbid"`, so a `.mrf` this
    module cannot fully validate would cost the two fixed-name copies as well.
    Here every field is optional and unknown ones are ignored — a `.mrf` that
    names no `.wpi` simply yields one file fewer.
    """

    model_config = ConfigDict(
        alias_generator=to_pascal,
        extra="ignore",
        populate_by_name=True,
    )


class _SamplePlate(_MrfNamesBase):
    """`<bts:MeasurementSamplePlate>`, for the `.wpi` and `.wpp` names."""

    well_plate_file_name: str | None = None
    well_plate_product_file_name: str | None = None


class _MeasurementDetailNames(_MrfNamesBase):
    """Root `<bts:MeasurementDetail>`, for the `.mes` name and the sample plate."""

    measurement_setting_file_name: str | None = None
    measurement_sample_plate: _SamplePlate | None = None


def _source_file_names(acquisition_dir: str) -> list[str]:
    """Basenames of the plate-level vendor files of `acquisition_dir`.

    The two fixed names are always returned, even when the `.mrf` is missing or
    unparsable — the `.mlf` is still worth copying on its own.

    Args:
        acquisition_dir: URL of the acquisition directory.

    Returns:
        Basenames, deduplicated, in a stable order. Each is resolved against
        `acquisition_dir` by the caller.
    """
    names = list(_FIXED_FILE_NAMES)

    mrf_url = join_url_paths(acquisition_dir, "MeasurementDetail.mrf")
    try:
        detail = _MeasurementDetailNames(**_parse_bts_xml(mrf_url)["MeasurementDetail"])
    except FileNotFoundError:
        logger.warning(
            f"No `MeasurementDetail.mrf` at {mrf_url}; the `.mes`, `.wpi` and `.wpp` "
            "cannot be located and will not be copied."
        )
        return names
    except Exception as e:
        logger.warning(
            f"Could not read the file names out of {mrf_url} ({e}); the `.mes`, "
            "`.wpi` and `.wpp` will not be copied."
        )
        return names

    plate = detail.measurement_sample_plate
    variable = [
        detail.measurement_setting_file_name,
        plate.well_plate_file_name if plate else None,
        plate.well_plate_product_file_name if plate else None,
    ]
    for name in variable:
        # Filenames are preserved verbatim, including non-ASCII and spaces
        # ("10_Greiner_μClear #655090.wpp"): they are the vendor's identifiers
        # and downstream tooling matches on them.
        if name and name not in names:
            names.append(name)
    return names


######################################################################
#
# Copying
#
######################################################################


def plate_urls_for_images(
    *, zarr_dir: str, tiled_images: list[TiledImage]
) -> list[str]:
    """Absolute URLs of the distinct plates `tiled_images` were written into.

    `ImageInPlate.plate_path()` is relative to `zarr_dir`, so it is joined here
    the same way the library's own plate setup does it. One acquisition can
    produce several plates — a CQ3K acquisition yields one per projection
    algorithm — and each of them gets its own copy of the metadata.

    Args:
        zarr_dir: Directory the plates were written into.
        tiled_images: Images of a single acquisition.

    Returns:
        Sorted, deduplicated plate URLs. Empty for collections that are not
        plates, which carry no `plate_path`.
    """
    plate_paths = {
        image.collection.plate_path()
        for image in tiled_images
        if isinstance(image.collection, ImageInPlate)
    }
    return sorted(join_url_paths(zarr_dir, path) for path in plate_paths)


def _split_name(file_name: str) -> tuple[str, str]:
    """Split a basename into stem and extension, e.g. `("a.b", ".c")`."""
    stem, dot, suffix = file_name.rpartition(".")
    if not dot:
        return file_name, ""
    return stem, f".{suffix}"


def _destination_url(
    *,
    metadata_dir: str,
    file_name: str,
    payload: bytes,
    acquisition_id: int,
) -> str | None:
    """Where to write `file_name` under `metadata_dir`, or `None` to skip.

    Several acquisitions can target one plate while shipping identically named
    files, so a taken name is never overwritten: the copy moves to
    `<stem>_acq{acquisition_id}<ext>`. An existing byte-identical copy is left
    alone, which also makes re-running a conversion idempotent.
    """
    fs = filesystem_for_url(metadata_dir)
    stem, ext = _split_name(file_name)
    candidates = [file_name, f"{stem}_acq{acquisition_id}{ext}"]
    candidates += [
        f"{stem}_acq{acquisition_id}_{k}{ext}"
        for k in range(1, _MAX_COLLISION_VARIANTS)
    ]

    for candidate in candidates:
        url = join_url_paths(metadata_dir, candidate)
        if not fs.exists(url):
            return url
        with fs.open(url, "rb") as f:
            if f.read() == payload:
                return None

    logger.warning(
        f"Gave up copying {file_name} into {metadata_dir}: "
        f"{_MAX_COLLISION_VARIANTS + 1} differing copies are already there."
    )
    return None


def copy_source_metadata(
    *,
    acquisition_dir: str,
    plate_urls: list[str],
    acquisition_id: int = 0,
) -> None:
    """Copy the acquisition's vendor metadata into every plate it produced.

    Each source is read once and fanned out, so a CQ3K acquisition split across
    several projection plates does not re-read it per plate.

    Source and destination are resolved through *separate* `filesystem_for_url`
    lookups and the bytes travel through this process. `fs.copy`/`fs.put` are
    deliberately not used: the two are routinely different filesystems (a local
    raw directory converted into an `s3://` `zarr_dir`), which is the whole point
    of the requirement.

    A missing or unreadable file is logged and skipped, never raised. That path
    is normal, not defensive: the in-repo CV7000 fixture names a `.mes`, a `.wpi`
    and a `.wpp` that were never shipped with it.

    Args:
        acquisition_dir: URL of the raw acquisition directory.
        plate_urls: URLs of the `<plate>.zarr` stores built from it.
        acquisition_id: Used to disambiguate a file name already taken in a
            plate by another acquisition.
    """
    if not plate_urls:
        return

    for file_name in _source_file_names(acquisition_dir):
        src = join_url_paths(acquisition_dir, file_name)
        try:
            src_fs = filesystem_for_url(src)
            with src_fs.open(src, "rb") as f:
                payload = f.read()
        except FileNotFoundError:
            logger.warning(
                f"Source metadata file {src} does not exist; it will not be copied "
                "into the plate."
            )
            continue
        except Exception as e:
            logger.warning(f"Could not read source metadata file {src} ({e}).")
            continue

        for plate_url in plate_urls:
            metadata_dir = join_url_paths(plate_url, METADATA_DIR_NAME)
            try:
                dst = _destination_url(
                    metadata_dir=metadata_dir,
                    file_name=file_name,
                    payload=payload,
                    acquisition_id=acquisition_id,
                )
                if dst is None:
                    continue
                dst_fs = filesystem_for_url(dst)
                dst_fs.makedirs(metadata_dir, exist_ok=True)
                with dst_fs.open(dst, "wb") as f:
                    f.write(payload)
            except Exception as e:
                logger.warning(
                    f"Could not copy {src} into {metadata_dir} ({e}); the "
                    "conversion itself is unaffected."
                )
