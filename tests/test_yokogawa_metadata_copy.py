"""Tests for copying the Yokogawa vendor metadata into the plate (#46).

The snapshot model is a closed set that only inspects zarr metadata, so it can
never see these files. Everything here asserts on the filesystem directly.

Plate directory names are always discovered, never hard-coded: the CQ3K
projection suffixes are being renamed in a separate change and these tests must
not care.
"""

import logging
from pathlib import Path

import pytest
from ome_zarr_converters_tools import SingleImage, join_url_paths

from fractal_uzh_converters.common import plate_urls_for_images
from fractal_uzh_converters.yokogawa import _source_metadata
from fractal_uzh_converters.yokogawa._source_metadata import (
    METADATA_DIR_NAME,
    copy_source_metadata,
)
from fractal_uzh_converters.yokogawa.cellvoyager import convert_cellvoyager
from fractal_uzh_converters.yokogawa.cq3k import convert_cq3k
from fractal_uzh_converters.yokogawa.cq3k._utils import (
    CQ3KAcquisitionModel,
    parse_cq3k_metadata,
)

from .utils import DATA_DIR

CQ3K_ACQUISITION = DATA_DIR / "Yokogawa-CQ3K" / "raw" / "hcs_2w1p1c1z1t_mip"
CELLVOYAGER_ACQUISITION = DATA_DIR / "Yokogawa-CellVoyager" / "raw" / "hcs_1w1p1c1z1t"

# The CQ3K fixture ships all five plate-level files. The `.wpp` name carries a
# non-ASCII character, a space and a `#` — the exact reason filenames are copied
# verbatim rather than sanitized.
CQ3K_WPP_NAME = "10_Greiner_μClear #655090.wpp"
CQ3K_METADATA_FILES = {
    "MeasurementData.mlf",
    "MeasurementDetail.mrf",
    "MeasurementProtocol.mes",
    "NoPlateID.wpi",
    CQ3K_WPP_NAME,
}

# The CV7000 fixture ships only these two; its `.mrf` names a `.mes`, a `.wpi`
# and a `.wpp` that were never shipped with it.
CELLVOYAGER_PRESENT_FILES = {"MeasurementData.mlf", "MeasurementDetail.mrf"}
CELLVOYAGER_MISSING_FILES = {
    "20200812-Joel-CardiomyocyteDifferentiation14-Cycle1.mes",
    "20200812-CardiomyocyteDifferentiation14-Cycle1.wpi",
    "1009602002_Greiner_#655090.wpp",
}

_MRF_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<bts:MeasurementDetail bts:Version="1.0" bts:OperatorName="{operator}" \
bts:MeasurementSettingFileName="{mes}" \
xmlns:bts="http://www.yokogawa.co.jp/BTS/BTSSchema/1.0">
  <bts:MeasurementSamplePlate bts:Name="P" bts:WellPlateFileName="{wpi}" \
bts:WellPlateProductFileName="{wpp}" />
</bts:MeasurementDetail>
"""


def _plate_dirs(zarr_dir: Path) -> list[Path]:
    """Every `*.zarr` plate written under `zarr_dir`."""
    return sorted(zarr_dir.glob("*.zarr"))


def _metadata_files(plate_dir: Path) -> dict[str, bytes]:
    """`{filename: content}` of the plate's `metadata/` directory."""
    metadata_dir = plate_dir / METADATA_DIR_NAME
    if not metadata_dir.is_dir():
        return {}
    return {f.name: f.read_bytes() for f in metadata_dir.iterdir() if f.is_file()}


def _build_acquisition(
    directory: Path,
    *,
    operator: str = "op",
    mes: str = "Protocol.mes",
    wpi: str = "Plate.wpi",
    wpp: str = "Product.wpp",
    write_optional: bool = True,
) -> Path:
    """A minimal acquisition directory: only the `.mrf` is ever parsed."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "MeasurementDetail.mrf").write_text(
        _MRF_TEMPLATE.format(operator=operator, mes=mes, wpi=wpi, wpp=wpp),
        encoding="utf-8",
    )
    (directory / "MeasurementData.mlf").write_bytes(f"mlf-{operator}".encode())
    if write_optional:
        for name in (mes, wpi, wpp):
            (directory / name).write_bytes(f"{name}-{operator}".encode())
    return directory


# ---------------------------------------------------------------------------
# End to end, through the converters
# ---------------------------------------------------------------------------


def test_cq3k_copies_every_file_byte_for_byte(tmp_path: Path, converter_options):
    """All five plate-level files land in the plate, unchanged."""
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()

    convert_cq3k(
        zarr_dir=str(zarr_dir),
        acquisitions=[{"path": str(CQ3K_ACQUISITION), "acquisition_id": 0}],
        converter_options=converter_options,
    )

    plates = _plate_dirs(zarr_dir)
    assert plates, "the conversion produced no plate"
    for plate_dir in plates:
        copied = _metadata_files(plate_dir)
        assert set(copied) == CQ3K_METADATA_FILES
        for name, content in copied.items():
            assert content == (CQ3K_ACQUISITION / name).read_bytes()


def test_cq3k_preserves_the_verbatim_filename(tmp_path: Path, converter_options):
    """Non-ASCII, spaces and `#` survive: the vendor name is the identifier."""
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()

    convert_cq3k(
        zarr_dir=str(zarr_dir),
        acquisitions=[{"path": str(CQ3K_ACQUISITION), "acquisition_id": 0}],
        converter_options=converter_options,
    )

    for plate_dir in _plate_dirs(zarr_dir):
        assert CQ3K_WPP_NAME in _metadata_files(plate_dir)


def test_cellvoyager_warns_for_files_the_mrf_names_but_ships(
    tmp_path: Path, converter_options, caplog
):
    """A missing source is a warning and a skip, never a failed conversion."""
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()

    with caplog.at_level(logging.WARNING, logger=_source_metadata.__name__):
        convert_cellvoyager(
            zarr_dir=str(zarr_dir),
            acquisitions=[
                {
                    "path": str(CELLVOYAGER_ACQUISITION),
                    "acquisition_id": 0,
                    "image_extension": ".png",
                }
            ],
            converter_options=converter_options,
        )

    plates = _plate_dirs(zarr_dir)
    assert plates, "the conversion produced no plate"
    for plate_dir in plates:
        assert set(_metadata_files(plate_dir)) == CELLVOYAGER_PRESENT_FILES

    warned = "\n".join(
        record.message
        for record in caplog.records
        if record.name == _source_metadata.__name__
    )
    for missing in CELLVOYAGER_MISSING_FILES:
        assert missing in warned


def test_rerunning_a_conversion_does_not_duplicate_the_metadata(
    tmp_path: Path, converter_options
):
    """An identical copy already in place is left alone, not suffixed."""
    zarr_dir = tmp_path / "output"
    zarr_dir.mkdir()

    acquisitions = [{"path": str(CQ3K_ACQUISITION), "acquisition_id": 0}]
    for _ in range(2):
        convert_cq3k(
            zarr_dir=str(zarr_dir),
            acquisitions=acquisitions,
            converter_options=converter_options,
            overwrite="Overwrite",
        )

    for plate_dir in _plate_dirs(zarr_dir):
        assert set(_metadata_files(plate_dir)) == CQ3K_METADATA_FILES


# ---------------------------------------------------------------------------
# `copy_source_metadata` on its own
# ---------------------------------------------------------------------------


def test_copies_into_every_plate_of_the_acquisition(tmp_path: Path):
    """One acquisition, several plates — a CQ3K projection split gets them all."""
    acquisition = _build_acquisition(tmp_path / "raw")
    plate_urls = [str(tmp_path / f"plate_{suffix}.zarr") for suffix in ("a", "b", "c")]

    copy_source_metadata(acquisition_dir=str(acquisition), plate_urls=plate_urls)

    for plate_url in plate_urls:
        assert set(_metadata_files(Path(plate_url))) == {
            "MeasurementData.mlf",
            "MeasurementDetail.mrf",
            "Protocol.mes",
            "Plate.wpi",
            "Product.wpp",
        }


def test_a_second_acquisition_does_not_overwrite_the_first(tmp_path: Path):
    """Same plate, same filenames, different content: both copies are kept."""
    first = _build_acquisition(tmp_path / "raw_0", operator="alice")
    second = _build_acquisition(tmp_path / "raw_1", operator="bob")
    plate_url = str(tmp_path / "plate.zarr")

    copy_source_metadata(
        acquisition_dir=str(first), plate_urls=[plate_url], acquisition_id=0
    )
    copy_source_metadata(
        acquisition_dir=str(second), plate_urls=[plate_url], acquisition_id=1
    )

    copied = _metadata_files(Path(plate_url))
    assert copied["MeasurementData.mlf"] == b"mlf-alice"
    assert copied["MeasurementData_acq1.mlf"] == b"mlf-bob"
    assert copied["Protocol.mes"] == b"Protocol.mes-alice"
    assert copied["Protocol_acq1.mes"] == b"Protocol.mes-bob"


def test_a_missing_mrf_still_copies_the_mlf(tmp_path: Path, caplog):
    """The two fixed names do not depend on the `.mrf` being readable."""
    acquisition = tmp_path / "raw"
    acquisition.mkdir()
    (acquisition / "MeasurementData.mlf").write_bytes(b"records")
    plate_url = str(tmp_path / "plate.zarr")

    with caplog.at_level(logging.WARNING, logger=_source_metadata.__name__):
        copy_source_metadata(acquisition_dir=str(acquisition), plate_urls=[plate_url])

    assert _metadata_files(Path(plate_url)) == {"MeasurementData.mlf": b"records"}
    assert "MeasurementDetail.mrf" in caplog.text


def test_a_malformed_mrf_still_copies_the_fixed_names(tmp_path: Path, caplog):
    """A `.mrf` that does not parse costs the three variable files, nothing more."""
    acquisition = tmp_path / "raw"
    acquisition.mkdir()
    (acquisition / "MeasurementData.mlf").write_bytes(b"records")
    (acquisition / "MeasurementDetail.mrf").write_bytes(b"<bts:Not>closed")
    plate_url = str(tmp_path / "plate.zarr")

    with caplog.at_level(logging.WARNING, logger=_source_metadata.__name__):
        copy_source_metadata(acquisition_dir=str(acquisition), plate_urls=[plate_url])

    assert set(_metadata_files(Path(plate_url))) == {
        "MeasurementData.mlf",
        "MeasurementDetail.mrf",
    }
    assert "MeasurementDetail.mrf" in caplog.text


def test_no_plates_is_a_no_op(tmp_path: Path):
    """Nothing to copy into, nothing read."""
    acquisition = _build_acquisition(tmp_path / "raw")
    copy_source_metadata(acquisition_dir=str(acquisition), plate_urls=[])
    assert not (tmp_path / "plate.zarr").exists()


def test_reads_and_writes_through_fsspec(tmp_path: Path, monkeypatch):
    """Source and destination are resolved through separate `filesystem_for_url`
    lookups, so a local raw directory can be converted into an `s3://` store.

    The library only resolves local and `s3://` URLs, so this asserts the calls
    rather than exercising a second backend. `fs.copy`/`fs.put` would take a
    single filesystem and silently rule the cross-backend case out.
    """
    acquisition = _build_acquisition(tmp_path / "raw")
    plate_url = str(tmp_path / "plate.zarr")

    urls: list[str] = []
    original = _source_metadata.filesystem_for_url

    def _spy(url, *args, **kwargs):
        urls.append(url)
        return original(url, *args, **kwargs)

    monkeypatch.setattr(_source_metadata, "filesystem_for_url", _spy)
    copy_source_metadata(acquisition_dir=str(acquisition), plate_urls=[plate_url])

    # Both prefixes go through `join_url_paths`, which normalises to forward
    # slashes. On Windows `str(tmp_path / "raw")` is backslash-separated and
    # would match none of the URLs the spy collected.
    acquisition_url = join_url_paths(str(acquisition))
    metadata_dir = join_url_paths(plate_url, METADATA_DIR_NAME)
    sources = {url for url in urls if url.startswith(acquisition_url)}
    destinations = {url for url in urls if url.startswith(metadata_dir)}

    assert sources == {
        join_url_paths(acquisition_url, name)
        for name in (
            "MeasurementDetail.mrf",
            "MeasurementData.mlf",
            "Protocol.mes",
            "Plate.wpi",
            "Product.wpp",
        )
    }
    assert destinations
    assert not sources & destinations


# ---------------------------------------------------------------------------
# `plate_urls_for_images`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "zarr_dir", ["/local/zarr_dir", "s3://bucket/zarr_dir"], ids=["local", "s3"]
)
def test_plate_urls_are_absolute_and_deduplicated(zarr_dir, converter_options):
    """`plate_path()` is relative to `zarr_dir`, so it must be joined onto it.

    The fixture's two wells give several images sharing one plate, which is what
    makes the deduplication observable.
    """
    tiled_images = parse_cq3k_metadata(
        acquisition_model=CQ3KAcquisitionModel(
            path=str(CQ3K_ACQUISITION), acquisition_id=0
        ),
        converter_options=converter_options,
    )
    assert len(tiled_images) > 1

    urls = plate_urls_for_images(zarr_dir=zarr_dir, tiled_images=tiled_images)

    assert len(urls) == 1
    assert urls[0].startswith(f"{zarr_dir}/")
    assert urls[0].endswith(".zarr")


def test_collections_without_a_plate_yield_no_urls(converter_options):
    """A single-image collection has no `plate_path`; it is skipped, not crashed on."""
    tiled_images = parse_cq3k_metadata(
        acquisition_model=CQ3KAcquisitionModel(
            path=str(CQ3K_ACQUISITION), acquisition_id=0
        ),
        converter_options=converter_options,
    )
    for image in tiled_images:
        image.collection = SingleImage(image_path="standalone")

    assert plate_urls_for_images(zarr_dir="/zarr_dir", tiled_images=tiled_images) == []
