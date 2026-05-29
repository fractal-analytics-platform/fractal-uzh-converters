"""Utility functions for Custom TIFF data."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import tifffile
import tomllib
from ome_zarr_converters_tools import (
    AcquisitionDetails,
    ChannelInfo,
    ConverterOptions,
    TiledImage,
    tiles_aggregation_pipeline,
)
from ome_zarr_converters_tools.core import (
    hcs_images_from_dataframe,
    single_images_from_dataframe,
)

from fractal_uzh_converters.common import BaseAcquisitionModel

logger = logging.getLogger(__name__)


######################################################################
#
# Acquisition Input Models
#
######################################################################


class HcsTiffAcquisitionModel(BaseAcquisitionModel):
    """Acquisition model for custom HCS TIFF datasets.

    The acquisition directory (``path``) must contain a ``tiles.csv`` file
    with columns: ``file_path``, ``row``, ``column``, and optionally
    ``fov_name``, ``start_*``, ``length_*``.  An optional
    ``acquisition_details.toml`` provides global metadata (pixel size,
    channels, axes, coordinate-space flags).
    """


class SingleTiffAcquisitionModel(BaseAcquisitionModel):
    """Acquisition model for custom single-image TIFF datasets.

    The acquisition directory (``path``) must contain a ``tiles.csv`` file
    with columns: ``file_path``, ``fov_name`` (used as the output zarr name),
    and optionally ``start_*``, ``length_*``.  An optional
    ``acquisition_details.toml`` provides global metadata.
    """


######################################################################
#
# TIFF metadata parsing
#
######################################################################

# Conversion factors to micrometers for common space units
_SPACE_UNIT_TO_UM: dict[str, float] = {
    "um": 1.0,
    "µm": 1.0,
    "micrometer": 1.0,
    "micrometers": 1.0,
    "nm": 0.001,
    "nanometer": 0.001,
    "nanometers": 0.001,
    "mm": 1000.0,
    "millimeter": 1000.0,
    "millimeters": 1000.0,
    "cm": 10000.0,
    "centimeter": 10000.0,
    "centimeters": 10000.0,
}

# Conversion factors to seconds for common time units
_TIME_UNIT_TO_S: dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 0.001,
    "millisecond": 0.001,
    "milliseconds": 0.001,
}


def _convert_space_unit(value: float, unit: str) -> float | None:
    factor = _SPACE_UNIT_TO_UM.get(unit.strip().lower())
    if factor is None:
        logger.warning(f"Unknown space unit '{unit}', skipping spacing value.")
        return None
    return value * factor


def _convert_time_unit(value: float, unit: str) -> float | None:
    factor = _TIME_UNIT_TO_S.get(unit.strip().lower())
    if factor is None:
        logger.warning(f"Unknown time unit '{unit}', skipping t_spacing value.")
        return None
    return value * factor


def _parse_tiff_metadata(path: Path) -> dict:
    """Parse metadata from a TIFF file. Never raises.

    Returns a partial dict with any subset of:
      pixelsize, z_spacing, t_spacing (floats, µm or s),
      length_x, length_y, length_z, length_t (ints, pixels).
    """
    result: dict = {}
    try:
        with tifffile.TiffFile(str(path)) as tif:
            # Shape from first page — always tried first as a baseline
            try:
                page_shape = tif.pages[0].shape
                if len(page_shape) >= 2:
                    result["length_y"] = int(page_shape[-2])
                    result["length_x"] = int(page_shape[-1])
            except Exception as exc:
                logger.warning(f"Could not read page shape from {path}: {exc}")

            # Shape from series — may provide z/t dims and refine x/y
            try:
                if tif.series:
                    series = tif.series[0]
                    for ax, size in zip(series.axes.lower(), series.shape, strict=True):
                        if ax == "z":
                            result["length_z"] = int(size)
                        elif ax == "t":
                            result["length_t"] = int(size)
                        elif ax == "y":
                            result["length_y"] = int(size)
                        elif ax == "x":
                            result["length_x"] = int(size)
            except Exception as exc:
                logger.warning(f"Could not read series shape from {path}: {exc}")

            # OME-TIFF metadata — spacing
            if tif.is_ome:
                try:
                    ome_xml = tif.ome_metadata
                    if ome_xml:
                        root = ET.fromstring(ome_xml)
                        pixels = root.find(".//{*}Pixels")
                        if pixels is not None:
                            psx = pixels.get("PhysicalSizeX")
                            psx_unit = pixels.get("PhysicalSizeXUnit", "µm")
                            if psx is not None:
                                val = _convert_space_unit(float(psx), psx_unit)
                                if val is not None and val > 0:
                                    result["pixelsize"] = val

                            psz = pixels.get("PhysicalSizeZ")
                            psz_unit = pixels.get("PhysicalSizeZUnit", "µm")
                            if psz is not None:
                                val = _convert_space_unit(float(psz), psz_unit)
                                if val is not None and val > 0:
                                    result["z_spacing"] = val

                            ti = pixels.get("TimeIncrement")
                            ti_unit = pixels.get("TimeIncrementUnit", "s")
                            if ti is not None:
                                val = _convert_time_unit(float(ti), ti_unit)
                                if val is not None and val > 0:
                                    result["t_spacing"] = val
                except Exception as exc:
                    logger.warning(f"Could not parse OME metadata from {path}: {exc}")

            # ImageJ TIFF metadata — spacing (only if not already from OME)
            elif tif.is_imagej:
                try:
                    ij = tif.imagej_metadata or {}
                    unit = ij.get("unit", "µm")

                    spacing = ij.get("spacing")
                    if spacing is not None:
                        val = _convert_space_unit(float(spacing), unit)
                        if val is not None and val > 0:
                            result["z_spacing"] = val

                    tags = tif.pages[0].tags
                    if "XResolution" in tags:
                        num_pixels, units = tags["XResolution"].value
                        if num_pixels > 0:
                            x = units / num_pixels
                            val = _convert_space_unit(x, unit)
                            if val is not None and val > 0:
                                result["pixelsize"] = val
                except Exception as exc:
                    logger.warning(
                        f"Could not parse ImageJ metadata from {path}: {exc}"
                    )

    except Exception as exc:
        logger.warning(f"Could not open {path} for metadata parsing: {exc}")

    return result


######################################################################
#
# Internal helpers
#
######################################################################

_ACQUISITION_DETAILS_SPACING_KEYS = frozenset({"pixelsize", "z_spacing", "t_spacing"})


def _load_tiles_table(acq_path: str) -> pd.DataFrame:
    """Load tiles table from ``tiles.csv``.

    Relative ``file_path`` values are resolved against ``acq_path``.
    """
    base = Path(acq_path)
    fpath = base / "tiles.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"No tiles.csv found in {acq_path}")
    df = pd.read_csv(fpath)
    df["file_path"] = df["file_path"].apply(
        lambda fp: fp if Path(fp.strip()).is_absolute() else str(base / fp.strip())
    )
    return df


def _load_toml_raw(acq_path: str) -> dict:
    """Load raw TOML dict from ``acquisition_details.toml`` if present."""
    toml_path = Path(acq_path) / "acquisition_details.toml"
    if not toml_path.exists():
        return {}
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


def _build_acquisition_details(raw: dict) -> AcquisitionDetails:
    """Build an AcquisitionDetails from a raw dict (TOML-format keys)."""
    raw = dict(raw)
    channel_names = raw.pop("channel_names", None)
    wavelengths = raw.pop("wavelengths", None)
    axes = raw.pop("axes", None)

    channels = None
    if channel_names is not None:
        if wavelengths is None:
            wavelengths = [None] * len(channel_names)
        channels = [
            ChannelInfo(channel_label=ch, wavelength_id=w)
            for ch, w in zip(channel_names, wavelengths, strict=True)
        ]

    kwargs: dict = {k: v for k, v in raw.items() if not k.startswith("#")}
    if channels is not None:
        kwargs["channels"] = channels
    if axes is not None:
        kwargs["axes"] = axes

    return AcquisitionDetails(**kwargs)


def _load_acquisition_details(
    acq_path: str,
    tiff_defaults: dict | None = None,
) -> AcquisitionDetails:
    """Load acquisition details, merging TIFF defaults with TOML values.

    Priority (lowest to highest): TIFF metadata → TOML → user input (applied later).
    Only spacing keys (pixelsize, z_spacing, t_spacing) are taken from tiff_defaults.
    """
    spacing_defaults = {
        k: v
        for k, v in (tiff_defaults or {}).items()
        if k in _ACQUISITION_DETAILS_SPACING_KEYS
    }
    toml_raw = _load_toml_raw(acq_path)
    merged = {**spacing_defaults, **toml_raw}
    return _build_acquisition_details(merged)


def _apply_table_defaults(
    tiles_table: pd.DataFrame,
    tiff_meta: dict,
) -> pd.DataFrame:
    """Add missing columns to tiles_table using TIFF metadata or hard-coded defaults.

    Existing columns are never overwritten.
    """
    df = tiles_table.copy()

    def _add_if_missing(col: str, value) -> None:
        if col not in df.columns and value is not None:
            df[col] = value

    _add_if_missing("fov_name", "Image")
    _add_if_missing("start_x", 0.0)
    _add_if_missing("start_y", 0.0)
    _add_if_missing("length_x", tiff_meta.get("length_x"))
    _add_if_missing("length_y", tiff_meta.get("length_y"))
    _add_if_missing("length_z", tiff_meta.get("length_z"))
    _add_if_missing("length_t", tiff_meta.get("length_t"))

    return df


######################################################################
#
# Metadata parsers
#
######################################################################


def parse_hcs_tiff_metadata(
    *,
    acquisition_model: HcsTiffAcquisitionModel,
    converter_options: ConverterOptions,
) -> list[TiledImage]:
    """Parse custom HCS TIFF metadata and return a list of TiledImages."""
    acq_path = acquisition_model.path
    tiles_table = _load_tiles_table(acq_path)

    first_path = Path(tiles_table["file_path"].iloc[0])
    tiff_meta = _parse_tiff_metadata(first_path)
    tiles_table = _apply_table_defaults(tiles_table, tiff_meta)

    acquisition_details = _load_acquisition_details(acq_path, tiff_defaults=tiff_meta)
    acquisition_details = acquisition_model.advanced.update_acquisition_details(
        acquisition_details
    )

    tiles = hcs_images_from_dataframe(
        tiles_table=tiles_table,
        acquisition_details=acquisition_details,
        plate_name=acquisition_model.normalized_plate_name,
        acquisition_id=acquisition_model.acquisition_id,
    )

    return tiles_aggregation_pipeline(
        tiles=tiles,
        converter_options=converter_options,
        filters=acquisition_model.advanced.filters,
        validators=None,
        resource=None,
    )


def parse_single_tiff_metadata(
    *,
    acquisition_model: SingleTiffAcquisitionModel,
    converter_options: ConverterOptions,
) -> list[TiledImage]:
    """Parse custom single-image TIFF metadata and return a list of TiledImages."""
    acq_path = acquisition_model.path
    path = Path(acq_path)

    if path.is_file():
        tiff_meta = _parse_tiff_metadata(path)
        if "length_x" not in tiff_meta or "length_y" not in tiff_meta:
            raise ValueError(
                f"Could not determine image dimensions from {path}. "
                "Ensure the file is a readable TIFF."
            )
        tiles_table = pd.DataFrame(
            {
                "file_path": [str(path.resolve())],
                "fov_name": [path.stem],
                "start_x": [0.0],
                "start_y": [0.0],
                "length_x": [tiff_meta["length_x"]],
                "length_y": [tiff_meta["length_y"]],
            }
        )
        acquisition_details = _load_acquisition_details(
            str(path.parent), tiff_defaults=tiff_meta
        )
    else:
        tiles_table = _load_tiles_table(acq_path)
        first_path = Path(tiles_table["file_path"].iloc[0])
        tiff_meta = _parse_tiff_metadata(first_path)
        tiles_table = _apply_table_defaults(tiles_table, tiff_meta)
        acquisition_details = _load_acquisition_details(
            acq_path, tiff_defaults=tiff_meta
        )

    acquisition_details = acquisition_model.advanced.update_acquisition_details(
        acquisition_details
    )

    tiles_table = tiles_table.copy()
    tiles_table["image_path"] = acquisition_model.normalized_plate_name

    tiles = single_images_from_dataframe(
        tiles_table=tiles_table,
        acquisition_details=acquisition_details,
    )

    return tiles_aggregation_pipeline(
        tiles=tiles,
        converter_options=converter_options,
        filters=acquisition_model.advanced.filters,
        validators=None,
        resource=None,
    )
