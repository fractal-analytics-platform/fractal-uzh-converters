"""Utility functions for Plain TIFF data."""

import logging
from pathlib import Path

import pandas as pd
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
    """Acquisition model for plain HCS TIFF datasets.

    The acquisition directory (``path``) must contain a ``tiles.csv`` file
    with columns: ``file_path``, ``row``, ``column``, and optionally
    ``fov_name``, ``start_*``, ``length_*``.  An optional
    ``acquisition_details.toml`` provides global metadata (pixel size,
    channels, axes, coordinate-space flags).
    """


class SingleTiffAcquisitionModel(BaseAcquisitionModel):
    """Acquisition model for plain single-image TIFF datasets.

    The acquisition directory (``path``) must contain a ``tiles.csv`` file
    with columns: ``file_path``, ``fov_name`` (used as the output zarr name),
    and optionally ``start_*``, ``length_*``.  An optional
    ``acquisition_details.toml`` provides global metadata.
    """


######################################################################
#
# Internal helpers
#
######################################################################


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


def _load_acquisition_details(acq_path: str) -> AcquisitionDetails:
    """Load acquisition details from ``acquisition_details.toml`` if present."""
    toml_path = Path(acq_path) / "acquisition_details.toml"
    if not toml_path.exists():
        return AcquisitionDetails()

    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

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
    """Parse plain HCS TIFF metadata and return a list of TiledImages."""
    acq_path = acquisition_model.path
    tiles_table = _load_tiles_table(acq_path)
    acquisition_details = _load_acquisition_details(acq_path)
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
    """Parse plain single-image TIFF metadata and return a list of TiledImages."""
    acq_path = acquisition_model.path
    tiles_table = _load_tiles_table(acq_path)
    acquisition_details = _load_acquisition_details(acq_path)
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
