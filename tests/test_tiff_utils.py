"""Unit tests for custom_tiff metadata parsing utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from fractal_uzh_converters.custom_tiff._utils import (
    _apply_table_defaults,
    _convert_space_unit,
    _convert_time_unit,
    _parse_tiff_metadata,
    _validate_series_axes,
)

# ---------------------------------------------------------------------------
# _convert_space_unit
# ---------------------------------------------------------------------------


class TestConvertSpaceUnit:
    def test_micrometers_passthrough(self):
        assert _convert_space_unit(1.0, "µm") == pytest.approx(1.0)
        assert _convert_space_unit(1.0, "um") == pytest.approx(1.0)
        assert _convert_space_unit(1.0, "micrometer") == pytest.approx(1.0)

    def test_nanometers_to_um(self):
        assert _convert_space_unit(1000.0, "nm") == pytest.approx(1.0)

    def test_millimeters_to_um(self):
        assert _convert_space_unit(1.0, "mm") == pytest.approx(1000.0)

    def test_centimeters_to_um(self):
        assert _convert_space_unit(1.0, "cm") == pytest.approx(10000.0)

    def test_unknown_unit_returns_none(self):
        assert _convert_space_unit(1.0, "inch") is None
        assert _convert_space_unit(1.0, "") is None


# ---------------------------------------------------------------------------
# _convert_time_unit
# ---------------------------------------------------------------------------


class TestConvertTimeUnit:
    def test_seconds_passthrough(self):
        assert _convert_time_unit(1.0, "s") == pytest.approx(1.0)
        assert _convert_time_unit(1.0, "second") == pytest.approx(1.0)
        assert _convert_time_unit(1.0, "seconds") == pytest.approx(1.0)

    def test_milliseconds_to_seconds(self):
        assert _convert_time_unit(500.0, "ms") == pytest.approx(0.5)

    def test_unknown_unit_returns_none(self):
        assert _convert_time_unit(1.0, "min") is None
        assert _convert_time_unit(1.0, "") is None


# ---------------------------------------------------------------------------
# _validate_series_axes
# ---------------------------------------------------------------------------

_FAKE_PATH = Path("fake.tif")


class TestValidateSeriesAxes:
    @pytest.mark.parametrize(
        "axes",
        ["tczyx", "czyx", "zyx", "yx", "tcyx", "tzyx", "cx", "tx", "x", "tcx"],
    )
    def test_valid_axes_do_not_raise(self, axes):
        _validate_series_axes(axes, _FAKE_PATH)

    @pytest.mark.parametrize("axes", ["iyx", "qyx", "syx", "abcde", "rgb"])
    def test_unknown_labels_raise(self, axes):
        with pytest.raises(ValueError, match="unrecognized"):
            _validate_series_axes(axes, _FAKE_PATH)

    @pytest.mark.parametrize("axes", ["zcyx", "yxz", "xyzct", "tcxzy", "zcy"])
    def test_wrong_order_raises(self, axes):
        with pytest.raises(ValueError, match="canonical order"):
            _validate_series_axes(axes, _FAKE_PATH)


# ---------------------------------------------------------------------------
# _parse_tiff_metadata — helpers
# ---------------------------------------------------------------------------


def _write_ome_tiff(path: Path, data: np.ndarray, axes: str, **metadata) -> None:
    with tifffile.TiffWriter(str(path), ome=True) as tw:
        tw.write(
            data,
            photometric="minisblack",
            metadata={"axes": axes, **metadata},
        )


def _write_imagej_tiff(
    path: Path,
    data: np.ndarray,
    *,
    resolution: tuple[float, float],
    spacing: float,
    unit: str,
) -> None:
    tifffile.imwrite(
        str(path),
        data,
        imagej=True,
        resolution=resolution,
        metadata={"unit": unit, "spacing": spacing, "axes": "ZYX"},
    )


# ---------------------------------------------------------------------------
# _parse_tiff_metadata
# ---------------------------------------------------------------------------


class TestParseTiffMetadata:
    def test_plain_2d(self, tmp_path):
        path = tmp_path / "plain2d.tif"
        tifffile.imwrite(str(path), np.zeros((64, 128), dtype=np.uint16))
        meta = _parse_tiff_metadata(str(path))
        assert meta["length_y"] == 64
        assert meta["length_x"] == 128
        assert "length_z" not in meta
        assert "length_c" not in meta

    def test_plain_3d_raises(self, tmp_path):
        # A plain 3D TIFF with no axis metadata — tifffile infers non-canonical
        # axis label (e.g. "Q") for the extra dimension.
        path = tmp_path / "plain3d.tif"
        tifffile.imwrite(
            str(path),
            np.zeros((4, 64, 64), dtype=np.uint16),
            photometric="minisblack",
        )
        with pytest.raises(ValueError):
            _parse_tiff_metadata(str(path))

    def test_ome_czyx_detects_length_c(self, tmp_path):
        path = tmp_path / "ome_czyx.tif"
        _write_ome_tiff(
            path,
            np.zeros((3, 5, 64, 128), dtype=np.uint16),
            axes="CZYX",
        )
        meta = _parse_tiff_metadata(str(path))
        assert meta["length_c"] == 3
        assert meta["length_z"] == 5
        assert meta["length_y"] == 64
        assert meta["length_x"] == 128
        assert "length_t" not in meta

    def test_ome_tczyx_detects_all_dims(self, tmp_path):
        path = tmp_path / "ome_tczyx.tif"
        _write_ome_tiff(
            path,
            np.zeros((2, 3, 5, 64, 128), dtype=np.uint16),
            axes="TCZYX",
        )
        meta = _parse_tiff_metadata(str(path))
        assert meta["length_t"] == 2
        assert meta["length_c"] == 3
        assert meta["length_z"] == 5
        assert meta["length_y"] == 64
        assert meta["length_x"] == 128

    def test_ome_spacing(self, tmp_path):
        path = tmp_path / "ome_spacing.tif"
        _write_ome_tiff(
            path,
            np.zeros((2, 3, 5, 64, 64), dtype=np.uint16),
            axes="TCZYX",
            PhysicalSizeX=0.65,
            PhysicalSizeXUnit="µm",
            PhysicalSizeZ=5.0,
            PhysicalSizeZUnit="µm",
            TimeIncrement=2.5,
            TimeIncrementUnit="s",
        )
        meta = _parse_tiff_metadata(str(path))
        assert meta["pixelsize"] == pytest.approx(0.65)
        assert meta["z_spacing"] == pytest.approx(5.0)
        assert meta["t_spacing"] == pytest.approx(2.5)

    def test_ome_spacing_nm_unit_conversion(self, tmp_path):
        path = tmp_path / "ome_nm.tif"
        _write_ome_tiff(
            path,
            np.zeros((1, 64, 64), dtype=np.uint16),
            axes="ZYX",
            PhysicalSizeX=500.0,
            PhysicalSizeXUnit="nm",
        )
        meta = _parse_tiff_metadata(str(path))
        assert meta["pixelsize"] == pytest.approx(0.5)

    def test_imagej_spacing(self, tmp_path):
        path = tmp_path / "imagej.tif"
        pixels_per_um = 1.0 / 0.65
        _write_imagej_tiff(
            path,
            np.zeros((5, 64, 64), dtype=np.uint16),
            resolution=(pixels_per_um, pixels_per_um),
            spacing=5.0,
            unit="um",
        )
        meta = _parse_tiff_metadata(str(path))
        assert meta["z_spacing"] == pytest.approx(5.0)
        assert meta["pixelsize"] == pytest.approx(0.65, rel=1e-4)

    def test_missing_file_returns_empty(self, tmp_path):
        meta = _parse_tiff_metadata(str(tmp_path / "nonexistent.tif"))
        assert meta == {}


# ---------------------------------------------------------------------------
# _apply_table_defaults
# ---------------------------------------------------------------------------


class TestApplyTableDefaults:
    def _base_df(self, **extra_cols) -> pd.DataFrame:
        data = {"file_path": ["a.tif", "b.tif"], **extra_cols}
        return pd.DataFrame(data)

    def test_fills_all_missing_from_tiff_meta(self):
        df = self._base_df()
        meta = {
            "length_x": 128,
            "length_y": 64,
            "length_z": 5,
            "length_t": 3,
            "length_c": 2,
        }
        result = _apply_table_defaults(df, meta)
        assert list(result["length_x"]) == [128, 128]
        assert list(result["length_y"]) == [64, 64]
        assert list(result["length_z"]) == [5, 5]
        assert list(result["length_t"]) == [3, 3]
        assert list(result["length_c"]) == [2, 2]
        assert list(result["fov_name"]) == ["Image", "Image"]
        assert list(result["start_x"]) == [0.0, 0.0]
        assert list(result["start_y"]) == [0.0, 0.0]

    def test_does_not_overwrite_existing_columns(self):
        df = self._base_df(length_x=[10, 20], length_c=[1, 1])
        meta = {"length_x": 999, "length_y": 64, "length_c": 5}
        result = _apply_table_defaults(df, meta)
        assert list(result["length_x"]) == [10, 20]
        assert list(result["length_c"]) == [1, 1]
        assert list(result["length_y"]) == [64, 64]

    def test_length_c_not_filled_when_absent_from_meta(self):
        df = self._base_df()
        meta = {"length_x": 64, "length_y": 64}
        result = _apply_table_defaults(df, meta)
        assert "length_c" not in result.columns

    def test_does_not_mutate_input_dataframe(self):
        df = self._base_df()
        original_cols = set(df.columns)
        _apply_table_defaults(df, {"length_x": 64, "length_y": 64})
        assert set(df.columns) == original_cols
