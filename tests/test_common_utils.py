"""Unit tests for common utility functions."""

import polars
import pytest

from fractal_uzh_converters.common._utils import (
    HCSBaseAcquisitionModel,
    SingleBaseAcquisitionModel,
    get_attributes_from_condition_table,
    parse_acquisitions,
)
from fractal_uzh_converters.scanr._utils import (
    _extract_well_position_id,
    _wellid_to_row_column,
)

# ---------------------------------------------------------------------------
# get_attributes_from_condition_table
# ---------------------------------------------------------------------------


def _make_table(**kwargs) -> polars.DataFrame:
    return polars.DataFrame(kwargs)


class TestGetAttributesFromConditionTable:
    def test_none_returns_empty(self):
        assert get_attributes_from_condition_table(None, "A", 1) == {}

    def test_basic_match(self):
        table = _make_table(row=["A", "B"], column=[1, 2], condition=["ctrl", "treat"])
        result = get_attributes_from_condition_table(table, "A", 1)
        assert result == {"condition": ["ctrl"]}

    def test_col_alias_instead_of_column(self):
        table = _make_table(row=["A"], col=[1], condition=["ctrl"])
        result = get_attributes_from_condition_table(table, "A", 1)
        assert result == {"condition": ["ctrl"]}

    def test_case_insensitive_row_column(self):
        table = _make_table(Row=["A"], Column=[1], label=["x"])
        result = get_attributes_from_condition_table(table, "A", 1)
        assert result == {"label": ["x"]}

    def test_missing_row_column_raises(self):
        table = _make_table(well=["A1"], condition=["ctrl"])
        with pytest.raises(ValueError, match="'row' column"):
            get_attributes_from_condition_table(table, "A", 1)

    def test_missing_column_and_col_raises(self):
        table = _make_table(row=["A"], well_col=[1], condition=["ctrl"])
        with pytest.raises(ValueError, match="'column' or 'col'"):
            get_attributes_from_condition_table(table, "A", 1)

    def test_no_match_returns_empty(self):
        table = _make_table(row=["B"], column=[2], condition=["treat"])
        result = get_attributes_from_condition_table(table, "A", 1)
        assert result == {}

    def test_acquisition_filtering(self):
        table = _make_table(
            row=["A", "A"],
            column=[1, 1],
            acquisition=[0, 1],
            condition=["ctrl", "treat"],
        )
        result0 = get_attributes_from_condition_table(table, "A", 1, acquisition=0)
        assert result0 == {"condition": ["ctrl"]}
        result1 = get_attributes_from_condition_table(table, "A", 1, acquisition=1)
        assert result1 == {"condition": ["treat"]}

    def test_placeholder_normalization(self):
        table = _make_table(
            row=["A", "B", "C", "D"],
            column=[1, 1, 1, 1],
            label=["", "Na", "N/A", "NA"],
        )
        result_a = get_attributes_from_condition_table(table, "A", 1)
        assert result_a["label"] == [None]
        result_b = get_attributes_from_condition_table(table, "B", 1)
        assert result_b["label"] == [None]
        result_c = get_attributes_from_condition_table(table, "C", 1)
        assert result_c["label"] == [None]
        result_d = get_attributes_from_condition_table(table, "D", 1)
        assert result_d["label"] == [None]

    def test_numeric_column_passthrough(self):
        table = _make_table(row=["A"], column=[1], dose=[0.5])
        result = get_attributes_from_condition_table(table, "A", 1)
        assert result == {"dose": [0.5]}

    def test_multiple_rows_same_well_both_returned(self):
        table = _make_table(
            row=["A", "A"],
            column=[1, 1],
            replicate=[1, 2],
        )
        result = get_attributes_from_condition_table(table, "A", 1)
        assert result == {"replicate": [1, 2]}

    def test_row_col_keys_excluded_from_result(self):
        table = _make_table(row=["A"], column=[1], label=["ctrl"])
        result = get_attributes_from_condition_table(table, "A", 1)
        assert "row" not in result
        assert "column" not in result


# ---------------------------------------------------------------------------
# _wellid_to_row_column
# ---------------------------------------------------------------------------


class TestWellidToRowColumn:
    @pytest.mark.parametrize(
        "layout,well_id,expected_row,expected_col",
        [
            # 96-well (8 rows x 12 cols)
            ("96-well", 1, "A", 1),
            ("96-well", 12, "A", 12),
            ("96-well", 13, "B", 1),
            ("96-well", 96, "H", 12),
            # 24-well (4 rows x 6 cols): well 1 -> A1, well 24 -> D6
            ("24-well", 1, "A", 1),
            ("24-well", 24, "D", 6),
            # 48-well (6 rows x 8 cols): well 1 -> A1, well 48 -> F8
            ("48-well", 1, "A", 1),
            ("48-well", 48, "F", 8),
            # 384-well (16 rows x 24 cols): well 1 -> A1, well 384 -> P24
            ("384-well", 1, "A", 1),
            ("384-well", 384, "P", 24),
        ],
    )
    def test_known_wells(self, layout, well_id, expected_row, expected_col):
        row, col = _wellid_to_row_column(well_id, layout)
        assert row == expected_row
        assert col == expected_col

    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            _wellid_to_row_column(97, "96-well")

    def test_out_of_bounds_384_raises(self):
        with pytest.raises(ValueError, match="out of bounds"):
            _wellid_to_row_column(385, "384-well")


# ---------------------------------------------------------------------------
# _extract_well_position_id
# ---------------------------------------------------------------------------


class TestExtractWellPositionId:
    def test_basic_extraction(self):
        (row, col), pos = _extract_well_position_id("W001P001", "96-well")
        assert row == "A"
        assert col == 1
        assert pos == 1

    def test_embedded_in_filename(self):
        (row, col), pos = _extract_well_position_id("scan_W013P003_ch1.tif", "96-well")
        assert row == "B"
        assert col == 1
        assert pos == 3

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Could not extract"):
            _extract_well_position_id("no_well_info.tif", "96-well")


# ---------------------------------------------------------------------------
# HCSBaseAcquisitionModel.normalized_plate_name
# ---------------------------------------------------------------------------


class TestNormalizedPlateName:
    def test_uses_last_path_component(self):
        model = HCSBaseAcquisitionModel(path="/data/experiment")
        assert model.normalized_plate_name == "experiment"

    def test_strips_trailing_slash(self):
        model = HCSBaseAcquisitionModel(path="/data/experiment/")
        assert model.normalized_plate_name == "experiment"

    def test_explicit_plate_name_overrides(self):
        model = HCSBaseAcquisitionModel(path="/data/experiment", plate_name="MyPlate")
        assert model.normalized_plate_name == "MyPlate"


# ---------------------------------------------------------------------------
# SingleBaseAcquisitionModel.normalized_image_name
# ---------------------------------------------------------------------------


class TestNormalizedImageName:
    def test_uses_last_path_component(self):
        model = SingleBaseAcquisitionModel(path="/data/stack.tif")
        assert model.normalized_image_name == "stack.tif"

    def test_strips_trailing_slash(self):
        model = SingleBaseAcquisitionModel(path="/data/images/")
        assert model.normalized_image_name == "images"

    def test_explicit_image_name_overrides(self):
        model = SingleBaseAcquisitionModel(path="/data/stack.tif", image_name="MyImage")
        assert model.normalized_image_name == "MyImage"


# ---------------------------------------------------------------------------
# parse_acquisitions — error paths
# ---------------------------------------------------------------------------


def _make_converter_options():
    from ome_zarr_converters_tools import BackendType, ConverterOptions, OmeZarrOptions

    return ConverterOptions(
        omezarr_options=OmeZarrOptions(
            ngff_version="0.5", table_backend=BackendType.CSV
        )
    )


def _dummy_parser(*, acquisition_model, converter_options):
    return []


class TestParseAcquisitions:
    def test_empty_acquisitions_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_acquisitions(
                parse_function=_dummy_parser,
                acquisitions=[],
                converter_options=_make_converter_options(),
            )

    def test_all_empty_results_raises(self):
        acq = HCSBaseAcquisitionModel(path="/some/path")
        with pytest.raises(ValueError, match="No images found"):
            parse_acquisitions(
                parse_function=_dummy_parser,
                acquisitions=[acq],
                converter_options=_make_converter_options(),
            )
