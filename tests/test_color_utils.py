import pytest
from ome_zarr_converters_tools.fractal._models import DefaultColor

from fractal_uzh_converters.imagexpress_hcs.color_utils import (
    wavelength_to_default_color,
)


@pytest.mark.parametrize(
    "wavelength, expected_color",
    [
        # UV -> magenta
        (300, DefaultColor.Magenta),
        (350, DefaultColor.Magenta),
        # Violet/Blue
        (380, DefaultColor.Blue),
        (405, DefaultColor.Blue),
        (449, DefaultColor.Blue),
        # Cyan
        (450, DefaultColor.Cyan),
        (488, DefaultColor.Cyan),
        # Green
        (495, DefaultColor.Green),
        (510, DefaultColor.Green),
        # Lime
        (520, DefaultColor.Lime),
        (540, DefaultColor.Lime),
        # Yellow
        (560, DefaultColor.Yellow),
        (575, DefaultColor.Yellow),
        # Orange
        (590, DefaultColor.Orange),
        (610, DefaultColor.Orange),
        # Red
        (620, DefaultColor.Red),
        (680, DefaultColor.Red),
        (749, DefaultColor.Red),
        # Far red/IR -> magenta
        (750, DefaultColor.Magenta),
        (800, DefaultColor.Magenta),
    ],
)
def test_wavelength_to_default_color(wavelength: float, expected_color: DefaultColor):
    assert wavelength_to_default_color(wavelength) == expected_color
