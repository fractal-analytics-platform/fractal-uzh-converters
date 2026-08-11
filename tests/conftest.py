import pytest
from ome_zarr_converters_tools import BackendType, ConverterOptions, OmeZarrOptions

# Load the shared snapshot-testing plugin: the --update-snapshots / --extended
# options, the `extended` marker and its skip behaviour, and the
# `update_snapshots` fixture.
pytest_plugins = ["ome_zarr_converters_tools.testing.plugin"]


def pytest_configure(config):
    # The converters' own warnings describe the input data and fire on healthy
    # runs (no CV8000 acquisition ships a `.wpi`), so the "error" filter would
    # fail the suite on them. `pytest.warns` installs its own filter and is
    # unaffected.
    #
    # The ignore is registered here rather than in the pytest ini because pytest
    # resolves a filter's category by importing the module naming it, and it
    # does so from `pytest_load_initial_conftests` — a hookwrapper that runs
    # before pytest-cov starts coverage. Naming a `fractal_uzh_converters.*`
    # category in the ini therefore imports the whole package unmeasured
    # (coverage's "module-not-measured"), reporting every module-level statement
    # as a miss. `pytest_configure` runs after coverage has started.
    config.addinivalue_line(
        "filterwarnings",
        "ignore::fractal_uzh_converters.common.ConverterWarning",
    )


@pytest.fixture
def converter_options():
    return ConverterOptions(
        omezarr_options=OmeZarrOptions(
            ngff_version="0.5", table_backend=BackendType.CSV
        )
    )
