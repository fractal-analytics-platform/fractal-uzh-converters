import pytest
from ome_zarr_converters_tools import ConverterOptions, OmeZarrOptions
from ome_zarr_converters_tools.models._converter_options import (
    BackendType,
)


def pytest_addoption(parser):
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Regenerate assertion snapshot YAMLs",
    )
    parser.addoption(
        "--extended",
        action="store_true",
        default=False,
        help="Run extended tests requiring large test datasets",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "extended: mark test as requiring the extended test datasets"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--extended"):
        skip_marker = pytest.mark.skip(reason="Pass --extended to run extended tests")
        for item in items:
            if "extended" in item.keywords:
                item.add_marker(skip_marker)


@pytest.fixture
def update_snapshots(request):
    return request.config.getoption("--update-snapshots")


@pytest.fixture
def converter_options():
    return ConverterOptions(
        omezarr_options=OmeZarrOptions(
            ngff_version="0.5", table_backend=BackendType.CSV
        )
    )
