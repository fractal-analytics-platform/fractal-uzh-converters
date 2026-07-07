import pytest
from ome_zarr_converters_tools import ConverterOptions, OmeZarrOptions
from ome_zarr_converters_tools.models._converter_options import BackendType

# The --update-snapshots / --extended options, the `extended` marker and its
# skip behaviour, and the `update_snapshots` fixture are provided by the
# ome_zarr_converters_tools.testing pytest plugin.


@pytest.fixture
def converter_options():
    return ConverterOptions(
        omezarr_options=OmeZarrOptions(
            ngff_version="0.5", table_backend=BackendType.CSV
        )
    )
