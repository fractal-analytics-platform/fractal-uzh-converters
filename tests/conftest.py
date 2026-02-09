import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Regenerate assertion snapshot YAMLs",
    )


@pytest.fixture
def update_snapshots(request):
    return request.config.getoption("--update-snapshots")
