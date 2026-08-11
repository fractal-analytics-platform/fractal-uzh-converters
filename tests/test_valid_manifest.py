import json

import requests
from jsonschema import validate

from . import MANIFEST, TASK_LIST

# The Fractal form renders the acquisition fields in schema order, so the order
# is part of the task's public surface. It is also fragile: pydantic keeps a
# redeclared field in the base class's slot, so moving `advanced` back onto
# `BaseAcquisitionModel` would silently pull it up to slot #2 in all seven
# models at once, even though each one declares it last. `z_processing` is
# top-level rather than an advanced option and must stay ahead of `advanced`
# too, so a converter-specific field cannot drift behind it.
EXPECTED_ACQUISITION_FIELD_ORDER = {
    "ScanRAcquisitionModel": [
        "path",
        "plate_name",
        "acquisition_id",
        "layout",
        "advanced",
    ],
    "CellVoyagerAcquisitionModel": [
        "path",
        "plate_name",
        "acquisition_id",
        "image_extension",
        "z_processing",
        "advanced",
    ],
    "CQ3KAcquisitionModel": [
        "path",
        "plate_name",
        "acquisition_id",
        "z_processing",
        "advanced",
    ],
    "OperettaAcquisitionModel": ["path", "plate_name", "acquisition_id", "advanced"],
    "MDImageXpressHCSaiAcquisitionModel": [
        "path",
        "plate_name",
        "acquisition_id",
        "z_processing",
        "advanced",
    ],
    "HcsTiffAcquisitionModel": ["path", "plate_name", "acquisition_id", "advanced"],
    "SingleTiffAcquisitionModel": ["path", "image_name", "advanced"],
}


def test_acquisition_field_order():
    """`advanced` must be the last field of every acquisition model."""
    found = {
        name: list(schema["properties"])
        for task in TASK_LIST
        for name, schema in task["args_schema_non_parallel"]["$defs"].items()
        if name.endswith("AcquisitionModel")
    }
    assert found == EXPECTED_ACQUISITION_FIELD_ORDER


def test_valid_manifest(tmp_path):
    """
    NOTE: to avoid adding a fractal-server dependency, we simply download the
    relevant file.
    """
    # Download JSON Schema for ManifestV2
    url = (
        "https://raw.githubusercontent.com/fractal-analytics-platform/"
        "fractal-server/main/"
        "fractal_server/json_schemas/manifest_v2.json"
    )
    r = requests.get(url)
    with (tmp_path / "manifest_schema.json").open("wb") as f:
        f.write(r.content)
    with (tmp_path / "manifest_schema.json").open("r") as f:
        manifest_schema = json.load(f)

    validate(instance=MANIFEST, schema=manifest_schema)
