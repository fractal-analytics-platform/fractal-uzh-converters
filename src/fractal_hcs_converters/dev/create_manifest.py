"""
Generate JSON schemas for task arguments afresh, and write them
to the package manifest.
"""

from fractal_tasks_core.dev.create_manifest import create_manifest

custom_pydantic_models = [
    (
        "fractal_hcs_converters",
        "olympus_scanr/convert_scanr_init_task.py",
        "AcquisitionInputModel",
    ),
    (
        "fractal_hcs_converters",
        "olympus_scanr/convert_scanr_init_task.py",
        "AdvancedOptions",
    ),
]


if __name__ == "__main__":
    PACKAGE = "fractal_hcs_converters"
    AUTHORS = "Lorenzo Cerrone"
    docs_link = "https://github.com/fractal-analytics-platform/fractal-hcs-converters"
    create_manifest(
        package=PACKAGE,
        authors=AUTHORS,
        docs_link=docs_link,
        custom_pydantic_models=custom_pydantic_models,
    )
