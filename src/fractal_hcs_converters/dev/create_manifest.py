"""
Generate JSON schemas for task arguments afresh, and write them
to the package manifest.
"""
from fractal_tasks_core.dev.create_manifest import create_manifest

if __name__ == "__main__":
    PACKAGE = "fractal_hcs_converters"
    AUTHORS = "Lorenzo Cerrone"
    docs_link = ""
    if docs_link:
        create_manifest(package=PACKAGE, authors=AUTHORS, docs_link=docs_link)
    else:
        create_manifest(package=PACKAGE, authors=AUTHORS)
