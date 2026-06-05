"""Register a collection setup handler for ``SingleImage`` outputs.

``ome_zarr_converters_tools`` ships with a built-in handler for ``ImageInPlate``
only. ``SingleImage`` outputs do not need an upfront skeleton (the zarr group
is created in the compute task), so the handler only enforces the
``OverwriteMode`` contract.
"""

from ome_zarr_converters_tools import (
    OverwriteMode,
    SingleImage,
    TiledImage,
    filesystem_for_url,
    join_url_paths,
)
from ome_zarr_converters_tools.pipelines import add_collection_handler


def setup_singleimage(
    zarr_dir: str,
    tiled_images: list[TiledImage],
    ngff_version=None,
    overwrite_mode: OverwriteMode = OverwriteMode.NO_OVERWRITE,
) -> None:
    """Set up a SingleImage collection (overwrite-mode enforcement only)."""
    for tiled_image in tiled_images:
        collection = tiled_image.collection
        if not isinstance(collection, SingleImage):
            raise ValueError(f"Expected SingleImage collection, got {type(collection)}")
        zarr_url = join_url_paths(zarr_dir, collection.path())
        if overwrite_mode == OverwriteMode.NO_OVERWRITE:
            fs = filesystem_for_url(zarr_url)
            if fs.exists(zarr_url):
                raise FileExistsError(
                    f"A zarr already exists at {zarr_url} "
                    f"(overwrite_mode={OverwriteMode.NO_OVERWRITE.value}). "
                    f"Set overwrite_mode={OverwriteMode.OVERWRITE.value} to "
                    f"replace it."
                )


try:
    add_collection_handler(
        function=setup_singleimage,
        collection_type="SingleImage",
    )
except ValueError:
    # Already registered (e.g. re-import); safe to ignore.
    pass
