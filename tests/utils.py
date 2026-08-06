from pathlib import Path

from ome_zarr_converters_tools.testing import build_snapshot

DATA_DIR = Path(__file__).parent / "data"
EXTENDED_DATA_DIR = Path(__file__).parent / "data-extended"


def channel_metadata(
    zarr_dir: Path, image_list_updates: list[dict]
) -> dict[str, tuple[list[str], list[str]]]:
    """Return `{image_path: (channel_labels, wavelength_ids)}` for a plate run."""
    snapshot = build_snapshot(
        zarr_dir=zarr_dir,
        image_list_updates=image_list_updates,
        output_type="plate",
    )
    return {
        image_path: (image.channel_labels, image.wavelength_ids)
        for plate in snapshot.plates.values()
        for image_path, image in plate.images.items()
    }
