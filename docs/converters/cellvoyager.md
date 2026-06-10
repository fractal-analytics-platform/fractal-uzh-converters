# Yokogawa CellVoyager

## Expected Data Structure

The CellVoyager converter expects an acquisition directory containing the measurement metadata files and image files:

```
my_acquisition/
├── MeasurementData.mlf      # Image measurement records (required)
├── MeasurementDetail.mrf    # Acquisition details and channel info (required)
├── image_001.png
├── image_002.png
└── ...
```

The image file paths are referenced inside `MeasurementData.mlf` (with `.tif` extension) and can be in subdirectories relative to the acquisition directory. The actual files may use `.png` or `.tif` extension — select the matching extension via the `image_extension` parameter.

## Metadata

The converter parses two XML files:

- **`MeasurementData.mlf`** — Contains one record per acquired image tile, including well position (row, column), field index, channel, Z-index, timepoint, stage coordinates (X, Y, Z), and the relative path to the image file.
- **`MeasurementDetail.mrf`** — Contains acquisition-level metadata: pixel dimensions, number of channels, rows/columns/fields/Z-planes/timepoints, and channel details (pixel size, bit depth).

## Z-Image Processing

Unlike the [CQ3K converter](cq3k.md), the CellVoyager converter does **not** support Z-image processing types (e.g., `focus`, `maximum_projection`). A single plate is always produced per acquisition.

## Task Parameters

The CellVoyager init task extends the base acquisition parameters with one additional field:

| Field | Type | Default | Description |
|---|---|---|---|
| `Path` | `str` | *required* | Path to the CellVoyager acquisition directory. |
| `Plate Name` | `str` or `null` | `null` | Custom plate name. Defaults to the directory name. |
| `Acquisition Id` | `int` | `0` | Acquisition identifier for multi-acquisition plates. |
| `Image Extension` | `"png"` or `"tif"` | `"png"` | File extension of the actual image files. The metadata always references `.tif`, but actual files may be `.png` or `.tif`. |
| `Advanced` | `AcquisitionOptions` | `{}` | Advanced options (condition table, overrides). |

!!! warning "Limited Testing"
    This converter has been tested on a limited set of acquisitions. It may not work correctly on all Yokogawa CellVoyager datasets.

## Python API

```python
from fractal_uzh_converters import convert_cellvoyager, CellVoyagerAcquisitionModel

acquisitions = [
    CellVoyagerAcquisitionModel(
        path="/path/to/cellvoyager/acquisition",
        plate_name="my_plate",
        acquisition_id=0,
        image_extension=".tif",
    )
]

convert_cellvoyager(
    zarr_dir="/output/zarr",
    acquisitions=acquisitions,
)
```

See [How to Run the Converters](../how_to_run_the_converters.md) for all common parameters and execution details.
