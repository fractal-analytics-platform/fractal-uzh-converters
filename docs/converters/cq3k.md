# Yokogawa CQ3K

## Expected Data Structure

The CQ3K converter expects an acquisition directory containing the measurement metadata files and TIFF images:

```
my_acquisition/
├── MeasurementData.mlf      # Image measurement records (required)
├── MeasurementDetail.mrf    # Acquisition details and channel info (required)
└── <subdirectories>/
    ├── image_001.tif
    ├── image_002.tif
    ├── ...
    └── image_NNN.tif
```

The TIFF file paths are referenced inside `MeasurementData.mlf` and can be in subdirectories relative to the acquisition directory.

## Metadata

The converter parses two XML files:

- **`MeasurementData.mlf`** — Contains one record per acquired image tile, including well position (row, column), field index, channel, Z-index, timepoint, stage coordinates (X, Y, Z), and the relative path to the TIFF file.
- **`MeasurementDetail.mrf`** — Contains acquisition-level metadata: pixel dimensions, number of channels, rows/columns/fields/Z-planes/timepoints, and channel details (pixel size, bit depth).

## Z-Image Processing

Some CQ3K acquisitions produce multiple image types per Z-stack (e.g., `focus`, `maximum_projection`). The converter groups these into separate plates automatically, using the `z_image_processing` value as a suffix on the plate name.

For example, an acquisition named `MyPlate` with two Z-processing types will produce:

- `MyPlate_focus.zarr`
- `MyPlate_maximum_projection.zarr`

If no Z-image processing is present, a single plate is created without any suffix.

## Task Parameters

The CQ3K init task uses the base acquisition parameters with no additional fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `Path` | `str` | *required* | Path to the CQ3K acquisition directory. |
| `Plate Name` | `str` or `null` | `null` | Custom plate name. Defaults to the directory name. |
| `Acquisition Id` | `int` | `0` | Acquisition identifier for multi-acquisition plates. |
| `Advanced` | `AcquisitionOptions` | `{}` | Advanced options (condition table, overrides). |

