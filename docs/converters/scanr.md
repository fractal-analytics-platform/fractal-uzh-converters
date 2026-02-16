# Olympus ScanR

## Expected Data Structure

The ScanR converter expects an acquisition directory containing a `data` subdirectory with the OME-XML metadata file and TIFF images:

```
my_acquisition/
└── data/
    ├── metadata.ome.xml     # OME-XML metadata file (required)
    ├── image_001.tiff
    ├── image_002.tiff
    ├── ...
    └── image_NNN.tiff
```

!!! info "Flexible path input"
    You can point `Path` to either the base directory (`my_acquisition/`) or directly to the `data` subdirectory (`my_acquisition/data/`). The converter handles both.

## Metadata

The converter parses the OME-XML metadata file using the `ome_types` library. It extracts:

- Well and position IDs from image names (pattern: `W<well_id>P<position_id>`)
- Channel names
- Pixel size (physical size X/Y)
- Stage positions per plane (X, Y, Z)
- Z-spacing (computed from plane positions)
- Timepoint information

## Well ID Mapping

ScanR uses sequential well IDs (W1, W2, ...) that the converter maps to row/column format based on the plate layout:

| Layout | Rows | Columns | Example: W1 = A01, W13 = B01 (96-well) |
|---|---|---|---|
| 24-well | 4 (A-D) | 6 | W1 = A01, W7 = B01 |
| 48-well | 6 (A-F) | 8 | W1 = A01, W9 = B01 |
| 96-well | 8 (A-H) | 12 | W1 = A01, W13 = B01 |
| 384-well | 16 (A-P) | 24 | W1 = A01, W25 = B01 |

## Task Parameters

The ScanR init task adds a `Layout` field to the base acquisition parameters:

| Field | Type | Default | Description |
|---|---|---|---|
| `Path` | `str` | *required* | Path to the ScanR acquisition directory. |
| `Plate Name` | `str` or `null` | `null` | Custom plate name. Defaults to the directory name. |
| `Acquisition Id` | `int` | `0` | Acquisition identifier for multi-acquisition plates. |
| `Layout` | `str` | `"96-well"` | Plate layout. One of: `24-well`, `48-well`, `96-well`, `384-well`. |
| `Advanced` | `AcquisitionOptions` | `{}` | Advanced options (condition table, overrides). |

!!! warning "Plate layout"
    Make sure to set the correct `Layout` for your plate. The well ID mapping depends on it — using the wrong layout will assign images to incorrect wells.

