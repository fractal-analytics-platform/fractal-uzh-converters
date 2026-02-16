# PerkinElmer Operetta / Opera Phenix

## Expected Data Structure

The Operetta converter expects an acquisition directory containing an `Images` subdirectory with the metadata XML file and TIFF images:

```
my_acquisition/
└── Images/
    ├── Index.idx.xml        # Metadata file (required)
    ├── r01c01f01p01-ch1sk1fk1fl1.tiff
    ├── r01c01f01p01-ch2sk1fk1fl1.tiff
    ├── ...
    └── rXXcXXfXXpXX-chXskXfkXflX.tiff
```

!!! info "Flexible path input"
    You can point `Path` to either the base directory (`my_acquisition/`) or directly to the `Images` subdirectory (`my_acquisition/Images/`). The converter handles both.

## Metadata

The converter reads all image metadata from `Images/Index.idx.xml`, including:

- Well position (row and column)
- Field of view index
- Channel names and IDs
- Pixel size (from `ImageResolutionX` / `ImageResolutionY`)
- Stage positions (X, Y, Z)
- Z-plane and timepoint indices
- Bit depth (auto-detected from max intensity values)

## Task Parameters

The Operetta init task uses the base acquisition parameters with no additional fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `Path` | `str` | *required* | Path to the Operetta acquisition directory. |
| `Plate Name` | `str` or `null` | `null` | Custom plate name. Defaults to the directory name. |
| `Acquisition Id` | `int` | `0` | Acquisition identifier for multi-acquisition plates. |
| `Advanced` | `AcquisitionOptions` | `{}` | Advanced options (condition table, overrides). |

## Multiple Acquisitions (same plate)

To combine multiple acquisitions (e.g., different timepoints or imaging rounds) into a single plate, use different `Acquisition Id` values while keeping the same `Plate Name`.
