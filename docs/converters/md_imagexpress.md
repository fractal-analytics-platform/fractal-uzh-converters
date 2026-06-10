# Molecular Devices ImageXpress HCS.ai

## Expected Data Structure

The MD ImageXpress converter expects an acquisition directory containing an `.mxprotocol` file and one or more experiment subdirectories:

```
{protocol_name}_{date-time}/
├── {protocol_name}.mxprotocol
├── autofocus/
└── experiment{_mode}/
    ├── {acquisition_name}.jdce
    ├── image_metadata_1.csv
    ├── timepoint0/
    │   └── {protocol_name}_t0_C05_s0_w0_z0.tif
    └── timepoint1/
        ├── {protocol_name}_t1_C05_s0_w0_z0.tif
        ├── {protocol_name}_t1_C05_s0_w0_z1.tif
        └── ...
```

!!! info "Flexible path input"
    You can point `Path` to either the acquisition directory (`{protocol_name}_{date-time}/`) or directly to the `.mxprotocol` file. The converter handles both.

## Metadata

The converter parses two files inside the experiment directory:

- **`.jdce`** (JSON) — Experiment-level metadata: channel/wavelength configuration, pixel size, objective calibration, Z-stack parameters, and time schedule.
- **`.csv`** — Per-image records: well position (row, column), field of view index, wavelength, Z index, timepoint, stage positions, and the relative path to each TIFF file.

Channel colors are automatically assigned based on the excitation wavelength defined in the `.jdce` file.

## Experiment Modes

The microscope can produce up to three experiment directories, each containing a different type of data:

| Directory | Content |
|---|---|
| `experiment` | Standard or projection images |
| `experiment_z_stack` | Full Z-stack acquisitions |
| `experiment_montage` | Montaged / stitched images |

By default, the converter prefers `experiment_z_stack` if present, otherwise falls back to `experiment`. Use the advanced options below to select a different mode.

## Task Parameters

The MD ImageXpress init task extends the base acquisition parameters with MD-specific advanced options:

| Field | Type | Default | Description |
|---|---|---|---|
| `Path` | `str` | *required* | Path to the acquisition directory or `.mxprotocol` file. |
| `Plate Name` | `str` or `null` | `null` | Custom plate name. Defaults to the directory name. |
| `Acquisition Id` | `int` | `0` | Acquisition identifier for multi-acquisition plates. |
| `Advanced` | `MDAcquisitionOptions` | `{}` | Advanced options (see below). |

### MD-Specific Advanced Options

In addition to the [common advanced options](index.md#acquisition-options-advanced), the MD converter adds:

| Field | Type | Default | Description |
|---|---|---|---|
| `Convert Only Projections` | `bool` | `false` | Only convert projection images from the `experiment` directory, ignoring Z-stacks. |
| `Convert Montages` | `bool` | `false` | Convert montaged / stitched images from `experiment_montage` instead of individual FOVs. |

!!! warning "Incompatible options"
    `Convert Only Projections` and `Convert Montages` cannot both be enabled when the montage data contains Z-stacks.

## Multiple Acquisitions (same plate)

To combine multiple acquisitions into a single plate, use different `Acquisition Id` values while keeping the same `Plate Name`.

## Python API

```python
from fractal_uzh_converters import convert_imagexpress_hcs, MDImageXpressHCSaiAcquisitionModel

acquisitions = [
    MDImageXpressHCSaiAcquisitionModel(
        path="/path/to/imagexpress/acquisition",
        plate_name="my_plate",
        acquisition_id=0,
    )
]

convert_imagexpress_hcs(
    zarr_dir="/output/zarr",
    acquisitions=acquisitions,
)
```

See [How to Run the Converters](../how_to_run_the_converters.md) for all common parameters and execution details.
