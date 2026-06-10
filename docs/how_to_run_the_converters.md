# How to Run the Converters

The converters in this package can be used in two ways:

- **As Fractal tasks** — configured and executed via the [Fractal Analytics Platform](https://fractal-analytics-platform.github.io/) web interface or API.
- **As Python functions** — called directly from your own scripts or Jupyter notebooks.

Both modes accept the same parameters; the Python API is a thin wrapper around the same underlying logic as the Fractal tasks.

## Running via Python API

You can also run any converter as a plain Python function. This is useful for scripting, local testing, or integrating the conversion into your own pipelines.

### Installation

```bash
pip install fractal-uzh-converters
```

### Import Pattern

```python
from fractal_uzh_converters import convert_<microscope>, <Microscope>AcquisitionModel
```

All converters and acquisition models are exported from the top-level package:

```python
from fractal_uzh_converters import (
    # Converters
    convert_operetta,
    convert_scanr,
    convert_cq3k,
    convert_cellvoyager,
    convert_imagexpress_hcs,
    convert_hcs_tiff,
    convert_single_tiff,
    # Acquisition models
    OperettaAcquisitionModel,
    ScanRAcquisitionModel,
    CQ3KAcquisitionModel,
    CellVoyagerAcquisitionModel,
    MDImageXpressHCSaiAcquisitionModel,
    MDAcquisitionOptions,
    HcsTiffAcquisitionModel,
    SingleTiffAcquisitionModel,
)
```

### Example (Operetta)

```python
from fractal_uzh_converters import convert_operetta, OperettaAcquisitionModel

acquisitions = [
    OperettaAcquisitionModel(
        path="/path/to/operetta/acquisition",
        plate_name="my_plate",
        acquisition_id=0,
    )
]

images = convert_operetta(
    zarr_dir="/output/zarr",
    acquisitions=acquisitions,
)
```

The function returns a list of `ImageListUpdateDict` objects describing the converted OME-Zarr images.

### Common Parameters

All functions share the same signature:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `zarr_dir` | `str` | *required* | Output directory where the OME-Zarr plate(s) will be written. |
| `acquisitions` | `list[<Model>]` | *required* | List of acquisition objects. Type varies by converter — see each converter page. |
| `converter_options` | `ConverterOptions \| None` | `None` | Advanced options (tiling, writer mode, chunking, OME-Zarr format). `None` uses the defaults. |
| `overwrite` | `OverwriteMode` | `NO_OVERWRITE` | What to do if the output already exists. |
| `runner` | `RunnerType \| None` | `None` | Execution strategy. `None` runs wells sequentially. |

### Multiple Acquisitions

Pass multiple acquisition objects to convert them all into a single run. To combine them into one plate (e.g. for multiplexed experiments), use the same `plate_name` with different `acquisition_id` values:

```python
from fractal_uzh_converters import convert_operetta, OperettaAcquisitionModel

acquisitions = [
    OperettaAcquisitionModel(
        path="/data/round1",
        plate_name="my_plate",
        acquisition_id=0,
    ),
    OperettaAcquisitionModel(
        path="/data/round2",
        plate_name="my_plate",
        acquisition_id=1,
    ),
]

convert_operetta(zarr_dir="/output/zarr", acquisitions=acquisitions)
```

### Per-Converter Examples

Each converter page includes a Python API example with the microscope-specific acquisition model:

- [Revvity Operetta / Opera Phenix](converters/operetta.md#python-api)
- [Evident ScanR](converters/scanr.md#python-api)
- [Yokogawa CQ3K](converters/cq3k.md#python-api)
- [Yokogawa CellVoyager](converters/cellvoyager.md#python-api)
- [Molecular Devices ImageXpress HCS.ai](converters/md_imagexpress.md#python-api)
- [Custom TIFF](converters/custom_tiff.md#python-api)
