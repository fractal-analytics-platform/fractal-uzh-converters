# Yokogawa CellVoyager

## Expected Data Structure

The CellVoyager converter expects an acquisition directory containing the measurement metadata files and image files:

```
my_acquisition/
├── MeasurementData.mlf      # Image measurement records (required)
├── MeasurementDetail.mrf    # Acquisition details and channel info (required)
├── protocol.mes             # Acquisition protocol (optional, see Channels)
├── image_001.png
├── image_002.png
└── ...
```

The image file paths are referenced inside `MeasurementData.mlf` (with `.tif` extension) and can be in subdirectories relative to the acquisition directory. The actual files may use `.png` or `.tif` extension — select the matching extension via the `image_extension` parameter.

## Metadata

The converter parses up to three XML files:

- **`MeasurementData.mlf`** — Contains one record per acquired image tile, including well position (row, column), field index, channel, Z-index, timepoint, stage coordinates (X, Y, Z), and the relative path to the image file.
- **`MeasurementDetail.mrf`** — Contains acquisition-level metadata: pixel dimensions, number of channels, rows/columns/fields/Z-planes/timepoints, and channel details (pixel size, bit depth).
- **the `.mes` protocol file** — Contains the channel definitions. Its filename varies per acquisition and is read out of the `.mrf`. Optional; see [Channels](#channels).

## Channels

Channel labels, display colours and wavelength ids come from the acquisition's `.mes` protocol file, whose filename is recorded in the `.mrf`. The label is the channel's `Target` (e.g. `405`, `DAPI`), and the wavelength id is `A{action}_C{channel}` — for example `A01_C04`. When no `.mes` is available the label falls back to the wavelength id.

To override them, fill in `Advanced` → `Channels`. **The list is ordered by the instrument's channel number**: element 0 is `Ch1`, element 1 is `Ch2`, and so on across the full channel range the protocol declares — it is *not* a dense list of the channels you can see in the output.

That distinction matters whenever an acquisition uses a subset of the instrument's channels, which on the CellVoyager includes the case where different wells acquire different channels:

> A 5-channel instrument, where one well acquires only `Ch1` and another only `Ch4`. The override still needs **four** entries, and it is elements 0 and 3 that end up on the two wells. A 2-entry list — one per channel actually visible — is rejected.

The list must have at least as many entries as the highest channel number the acquisition uses. If it is too short the conversion fails with an error naming the number required, so running once and reading the error is the quickest way to find it. Beyond that:

- Extra entries are discarded.
- Entries past the end of your list keep their `.mes` metadata.
- Leaving `Wavelength ID` empty keeps the computed `A{action}_C{channel}`.
- Leaving the colour on `Auto` keeps the colour from the `.mes`.

## Z-Image Processing

Unlike the [CQ3K converter](cq3k.md), the CellVoyager converter does **not** support Z-image processing types. A single plate is always produced per acquisition.

## Task Parameters

The CellVoyager init task extends the base acquisition parameters with one additional field:

| Field | Type | Default | Description |
|---|---|---|---|
| `Path` | `str` | *required* | Path to the CellVoyager acquisition directory. |
| `Plate Name` | `str` or `null` | `null` | Custom plate name. Defaults to the directory name. |
| `Acquisition Id` | `int` | `0` | Acquisition identifier for multi-acquisition plates. |
| `Image Extension` | `"png"` or `"tif"` | `"png"` | File extension of the actual image files. The metadata always references `.tif`, but actual files may be `.png` or `.tif`. |
| `Advanced` | `AcquisitionOptions` | `{}` | Advanced options (condition table, [channel overrides](#channels), filters). |

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
