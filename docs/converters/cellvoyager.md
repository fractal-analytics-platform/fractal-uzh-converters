# Yokogawa CellVoyager

## Expected Data Structure

The CellVoyager converter expects an acquisition directory containing the measurement metadata files and image files:

```
my_acquisition/
├── MeasurementData.mlf      # Image measurement records (required)
├── MeasurementDetail.mrf    # Acquisition details and channel info (required)
├── protocol.mes             # Acquisition protocol (optional, see Channels)
├── plate.wpi                # Plate definition (optional)
├── plate.wpp                # Plate product (optional)
├── image_001.png
├── image_002.png
└── ...
```

The image file paths are referenced inside `MeasurementData.mlf` (with `.tif` extension) and can be in subdirectories relative to the acquisition directory. The actual files may use `.png` or `.tif` extension — select the matching extension via the `image_extension` parameter.

The `.mes`, `.wpi` and `.wpp` filenames vary per acquisition and are read out of the `.mrf` — the names above are only examples.

## Metadata

The converter parses up to three XML files:

- **`MeasurementData.mlf`** — Contains one record per acquired image tile, including well position (row, column), field index, channel, Z-index, timepoint, stage coordinates (X, Y, Z), and the relative path to the image file.
- **`MeasurementDetail.mrf`** — Contains acquisition-level metadata: pixel dimensions, number of channels, rows/columns/fields/Z-planes/timepoints, and channel details (pixel size, bit depth).
- **the `.mes` protocol file** — Contains the channel definitions. Its filename varies per acquisition and is read out of the `.mrf`. Optional; see [Channels](#channels).

### Copied into the plate

The converter models only the fraction of the vendor metadata it needs, so the five plate-level files are also copied verbatim into `<plate>.zarr/metadata/`:

```
MyPlate.zarr/
└── metadata/
    ├── MeasurementData.mlf
    ├── MeasurementDetail.mrf
    ├── protocol.mes
    ├── plate.wpi
    └── plate.wpp
```

The `.mlf` and `.mrf` are copied under their fixed names; the `.mes`, `.wpi` and `.wpp` under the names recorded inside the `.mrf`. Filenames are preserved exactly, spaces and non-ASCII characters included.

- **Every plate the acquisition produced gets its own copy**, so each [projection plate](#z-image-processing) carries the same five files as the unsuffixed one.
- **A file the acquisition does not ship is a warning, not an error.** The conversion itself is unaffected. This is a common case rather than a defect: CV8000 acquisitions routinely name a `.wpi` in the `.mrf` that was never written.
- When several acquisitions land in the same plate with identically named files, the second copy becomes `<stem>_acq{acquisition_id}<ext>` rather than overwriting the first. A byte-identical copy is left alone, so re-running a conversion does not accumulate duplicates.

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

### With projection plates

One list covers the **whole acquisition**, every plate included, and it is numbered in the acquisition-wide channel space — not per plate. Because each [projection algorithm](#z-image-processing) carries its own channel numbers, the entries are split across plates rather than repeated in each.

> An acquisition that min-projects `Ch1` and max-projects `Ch2`–`Ch5` needs one 5-entry list. Element 0 surfaces in `MyPlate_MinIP.zarr`; elements 1–4 surface in `MyPlate_MIP.zarr`.

So do not write a separate list per plate, and do not size the list by the channels visible in one plate. Selecting a single plate with `Z Processing` does relax the requirement to that plate's own channels.

## Z-Image Processing

A CellVoyager acquisition can write projections alongside — or instead of — the raw Z slices, tagged `bts:ZImageProcessing` in the `.mlf`. Each projection algorithm carries its own channel numbers, so the converter writes one plate per algorithm, suffixing the plate name:

| `ZImageProcessing` | Plate suffix |
|---|---|
| *(absent — raw Z slices)* | *(none)* |
| `Maximum` | `_MIP` |
| `Minimum` | `_MinIP` |
| `Sum` | `_SIP` |

Any other value is used verbatim as the suffix, with a warning.

The suffixed plates coexist with the unsuffixed one. An acquisition named `MyPlate` that stores raw slices plus a maximum and a minimum projection produces:

- `MyPlate.zarr` — the raw Z slices
- `MyPlate_MIP.zarr`
- `MyPlate_MinIP.zarr`

The split follows the acquisition, not the channel set: an acquisition that max-projects its fluorescence channels and min-projects a brightfield channel yields a `_MIP` plate and a `_MinIP` plate, each holding only its own channels.

Channel labels do **not** carry the algorithm — it is already in the plate name — so a channel keeps the same label across the raw and the projection plates.

Most CellVoyager acquisitions carry no `bts:ZImageProcessing` at all and produce the single unsuffixed plate they always have.

### Converting only some of them

`Advanced` → `Z Processing` selects which of these plates are written. Leave it unset — the default — to convert every kind the acquisition contains. Otherwise it is one switch per kind:

| Switch | Default | Selects |
|---|---|---|
| `Z slices` | on | the unsuffixed raw-stack plate |
| `MIP` | off | `_MIP` |
| `MinIP` | off | `_MinIP` |
| `SIP` | off | `_SIP` |

Note that `Z slices` is the only one on by default, so enabling a projection *adds* to the raw stack rather than replacing it:

```python
advanced={"z_processing": {"mip": True}}                     # Z slices + MIP
advanced={"z_processing": {"z_slices": False, "mip": True}}  # MIP only
advanced={"z_processing": {}}                                # Z slices only
```

| Selection | Result |
|---|---|
| unset | Every kind the acquisition contains. |
| nothing enabled | An error — enable something, or leave the option unset. |
| all enabled kinds present | Exactly those. |
| some enabled kinds absent | The ones that matched, plus a warning naming the rest. One selection stays usable across a batch of mixed acquisitions. |
| no enabled kind present | An error naming what the acquisition actually contains. |

A `ZImageProcessing` value outside the four above cannot be named here. To single one out, use a `Path Regex Filter` on the plate name instead:

```python
advanced={"filters": [
    {"name": "Path Regex Filter", "mode": "Exclude", "regex": r"_Median\.zarr"}
]}
```

## Task Parameters

The CellVoyager init task extends the base acquisition parameters with one additional field:

| Field | Type | Default | Description |
|---|---|---|---|
| `Path` | `str` | *required* | Path to the CellVoyager acquisition directory. |
| `Plate Name` | `str` or `null` | `null` | Custom plate name. Defaults to the directory name. |
| `Acquisition Id` | `int` | `0` | Acquisition identifier for multi-acquisition plates. |
| `Image Extension` | `"png"` or `"tif"` | `"png"` | File extension of the actual image files. The metadata always references `.tif`, but actual files may be `.png` or `.tif`. |
| `Advanced` | `CellVoyagerAcquisitionOptions` | `{}` | Advanced options (condition table, [channel overrides](#channels), [`Z Processing`](#converting-only-some-of-them), filters). |

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
