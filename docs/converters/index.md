# Converters Overview

All converters in this package follow the same Fractal Compound Task structure and share common parameters. This page provides an overview of the shared parameters and how the converters work, as well as links to the individual guides for each supported microscope.

## Main Parameters

All init tasks accept the following parameters:

| Parameter | Type | Description |
|---|---|---|
| `Acquisitions` | `list` | List of acquisition objects (microscope-specific, see below). |
| `Converter Options` | `ConverterOptions` | Advanced converter options (tiling, registration, writer mode). Defaults are usually fine. |
| `Overwrite` | `OverwriteMode` | What to do if output already exists: `No Overwrite` (default), `Overwrite`, or `Extend`. |

## Acquisition Parameters

Every acquisition object starts with `Path` and ends with `Advanced`, whatever converter-specific fields come in between:

| Field | Type | Default | Description |
|---|---|---|---|
| `Path` | `str` | *required* | Path to the raw acquisition directory or file. |
| `Advanced` | `AcquisitionOptions` | `{}` | Advanced options including `Condition Table Path` and acquisition detail overrides. See [Acquisition Options (Advanced)](#acquisition-options-advanced) below. |

### HCS (plate) acquisitions

HCS converters add the following plate-specific fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `Plate Name` | `str` or `null` | `null` | Custom plate name. If not set, the directory name is used. |
| `Acquisition Id` | `int` | `0` | Identifies the acquisition when combining multiple acquisitions into one plate. |

Some HCS converters add further fields — `Layout` for ScanR, `Image Extension` for CellVoyager, and `Z Processing` for CQ3K, CellVoyager and MD ImageXpress. See the individual converter pages for details.

If multiple acquisitions need to be combined into a single plate, simply provide multiple acquisition objects with the same `Plate Name`, but different `Acquisition Id` values (e.g. in case of multiplexed experiments).

### Single-image acquisitions

Single-image converters (e.g., Custom TIFF Single Image) add:

| Field | Type | Default | Description |
|---|---|---|---|
| `Image Name` | `str` or `null` | `null` | Custom output OME-Zarr image name. If not set, the directory or file name is used. |

### Acquisition Options (Advanced)

The `Advanced` field on each acquisition allows per-acquisition overrides and filtering. Most users only need `Condition Table Path` here.

| Field | Type | Default | Description |
|---|---|---|---|
| `Channels` | `list[ChannelInfoUI]` or `null` | `null` | Override channel names, wavelength ids and colors. The list is positional, and how it is indexed depends on the converter — see the [CQ3K](cq3k.md#channels) and [CellVoyager](cellvoyager.md#channels) pages, where it is indexed by the instrument channel number rather than by the channels present in the output. |
| `Pixel Size Information` | `PixelSizeModel` or `null` | `null` | Override pixel size (`Pixelsize`, `Z Spacing`, `T Spacing` in micrometers). |
| `Condition Table Path` | `str` or `null` | `null` | Absolute path to a [condition table](../condition_tables.md) CSV file. |
| `Axes` | `str` or `null` | `null` | Override axes string (e.g., `"czyx"`). |
| `Data Type` | `DataTypeEnum` | `"autodetect"` | Pixel data type of the output: `autodetect`, `uint8`, `uint16` or `uint32`. `autodetect` infers it from the input images. |
| `Stage Orientation` | `StageOrientation` | `{}` | Flip or swap stage axes (see below). |
| `Filters` | `list` | `[]` | Filters selecting which tiles are converted (see below). |

#### Stage Orientation

If the microscope stage orientation does not match the expected coordinate system, you can apply corrections:

| Field | Type | Default | Description |
|---|---|---|---|
| `Flip X` | `bool` | `false` | Flip positions along the X axis. |
| `Flip Y` | `bool` | `false` | Flip positions along the Y axis. |
| `Swap XY` | `bool` | `false` | Swap the X and Y axes. |

#### Filters

`Filters` is a list of filter objects, each identified by its `Name`. Most of them take a `Mode` of `Include` (keep only the matching tiles) or `Exclude` (drop them), defaulting to `Include`:

| Name | Mode | Selects on |
|---|---|---|
| `Path Regex Filter` | yes | `Regex` matched against the tile's file path. |
| `Well Filter` | yes | `Wells` — a list of well IDs (e.g. `["A1", "B2"]`). Plates only. |
| `Acquisition Filter` | yes | `Acquisitions` — a list of acquisition indices. Plates only. |
| `Channel Filter` | yes | `Channel Labels` — a list of channel labels. |
| `FOV Name Filter` | yes | `Regex` matched against the field of view name. |
| `Attribute Filter` | yes | `Key` and `Values` — any tile attribute. |
| `Z Range Filter` | no | `Min Z` / `Max Z` — keeps tiles whose starting Z position lies in the range. |
| `Time Range Filter` | no | `Min T` / `Max T` — keeps tiles whose starting time point lies in the range. |

In the Python API a filter is a plain dict:

```python
advanced={"filters": [
    {"name": "Path Regex Filter", "mode": "Exclude", "regex": r"_Median\.zarr"},
    {"name": "Well Filter", "mode": "Include", "wells": ["A1", "B2"]},
]}
```

## Converter Options

The `Converter Options` parameter controls how tiles are assembled, written, and stored. The defaults work well for most cases — only adjust these if you have specific requirements. It has five sections: `Writer Mode`, `Grouping`, `Stage Position Corrections`, `OME-Zarr Options` and `Runtime Settings`.

### Grouping

Controls how individual fields of view (FOVs) are turned into output images.

| Mode | Description |
|---|---|
| `Mosaic` (default) | Aggregate all FOVs of an acquisition into one OME-Zarr image, arranged by the nested `Tiling Strategy`. |
| `Per-FOV` | Write each FOV as its own OME-Zarr image — no mosaic, and no tiling strategy. |

Under `Mosaic`, `Tiling Strategy` picks how the FOVs are arranged:

| Strategy | Description |
|---|---|
| `Auto` (default) | Uses `Snap to Grid` if the positions align to a grid, otherwise falls back to `Snap to Corners`. Takes a `Tiling Tolerance (in pixels)`, default `1`. |
| `Snap to Grid` | Tiles images onto a regular grid. Only works if stage positions align to a grid (with possible overlap). Also takes a tolerance. |
| `Snap to Corners` | Tiles images onto a grid defined by the corner positions of the FOVs. |
| `Inplace` | Writes tiles at their original stage positions without snapping. May produce artifacts if stage positions are imprecise. |

### Writer Mode

Controls how image data is loaded into memory and written to disk.

| Mode | Description |
|---|---|
| `By FOV` (default) | Loads and writes one FOV at a time. Good balance of speed and memory usage. |
| `By Tile` | Writes one tile (single Z/C/T plane) at a time. Lowest memory usage but slower. |
| `By FOV (Using Dask)` | Parallel FOV writing via Dask. Faster but uses more memory. |
| `By Tile (Using Dask)` | Parallel tile writing via Dask. |
| `In Memory` | Loads all data into memory before writing. Fastest but requires enough RAM. |

### Stage Position Corrections

Corrections applied to the stage positions before the image is written.

| Field | Type | Default | Description |
|---|---|---|---|
| `Remove XY Offset` | `Keep` or `Global` | `Global` | `Global` shifts all positions together so the image XY origin is 0. `Keep` uses them as-is, which fails on a negative position and pads the origin on a positive one. |
| `Remove Z Offset` | `Keep`, `Per-FOV` or `Global` | `Global` | As above for Z; `Per-FOV` shifts each field of view independently to Z origin 0. |
| `Remove T Offset` | `Keep` or `Global` | `Global` | As above for the time axis. |
| `Remove XY Jitter` | `bool` | `true` | Correct minor stage positioning errors across FOVs. |
| `Reindex Channels` | `bool` | `true` | Pack the channels present in the acquisition into a dense 0-based axis. Disable it to keep the instrument's channel numbering, which writes the channels the acquisition did not use as empty planes. |

### OME-Zarr Options

Controls the output OME-Zarr format.

| Field | Type | Default | Description |
|---|---|---|---|
| `Resolution Levels` | `Number of Levels` or `Custom Names` | 5 levels | `Number of Levels` creates that many levels with the default names (`0`, `1`, …); `Custom Names` takes an explicit `Level Names` list, highest resolution first. |
| `Chunks` | `Same as FOV` or `Fixed Size` | `Same as FOV` | How to chunk the data on disk (see below). |
| `Ngff Version` | `str` | `"0.4"` | OME-NGFF specification version to target (`"0.4"` or `"0.5"`). |
| `Table Backend` | `str` | `"csv"` | Backend for storing tables. One of: `anndata`, `json`, `csv`, `parquet`. |

**Chunking strategies:**

=== "Same as FOV (default)"

    Chunk size matches the FOV dimensions, optionally scaled.

    | Field | Default | Description |
    |---|---|---|
    | `XY Scaling Factor` | `1` | Scale factor for XY chunk size relative to FOV (`0.25`, `0.5`, `1`, `2`, `4`). |
    | `Chunk Size for Z` | `10` | Chunk size for the Z dimension. |
    | `Chunk Size for C` | `1` | Chunk size for the C (channel) dimension. |
    | `Chunk Size for T` | `1` | Chunk size for the T (time) dimension. |

=== "Fixed Size"

    Fixed chunk size in pixels, independent of FOV dimensions.

    | Field | Default | Description |
    |---|---|---|
    | `Chunk Size for XY` | `4096` | Chunk size in pixels for XY dimensions. |
    | `Chunk Size for Z` | `10` | Chunk size for the Z dimension. |
    | `Chunk Size for C` | `1` | Chunk size for the C (channel) dimension. |
    | `Chunk Size for T` | `1` | Chunk size for the T (time) dimension. |

### Runtime Settings

How the conversion itself is executed. These do not change the output, only how it is produced.

| Field | Type | Default | Description |
|---|---|---|---|
| `Use Zarrs Codec Pipeline` | `bool` | `false` | Read and write image data with the `zarrs` Rust backend, which is usually faster. Requires the optional `zarrs` dependency. |
| `Dask Scheduler` | `Default`, `Threads`, `Processes` or `Synchronous` | `Default` | How the work is parallelized. `Default` leaves the environment's own settings unchanged; `Synchronous` runs sequentially. |
| `Temporary JSON Options` | `TempJsonOptions` | see below | Where and when intermediate conversion data is stored on disk. |

`Temporary JSON Options` takes a `Temporary Storage URL` (default `{zarr_dir}/_tmp_json`), a `Serialization` mode (`Auto`, `Memory` or `JSON`) and a `Max In-Memory Bytes` threshold (default 10 MiB) above which `Auto` spills to disk.


## Overwrite Modes

All converters support three overwrite modes when the output plate already exists:

- `No Overwrite` (default): The converter will raise an error if the output plate already exists, preventing accidental data loss.
- `Overwrite`: The converter will delete the existing plate and create a new one from scratch.
- `Extend`: The converter will add new acquisitions to the existing plate, and it will ignore any acquisitions that are already present.
This mode can be used to incrementally add acquisitions to a plate without reprocessing everything, or to recover from an error by re-running only the failed acquisition.

## Supported Converters

- [Revity Operetta / Opera Phenix](operetta.md)
- [Evident ScanR](scanr.md)
- [Yokogawa CQ3K](cq3k.md)
- [Yokogawa CellVoyager](cellvoyager.md)
- [Molecular Devices ImageXpress HCS.ai](md_imagexpress.md)
- [Custom TIFF (HCS & Single Image)](custom_tiff.md)
