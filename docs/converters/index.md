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

Every acquisition object shares these base fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `Path` | `str` | *required* | Path to the raw acquisition directory. |
| `Plate Name` | `str` or `null` | `null` | Custom plate name. If not set, the directory name is used. |
| `Acquisition Id` | `int` | `0` | Identifies the acquisition when combining multiple acquisitions into one plate. |
| `Advanced` | `AcquisitionOptions` | `{}` | Advanced options including `Condition Table Path` and acquisition detail overrides. |

Some converters add extra fields (e.g., `Layout` for ScanR). See the individual converter pages for details.

If multiple acquisitions need to be combined into a single plate, simply provide multiple acquisition objects with the same `Plate Name`, but different `Acquisition Id` values (e.g. in case of multiplexed experiments).

### Acquisition Options (Advanced)

The `Advanced` field on each acquisition allows per-acquisition overrides and filtering. Most users only need `Condition Table Path` here.

| Field | Type | Default | Description |
|---|---|---|---|
| `Condition Table Path` | `str` or `null` | `null` | Absolute path to a [condition table](../condition_tables.md) CSV file. |
| `Channels` | `list[ChannelInfo]` or `null` | `null` | Override channel names and colors. |
| `Pixel Size Information` | `PixelSizeModel` or `null` | `null` | Override pixel size (`Pixelsize`, `Z Spacing`, `T Spacing` in micrometers). |
| `Axes` | `str` or `null` | `null` | Override axes string (e.g., `"czyx"`). |
| `Data Type` | `str` or `null` | `null` | Override data type: `uint8`, `uint16`, or `uint32`. |
| `Stage Corrections` | `StageCorrections` | `{}` | Flip or swap stage axes (see below). |
| `Filters` | `list` | `[]` | Filters to include/exclude specific tiles. |

#### Stage Corrections

If the microscope stage orientation does not match the expected coordinate system, you can apply corrections:

| Field | Type | Default | Description |
|---|---|---|---|
| `Flip X` | `bool` | `false` | Flip positions along the X axis. |
| `Flip Y` | `bool` | `false` | Flip positions along the Y axis. |
| `Swap XY` | `bool` | `false` | Swap the X and Y axes. |

#### Filters

You can filter tiles during conversion using the `Filters` list:

- **Well Filter** — Remove specific wells by ID (e.g., `["A1", "B2"]`).
- **Path Regex Include Filter** — Only include tiles whose file path matches a regex.
- **Path Regex Exclude Filter** — Exclude tiles whose file path matches a regex.

## Converter Options

The `Converter Options` parameter controls how tiles are assembled, written, and stored. The defaults work well for most cases — only adjust these if you have specific requirements.

### Tiling Mode

Controls how individual fields of view (FOVs) are assembled into the final image.

| Mode | Description |
|---|---|
| `Auto` (default) | Automatically picks `Snap to Grid` if positions align to a grid, otherwise falls back to `Snap to Corners`. |
| `Snap to Grid` | Tiles images onto a regular grid. Only works if stage positions align to a grid (with possible overlap). |
| `Snap to Corners` | Tiles images onto a grid defined by the corner positions of the FOVs. |
| `Inplace` | Writes tiles at their original stage positions without snapping. May produce artifacts if stage positions are imprecise. |
| `No Tiling` | Each FOV is written as a separate OME-Zarr image (no stitching). |

### Writer Mode

Controls how image data is loaded into memory and written to disk.

| Mode | Description |
|---|---|
| `By FOV` (default) | Loads and writes one FOV at a time. Good balance of speed and memory usage. |
| `By Tile` | Writes one tile (single Z/C/T plane) at a time. Lowest memory usage but slower. |
| `By FOV (Using Dask)` | Parallel FOV writing via Dask. Faster but uses more memory. |
| `By Tile (Using Dask)` | Parallel tile writing via Dask. |
| `In Memory` | Loads all data into memory before writing. Fastest but requires enough RAM. |

### Alignment Corrections

Corrects for minor stage positioning errors across FOVs.

| Field | Type | Default | Description |
|---|---|---|---|
| `Align XY` | `bool` | `false` | Align FOV positions in the XY plane. |
| `Align Z` | `bool` | `false` | Align FOV positions along the Z axis. |
| `Align T` | `bool` | `false` | Align FOV positions along the T axis. |

### OME-Zarr Options

Controls the output OME-Zarr format.

| Field | Type | Default | Description |
|---|---|---|---|
| `Num Levels` | `int` | `5` | Number of resolution pyramid levels. |
| `Chunking Strategy` | `ChunkingStrategy` | FOV-based | How to chunk the data on disk (see below). |
| `Ngff Version` | `str` | `"0.4"` | OME-NGFF specification version to target (`"0.4"` or `"0.5"`). |
| `Table Backend` | `str` | `"anndata"` | Backend for storing tables. One of: `anndata`, `json`, `csv`, `parquet`. |

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


## Overwrite Modes

All converters support three overwrite modes when the output plate already exists:

- `No Overwrite` (default): The converter will raise an error if the output plate already exists, preventing accidental data loss.
- `Overwrite`: The converter will delete the existing plate and create a new one from scratch.
- `Extend`: The converter will add new acquisitions to the existing plate, and it will ignore any acquisitions that are already present.
This mode can be used to incrementally add acquisitions to a plate without reprocessing everything, or to recover from an error by re-running only the failed acquisition.

## Supported Converters

- [PerkinElmer Operetta / Opera Phenix](operetta.md)
- [Olympus ScanR](scanr.md)
- [Yokogawa CQ3K / CellVoyager](cq3k.md)
