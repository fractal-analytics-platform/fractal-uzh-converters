# Custom TIFF

The Custom TIFF converters handle generic TIFF image files without any proprietary microscope metadata. Instead of reading a vendor-specific format, these converters rely on a user-supplied `tiles.csv` file that describes the location and dimensions of each tile. An optional `acquisition_details.toml` provides global metadata such as pixel size and channel names.

Two variants are available:

- **HCS (plate)** — organises images into an OME-Zarr HCS plate with well structure. Use task `Convert Custom TIFF HCS Plate to OME-Zarr`.
- **Single Image** — produces flat OME-Zarr images with no well structure. Use task `Convert Custom TIFF Images to OME-Zarr`.

## Expected Data Structure

Each acquisition directory must contain a `tiles.csv` file. The TIFF files can live anywhere on disk; paths in `tiles.csv` can be relative (resolved against the acquisition directory) or absolute.

```
my_acquisition/
├── tiles.csv                     # Required
├── acquisition_details.toml      # Optional
└── data/
    ├── image_001.tif
    ├── image_002.tif
    └── ...
```

## tiles.csv

### HCS Mode

The CSV must contain at least `file_path`, `row`, and `column`. All other columns are optional.

| Column | Required | Description |
|---|---|---|
| `file_path` | Yes | Path to the TIFF file. Relative paths are resolved against the acquisition directory. |
| `row` | Yes | Well row identifier (e.g., `A`, `B`). |
| `column` | Yes | Well column identifier (integer, e.g., `1`, `2`). |
| `fov_name` | No | Field-of-view name. Tiles sharing the same `fov_name` within a well are assembled into one FOV. If absent, each tile becomes its own FOV. |
| `start_x` | No | X position of the tile (in micrometers, or pixels when `start_z_coo = "pixel"`). |
| `start_y` | No | Y position of the tile. |
| `start_z` | No | Z index or position of the tile. |
| `start_c` | No | Channel index of the tile. |
| `start_t` | No | Timepoint index of the tile. |
| `length_x` | No | Tile width in pixels. Inferred from the TIFF file if absent. |
| `length_y` | No | Tile height in pixels. |
| `length_z` | No | Extent in Z (number of Z planes covered). Defaults to `1`. |
| `length_c` | No | Extent in C (number of channels covered). Defaults to `1`. |
| `length_t` | No | Extent in T (number of timepoints covered). Defaults to `1`. |
| *any other column* | No | Treated as a well attribute and stored in the condition table (e.g., `drug`, `concentration`). |

!!! tip "Minimal example"
    ```csv
    file_path,row,column
    data/well_A1.tif,A,1
    data/well_B2.tif,B,2
    ```

Full example with FOVs, Z-planes, and a condition column:

```csv
file_path,row,column,fov_name,start_x,start_y,start_z,start_c,start_t,length_x,length_y,length_z,length_c,length_t,drug
data/fov1_z0.tif,A,1,FOV_1,10.0,10.0,0,0,0,64,64,1,1,1,DMSO
data/fov1_z1.tif,A,1,FOV_1,10.0,10.0,1,0,0,64,64,1,1,1,DMSO
data/fov2_z0.tif,A,1,FOV_2,1000.0,1000.0,0,0,0,64,64,1,1,1,DMSO
data/fov2_z1.tif,A,1,FOV_2,1000.0,1000.0,1,0,0,64,64,1,1,1,DMSO
```

### Single Image Mode

The CSV must contain at least `file_path` and `fov_name`. All other columns are optional.

| Column | Required | Description |
|---|---|---|
| `file_path` | Yes | Path to the TIFF file. Relative paths are resolved against the acquisition directory. |
| `fov_name` | Yes | Output image name. Tiles sharing the same `fov_name` are assembled into one OME-Zarr image. |
| `start_x` | No | X position of the tile. |
| `start_y` | No | Y position of the tile. |
| `start_z` | No | Z index or position of the tile. |
| `start_c` | No | Channel index of the tile. |
| `start_t` | No | Timepoint index of the tile. |
| `length_x` | No | Tile width in pixels. |
| `length_y` | No | Tile height in pixels. |
| `length_z` | No | Extent in Z. Defaults to `1`. |
| `length_c` | No | Extent in C. Defaults to `1`. |
| `length_t` | No | Extent in T. Defaults to `1`. |

Example with two FOVs and two Z-planes each:

```csv
file_path,fov_name,start_x,start_y,start_z,start_c,start_t,length_x,length_y,length_z,length_c,length_t
data/fov1_z0.tif,FOV_1,10.0,10.0,0,0,0,64,64,1,1,1
data/fov1_z1.tif,FOV_1,10.0,10.0,1,0,0,64,64,1,1,1
data/fov2_z0.tif,FOV_2,1000.0,1000.0,0,0,0,64,64,1,1,1
data/fov2_z1.tif,FOV_2,1000.0,1000.0,1,0,0,64,64,1,1,1
```

## acquisition_details.toml

This file is entirely optional. When present it provides global metadata that applies to all tiles in the acquisition. Any field can be overridden per-acquisition via the `Advanced` parameter (see [Acquisition Options](index.md#acquisition-options-advanced)).

| Field | Type | Description |
|---|---|---|
| `pixelsize` | `float` | Physical pixel size in micrometers (XY). |
| `z_spacing` | `float` | Distance between Z planes in micrometers. |
| `start_z_coo` | `str` | Coordinate system for `start_z` values in `tiles.csv`. Use `"pixel"` to treat them as integer Z indices; omit (or set to `"micrometer"`) to treat them as physical positions in µm. |
| `start_t_coo` | `str` | Coordinate system for `start_t` values. Same values as `start_z_coo`. |
| `axes` | `str` | Override the axes string (e.g., `"czyx"`, `"tczyx"`). |
| `[[channels]]` | array | List of channel definitions. Each entry has `channel_label` (display name) and optionally `wavelength_id` (e.g., `"405"`). |

Example:

```toml
pixelsize = 0.65
z_spacing = 5.0
start_z_coo = "pixel"
start_t_coo = "pixel"

[[channels]]
channel_label = "DAPI"
wavelength_id = "405"

[[channels]]
channel_label = "GFP"
wavelength_id = "488"
```

!!! note "Channel order"
    Channel definitions must appear in the same order as the channel indices (`start_c`) used in `tiles.csv`.

## Task Parameters

Both tasks use the standard base acquisition parameters. There are no converter-specific extra fields.

| Field | Type | Default | Description |
|---|---|---|---|
| `Path` | `str` | *required* | Path to the acquisition directory containing `tiles.csv`. |
| `Plate Name` | `str` or `null` | `null` | Custom output name. Defaults to the directory name. |
| `Acquisition Id` | `int` | `0` | Acquisition identifier for combining multiple acquisitions into one plate. |
| `Advanced` | `AcquisitionOptions` | `{}` | Advanced options (condition table, channel/pixel-size overrides). |
