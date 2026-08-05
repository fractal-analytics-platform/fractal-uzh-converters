# Custom TIFF

The Custom TIFF converters handle generic TIFF image files without any proprietary microscope metadata. Instead of reading a vendor-specific format, these converters rely on a user-supplied `tiles.csv` file that describes the location and dimensions of each tile. An optional `acquisition_details.toml` provides global metadata such as pixel size and channel names.

When the TIFF files are OME-TIFF or ImageJ TIFF, the converter automatically reads embedded metadata (pixel size, Z/T spacing, image dimensions) as a fallback. Values supplied explicitly in `tiles.csv` or `acquisition_details.toml` always take priority over auto-detected ones, and the user `Advanced` options take priority over both.

Two variants are available:

- **HCS (plate)** — organises images into an OME-Zarr HCS plate with well structure. Use task `Convert Custom TIFF HCS Plate to OME-Zarr`.
- **Single Image** — produces flat OME-Zarr images with no well structure. Use task `Convert Custom TIFF Images to OME-Zarr`.

!!! note "Metadata priority"
    Metadata is applied in this order (later values override earlier ones):

    1. Parsed from the TIFF file (NOTE: only OME-TIFF and ImageJ TIFF are supported, and only the first TIFF file is parsed for metadata)
    2. Columns in `tiles.csv` and values in `acquisition_details.toml`
    3. `Advanced` acquisition options set by the user

!!! warning "TIFF axis requirements"
    The converter only supports TIFF files whose axes are a canonical
    subsequence of `t, c, z, y, x` in that order. Files with non-canonical or
    unrecognized axis labels are **not supported** and will raise a `ValueError`
    at startup.

    - **OME-TIFF** with standard `DimensionOrder` (e.g. `XYZCT`): tifffile
      normalizes to `TCZYX` — fully supported.
    - **ImageJ TIFF** with explicit channel metadata: supported.
    - **Plain TIFF** (no axis metadata): tifffile may assign arbitrary labels
      (`I`, `Q`, …). Such files are not supported; convert them to OME-TIFF
      first.

## HCS Mode

### Expected Data Structure

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

### tiles.csv

The only truly required columns are `file_path`, `row`, and `column`. All other columns have automatic defaults derived from TIFF metadata or fixed fallback values.

| Column | Required | Default / Source | Description |
|---|---|---|---|
| `file_path` | Yes | — | Path to the TIFF file. Relative paths are resolved against the acquisition directory. |
| `row` | Yes | — | Well row identifier (e.g., `A`, `B`). |
| `column` | Yes | — | Well column identifier (integer, e.g., `1`, `2`). |
| `fov_name` | No | `"Image"` | Field-of-view name. Tiles sharing the same `fov_name` within a well are assembled into one FOV. |
| `start_x` | No | `0.0` | X position of the tile in micrometers. |
| `start_y` | No | `0.0` | Y position of the tile in micrometers. |
| `length_x` | No | From TIFF dimensions | Tile width in pixels. Read from the TIFF file if not provided. |
| `length_y` | No | From TIFF dimensions | Tile height in pixels. Read from the TIFF file if not provided. |
| `start_z` | No | `0` | Z index or position of the tile. |
| `start_c` | No | `0` | Channel index of the tile. |
| `start_t` | No | `0` | Timepoint index of the tile. |
| `length_z` | No | From TIFF dimensions, else `1` | Extent in Z (number of Z planes covered). |
| `length_c` | No | From TIFF dimensions for OME-TIFF/ImageJ TIFF, else `1` | Extent in C (number of channels covered). |
| `length_t` | No | From TIFF dimensions, else `1` | Extent in T (number of timepoints covered). |
| *any other column* | No | — | Treated as a well attribute and stored in the condition table (e.g., `drug`, `concentration`). |

!!! tip "Minimal example"
    With OME-TIFF or ImageJ TIFF files you can omit all dimension and position columns:

    ```csv
    file_path,row,column
    data/well_A1.tif,A,1
    data/well_B2.tif,B,2
    ```

    Dimensions are read from the TIFF files; positions default to `0`; `fov_name` defaults to `"Image"`.

Full example with FOVs, Z-planes, and a condition column:

```csv
file_path,row,column,fov_name,start_x,start_y,start_z,start_c,start_t,length_x,length_y,length_z,length_c,length_t,drug
data/fov1_z0.tif,A,1,FOV_1,10.0,10.0,0,0,0,64,64,1,1,1,DMSO
data/fov1_z1.tif,A,1,FOV_1,10.0,10.0,1,0,0,64,64,1,1,1,DMSO
data/fov2_z0.tif,A,1,FOV_2,1000.0,1000.0,0,0,0,64,64,1,1,1,DMSO
data/fov2_z1.tif,A,1,FOV_2,1000.0,1000.0,1,0,0,64,64,1,1,1,DMSO
```

### acquisition_details.toml

This file is entirely optional. When present it provides global metadata that applies to all tiles in the acquisition. Any field can be overridden per-acquisition via the `Advanced` parameter (see [Acquisition Options](index.md#acquisition-options-advanced)).

Spacing values (`xy_pixel_size`, `z_spacing`, `t_spacing`) are automatically read from OME-TIFF or ImageJ TIFF metadata when the TIFF files contain them. Explicit values in this file take priority over auto-detected ones.

> **Migration note (ome-zarr-converters-tools v1):** the keys `pixelsize`,
> `start_z_coo` and `start_t_coo` were renamed to `xy_pixel_size`, `start_z_space`
> and `start_t_space`. Update any existing `acquisition_details.toml` files accordingly.

| Field | Type | Description |
|---|---|---|
| `xy_pixel_size` | `float` | Physical pixel size in micrometers (XY). Auto-detected from OME `PhysicalSizeX` if absent. |
| `z_spacing` | `float` | Distance between Z planes in micrometers. Auto-detected from OME `PhysicalSizeZ` or ImageJ `spacing` if absent. |
| `t_spacing` | `float` | Time interval between frames in seconds. Auto-detected from OME `TimeIncrement` if absent. |
| `start_z_space` | `str` | Coordinate system for `start_z` values in `tiles.csv`. Use `"pixel"` to treat them as integer Z indices; omit (or set to `"world"`) to treat them as physical positions in µm. |
| `start_t_space` | `str` | Coordinate system for `start_t` values. Same values as `start_z_space`. |
| `axes` | `str` | Override the axes string (e.g., `"czyx"`, `"tczyx"`). |
| `[[channels]]` | array | List of channel definitions. Each entry has `channel_label` (display name) and optionally `wavelength_id` (e.g., `"405"`). |

Example:

```toml
xy_pixel_size = 0.65
z_spacing = 5.0
start_z_space = "pixel"
start_t_space = "pixel"

[[channels]]
channel_label = "DAPI"
wavelength_id = "405"

[[channels]]
channel_label = "GFP"
wavelength_id = "488"
```

!!! note "Channel order"
    Channel definitions must appear in the same order as the channel indices (`start_c`) used in `tiles.csv`.

### Task Parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `Path` | `str` | *required* | Path to the acquisition directory containing `tiles.csv`. |
| `Plate Name` | `str` or `null` | `null` | Custom output plate name. Defaults to the directory name. |
| `Acquisition Id` | `int` | `0` | Acquisition identifier for combining multiple acquisitions into one plate. |
| `Advanced` | `AcquisitionOptions` | `{}` | Advanced options (condition table, channel/pixel-size overrides). |


## Single Image Mode


### Expected Data Structure

Each acquisition directory must contain a `tiles.csv` file. The TIFF files can live anywhere on disk; paths in `tiles.csv` can be relative (resolved against the acquisition directory) or absolute.

**Single TIFF mode:** To convert a single TIFF file, simply provide the path to the file instead of a directory. The converter detects that it is a file, skips looking for `tiles.csv`, and automatically reads dimensions and spacing from the file metadata.
```
my_acquisition.tiff
```

**Multiple TIFF mode:** To convert multiple TIFF files, organise them in a directory with a `tiles.csv` that describes their layout. The structure is the same as for HCS mode, but `row` and `column` are not required since there is no well structure.
```
my_acquisition/
├── tiles.csv                     # Required
├── acquisition_details.toml      # Optional
└── data/
    ├── image_001.tif
    ├── image_002.tif
    └── ...
```

### tiles.csv

The only truly required column is `file_path`. All other columns have automatic defaults derived from TIFF metadata or fixed fallback values.

| Column | Required | Default / Source | Description |
|---|---|---|---|
| `file_path` | Yes | — | Path to the TIFF file. Relative paths are resolved against the acquisition directory. |
| `fov_name` | No | `"Image"` | Field-of-view name. Tiles sharing the same `fov_name` are assembled into one FOV. Also used as the output Zarr name. |
| `start_x` | No | `0.0` | X position of the tile in micrometers. |
| `start_y` | No | `0.0` | Y position of the tile in micrometers. |
| `length_x` | No | From TIFF dimensions | Tile width in pixels. Read from the TIFF file if not provided. |
| `length_y` | No | From TIFF dimensions | Tile height in pixels. Read from the TIFF file if not provided. |
| `start_z` | No | `0` | Z index or position of the tile. |
| `start_c` | No | `0` | Channel index of the tile. |
| `start_t` | No | `0` | Timepoint index of the tile. |
| `length_z` | No | From TIFF dimensions, else `1` | Extent in Z. |
| `length_c` | No | `1` | Extent in C. |
| `length_t` | No | From TIFF dimensions, else `1` | Extent in T. |

!!! tip "Minimal example"
    With OME-TIFF or ImageJ TIFF files you can provide only `file_path`:

    ```csv
    file_path
    data/image.tif
    ```

    Dimensions are read from the TIFF file; positions default to `0`; `fov_name` defaults to `"Image"`.

Example with two FOVs and two Z-planes each:

```csv
file_path,fov_name,start_x,start_y,start_z,start_c,start_t,length_x,length_y,length_z,length_c,length_t
data/fov1_z0.tif,FOV_1,10.0,10.0,0,0,0,64,64,1,1,1
data/fov1_z1.tif,FOV_1,10.0,10.0,1,0,0,64,64,1,1,1
data/fov2_z0.tif,FOV_2,1000.0,1000.0,0,0,0,64,64,1,1,1
data/fov2_z1.tif,FOV_2,1000.0,1000.0,1,0,0,64,64,1,1,1
```

### acquisition_details.toml

Same format and fields as described in the HCS Mode section above.

### Task Parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `Path` | `str` | *required* | Path to the acquisition directory containing `tiles.csv`, or directly to a TIFF file. |
| `Image Name` | `str` or `null` | `null` | Custom output OME-Zarr image name. Defaults to the directory or file name. |
| `Advanced` | `AcquisitionOptions` | `{}` | Advanced options (channel/pixel-size overrides). |

## Python API

=== "HCS Plate"

    ```python
    from fractal_uzh_converters import convert_hcs_tiff, HcsTiffAcquisitionModel

    acquisitions = [
        HcsTiffAcquisitionModel(
            path="/path/to/hcs_acquisition",
            plate_name="my_plate",
            acquisition_id=0,
        )
    ]

    convert_hcs_tiff(
        zarr_dir="/output/zarr",
        acquisitions=acquisitions,
    )
    ```

=== "Single Image"

    ```python
    from fractal_uzh_converters import convert_single_tiff, SingleTiffAcquisitionModel

    acquisitions = [
        SingleTiffAcquisitionModel(
            path="/path/to/image.tif",
            image_name="my_image",
        )
    ]

    convert_single_tiff(
        zarr_dir="/output/zarr",
        acquisitions=acquisitions,
    )
    ```

See [How to Run the Converters](../how_to_run_the_converters.md) for all common parameters and execution details.
