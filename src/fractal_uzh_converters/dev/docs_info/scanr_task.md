### Purpose

- Convert images acquired with an Olympus ScanR microscope to an OME-Zarr Plate.

### Outputs

- An OME-Zarr Plate.

### Limitations

- This task has been tested on a limited set of acquisitions. It may not work on all Olympus ScanR acquisitions.
- Make sure to set the correct `Layout` for your plate. The well ID mapping depends on it — using the wrong layout will assign images to incorrect wells.

### Expected inputs

The following directory structure is expected:

```text
my_acquisition/
└── data/
    ├── metadata.ome.xml     # OME-XML metadata file (required)
    ├── image_001.tiff
    ├── image_002.tiff
    └── ...
```

`Path` can point to either the base directory (`my_acquisition/`) or directly to the `data` subdirectory (`my_acquisition/data/`).
