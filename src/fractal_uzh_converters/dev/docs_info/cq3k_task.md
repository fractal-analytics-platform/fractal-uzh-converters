### Purpose

- Convert images acquired with a Yokogawa CQ3K / CellVoyager microscope to an OME-Zarr Plate.

### Outputs

- An OME-Zarr Plate.
- If the acquisition contains multiple Z-image processing types (e.g., `focus`, `maximum_projection`), a separate plate is created for each type.

### Limitations

- This task has been tested on a limited set of acquisitions. It may not work on all Yokogawa CQ3K acquisitions.

### Expected inputs

The following directory structure is expected:

```text
my_acquisition/
├── MeasurementData.mlf      # Image measurement records (required)
├── MeasurementDetail.mrf    # Acquisition details and channel info (required)
└── <subdirectories>/
    ├── image_001.tif
    └── ...
```

The TIFF file paths are referenced inside `MeasurementData.mlf` and can be in subdirectories relative to the acquisition directory.
