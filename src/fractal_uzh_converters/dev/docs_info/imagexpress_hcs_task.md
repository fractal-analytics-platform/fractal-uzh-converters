### Purpose

- Convert images acquired with an MD ImageXpress HCS.ai microscope to an OME-Zarr Plate.

### Outputs

- An OME-Zarr Plate.

### Limitations

- This task has been tested on a limited set of acquisitions. It may not work on all MD ImageXpress acquisitions.

### Expected inputs

The following directory structure is expected:

```text
{protocol_name}_{date-time}
├── {protocol_name}.mxprotocol
├── autofocus/
└── experiment{_mode}/
    ├── {acquisition_name}.jdce
    ├── image_metadata_1.csv
    ├── timepoint0/
    └── timepoint1/
        ├── {protocol_name}_t1_C05_s0_w0_z0.tif
        ├── {protocol_name}_t1_C05_s0_w0_z1.tif
        └── ...
```

`Path` should point to the `{protocol_name}_{date-time}` folder or the `{protocol_name}.mxprotocol` file.
