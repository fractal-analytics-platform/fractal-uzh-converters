### Purpose

- Convert images acquired with an MD ImageXpress HCS.ai microscope to an OME-Zarr Plate.

### Outputs

- An OME-Zarr Plate.
- One acquisition is read from a single `experiment{_mode}` directory. `Z Processing` picks which: `Raw` (the default) prefers `experiment_z_stack`, falling back to `experiment`; `MIP` reads `experiment`, where MD writes the projections of a Z stack. Only one of the two can be enabled, since both would produce the same plate.
- `Advanced` → `Convert Montages` reads `experiment_montage` instead, taking precedence over `Z Processing`; combining it with `MIP` is an error when the montage data is a Z stack.

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
