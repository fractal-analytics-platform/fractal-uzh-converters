### Purpose

- Convert images acquired with a PerkinElmer Operetta / Opera Phenix microscope to an OME-Zarr Plate.

### Outputs

- An OME-Zarr Plate.

### Limitations

- This task has been tested on a limited set of acquisitions. It may not work on all PerkinElmer Operetta acquisitions.

### Expected inputs

The following directory structure is expected:

```text
my_acquisition/
└── Images/
    ├── Index.idx.xml        # Metadata file (required)
    ├── r01c01f01p01-ch1sk1fk1fl1.tiff
    ├── r01c01f01p01-ch2sk1fk1fl1.tiff
    └── ...
```

`Path` can point to either the base directory (`my_acquisition/`) or directly to the `Images` subdirectory (`my_acquisition/Images/`).
