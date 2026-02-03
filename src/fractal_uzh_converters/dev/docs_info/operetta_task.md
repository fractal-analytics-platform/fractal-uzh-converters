### Purpose

- Convert images acquired with a PerkinElmer Operetta microscope to a OME-Zarr Plate.

### Outputs

- A OME-Zarr Plate.

### Limitations

- This task has been tested on a limited set of acquisitions. It may not work on all PerkinElmer Operetta acquisitions.

### Expected inputs

The following directory structure is expected:

```text
/plate_dir/
----/Images/
--------/Index.idx.xml
--------/r03c11f01p01-ch1sk1fk1fl1.tiff
--------/...
```