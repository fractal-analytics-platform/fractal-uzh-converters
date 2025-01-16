### Purpose

- Convert images acquired with an Olympus ScanR microscope to a OME-Zarr Plate.

### Outputs

- A OME-Zarr Plate.

### Limitations

- This task has been tested on a limited set of acquisitions. It may not work on all Olympus ScanR acquisitions.

### Expected inputs

The following directory structure is expected:

```text
/plate_dir/
----/data/
--------/metadata.ome.xml
--------/B2--W00014--P00001--Z00000--T00000--DAPi.tif
--------/B2--W00014--P00001--Z00000--T00000--TRITC.tif
--------/...
```
