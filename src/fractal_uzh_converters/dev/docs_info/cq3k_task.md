### Purpose

- Convert images acquired with a Yokogawa CQ3K microscope to a OME-Zarr Plate.

### Outputs

- A OME-Zarr Plate.

### Limitations

- This task has been tested on a limited set of acquisitions. It may not work on all Yokogawa CQ3K acquisitions.

### Expected inputs

The following directory structure is expected:

```text
/plate_dir/
----/MeasurementData.mlf
----/MeasurementDetail.mrf
----/Image/
--------/W0000F0001T0001Z000C1.tif
----/Projection/ (optional)
--------/W0000F0001T0001Z000C1.tif
```