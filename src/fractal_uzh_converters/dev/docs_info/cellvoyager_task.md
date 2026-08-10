### Purpose

- Convert images acquired with a Yokogawa CellVoyager microscope to an OME-Zarr Plate.

### Outputs

- An OME-Zarr Plate.

### Channels

- Channel labels, colors and wavelength ids (`A{action}_C{channel}`) are read from the acquisition's `.mes` protocol file, whose name is recorded in the `.mrf`. Without a `.mes`, the label is the wavelength id.
- To override them, `Advanced` → `Channels` is ordered by the **instrument channel number**: element 0 is `Ch1`, element 1 is `Ch2`, over the full range the protocol declares — not one entry per channel visible in the output. It must be at least as long as the highest channel the acquisition uses; the error names that number if it is too short.

### Metadata

- The acquisition's five plate-level vendor files are copied verbatim into `<plate>.zarr/metadata/`: the `.mlf` and `.mrf` under their fixed names, the `.mes`, `.wpi` and `.wpp` under the names recorded in the `.mrf`.
- A file the acquisition does not ship is only a warning — CV8000 acquisitions routinely name a `.wpi` that was never written — and a name already taken by another acquisition becomes `<stem>_acq{id}<ext>` instead of overwriting.

### Limitations

- Unlike the CQ3K converter, this task does not support Z-image processing types.

### Expected inputs

The following directory structure is expected:

```text
my_acquisition/
├── MeasurementData.mlf      # Image measurement records (required)
├── MeasurementDetail.mrf    # Acquisition details and channel info (required)
├── protocol.mes             # Acquisition protocol (optional, sets channel names)
├── plate.wpi                # Plate definition (optional)
├── plate.wpp                # Plate product (optional)
├── image_001.png
└── ...
```

The image file paths are referenced inside `MeasurementData.mlf` (with `.tif` extension) and can be in subdirectories relative to the acquisition directory. The actual files may use `.png` or `.tif` extension — select the matching extension via the `image_extension` parameter. The `.mes`, `.wpi` and `.wpp` filenames vary per acquisition and are read out of the `.mrf` — the names above are only examples.
