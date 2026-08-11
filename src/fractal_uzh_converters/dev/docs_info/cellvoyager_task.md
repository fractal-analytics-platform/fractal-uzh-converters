### Purpose

- Convert images acquired with a Yokogawa CellVoyager microscope to an OME-Zarr Plate.

### Outputs

- An OME-Zarr Plate.
- One plate per Z-image processing type. Projections are suffixed `_MIP` (`Maximum`), `_MinIP` (`Minimum`) or `_SIP` (`Sum`), and coexist with the unsuffixed plate holding the raw Z slices. Channel labels are the same across all of them. Most acquisitions carry no `bts:ZImageProcessing` and produce the single unsuffixed plate.
- `Z Processing` is one switch per output. Leave it unset to convert all of them; otherwise only `Raw` is on by default, so enabling a projection adds to the raw stack rather than replacing it. An enabled kind the acquisition does not contain is a warning; enabling nothing, or nothing that the acquisition contains, is an error.

### Channels

- Channel labels, colors and wavelength ids (`A{action}_C{channel}`) are read from the acquisition's `.mes` protocol file, whose name is recorded in the `.mrf`. Without a `.mes`, the label is the wavelength id.
- To override them, `Advanced` → `Channels` is ordered by the **instrument channel number**: element 0 is `Ch1`, element 1 is `Ch2`, over the full range the protocol declares — not one entry per channel visible in the output. It must be at least as long as the highest channel the acquisition uses; the error names that number if it is too short.
- One list covers the whole acquisition, projection plates included: each projection algorithm carries its own channel numbers, so the entries are split across plates rather than repeated in each.

### Metadata

- The acquisition's five plate-level vendor files are copied verbatim into `<plate>.zarr/metadata/`: the `.mlf` and `.mrf` under their fixed names, the `.mes`, `.wpi` and `.wpp` under the names recorded in the `.mrf`.
- Every plate the acquisition produced gets a copy, projection plates included. A file the acquisition does not ship is only a warning — CV8000 acquisitions routinely name a `.wpi` that was never written — and a name already taken by another acquisition becomes `<stem>_acq{id}<ext>` instead of overwriting.

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
