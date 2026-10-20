### Purpose

- Convert images acquired with a Yokogawa CQ3K microscope to an OME-Zarr Plate.

### Outputs

- An OME-Zarr Plate.
- One plate per Z-image processing type. Projections are suffixed `_MIP` (`Maximum`), `_MinIP` (`Minimum`) or `_SIP` (`Sum`), and coexist with the unsuffixed plate holding the raw Z slices. Channel labels are the same across all of them.
- `Z Processing` is one switch per output. Leave it unset to convert all of them; otherwise only `Raw` is on by default, so enabling a projection adds to the raw stack rather than replacing it. An enabled kind the acquisition does not contain is a warning; enabling nothing, or nothing that the acquisition contains, is an error.

### Channels

- Channel labels, colors and wavelength ids (`A{action}_C{channel}`) are read from the acquisition's `.mes` protocol file, whose name is recorded in the `.mrf`. Without a `.mes`, the label is the wavelength id.
- To override them, add one entry per channel to change under `Advanced` → `Channels`, each with a `Channel Key` — the instrument's `bts:Ch` channel number, the `C` number of the default wavelength id. Channels without an entry keep their `.mes` metadata. A key the protocol does not declare is an error; one it declares but the acquisition never used is accepted. An acquired channel left without an entry is only a warning.
- One list covers the whole acquisition, projection plates included: channel numbers are acquisition-wide, so an entry surfaces in whichever plate acquires that channel.

### Metadata

- The acquisition's five plate-level vendor files are copied verbatim into `<plate>.zarr/metadata/`: the `.mlf` and `.mrf` under their fixed names, the `.mes`, `.wpi` and `.wpp` under the names recorded in the `.mrf`.
- Every plate the acquisition produced gets a copy, projection plates included. A file the acquisition does not ship is only a warning, and a name already taken by another acquisition becomes `<stem>_acq{id}<ext>` instead of overwriting.

### Expected inputs

The following directory structure is expected:

```text
my_acquisition/
├── MeasurementData.mlf         # Image measurement records (required)
├── MeasurementDetail.mrf       # Acquisition details and channel info (required)
├── MeasurementProtocol.mes     # Acquisition protocol (optional, sets channel names)
├── NoPlateID.wpi               # Plate definition (optional)
├── 10_Greiner_μClear.wpp       # Plate product (optional)
└── <subdirectories>/
    ├── image_001.tif
    └── ...
```

The TIFF file paths are referenced inside `MeasurementData.mlf` and can be in subdirectories relative to the acquisition directory. The `.mes`, `.wpi` and `.wpp` filenames vary per acquisition and are read out of the `.mrf` — the names above are only examples.
