# Plan: Extended Test Data Intake — Detailed Workflow

## Context

New extended test datasets exist but are not yet formatted to match the `hcs_{W}w{P}p{C}c{Z}z{T}t_{Descriptor}` naming convention used by the existing extended test suites. This document is the full, step-by-step workflow to process new raw acquisition folders — regardless of which of the five supported instruments they come from — into properly named, snapshotted extended test entries.

---

## Converter Reference Table

The extended tests call the public converter API function (`api_fn=`), not the init task.

| Instrument | Converter API function | data-extended subfolder | Extended test file | Test function name |
|---|---|---|---|---|
| Evident ScanR | `from fractal_uzh_converters.scanr import convert_scanr` | `Evident-scanR/` | `test_evident_scanr_extended.py` | `test_scanr_extended` |
| Yokogawa CQ3K | `from fractal_uzh_converters.cq3k import convert_cq3k` | `Yokogawa-CQ3K/` | `test_yokogawa_cq3k_extended.py` | `test_cq3k_extended` |
| Revvity Operetta | `from fractal_uzh_converters.operetta import convert_operetta` | `Revvity-Operetta/` | `test_revvity_operetta_extended.py` | `test_operetta_extended` |
| Yokogawa CellVoyager | `from fractal_uzh_converters.cellvoyager import convert_cellvoyager` | `Yokogawa-CellVoyager/` | `test_yokogawa_cellvoyager_extended.py` | `test_cellvoyager_extended` |
| Molecular Devices ImageXpress | `from fractal_uzh_converters.imagexpress_hcs import convert_imagexpress_hcs` | `MolecularDevices-ImageXpressHCSai/` | `test_molecular_devices_imagexpress_extended.py` | `test_imagexpress_hcs_extended` |

---

## Naming Convention

All dataset directories follow the pattern:

```
hcs_{W}w{P}p{C}c{Z}z{T}t_{Descriptor}
```

| Token | Meaning | Source in the snapshot JSON |
|---|---|---|
| `{W}` | Number of wells | `len(plates[plate].wells)` |
| `{P}` | Fields of view per well | `len(images[img].tables.FOV_ROI_table.rois)` — use `1` if table absent |
| `{C}` | Number of channels | `images[img].shape` at `axes.index('c')` |
| `{Z}` | Number of Z slices | `images[img].shape` at `axes.index('z')` |
| `{T}` | Number of time points | `images[img].shape` at `axes.index('t')`, or `1` if `t` not in axes |
| `{Descriptor}` | Human-readable variant | Inferred from acquisition characteristics (see below) |

**None of these is guaranteed uniform.** A converter may prune unacquired channels per
image, wells may hold different field counts, and one acquisition may produce several
plates with different shapes (the CQ3K converter emits one plate per Z-projection
algorithm, each carrying its own channel subset and `Z = 1` for the projections). **Take
each token as the maximum over every image of every plate the acquisition produces**, and
record the variation in the store `README.md` row.

**Descriptor guidance** — use PascalCase words separated by underscores to describe what makes the dataset distinctive. It is
usually written in the raw folder name, and can not be reliably parsed from the snapshot JSON. Examples include:
- Projection type: `MIP`, `SUM`, `MIP_SUM`, `MIP_Slice`, `MIP_SUM_Slice`
- Tiling arrangement: `Centered`, `Grid`
- Resolution level: `FP` (field precision / reduced res), `SP` (super/standard res)
- Scanning field: `SF`
- Acquisition mode: `dual`, `seq`
- Fov spacing variant: `nospacing`, `10overlap`, `1000spacing`
- Binning: `2bin`
- Channel emphasis: `Channel` (for multi-channel datasets)

---

## Step-by-Step Workflow

### Step 0 — Check if extended test infrastructure exists for the instrument

**If `tests/data-extended/{InstrumentDir}/` does not exist yet:**

a. Create the directory structure (or symlink to an external data store following the pattern of `Evident-scanR` and `Yokogawa-CQ3K`):
```
tests/data-extended/{InstrumentDir}/
├── raw/
├── snapshots/
└── output/
```

b. Create `tests/test_{instrument}_extended.py` by copying an existing extended test file — `tests/test_yokogawa_cq3k_extended.py` is the reference — and replacing the instrument-specific values (the `convert_*` import, `SNAPSHOT_DIR`, `RAW_DIR`, the test function name):

```python
from pathlib import Path

import pytest
from ome_zarr_converters_tools.testing import run_converter_test

from fractal_uzh_converters.{module} import {convert_fn}

EXTENDED_DATA_DIR = Path(__file__).parent / "data-extended"
SNAPSHOT_DIR = EXTENDED_DATA_DIR / "{InstrumentDir}" / "snapshots"
RAW_DIR = EXTENDED_DATA_DIR / "{InstrumentDir}" / "raw"

_DATASETS: list[str] = []


@pytest.mark.extended
@pytest.mark.parametrize(
    "init_task_kwargs, snapshot_name",
    [
        (
            {"acquisitions": [{"path": str(RAW_DIR / name), "acquisition_id": 0}]},
            name,
        )
        for name in _DATASETS
    ],
)
def test_{instrument}_extended(
    tmp_path: Path,
    init_task_kwargs: dict,
    snapshot_name: str,
    update_snapshots: bool,
    converter_options,
):
    run_converter_test(
        tmp_path=tmp_path,
        api_fn={convert_fn},
        api_kwargs=init_task_kwargs,
        snapshot_path=SNAPSHOT_DIR / f"{snapshot_name}.json",
        update_snapshots=update_snapshots,
        converter_options=converter_options,
    )
```

`run_converter_test`, the `--extended` / `--update-snapshots` options, the `extended`
marker and the `update_snapshots` fixture all come from
`ome_zarr_converters_tools.testing` (the plugin is loaded via `pytest_plugins` in
`tests/conftest.py`). `converter_options` is a local fixture defined there.

---

### Step 1 — Place the raw data with a temporary name

Copy or symlink the new acquisition folder into:
```
tests/data-extended/{InstrumentDir}/raw/{tmp_name}/
```
`{tmp_name}` can be the original folder name as-is (no renaming yet).

---

### Step 2 — Add temp name to `_DATASETS` and run converter with `--update-snapshots`

In the extended test file, add `"{tmp_name}"` to `_DATASETS`, then run:

```bash
pixi run -e test pytest tests/test_{instrument}_extended.py \
    -k "{tmp_name}" --extended --update-snapshots
```

This produces:
- `tests/data-extended/{InstrumentDir}/snapshots/{tmp_name}.json` — the snapshot
- `tests/data-extended/{InstrumentDir}/output/{tmp_name}/` — the zarr output (kept for inspection)

If the converter fails, debug the raw data format before continuing.

---

### Step 3 — Parse the snapshot to derive `{W}`, `{P}`, `{C}`, `{Z}`, `{T}`

Open `tests/data-extended/{InstrumentDir}/snapshots/{tmp_name}.json` and read:

```jsonc
{
  "versions": { ... },              // library versions, not used for naming
  "plates": {
    "{plate_name}.zarr": {
      "wells": ["C/05", "F/09"],    // → W = len(this list)
      "images": {
        "C/05/0": {
          "axes":  ["c", "z", "y", "x"],   // or ["t", "c", "z", "y", "x"]
          "shape": [1, 1, 2000, 2000],
          "tables": {
            "FOV_ROI_table": {
              "rois": {             // → P = len(rois); key absent → P = 1
                "FOV_1": { ... },
                "FOV_2": { ... }
              }
            }
          }
        }
      }
    }
  }
}
```

Per the rule above, each token is the **maximum** over every image of every plate:

- **W**: `len(plates[plate]["wells"])`
- **P**: `len(plates[plate]["images"][img]["tables"]["FOV_ROI_table"]["rois"])`. If `FOV_ROI_table` is absent → `P = 1`.
- **C**: `shape[axes.index("c")]`
- **Z**: `shape[axes.index("z")]` — for projection-only datasets this will be `1`
- **T**: `shape[axes.index("t")]` if `"t"` in axes, else `1`

Do not eyeball this. Derive the tokens with a script and assert they match the folder name
you chose, then note in the store `README.md` wherever a value actually varies — which
well has fewer fields, which plate has fewer channels.

Construct the canonical name: `hcs_{W}w{P}p{C}c{Z}z{T}t_{Descriptor}`

---

### Step 4 — Rename and flatten the raw directory

- Rename `tests/data-extended/{InstrumentDir}/raw/{tmp_name}/` → `tests/data-extended/{InstrumentDir}/raw/{canonical_name}/`
- If the folder has unnecessary nesting (e.g., a single intermediate directory containing all the actual files), move the contents up one level so that acquisition files live directly inside `raw/{canonical_name}/`

---

### Step 5 — Update the instrument-level `README.md`

There is a single `README.md` per instrument at `tests/data-extended/{InstrumentDir}/README.md` (not one per dataset). Add a row for the new dataset to the overview table:

```markdown
| {canonical_name} | HCS | {W} | {P} | {C} | {Z} | {T} | {One-sentence description} |
```

If the `README.md` does not yet exist for the instrument, create it following the structure of `tests/data-extended/Evident-scanR/README.md`:

```markdown
# {InstrumentDir} Testing Dataset

{One paragraph describing the instrument and data format.}

## Details

- *Authors*: ...
- *Acquisition Date*: ...
- *Acquisition Location*: ...
- *Modality*: Fluorescence microscopy.
- *Microscope*: ...
- *Pixel size*: ...

## Overview

| Dataset Name | Type | Wells | FoV | Channels | Z-Stacks | Time Points | Extra Info |
|---|---|---|---|---|---|---|---|
| {canonical_name} | HCS | {W} | {P} | {C} | {Z} | {T} | {description} |

## Dataset Structure

- *./raw*: Contains the original microscopy data in the original format.
- *./snapshots*: Contains snapshot `.json` files used for automated testing.
```

---

### Step 6 — Update `_DATASETS` and re-generate the snapshot with canonical name

In the extended test file:
- Replace `"{tmp_name}"` → `"{canonical_name}"` in `_DATASETS`
- Delete the old snapshot: `tests/data-extended/{InstrumentDir}/snapshots/{tmp_name}.json`
- Delete the old output dir: `tests/data-extended/{InstrumentDir}/output/{tmp_name}/`

Re-run with the canonical name:

```bash
pixi run -e test pytest tests/test_{instrument}_extended.py \
    -k "{canonical_name}" --extended --update-snapshots
```

Verify the new snapshot:
- Plate name(s) contain the canonical dataset name
- Well IDs are correct
- `axes`, `shape`, `pixelsize`, `channel_labels` look sensible

---

### Step 7 — Validation run (no `--update-snapshots`)

```bash
pixi run -e test pytest tests/test_{instrument}_extended.py --extended
```

All tests — including the new one — must pass.

---

## Edge Cases

**Multiple plates per dataset** (e.g., CQ3K with MIP + SUM + Slice):  
The snapshot JSON will have multiple entries under `"plates"`, and they do **not** all
have the same shape. `Z` differs (projections = 1, slices = N) and so does `C`, because
each projection algorithm carries its own `bts:Ch` numbers — in
`hcs_3w2p4c1z1t_Channels_MIP_MinIP` the `_MIP` plate has 4 channels and the `_MinIP` plate
has 1. Take the maximum across plates for every token, as above.

**No `FOV_ROI_table`** (single-FOV wells):  
`P = 1`. The `well_ROI_table` with an `image` ROI will be the only table.

**Dataset that also tests `condition_table` or other advanced kwargs**:  
The acquisition path stays the same; the snapshot name differs. Add a second entry to `_DATASETS` with the variant suffix (e.g., `{canonical_name}_with_condition_table`) and pass the extra kwarg in the test parametrization manually (not via the shared list comprehension).
