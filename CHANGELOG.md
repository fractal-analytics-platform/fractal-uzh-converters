# Changelog

## [Unreleased]

Migration to `ome-zarr-converters-tools` v1.

### Features
- Yokogawa (CQ3K, CellVoyager): channel labels, wavelength ids (`A{action}_C{channel}`) and
  colours are now read from the acquisition's `.mes` protocol file (#27).
- CQ3K: new `z_processing` option selecting which Z-image processing outputs to convert,
  so a projection or the raw Z stack can be skipped without a path regex filter.
- Yokogawa (CQ3K, CellVoyager): the acquisition's `.mlf`, `.mrf`, `.mes`, `.wpi` and `.wpp`
  are copied verbatim into `<plate>.zarr/metadata/`, so the vendor metadata the converters
  do not model travels with the converted plate (#46).

### API Breaking Changes
- **CellVoyager acquisitions tagged `bts:ZImageProcessing` now split into one plate per
  algorithm** (`_MIP`/`_MinIP`/`_SIP`), with the same `z_processing` selection as CQ3K;
  the attribute was previously dropped, overlapping the tiles of both plates.
- **`z_processing` is now a top-level acquisition field** on CQ3K, CellVoyager and MD
  ImageXpress instead of an advanced option, sharing one `Z Processing Selection` shape
  per converter; the Yokogawa switch `Z slices` is renamed `Raw`.
- **MD ImageXpress `advanced.convert_only_projections` is removed** in favour of
  `z_processing`, where `MIP` reads `experiment` and `Raw` prefers `experiment_z_stack`;
  the two cannot be enabled together, since MD reads a single source directory.
- **Yokogawa default channel labels are no longer `channel_N`** but the `.mes` channel
  target, falling back to the wavelength id when no `.mes` is available. With
  `reindex_channels` disabled, instrument channels the acquisition did not use are now
  written as empty planes.
- **CQ3K projection plates are now suffixed `_MIP`/`_MinIP`/`_SIP`** instead of the raw
  `bts:ZImageProcessing` values `_Maximum`/`_Minimum`/`_Sum`; an unrecognised value is
  still used verbatim (#45).
- **CustomTiff `acquisition_details.toml` keys renamed** (follows the v1 field renames):
  `pixelsize` → `xy_pixel_size`, `start_z_coo`/`start_t_coo` → `start_z_space`/`start_t_space`
  (and any other `*_coo` → `*_space`). Existing user `acquisition_details.toml` files must
  be updated. See `docs/converters/custom_tiff.md`.
- **`Advanced` is now the last acquisition field** in all seven converters instead of the
  second, so the form reads `Path → Plate Name → Acquisition Id → … → Advanced`.

### Fix
- All converters: strip leading and trailing whitespace from channel labels and
  wavelength ids — ngio treats `"DAPI"` and `"DAPI "` as two distinct valid channels, so
  a stray space silently broke per-name channel matching downstream.
- Yokogawa (CQ3K, CellVoyager): channel indices are now 0-based, so an
  `advanced.channels` override maps its first entry to `Ch1` instead of failing (#27).
- Yokogawa (CQ3K, CellVoyager): build one `AcquisitionDetails` per plate instead of one
  per field of view, so fields merged into a single image cannot disagree on it.
- CQ3K: a `.mlf` holding a single measurement record now converts; `xmltodict` returns a
  bare dict for it, which the list-only type rejected.
- CQ3K: unknown `bts:` attributes are ignored rather than rejected, so a firmware update
  adding one is no longer a hard parse failure. Matches CellVoyager.
- Yokogawa (CQ3K, CellVoyager): warn when the `.mrf` channels disagree on pixel size or
  frame size — channel 1's geometry is applied to all of them.
- Yokogawa (CQ3K, CellVoyager): each well's `bts:TimePoint` values are mapped onto a dense
  0-based time axis; the raw value counts timelines, so it left leading frames empty.
- Yokogawa (CQ3K, CellVoyager): a `bts:Type="ERR"` record no longer makes the whole `.mlf`
  unparsable — it carries no `bts:FieldIndex`, which was required on the shared record
  base. Such records are skipped, with one warning per acquisition (#41).

### Chores
- Unify the two Yokogawa converters into a `yokogawa/` package with one shared parser and
  thin `cq3k`/`cellvoyager` layers; `fractal_uzh_converters.{cq3k,cellvoyager}` move to
  `fractal_uzh_converters.yokogawa.{cq3k,cellvoyager}`, package-root names are unchanged.
- Bump `ome-zarr-converters-tools` to `>=1.0.2` for its channel-metadata compaction fix,
  and drop the `ngff_version=` argument it deprecates from every init task.
- Replace the Yokogawa CQ3K test data — both the in-repo fixture and the extended
  datasets — with BSSE-CQ3000 acquisitions; the previous data was not redistributable.
  The new fixture also ships `.mes`, `.wpi` and `.wpp`.
- Add an extended test suite for the Yokogawa CellVoyager converter.
- Rename the Yokogawa extended test datasets to the canonical `hcs_…` convention and
  regenerate their reference snapshots, now that the channel counts and plate names have
  settled.
- Bump `ome-zarr-converters-tools` to `[s3]>=1.0.0,<2.0.0`. The `[s3]` extra is now
  required: v1 makes `s3fs` optional, so it is pinned here to keep `s3://` inputs working
  (it was previously pulled in transitively).
- Adopt the v1 `AcquisitionDetails` renames in all builders: `pixelsize=` → `xy_pixel_size=`
  and `start_*_coo=`/`length_*_coo=` → `*_space=` (values unchanged). Import `BackendType`
  from the package root instead of the private `models._converter_options`.
- Regenerate `__FRACTAL_MANIFEST__.json` against the v1 `AcquisitionOptions` schema (new
  built-in filters, `grouping`/tiling split, `remove_*` stage corrections, scheduler
  `mode`).

### Documentation
- Document Yokogawa channel handling: where the labels come from, and that an
  `advanced.channels` override is indexed by instrument channel number rather than by the
  channels present in the output.
- Fix `tests/data_intake_instructions.md` and `tests/cleanup_test_data.sh`: snapshots are
  JSON, not YAML, and the extended-test template was outdated.
- Document the Yokogawa vendor-metadata copy on both converter pages, and drop the
  limited-testing caveats now that both converters have an extended test suite.

## [v0.7.2]

### Fix
- Bug in extended testing using the wrong api
- Fix typo in the conversion log message (`Successfully`).

### Chores
- Load the shared snapshot-testing plugin via `pytest_plugins` in `tests/conftest.py` instead of `-p` in `pytest` `addopts`, matching the sibling converters.
- Align repository tooling with `ome-zarr-converters-tools`: adopt its `.pre-commit-config.yaml` (`validate-pyproject` v0.25, `crate-ci/typos`, `astral-sh/ruff-pre-commit` v0.15.17, `nbstripout`) with a per-repo `_typos.toml`, add a `chores` pixi task, bump GitHub Actions pins (`checkout` v7, `codecov-action` v7, `action-gh-release` v3, `setup-python` v6), and trim `CLAUDE.md`.

## [0.7.1]

### Fix
- Remove passing explicit `logger_name` to the `run_fractal_task` wrapper.

## [0.7.0]

### Features
- Add Python API functions (`convert_cellvoyager`, `convert_cq3k`, `convert_operetta`, `convert_imagexpress_hcs`, `convert_hcs_tiff`, `convert_single_tiff`) for programmatic use outside Fractal, following the existing `convert_scanr` pattern.
- Update test utilities to call the high-level API functions, providing end-to-end coverage of both the API layer and the underlying init/compute tasks.

### Docs
- Add "How to Run the Converters" page documenting the Python API with usage examples, common parameters, and per-converter code snippets.
- Add Python API section to each converter page with a copy-paste ready code example.

### Chores
- Remove `custom_tiff/_setup.py` and its import: `ome-zarr-converters-tools>=0.10.0` now ships a built-in `setup_singleimage` handler for `SingleImage` collections, making the local copy redundant.
- Bump to `ome-zarr-converters-tools>=0.10.0,<0.11.0`
- Rename internal modules to `_{module_name}.py` across all converter packages and `common/` to signal they are private implementation details.

## [0.6.0]

### Features
- Add `custom_tiff` converter: two new tasks for converting plain TIFF data (HCS plate and single-image) to OME-Zarr.

### Fix
- Fix `get_attributes_from_condition_table` to dynamically exclude the acquisition column from results.

### Chores
- Bump to `ome-zarr-converters-tools>=0.9.0,<0.10.0`
