# Changelog

## [v0.8.0]

Migration to `ome-zarr-converters-tools` v1 and `ngio` 1.1, plus a rework of the Yokogawa
converters. Written OME-Zarr output is unchanged for Operetta, ScanR, ImageXpress and
CustomTiff; Yokogawa plates change in channel labels, plate suffixes and vendor metadata.

### Features
- Sharded OME-Zarr output: `Converter Options → OME-Zarr Options → Sharding Strategy`
  (`Multiple of Chunks` groups a fixed number of chunks per axis into one shard). Requires
  `Ngff Version` `"0.5"`. Available in every task via `ome-zarr-converters-tools` 1.1.0.
- The Dask writer modes stream the image block by block instead of materialising it first,
  so peak memory no longer scales with the image size (`ome-zarr-converters-tools` 1.1.0).
- Conversion errors name the image, tile and source file they concern, with the expected
  and actual values (`ome-zarr-converters-tools` 1.1.1).
- Yokogawa (CQ3K, CellVoyager): channel labels, wavelength ids (`A{action}_C{channel}`) and
  colours are now read from the acquisition's `.mes` protocol file (#27).
- Yokogawa (CQ3K, CellVoyager): `advanced.channels` entries carry a `Channel Key` (the `bts:Ch`
  channel number), so channels the protocol declares but the acquisition never used need no
  placeholder entry; an acquired channel left without an entry is reported as a warning.
- New `z_processing` acquisition option on CQ3K, CellVoyager and MD ImageXpress, selecting
  which Z-image processing outputs to convert without resorting to a path regex filter.
- Yokogawa (CQ3K, CellVoyager): the acquisition's `.mlf`, `.mrf`, `.mes`, `.wpi` and `.wpp`
  are copied verbatim into `<plate>.zarr/metadata/`, so the vendor metadata the converters
  do not model travels with the converted plate (#46).

### API Breaking Changes
- **Yokogawa `advanced.channels` entries now carry a required `Channel Key` (the `bts:Ch` number)** and
  override only that channel; the positional, full-length list is no longer accepted, and
  `Advanced` is `YokogawaAcquisitionOptions` with `YokogawaChannelInfoUI` entries.
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
- **`fractal_uzh_converters.{cq3k,cellvoyager}` move to
  `fractal_uzh_converters.yokogawa.{cq3k,cellvoyager}`**, which unifies the two behind one
  shared parser. Package-root imports are unchanged; the old submodule paths are not.
- **CustomTiff `acquisition_details.toml` keys renamed** (follows the v1 field renames):
  `pixelsize` → `xy_pixel_size`, `start_z_coo`/`start_t_coo` → `start_z_space`/`start_t_space`
  (and any other `*_coo` → `*_space`). Existing user `acquisition_details.toml` files must
  be updated. See `docs/converters/custom_tiff.md`.
- **`Advanced` is now the last acquisition field** in all seven converters instead of the
  second.
- Require `ome-zarr-converters-tools[s3]>=1.1.1,<2.0.0` and `ngio>=1.1.2`. The `[s3]`
  extra is now mandatory because v1 makes `s3fs` optional and `s3://` inputs must keep working.
  The `ngio` floor is 1.1.2 because earlier 1.1.x releases could silently lose pixels in
  Dask writes (fixed upstream in `ngio` 1.1.2, required by `ome-zarr-converters-tools` 1.1.0).

### Fix
- All converters: strip whitespace from channel labels and wavelength ids — ngio treats
  `"DAPI"` and `"DAPI "` as two distinct channels, so a stray space silently broke
  per-name channel matching downstream.
- Yokogawa: the tile channel index is now 0-based like the Z and time indices, so an
  `advanced.channels` override no longer fails with a channel-index validation error (#27).
- Yokogawa: build one `AcquisitionDetails` per plate rather than per field of view, so
  fields merged into a single image cannot disagree on it.
- Yokogawa: each well's `bts:TimePoint` values map onto a dense 0-based time axis; the raw
  value counts timelines, so it left leading frames empty.
- Yokogawa: a `bts:Type="ERR"` record no longer makes the whole `.mlf` unparsable. Such
  records are skipped, with one warning per acquisition (#41).
- Yokogawa: warn when the `.mrf` channels disagree on pixel size or frame size — channel
  1's geometry is applied to all of them.
- CQ3K: a single-record `.mlf` now converts, and unknown `bts:` attributes are ignored
  rather than rejected. Both already held on CellVoyager.
- Bug in extended testing using the wrong api
- Fix typo in the conversion log message (`Successfully`).

### Chores
- All converters: what a converter reports about the input data is now a `warnings.warn`
  under a `ConverterWarning` hierarchy instead of a `logger.warning`, so a caller can
  filter or escalate it; the init tasks keep it visible in the task log.
- Drop the `ngff_version=` argument from all seven `setup_images_for_conversion(...)` calls
  (removed in `ome-zarr-converters-tools` 1.1.0; the value already came from `converter_options`).
- Adopt the v1 `AcquisitionDetails` renames in all builders: `pixelsize=` → `xy_pixel_size=`
  and `start_*_coo=`/`length_*_coo=` → `*_space=` (values unchanged). Import `BackendType`
  from the package root instead of the private `models._converter_options`.
- Regenerate `__FRACTAL_MANIFEST__.json` against the v1 `AcquisitionOptions` schema (new
  built-in filters, `grouping`/tiling split, `remove_*` stage corrections, scheduler
  `mode`).
- Regenerate `pixi.lock` from PyPI releases only (the previous lock pointed at a local
  editable path); ngio 1.x no longer pulls in the `distributed` dependency cluster.
- Replace the Yokogawa CQ3K test data with BSSE-CQ3000 acquisitions, the previous data not
  being redistributable; add an extended CellVoyager suite and rename the Yokogawa
  extended datasets to the canonical `hcs_…` convention.
- Tests: rename the stale `init_task_kwargs` parametrize variable to `api_kwargs`.
- Tests: register the `ConverterWarning` ignore from `conftest.py` instead of the pytest
  ini, which imported the package before coverage started and under-reported coverage by
  ~37 points.
- Load the shared snapshot-testing plugin via `pytest_plugins` in `tests/conftest.py` instead of `-p` in `pytest` `addopts`, matching the sibling converters.
- Align repository tooling with `ome-zarr-converters-tools`: adopt its `.pre-commit-config.yaml` (`validate-pyproject` v0.25, `crate-ci/typos`, `astral-sh/ruff-pre-commit` v0.15.17, `nbstripout`) with a per-repo `_typos.toml`, add a `chores` pixi task, bump GitHub Actions pins (`checkout` v7, `codecov-action` v7, `action-gh-release` v3, `setup-python` v6), and trim `CLAUDE.md`.

### Documentation
- Document Yokogawa channel handling: where the labels come from, and that an
  `advanced.channels` override is indexed by instrument channel number rather than by the
  channels present in the output.
- Document the Yokogawa vendor-metadata copy on both converter pages, and drop the
  limited-testing caveats now that both converters have an extended test suite.
- Bring the shared converter-options reference up to date with the v1 schema, including
  the new sharding strategies and runtime settings.
- Fix `tests/data_intake_instructions.md` and `tests/cleanup_test_data.sh`: snapshots are
  JSON, not YAML, and the extended-test template was outdated.

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
