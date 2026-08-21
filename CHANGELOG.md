# Changelog

## [Unreleased]

Migration to `ome-zarr-converters-tools` v1.

### API Breaking Changes
- **CustomTiff `acquisition_details.toml` keys renamed** (follows the v1 field renames):
  `pixelsize` → `xy_pixel_size`, `start_z_coo`/`start_t_coo` → `start_z_space`/`start_t_space`
  (and any other `*_coo` → `*_space`). Existing user `acquisition_details.toml` files must
  be updated. See `docs/converters/custom_tiff.md`.

### Chores
- Migrate to `ngio` 1.1: pin `ngio==1.1.0a2` and take `ome-zarr-converters-tools[s3]`
  from its `refactor/ngio-1.1` git branch (both temporary until the final releases;
  the lock drops the `distributed` dependency cluster that ngio 1.x no longer needs).
  No converter code touches ngio directly, so no ngio API changes were needed.
- Drop the deprecated `ngff_version=` argument from all seven
  `setup_images_for_conversion(...)` calls (deprecated in `ome-zarr-converters-tools`
  1.0.2; the value is taken from `converter_options.omezarr_options.ngff_version`,
  which is exactly what was being passed).
- Temporarily filter the ngio `NgioFutureWarning` about plate-wide `max_workers`:
  `plate.get_images(max_workers="auto")` in ngio 1.1.0a2 still warns on multi-well
  plates despite the opt-in (upstream bug, fires via the converters-tools snapshot
  helper). Remove the filter once ngio forwards `max_workers` to `images_paths`.
- Bump `ome-zarr-converters-tools` to `[s3]>=1.0.0,<2.0.0`. The `[s3]` extra is now
  required: v1 makes `s3fs` optional, so it is pinned here to keep `s3://` inputs working
  (it was previously pulled in transitively).
- Adopt the v1 `AcquisitionDetails` renames in all builders: `pixelsize=` → `xy_pixel_size=`
  and `start_*_coo=`/`length_*_coo=` → `*_space=` (values unchanged). Import `BackendType`
  from the package root instead of the private `models._converter_options`.
- Regenerate `__FRACTAL_MANIFEST__.json` against the v1 `AcquisitionOptions` schema (new
  built-in filters, `grouping`/tiling split, `remove_*` stage corrections, scheduler
  `mode`).

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
