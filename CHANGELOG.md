# Changelog

## [Unreleased]

### Features
- Add Python API functions (`convert_cellvoyager`, `convert_cq3k`, `convert_operetta`, `convert_imagexpress_hcs`, `convert_hcs_tiff`, `convert_single_tiff`) for programmatic use outside Fractal, following the existing `convert_scanr` pattern.
- Update test utilities to call the high-level API functions, providing end-to-end coverage of both the API layer and the underlying init/compute tasks.

## [0.6.0]

### Features
- Add `custom_tiff` converter: two new tasks for converting plain TIFF data (HCS plate and single-image) to OME-Zarr.

### Bug Fixes
- Fix `get_attributes_from_condition_table` to dynamically exclude the acquisition column from results.

### Chores
- Bump to `ome-zarr-converters-tools=0.9.0`