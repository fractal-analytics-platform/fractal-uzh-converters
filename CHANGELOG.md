# Changelog

## [0.6.0]

### Features
- Add `custom_tiff` converter: two new tasks for converting plain TIFF data (HCS plate and single-image) to OME-Zarr.

### Bug Fixes
- Fix `get_attributes_from_condition_table` to dynamically exclude the acquisition column from results.

### Chores
- Bump to `ome-zarr-converters-tools=0.9.0`