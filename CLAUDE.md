# CLAUDE.md

Fractal tasks to convert HCS plate data into OME-Zarr. Supports several microscope
systems (Revvity Operetta, Evident ScanR, Yokogawa CQ3K/CellVoyager, Molecular
Devices ImageXpress, custom TIFF). Conversion engine lives in
`ome-zarr-converters-tools` (shared across the sibling converters).

## Commands

All commands need a `pixi run -e dev` / `-e test` prefix (never bare python/pytest/ruff):

- `pixi run -e test pytest tests/` — test suite
- `pixi run -e dev chores` — full gate (ruff format/fix → pytest → pre-commit)
- `pixi run -e dev fractal-manifest check --package fractal_uzh_converters` — validate the Fractal manifest

## Testing

Snapshot-based via the shared `ome_zarr_converters_tools.testing` helper; reference
JSON in `tests/data/*/snapshots/` (one dir per microscope). Regenerate with
`--update-snapshots`. The `--update-snapshots`/`--extended` options, the `extended`
marker, and the `update_snapshots` fixture come from
`ome_zarr_converters_tools.testing.plugin`, loaded via `pytest_plugins` in
`tests/conftest.py` (deliberately not a pytest11 entry point upstream, so coverage
can measure it). `--extended` tests need the git-ignored `tests/data-extended/`.

## Code Style

- Ruff: line length 88, target py311; Google-style docstrings; type-checking via `ty`
- Spell-check via `typos` (false positives go in `_typos.toml`)
- Pydantic v2 models; `@validate_call` on task functions

## Changelog

Always update `CHANGELOG.md` (Features / Fix / API Breaking Changes / Chores / Documentation).
