# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fractal tasks to convert HCS (High-Content Screening) plate data from microscopes into OME-Zarr format. Supports three microscope systems: Revvity Operetta, Evident ScanR, and Yokogawa CQ3K/CellVoyager.

## Environment & Commands

Uses **pixi** for environment management. Available environments: `default`, `dev`, `test`, `docs`.

```bash
# Run tests
pixi run -e test pytest tests
pixi run -e test pytest tests/test_operetta_task.py       # single test file
pixi run -e test pytest tests/test_operetta_task.py -k "test_name"  # single test

# Lint and format
pixi run -e dev ruff check .
pixi run -e dev ruff check . --fix
pixi run -e dev ruff format .

# Docs
pixi run -e docs mkdocs serve

# Validate Fractal manifest
pixi run -e dev python src/fractal_uzh_converters/dev/task_list.py
```

## Key Dependencies

- `ome-zarr-converters-tools` — Core conversion engine (see below)
- `fractal-task-tools` — Fractal task execution framework
- `ngio` — OME-Zarr I/O library (transitive dependency via converters-tools, not imported directly here)
- For local development of dependencies, editable paths can be uncommented in `pyproject.toml` under `[tool.pixi.pypi-dependencies]`

## Testing

Tests use **snapshot-based assertions** — reference YAML files in `tests/data/` store expected image fingerprints (mean, std, min, max, hash). Use `--update-snapshots` pytest flag to regenerate reference data.

## Code Style

- Ruff with 88-char line length, Google-style docstrings
- Pydantic models with `@validate_call` on task functions
- Python 3.10+ type hint syntax (`list[T]`, `str | None`)
- Pre-commit hooks: `validate-pyproject`, `ruff`, `ruff-format`
