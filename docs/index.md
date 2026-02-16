# Fractal UZH Converters

Fractal UZH Converters is a collection of [Fractal](https://fractal-analytics-platform.github.io/) tasks that convert High-Content Screening (HCS) plate data from various microscopes into [OME-Zarr](https://ngff.openmicroscopy.org/) format.

## Supported Microscopes

| Microscope | Manufacturer | Task Name |
|---|---|---|
| Opera Phenix / Operetta | PerkinElmer | `Convert Operetta to OME-Zarr` |
| ScanR | Olympus | `Convert ScanR to OME-Zarr` |
| CQ3K | Yokogawa | `Convert CQ3K to OME-Zarr` |

Each converter reads the microscope's native metadata and image files, then produces a well-structured OME-Zarr HCS plate that can be viewed in tools like [napari](https://napari.org/) or processed with downstream Fractal tasks.

## Installation

```bash
pip install fractal-uzh-converters
```

## How It Works

Each converter is implemented as a Fractal Compoud Task that consists of two steps:

1. **Init task** — Parses the microscope metadata, creates the OME-Zarr plate structure, and generates a parallelization list.
2. **Compute task** — Reads the raw image tiles and writes them into the OME-Zarr dataset. This task runs in parallel across wells.

You configure the init task with one or more **acquisitions** (paths to your raw data directories) and the converter handles the rest.

!!! tip "Condition Tables"
    You can attach experimental metadata (drug treatments, concentrations, replicates, etc.) to wells using a **condition table** CSV file. See the [Condition Tables](condition_tables.md) guide for details.

## Quick Links

- [Converters overview](converters/index.md) — Common parameters and per-microscope guides
- [Condition Tables](condition_tables.md) — How to associate experimental metadata with wells
- [Fractal Analytics Platform](https://fractal-analytics-platform.github.io/) — The task runner used to execute these converters
