# Fractal UZH Converters

A collection of [Fractal](https://fractal-analytics-platform.github.io/) tasks to convert High-Content Screening (HCS) plate data from various microscopes into [OME-Zarr](https://ngff.openmicroscopy.org/) format.

## Supported Microscopes

| Microscope | Manufacturer | Task Name |
|---|---|---|
| Operetta / Opera Phenix | Revvity | `Convert Operetta Plate to OME-Zarr` |
| ScanR | Evident | `Convert Evident ScanR Plate to OME-Zarr` |
| CQ3K | Yokogawa | `Convert Yokogawa CQ3K Plate to OME-Zarr` |
| CellVoyager | Yokogawa | `Convert Yokogawa CellVoyager Plate to OME-Zarr` |
| ImageXpress HCS.ai | Molecular Devices | `Convert MD ImageXpress HCS.ai Plate to OME-Zarr` |
| Custom TIFF (HCS plate) | Any | `Convert Custom TIFF HCS Plate to OME-Zarr` |
| Custom TIFF (single images) | Any | `Convert Custom TIFF Images to OME-Zarr` |

Each converter reads the microscope's native metadata and image files, then produces a well-structured OME-Zarr HCS plate.
## Installation

```bash
pip install fractal-uzh-converters
```

## How It Works

Each converter is implemented as a Fractal Compound Task consisting of two steps:

1. **Init task** — Parses the microscope metadata, creates the OME-Zarr plate structure, and generates a parallelization list.
2. **Compute task** — Reads the raw image tiles and writes them into the OME-Zarr dataset. This task runs in parallel across wells.

You configure the init task with one or more **acquisitions** (paths to your raw data directories) and the converter handles the rest.

## Key Features

- **Multiple tiling modes** — Snap to grid, snap to corners, inplace, or no tiling depending on your acquisition layout.
- **Condition tables** — Attach experimental metadata (drug treatments, concentrations, replicates) to wells via a CSV file.
- **Flexible writer modes** — Choose between per-FOV, per-tile, Dask-parallel, or in-memory writing strategies to balance speed and memory usage.
- **Overwrite control** — No overwrite, full overwrite, or extend mode to incrementally add acquisitions.
- **OME-NGFF 0.4 and 0.5** — Target either specification version for the output.

## Documentation

Full documentation is available at: https://fractal-analytics-platform.github.io/fractal-uzh-converters/
