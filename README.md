# Fractal UZH Converters

[![CI (build and test)](https://github.com/fractal-analytics-platform/fractal-uzh-converters/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/fractal-analytics-platform/fractal-uzh-converters/actions/workflows/build_and_test.yml)
[![codecov](https://codecov.io/gh/fractal-analytics-platform/fractal-uzh-converters/graph/badge.svg)](https://codecov.io/gh/fractal-analytics-platform/fractal-uzh-converters)

A collection of [Fractal](https://fractal-analytics-platform.github.io/) tasks to convert High-Content Screening (HCS) plate data from several microscopes into [OME-Zarr](https://ngff.openmicroscopy.org/) format.

## Tasks

| Task | Use case |
|---|---|
| `Convert Operetta Plate to OME-Zarr` | Operetta / Opera Phenix (Revvity) HCS plates. |
| `Convert Evident ScanR Plate to OME-Zarr` | ScanR (Evident) HCS plates. |
| `Convert Yokogawa CQ3K Plate to OME-Zarr` | CQ3K (Yokogawa) HCS plates. |
| `Convert Yokogawa CellVoyager Plate to OME-Zarr` | CellVoyager (Yokogawa) HCS plates. |
| `Convert MD ImageXpress HCS.ai Plate to OME-Zarr` | ImageXpress HCS.ai (Molecular Devices) HCS plates. |
| `Convert Custom TIFF HCS Plate to OME-Zarr` | Custom TIFF HCS plates from any source. |
| `Convert Custom TIFF Images to OME-Zarr` | Custom TIFF single images from any source. |

Each task is a Fractal **compound task**: an init step parses the microscope
metadata and builds the parallelization list, and a compute step writes the
image data well-by-well (or image-by-image).

## Installation

```bash
pip install fractal-uzh-converters
```

## Part of the OME-Zarr converters ecosystem

This converter is a thin, format-specific layer built on
[`ome-zarr-converters-tools`](https://github.com/BioVisionCenter/ome-zarr-converters-tools),
the shared engine that handles tiling, image registration, and OME-Zarr writing for
the whole Fractal converter family. Because they all share that engine, every
converter offers the same options, behavior, and development workflow.

Sibling converters built on the same tooling:

- [`fractal-czi-converters`](https://github.com/fractal-analytics-platform/fractal-czi-converters) — Zeiss `.czi`
- [`fractal-lif-converters`](https://github.com/fractal-analytics-platform/fractal-lif-converters) — Leica `.lif`
- [`fractal-nd2-converters`](https://github.com/fractal-analytics-platform/fractal-nd2-converters) — Nikon `.nd2`

## Documentation

Full documentation — including the supported file layouts, all converter
parameters, and the condition-table format — is available at
<https://fractal-analytics-platform.github.io/fractal-uzh-converters/>.
