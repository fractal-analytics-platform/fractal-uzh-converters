"""A collection of fractal tasks to convert HCS Plates to OME-Zarr"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fractal-uzh-converters")
except PackageNotFoundError:
    __version__ = "uninstalled"
