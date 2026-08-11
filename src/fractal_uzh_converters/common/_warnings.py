"""The warning categories the converters raise, and how to log them.

Almost everything a converter has to report is about the *input data*: a vendor
metadata file that is absent, an attribute that is malformed, two channels that
disagree about the pixel size. That is what `warnings.warn` is for. A
`logger.warning` cannot be filtered, promoted to an error, or asserted on by a
caller — a warning can, which matters most for the API entry points, where the
caller is a script rather than the Fractal runner.

Neither `ome-zarr-converters-tools` nor `ngio` defines a `Warning` subclass, so
this hierarchy is the repo's own. It is deliberately shallow: the base exists so
that a user can silence or escalate *everything* the converters say with one
filter, and the three subclasses exist so that a caller who only cares about,
say, channel metadata can be that specific.
"""

import logging
import warnings

__all__ = [
    "ChannelMetadataWarning",
    "ConverterWarning",
    "GeometryWarning",
    "SourceMetadataWarning",
    "log_converter_warnings",
]


class ConverterWarning(UserWarning):
    """Something about the acquisition that the conversion worked around.

    Never raised on its own where a more specific subclass fits; the base is used
    directly only for reports that belong to no single category.
    """


class SourceMetadataWarning(ConverterWarning):
    """A vendor metadata file is missing, unreadable, or could not be copied.

    The conversion continues without whatever that file would have contributed.
    """


class ChannelMetadataWarning(ConverterWarning):
    """A channel's label, wavelength id or colour could not be taken at face value.

    The channel is still converted, under a fallback name or without its colour.
    """


class GeometryWarning(ConverterWarning):
    """The acquisition's declared geometry is not self-consistent.

    Pixel sizes that differ between x and y, a Z spacing that is not constant, or
    two channels declaring different frame sizes. The converter picks one value
    and says which; it does not attempt to reconcile them.
    """


def log_converter_warnings() -> None:
    """Route the converters' warnings into the task log, one line per emission.

    Fractal runs each task in its own process and captures its logging output, so
    a warning printed to `stderr` by the default `showwarning` would be lost.
    `logging.captureWarnings` redirects it to the `py.warnings` logger instead,
    which has no handler of its own and therefore propagates to the root handler
    `fractal_task_tools` installs.

    The `"always"` filter is not redundant. Python's default action shows a given
    message only once per source line per process, so a batch of ten acquisitions
    that all have a non-constant Z spacing would report it once, naming no
    acquisition. Warnings that *should* be aggregated are aggregated at the site
    that knows the count — see the `bts:Type="ERR"` record tally in
    `yokogawa/_parse.py`.

    It is appended rather than prepended, which makes it a *fallback*: it applies
    only when nothing has already configured the category, so a caller's own
    `filterwarnings` (or this repo's `ignore::ConverterWarning` under pytest) still
    wins. Appending is also idempotent, so the seven init tasks add one entry
    between them rather than seven.

    Both settings are process-global, which is what a one-shot task process wants.
    Under pytest each test runs inside a `warnings.catch_warnings` block, so the
    call is undone at the end of any test that reaches it.
    """
    logging.captureWarnings(True)
    warnings.filterwarnings("always", category=ConverterWarning, append=True)
