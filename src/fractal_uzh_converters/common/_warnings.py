"""The warning categories the converters raise, and how to log them.

What a converter has to report is almost always about the *input data*, so it is a
`warnings.warn` rather than a `logger.warning`: a caller can filter it, escalate it
to an error, or assert on it. Filter the base to catch everything, a subclass to be
specific.
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

    Used directly only for reports that fit none of the subclasses below.
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

    The converter picks one value and says which; it does not reconcile them.
    """


def log_converter_warnings() -> None:
    """Route the converters' warnings into the task log, one line per emission.

    Fractal captures a task's logging output, not its stderr, so `captureWarnings`
    redirects to the `py.warnings` logger. `"always"` keeps a repeated warning from
    being shown once for a whole batch, naming no acquisition; `append=True` makes
    it a fallback, so a caller's own filter still wins.
    """
    logging.captureWarnings(True)
    warnings.filterwarnings("always", category=ConverterWarning, append=True)
