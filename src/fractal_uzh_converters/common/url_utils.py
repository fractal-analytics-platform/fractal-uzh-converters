import os.path
from pathlib import Path

import fsspec.core
from ome_zarr_converters_tools import filesystem_for_url

######################################################################
#
# Custom URL path utilities
# TODO(PR): consider moving these to ome-zarr-converters-tools
#
######################################################################


def url_path_parent(url: str) -> str:
    """Parent directory of a URL path, robust to fwd/back slashes and URL protocols.

    For remote protocols the first path component after the protocol is considered the
    network location (like a bucket name for S3). Examples:
    /path/to/dir/ -> /path/to
    /path/to/file.txt -> /path/to
    / -> ValueError: No parent directory for URL: /
    s3://bucket/dir/ -> s3://bucket
    s3://bucket/dir/file.txt -> s3://bucket/dir
    s3://bucket/ -> ValueError: No parent directory for URL: s3://bucket
    """
    protocol, path = fsspec.core.split_protocol(url)
    path = path.rstrip("\\").rstrip("/")
    if protocol is None:
        # for local and relative paths it's easier to use pathlib
        parent = str(Path(path).parent)
        if parent == path:
            # This means the path is root (e.g., / or C:\), which has no parent
            raise ValueError(f"No parent directory for URL: {url}")
        return parent
    else:
        parent = os.path.dirname(path)
        if parent == "":
            # for remote urls like s3://bucket, the parent of the bucket is not defined
            raise ValueError(f"No parent directory for URL: {url}")
        return f"{protocol}://{parent}"


def url_path_basename(url: str) -> str:
    """Last path component, robust to fwd/back slashes and URL protocols.

    The trailing slash is stripped so that the basename of a directory is returned:
    /path/to/dir/ -> dir
    /path/to/dir -> dir
    /path/to/file.txt -> file.txt
    """
    path = fsspec.core.strip_protocol(url)
    return os.path.basename(path.rstrip("\\").rstrip("/"))


def is_url_path_absolute(url: str) -> bool:
    """Check if a URL path is absolute, robust to fwd/back slashes and URL protocols.

    If the URL starts with a protocol (e.g., "s3://"), it is considered absolute.
    """
    protocol, path = fsspec.core.split_protocol(url)
    # If the URL has a protocol (e.g., "s3://"), we consider it absolute.
    return True if protocol is not None else Path(path).is_absolute()


def join_url_paths(base_url: str, *relative_paths: str) -> str:
    """Join a base URL with multiple relative paths, ensuring proper slashes.

    Uses ``fsspec`` to preserve URL protocols (e.g. ``s3://``) and normalizes
    Windows/UNC backslash separators as well as ``.`` / ``..`` components,
    while never mangling remote URLs.
    """
    protocol, base = fsspec.core.split_protocol(base_url)
    # Unify separators (Windows/UNC) and resolve any "." / ".." segments.
    # normpath also collapses the redundant slashes from the join itself.
    joined = os.path.normpath(f"{base}/{'/'.join(relative_paths)}")
    return joined if protocol is None else f"{protocol}://{joined}"


def url_path_glob(*, base_url: str | None, pattern: str) -> list[str]:
    """Glob a URL path pattern, returning a list of matching URL paths.

    If base_url is None, the pattern is treated as an absolute URL path.
    """
    if base_url is not None:
        pattern = join_url_paths(base_url, pattern)
    fs = filesystem_for_url(pattern)
    protocol = fsspec.core.split_protocol(pattern)[0]
    matched_paths = fs.glob(pattern)
    if protocol is not None:
        return [f"{protocol}://{path}" for path in matched_paths]
    return matched_paths


######################################################################
# end of custom URL path utilities
######################################################################
