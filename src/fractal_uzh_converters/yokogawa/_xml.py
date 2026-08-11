"""Reading the Yokogawa BTS XML documents (`.mlf`, `.mrf`, `.mes`, `.wpi`, `.wpp`)."""

from typing import Any

import xmltodict
from ome_zarr_converters_tools import filesystem_for_url

BTS_NAMESPACE = "http://www.yokogawa.co.jp/BTS/BTSSchema/1.0"


def parse_bts_xml(url: str) -> dict[str, Any]:
    """Parse a BTS XML document, stripping the `bts:` namespace.

    Raises whatever the read or the parse raises; callers decide what a missing or
    malformed file means for them.
    """
    fs = filesystem_for_url(url)
    with fs.open(url, encoding="utf-8") as f:
        return xmltodict.parse(
            f.read(),
            process_namespaces=True,
            namespaces={BTS_NAMESPACE: None},
            attr_prefix="",
            cdata_key="Value",
        )
