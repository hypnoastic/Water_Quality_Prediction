"""Remove stale, unreferenced content-type declarations from a generated PPTX.

PptxGenJS 4.0.1 can emit slideMaster overrides for absent master parts. Removing
these declarations repairs package integrity without changing a slide or its data.
"""

import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def repair(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(parts["[Content_Types].xml"])
    stale = [
        node
        for node in root
        if node.tag.endswith("Override") and node.attrib["PartName"].lstrip("/") not in parts
    ]
    for node in stale:
        root.remove(node)
    if stale:
        ET.register_namespace("", "http://schemas.openxmlformats.org/package/2006/content-types")
        parts["[Content_Types].xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        candidate = path.with_suffix(".fixed.pptx")
        with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in parts.items():
                archive.writestr(name, content)
        candidate.replace(path)
    print(f"Removed {len(stale)} stale content-type declarations.")


if __name__ == "__main__":
    repair(sys.argv[1])
