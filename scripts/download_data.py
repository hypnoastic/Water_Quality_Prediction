"""Download the versioned combined CSV, verify it, and leave existing data intact.

Usage: python scripts/download_data.py
Only Python's standard library is required. About 320 MB of disk space is needed.
"""

import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "data/raw/combined/Combined_dataset.csv"
    expected = json.loads((root / "data/source_checksums.json").read_text())[
        "data/raw/combined/Combined_dataset.csv"
    ]["sha256"]
    if target.exists():
        with target.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != expected:
            raise ValueError("Existing combined CSV does not match the documented source hash")
        print(f"Verified existing source: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".csv.part")
    digest = hashlib.sha256()
    try:
        with (
            urllib.request.urlopen(
                "https://ndownloader.figshare.com/files/50757321", timeout=60
            ) as response,
            partial.open("wb") as output,
        ):
            for block in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != expected:
            raise ValueError("Downloaded data do not match the supplied source hash")
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(f"Verified and saved {target}")


if __name__ == "__main__":
    main()
