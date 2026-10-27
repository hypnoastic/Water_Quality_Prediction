"""Create a compact handoff ZIP with code, trained models, reports and slides.

The two original-data archives remain separate in the public repository. This
handoff includes the exact-version downloader so training data can be restored.
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {
    ".github",
    "src",
    "tests",
    "scripts",
    "docs",
    "notebooks",
    "reports",
    "artifacts",
    "configs",
    "examples",
    "presentation",
    "submission",
    "data",
}
FILES = {
    ".gitattributes",
    ".gitignore",
    "README.md",
    "learning.md",
    "Makefile",
    "pyproject.toml",
    "requirements.lock.txt",
    "LICENSE",
    "CITATION.cff",
    "app.py",
}


def publishable(path):
    relative = path.relative_to(ROOT)
    if relative.parts[0] not in DIRECTORIES | FILES:
        return False
    if any(part in {"node_modules", "__pycache__", ".DS_Store"} for part in relative.parts):
        return False
    if path.is_symlink() or path.suffix in {".pyc", ".part"}:
        return False
    if str(relative).startswith("data/raw/") and path.suffix == ".csv":
        return False
    return path.is_file()


def main():
    output = ROOT / "submission/Water_Quality_Submission.zip"
    paths = [
        p
        for p in ROOT.rglob("*")
        if publishable(p) and p != output and not str(p.relative_to(ROOT)).startswith("data/raw/")
    ]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            archive.write(path, Path("Water_Quality_Prediction") / path.relative_to(ROOT))
    print(f"Packaged {len(paths)} files: {output.name} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
