# Dataset location and attribution

Original release: Karim et al., [Figshare v2](https://doi.org/10.6084/m9.figshare.27800394.v2).
Dataset licence: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Credit the dataset authors and original paper when reusing their work, as requested
on the source page. This repository preserves the supplied release and identifies
new analysis separately.

The project began with the downloaded release already present. It was reorganized
without modifying the source CSVs, spreadsheets or notebooks:

| Original release path | Current location |
| --- | --- |
| `Dataset/Combined Data/Combined_dataset.csv` | `data/raw/combined/Combined_dataset.csv` |
| `Dataset/Country-Wise Data/*_dataset.csv` | `data/raw/countries/` |
| `Dataset/Combined Data/Summary.xlsx` | `data/reference/Summary.xlsx` |
| `Dataset/Country-Wise Data/Summary_country.xlsx` | `data/reference/Summary_country.xlsx` |
| `Data Collection Sources/DataCollectionSources.xlsx` | `data/reference/DataCollectionSources.xlsx` |
| `Code/*.ipynb` | `notebooks/reference/` |

For a fresh clone, run `python scripts/download_data.py` to restore and hash-verify
the combined CSV from `data/raw/combined_source.zip`. If that archive is absent, the
script downloads the exact source file. `data/raw/country_sources.zip` contains the
five country views. Unpack those only if needed:

```bash
python -m zipfile -e data/raw/country_sources.zip data/raw/
```

Both archives retain the original CSV bytes and are below GitHub's per-file size
limit. Training needs only the combined CSV (~320 MB uncompressed). Extracted CSVs
are ignored by Git. Do not combine the country copies with the combined CSV.

[`source_checksums.json`](source_checksums.json) records SHA-256 for all supplied
source files. The experiment records its input hash in `reports/run_manifest.json`.
See [the data card](../docs/DATA.md) for units, counts and label limitations.
