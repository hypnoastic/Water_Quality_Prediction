# Data source

Karim et al., [A Comprehensive Surface Water Quality Monitoring Dataset
(1940–2023), Figshare version 2](https://doi.org/10.6084/m9.figshare.27800394.v2).
The release uses [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Training uses only `data/raw/combined/Combined_dataset.csv`: 2,827,977 rows,
14 columns and about 320 MB. Restore or verify it with:

```bash
python scripts/download_data.py
```

The script downloads the exact release file and checks its SHA-256. Extracted data
stay local and are ignored by Git. Country CSVs, data archives and source summary
spreadsheets were removed because they repeat information already in the combined
CSV and generated audit. The original release remains available at the DOI.

`source_checksums.json` verifies the two original files retained for research:
the combined training CSV and the unchanged harmonization notebook in
`notebooks/reference/`. That notebook documents the source-derived labels,
including the denominator and objective directions.

See [docs/DATA.md](../docs/DATA.md) for units, target construction, quality rules and
scientific limitations. Inference uses the saved models and needs no raw CSV.
