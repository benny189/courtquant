# Data

CourtQuant uses historical German handball match results for probabilistic count modelling and model-risk analysis.

## Expected raw file

Place the local source file at:

```text
data/raw/handball.csv
```

The file must be semicolon-separated and encoded as UTF-8 or UTF-8 with BOM.

## Expected schema

| Column | Description |
|---|---|
| `saison` | Season identifier such as `18/19` |
| `spieltag` | Positive matchday number |
| `spiel` | Positive match number within the matchday |
| `tore_mannschaft1` | Non-negative goals scored by team 1 |
| `tore_mannschaft2` | Non-negative goals scored by team 2 |

Each combination of `saison`, `spieltag` and `spiel` must be unique. Missing values, duplicate rows and additional columns are rejected.

## Data policy

Raw and processed datasets are excluded from version control. The source dataset is not redistributed until its licensing and redistribution rights have been confirmed.

The committed research outputs contain derived model results rather than the original match-level source data.