# CourtQuant

A reproducible probabilistic pricing and model-risk engine for handball goal markets.

CourtQuant investigates how strongly simplified count models can misestimate
event probabilities when score data exhibit dispersion, dependence and
time-varying behaviour. The project develops the complete workflow from data
validation and statistical diagnostics to out-of-sample evaluation and
probability-based pricing.

> **Status:** Active research project. The repository is currently private
> while the data pipeline and baseline model are being validated.

## Research question

How much does a textbook Poisson model misprice handball goal markets when its
assumptions of equidispersion, independence and stationarity are violated?

## Research hypotheses

1. A stationary Poisson model misestimates tail probabilities when the
   variance of observed scores differs materially from the mean.
2. Ignoring score dependence and season-level regime shifts reduces
   out-of-sample calibration.
3. More flexible count models improve probabilistic forecasts, particularly
   for goal totals and handicap thresholds.

## Planned methodology

### Data integrity

- Explicit schema and type validation
- Missing-value and duplicate detection
- Reproducible anomaly reporting
- Strict separation of raw and processed data

### Statistical models

- Stationary Poisson baseline
- Season-aware Poisson model
- Skellam model for score differences
- Conway-Maxwell-Poisson model for flexible dispersion
- Dependent score models as a later extension

### Evaluation

- Expanding-window and season-based backtesting
- Logarithmic score and Brier score
- Calibration diagnostics
- Empirical and model-implied distribution comparisons
- Bootstrap and distribution-free uncertainty bounds

### Pricing layer

The fitted distributions will be translated into fair probabilities for
markets such as:

- Total goals over/under a specified threshold
- Goal ranges
- Score differences
- Handicap thresholds
- Extreme-score tail events

## Repository structure

```text
courtquant/
├── data/                 # Local raw data and generated datasets
├── notebooks/            # Reproducible research notebooks
├── reports/figures/      # Publication-ready figures
├── src/courtquant/       # Tested production code
├── tests/                # Automated test suite
├── pyproject.toml        # Project and quality configuration
└── uv.lock               # Fully resolved dependency versions
```

## Local setup

CourtQuant uses [uv](https://docs.astral.sh/uv/) for dependency and environment
management.

```bash
git clone https://github.com/benny189/courtquant.git
cd courtquant
uv sync
```

Place the source dataset at:

```text
data/raw/handball.csv
```

Then run the quality checks:

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

Start the research environment with:

```bash
uv run jupyter lab
```

## Data availability

The source dataset is not included in the repository until redistribution
rights have been confirmed. See [`data/README.md`](data/README.md) for the
expected schema and data-handling policy.

## Reproducibility

All model logic will be implemented in the installable `courtquant` package.
Notebooks are reserved for transparent analysis and presentation. Dependency
versions are locked in `uv.lock`, and all production code is checked using
Ruff, Mypy and Pytest.

## Disclaimer

This project is intended solely for statistical research and education. It
does not provide betting recommendations or financial advice.

## License

The source code is released under the [MIT License](LICENSE).