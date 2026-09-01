# CourtQuant Research Summary

## Research question

Can a dispersion-aware count model improve probabilistic pricing of German handball total-goals markets relative to a standard Poisson benchmark?

## Data and validation design

- Matches in dataset: **1537**
- Minimum initial training sample: **200 matches**
- Rolling window: **306 matches**
- Out-of-sample predictions: **1330**
- Evaluation matchdays: **144**
- Validation: chronological walk-forward backtesting with a matchday embargo
- Primary metric: mean negative log-score, where lower is better

## Out-of-sample model ranking

1. `com_poisson_rolling_306` — mean negative log-score 3.285576, MAE 5.1046, RMSE 6.4759
2. `poisson_rolling_306` — mean negative log-score 3.304016, MAE 5.1046, RMSE 6.4759
3. `com_poisson_expanding` — mean negative log-score 3.323466, MAE 5.2777, RMSE 6.7127
4. `poisson_expanding` — mean negative log-score 3.332047, MAE 5.2777, RMSE 6.7127

## Statistical model comparison

The rolling COM-Poisson model improves the mean negative log-score by **0.558%** relative to rolling Poisson.

- Mean paired score difference: **-0.018440**
- 95% matchday-block bootstrap interval: **[-0.030803, -0.005325]**
- Bootstrap samples: **20000**
- Share of bootstrap samples favouring COM-Poisson: **99.66%**

A negative score difference favours COM-Poisson. The bootstrap frequency is a measure of resampling support, not a Bayesian posterior probability.

## Current rolling-window distribution

- Poisson mean and variance: **58.8954**
- COM-Poisson mean: **58.8954**
- COM-Poisson variance: **49.0627**
- COM-Poisson dispersion index: **0.8330**

The models can share the same expected total while producing different uncertainty estimates and therefore different fair market prices.

## Pricing and model risk

The pricing surface covers total lines from **54.0** to **64.0** goals.

The maximum absolute probability difference between the two models is **215.72 basis points**. Integer lines include push probabilities and fair odds return the stake in the push state.

These are model-implied fair prices. No bookmaker odds are used, so the analysis does not claim a profitable betting strategy.

## Generated artifacts

- `model_summary.csv`
- `bootstrap_summary.csv`
- `pricing_surface.csv`
- `cumulative_score_path.csv`
- `figures/cumulative_score_advantage.png`
- `figures/pricing_surface.png`
