"""Generate the complete CourtQuant research report."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd

from courtquant.backtesting import (
    walk_forward_com_poisson,
    walk_forward_poisson,
)
from courtquant.data import load_matches
from courtquant.diagnostics import add_match_features
from courtquant.evaluation import (
    PairedBootstrapResult,
    paired_matchday_bootstrap,
)
from courtquant.models.com_poisson import (
    ConwayMaxwellPoissonModel,
)
from courtquant.models.poisson import (
    PoissonTotalModel,
)
from courtquant.reporting import (
    build_bootstrap_summary,
    build_cumulative_score_path,
    build_model_summary,
    build_pricing_surface,
    save_cumulative_score_chart,
    save_pricing_surface_chart,
)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=("Generate CourtQuant model evaluation and pricing artifacts.")
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/handball.csv"),
        help="Path to the raw handball CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports"),
        help="Directory for generated artifacts.",
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=200,
        help="Minimum observations before evaluation.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=306,
        help="Rolling training-window size.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=20_000,
        help="Number of matchday bootstrap samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the bootstrap.",
    )
    return parser


def _latest_training_counts(
    matches: pd.DataFrame,
    window_size: int,
) -> list[int]:
    """Return the most recent chronological totals."""
    featured = add_match_features(matches)
    featured["_season_start"] = featured["saison"].str[:2].astype("int64")
    ordered = featured.sort_values(
        [
            "_season_start",
            "spieltag",
            "spiel",
        ],
        kind="stable",
    )

    if len(ordered) < window_size:
        raise ValueError("The dataset contains fewer matches than the requested rolling window.")

    return [int(value) for value in ordered["total_goals"].tail(window_size)]


def _model_ranking_text(
    model_summary: pd.DataFrame,
) -> str:
    """Format the out-of-sample model ranking."""
    ranking: list[str] = []

    for position in range(len(model_summary)):
        row = model_summary.iloc[position]
        ranking.append(
            f"{int(row['rank'])}. "
            f"`{row['model']}` — "
            "mean negative log-score "
            f"{float(row['mean_negative_log_score']):.6f}, "
            f"MAE {float(row['mean_absolute_error']):.4f}, "
            "RMSE "
            f"{float(row['root_mean_squared_error']):.4f}"
        )

    return "\n".join(ranking)


def _write_research_summary(
    destination: Path,
    *,
    matches: int,
    min_train_size: int,
    window_size: int,
    model_summary: pd.DataFrame,
    bootstrap: PairedBootstrapResult,
    poisson_mean: float,
    com_mean: float,
    com_variance: float,
    pricing_surface: pd.DataFrame,
) -> None:
    """Write the human-readable research summary."""
    maximum_model_risk = float(pricing_surface["maximum_model_risk_bps"].max())
    minimum_line = float(pricing_surface["line"].min())
    maximum_line = float(pricing_surface["line"].max())
    dispersion_index = com_variance / com_mean
    ranking = _model_ranking_text(model_summary)

    content = f"""# CourtQuant Research Summary

## Research question

Can a dispersion-aware count model improve probabilistic pricing of German handball total-goals markets relative to a standard Poisson benchmark?

## Data and validation design

- Matches in dataset: **{matches}**
- Minimum initial training sample: **{min_train_size} matches**
- Rolling window: **{window_size} matches**
- Out-of-sample predictions: **{bootstrap.predictions}**
- Evaluation matchdays: **{bootstrap.matchdays}**
- Validation: chronological walk-forward backtesting with a matchday embargo
- Primary metric: mean negative log-score, where lower is better

## Out-of-sample model ranking

{ranking}

## Statistical model comparison

The rolling COM-Poisson model improves the mean negative log-score by **{bootstrap.relative_improvement_percent:.3f}%** relative to rolling Poisson.

- Mean paired score difference: **{bootstrap.mean_score_difference:.6f}**
- 95% matchday-block bootstrap interval: **[{bootstrap.confidence_interval_low:.6f}, {bootstrap.confidence_interval_high:.6f}]**
- Bootstrap samples: **{bootstrap.bootstrap_samples}**
- Share of bootstrap samples favouring COM-Poisson: **{bootstrap.candidate_win_probability * 100.0:.2f}%**

A negative score difference favours COM-Poisson. The bootstrap frequency is a measure of resampling support, not a Bayesian posterior probability.

## Current rolling-window distribution

- Poisson mean and variance: **{poisson_mean:.4f}**
- COM-Poisson mean: **{com_mean:.4f}**
- COM-Poisson variance: **{com_variance:.4f}**
- COM-Poisson dispersion index: **{dispersion_index:.4f}**

The models can share the same expected total while producing different uncertainty estimates and therefore different fair market prices.

## Pricing and model risk

The pricing surface covers total lines from **{minimum_line:.1f}** to **{maximum_line:.1f}** goals.

The maximum absolute probability difference between the two models is **{maximum_model_risk:.2f} basis points**. Integer lines include push probabilities and fair odds return the stake in the push state.

These are model-implied fair prices. No bookmaker odds are used, so the analysis does not claim a profitable betting strategy.

## Generated artifacts

- `model_summary.csv`
- `bootstrap_summary.csv`
- `pricing_surface.csv`
- `cumulative_score_path.csv`
- `figures/cumulative_score_advantage.png`
- `figures/pricing_surface.png`
"""

    destination.write_text(
        content,
        encoding="utf-8",
    )


def generate_report(
    args: Namespace,
) -> Path:
    """Run the complete CourtQuant research pipeline."""
    data_path = Path(args.data)
    output_directory = Path(args.output)
    min_train_size = int(args.min_train_size)
    window_size = int(args.window_size)
    bootstrap_samples = int(args.bootstrap_samples)
    seed = int(args.seed)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    matches = load_matches(str(data_path))

    poisson_expanding = walk_forward_poisson(
        matches,
        min_train_size=min_train_size,
    )
    poisson_rolling = walk_forward_poisson(
        matches,
        min_train_size=min_train_size,
        window_size=window_size,
    )
    com_expanding = walk_forward_com_poisson(
        matches,
        min_train_size=min_train_size,
    )
    com_rolling = walk_forward_com_poisson(
        matches,
        min_train_size=min_train_size,
        window_size=window_size,
    )

    model_summary = build_model_summary(
        [
            poisson_expanding,
            poisson_rolling,
            com_expanding,
            com_rolling,
        ]
    )
    bootstrap = paired_matchday_bootstrap(
        poisson_rolling,
        com_rolling,
        bootstrap_samples=bootstrap_samples,
        confidence_level=0.95,
        seed=seed,
    )
    bootstrap_summary = build_bootstrap_summary(bootstrap)
    cumulative_path = build_cumulative_score_path(
        poisson_rolling,
        com_rolling,
    )

    latest_counts = _latest_training_counts(
        matches,
        window_size,
    )
    poisson_model = PoissonTotalModel.fit(latest_counts)
    com_model = ConwayMaxwellPoissonModel.fit(latest_counts)
    com_mean, com_variance = com_model.moments()
    central_line = round(com_mean * 2.0) / 2.0
    pricing_lines = [central_line + offset * 0.5 for offset in range(-10, 11)]
    pricing_surface = build_pricing_surface(
        poisson_model,
        com_model,
        pricing_lines,
        baseline_name=(f"poisson_rolling_{window_size}"),
        candidate_name=(f"com_poisson_rolling_{window_size}"),
    )

    model_summary.to_csv(
        output_directory / "model_summary.csv",
        index=False,
    )
    bootstrap_summary.to_csv(
        output_directory / "bootstrap_summary.csv",
        index=False,
    )
    pricing_surface.to_csv(
        output_directory / "pricing_surface.csv",
        index=False,
    )
    cumulative_path.to_csv(
        output_directory / "cumulative_score_path.csv",
        index=False,
    )

    save_cumulative_score_chart(
        cumulative_path,
        output_directory / "figures" / "cumulative_score_advantage.png",
    )
    save_pricing_surface_chart(
        pricing_surface,
        output_directory / "figures" / "pricing_surface.png",
    )

    summary_path = output_directory / "research_summary.md"
    _write_research_summary(
        summary_path,
        matches=len(matches),
        min_train_size=min_train_size,
        window_size=window_size,
        model_summary=model_summary,
        bootstrap=bootstrap,
        poisson_mean=poisson_model.rate,
        com_mean=com_mean,
        com_variance=com_variance,
        pricing_surface=pricing_surface,
    )

    print(model_summary.to_string(index=False))
    print(f"\nResearch report written to {summary_path}")
    return summary_path


def main() -> None:
    """Run the report generator from the command line."""
    parser = _build_parser()
    generate_report(parser.parse_args())


if __name__ == "__main__":
    main()
