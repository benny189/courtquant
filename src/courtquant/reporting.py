"""Research tables and charts for CourtQuant results."""

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from courtquant.backtesting import (
    summarize_backtest,
)
from courtquant.evaluation import (
    PairedBootstrapResult,
)
from courtquant.pricing import (
    TotalProbabilityModel,
    build_total_distribution,
    compare_total_prices,
    price_total_market,
)

PAIRING_COLUMNS: Final[tuple[str, ...]] = (
    "observation_index",
    "saison",
    "spieltag",
    "spiel",
)
PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "model",
    *PAIRING_COLUMNS,
    "negative_log_score",
)
BASELINE_COLOR: Final[str] = "#315C8A"
CANDIDATE_COLOR: Final[str] = "#D46A3A"
REFERENCE_COLOR: Final[str] = "#6B7280"


class ReportingValidationError(ValueError):
    """Raised when research output cannot be built safely."""


def build_model_summary(
    prediction_sets: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Build a ranked table of backtest metrics."""
    if not prediction_sets:
        raise ReportingValidationError("At least one prediction set is required.")

    summary = pd.DataFrame.from_records(
        [asdict(summarize_backtest(predictions)) for predictions in prediction_sets]
    )
    summary = summary.sort_values(
        "mean_negative_log_score",
        kind="stable",
    ).reset_index(drop=True)
    summary.insert(
        0,
        "rank",
        range(1, len(summary) + 1),
    )
    return summary


def build_bootstrap_summary(
    result: PairedBootstrapResult,
) -> pd.DataFrame:
    """Convert a paired bootstrap result to one table row."""
    return pd.DataFrame.from_records([asdict(result)])


def build_pricing_surface(
    baseline_model: TotalProbabilityModel,
    candidate_model: TotalProbabilityModel,
    lines: Sequence[float],
    *,
    baseline_name: str,
    candidate_name: str,
    maximum_total: int = 200,
    tail_tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Build fair prices and model-risk measures by line."""
    _validate_model_names(
        baseline_name,
        candidate_name,
    )
    if not lines:
        raise ReportingValidationError("At least one total line is required.")

    baseline_distribution = build_total_distribution(
        baseline_model,
        maximum_total=maximum_total,
        tail_tolerance=tail_tolerance,
    )
    candidate_distribution = build_total_distribution(
        candidate_model,
        maximum_total=maximum_total,
        tail_tolerance=tail_tolerance,
    )

    rows: list[dict[str, object]] = []
    seen_lines: set[float] = set()
    for line in lines:
        baseline_price = price_total_market(
            baseline_distribution,
            line,
        )
        candidate_price = price_total_market(
            candidate_distribution,
            line,
        )
        numeric_line = baseline_price.line

        if numeric_line in seen_lines:
            raise ReportingValidationError("Pricing lines must be unique.")
        seen_lines.add(numeric_line)

        risk = compare_total_prices(
            baseline_price,
            candidate_price,
        )
        rows.append(
            {
                "line": numeric_line,
                "baseline_model": baseline_name,
                "candidate_model": candidate_name,
                "baseline_under_probability": (baseline_price.under_probability),
                "candidate_under_probability": (candidate_price.under_probability),
                "under_difference_bps": (risk.under_probability_difference * 10_000.0),
                "baseline_push_probability": (baseline_price.push_probability),
                "candidate_push_probability": (candidate_price.push_probability),
                "push_difference_bps": (risk.push_probability_difference * 10_000.0),
                "baseline_over_probability": (baseline_price.over_probability),
                "candidate_over_probability": (candidate_price.over_probability),
                "over_difference_bps": (risk.over_probability_difference * 10_000.0),
                "baseline_under_fair_odds": (baseline_price.under_fair_decimal_odds),
                "candidate_under_fair_odds": (candidate_price.under_fair_decimal_odds),
                "baseline_over_fair_odds": (baseline_price.over_fair_decimal_odds),
                "candidate_over_fair_odds": (candidate_price.over_fair_decimal_odds),
                "maximum_model_risk_bps": (risk.maximum_absolute_difference_basis_points),
            }
        )

    return (
        pd.DataFrame.from_records(rows)
        .sort_values(
            "line",
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_cumulative_score_path(
    baseline_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build the cumulative candidate log-score advantage."""
    baseline, baseline_model = _validated_prediction_subset(
        baseline_predictions,
        "baseline",
    )
    candidate, candidate_model = _validated_prediction_subset(
        candidate_predictions,
        "candidate",
    )

    if baseline_model == candidate_model:
        raise ReportingValidationError("Baseline and candidate models must differ.")

    paired = baseline.merge(
        candidate,
        on=list(PAIRING_COLUMNS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not bool(paired["_merge"].eq("both").all()):
        raise ReportingValidationError(
            "Models must contain scores for exactly the same observations."
        )

    paired = (
        paired.drop(columns="_merge")
        .sort_values(
            "observation_index",
            kind="stable",
        )
        .reset_index(drop=True)
    )
    paired["baseline_model"] = baseline_model
    paired["candidate_model"] = candidate_model
    paired["candidate_score_advantage"] = (
        paired["baseline_negative_log_score"] - paired["candidate_negative_log_score"]
    )
    paired["cumulative_candidate_score_advantage"] = paired["candidate_score_advantage"].cumsum()

    return paired.loc[
        :,
        [
            *PAIRING_COLUMNS,
            "baseline_model",
            "candidate_model",
            "baseline_negative_log_score",
            "candidate_negative_log_score",
            "candidate_score_advantage",
            ("cumulative_candidate_score_advantage"),
        ],
    ]


def save_cumulative_score_chart(
    cumulative_path: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save the cumulative probabilistic advantage chart."""
    required_columns = (
        "observation_index",
        "baseline_model",
        "candidate_model",
        "cumulative_candidate_score_advantage",
    )
    _require_table_columns(
        cumulative_path,
        required_columns,
        "Cumulative score path",
    )
    baseline_model = _single_model_value(
        cumulative_path,
        "baseline_model",
    )
    candidate_model = _single_model_value(
        cumulative_path,
        "candidate_model",
    )

    figure, axis = plt.subplots(figsize=(10.0, 5.5))
    x_values = cumulative_path["observation_index"].to_numpy(dtype=np.float64)
    cumulative_advantage = cumulative_path["cumulative_candidate_score_advantage"].to_numpy(
        dtype=np.float64
    )

    axis.plot(
        x_values,
        cumulative_advantage,
        color=CANDIDATE_COLOR,
        linewidth=2.2,
        label=(f"{candidate_model} advantage over {baseline_model}"),
    )
    axis.axhline(
        0.0,
        color=REFERENCE_COLOR,
        linewidth=1.0,
        linestyle="--",
    )
    axis.set_title("Cumulative out-of-sample log-score advantage")
    axis.set_xlabel("Chronological prediction index")
    axis.set_ylabel("Cumulative log-score advantage")
    axis.grid(
        alpha=0.2,
        linewidth=0.8,
    )
    axis.legend(
        frameon=False,
        loc="best",
    )
    return _save_figure(
        figure,
        output_path,
    )


def save_pricing_surface_chart(
    pricing_surface: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save fair Over probabilities across total lines."""
    required_columns = (
        "line",
        "baseline_model",
        "candidate_model",
        "baseline_over_probability",
        "candidate_over_probability",
    )
    _require_table_columns(
        pricing_surface,
        required_columns,
        "Pricing surface",
    )
    baseline_model = _single_model_value(
        pricing_surface,
        "baseline_model",
    )
    candidate_model = _single_model_value(
        pricing_surface,
        "candidate_model",
    )

    figure, axis = plt.subplots(figsize=(10.0, 5.5))
    lines = pricing_surface["line"].to_numpy(dtype=np.float64)
    baseline_probabilities = pricing_surface["baseline_over_probability"].to_numpy(dtype=np.float64)
    candidate_probabilities = pricing_surface["candidate_over_probability"].to_numpy(
        dtype=np.float64
    )

    axis.plot(
        lines,
        baseline_probabilities,
        color=BASELINE_COLOR,
        linewidth=2.0,
        marker="o",
        label=baseline_model,
    )
    axis.plot(
        lines,
        candidate_probabilities,
        color=CANDIDATE_COLOR,
        linewidth=2.0,
        marker="o",
        label=candidate_model,
    )
    axis.set_title("Fair Over probability by market line")
    axis.set_xlabel("Total-goals line")
    axis.set_ylabel("Fair Over probability")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(
        alpha=0.2,
        linewidth=0.8,
    )
    axis.legend(
        frameon=False,
        loc="best",
    )
    return _save_figure(
        figure,
        output_path,
    )


def _validated_prediction_subset(
    predictions: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, str]:
    """Validate and select one model's score data."""
    if predictions.empty:
        raise ReportingValidationError(f"{label.capitalize()} predictions cannot be empty.")

    missing_columns = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    if missing_columns:
        raise ReportingValidationError(
            f"{label.capitalize()} predictions are missing columns: {missing_columns}"
        )

    required = predictions.loc[
        :,
        list(PREDICTION_COLUMNS),
    ]
    if bool(required.isna().any().any()):
        raise ReportingValidationError(
            f"{label.capitalize()} predictions cannot contain missing values."
        )

    if bool(
        predictions.duplicated(
            subset=list(PAIRING_COLUMNS),
        ).any()
    ):
        raise ReportingValidationError(
            f"{label.capitalize()} predictions contain duplicate observations."
        )

    model_names = predictions["model"].astype("string").unique().tolist()
    if len(model_names) != 1:
        raise ReportingValidationError(
            f"{label.capitalize()} predictions must contain exactly one model."
        )

    scores = predictions["negative_log_score"].to_numpy(dtype=np.float64)
    if not bool(np.isfinite(scores).all()):
        raise ReportingValidationError(f"{label.capitalize()} scores must be finite.")

    subset = predictions.loc[
        :,
        [
            *PAIRING_COLUMNS,
            "negative_log_score",
        ],
    ].rename(columns={"negative_log_score": (f"{label}_negative_log_score")})
    return subset, str(model_names[0])


def _validate_model_names(
    baseline_name: str,
    candidate_name: str,
) -> None:
    """Validate labels used in research outputs."""
    if not baseline_name.strip() or not candidate_name.strip():
        raise ReportingValidationError("Model names cannot be empty.")

    if baseline_name == candidate_name:
        raise ReportingValidationError("Baseline and candidate names must differ.")


def _require_table_columns(
    table: pd.DataFrame,
    required_columns: Sequence[str],
    label: str,
) -> None:
    """Validate a table before plotting."""
    if table.empty:
        raise ReportingValidationError(f"{label} cannot be empty.")

    missing_columns = sorted(set(required_columns) - set(table.columns))
    if missing_columns:
        raise ReportingValidationError(f"{label} is missing columns: {missing_columns}")


def _single_model_value(
    table: pd.DataFrame,
    column: str,
) -> str:
    """Return one unique model label from a table."""
    model_names = table[column].astype("string").unique().tolist()
    if len(model_names) != 1:
        raise ReportingValidationError(f"{column} must contain exactly one model.")

    return str(model_names[0])


def _save_figure(
    figure: Figure,
    output_path: str | Path,
) -> Path:
    """Save and close one research figure."""
    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.tight_layout()
    figure.savefig(
        destination,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return destination
