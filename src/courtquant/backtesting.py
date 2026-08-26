"""Leakage-resistant walk-forward backtesting for CourtQuant models."""

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from courtquant.diagnostics import add_match_features
from courtquant.models.poisson import PoissonTotalModel

PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "model",
    "observation_index",
    "saison",
    "spieltag",
    "spiel",
    "train_start",
    "train_size",
    "observed_total",
    "predicted_rate",
    "log_probability",
    "negative_log_score",
    "forecast_error",
)


class BacktestValidationError(ValueError):
    """Raised when a backtest configuration or result is invalid."""


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    """Proper scoring and point-forecast metrics for one backtest."""

    model: str
    predictions: int
    mean_negative_log_score: float
    mean_absolute_error: float
    root_mean_squared_error: float
    forecast_bias: float
    mean_observed_total: float
    mean_predicted_rate: float


def walk_forward_poisson(
    matches: pd.DataFrame,
    *,
    min_train_size: int = 200,
    window_size: int | None = None,
) -> pd.DataFrame:
    """Run a matchday-level walk-forward backtest of the Poisson benchmark.

    Every game on a matchday is priced from the same information set. The
    model is updated only after the entire matchday, preventing within-round
    look-ahead bias when exact kickoff timestamps are unavailable.
    """
    _validate_configuration(min_train_size, window_size)
    ordered = _chronological_matches(matches)

    if len(ordered) <= min_train_size:
        raise BacktestValidationError("Match data must contain more rows than min_train_size.")

    totals = ordered["total_goals"].to_numpy(dtype=np.int64)
    model_name = "poisson_expanding" if window_size is None else f"poisson_rolling_{window_size}"
    records: list[dict[str, object]] = []

    matchdays = ordered.groupby(["saison", "spieltag"], sort=False)
    for _, current_matchday in matchdays:
        first_position = int(current_matchday.index[0])
        if first_position < min_train_size:
            continue

        train_start = 0 if window_size is None else max(0, first_position - window_size)
        training_counts = totals[train_start:first_position]
        model = PoissonTotalModel.fit(training_counts.tolist())

        for position in current_matchday.index:
            observation_index = int(position)
            observed_total = int(totals[observation_index])
            log_probability = model.log_likelihood([observed_total])
            row = ordered.iloc[observation_index]

            records.append(
                {
                    "model": model_name,
                    "observation_index": observation_index,
                    "saison": str(row["saison"]),
                    "spieltag": int(row["spieltag"]),
                    "spiel": int(row["spiel"]),
                    "train_start": train_start,
                    "train_size": model.sample_size,
                    "observed_total": observed_total,
                    "predicted_rate": model.rate,
                    "log_probability": log_probability,
                    "negative_log_score": -log_probability,
                    "forecast_error": model.rate - observed_total,
                }
            )

    if not records:
        raise BacktestValidationError(
            "No complete evaluation matchday follows the minimum training sample."
        )

    return pd.DataFrame.from_records(
        records,
        columns=list(PREDICTION_COLUMNS),
    )


def summarize_backtest(
    predictions: pd.DataFrame,
) -> BacktestSummary:
    """Summarize one model's walk-forward predictions."""
    if predictions.empty:
        raise BacktestValidationError("Backtest predictions cannot be empty.")

    missing_columns = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    if missing_columns:
        raise BacktestValidationError(
            f"Backtest predictions are missing columns: {missing_columns}"
        )

    model_names = predictions["model"].astype("string").unique().tolist()
    if len(model_names) != 1:
        raise BacktestValidationError(
            "Backtest summary requires predictions from exactly one model."
        )

    observed = predictions["observed_total"].to_numpy(dtype=np.float64)
    predicted = predictions["predicted_rate"].to_numpy(dtype=np.float64)
    negative_log_scores = predictions["negative_log_score"].to_numpy(dtype=np.float64)
    forecast_errors = predicted - observed

    return BacktestSummary(
        model=str(model_names[0]),
        predictions=len(predictions),
        mean_negative_log_score=float(negative_log_scores.mean()),
        mean_absolute_error=float(np.abs(forecast_errors).mean()),
        root_mean_squared_error=float(np.sqrt(np.square(forecast_errors).mean())),
        forecast_bias=float(forecast_errors.mean()),
        mean_observed_total=float(observed.mean()),
        mean_predicted_rate=float(predicted.mean()),
    )


def _chronological_matches(
    matches: pd.DataFrame,
) -> pd.DataFrame:
    ordered = add_match_features(matches)
    ordered["_season_start"] = ordered["saison"].str[:2].astype("int64")
    ordered = ordered.sort_values(
        ["_season_start", "spieltag", "spiel"],
        kind="stable",
    )
    return ordered.drop(columns="_season_start").reset_index(drop=True)


def _validate_configuration(
    min_train_size: int,
    window_size: int | None,
) -> None:
    if isinstance(min_train_size, bool) or min_train_size < 1:
        raise BacktestValidationError("min_train_size must be a positive integer.")
    if window_size is not None and (isinstance(window_size, bool) or window_size < 1):
        raise BacktestValidationError("window_size must be a positive integer or None.")
