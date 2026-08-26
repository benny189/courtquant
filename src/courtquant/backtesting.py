"""Leakage-resistant walk-forward backtesting for CourtQuant models."""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from courtquant.diagnostics import add_match_features
from courtquant.models.com_poisson import (
    ConwayMaxwellPoissonModel,
)
from courtquant.models.poisson import PoissonTotalModel

type ModelFamily = Literal["poisson", "com_poisson"]
type SupportedModel = PoissonTotalModel | ConwayMaxwellPoissonModel

PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "model",
    "observation_index",
    "saison",
    "spieltag",
    "spiel",
    "train_start",
    "train_size",
    "observed_total",
    "model_intensity",
    "dispersion_parameter",
    "predicted_mean",
    "predicted_variance",
    "predicted_dispersion_index",
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
    mean_predicted_total: float
    mean_predicted_variance: float
    mean_predicted_dispersion_index: float


def walk_forward_poisson(
    matches: pd.DataFrame,
    *,
    min_train_size: int = 200,
    window_size: int | None = None,
) -> pd.DataFrame:
    """Run a matchday-level walk-forward Poisson backtest."""
    return _walk_forward_model(
        matches,
        model_family="poisson",
        min_train_size=min_train_size,
        window_size=window_size,
    )


def walk_forward_com_poisson(
    matches: pd.DataFrame,
    *,
    min_train_size: int = 200,
    window_size: int | None = None,
) -> pd.DataFrame:
    """Run a matchday-level walk-forward COM-Poisson backtest."""
    return _walk_forward_model(
        matches,
        model_family="com_poisson",
        min_train_size=min_train_size,
        window_size=window_size,
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
    predicted = predictions["predicted_mean"].to_numpy(dtype=np.float64)
    predicted_variance = predictions["predicted_variance"].to_numpy(dtype=np.float64)
    predicted_dispersion = predictions["predicted_dispersion_index"].to_numpy(dtype=np.float64)
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
        mean_predicted_total=float(predicted.mean()),
        mean_predicted_variance=float(predicted_variance.mean()),
        mean_predicted_dispersion_index=float(predicted_dispersion.mean()),
    )


def _walk_forward_model(
    matches: pd.DataFrame,
    *,
    model_family: ModelFamily,
    min_train_size: int,
    window_size: int | None,
) -> pd.DataFrame:
    """Run one model family without within-matchday leakage."""
    _validate_configuration(
        min_train_size,
        window_size,
    )
    ordered = _chronological_matches(matches)

    if len(ordered) <= min_train_size:
        raise BacktestValidationError("Match data must contain more rows than min_train_size.")

    totals = ordered["total_goals"].to_numpy(dtype=np.int64)
    window_label = "expanding" if window_size is None else f"rolling_{window_size}"
    model_name = f"{model_family}_{window_label}"
    records: list[dict[str, object]] = []

    matchdays = ordered.groupby(
        ["saison", "spieltag"],
        sort=False,
    )
    for _, current_matchday in matchdays:
        first_position = int(current_matchday.index[0])
        if first_position < min_train_size:
            continue

        train_start = (
            0
            if window_size is None
            else max(
                0,
                first_position - window_size,
            )
        )
        training_counts = [int(value) for value in totals[train_start:first_position]]
        model = _fit_model(
            training_counts,
            model_family,
        )
        (
            model_intensity,
            dispersion_parameter,
            predicted_mean,
            predicted_variance,
        ) = _model_characteristics(model)
        predicted_dispersion_index = predicted_variance / predicted_mean

        for position in current_matchday.index:
            observation_index = int(position)
            observed_total = int(totals[observation_index])
            log_probability = model.log_likelihood([observed_total])
            row = ordered.iloc[observation_index]

            records.append(
                {
                    "model": model_name,
                    "observation_index": (observation_index),
                    "saison": str(row["saison"]),
                    "spieltag": int(row["spieltag"]),
                    "spiel": int(row["spiel"]),
                    "train_start": train_start,
                    "train_size": model.sample_size,
                    "observed_total": observed_total,
                    "model_intensity": (model_intensity),
                    "dispersion_parameter": (dispersion_parameter),
                    "predicted_mean": (predicted_mean),
                    "predicted_variance": (predicted_variance),
                    "predicted_dispersion_index": (predicted_dispersion_index),
                    "log_probability": (log_probability),
                    "negative_log_score": (-log_probability),
                    "forecast_error": (predicted_mean - observed_total),
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


def _fit_model(
    counts: list[int],
    model_family: ModelFamily,
) -> SupportedModel:
    if model_family == "poisson":
        return PoissonTotalModel.fit(counts)

    return ConwayMaxwellPoissonModel.fit(counts)


def _model_characteristics(
    model: SupportedModel,
) -> tuple[float, float, float, float]:
    if isinstance(model, PoissonTotalModel):
        return (
            model.rate,
            1.0,
            model.rate,
            model.rate,
        )

    mean, variance = model.moments()
    return (
        model.intensity,
        model.dispersion,
        mean,
        variance,
    )


def _chronological_matches(
    matches: pd.DataFrame,
) -> pd.DataFrame:
    ordered = add_match_features(matches)
    ordered["_season_start"] = ordered["saison"].str[:2].astype("int64")
    ordered = ordered.sort_values(
        [
            "_season_start",
            "spieltag",
            "spiel",
        ],
        kind="stable",
    )
    return ordered.drop(columns="_season_start").reset_index(drop=True)


def _validate_configuration(
    min_train_size: int,
    window_size: int | None,
) -> None:
    if (
        isinstance(min_train_size, bool)
        or not isinstance(
            min_train_size,
            int,
        )
        or min_train_size < 1
    ):
        raise BacktestValidationError("min_train_size must be a positive integer.")

    if window_size is not None and (
        isinstance(window_size, bool)
        or not isinstance(
            window_size,
            int,
        )
        or window_size < 1
    ):
        raise BacktestValidationError("window_size must be a positive integer or None.")
