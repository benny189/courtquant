"""Statistical evaluation of competing CourtQuant models."""

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

PAIRING_COLUMNS: Final[tuple[str, ...]] = (
    "saison",
    "spieltag",
    "spiel",
)
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "model",
    *PAIRING_COLUMNS,
    "observed_total",
    "negative_log_score",
)


class EvaluationValidationError(ValueError):
    """Raised when model predictions cannot be compared safely."""


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """Result of a paired matchday-block bootstrap."""

    baseline_model: str
    candidate_model: str
    predictions: int
    matchdays: int
    baseline_mean_negative_log_score: float
    candidate_mean_negative_log_score: float
    mean_score_difference: float
    relative_improvement_percent: float
    confidence_interval_low: float
    confidence_interval_high: float
    candidate_win_probability: float
    bootstrap_samples: int
    confidence_level: float


def paired_matchday_bootstrap(
    baseline_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    *,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Compare two models using paired matchday resampling.

    A negative score difference means that the candidate model has
    the better, lower negative log-score.
    """
    _validate_configuration(
        bootstrap_samples,
        confidence_level,
        seed,
    )
    (
        paired,
        baseline_model,
        candidate_model,
    ) = _prepare_paired_predictions(
        baseline_predictions,
        candidate_predictions,
    )

    paired["score_difference"] = paired["candidate_score"] - paired["baseline_score"]
    matchday_blocks = (
        paired.groupby(
            ["saison", "spieltag"],
            sort=False,
        )
        .agg(
            score_difference_sum=(
                "score_difference",
                "sum",
            ),
            predictions=(
                "score_difference",
                "size",
            ),
        )
        .reset_index()
    )

    block_score_sums = matchday_blocks["score_difference_sum"].to_numpy(dtype=np.float64)
    block_sizes = matchday_blocks["predictions"].to_numpy(dtype=np.float64)
    matchday_count = len(matchday_blocks)

    generator = np.random.default_rng(seed)
    sampled_blocks = generator.integers(
        0,
        matchday_count,
        size=(
            bootstrap_samples,
            matchday_count,
        ),
    )
    sampled_mean_differences = block_score_sums[sampled_blocks].sum(axis=1) / block_sizes[
        sampled_blocks
    ].sum(axis=1)

    tail_probability = (1.0 - confidence_level) / 2.0
    confidence_interval = np.quantile(
        sampled_mean_differences,
        [
            tail_probability,
            1.0 - tail_probability,
        ],
    )

    baseline_scores = paired["baseline_score"].to_numpy(dtype=np.float64)
    candidate_scores = paired["candidate_score"].to_numpy(dtype=np.float64)
    baseline_mean = float(baseline_scores.mean())
    candidate_mean = float(candidate_scores.mean())
    mean_difference = candidate_mean - baseline_mean

    return PairedBootstrapResult(
        baseline_model=baseline_model,
        candidate_model=candidate_model,
        predictions=len(paired),
        matchdays=matchday_count,
        baseline_mean_negative_log_score=baseline_mean,
        candidate_mean_negative_log_score=candidate_mean,
        mean_score_difference=mean_difference,
        relative_improvement_percent=float(
            (baseline_mean - candidate_mean) / baseline_mean * 100.0
        ),
        confidence_interval_low=float(confidence_interval[0]),
        confidence_interval_high=float(confidence_interval[1]),
        candidate_win_probability=float(np.mean(sampled_mean_differences < 0.0)),
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
    )


def _prepare_paired_predictions(
    baseline_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    """Validate and align predictions from two models."""
    baseline_model = _validate_prediction_frame(
        baseline_predictions,
        "baseline",
    )
    candidate_model = _validate_prediction_frame(
        candidate_predictions,
        "candidate",
    )

    if baseline_model == candidate_model:
        raise EvaluationValidationError("Baseline and candidate models must differ.")

    baseline = baseline_predictions.loc[
        :,
        [
            *PAIRING_COLUMNS,
            "observed_total",
            "negative_log_score",
        ],
    ].rename(
        columns={
            "observed_total": ("baseline_observed_total"),
            "negative_log_score": ("baseline_score"),
        }
    )
    candidate = candidate_predictions.loc[
        :,
        [
            *PAIRING_COLUMNS,
            "observed_total",
            "negative_log_score",
        ],
    ].rename(
        columns={
            "observed_total": ("candidate_observed_total"),
            "negative_log_score": ("candidate_score"),
        }
    )

    paired = baseline.merge(
        candidate,
        on=list(PAIRING_COLUMNS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not bool(paired["_merge"].eq("both").all()):
        raise EvaluationValidationError(
            "Models must contain predictions for exactly the same matches."
        )

    baseline_observed = paired["baseline_observed_total"].to_numpy()
    candidate_observed = paired["candidate_observed_total"].to_numpy()
    if not np.array_equal(
        baseline_observed,
        candidate_observed,
    ):
        raise EvaluationValidationError("Paired predictions contain different observed totals.")

    paired = paired.drop(columns="_merge")
    paired = paired.sort_values(
        list(PAIRING_COLUMNS),
        kind="stable",
    ).reset_index(drop=True)
    return (
        paired,
        baseline_model,
        candidate_model,
    )


def _validate_prediction_frame(
    predictions: pd.DataFrame,
    label: str,
) -> str:
    """Validate one model's prediction table."""
    if predictions.empty:
        raise EvaluationValidationError(f"{label.capitalize()} predictions cannot be empty.")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(predictions.columns))
    if missing_columns:
        raise EvaluationValidationError(
            f"{label.capitalize()} predictions are missing columns: {missing_columns}"
        )

    required = predictions.loc[
        :,
        list(REQUIRED_COLUMNS),
    ]
    if bool(required.isna().any().any()):
        raise EvaluationValidationError(
            f"{label.capitalize()} predictions cannot contain missing values."
        )

    if bool(
        predictions.duplicated(
            subset=list(PAIRING_COLUMNS),
        ).any()
    ):
        raise EvaluationValidationError(
            f"{label.capitalize()} predictions contain duplicate matches."
        )

    model_names = predictions["model"].astype("string").unique().tolist()
    if len(model_names) != 1:
        raise EvaluationValidationError(
            f"{label.capitalize()} predictions must contain exactly one model."
        )

    scores = predictions["negative_log_score"].to_numpy(dtype=np.float64)
    if not bool(np.isfinite(scores).all()) or bool(np.any(scores < 0.0)):
        raise EvaluationValidationError(
            f"{label.capitalize()} negative log-scores must be finite and non-negative."
        )

    return str(model_names[0])


def _validate_configuration(
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> None:
    """Validate bootstrap configuration values."""
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(
            bootstrap_samples,
            int,
        )
        or bootstrap_samples < 1
    ):
        raise EvaluationValidationError("bootstrap_samples must be a positive integer.")

    if (
        isinstance(confidence_level, bool)
        or not isinstance(
            confidence_level,
            (int, float),
        )
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise EvaluationValidationError("confidence_level must be between zero and one.")

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvaluationValidationError("seed must be a non-negative integer.")
