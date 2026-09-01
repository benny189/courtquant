"""Tests for statistical model comparison."""

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
import pytest

from courtquant.evaluation import (
    EvaluationValidationError,
    PairedBootstrapResult,
    paired_matchday_bootstrap,
)


def _predictions(
    model: str,
    scores: Sequence[float],
    *,
    observed_totals: Sequence[int] = (
        50,
        51,
        52,
        53,
    ),
) -> pd.DataFrame:
    """Create two matchdays of model predictions."""
    return pd.DataFrame(
        {
            "model": [model] * 4,
            "saison": ["18/19"] * 4,
            "spieltag": [1, 1, 2, 2],
            "spiel": [1, 2, 1, 2],
            "observed_total": list(observed_totals),
            "negative_log_score": list(scores),
        }
    )


def test_bootstrap_reports_uniform_candidate_improvement() -> None:
    baseline = _predictions(
        "poisson",
        [2.0, 3.0, 4.0, 5.0],
    )
    candidate = _predictions(
        "com_poisson",
        [1.5, 2.5, 3.5, 4.5],
    )

    result = paired_matchday_bootstrap(
        baseline,
        candidate,
        bootstrap_samples=200,
        confidence_level=0.90,
        seed=7,
    )

    assert isinstance(
        result,
        PairedBootstrapResult,
    )
    assert result.baseline_model == "poisson"
    assert result.candidate_model == "com_poisson"
    assert result.predictions == 4
    assert result.matchdays == 2
    assert result.baseline_mean_negative_log_score == pytest.approx(3.5)
    assert result.candidate_mean_negative_log_score == pytest.approx(3.0)
    assert result.mean_score_difference == pytest.approx(-0.5)
    assert result.relative_improvement_percent == pytest.approx(14.2857142857)
    assert result.confidence_interval_low == pytest.approx(-0.5)
    assert result.confidence_interval_high == pytest.approx(-0.5)
    assert result.candidate_win_probability == pytest.approx(1.0)
    assert result.bootstrap_samples == 200
    assert result.confidence_level == pytest.approx(0.90)


def test_bootstrap_resamples_complete_matchdays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedGenerator:
        def integers(
            self,
            low: int,
            high: int,
            *,
            size: tuple[int, int],
        ) -> np.ndarray:
            assert low == 0
            assert high == 2
            assert size == (3, 2)
            return np.asarray(
                [
                    [0, 0],
                    [1, 1],
                    [0, 1],
                ],
                dtype=np.int64,
            )

    def fixed_default_rng(
        seed: int,
    ) -> FixedGenerator:
        assert seed == 7
        return FixedGenerator()

    monkeypatch.setattr(
        np.random,
        "default_rng",
        fixed_default_rng,
    )

    baseline = _predictions(
        "poisson",
        [5.0, 5.0, 5.0, 5.0],
    )
    candidate = _predictions(
        "com_poisson",
        [2.0, 6.0, 7.0, 7.0],
    )

    result = paired_matchday_bootstrap(
        baseline,
        candidate,
        bootstrap_samples=3,
        confidence_level=0.50,
        seed=7,
    )

    assert result.mean_score_difference == pytest.approx(0.5)
    assert result.confidence_interval_low == pytest.approx(-0.25)
    assert result.confidence_interval_high == pytest.approx(1.25)
    assert result.candidate_win_probability == pytest.approx(1.0 / 3.0)


def test_bootstrap_aligns_shuffled_predictions() -> None:
    baseline = _predictions(
        "poisson",
        [2.0, 3.0, 4.0, 5.0],
    )
    candidate = (
        _predictions(
            "com_poisson",
            [1.75, 2.75, 3.75, 4.75],
        )
        .iloc[[3, 1, 0, 2]]
        .reset_index(drop=True)
    )

    result = paired_matchday_bootstrap(
        baseline,
        candidate,
        bootstrap_samples=10,
    )

    assert result.mean_score_difference == pytest.approx(-0.25)


def test_bootstrap_rejects_empty_predictions() -> None:
    with pytest.raises(
        EvaluationValidationError,
        match="cannot be empty",
    ):
        paired_matchday_bootstrap(
            pd.DataFrame(),
            _predictions(
                "com_poisson",
                [1.0, 2.0, 3.0, 4.0],
            ),
        )


def test_bootstrap_rejects_missing_columns() -> None:
    baseline = _predictions(
        "poisson",
        [1.0, 2.0, 3.0, 4.0],
    ).drop(columns="observed_total")

    with pytest.raises(
        EvaluationValidationError,
        match="missing columns",
    ):
        paired_matchday_bootstrap(
            baseline,
            _predictions(
                "com_poisson",
                [1.0, 2.0, 3.0, 4.0],
            ),
        )


def test_bootstrap_rejects_missing_values() -> None:
    baseline = _predictions(
        "poisson",
        [1.0, 2.0, 3.0, 4.0],
    )
    baseline.loc[
        0,
        "negative_log_score",
    ] = np.nan

    with pytest.raises(
        EvaluationValidationError,
        match="missing values",
    ):
        paired_matchday_bootstrap(
            baseline,
            _predictions(
                "com_poisson",
                [1.0, 2.0, 3.0, 4.0],
            ),
        )


def test_bootstrap_rejects_duplicate_matches() -> None:
    baseline = _predictions(
        "poisson",
        [1.0, 2.0, 3.0, 4.0],
    )
    baseline = pd.concat(
        [
            baseline,
            baseline.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        EvaluationValidationError,
        match="duplicate matches",
    ):
        paired_matchday_bootstrap(
            baseline,
            _predictions(
                "com_poisson",
                [1.0, 2.0, 3.0, 4.0],
            ),
        )


def test_bootstrap_rejects_multiple_models() -> None:
    baseline = _predictions(
        "poisson",
        [1.0, 2.0, 3.0, 4.0],
    )
    baseline.loc[3, "model"] = "other_model"

    with pytest.raises(
        EvaluationValidationError,
        match="exactly one model",
    ):
        paired_matchday_bootstrap(
            baseline,
            _predictions(
                "com_poisson",
                [1.0, 2.0, 3.0, 4.0],
            ),
        )


@pytest.mark.parametrize(
    "invalid_score",
    [
        float("inf"),
        -0.1,
    ],
)
def test_bootstrap_rejects_invalid_scores(
    invalid_score: float,
) -> None:
    baseline = _predictions(
        "poisson",
        [
            invalid_score,
            2.0,
            3.0,
            4.0,
        ],
    )

    with pytest.raises(
        EvaluationValidationError,
        match="finite and non-negative",
    ):
        paired_matchday_bootstrap(
            baseline,
            _predictions(
                "com_poisson",
                [1.0, 2.0, 3.0, 4.0],
            ),
        )


def test_bootstrap_requires_different_models() -> None:
    baseline = _predictions(
        "same_model",
        [1.0, 2.0, 3.0, 4.0],
    )
    candidate = _predictions(
        "same_model",
        [1.0, 2.0, 3.0, 4.0],
    )

    with pytest.raises(
        EvaluationValidationError,
        match="must differ",
    ):
        paired_matchday_bootstrap(
            baseline,
            candidate,
        )


def test_bootstrap_requires_identical_matches() -> None:
    baseline = _predictions(
        "poisson",
        [1.0, 2.0, 3.0, 4.0],
    )
    candidate = _predictions(
        "com_poisson",
        [1.0, 2.0, 3.0, 4.0],
    )
    candidate.loc[3, "spiel"] = 3

    with pytest.raises(
        EvaluationValidationError,
        match="exactly the same matches",
    ):
        paired_matchday_bootstrap(
            baseline,
            candidate,
        )


def test_bootstrap_requires_identical_outcomes() -> None:
    baseline = _predictions(
        "poisson",
        [1.0, 2.0, 3.0, 4.0],
    )
    candidate = _predictions(
        "com_poisson",
        [1.0, 2.0, 3.0, 4.0],
        observed_totals=[
            50,
            51,
            52,
            99,
        ],
    )

    with pytest.raises(
        EvaluationValidationError,
        match="different observed totals",
    ):
        paired_matchday_bootstrap(
            baseline,
            candidate,
        )


@pytest.mark.parametrize(
    "bootstrap_samples",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_bootstrap_rejects_invalid_sample_count(
    bootstrap_samples: object,
) -> None:
    with pytest.raises(
        EvaluationValidationError,
        match="bootstrap_samples must be a positive integer",
    ):
        paired_matchday_bootstrap(
            _predictions(
                "poisson",
                [1.0, 2.0, 3.0, 4.0],
            ),
            _predictions(
                "com_poisson",
                [1.0, 2.0, 3.0, 4.0],
            ),
            bootstrap_samples=cast(
                int,
                bootstrap_samples,
            ),
        )


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
        True,
        "0.95",
    ],
)
def test_bootstrap_rejects_invalid_confidence_level(
    confidence_level: object,
) -> None:
    with pytest.raises(
        EvaluationValidationError,
        match="confidence_level must be between zero and one",
    ):
        paired_matchday_bootstrap(
            _predictions(
                "poisson",
                [1.0, 2.0, 3.0, 4.0],
            ),
            _predictions(
                "com_poisson",
                [1.0, 2.0, 3.0, 4.0],
            ),
            confidence_level=cast(
                float,
                confidence_level,
            ),
        )


@pytest.mark.parametrize(
    "seed",
    [
        -1,
        True,
        1.5,
    ],
)
def test_bootstrap_rejects_invalid_seed(
    seed: object,
) -> None:
    with pytest.raises(
        EvaluationValidationError,
        match="seed must be a non-negative integer",
    ):
        paired_matchday_bootstrap(
            _predictions(
                "poisson",
                [1.0, 2.0, 3.0, 4.0],
            ),
            _predictions(
                "com_poisson",
                [1.0, 2.0, 3.0, 4.0],
            ),
            seed=cast(
                int,
                seed,
            ),
        )
