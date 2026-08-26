"""Tests for leakage-resistant walk-forward backtesting."""

from collections.abc import Sequence
from typing import cast

import pandas as pd
import pytest

from courtquant.backtesting import (
    PREDICTION_COLUMNS,
    BacktestSummary,
    BacktestValidationError,
    summarize_backtest,
    walk_forward_com_poisson,
    walk_forward_poisson,
)
from courtquant.models.com_poisson import (
    ConwayMaxwellPoissonModel,
)


def _sample_matches() -> pd.DataFrame:
    """Create three matchdays with two matches each."""
    return pd.DataFrame(
        {
            "saison": ["18/19"] * 6,
            "spieltag": [1, 1, 2, 2, 3, 3],
            "spiel": [1, 2, 1, 2, 1, 2],
            "tore_mannschaft1": [
                4,
                9,
                14,
                19,
                24,
                29,
            ],
            "tore_mannschaft2": [
                6,
                11,
                16,
                21,
                26,
                31,
            ],
        }
    )


def _summary_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["model_a", "model_a"],
            "observation_index": [2, 3],
            "saison": ["18/19", "18/19"],
            "spieltag": [2, 2],
            "spiel": [1, 2],
            "train_start": [0, 0],
            "train_size": [2, 2],
            "observed_total": [10, 20],
            "model_intensity": [12.0, 18.0],
            "dispersion_parameter": [1.0, 1.0],
            "predicted_mean": [12.0, 18.0],
            "predicted_variance": [12.0, 18.0],
            "predicted_dispersion_index": [
                1.0,
                1.0,
            ],
            "log_probability": [-2.0, -4.0],
            "negative_log_score": [2.0, 4.0],
            "forecast_error": [2.0, -2.0],
        },
        columns=list(PREDICTION_COLUMNS),
    )


def test_expanding_poisson_uses_matchday_embargo() -> None:
    predictions = walk_forward_poisson(
        _sample_matches(),
        min_train_size=2,
    )

    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert predictions["model"].unique().tolist() == ["poisson_expanding"]
    assert predictions["observation_index"].tolist() == [
        2,
        3,
        4,
        5,
    ]

    second_matchday = predictions.loc[predictions["spieltag"] == 2]
    assert second_matchday["predicted_mean"].tolist() == pytest.approx([15.0, 15.0])
    assert second_matchday["predicted_variance"].tolist() == pytest.approx([15.0, 15.0])
    assert second_matchday["predicted_dispersion_index"].tolist() == pytest.approx([1.0, 1.0])
    assert second_matchday["model_intensity"].tolist() == pytest.approx([15.0, 15.0])
    assert second_matchday["dispersion_parameter"].tolist() == pytest.approx([1.0, 1.0])
    assert second_matchday["train_size"].tolist() == [
        2,
        2,
    ]
    assert second_matchday["train_start"].tolist() == [
        0,
        0,
    ]

    third_matchday = predictions.loc[predictions["spieltag"] == 3]
    assert third_matchday["predicted_mean"].tolist() == pytest.approx([25.0, 25.0])
    assert third_matchday["train_size"].tolist() == [
        4,
        4,
    ]
    assert third_matchday["train_start"].tolist() == [
        0,
        0,
    ]


def test_backtest_sorts_matches_chronologically() -> None:
    shuffled = _sample_matches().iloc[[5, 2, 0, 4, 1, 3]].reset_index(drop=True)

    predictions = walk_forward_poisson(
        shuffled,
        min_train_size=2,
    )

    assert predictions["spieltag"].tolist() == [
        2,
        2,
        3,
        3,
    ]
    assert predictions["spiel"].tolist() == [
        1,
        2,
        1,
        2,
    ]
    assert predictions["observed_total"].tolist() == [
        30,
        40,
        50,
        60,
    ]


def test_rolling_poisson_uses_requested_window() -> None:
    predictions = walk_forward_poisson(
        _sample_matches(),
        min_train_size=2,
        window_size=2,
    )

    assert predictions["model"].unique().tolist() == ["poisson_rolling_2"]

    second_matchday = predictions.loc[predictions["spieltag"] == 2]
    assert second_matchday["predicted_mean"].tolist() == pytest.approx([15.0, 15.0])
    assert second_matchday["train_start"].tolist() == [
        0,
        0,
    ]

    third_matchday = predictions.loc[predictions["spieltag"] == 3]
    assert third_matchday["predicted_mean"].tolist() == pytest.approx([35.0, 35.0])
    assert third_matchday["train_start"].tolist() == [
        2,
        2,
    ]
    assert third_matchday["train_size"].tolist() == [
        2,
        2,
    ]


def test_com_poisson_backtest_records_dispersion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fitted_model(
        counts: Sequence[int],
    ) -> ConwayMaxwellPoissonModel:
        return ConwayMaxwellPoissonModel(
            intensity=2_500.0,
            dispersion=2.0,
            sample_size=len(counts),
        )

    monkeypatch.setattr(
        ConwayMaxwellPoissonModel,
        "fit",
        staticmethod(fitted_model),
    )

    reference = fitted_model([10, 20])
    expected_mean, expected_variance = reference.moments()

    predictions = walk_forward_com_poisson(
        _sample_matches(),
        min_train_size=2,
        window_size=2,
    )

    assert predictions["model"].unique().tolist() == ["com_poisson_rolling_2"]
    assert predictions["model_intensity"].tolist() == pytest.approx([2_500.0] * 4)
    assert predictions["dispersion_parameter"].tolist() == pytest.approx([2.0] * 4)
    assert predictions["predicted_mean"].tolist() == pytest.approx([expected_mean] * 4)
    assert predictions["predicted_variance"].tolist() == pytest.approx([expected_variance] * 4)
    assert predictions["predicted_dispersion_index"].tolist() == pytest.approx(
        [expected_variance / expected_mean] * 4
    )
    assert predictions["train_size"].tolist() == [
        2,
        2,
        2,
        2,
    ]


def test_scores_are_internally_consistent() -> None:
    predictions = walk_forward_poisson(
        _sample_matches(),
        min_train_size=2,
    )

    assert predictions["negative_log_score"].tolist() == pytest.approx(
        (-predictions["log_probability"]).tolist()
    )
    assert predictions["forecast_error"].tolist() == pytest.approx(
        (predictions["predicted_mean"] - predictions["observed_total"]).tolist()
    )


@pytest.mark.parametrize(
    "min_train_size",
    [0, -1, True, 1.5],
)
def test_backtest_rejects_invalid_minimum_training_size(
    min_train_size: object,
) -> None:
    with pytest.raises(
        BacktestValidationError,
        match="min_train_size must be a positive integer",
    ):
        walk_forward_poisson(
            _sample_matches(),
            min_train_size=cast(
                int,
                min_train_size,
            ),
        )


@pytest.mark.parametrize(
    "window_size",
    [0, -1, True, 1.5],
)
def test_backtest_rejects_invalid_window_size(
    window_size: object,
) -> None:
    with pytest.raises(
        BacktestValidationError,
        match="window_size must be a positive integer or None",
    ):
        walk_forward_poisson(
            _sample_matches(),
            min_train_size=2,
            window_size=cast(
                int,
                window_size,
            ),
        )


def test_backtest_rejects_insufficient_history() -> None:
    with pytest.raises(
        BacktestValidationError,
        match="more rows than min_train_size",
    ):
        walk_forward_poisson(
            _sample_matches(),
            min_train_size=6,
        )


def test_backtest_requires_complete_evaluation_matchday() -> None:
    one_matchday = _sample_matches().iloc[:3].copy()
    one_matchday["spieltag"] = 1
    one_matchday["spiel"] = [1, 2, 3]

    with pytest.raises(
        BacktestValidationError,
        match="No complete evaluation matchday",
    ):
        walk_forward_poisson(
            one_matchday,
            min_train_size=2,
        )


def test_summarize_backtest_calculates_expected_metrics() -> None:
    summary = summarize_backtest(_summary_predictions())

    assert summary == BacktestSummary(
        model="model_a",
        predictions=2,
        mean_negative_log_score=3.0,
        mean_absolute_error=2.0,
        root_mean_squared_error=2.0,
        forecast_bias=0.0,
        mean_observed_total=15.0,
        mean_predicted_total=15.0,
        mean_predicted_variance=15.0,
        mean_predicted_dispersion_index=1.0,
    )


def test_summarize_backtest_rejects_empty_predictions() -> None:
    empty = pd.DataFrame(columns=list(PREDICTION_COLUMNS))

    with pytest.raises(
        BacktestValidationError,
        match="cannot be empty",
    ):
        summarize_backtest(empty)


def test_summarize_backtest_rejects_missing_columns() -> None:
    incomplete = pd.DataFrame({"model": ["model_a"]})

    with pytest.raises(
        BacktestValidationError,
        match="missing columns",
    ):
        summarize_backtest(incomplete)


def test_summarize_backtest_rejects_multiple_models() -> None:
    predictions = _summary_predictions()
    predictions.loc[1, "model"] = "model_b"

    with pytest.raises(
        BacktestValidationError,
        match="exactly one model",
    ):
        summarize_backtest(predictions)
