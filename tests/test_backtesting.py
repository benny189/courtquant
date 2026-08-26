"""Tests for leakage-resistant walk-forward backtesting."""

import pandas as pd
import pytest

from courtquant.backtesting import (
    PREDICTION_COLUMNS,
    BacktestSummary,
    BacktestValidationError,
    summarize_backtest,
    walk_forward_poisson,
)


def _sample_matches() -> pd.DataFrame:
    """Create three matchdays with two matches each."""
    return pd.DataFrame(
        {
            "saison": ["18/19"] * 6,
            "spieltag": [1, 1, 2, 2, 3, 3],
            "spiel": [1, 2, 1, 2, 1, 2],
            "tore_mannschaft1": [4, 9, 14, 19, 24, 29],
            "tore_mannschaft2": [6, 11, 16, 21, 26, 31],
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
            "predicted_rate": [12.0, 18.0],
            "log_probability": [-2.0, -4.0],
            "negative_log_score": [2.0, 4.0],
            "forecast_error": [2.0, -2.0],
        },
        columns=list(PREDICTION_COLUMNS),
    )


def test_expanding_backtest_uses_matchday_embargo() -> None:
    predictions = walk_forward_poisson(
        _sample_matches(),
        min_train_size=2,
    )

    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert predictions["model"].unique().tolist() == ["poisson_expanding"]
    assert predictions["observation_index"].tolist() == [2, 3, 4, 5]

    second_matchday = predictions.loc[predictions["spieltag"] == 2]
    assert second_matchday["predicted_rate"].tolist() == pytest.approx([15.0, 15.0])
    assert second_matchday["train_size"].tolist() == [2, 2]
    assert second_matchday["train_start"].tolist() == [0, 0]

    third_matchday = predictions.loc[predictions["spieltag"] == 3]
    assert third_matchday["predicted_rate"].tolist() == pytest.approx([25.0, 25.0])
    assert third_matchday["train_size"].tolist() == [4, 4]
    assert third_matchday["train_start"].tolist() == [0, 0]


def test_backtest_sorts_matches_chronologically() -> None:
    shuffled = _sample_matches().iloc[[5, 2, 0, 4, 1, 3]].reset_index(drop=True)

    predictions = walk_forward_poisson(
        shuffled,
        min_train_size=2,
    )

    assert predictions["spieltag"].tolist() == [2, 2, 3, 3]
    assert predictions["spiel"].tolist() == [1, 2, 1, 2]
    assert predictions["observed_total"].tolist() == [30, 40, 50, 60]


def test_rolling_backtest_uses_only_requested_window() -> None:
    predictions = walk_forward_poisson(
        _sample_matches(),
        min_train_size=2,
        window_size=2,
    )

    assert predictions["model"].unique().tolist() == ["poisson_rolling_2"]

    second_matchday = predictions.loc[predictions["spieltag"] == 2]
    assert second_matchday["predicted_rate"].tolist() == pytest.approx([15.0, 15.0])
    assert second_matchday["train_start"].tolist() == [0, 0]

    third_matchday = predictions.loc[predictions["spieltag"] == 3]
    assert third_matchday["predicted_rate"].tolist() == pytest.approx([35.0, 35.0])
    assert third_matchday["train_start"].tolist() == [2, 2]
    assert third_matchday["train_size"].tolist() == [2, 2]


def test_scores_are_internally_consistent() -> None:
    predictions = walk_forward_poisson(
        _sample_matches(),
        min_train_size=2,
    )

    assert predictions["negative_log_score"].tolist() == pytest.approx(
        (-predictions["log_probability"]).tolist()
    )
    assert predictions["forecast_error"].tolist() == pytest.approx(
        (predictions["predicted_rate"] - predictions["observed_total"]).tolist()
    )


@pytest.mark.parametrize("min_train_size", [0, -1, True])
def test_backtest_rejects_invalid_minimum_training_size(
    min_train_size: int,
) -> None:
    with pytest.raises(
        BacktestValidationError,
        match="min_train_size must be a positive integer",
    ):
        walk_forward_poisson(
            _sample_matches(),
            min_train_size=min_train_size,
        )


@pytest.mark.parametrize("window_size", [0, -1, True])
def test_backtest_rejects_invalid_window_size(window_size: int) -> None:
    with pytest.raises(
        BacktestValidationError,
        match="window_size must be a positive integer or None",
    ):
        walk_forward_poisson(
            _sample_matches(),
            min_train_size=2,
            window_size=window_size,
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
        mean_predicted_rate=15.0,
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
