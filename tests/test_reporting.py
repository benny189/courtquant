"""Tests for CourtQuant research tables and charts."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from courtquant.backtesting import (
    PREDICTION_COLUMNS as BACKTEST_COLUMNS,
)
from courtquant.evaluation import (
    PairedBootstrapResult,
)
from courtquant.reporting import (
    ReportingValidationError,
    build_bootstrap_summary,
    build_cumulative_score_path,
    build_model_summary,
    build_pricing_surface,
    save_cumulative_score_chart,
    save_pricing_surface_chart,
)


class LogProbabilityModel:
    """Simple model backed by discrete probabilities."""

    def __init__(
        self,
        probabilities: Sequence[float],
    ) -> None:
        self.probabilities = tuple(probabilities)

    def log_likelihood(
        self,
        counts: Sequence[int],
    ) -> float:
        probability = self.probabilities[counts[0]]
        if probability <= 0.0:
            return float("-inf")

        return float(np.log(probability))


def _backtest_predictions(
    model: str,
    negative_log_scores: Sequence[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [model, model],
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
            "log_probability": [-float(score) for score in negative_log_scores],
            "negative_log_score": list(negative_log_scores),
            "forecast_error": [2.0, -2.0],
        },
        columns=list(BACKTEST_COLUMNS),
    )


def _score_predictions(
    model: str,
    scores: Sequence[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [model] * 4,
            "observation_index": [
                2,
                3,
                4,
                5,
            ],
            "saison": ["18/19"] * 4,
            "spieltag": [2, 2, 3, 3],
            "spiel": [1, 2, 1, 2],
            "negative_log_score": list(scores),
        }
    )


def _bootstrap_result() -> PairedBootstrapResult:
    return PairedBootstrapResult(
        baseline_model="poisson",
        candidate_model="com_poisson",
        predictions=100,
        matchdays=12,
        baseline_mean_negative_log_score=3.3,
        candidate_mean_negative_log_score=3.2,
        mean_score_difference=-0.1,
        relative_improvement_percent=3.0,
        confidence_interval_low=-0.15,
        confidence_interval_high=-0.05,
        candidate_win_probability=0.99,
        bootstrap_samples=10_000,
        confidence_level=0.95,
    )


def test_model_summary_ranks_by_log_score() -> None:
    baseline = _backtest_predictions(
        "poisson",
        [3.0, 3.0],
    )
    candidate = _backtest_predictions(
        "com_poisson",
        [2.0, 2.0],
    )

    summary = build_model_summary(
        [
            baseline,
            candidate,
        ]
    )

    assert summary["rank"].tolist() == [
        1,
        2,
    ]
    assert summary["model"].tolist() == [
        "com_poisson",
        "poisson",
    ]
    assert summary["mean_negative_log_score"].tolist() == pytest.approx(
        [
            2.0,
            3.0,
        ]
    )


def test_model_summary_requires_predictions() -> None:
    with pytest.raises(
        ReportingValidationError,
        match="At least one prediction set",
    ):
        build_model_summary([])


def test_bootstrap_summary_builds_one_row() -> None:
    summary = build_bootstrap_summary(_bootstrap_result())

    assert len(summary) == 1
    assert (
        summary.loc[
            0,
            "baseline_model",
        ]
        == "poisson"
    )
    assert (
        summary.loc[
            0,
            "candidate_model",
        ]
        == "com_poisson"
    )
    assert summary.loc[
        0,
        "candidate_win_probability",
    ] == pytest.approx(0.99)


def test_pricing_surface_sorts_lines_and_reports_risk() -> None:
    surface = build_pricing_surface(
        LogProbabilityModel(
            [
                0.1,
                0.2,
                0.3,
                0.4,
            ]
        ),
        LogProbabilityModel(
            [
                0.05,
                0.2,
                0.25,
                0.5,
            ]
        ),
        [
            1.5,
            1.0,
        ],
        baseline_name="poisson",
        candidate_name="com_poisson",
        maximum_total=3,
        tail_tolerance=1e-12,
    )

    assert surface["line"].tolist() == [
        1.0,
        1.5,
    ]
    assert surface["baseline_model"].unique().tolist() == ["poisson"]
    assert surface["candidate_model"].unique().tolist() == ["com_poisson"]
    assert surface["baseline_over_probability"].tolist() == pytest.approx(
        [
            0.7,
            0.7,
        ]
    )
    assert surface["candidate_over_probability"].tolist() == pytest.approx(
        [
            0.75,
            0.75,
        ]
    )
    assert surface["over_difference_bps"].tolist() == pytest.approx(
        [
            500.0,
            500.0,
        ]
    )
    assert surface["maximum_model_risk_bps"].tolist() == pytest.approx(
        [
            500.0,
            500.0,
        ]
    )


def test_pricing_surface_requires_lines() -> None:
    with pytest.raises(
        ReportingValidationError,
        match="At least one total line",
    ):
        build_pricing_surface(
            LogProbabilityModel([0.5, 0.5]),
            LogProbabilityModel([0.4, 0.6]),
            [],
            baseline_name="poisson",
            candidate_name="com_poisson",
            maximum_total=1,
        )


@pytest.mark.parametrize(
    (
        "baseline_name",
        "candidate_name",
    ),
    [
        (
            "",
            "candidate",
        ),
        (
            "baseline",
            " ",
        ),
    ],
)
def test_pricing_surface_rejects_empty_model_names(
    baseline_name: str,
    candidate_name: str,
) -> None:
    with pytest.raises(
        ReportingValidationError,
        match="Model names cannot be empty",
    ):
        build_pricing_surface(
            LogProbabilityModel([0.5, 0.5]),
            LogProbabilityModel([0.4, 0.6]),
            [0.5],
            baseline_name=baseline_name,
            candidate_name=candidate_name,
            maximum_total=1,
        )


def test_pricing_surface_requires_different_names() -> None:
    with pytest.raises(
        ReportingValidationError,
        match="names must differ",
    ):
        build_pricing_surface(
            LogProbabilityModel([0.5, 0.5]),
            LogProbabilityModel([0.4, 0.6]),
            [0.5],
            baseline_name="same",
            candidate_name="same",
            maximum_total=1,
        )


def test_pricing_surface_rejects_duplicate_lines() -> None:
    with pytest.raises(
        ReportingValidationError,
        match="lines must be unique",
    ):
        build_pricing_surface(
            LogProbabilityModel(
                [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                ]
            ),
            LogProbabilityModel(
                [
                    0.05,
                    0.2,
                    0.25,
                    0.5,
                ]
            ),
            [
                1.5,
                1.5,
            ],
            baseline_name="poisson",
            candidate_name="com_poisson",
            maximum_total=3,
        )


def test_cumulative_score_path_aligns_predictions() -> None:
    baseline = _score_predictions(
        "poisson",
        [
            3.0,
            4.0,
            5.0,
            6.0,
        ],
    )
    candidate = (
        _score_predictions(
            "com_poisson",
            [
                2.0,
                5.0,
                4.0,
                6.0,
            ],
        )
        .iloc[[3, 1, 0, 2]]
        .reset_index(drop=True)
    )

    path = build_cumulative_score_path(
        baseline,
        candidate,
    )

    assert path["observation_index"].tolist() == [
        2,
        3,
        4,
        5,
    ]
    assert path["candidate_score_advantage"].tolist() == pytest.approx(
        [
            1.0,
            -1.0,
            1.0,
            0.0,
        ]
    )
    assert path["cumulative_candidate_score_advantage"].tolist() == pytest.approx(
        [
            1.0,
            0.0,
            1.0,
            1.0,
        ]
    )
    assert path["baseline_model"].unique().tolist() == ["poisson"]
    assert path["candidate_model"].unique().tolist() == ["com_poisson"]


def test_cumulative_path_rejects_empty_predictions() -> None:
    with pytest.raises(
        ReportingValidationError,
        match="cannot be empty",
    ):
        build_cumulative_score_path(
            pd.DataFrame(),
            _score_predictions(
                "com_poisson",
                [2.0, 3.0, 4.0, 5.0],
            ),
        )


def test_cumulative_path_rejects_missing_columns() -> None:
    baseline = _score_predictions(
        "poisson",
        [2.0, 3.0, 4.0, 5.0],
    ).drop(columns="spiel")

    with pytest.raises(
        ReportingValidationError,
        match="missing columns",
    ):
        build_cumulative_score_path(
            baseline,
            _score_predictions(
                "com_poisson",
                [2.0, 3.0, 4.0, 5.0],
            ),
        )


def test_cumulative_path_rejects_missing_values() -> None:
    baseline = _score_predictions(
        "poisson",
        [2.0, 3.0, 4.0, 5.0],
    )
    baseline.loc[
        0,
        "negative_log_score",
    ] = np.nan

    with pytest.raises(
        ReportingValidationError,
        match="missing values",
    ):
        build_cumulative_score_path(
            baseline,
            _score_predictions(
                "com_poisson",
                [2.0, 3.0, 4.0, 5.0],
            ),
        )


def test_cumulative_path_rejects_duplicates() -> None:
    baseline = _score_predictions(
        "poisson",
        [2.0, 3.0, 4.0, 5.0],
    )
    baseline = pd.concat(
        [
            baseline,
            baseline.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ReportingValidationError,
        match="duplicate observations",
    ):
        build_cumulative_score_path(
            baseline,
            _score_predictions(
                "com_poisson",
                [2.0, 3.0, 4.0, 5.0],
            ),
        )


def test_cumulative_path_rejects_multiple_models() -> None:
    baseline = _score_predictions(
        "poisson",
        [2.0, 3.0, 4.0, 5.0],
    )
    baseline.loc[3, "model"] = "other"

    with pytest.raises(
        ReportingValidationError,
        match="exactly one model",
    ):
        build_cumulative_score_path(
            baseline,
            _score_predictions(
                "com_poisson",
                [2.0, 3.0, 4.0, 5.0],
            ),
        )


def test_cumulative_path_rejects_non_finite_scores() -> None:
    baseline = _score_predictions(
        "poisson",
        [
            float("inf"),
            3.0,
            4.0,
            5.0,
        ],
    )

    with pytest.raises(
        ReportingValidationError,
        match="scores must be finite",
    ):
        build_cumulative_score_path(
            baseline,
            _score_predictions(
                "com_poisson",
                [2.0, 3.0, 4.0, 5.0],
            ),
        )


def test_cumulative_path_requires_different_models() -> None:
    baseline = _score_predictions(
        "same_model",
        [2.0, 3.0, 4.0, 5.0],
    )
    candidate = _score_predictions(
        "same_model",
        [2.0, 3.0, 4.0, 5.0],
    )

    with pytest.raises(
        ReportingValidationError,
        match="models must differ",
    ):
        build_cumulative_score_path(
            baseline,
            candidate,
        )


def test_cumulative_path_requires_same_observations() -> None:
    baseline = _score_predictions(
        "poisson",
        [2.0, 3.0, 4.0, 5.0],
    )
    candidate = _score_predictions(
        "com_poisson",
        [2.0, 3.0, 4.0, 5.0],
    )
    candidate.loc[
        3,
        "observation_index",
    ] = 99

    with pytest.raises(
        ReportingValidationError,
        match="exactly the same observations",
    ):
        build_cumulative_score_path(
            baseline,
            candidate,
        )


def test_save_cumulative_score_chart(
    tmp_path: Path,
) -> None:
    path = build_cumulative_score_path(
        _score_predictions(
            "poisson",
            [3.0, 4.0, 5.0, 6.0],
        ),
        _score_predictions(
            "com_poisson",
            [2.0, 3.5, 4.0, 5.0],
        ),
    )
    destination = tmp_path / "figures" / "cumulative.png"

    returned = save_cumulative_score_chart(
        path,
        destination,
    )

    assert returned == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0


def test_save_pricing_surface_chart(
    tmp_path: Path,
) -> None:
    surface = build_pricing_surface(
        LogProbabilityModel(
            [
                0.1,
                0.2,
                0.3,
                0.4,
            ]
        ),
        LogProbabilityModel(
            [
                0.05,
                0.2,
                0.25,
                0.5,
            ]
        ),
        [
            0.5,
            1.0,
            1.5,
            2.0,
        ],
        baseline_name="poisson",
        candidate_name="com_poisson",
        maximum_total=3,
    )
    destination = tmp_path / "figures" / "pricing.png"

    returned = save_pricing_surface_chart(
        surface,
        destination,
    )

    assert returned == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0


def test_chart_rejects_empty_table(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ReportingValidationError,
        match="cannot be empty",
    ):
        save_cumulative_score_chart(
            pd.DataFrame(),
            tmp_path / "empty.png",
        )


def test_chart_rejects_missing_columns(
    tmp_path: Path,
) -> None:
    incomplete = pd.DataFrame(
        {
            "observation_index": [1],
        }
    )

    with pytest.raises(
        ReportingValidationError,
        match="missing columns",
    ):
        save_cumulative_score_chart(
            incomplete,
            tmp_path / "incomplete.png",
        )


def test_chart_requires_single_model_label(
    tmp_path: Path,
) -> None:
    path = build_cumulative_score_path(
        _score_predictions(
            "poisson",
            [3.0, 4.0, 5.0, 6.0],
        ),
        _score_predictions(
            "com_poisson",
            [2.0, 3.0, 4.0, 5.0],
        ),
    )
    path.loc[
        3,
        "baseline_model",
    ] = "other_model"

    with pytest.raises(
        ReportingValidationError,
        match="exactly one model",
    ):
        save_cumulative_score_chart(
            path,
            tmp_path / "multiple.png",
        )
