"""Tests for total-market pricing and model risk."""

from collections.abc import Sequence
from math import log
from typing import cast

import numpy as np
import pytest

from courtquant.pricing import (
    PricingValidationError,
    TotalDistribution,
    TotalMarketModelRisk,
    build_total_distribution,
    compare_total_prices,
    price_total_market,
)


class LogProbabilityModel:
    """Simple discrete model backed by log-probabilities."""

    def __init__(
        self,
        log_probabilities: Sequence[float],
    ) -> None:
        self.log_probabilities = tuple(log_probabilities)

    def log_likelihood(
        self,
        counts: Sequence[int],
    ) -> float:
        assert len(counts) == 1
        return self.log_probabilities[counts[0]]


def _model_from_probabilities(
    probabilities: Sequence[float],
) -> LogProbabilityModel:
    return LogProbabilityModel(
        [
            (log(probability) if probability > 0.0 else float("-inf"))
            for probability in probabilities
        ]
    )


def _complete_distribution() -> TotalDistribution:
    return TotalDistribution(
        probabilities=(
            0.1,
            0.2,
            0.3,
            0.4,
        ),
        covered_probability_mass=1.0,
        residual_probability=0.0,
        maximum_total=3,
    )


def test_build_total_distribution() -> None:
    distribution = build_total_distribution(
        _model_from_probabilities(
            [
                0.1,
                0.2,
                0.3,
                0.4,
            ]
        ),
        maximum_total=3,
        tail_tolerance=1e-12,
    )

    assert distribution.maximum_total == 3
    assert distribution.probabilities == pytest.approx(
        (
            0.1,
            0.2,
            0.3,
            0.4,
        )
    )
    assert distribution.covered_probability_mass == pytest.approx(1.0)
    assert distribution.residual_probability == pytest.approx(0.0)


def test_distribution_retains_small_residual_mass() -> None:
    distribution = build_total_distribution(
        _model_from_probabilities(
            [
                0.2,
                0.3,
                0.4,
            ]
        ),
        maximum_total=2,
        tail_tolerance=0.2,
    )

    assert distribution.covered_probability_mass == pytest.approx(0.9)
    assert distribution.residual_probability == pytest.approx(0.1)

    price = price_total_market(
        distribution,
        1.5,
    )

    assert price.under_probability == pytest.approx(0.5)
    assert price.over_probability == pytest.approx(0.5)


def test_distribution_normalizes_tiny_excess_mass() -> None:
    distribution = build_total_distribution(
        _model_from_probabilities(
            [
                0.5,
                0.5000000001,
            ]
        ),
        maximum_total=1,
        tail_tolerance=1e-6,
    )

    assert sum(distribution.probabilities) == pytest.approx(1.0)
    assert distribution.covered_probability_mass == pytest.approx(1.0)
    assert distribution.residual_probability == pytest.approx(0.0)


def test_price_integer_total_market() -> None:
    price = price_total_market(
        _complete_distribution(),
        1.0,
    )

    assert price.line == pytest.approx(1.0)
    assert price.under_probability == pytest.approx(0.1)
    assert price.push_probability == pytest.approx(0.2)
    assert price.over_probability == pytest.approx(0.7)
    assert price.under_fair_decimal_odds == pytest.approx(8.0)
    assert price.over_fair_decimal_odds == pytest.approx(0.8 / 0.7)
    assert price.covered_probability_mass == pytest.approx(1.0)
    assert price.residual_probability == pytest.approx(0.0)
    assert price.maximum_total == 3


def test_price_half_goal_total_market() -> None:
    price = price_total_market(
        _complete_distribution(),
        1.5,
    )

    assert price.under_probability == pytest.approx(0.3)
    assert price.push_probability == pytest.approx(0.0)
    assert price.over_probability == pytest.approx(0.7)
    assert price.under_fair_decimal_odds == pytest.approx(1.0 / 0.3)
    assert price.over_fair_decimal_odds == pytest.approx(1.0 / 0.7)


def test_zero_win_probability_has_infinite_odds() -> None:
    distribution = TotalDistribution(
        probabilities=(
            0.2,
            0.8,
        ),
        covered_probability_mass=1.0,
        residual_probability=0.0,
        maximum_total=1,
    )

    price = price_total_market(
        distribution,
        0.0,
    )

    assert price.under_probability == pytest.approx(0.0)
    assert price.under_fair_decimal_odds == float("inf")
    assert price.push_probability == pytest.approx(0.2)
    assert price.over_probability == pytest.approx(0.8)


def test_compare_total_prices_quantifies_model_risk() -> None:
    baseline = price_total_market(
        _complete_distribution(),
        1.0,
    )
    candidate = price_total_market(
        TotalDistribution(
            probabilities=(
                0.05,
                0.25,
                0.25,
                0.45,
            ),
            covered_probability_mass=1.0,
            residual_probability=0.0,
            maximum_total=3,
        ),
        1.0,
    )

    risk = compare_total_prices(
        baseline,
        candidate,
    )

    assert isinstance(
        risk,
        TotalMarketModelRisk,
    )
    assert risk.line == pytest.approx(1.0)
    assert risk.baseline_price == baseline
    assert risk.candidate_price == candidate
    assert risk.under_probability_difference == pytest.approx(-0.05)
    assert risk.push_probability_difference == pytest.approx(0.05)
    assert risk.over_probability_difference == pytest.approx(0.0)
    assert risk.maximum_absolute_probability_difference == pytest.approx(0.05)
    assert risk.maximum_absolute_difference_basis_points == pytest.approx(500.0)


def test_compare_total_prices_requires_same_line() -> None:
    baseline = price_total_market(
        _complete_distribution(),
        1.0,
    )
    candidate = price_total_market(
        _complete_distribution(),
        1.5,
    )

    with pytest.raises(
        PricingValidationError,
        match="same total line",
    ):
        compare_total_prices(
            baseline,
            candidate,
        )


@pytest.mark.parametrize(
    "maximum_total",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_distribution_rejects_invalid_maximum(
    maximum_total: object,
) -> None:
    with pytest.raises(
        PricingValidationError,
        match="maximum_total must be a positive integer",
    ):
        build_total_distribution(
            _model_from_probabilities([1.0]),
            maximum_total=cast(
                int,
                maximum_total,
            ),
        )


@pytest.mark.parametrize(
    "tail_tolerance",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
        True,
        "0.1",
    ],
)
def test_distribution_rejects_invalid_tail_tolerance(
    tail_tolerance: object,
) -> None:
    with pytest.raises(
        PricingValidationError,
        match="tail_tolerance must be between zero and one",
    ):
        build_total_distribution(
            _model_from_probabilities(
                [
                    0.5,
                    0.5,
                ]
            ),
            maximum_total=1,
            tail_tolerance=cast(
                float,
                tail_tolerance,
            ),
        )


@pytest.mark.parametrize(
    "invalid_log_probability",
    [
        float("nan"),
        float("inf"),
    ],
)
def test_distribution_rejects_non_finite_log_probabilities(
    invalid_log_probability: float,
) -> None:
    model = LogProbabilityModel(
        [
            invalid_log_probability,
            log(0.5),
        ]
    )

    with pytest.raises(
        PricingValidationError,
        match="NaN or positive infinity",
    ):
        build_total_distribution(
            model,
            maximum_total=1,
            tail_tolerance=0.9,
        )


def test_distribution_rejects_positive_log_probability() -> None:
    model = LogProbabilityModel(
        [
            0.1,
            log(0.5),
        ]
    )

    with pytest.raises(
        PricingValidationError,
        match="cannot be positive",
    ):
        build_total_distribution(
            model,
            maximum_total=1,
            tail_tolerance=0.9,
        )


def test_distribution_rejects_non_finite_probability_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def infinite_probabilities(
        values: np.ndarray,
    ) -> np.ndarray:
        return np.full_like(
            values,
            float("inf"),
        )

    monkeypatch.setattr(
        "courtquant.pricing.np.exp",
        infinite_probabilities,
    )

    with pytest.raises(
        PricingValidationError,
        match="positive finite total mass",
    ):
        build_total_distribution(
            LogProbabilityModel(
                [
                    0.0,
                    0.0,
                ]
            ),
            maximum_total=1,
            tail_tolerance=0.5,
        )


def test_distribution_rejects_zero_probability_mass() -> None:
    with pytest.raises(
        PricingValidationError,
        match="positive finite total mass",
    ):
        build_total_distribution(
            LogProbabilityModel(
                [
                    float("-inf"),
                    float("-inf"),
                ]
            ),
            maximum_total=1,
            tail_tolerance=0.5,
        )


def test_distribution_rejects_excess_probability_mass() -> None:
    with pytest.raises(
        PricingValidationError,
        match="exceed total mass one",
    ):
        build_total_distribution(
            _model_from_probabilities(
                [
                    0.7,
                    0.7,
                ]
            ),
            maximum_total=1,
            tail_tolerance=0.1,
        )


def test_distribution_rejects_uncovered_tail() -> None:
    with pytest.raises(
        PricingValidationError,
        match="does not cover enough",
    ):
        build_total_distribution(
            _model_from_probabilities(
                [
                    0.2,
                    0.3,
                ]
            ),
            maximum_total=1,
            tail_tolerance=0.1,
        )


@pytest.mark.parametrize(
    "line",
    [
        True,
        "1.5",
        float("nan"),
        float("inf"),
    ],
)
def test_market_rejects_non_finite_numeric_line(
    line: object,
) -> None:
    with pytest.raises(
        PricingValidationError,
        match="finite number",
    ):
        price_total_market(
            _complete_distribution(),
            cast(float, line),
        )


@pytest.mark.parametrize(
    "line",
    [
        -0.5,
        3.0,
    ],
)
def test_market_rejects_line_outside_support(
    line: float,
) -> None:
    with pytest.raises(
        PricingValidationError,
        match="below maximum_total",
    ):
        price_total_market(
            _complete_distribution(),
            line,
        )


@pytest.mark.parametrize(
    "line",
    [
        1.25,
        1.75,
    ],
)
def test_market_rejects_quarter_goal_line(
    line: float,
) -> None:
    with pytest.raises(
        PricingValidationError,
        match="integer or half-goal increments",
    ):
        price_total_market(
            _complete_distribution(),
            line,
        )


def test_market_rejects_invalid_probability_partition() -> None:
    malformed = TotalDistribution(
        probabilities=(
            0.2,
            0.2,
            0.2,
        ),
        covered_probability_mass=0.6,
        residual_probability=0.0,
        maximum_total=2,
    )

    with pytest.raises(
        PricingValidationError,
        match="must sum to one",
    ):
        price_total_market(
            malformed,
            1.5,
        )
