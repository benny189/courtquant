"""Tests for the Poisson total-goal benchmark model."""

from collections.abc import Iterable
from math import exp, isinf
from typing import cast

import pytest

from courtquant.models.poisson import (
    ModelValidationError,
    PoissonTotalModel,
    TotalMarketProbabilities,
)


def fitted_model() -> PoissonTotalModel:
    """Return a Poisson model with a known rate of two."""
    return PoissonTotalModel.fit([1, 2, 3])


def test_fit_uses_sample_mean_as_mle() -> None:
    """The Poisson maximum-likelihood estimator is the sample mean."""
    model = fitted_model()

    assert model.rate == pytest.approx(2.0)
    assert model.sample_size == 3


@pytest.mark.parametrize(
    "rate",
    [-1.0, float("inf"), float("nan")],
)
def test_model_rejects_invalid_rate(rate: float) -> None:
    """The Poisson rate must be finite and non-negative."""
    with pytest.raises(ModelValidationError, match="Poisson rate"):
        PoissonTotalModel(rate=rate, sample_size=1)


def test_model_rejects_nonpositive_sample_size() -> None:
    """A fitted model must record at least one observation."""
    with pytest.raises(ModelValidationError, match="Sample size"):
        PoissonTotalModel(rate=1.0, sample_size=0)


def test_fit_rejects_empty_counts() -> None:
    """An empty estimation sample is invalid."""
    with pytest.raises(ModelValidationError, match="non-empty"):
        PoissonTotalModel.fit([])


def test_fit_rejects_multidimensional_counts() -> None:
    """Count observations must form a one-dimensional sequence."""
    nested_counts = cast(
        Iterable[int | float],
        [[1, 2], [3, 4]],
    )

    with pytest.raises(ModelValidationError, match="one-dimensional"):
        PoissonTotalModel.fit(nested_counts)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ([1.0, float("inf")], "finite"),
        ([1.0, -1.0], "negative"),
        ([1.0, 1.5], "integers"),
    ],
)
def test_fit_rejects_invalid_counts(
    counts: list[float],
    message: str,
) -> None:
    """Counts must be finite, non-negative integers."""
    with pytest.raises(ModelValidationError, match=message):
        PoissonTotalModel.fit(counts)


def test_probability_mass_matches_poisson_formula() -> None:
    """The PMF matches the analytical Poisson probabilities."""
    probabilities = fitted_model().probability_mass([0, 1, 2])

    expected = [
        exp(-2.0),
        2.0 * exp(-2.0),
        2.0 * exp(-2.0),
    ]
    assert probabilities.tolist() == pytest.approx(expected)


def test_log_likelihood_matches_known_value() -> None:
    """Joint log-likelihood is the sum of individual log probabilities."""
    log_likelihood = fitted_model().log_likelihood([1, 2, 3])

    assert log_likelihood == pytest.approx(-4.326023566428328)


def test_price_half_goal_line_has_no_push() -> None:
    """A half-goal market partitions probability into over and under."""
    market = fitted_model().price_total(1.5)

    assert market.line == pytest.approx(1.5)
    assert market.under == pytest.approx(0.40600584970983794)
    assert market.over == pytest.approx(0.5939941502901616)
    assert market.push == pytest.approx(0.0)
    assert market.over + market.under == pytest.approx(1.0)


def test_price_whole_goal_line_includes_push() -> None:
    """A whole-goal market assigns probability to the push outcome."""
    market = fitted_model().price_total(2.0)

    assert market.under == pytest.approx(0.40600584970983794)
    assert market.push == pytest.approx(0.2706705664732254)
    assert market.over == pytest.approx(0.32332358381693654)
    assert market.over + market.under + market.push == pytest.approx(1.0)


@pytest.mark.parametrize(
    "line",
    [-0.5, float("inf"), float("nan"), 1.25],
)
def test_price_total_rejects_invalid_line(line: float) -> None:
    """Only finite, non-negative whole- and half-goal lines are supported."""
    with pytest.raises(ModelValidationError, match="Market line"):
        fitted_model().price_total(line)


@pytest.mark.parametrize(
    ("over", "under", "push"),
    [
        (-0.1, 1.1, 0.0),
        (float("nan"), 0.5, 0.5),
    ],
)
def test_market_rejects_invalid_probabilities(
    over: float,
    under: float,
    push: float,
) -> None:
    """Every market probability must lie inside the unit interval."""
    with pytest.raises(ModelValidationError, match="finite and lie"):
        TotalMarketProbabilities(
            line=1.0,
            over=over,
            under=under,
            push=push,
        )


def test_market_probabilities_must_sum_to_one() -> None:
    """The three mutually exclusive outcomes exhaust the market."""
    with pytest.raises(ModelValidationError, match="sum to one"):
        TotalMarketProbabilities(
            line=1.0,
            over=0.4,
            under=0.4,
            push=0.1,
        )


def test_fair_odds_adjust_for_push_probability() -> None:
    """Fair decimal odds condition on the wager not pushing."""
    market = TotalMarketProbabilities(
        line=2.0,
        over=0.3,
        under=0.5,
        push=0.2,
    )

    assert market.fair_over_odds == pytest.approx(0.8 / 0.3)
    assert market.fair_under_odds == pytest.approx(0.8 / 0.5)


def test_zero_win_probability_has_infinite_fair_odds() -> None:
    """An impossible winning outcome has infinite fair decimal odds."""
    market = TotalMarketProbabilities(
        line=0.0,
        over=0.0,
        under=0.0,
        push=1.0,
    )

    assert isinf(market.fair_over_odds)
    assert isinf(market.fair_under_odds)
