"""Fair pricing and model-risk analysis for goal-total markets."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Protocol

import numpy as np


class TotalProbabilityModel(Protocol):
    """Discrete model capable of scoring goal totals."""

    def log_likelihood(
        self,
        counts: Sequence[int],
    ) -> float:
        """Return the joint log-likelihood of counts."""
        ...


class PricingValidationError(ValueError):
    """Raised when a market cannot be priced safely."""


@dataclass(frozen=True, slots=True)
class TotalDistribution:
    """Finite representation of a total-goals distribution."""

    probabilities: tuple[float, ...]
    covered_probability_mass: float
    residual_probability: float
    maximum_total: int


@dataclass(frozen=True, slots=True)
class TotalMarketPrice:
    """Fair probabilities and decimal odds for one total line."""

    line: float
    under_probability: float
    push_probability: float
    over_probability: float
    under_fair_decimal_odds: float
    over_fair_decimal_odds: float
    covered_probability_mass: float
    residual_probability: float
    maximum_total: int


@dataclass(frozen=True, slots=True)
class TotalMarketModelRisk:
    """Pricing differences between two probability models."""

    line: float
    baseline_price: TotalMarketPrice
    candidate_price: TotalMarketPrice
    under_probability_difference: float
    push_probability_difference: float
    over_probability_difference: float
    maximum_absolute_probability_difference: float
    maximum_absolute_difference_basis_points: float


def build_total_distribution(
    model: TotalProbabilityModel,
    *,
    maximum_total: int = 200,
    tail_tolerance: float = 1e-9,
) -> TotalDistribution:
    """Evaluate a model's probability mass over finite support."""
    _validate_distribution_configuration(
        maximum_total,
        tail_tolerance,
    )

    log_probabilities = np.asarray(
        [model.log_likelihood([total]) for total in range(maximum_total + 1)],
        dtype=np.float64,
    )
    if bool(np.isnan(log_probabilities).any()) or bool(np.isposinf(log_probabilities).any()):
        raise PricingValidationError(
            "Model log-probabilities must not contain NaN or positive infinity."
        )

    if bool(np.any(log_probabilities > 1e-12)):
        raise PricingValidationError("A discrete log-probability cannot be positive.")

    probabilities = np.exp(log_probabilities)
    covered_probability_mass = float(probabilities.sum())
    if not isfinite(covered_probability_mass) or covered_probability_mass <= 0.0:
        raise PricingValidationError("Model probabilities must have positive finite total mass.")

    if covered_probability_mass > 1.0 + tail_tolerance:
        raise PricingValidationError("Model probabilities exceed total mass one.")

    if covered_probability_mass > 1.0:
        probabilities = probabilities / covered_probability_mass
        covered_probability_mass = 1.0

    residual_probability = max(
        0.0,
        1.0 - covered_probability_mass,
    )
    if residual_probability > tail_tolerance:
        raise PricingValidationError("Maximum total does not cover enough probability mass.")

    return TotalDistribution(
        probabilities=tuple(float(probability) for probability in probabilities),
        covered_probability_mass=(covered_probability_mass),
        residual_probability=(residual_probability),
        maximum_total=maximum_total,
    )


def price_total_market(
    distribution: TotalDistribution,
    line: float,
) -> TotalMarketPrice:
    """Price an integer or half-goal Over/Under market."""
    numeric_line = _validate_market_line(
        line,
        distribution.maximum_total,
    )
    probabilities = np.asarray(
        distribution.probabilities,
        dtype=np.float64,
    )
    totals = np.arange(
        len(probabilities),
        dtype=np.float64,
    )

    under_probability = float(probabilities[totals < numeric_line].sum())
    push_probability = float(probabilities[totals == numeric_line].sum())
    over_probability = float(
        probabilities[totals > numeric_line].sum() + distribution.residual_probability
    )

    total_probability = under_probability + push_probability + over_probability
    if not isclose(
        total_probability,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise PricingValidationError("Under, push, and over probabilities must sum to one.")

    return TotalMarketPrice(
        line=numeric_line,
        under_probability=under_probability,
        push_probability=push_probability,
        over_probability=over_probability,
        under_fair_decimal_odds=(
            _fair_decimal_odds(
                under_probability,
                push_probability,
            )
        ),
        over_fair_decimal_odds=(
            _fair_decimal_odds(
                over_probability,
                push_probability,
            )
        ),
        covered_probability_mass=(distribution.covered_probability_mass),
        residual_probability=(distribution.residual_probability),
        maximum_total=(distribution.maximum_total),
    )


def compare_total_prices(
    baseline_price: TotalMarketPrice,
    candidate_price: TotalMarketPrice,
) -> TotalMarketModelRisk:
    """Quantify model risk between two prices for one line."""
    if not isclose(
        baseline_price.line,
        candidate_price.line,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PricingValidationError("Model-risk comparison requires the same total line.")

    under_difference = candidate_price.under_probability - baseline_price.under_probability
    push_difference = candidate_price.push_probability - baseline_price.push_probability
    over_difference = candidate_price.over_probability - baseline_price.over_probability
    maximum_absolute_difference = max(
        abs(under_difference),
        abs(push_difference),
        abs(over_difference),
    )

    return TotalMarketModelRisk(
        line=baseline_price.line,
        baseline_price=baseline_price,
        candidate_price=candidate_price,
        under_probability_difference=(under_difference),
        push_probability_difference=(push_difference),
        over_probability_difference=(over_difference),
        maximum_absolute_probability_difference=(maximum_absolute_difference),
        maximum_absolute_difference_basis_points=(maximum_absolute_difference * 10_000.0),
    )


def _fair_decimal_odds(
    win_probability: float,
    push_probability: float,
) -> float:
    """Return break-even decimal odds with stake returned on push."""
    if win_probability <= 0.0:
        return float("inf")

    active_probability = 1.0 - push_probability
    return active_probability / win_probability


def _validate_distribution_configuration(
    maximum_total: int,
    tail_tolerance: float,
) -> None:
    """Validate finite-support approximation settings."""
    if isinstance(maximum_total, bool) or not isinstance(maximum_total, int) or maximum_total < 1:
        raise PricingValidationError("maximum_total must be a positive integer.")

    if (
        isinstance(tail_tolerance, bool)
        or not isinstance(
            tail_tolerance,
            (int, float),
        )
        or not 0.0 < float(tail_tolerance) < 1.0
    ):
        raise PricingValidationError("tail_tolerance must be between zero and one.")


def _validate_market_line(
    line: float,
    maximum_total: int,
) -> float:
    """Validate an integer or half-goal market line."""
    if isinstance(line, bool) or not isinstance(line, (int, float)) or not isfinite(float(line)):
        raise PricingValidationError("Total line must be a finite number.")

    numeric_line = float(line)
    if numeric_line < 0.0 or numeric_line >= maximum_total:
        raise PricingValidationError("Total line must be non-negative and below maximum_total.")

    doubled_line = numeric_line * 2.0
    if not isclose(
        doubled_line,
        round(doubled_line),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise PricingValidationError("Total line must use integer or half-goal increments.")

    return numeric_line
