"""Poisson benchmark model for handball total-goal markets."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import floor, isclose, isfinite
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson


class ModelValidationError(ValueError):
    """Raised when model inputs violate the Poisson count-data contract."""


@dataclass(frozen=True, slots=True)
class TotalMarketProbabilities:
    """Probabilities for an over/under market, including an optional push."""

    line: float
    over: float
    under: float
    push: float

    def __post_init__(self) -> None:
        _validated_line(self.line)
        probabilities = (self.over, self.under, self.push)
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ModelValidationError("Market probabilities must be finite and lie in [0, 1].")
        if not isclose(
            sum(probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ModelValidationError("Over, under and push probabilities must sum to one.")

    @property
    def fair_over_odds(self) -> float:
        """Return fair decimal over odds conditional on the bet not pushing."""
        return (1.0 - self.push) / self.over if self.over > 0.0 else float("inf")

    @property
    def fair_under_odds(self) -> float:
        """Return fair decimal under odds conditional on the bet not pushing."""
        return (1.0 - self.push) / self.under if self.under > 0.0 else float("inf")


@dataclass(frozen=True, slots=True)
class PoissonTotalModel:
    """Maximum-likelihood Poisson benchmark for total goals."""

    rate: float
    sample_size: int

    def __post_init__(self) -> None:
        if not isfinite(self.rate) or self.rate < 0.0:
            raise ModelValidationError("Poisson rate must be finite and non-negative.")
        if self.sample_size < 1:
            raise ModelValidationError("Sample size must be positive.")

    @classmethod
    def fit(cls, counts: Iterable[int | float]) -> Self:
        """Estimate the Poisson rate by maximum likelihood."""
        values = _validated_counts(counts)
        return cls(
            rate=float(values.mean()),
            sample_size=int(values.size),
        )

    def probability_mass(
        self,
        counts: Iterable[int | float],
    ) -> NDArray[np.float64]:
        """Evaluate the fitted probability mass function at integer counts."""
        values = _validated_counts(counts)
        return np.asarray(
            poisson.pmf(values, mu=self.rate),
            dtype=np.float64,
        )

    def log_likelihood(self, counts: Iterable[int | float]) -> float:
        """Return the joint out-of-sample log-likelihood of observed counts."""
        values = _validated_counts(counts)
        log_probabilities = np.asarray(
            poisson.logpmf(values, mu=self.rate),
            dtype=np.float64,
        )
        return float(log_probabilities.sum())

    def price_total(self, line: float) -> TotalMarketProbabilities:
        """Price a whole- or half-goal over/under line.

        Whole-number lines can push when total goals equal the line. Half-goal
        lines have no push outcome.
        """
        validated_line = _validated_line(line)
        cutoff = floor(validated_line)

        if validated_line.is_integer():
            under = float(poisson.cdf(cutoff - 1, mu=self.rate))
            push = float(poisson.pmf(cutoff, mu=self.rate))
            over = float(poisson.sf(cutoff, mu=self.rate))
        else:
            under = float(poisson.cdf(cutoff, mu=self.rate))
            push = 0.0
            over = float(poisson.sf(cutoff, mu=self.rate))

        return TotalMarketProbabilities(
            line=validated_line,
            over=over,
            under=under,
            push=push,
        )


def _validated_counts(
    counts: Iterable[int | float],
) -> NDArray[np.float64]:
    values = np.asarray(list(counts), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ModelValidationError("Counts must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all():
        raise ModelValidationError("Counts must be finite.")
    if bool((values < 0.0).any()):
        raise ModelValidationError("Counts cannot be negative.")
    if not np.equal(values, np.floor(values)).all():
        raise ModelValidationError("Counts must be integers.")
    return values


def _validated_line(line: float) -> float:
    numeric_line = float(line)
    if not isfinite(numeric_line) or numeric_line < 0.0:
        raise ModelValidationError("Market line must be finite and non-negative.")

    doubled_line = numeric_line * 2.0
    if not isclose(
        doubled_line,
        round(doubled_line),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ModelValidationError("Market line must use whole- or half-goal increments.")
    return numeric_line
