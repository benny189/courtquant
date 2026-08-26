"""Dispersion-aware Conway-Maxwell-Poisson models for goal totals."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, floor, isclose, isfinite, log, log1p
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln

from courtquant.models.poisson import (
    ModelValidationError,
    TotalMarketProbabilities,
)

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]

DEFAULT_TAIL_TOLERANCE: Final[float] = 1e-12
DEFAULT_MAXIMUM_SUPPORT: Final[int] = 10_000
MINIMUM_DISPERSION: Final[float] = 0.05
MAXIMUM_DISPERSION: Final[float] = 20.0


class ComPoissonConvergenceError(RuntimeError):
    """Raised when COM-Poisson estimation or normalization fails."""


@dataclass(frozen=True, slots=True)
class ConwayMaxwellPoissonModel:
    """A Conway-Maxwell-Poisson model estimated by maximum likelihood.

    ``intensity`` is the conventional lambda parameter and is not generally
    equal to the distribution mean. ``dispersion`` is the conventional nu
    parameter: values above one indicate underdispersion, while a value of one
    reduces the model to a Poisson distribution.
    """

    intensity: float
    dispersion: float
    sample_size: int
    tail_tolerance: float = DEFAULT_TAIL_TOLERANCE
    maximum_support: int = DEFAULT_MAXIMUM_SUPPORT

    def __post_init__(self) -> None:
        if not isfinite(self.intensity) or self.intensity <= 0.0:
            raise ModelValidationError("intensity must be finite and positive.")
        if not isfinite(self.dispersion) or self.dispersion <= 0.0:
            raise ModelValidationError("dispersion must be finite and positive.")
        if (
            isinstance(self.sample_size, bool)
            or not isinstance(self.sample_size, int)
            or self.sample_size < 1
        ):
            raise ModelValidationError("sample_size must be a positive integer.")
        _validate_numerical_configuration(
            self.tail_tolerance,
            self.maximum_support,
        )

    @classmethod
    def fit(
        cls,
        counts: Sequence[int],
        *,
        tail_tolerance: float = DEFAULT_TAIL_TOLERANCE,
        maximum_support: int = DEFAULT_MAXIMUM_SUPPORT,
    ) -> "ConwayMaxwellPoissonModel":
        """Estimate intensity and dispersion using maximum likelihood."""
        _validate_numerical_configuration(
            tail_tolerance,
            maximum_support,
        )
        observed = _validate_counts(counts)
        maximum_observed = int(observed.max())

        if maximum_observed > maximum_support:
            raise ModelValidationError(
                "maximum_support must be at least the largest observed count."
            )

        observed_float = observed.astype(np.float64)
        sample_mean = float(observed_float.mean())

        if sample_mean <= 0.0:
            raise ModelValidationError("COM-Poisson estimation requires a positive sample mean.")

        sample_variance = float(observed_float.var())
        initial_dispersion = (
            1.0
            if sample_variance <= 0.0
            else float(
                np.clip(
                    sample_mean / sample_variance,
                    0.25,
                    4.0,
                )
            )
        )
        initial_parameters = np.asarray(
            [log(sample_mean), log(initial_dispersion)],
            dtype=np.float64,
        )

        sample_size = len(observed)
        sum_counts = float(observed_float.sum())
        log_factorials = np.asarray(
            gammaln(observed_float + 1.0),
            dtype=np.float64,
        )
        sum_log_factorials = float(log_factorials.sum())
        maximum_location = max(
            2.0 * maximum_observed + 10.0,
            2.0 * sample_mean + 10.0,
            10.0,
        )

        def objective_and_gradient(
            parameters: FloatArray,
        ) -> tuple[float, FloatArray]:
            log_location = float(parameters[0])
            dispersion = exp(float(parameters[1]))
            log_intensity = dispersion * log_location

            log_weights, log_normalizer = _log_weights(
                log_intensity=log_intensity,
                dispersion=dispersion,
                minimum_support=maximum_observed,
                tail_tolerance=tail_tolerance,
                maximum_support=maximum_support,
            )
            probabilities = np.exp(log_weights - log_normalizer)
            support = np.arange(
                len(log_weights),
                dtype=np.float64,
            )
            expected_count = float(np.dot(support, probabilities))
            expected_log_factorial = float(
                np.dot(
                    np.asarray(
                        gammaln(support + 1.0),
                        dtype=np.float64,
                    ),
                    probabilities,
                )
            )

            negative_log_likelihood = (
                sample_size * log_normalizer
                - sum_counts * log_intensity
                + dispersion * sum_log_factorials
            )
            intensity_gradient = sample_size * expected_count - sum_counts
            dispersion_gradient = sum_log_factorials - sample_size * expected_log_factorial
            gradient = np.asarray(
                [
                    dispersion * intensity_gradient,
                    dispersion * (log_location * intensity_gradient + dispersion_gradient),
                ],
                dtype=np.float64,
            )
            return float(negative_log_likelihood), gradient

        def objective(parameters: FloatArray) -> float:
            return objective_and_gradient(parameters)[0]

        def gradient(parameters: FloatArray) -> FloatArray:
            return objective_and_gradient(parameters)[1]

        parameter_bounds = [
            (
                log(1e-6),
                log(maximum_location),
            ),
            (
                log(MINIMUM_DISPERSION),
                log(MAXIMUM_DISPERSION),
            ),
        ]
        result = minimize(
            objective,
            initial_parameters,
            method="L-BFGS-B",
            jac=gradient,
            bounds=parameter_bounds,
            options={
                "maxiter": 500,
                "maxls": 50,
                "ftol": 1e-12,
                "gtol": 1e-8,
            },
        )

        if not bool(result.success):
            result = minimize(
                objective,
                np.asarray(
                    result.x,
                    dtype=np.float64,
                ),
                method="SLSQP",
                jac=gradient,
                bounds=parameter_bounds,
                options={
                    "maxiter": 500,
                    "ftol": 1e-12,
                },
            )

        if not bool(result.success):
            raise ComPoissonConvergenceError(
                f"COM-Poisson optimization failed after L-BFGS-B and SLSQP: {result.message}"
            )

        optimum = np.asarray(result.x, dtype=np.float64)
        dispersion = exp(float(optimum[1]))
        log_intensity = dispersion * float(optimum[0])
        intensity = exp(log_intensity)

        if not isfinite(intensity) or not isfinite(dispersion):
            raise ComPoissonConvergenceError(
                "COM-Poisson optimization returned non-finite parameters."
            )

        return cls(
            intensity=intensity,
            dispersion=dispersion,
            sample_size=sample_size,
            tail_tolerance=tail_tolerance,
            maximum_support=maximum_support,
        )

    @property
    def is_underdispersed(self) -> bool:
        """Return whether the fitted dispersion parameter exceeds one."""
        return self.dispersion > 1.0

    def moments(self) -> tuple[float, float]:
        """Return the model-implied mean and variance."""
        log_weights, log_normalizer = self._log_distribution()
        probabilities = np.exp(log_weights - log_normalizer)
        support = np.arange(
            len(log_weights),
            dtype=np.float64,
        )
        mean = float(np.dot(support, probabilities))
        variance = float(
            np.dot(
                np.square(support - mean),
                probabilities,
            )
        )
        return mean, variance

    @property
    def mean(self) -> float:
        """Return the model-implied expected count."""
        return self.moments()[0]

    @property
    def variance(self) -> float:
        """Return the model-implied count variance."""
        return self.moments()[1]

    @property
    def dispersion_index(self) -> float:
        """Return the model-implied variance-to-mean ratio."""
        mean, variance = self.moments()
        return variance / mean

    def probability_mass(
        self,
        counts: Sequence[int],
    ) -> FloatArray:
        """Evaluate the probability mass function at given counts."""
        evaluated = _validate_counts(counts)
        log_intensity = log(self.intensity)

        _, log_normalizer = _log_weights(
            log_intensity=log_intensity,
            dispersion=self.dispersion,
            minimum_support=int(evaluated.max()),
            tail_tolerance=self.tail_tolerance,
            maximum_support=self.maximum_support,
        )
        evaluated_float = evaluated.astype(np.float64)
        log_probabilities = (
            evaluated_float * log_intensity
            - self.dispersion * gammaln(evaluated_float + 1.0)
            - log_normalizer
        )
        return np.asarray(
            np.exp(log_probabilities),
            dtype=np.float64,
        )

    def log_likelihood(
        self,
        counts: Sequence[int],
    ) -> float:
        """Return the joint log-likelihood of observations."""
        evaluated = _validate_counts(counts)
        log_intensity = log(self.intensity)

        _, log_normalizer = _log_weights(
            log_intensity=log_intensity,
            dispersion=self.dispersion,
            minimum_support=int(evaluated.max()),
            tail_tolerance=self.tail_tolerance,
            maximum_support=self.maximum_support,
        )
        evaluated_float = evaluated.astype(np.float64)

        return float(
            (
                evaluated_float * log_intensity
                - self.dispersion * gammaln(evaluated_float + 1.0)
                - log_normalizer
            ).sum()
        )

    def price_total(
        self,
        line: float,
    ) -> TotalMarketProbabilities:
        """Price over/under probabilities for a valid goal line."""
        validated_line = _validate_market_line(line)
        cutoff = floor(validated_line)
        log_weights, log_normalizer = self._log_distribution(
            minimum_support=cutoff,
        )
        probabilities = np.exp(log_weights - log_normalizer)

        if validated_line.is_integer():
            under = float(probabilities[:cutoff].sum())
            push = float(probabilities[cutoff])
        else:
            under = float(probabilities[: cutoff + 1].sum())
            push = 0.0

        over = max(0.0, 1.0 - under - push)

        return TotalMarketProbabilities(
            line=validated_line,
            over=over,
            under=under,
            push=push,
        )

    def _log_distribution(
        self,
        *,
        minimum_support: int = 0,
    ) -> tuple[FloatArray, float]:
        return _log_weights(
            log_intensity=log(self.intensity),
            dispersion=self.dispersion,
            minimum_support=minimum_support,
            tail_tolerance=self.tail_tolerance,
            maximum_support=self.maximum_support,
        )


def _log_weights(
    *,
    log_intensity: float,
    dispersion: float,
    minimum_support: int,
    tail_tolerance: float,
    maximum_support: int,
) -> tuple[FloatArray, float]:
    if minimum_support > maximum_support:
        raise ModelValidationError("minimum_support cannot exceed maximum_support.")

    log_weights = [0.0]
    log_weight = 0.0
    log_normalizer = 0.0
    log_tolerance = log(tail_tolerance)

    for count in range(1, maximum_support + 1):
        log_weight += log_intensity - dispersion * log(count)
        log_weights.append(log_weight)
        log_normalizer = float(np.logaddexp(log_normalizer, log_weight))

        if count < minimum_support:
            continue

        log_next_ratio = log_intensity - dispersion * log(count + 1)
        if log_next_ratio >= 0.0:
            continue

        next_ratio = exp(log_next_ratio)
        if next_ratio >= 1.0:
            continue

        log_tail_bound = log_weight + log_next_ratio - log1p(-next_ratio)
        if log_tail_bound - log_normalizer <= log_tolerance:
            return (
                np.asarray(
                    log_weights,
                    dtype=np.float64,
                ),
                log_normalizer,
            )

    raise ComPoissonConvergenceError(
        "COM-Poisson normalizing constant did not converge before "
        f"maximum_support={maximum_support}."
    )


def _validate_counts(
    counts: Sequence[int],
) -> IntArray:
    values = list(counts)

    if not values:
        raise ModelValidationError("Counts cannot be empty.")
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ModelValidationError("Counts must be non-negative integers.")

    try:
        numeric = np.asarray(
            values,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise ModelValidationError("Counts must be non-negative integers.") from error

    if numeric.ndim != 1:
        raise ModelValidationError("Counts must be one-dimensional.")
    if (
        not np.isfinite(numeric).all()
        or (numeric < 0.0).any()
        or not np.equal(
            numeric,
            np.floor(numeric),
        ).all()
    ):
        raise ModelValidationError("Counts must be non-negative integers.")

    return numeric.astype(np.int64)


def _validate_market_line(line: float) -> float:
    if isinstance(line, bool) or not isfinite(line) or line < 0.0:
        raise ModelValidationError("Market line must be finite and non-negative.")

    doubled_line = 2.0 * line
    if not isclose(
        doubled_line,
        round(doubled_line),
        abs_tol=1e-10,
    ):
        raise ModelValidationError("Market line must be an integer or half-integer.")

    return float(line)


def _validate_numerical_configuration(
    tail_tolerance: float,
    maximum_support: int,
) -> None:
    if not isfinite(tail_tolerance) or tail_tolerance <= 0.0 or tail_tolerance >= 1.0:
        raise ModelValidationError(
            "tail_tolerance must be finite and strictly between zero and one."
        )
    if (
        isinstance(maximum_support, bool)
        or not isinstance(maximum_support, int)
        or maximum_support < 1
    ):
        raise ModelValidationError("maximum_support must be a positive integer.")
