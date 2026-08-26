"""Tests for the Conway-Maxwell-Poisson model."""

from collections.abc import Sequence
from math import log
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from scipy.stats import poisson

import courtquant.models.com_poisson as com_poisson
from courtquant.models.com_poisson import (
    ComPoissonConvergenceError,
    ConwayMaxwellPoissonModel,
)
from courtquant.models.poisson import ModelValidationError


def _model(
    *,
    intensity: float = 8.0,
    dispersion: float = 1.0,
    sample_size: int = 100,
    tail_tolerance: float = 1e-12,
    maximum_support: int = 10_000,
) -> ConwayMaxwellPoissonModel:
    return ConwayMaxwellPoissonModel(
        intensity=intensity,
        dispersion=dispersion,
        sample_size=sample_size,
        tail_tolerance=tail_tolerance,
        maximum_support=maximum_support,
    )


def _as_integer_sequence(values: object) -> Sequence[int]:
    return cast(Sequence[int], values)


@pytest.mark.parametrize(
    "intensity",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_model_rejects_invalid_intensity(
    intensity: float,
) -> None:
    with pytest.raises(
        ModelValidationError,
        match="intensity must be finite and positive",
    ):
        _model(intensity=intensity)


@pytest.mark.parametrize(
    "dispersion",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_model_rejects_invalid_dispersion(
    dispersion: float,
) -> None:
    with pytest.raises(
        ModelValidationError,
        match="dispersion must be finite and positive",
    ):
        _model(dispersion=dispersion)


@pytest.mark.parametrize(
    "sample_size",
    [0, -1, True, 1.5],
)
def test_model_rejects_invalid_sample_size(
    sample_size: object,
) -> None:
    with pytest.raises(
        ModelValidationError,
        match="sample_size must be a positive integer",
    ):
        _model(sample_size=cast(int, sample_size))


@pytest.mark.parametrize(
    "tail_tolerance",
    [
        0.0,
        -1.0,
        1.0,
        float("inf"),
        float("nan"),
    ],
)
def test_model_rejects_invalid_tail_tolerance(
    tail_tolerance: float,
) -> None:
    with pytest.raises(
        ModelValidationError,
        match="tail_tolerance must be finite",
    ):
        _model(tail_tolerance=tail_tolerance)


@pytest.mark.parametrize(
    "maximum_support",
    [0, -1, True, 1.5],
)
def test_model_rejects_invalid_maximum_support(
    maximum_support: object,
) -> None:
    with pytest.raises(
        ModelValidationError,
        match="maximum_support must be a positive integer",
    ):
        _model(
            maximum_support=cast(
                int,
                maximum_support,
            )
        )


def test_poisson_limit_matches_scipy_probability_mass() -> None:
    model = _model(
        intensity=12.5,
        dispersion=1.0,
    )
    counts = list(range(31))

    actual = model.probability_mass(counts)
    expected = poisson.pmf(counts, mu=12.5)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


def test_poisson_limit_has_expected_moments() -> None:
    model = _model(
        intensity=12.5,
        dispersion=1.0,
    )

    mean, variance = model.moments()

    assert mean == pytest.approx(12.5, abs=1e-8)
    assert variance == pytest.approx(12.5, abs=1e-7)
    assert model.mean == pytest.approx(12.5, abs=1e-8)
    assert model.variance == pytest.approx(12.5, abs=1e-7)
    assert model.dispersion_index == pytest.approx(
        1.0,
        abs=1e-8,
    )
    assert model.is_underdispersed is False


def test_underdispersed_model_reports_lower_variance() -> None:
    model = _model(
        intensity=25.0,
        dispersion=2.0,
    )

    assert model.is_underdispersed is True
    assert model.variance < model.mean
    assert model.dispersion_index < 1.0


def test_probability_mass_sums_to_one() -> None:
    model = _model(
        intensity=25.0,
        dispersion=2.0,
    )

    probabilities = model.probability_mass(list(range(101)))

    assert float(probabilities.sum()) == pytest.approx(
        1.0,
        abs=1e-11,
    )


def test_log_likelihood_matches_probability_mass() -> None:
    model = _model(
        intensity=9.0,
        dispersion=1.4,
    )
    counts = [2, 4, 7, 8]

    expected = float(np.log(model.probability_mass(counts)).sum())

    assert model.log_likelihood(counts) == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_half_goal_market_matches_poisson_limit() -> None:
    model = _model(
        intensity=8.0,
        dispersion=1.0,
    )

    market = model.price_total(8.5)

    assert market.under == pytest.approx(
        poisson.cdf(8, mu=8.0),
        abs=1e-11,
    )
    assert market.over == pytest.approx(
        poisson.sf(8, mu=8.0),
        abs=1e-11,
    )
    assert market.push == 0.0
    assert market.over + market.under == pytest.approx(1.0)


def test_integer_market_includes_push_probability() -> None:
    model = _model(
        intensity=8.0,
        dispersion=1.0,
    )

    market = model.price_total(8.0)

    assert market.under == pytest.approx(
        poisson.cdf(7, mu=8.0),
        abs=1e-11,
    )
    assert market.push == pytest.approx(
        poisson.pmf(8, mu=8.0),
        abs=1e-11,
    )
    assert market.over == pytest.approx(
        poisson.sf(8, mu=8.0),
        abs=1e-11,
    )
    assert (market.over + market.under + market.push) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "line",
    [
        -0.5,
        1.25,
        float("inf"),
        float("nan"),
        True,
    ],
)
def test_market_rejects_invalid_lines(
    line: object,
) -> None:
    with pytest.raises(ModelValidationError):
        _model().price_total(cast(float, line))


@pytest.mark.parametrize(
    "counts",
    [
        [],
        [True],
        ["not-a-count"],
        [[1, 2]],
        [-1],
        [1.5],
        [float("inf")],
        [float("nan")],
    ],
)
def test_probability_mass_rejects_invalid_counts(
    counts: object,
) -> None:
    with pytest.raises(ModelValidationError):
        _model().probability_mass(_as_integer_sequence(counts))


def test_fit_recovers_synthetic_underdispersion() -> None:
    true_model = _model(
        intensity=25.0,
        dispersion=2.0,
        sample_size=2_000,
    )
    support = np.arange(30, dtype=np.int64)
    support_list = [int(value) for value in support]
    probabilities = true_model.probability_mass(support_list)
    probabilities /= probabilities.sum()

    generator = np.random.default_rng(42)
    simulated = generator.choice(
        support,
        size=2_000,
        p=probabilities,
    )
    sample = [int(value) for value in simulated]

    fitted = ConwayMaxwellPoissonModel.fit(sample)

    assert fitted.sample_size == 2_000
    assert fitted.dispersion == pytest.approx(
        2.0,
        rel=0.15,
    )
    assert fitted.mean == pytest.approx(
        true_model.mean,
        rel=0.03,
    )
    assert fitted.is_underdispersed is True


def test_fit_rejects_zero_sample_mean() -> None:
    with pytest.raises(
        ModelValidationError,
        match="positive sample mean",
    ):
        ConwayMaxwellPoissonModel.fit([0, 0, 0])


def test_fit_rejects_support_below_observed_maximum() -> None:
    with pytest.raises(
        ModelValidationError,
        match="at least the largest observed count",
    ):
        ConwayMaxwellPoissonModel.fit(
            [1, 2, 3],
            maximum_support=2,
        )


def test_fit_handles_zero_variance_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def successful_minimize(
        *_args: object,
        **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            success=True,
            message="success",
            x=np.asarray(
                [log(5.0), 0.0],
                dtype=np.float64,
            ),
        )

    monkeypatch.setattr(
        com_poisson,
        "minimize",
        successful_minimize,
    )

    fitted = ConwayMaxwellPoissonModel.fit([5, 5, 5])

    assert fitted.intensity == pytest.approx(5.0)
    assert fitted.dispersion == pytest.approx(1.0)
    assert fitted.sample_size == 3


def test_fit_raises_when_optimizer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_minimize(
        *_args: object,
        **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            success=False,
            message="forced failure",
            x=np.zeros(2, dtype=np.float64),
        )

    monkeypatch.setattr(
        com_poisson,
        "minimize",
        failed_minimize,
    )

    with pytest.raises(
        ComPoissonConvergenceError,
        match="forced failure",
    ):
        ConwayMaxwellPoissonModel.fit([1, 2, 3])


def test_fit_rejects_non_finite_optimizer_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def non_finite_minimize(
        *_args: object,
        **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            success=True,
            message="success",
            x=np.asarray(
                [0.0, float("nan")],
                dtype=np.float64,
            ),
        )

    monkeypatch.setattr(
        com_poisson,
        "minimize",
        non_finite_minimize,
    )

    with pytest.raises(
        ComPoissonConvergenceError,
        match="non-finite parameters",
    ):
        ConwayMaxwellPoissonModel.fit([1, 2, 3])


def test_normalizer_reports_insufficient_support() -> None:
    model = _model(
        intensity=50.0,
        dispersion=1.0,
        maximum_support=2,
    )

    with pytest.raises(
        ComPoissonConvergenceError,
        match="did not converge",
    ):
        model.moments()


def test_market_rejects_line_above_maximum_support() -> None:
    model = _model(
        intensity=1.0,
        dispersion=1.0,
        maximum_support=10,
    )

    with pytest.raises(
        ModelValidationError,
        match="minimum_support cannot exceed maximum_support",
    ):
        model.price_total(11.5)


def test_normalizer_handles_ratio_rounded_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rounded_exponential(_value: float) -> float:
        return 1.0

    monkeypatch.setattr(
        com_poisson,
        "exp",
        rounded_exponential,
    )

    with pytest.raises(
        ComPoissonConvergenceError,
        match="did not converge",
    ):
        com_poisson._log_weights(
            log_intensity=-1.0,
            dispersion=1.0,
            minimum_support=0,
            tail_tolerance=1e-12,
            maximum_support=1,
        )
