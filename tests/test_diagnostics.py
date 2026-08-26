"""Tests for CourtQuant distribution diagnostics."""

from math import isnan

import pandas as pd
import pytest

from courtquant.diagnostics import (
    DIAGNOSTIC_COLUMNS,
    add_match_features,
    count_diagnostics,
)


def sample_matches() -> pd.DataFrame:
    """Return a deterministic two-season dataset."""
    return pd.DataFrame(
        {
            "saison": ["18/19", "18/19", "19/20", "19/20"],
            "spieltag": [1, 1, 1, 1],
            "spiel": [1, 2, 1, 2],
            "tore_mannschaft1": [10, 20, 30, 40],
            "tore_mannschaft2": [5, 15, 20, 25],
        }
    )


def test_add_match_features_calculates_pricing_variables() -> None:
    """Total goals and goal difference are calculated without mutation."""
    matches = sample_matches()

    featured = add_match_features(matches)

    assert featured["total_goals"].tolist() == [15, 35, 50, 65]
    assert featured["goal_difference"].tolist() == [5, 5, 10, 15]
    assert "total_goals" not in matches.columns
    assert "goal_difference" not in matches.columns


def test_count_diagnostics_returns_overall_and_season_rows() -> None:
    """The result contains one overall row followed by every season."""
    diagnostics = count_diagnostics(sample_matches())

    assert diagnostics.columns.tolist() == list(DIAGNOSTIC_COLUMNS)
    assert diagnostics["scope"].tolist() == ["overall", "18/19", "19/20"]
    assert diagnostics["matches"].tolist() == [4, 2, 2]


def test_count_diagnostics_uses_sample_variance() -> None:
    """Dispersion is based on unbiased sample variance."""
    diagnostics = count_diagnostics(sample_matches())
    season = diagnostics.loc[diagnostics["scope"] == "18/19"].iloc[0]

    assert float(season["mean_home_goals"]) == pytest.approx(15.0)
    assert float(season["mean_away_goals"]) == pytest.approx(10.0)
    assert float(season["home_advantage"]) == pytest.approx(5.0)
    assert float(season["mean_total_goals"]) == pytest.approx(25.0)
    assert float(season["variance_total_goals"]) == pytest.approx(200.0)
    assert float(season["dispersion_index"]) == pytest.approx(8.0)
    assert float(season["home_away_correlation"]) == pytest.approx(1.0)


def test_count_diagnostics_handles_single_match() -> None:
    """Variance, dispersion and correlation are undefined for one match."""
    diagnostics = count_diagnostics(sample_matches().iloc[[0]])
    overall = diagnostics.iloc[0]

    assert int(overall["matches"]) == 1
    assert isnan(float(overall["variance_total_goals"]))
    assert isnan(float(overall["dispersion_index"]))
    assert isnan(float(overall["home_away_correlation"]))


def test_count_diagnostics_handles_zero_mean() -> None:
    """A zero scoring mean has no defined dispersion ratio."""
    matches = pd.DataFrame(
        {
            "saison": ["18/19", "18/19"],
            "spieltag": [1, 1],
            "spiel": [1, 2],
            "tore_mannschaft1": [0, 0],
            "tore_mannschaft2": [0, 0],
        }
    )

    overall = count_diagnostics(matches).iloc[0]

    assert float(overall["mean_total_goals"]) == pytest.approx(0.0)
    assert float(overall["variance_total_goals"]) == pytest.approx(0.0)
    assert isnan(float(overall["dispersion_index"]))
    assert isnan(float(overall["home_away_correlation"]))


def test_count_diagnostics_handles_constant_home_goals() -> None:
    """Correlation is undefined when home goals have no variation."""
    matches = pd.DataFrame(
        {
            "saison": ["18/19", "18/19"],
            "spieltag": [1, 1],
            "spiel": [1, 2],
            "tore_mannschaft1": [10, 10],
            "tore_mannschaft2": [5, 6],
        }
    )

    overall = count_diagnostics(matches).iloc[0]

    assert isnan(float(overall["home_away_correlation"]))


def test_count_diagnostics_handles_constant_away_goals() -> None:
    """Correlation is undefined when away goals have no variation."""
    matches = pd.DataFrame(
        {
            "saison": ["18/19", "18/19"],
            "spieltag": [1, 1],
            "spiel": [1, 2],
            "tore_mannschaft1": [10, 11],
            "tore_mannschaft2": [5, 5],
        }
    )

    overall = count_diagnostics(matches).iloc[0]

    assert isnan(float(overall["home_away_correlation"]))
