"""Distribution diagnostics for handball goal counts."""

from dataclasses import asdict, dataclass
from math import isfinite, nan
from typing import Final

import pandas as pd

from courtquant.data import validate_matches

DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "scope",
    "matches",
    "mean_home_goals",
    "mean_away_goals",
    "home_advantage",
    "mean_total_goals",
    "variance_total_goals",
    "dispersion_index",
    "home_away_correlation",
)


@dataclass(frozen=True, slots=True)
class CountDiagnostics:
    """Summary statistics for one collection of handball matches."""

    scope: str
    matches: int
    mean_home_goals: float
    mean_away_goals: float
    home_advantage: float
    mean_total_goals: float
    variance_total_goals: float
    dispersion_index: float
    home_away_correlation: float


def add_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Validate raw matches and add pricing-relevant count features."""
    featured = validate_matches(matches)
    featured["total_goals"] = featured["tore_mannschaft1"] + featured["tore_mannschaft2"]
    featured["goal_difference"] = featured["tore_mannschaft1"] - featured["tore_mannschaft2"]
    return featured


def count_diagnostics(matches: pd.DataFrame) -> pd.DataFrame:
    """Calculate overall and season-level goal-count diagnostics.

    The variance uses the unbiased sample definition with ``ddof=1``. The
    dispersion index is sample variance divided by the sample mean, matching
    the Poisson equidispersion benchmark of one.
    """
    featured = add_match_features(matches)
    summaries = [_summarize(featured, scope="overall")]

    for season in featured["saison"].drop_duplicates().tolist():
        season_name = str(season)
        season_matches = featured.loc[featured["saison"] == season]
        summaries.append(_summarize(season_matches, scope=season_name))

    records = [asdict(summary) for summary in summaries]
    return pd.DataFrame.from_records(
        records,
        columns=list(DIAGNOSTIC_COLUMNS),
    )


def _summarize(matches: pd.DataFrame, scope: str) -> CountDiagnostics:
    home_goals = matches["tore_mannschaft1"].astype("float64")
    away_goals = matches["tore_mannschaft2"].astype("float64")
    total_goals = matches["total_goals"].astype("float64")

    mean_home = float(home_goals.mean())
    mean_away = float(away_goals.mean())
    mean_total = float(total_goals.mean())

    variance_total = float(total_goals.var(ddof=1)) if len(matches) > 1 else nan
    dispersion = (
        variance_total / mean_total if mean_total > 0.0 and isfinite(variance_total) else nan
    )

    correlation = nan
    if len(matches) > 1 and home_goals.nunique() > 1 and away_goals.nunique() > 1:
        correlation = float(home_goals.corr(away_goals))

    return CountDiagnostics(
        scope=scope,
        matches=len(matches),
        mean_home_goals=mean_home,
        mean_away_goals=mean_away,
        home_advantage=mean_home - mean_away,
        mean_total_goals=mean_total,
        variance_total_goals=variance_total,
        dispersion_index=dispersion,
        home_away_correlation=correlation,
    )
