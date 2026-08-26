"""Loading and validation for CourtQuant match data."""

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

EXPECTED_COLUMNS: Final[tuple[str, ...]] = (
    "saison",
    "spieltag",
    "spiel",
    "tore_mannschaft1",
    "tore_mannschaft2",
)
INTEGER_COLUMNS: Final[tuple[str, ...]] = (
    "spieltag",
    "spiel",
    "tore_mannschaft1",
    "tore_mannschaft2",
)
MATCH_KEY: Final[tuple[str, ...]] = ("saison", "spieltag", "spiel")


class DataValidationError(ValueError):
    """Raised when a match dataset violates the CourtQuant data contract."""


def load_matches(path: str | Path) -> pd.DataFrame:
    """Load and validate a semicolon-separated handball match dataset.

    Args:
        path: Location of the raw CSV file.

    Returns:
        A validated copy with stable column order and dtypes.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        DataValidationError: If the file violates the expected schema or domain rules.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Match data not found: {csv_path}")

    try:
        frame = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DataValidationError(f"Could not parse match data: {csv_path}") from exc

    return validate_matches(frame)


def validate_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an in-memory match table and return a normalized copy."""
    actual_columns = tuple(str(column) for column in frame.columns)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(actual_columns))
    unexpected_columns = sorted(set(actual_columns) - set(EXPECTED_COLUMNS))
    if missing_columns or unexpected_columns:
        raise DataValidationError(
            "Invalid schema. "
            f"Missing columns: {missing_columns or 'none'}; "
            f"unexpected columns: {unexpected_columns or 'none'}."
        )

    if frame.empty:
        raise DataValidationError("Match data must contain at least one row.")

    validated = frame.loc[:, list(EXPECTED_COLUMNS)].copy()
    missing_counts = validated.isna().sum()
    columns_with_missing = {
        str(column): int(count) for column, count in missing_counts.items() if count > 0
    }
    if columns_with_missing:
        raise DataValidationError(f"Missing values detected: {columns_with_missing}")

    seasons = validated["saison"].astype("string").str.strip()
    invalid_season_format = ~seasons.str.fullmatch(r"\d{2}/\d{2}")
    if bool(invalid_season_format.any()):
        invalid_values = sorted(seasons.loc[invalid_season_format].astype(str).unique().tolist())
        raise DataValidationError(f"Invalid season identifiers: {invalid_values}")

    season_start = seasons.str[:2].astype("int64")
    season_end = seasons.str[-2:].astype("int64")
    invalid_season_sequence = season_end.ne((season_start + 1) % 100)
    if bool(invalid_season_sequence.any()):
        invalid_values = sorted(seasons.loc[invalid_season_sequence].astype(str).unique().tolist())
        raise DataValidationError(
            f"Season identifiers must span consecutive years: {invalid_values}"
        )
    validated["saison"] = seasons

    for column in INTEGER_COLUMNS:
        numeric = pd.to_numeric(validated[column], errors="coerce")
        values = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise DataValidationError(f"Column '{column}' must contain finite integers only.")
        validated[column] = numeric.astype("int64")

    if bool((validated["spieltag"] < 1).any()):
        raise DataValidationError("Matchdays must be positive integers.")
    if bool((validated["spiel"] < 1).any()):
        raise DataValidationError("Match numbers must be positive integers.")
    if bool((validated[["tore_mannschaft1", "tore_mannschaft2"]] < 0).any().any()):
        raise DataValidationError("Goal counts cannot be negative.")

    if bool(validated.duplicated().any()):
        raise DataValidationError("Duplicate rows detected.")
    if bool(validated.duplicated(subset=list(MATCH_KEY)).any()):
        raise DataValidationError(f"Duplicate match keys detected for {MATCH_KEY}.")

    return validated.reset_index(drop=True)
