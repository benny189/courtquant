"""Tests for CourtQuant match-data loading and validation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from courtquant.data import DataValidationError, load_matches, validate_matches


def valid_frame() -> pd.DataFrame:
    """Return a minimal valid handball dataset."""
    return pd.DataFrame(
        {
            "saison": ["18/19", "18/19"],
            "spieltag": [1, 1],
            "spiel": [1, 2],
            "tore_mannschaft1": [26, 21],
            "tore_mannschaft2": [27, 18],
        }
    )


def test_load_matches_reads_semicolon_csv(tmp_path: Path) -> None:
    """A valid semicolon-separated CSV is loaded and normalized."""
    csv_path = tmp_path / "matches.csv"
    valid_frame().to_csv(csv_path, sep=";", index=False)

    matches = load_matches(csv_path)

    assert matches.shape == (2, 5)
    assert matches["saison"].tolist() == ["18/19", "18/19"]
    assert str(matches["saison"].dtype).startswith("string")
    assert str(matches["spieltag"].dtype) == "int64"


def test_load_matches_rejects_missing_file(tmp_path: Path) -> None:
    """A missing source file raises a descriptive error."""
    with pytest.raises(FileNotFoundError, match="Match data not found"):
        load_matches(tmp_path / "missing.csv")


def test_load_matches_wraps_parser_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Low-level parser failures are exposed as data-validation errors."""
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text("broken", encoding="utf-8")

    def raise_parser_error(*_: object, **__: object) -> pd.DataFrame:
        raise pd.errors.ParserError("broken CSV")

    monkeypatch.setattr(pd, "read_csv", raise_parser_error)

    with pytest.raises(DataValidationError, match="Could not parse match data"):
        load_matches(csv_path)


def test_validate_matches_rejects_missing_column() -> None:
    """All required source columns must be present."""
    frame = valid_frame().drop(columns=["spiel"])

    with pytest.raises(DataValidationError, match="Missing columns"):
        validate_matches(frame)


def test_validate_matches_rejects_unexpected_column() -> None:
    """Unexpected source columns cannot silently enter the model."""
    frame = valid_frame()
    frame["unexpected"] = 1

    with pytest.raises(DataValidationError, match="unexpected columns"):
        validate_matches(frame)


def test_validate_matches_rejects_empty_data() -> None:
    """An empty dataset cannot be used for model estimation."""
    with pytest.raises(DataValidationError, match="at least one row"):
        validate_matches(valid_frame().iloc[0:0])


def test_validate_matches_rejects_missing_values() -> None:
    """Missing observations are reported instead of silently removed."""
    frame = valid_frame()
    frame.loc[0, "saison"] = None

    with pytest.raises(DataValidationError, match="Missing values detected"):
        validate_matches(frame)


def test_validate_matches_rejects_invalid_season_format() -> None:
    """Season identifiers must use the YY/YY convention."""
    frame = valid_frame()
    frame.loc[0, "saison"] = "2018/19"

    with pytest.raises(DataValidationError, match="Invalid season identifiers"):
        validate_matches(frame)


def test_validate_matches_rejects_nonconsecutive_season() -> None:
    """A season must connect two consecutive calendar years."""
    frame = valid_frame()
    frame.loc[0, "saison"] = "18/20"

    with pytest.raises(DataValidationError, match="consecutive years"):
        validate_matches(frame)


def test_validate_matches_rejects_nonfinite_integer() -> None:
    """Infinite numeric observations are invalid."""
    frame = valid_frame()
    frame["spiel"] = pd.Series([np.inf, 2.0])

    with pytest.raises(DataValidationError, match="finite integers"):
        validate_matches(frame)


def test_validate_matches_rejects_fractional_integer() -> None:
    """Count and identifier columns cannot contain fractions."""
    frame = valid_frame()
    frame["spieltag"] = pd.Series([1.5, 1.0])

    with pytest.raises(DataValidationError, match="finite integers"):
        validate_matches(frame)


def test_validate_matches_rejects_nonpositive_matchday() -> None:
    """Matchdays start at one."""
    frame = valid_frame()
    frame.loc[0, "spieltag"] = 0

    with pytest.raises(DataValidationError, match="Matchdays must be positive"):
        validate_matches(frame)


def test_validate_matches_rejects_nonpositive_match_number() -> None:
    """Match numbers start at one."""
    frame = valid_frame()
    frame.loc[0, "spiel"] = 0

    with pytest.raises(DataValidationError, match="Match numbers must be positive"):
        validate_matches(frame)


def test_validate_matches_rejects_negative_goals() -> None:
    """Goal counts cannot be negative."""
    frame = valid_frame()
    frame.loc[0, "tore_mannschaft1"] = -1

    with pytest.raises(DataValidationError, match="Goal counts cannot be negative"):
        validate_matches(frame)


def test_validate_matches_rejects_duplicate_rows() -> None:
    """Exact duplicate observations are rejected."""
    frame = pd.concat([valid_frame(), valid_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(DataValidationError, match="Duplicate rows detected"):
        validate_matches(frame)


def test_validate_matches_rejects_duplicate_match_keys() -> None:
    """A season, matchday and match number identify exactly one game."""
    frame = valid_frame()
    duplicate_key = frame.iloc[[0]].copy()
    duplicate_key["tore_mannschaft1"] = 30
    frame = pd.concat([frame, duplicate_key], ignore_index=True)

    with pytest.raises(DataValidationError, match="Duplicate match keys detected"):
        validate_matches(frame)
