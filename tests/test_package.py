"""Smoke tests for the CourtQuant package."""

from courtquant import __version__


def test_package_version() -> None:
    """The installed package exposes the expected version."""
    assert __version__ == "0.1.0"
