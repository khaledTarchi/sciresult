"""Shared test fixtures and helpers.

Nothing in this module imports private production helpers for the purpose of
establishing correct behaviour. Private seams are used only where the test must
exercise an implementation control directly, such as the block size.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

from sciresult import numeric

FIXTURE_DIR = Path(__file__).parent / "fixtures"
UPDATE_SNAPSHOTS_ENV = "SCIRESULT_UPDATE_SNAPSHOTS"

settings.register_profile(
    "sciresult",
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "sciresult-thorough",
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "sciresult"))


@pytest.fixture
def set_block_size(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    """Return a helper that overrides the private numeric block size."""

    def _set(value: int) -> None:
        assert value > 0
        monkeypatch.setattr(numeric, "_NUMERIC_BLOCK_SIZE", value)

    return _set


@pytest.fixture
def memmap_factory(tmp_path: Path) -> Callable[..., np.memmap]:
    """Return a helper that writes a real on-disk NumPy memmap."""

    def _make(values: np.ndarray, name: str = "data.bin", dtype: str = "float64") -> np.memmap:
        path = tmp_path / name
        array = np.asarray(values, dtype=dtype)
        mapped = np.memmap(path, dtype=array.dtype, mode="w+", shape=array.shape)
        mapped[...] = array
        mapped.flush()
        return mapped

    return _make


def assert_text_snapshot(name: str, text: str) -> None:
    """Compare rendered text against a reviewed fixture.

    Set ``SCIRESULT_UPDATE_SNAPSHOTS=1`` to rewrite the fixtures after an
    intentional formatting change, then review the diff before committing.
    """
    path = FIXTURE_DIR / f"{name}.txt"
    if os.environ.get(UPDATE_SNAPSHOTS_ENV) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return
    assert path.exists(), f"missing text fixture {path.name}; regenerate it deliberately"
    assert text == path.read_text(encoding="utf-8"), f"{path.name} does not match"


@pytest.fixture
def text_snapshot() -> Callable[[str, str], None]:
    """Return the text snapshot assertion helper."""
    return assert_text_snapshot
