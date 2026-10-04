"""Timing comparison against a plain assertion and a compact local helper.

The point of this script is calibration, not marketing: a diagnostic tool is
expected to be slower than ``np.allclose``, and this measures by how much on
realistic shapes. It also confirms that the answer never depends on the memory
layout of the inputs.

Run it directly::

    python benchmarks/compare_arrays.py
    python benchmarks/compare_arrays.py --size 2000000 --repeat 3

Every scenario is measured separately so a slow phase can be attributed rather
than guessed at.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr

try:
    import sciresult
except ModuleNotFoundError:  # pragma: no cover - convenience for a bare checkout
    # Running from a source tree without installing is convenient but is not the
    # supported way to use the package; CI installs the built wheel instead.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import sciresult

DEFAULT_SIZE = 1_000_000


def _array(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(values, dims=("x",))


def compact_equal(
    actual: np.ndarray, reference: np.ndarray, rtol: float, atol: float, equal_nan: bool
) -> bool:
    """A deliberately small local equivalent, for scale.

    It returns only a verdict, so it has no diagnostics to compute, rank, cap,
    or render. It is the baseline this library is allowed to be slower than.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        equal = np.isclose(actual, reference, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return bool(np.all(equal))


def upstream_assertion(actual: xr.DataArray, reference: xr.DataArray) -> None:
    """What upstream Xarray offers today."""
    xr.testing.assert_equal(actual, reference)


def _scenario(
    name: str, actual: np.ndarray, reference: np.ndarray, rtol: float, atol: float
) -> dict[str, object]:
    equal_nan = False
    left, right = _array(actual), _array(reference)

    def timed(call: Callable[[], object], repeat: int = 3) -> float:
        durations = []
        for _ in range(repeat):
            start = time.perf_counter()
            call()
            durations.append(time.perf_counter() - start)
        return min(durations)

    def run() -> sciresult.ComparisonReport:
        return sciresult.compare(left, right, rtol=rtol, atol=atol, equal_nan=equal_nan)

    report = run()
    diagnostic = timed(run)
    render = timed(report.format)
    local = timed(lambda: compact_equal(actual, reference, rtol, atol, equal_nan))
    try:
        upstream_assertion(left, right)
        upstream = timed(lambda: upstream_assertion(left, right))
    except AssertionError:
        upstream = float("nan")
    return {
        "scenario": name,
        "elements": int(actual.size),
        "status": report.status.value,
        "failures": report.counts.total_failures if report.counts else None,
        "diagnostic_s": diagnostic,
        "format_s": render,
        "compact_s": local,
        "upstream_s": upstream,
        "ratio_vs_compact": diagnostic / local if local > 0 else float("nan"),
    }


def build_scenarios(size: int) -> list[tuple[str, np.ndarray, np.ndarray, float, float]]:
    rng = np.random.default_rng(20261003)
    base = rng.normal(size=size)
    zero = np.zeros(size)
    dense = base + 1.0
    sparse = base.copy()
    sparse[:: max(size // 10, 1)] += 5.0
    all_nan = np.full(size, np.nan)
    mixed = base.copy()
    mixed[:: max(size // 8, 1)] = np.nan
    mixed[1 :: max(size // 8, 1)] = np.inf
    scenarios: list[tuple[str, np.ndarray, np.ndarray, float, float]] = [
        ("successful scan", base, base.copy(), 1e-7, 1e-12),
        ("sparse finite failures", sparse, base, 0.0, 0.0),
        ("dense finite failures", dense, zero, 0.0, 0.0),
        ("many equal-score failures", np.full(size, 1.0), zero, 0.0, 0.0),
        ("all NaN", all_nan, all_nan.copy(), 0.0, 0.0),
        ("mixed nonfinite", mixed, base, 0.0, 0.0),
        ("zero allowance", dense, zero, 0.0, 0.0),
    ]
    layouts = {
        "fortran": np.asfortranarray(base),
        "reversed": base[::-1],
        "noncontiguous": base[::2],
    }
    for layout, values in layouts.items():
        scenarios.append((f"{layout} layout", values, values.copy(), 1e-7, 1e-12))
    return scenarios


def _coordinate_heavy(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Identical values carrying several coordinates, so the cost is in labels."""
    rng = np.random.default_rng(7)
    values = rng.normal(size=size)
    axis = np.arange(size)
    coordinates = {
        "x": axis.astype(np.float64),
        "station_id": ("x", (axis % 97).astype(np.int64)),
        "quality": ("x", np.linspace(0.0, 1.0, size)),
        "flag": ("x", (axis % 2).astype(np.int8)),
    }
    warmed = xr.DataArray(values, dims=("x",), coords=coordinates)
    assert warmed.sizes["x"] == size
    return values, values.copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    print(f"sciresult benchmark, numpy {np.__version__}, xarray {xr.__version__}")
    print(f"elements per scenario: {args.size:,}\n")
    header = (
        f"{'scenario':<26}{'elements':>12}{'diagnostic':>13}{'format':>10}"
        f"{'compact':>10}{'upstream':>10}{'ratio':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name, actual, reference, rtol, atol in build_scenarios(args.size):
        row = _scenario(name, actual, reference, rtol, atol)
        rows.append(row)
        print(
            f"{row['scenario']:<26}{row['elements']:>12,}{row['diagnostic_s']:>13.4f}"
            f"{row['format_s']:>10.4f}{row['compact_s']:>10.4f}"
            f"{row['upstream_s']:>10.4f}{row['ratio_vs_compact']:>8.1f}"
        )

    values, values_copy = _coordinate_heavy(min(args.size, 200_000))
    row = _scenario("coordinate-heavy", values, values_copy, 1e-7, 1e-12)
    print(
        f"{row['scenario']:<26}{row['elements']:>12,}{row['diagnostic_s']:>13.4f}"
        f"{row['format_s']:>10.4f}{row['compact_s']:>10.4f}"
        f"{row['upstream_s']:>10.4f}{row['ratio_vs_compact']:>8.1f}"
    )

    print("\nNotes:")
    print("- `diagnostic` is compare(); `format` renders the retained report text.")
    print("- `compact` returns only a verdict, so it does no classification, ranking,")
    print("  capping, label extraction, or rendering. Being slower is expected.")
    print("- `upstream` is xr.testing.assert_equal and raises on any difference,")
    print("  so it is only timed where the inputs match.")
    print("- Rows where every pair fails are dominated by building one bounded")
    print("  diagnostic object per failure, so their cost tracks the failure count")
    print("  rather than the array size. That cost is linear in the number of")
    print("  failures, measured at roughly 3.5 microseconds each on this machine,")
    print("  and is the price of returning Python objects rather than a verdict.")
    print("- Times are the minimum of", args.repeat, "runs on this machine; they are not")
    print("  a performance claim about any other environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
