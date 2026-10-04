"""Peak resident memory of a comparison, measured in a fresh subprocess.

Python allocation tracing is not enough here: NumPy allocates its buffers in
native code, so ``tracemalloc`` cannot see them. This script therefore reads the
operating system's peak resident set size for a child process that performs one
scenario, and reports the growth over a baseline taken after the inputs are
built but before the comparison starts.

Run it directly::

    python benchmarks/memory_probe.py
    python benchmarks/memory_probe.py --size 5000000

The result is an observation about this machine and these inputs. It is not a
process-wide memory guarantee and says nothing about any other platform.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent

SCENARIOS = (
    "match",
    "sparse_failures",
    "dense_failures",
    "all_nan",
    "mixed_nonfinite",
    "coordinate_heavy",
    "memmap",
)

# Injected verbatim into the child. Kept as source text because the child must
# not import this module, whose own imports would skew its baseline.
PEAK_RSS_HELPER = '''
def peak_rss_bytes():
    """Peak resident set size of this process, in bytes, from the OS."""
    try:
        import resource
        import sys as _sys
    except ImportError:
        pass
    else:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kibibytes; the BSDs and macOS report bytes.
        reports_bytes = _sys.platform.startswith("darwin") or "bsd" in _sys.platform
        return int(raw) if reports_bytes else int(raw) * 1024

    import ctypes

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    # The pseudo-handle from GetCurrentProcess must be declared as a pointer, or
    # the default c_int return type truncates it and the query fails.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Counters),
        ctypes.c_uint32,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        code = ctypes.get_last_error()
        raise OSError(f"GetProcessMemoryInfo failed with Win32 error {code}")
    return int(counters.PeakWorkingSetSize)
'''

CHILD = (
    """
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import sciresult

"""
    + PEAK_RSS_HELPER
    + """

scenario, size, rtol, atol, equal_nan, cap = json.loads(sys.argv[1])
rng = np.random.default_rng(20261003)
base = rng.normal(size=size)
zero = np.zeros(size)


def build(name):
    if name == "match":
        return base, base.copy(), None
    if name == "sparse_failures":
        left = base.copy()
        left[:: max(size // 1000, 1)] += 5.0
        return left, base, None
    if name == "dense_failures":
        return base + 1.0, zero, None
    if name == "all_nan":
        missing = np.full(size, np.nan)
        return missing, missing.copy(), None
    if name == "mixed_nonfinite":
        left = base.copy()
        left[:: max(size // 8, 1)] = np.nan
        left[1:: max(size // 8, 1)] = np.inf
        return left, base, None
    if name == "coordinate_heavy":
        # Auxiliary coordinates must be given their dimension explicitly, or
        # xarray would infer a dimension of the same name for them.
        axis = np.arange(size)
        coords = {
            "x": axis.astype(np.float64),
            "station_id": ("x", (axis % 97).astype(np.int64)),
            "quality": ("x", np.linspace(0.0, 1.0, size)),
            "flag": ("x", (axis % 2).astype(np.int8)),
        }
        return base, base.copy(), coords
    if name == "memmap":
        # Scratch files live in a temporary directory that is removed when this
        # short-lived child process exits. Writing them into the current working
        # directory instead would litter the checkout, and a benchmark that
        # leaves megabytes behind is a bug, not a feature.
        # The scratch directory is created and removed by the parent, because
        # Windows refuses to delete a file that is still mapped, and this
        # process is what holds the mappings open.
        scratch = Path(sys.argv[2])
        left = np.memmap(scratch / "left.bin", dtype=np.float64, mode="w+", shape=(size,))
        right = np.memmap(scratch / "right.bin", dtype=np.float64, mode="w+", shape=(size,))
        left[:] = base
        right[:] = base
        return left, right, None
    raise SystemExit("unknown scenario " + scenario)


left_values, right_values, coords = build(scenario)
actual = xr.DataArray(left_values, dims=("x",), coords=coords)
reference = xr.DataArray(right_values, dims=("x",), coords=coords)

# One warm-up pass first. Freshly allocated pages may still be uncommitted, so
# the first call would otherwise charge the input arrays' own first touch to the
# comparison. The reported growth is the steady-state cost of a repeated call.
report = sciresult.compare(
    actual, reference, rtol=rtol, atol=atol, equal_nan=equal_nan, max_examples=cap
)
report.format()
del report
baseline = peak_rss_bytes()
report = sciresult.compare(
    actual, reference, rtol=rtol, atol=atol, equal_nan=equal_nan, max_examples=cap
)
text = report.format()
peak = peak_rss_bytes()

print(json.dumps({
    "scenario": scenario,
    "elements": size,
    "status": report.status.value,
    "examples": len(report.examples),
    "text_bytes": len(text),
    "baseline_bytes": baseline,
    "peak_bytes": peak,
    "growth_bytes": max(0, peak - baseline),
    "input_bytes": int(np.asarray(actual.data).nbytes + np.asarray(reference.data).nbytes),
}))
"""
)


def _child_environment() -> dict[str, str]:
    """Make the package importable in the child without installing it."""
    environment = dict(os.environ)
    entries = [str(_HERE.parent / "src"), str(_HERE)]
    existing = environment.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def measure(
    scenario: str,
    size: int,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    equal_nan: bool = False,
    cap: int = 5,
) -> dict[str, Any]:
    """Run one scenario in a fresh interpreter and return its measurement."""
    request = json.dumps([scenario, size, rtol, atol, equal_nan, cap])
    with tempfile.TemporaryDirectory(prefix="sciresult-memmap-") as scratch:
        completed = subprocess.run(
            [sys.executable, "-c", CHILD, request, scratch],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(_HERE),
            env=_child_environment(),
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"memory probe child failed for scenario {scenario!r} "
            f"(exit {completed.returncode}):\n{completed.stderr.strip()}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2_000_000)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="emit raw JSON only")
    args = parser.parse_args()

    rows = [
        measure(
            scenario,
            args.size,
            rtol=args.rtol,
            atol=args.atol,
            equal_nan=False,
            cap=args.max_examples,
        )
        for scenario in SCENARIOS
    ]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    header = (
        f"{'scenario':<20}{'elements':>12}{'input MiB':>12}"
        f"{'growth MiB':>13}{'ratio':>8}{'status':>12}"
    )
    print(f"peak resident memory probe, {args.size:,} elements per scenario")
    print(header)
    print("-" * len(header))
    for row in rows:
        input_mib = row["input_bytes"] / (1024 * 1024)
        growth_mib = row["growth_bytes"] / (1024 * 1024)
        ratio = row["growth_bytes"] / row["input_bytes"] if row["input_bytes"] else 0.0
        print(
            f"{row['scenario']:<20}{row['elements']:>12,}{input_mib:>12.1f}"
            f"{growth_mib:>13.1f}{ratio:>8.2f}{row['status']:>12}"
        )

    print("\nNotes:")
    print("- Every row is a fresh interpreter, so no measurement inherits another's")
    print("  allocator state.")
    print("- Each scenario runs compare() once as a warm-up, then measures a second")
    print("  call. Freshly allocated pages can still be uncommitted, so the first call")
    print("  would charge the inputs' own first touch to the comparison. The figure")
    print("  reported is the steady-state cost of a repeated call, not the first.")
    print("- Peak resident set size comes from the operating system: getrusage on Unix")
    print("  and GetProcessMemoryInfo on Windows. Both are normalised to bytes.")
    print("- Growth tracks the scan block and its temporaries rather than the element")
    print("  count, so the ratio is expected to shrink as the arrays grow.")
    print("- This is a measurement on this machine and is not a memory guarantee.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
