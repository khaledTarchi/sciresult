# sciresult

Strict, coordinate-aware comparison of two Xarray `DataArray` values, with
diagnostics that name the position and label of what differs.

`sciresult` answers one narrow question precisely: **do these two arrays hold
the same scientific result?** It never aligns, interpolates, converts units, or
repairs anything. When two arrays are not directly comparable it says so and
stops, instead of guessing what you meant.

```python
import numpy as np
import xarray as xr
import sciresult

actual = xr.DataArray([10.0, 20.0, 30.0], dims="time",
                      coords={"time": [1.0, 2.0, 3.0]}, name="run_b")
reference = xr.DataArray([10.0, 20.0, 30.0], dims="time",
                         coords={"time": [1.0, 2.0, 3.0]}, name="run_a")

report = sciresult.compare(actual, reference, rtol=1e-7, atol=1e-12)
report.status            # <ComparisonStatus.MATCH: 'match'>
```

## Why not `xr.testing.assert_allclose`?

Upstream Xarray will transpose, align, or broadcast to make two arrays
comparable. That is helpful right up to the moment it is not: a comparison that
silently reinterprets your coordinates can report `MATCH` for two arrays that
are not the same result. `sciresult` is the opposite trade. It refuses anything
it cannot compare position by position, and when it does compare, it tells you
exactly which positions disagree and what they were.

| Question | `sciresult` |
| --- | --- |
| Did the time axis shift by one step? | `different`, coordinate values differ, no numerical scan |
| Which of 10 million values is wrong? | The top few, with labels and ratios |
| Is a NaN in the same place? | A separate category, never folded into a tolerance |
| Did the dtype change from `float32` to `float64`? | `different`, reported as a structural issue |
| Do the units match? | `different`, with both strings quoted and unconverted |
| How much memory does it use? | Bounded by a scan block, not the array size |

## Installation

```console
$ pip install sciresult
```

Requires Python 3.11 or newer, NumPy 1.26 or newer, and Xarray 2024.10.0 or
newer. The package is pure Python, ships `py.typed`, and has no optional
acceleration dependency.

> **Status: not published.** No license has been selected and the maintainer
> identity is not yet filled in, so this package is **not** available on PyPI
> and must not be treated as released. See [docs/release.md](docs/release.md)
> for exactly what is blocking publication and what has been verified.

## The two entry points

`compare` always returns a report:

```python
report = sciresult.compare(actual, reference, rtol=1e-7, atol=1e-12)

if report.status is sciresult.ComparisonStatus.DIFFERENT:
    print(report.format())
```

`assert_close` raises instead, for use in a test:

```python
sciresult.assert_close(actual, reference, rtol=1e-7, atol=1e-12)
# raises ComparisonMismatch, whose message is the rendered report
```

There is no third form, no boolean return, and no implicit truthiness: a
report has no `__bool__`, so `if report:` raises `TypeError` and you are made to
read `.status` explicitly.

## What it guarantees

- **No repair.** No broadcasting, transposition, alignment, reindexing,
  sorting, interpolation, unit conversion, or dtype promotion. If the two
  arrays are not already comparable element for element, that is reported.
- **Strict structural comparison.** Dimension names and order, shapes, data
  dtype, coordinate names, dimensions, shapes, dtypes, values, index presence,
  and index values are each checked in a fixed order.
- **Honest numerical evaluation.** Finite pairs pass when
  `abs(actual - reference) <= atol + rtol * abs(reference)`, evaluated in
  binary64 under an error state that turns overflow and invalid operations into
  an exception instead of a silently wrong answer.
- **Nonfinite values are their own category.** A NaN, a signed infinity, and a
  finite number are never compared with a tolerance, because no tolerance makes
  physical sense for them.
- **Deterministic, bounded diagnostics.** Failure examples are ranked by a
  total order and capped at `max_examples`. The same inputs always produce the
  same report.
- **Read-only.** Inputs are never mutated, and the report retains no reference
  to them.

## What it does not do

Read [docs/limitations.md](docs/limitations.md) before relying on it. In short:
no alignment, no unit awareness beyond reporting attribute differences, no
distributed or lazy arrays, no complex or integer data, no dimension names
beyond those you declare, and no guarantee about performance on any machine but
the one you measure.

## Documentation

- [docs/quickstart.md](docs/quickstart.md) - installation and first comparison
- [docs/semantics.md](docs/semantics.md) - the exact comparison rules
- [docs/diagnostics.md](docs/diagnostics.md) - the report and its ranking
- [docs/limitations.md](docs/limitations.md) - what is deliberately not supported
- [docs/release.md](docs/release.md) - verification status and release procedure
- [CHANGELOG.md](CHANGELOG.md) - release history
- [CONTRIBUTING.md](CONTRIBUTING.md) - how to contribute
- [SECURITY.md](SECURITY.md) - reporting a vulnerability

## Development

```console
$ pip install -e ".[test,dev]"
$ python -m pytest              # unit, property, and differential tests
$ python -m ruff check src tests benchmarks
$ python -m mypy src
```

Two independent implementations of the numerical specification are compared
against each other on generated inputs: `src/sciresult/` and
`tests/oracles.py`. Any disagreement fails the suite, including the case where
one implementation reports an arithmetic overflow and the other does not.

## Licence

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 Khaled Tarchi.

## Acknowledgements

This package implements a specification written for this project. It borrows
nothing from `xarray.testing`; the upstream comparison helpers are used in the
benchmark as a baseline to measure against, not as a dependency.