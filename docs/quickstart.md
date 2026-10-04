# Quickstart

## Installation

```console
$ pip install sciresult
```

From a checkout, for development:

```console
$ pip install -e ".[test,dev]"
```

Requires Python 3.11 or newer, NumPy 1.26 or newer, Xarray 2024.10.0 or newer.

To work against a checkout instead of the installed package, set
`PYTHONPATH=src`. See [release.md](release.md).

## A selected DataArray

Everything below uses this pair of arrays: the same values at the same labels,
with a name to make the rendered report readable.

```python
import numpy as np
import xarray as xr

actual = xr.DataArray(
    np.array([10.0, 20.0, 30.0]),
    dims="time",
    coords={"time": np.array([2020.0, 2021.0, 2022.0])},
    attrs={"units": "mm/day"},
    name="run_b",
)

reference = xr.DataArray(
    np.array([10.0, 20.0, 30.0]),
    dims="time",
    coords={"time": np.array([2020.0, 2021.0, 2022.0])},
    attrs={"units": "mm/day"},
    name="run_a",
)
```

## A passing call

`compare` returns a report. Read `.status`; do not test the report for
truthiness.

```python
import sciresult

passing = sciresult.compare(actual, reference, rtol=1e-7, atol=1e-12)

passing.status        # <ComparisonStatus.MATCH: 'match'>
passing.counts.total  # 3
passing.examples      # ()
```

`MATCH` means: every structural property checked was equal, and every pair of
values agreed under the tolerance relation. It does not mean the two arrays are
interchangeable in every downstream calculation.

## A structural mismatch

Change the time axis. The values are untouched, but the labels moved.

```python
shifted = reference.assign_coords(time=reference.time + 1.0)

structural = sciresult.compare(actual, shifted, rtol=1e-7, atol=1e-12)

structural.status                  # <ComparisonStatus.DIFFERENT: 'different'>
structural.structural_issue_count  # 2
[issue.code for issue in structural.structural_issues]
# [StructuralCode.COORDINATE_VALUES, StructuralCode.INDEX_VALUES]

structural.numeric_scan_complete   # False
structural.counts                   # None
structural.examples                 # ()
```

This is the important behaviour. The arrays are not comparable element for
element any more, so **no numerical scan happens and no counts are claimed**.
`counts` is `None` rather than zero-filled, because "we did not look" and "we
looked and found nothing" must not look the same.

It is tempting to reach for `xr.testing.assert_allclose` here, which would
happily align the two arrays and report success. That alignment is an assumption
about your intent, and it is exactly the assumption this library refuses to make.

## A numerical mismatch

Now keep the coordinates and change one value.

```python
perturbed = reference.copy()
perturbed[1] = 20.5

report = sciresult.compare(actual, perturbed, rtol=1e-7, atol=1e-12)

report.status                     # <ComparisonStatus.DIFFERENT: 'different'>
report.structural_issue_count     # 0
report.numeric_scan_complete      # True

report.counts.total               # 3
report.counts.finite_pairs        # 3
report.counts.finite_failures     # 1

example = report.examples[0]
example.kind                      # <FailureKind.FINITE_TOLERANCE: 'finite_tolerance'>
example.index                     # (1,)
example.actual, example.reference # (20.0, 20.5)
example.absolute_error            # 0.5
example.allowed_error             # 2.0500009999999997e-06
example.score                     # 243902.3200476488
example.score_saturated           # False
dict(example.labels)["time"].kind  # <LabelKind.FLOAT: 'float'>
dict(example.labels)["time"].value # 2021.0
```

The example names the position by integer index *and* by coordinate label, so
you can find it in the source data rather than counting from the end. `labels` is
a tuple of `(dimension_name, LabelValue)` pairs, one per declared dimension, in
dimension order; `dict(example.labels)` turns it into a mapping if that reads
better.

## Pytest use

`assert_close` raises instead of returning a report, and the exception message
is the rendered report.

```python
import pytest
import sciresult


def test_run_b_matches_run_a():
    sciresult.assert_close(actual, reference, rtol=1e-7, atol=1e-12)
```

A failure prints the whole report:

```text
sciresult comparison report
status: different (numerical difference)
actual: name=run_b dims=(time,) shape=(3,) dtype=<f8
reference: name=run_a dims=(time,) shape=(3,) dtype=<f8
options: rtol=1e-07 atol=1e-12 equal_nan=False max_examples=5
...
  example[1] finite_tolerance: flat_index=1 index=(1,) {time="2021.0"} actual=20.0 reference=20.5 ...
```

Two details worth knowing:

- If you want a custom message, catch it and re-raise, or compare and assert
  yourself:

  ```python
  report = sciresult.compare(actual, reference, rtol=1e-7, atol=1e-12)
  assert report.status is sciresult.ComparisonStatus.MATCH, report.format()  # noqa: E501
  ```

- Operational problems are **not** reported as assertion failures. An
  unsupported dtype raises `UnsupportedInputError`, an empty array raises
  `EmptyComparisonError`, and a bad tolerance raises
  `InvalidComparisonArgument`. Those are bugs in the test, not results, and they
  are not subclasses of `AssertionError`.

## Next

- [semantics.md](semantics.md) - the exact rules, including what is checked and
  what is ignored.
- [diagnostics.md](diagnostics.md) - how examples are ranked, labelled, and
  capped.
- [limitations.md](limitations.md) - read this before relying on the library.

Every example in this file is executed by the test suite, so it cannot drift
away from the implementation.