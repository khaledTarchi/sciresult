# Semantics

This document is the contract. `tests/oracles.py` is written from this document
independently of the implementation, and `tests/test_properties.py` compares the
two on generated inputs. Where the code and this document disagree, one of them
is a bug.

## The two operands

`compare(actual, reference, *, rtol, atol, equal_nan=False, max_examples=5)`

Exactly two `xarray.DataArray` objects. Both tolerances are required and
keyword-only; there is no default tolerance, because a default would silently
decide how wrong your science may be.

The pair below is the reference example used throughout this document:

```python
import numpy as np
import xarray as xr

import sciresult

actual = xr.DataArray(
    np.array([10.0, 20.0, 30.0]),
    dims="time",
    coords={"time": np.array([2020.0, 2021.0, 2022.0])},
    name="run_b",
)
reference = xr.DataArray(
    np.array([10.0, 20.0, 30.0]),
    dims="time",
    coords={"time": np.array([2020.0, 2021.0, 2022.0])},
    name="run_a",
)

matching = sciresult.compare(actual, reference, rtol=1e-7, atol=1e-12)
assert matching.status is sciresult.ComparisonStatus.MATCH
assert matching.structural_issue_count == 0
```

## Supported inputs

| Property | Accepted | Rejected |
| --- | --- | --- |
| Container | `xarray.DataArray` | anything else, including raw `ndarray` |
| Data backend | exact `numpy.ndarray`, exact `numpy.memmap` | `dask.array.Array`, `cupy` arrays, masked arrays, `np.matrix`, any `ndarray` subclass |
| Data dtype | real binary32, real binary64 | integer, boolean, complex, float16, float128, object, string, datetime, timedelta |
| Coordinate dtype | real integer or floating (float32/float64), datetime64, timedelta64, UTF-8 bytes, unicode text | complex, float16, object |
| Index | `xarray.indexes.PandasIndex` | any custom or foreign index family |
| Shape | non-empty in total | any zero-length dimension, scalar data |

Rejections happen during admission, before any structural or numerical work,
and each carries a role-prefixed reason code such as
`actual_data_dtype_unsupported` or `reference_coordinate_dtype_unsupported`.

Float16 is rejected for data. It is not accepted as data because binary16
arithmetic is not a supported comparison basis, and accepting it silently would
be worse than refusing it. The admission check rejects it for coordinates too,
though on the pinned pandas a float16 coordinate cannot be indexed at all:
`pandas.Index` raises a `FutureWarning` and casts to float64, so in practice the
library refuses the input before it ever reaches comparison.

## Structural relation

Structural comparison runs in this fixed order and stops reporting new
categories once each has been recorded once. It is one aggregate issue per
category, not one issue per element.

| Order | Property | Issue code |
| --- | --- | --- |
| 1 | dimension names and order | `dimensions` |
| 2 | shape | `shape` |
| 3 | data dtype kind and width | `data_dtype` |
| 4 | coordinate names | `coordinate_names` |
| 5 | coordinate dimensions | `coordinate_dimensions` |
| 6 | coordinate shape | `coordinate_shape` |
| 7 | coordinate dtype kind, width, and temporal unit | `coordinate_dtype` |
| 8 | coordinate values | `coordinate_values` |
| 9 | index presence | `index_presence` |
| 10 | index values | `index_values` |
| 11 | guarded attributes | `guarded_attribute` |

### What "compatible" means

Every one of these must hold. There is no tolerance on any structural property,
including coordinates: a coordinate that differs by `1e-12` is a different
coordinate, and `sciresult` says so rather than deciding that it is close
enough.

### Data dtype

Compared by kind and itemsize only, so binary64 and big-endian binary64 are the
same dtype. The report still shows the exact `dtype.str` for both operands, so
byte order remains visible.

### Coordinate equality rules

- NaN and NaT at corresponding positions count as equal missing labels.
- Signed zeros compare equal.
- Duplicate labels are allowed. They are never an error, and they never change a
  verdict. Diagnostics always include integer positions so repeated labels stay
  unambiguous.
- Temporal coordinates compare as integer ticks plus unit. The unit is part of
  the comparison.

### When the numerical scan is withheld

The scan is withheld, `counts` is `None`, `numeric_scan_complete` is `False`,
`max_finite_absolute_error` is `None`, and `examples` is empty, when any of the
following break correspondence between the two arrays:

`dimensions`, `shape`, `coordinate_names`, `coordinate_dimensions`,
`coordinate_shape`, `coordinate_values`, `index_presence`, `index_values`

A data dtype difference or a guarded-attribute difference does **not** withhold
the scan. Neither changes which position pairs with which position, so the values
are still compared and still counted. The report is `DIFFERENT` because of the
dtype or attribute difference, while `counts` describes the values that were
examined.

## Metadata guards

Only these `attrs` keys are compared: `units`, `long_name`, `standard_name`,
`axis`, `positive`, `calendar`, `leap_month`, `leap_year`, `month_lengths`.

- Missing on one side only is a difference.
- Values must be text, integers, or booleans.
- Values are compared exactly, including case. No unit conversion is attempted,
  so `m/s` and `km/h` are reported as differing and left to you.
- All other `attrs` keys are ignored entirely.

The data variable's `name` is reported but never affects the verdict: `run_a` and
`run_b` are two names for the same comparison.

## Numerical relation

A pair is finite only when both values are finite. For a finite pair the pair
passes when

```text
absolute_error <= allowed_error
absolute_error = abs(actual - reference)
allowed_error  = atol + rtol * abs(reference)
```

The inequality is inclusive: a discrepancy of exactly the allowance passes.

### Binary64 operation order

The three steps are evaluated as separate binary64 operations, in this order,
under `np.errstate(over="raise", invalid="raise", under="ignore",
divide="raise")`:

1. `difference = actual - reference`
2. `absolute_error = abs(difference)`
3. `relative_part = rtol * abs(reference)`, then `allowed_error = atol + relative_part`

There is no fused multiply-add and no whole-array cast. The order is fixed
because a different order gives different rounding, and a comparison whose
rounding depends on rearrangement is not reproducible.

Underflow is ignored deliberately: an accepted scalar wider than binary64
narrowed to binary64 can underflow to zero, and that is arithmetic, not a
verdict.

### Overflow is an error, not a verdict

If a finite step overflows or becomes invalid, the comparison raises
`ComparisonArithmeticError` and produces **no report**. A report is never
returned with a partially computed population, because a count that silently
excluded overflowed pairs would be indistinguishable from a count of zero
failures.

The error state is always restored, including when an exception escapes.

## NaN and infinity

Every pair falls into exactly one category. A NaN paired with an infinity is a
NaN-position mismatch and nothing else, so the categories sum to the number of
pairs.

| Category | Condition | Outcome |
| --- | --- | --- |
| finite pair | both values finite | compared against the tolerance relation |
| NaN-position mismatch | exactly one side is NaN | failure |
| matching NaN | both NaN, `equal_nan=True` | pass |
| rejected NaN pair | both NaN, `equal_nan=False` | failure |
| matching infinity | both are the same signed infinity | pass |
| infinity mismatch | either side is infinite and the above does not apply | failure |

Every pair falls into exactly one category, so the categories sum to the number
of pairs:

```python
import numpy as np
import xarray as xr

import sciresult

# One pair of each kind: finite, NaN on one side, NaN on both, infinity pair,
# and an infinity against a finite value.
left = xr.DataArray(np.array([1.0, np.nan, np.nan, np.inf, np.inf]), dims="x")
right = xr.DataArray(np.array([1.0, np.nan, 2.0, np.inf, 1.0]), dims="x")

for equal_nan in (True, False):
    report = sciresult.compare(left, right, rtol=0.0, atol=0.0, equal_nan=equal_nan)
    counts = report.counts
    assert counts is not None
    assert counts.total == 5
    assert (
        counts.total
        == counts.finite_pairs
        + counts.nan_position_mismatches
        + counts.matching_nan_pairs
        + counts.rejected_nan_pairs
        + counts.matching_infinity_pairs
        + counts.infinity_mismatches
    )
```

Consequences worth stating:

- No tolerance rescues an infinity. `rtol=1.0, atol=1e308` still reports
  `actual=inf, reference=1e308` as an infinity mismatch.
- `equal_nan` affects data only. It never affects coordinate equality, where
  NaN labels always match.
- `NaT` behaves as a NaN label.

## Empty input

Any input with zero elements, including a zero-length dimension or a scalar
array, raises `EmptyComparisonError`. A population of zero pairs has no verdict:
reporting `MATCH` would claim an agreement that was never tested.

## Checked and ignored

**Checked**

- dimension names and order; shape; data dtype kind and width
- coordinate names, dimensions, shapes, dtypes, and values; index presence and
  values; guarded attributes; data variable name (reported only)
- every pair of values, classified into exactly one category

**Ignored**

- all other `attrs`; anything not in the guarded list
- anything not attached to the two `DataArray` objects: the global Xarray
  options, other variables, and anything outside these arrays
- memory layout: C, Fortran, reversed, and non-contiguous inputs all produce
  the same verdict, counts, and example ordering

**Never performed**

alignment, reindexing, sorting, transposition, broadcasting, interpolation,
unit conversion, dtype promotion, coordinate rounding, and input mutation.