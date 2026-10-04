# Diagnostics

What a report contains, how examples are chosen, and what the numbers mean.

## Population counts

When `numeric_scan_complete` is `True`, `report.counts` is a `PairCounts` with
these fields, each an exact integer:

| Field | Meaning |
| --- | --- |
| `total` | number of pairs examined |
| `finite_pairs` | pairs where both values are finite |
| `finite_failures` | finite pairs that exceeded their allowance |
| `nan_position_mismatches` | exactly one side is NaN |
| `matching_nan_pairs` | both NaN, `equal_nan=True` |
| `rejected_nan_pairs` | both NaN, `equal_nan=False` |
| `matching_infinity_pairs` | both the same signed infinity |
| `infinity_mismatches` | infinity involved, not matching |

The categories partition the pairs:

```text
total == finite_pairs + nan_position_mismatches + matching_nan_pairs
             + rejected_nan_pairs + matching_infinity_pairs + infinity_mismatches
```

`finite_pairs == finite_matches + finite_failures`, where `finite_matches` is
derived as `finite_pairs - finite_failures`.

No failures at all is reported as a string reason, not as an empty list, so
`report.max_finite_absolute_error` and the category counts can never be read as
"zero" when they mean "absent".

When `numeric_scan_complete` is `False`, `counts` is `None`, not zero-filled.
Zero would mean "we looked and found nothing", which is a different claim.

## Failure categories

Each `FailureExample` carries exactly one `FailureKind`:

| Kind | Meaning |
| --- | --- |
| `finite_tolerance` | both finite, discrepancy exceeded the allowance |
| `nan_position_mismatch` | one side NaN, the other not |
| `nan_rejected` | both NaN, `equal_nan=False` |
| `infinity_mismatch` | infinities that do not agree, or an infinity against a finite value |

The finite fields of an example (`absolute_error`, `allowed_error`, `score`,
`score_saturated`) are populated only for `finite_tolerance`. For every other
kind they are `None`. A NaN example never displays a fabricated zero error.

## The finite score

For a finite pair that failed:

```text
score = absolute_error / allowed_error
```

It is a dimensionless ratio, so it is comparable across positions with different
magnitudes. The property name is `FailureExample.score`; it is not `exceedance`,
because a value of 243902.32 does **not** mean the error is 243902.32 times
*over* the allowance. It means the error is 243902.32 times the allowance, which
every failing pair is by definition.

`allowed_error` is reported alongside it, in the same units as the data, because
the ratio alone cannot tell you whether the absolute numbers matter.

A saturated score renders as `score=+inf (score saturated at positive
infinity)`, and adds a report note explaining that computed-score ranking is not
exact-real ranking at saturated positions. A non-saturated score renders with no
marker at all.

## Saturation

Binary64 has a finite range, so `absolute_error / allowed_error` can overflow.

- **Positive allowance, quotient overflows.** `score` is `+inf` and
  `score_saturated` is `True`. The exact-real ratio is finite, so ranking these
  examples by computed score is *not* ranking by exact-real ratio. A note on the
  report says so. Two saturated examples cannot be ordered reliably against each
  other, and the tie-break falls through to position.
- **Zero allowance.** `score` is `+inf` and `score_saturated` is `False`. The
  exact-real ratio really is infinite, so these examples tie in exact-real
  ranking exactly as they tie here, and no fidelity has been lost.

The invariant is precise: `score_saturated` is `True` only when the score is
infinite *and* the true ratio is finite. Distinguishing the two cases matters,
because conflating them would report a fidelity loss that does not exist and
would make a zero-allowance failure look like an arithmetic limitation.

## Category ordering

Examples are ranked worst-first by kind, then by score, then by position. The
kind ordering is a fixed severity order:

```text
nan_position_mismatch > nan_rejected > infinity_mismatch > finite_tolerance
```

A NaN in a position that should hold a number is a more serious finding than a
value that is merely too large, so it outranks one regardless of magnitudes. The
ordering is a total order: integer positions are unique, so the result is fully
deterministic and stable across runs and platforms. It is a total order, not a
partial one, so the retained set is reproducible, not merely reproducible for
most inputs.

## Maximum finite error

`max_finite_absolute_error` reports the largest absolute error over *all* finite
pairs, including those that passed. It answers "how close was the worst point",
which the failure examples alone cannot: with a small `max_examples`, a large
passing error can be absent from the examples.

It is a `LocatedMetric` carrying the value, its integer position, and its
coordinate labels, so you can look it up. It is `None` when no finite pair was
examined. It is not a score and not a normalised quantity; ranking is done by
score, and this is reported separately for that reason.

## Coordinate labels

`FailureExample.labels` is a tuple of `(dimension_name, LabelValue)` pairs, one
entry per declared dimension, in dimension order. Wrap it in `dict()` when a
mapping reads better. Each value records its kind, its value, its unit, and
whether it was rendered by truncation:

```python
import numpy as np
import xarray as xr

import sciresult

actual = xr.DataArray(
    np.array([10.0, 20.0]), dims="time", coords={"time": np.array([2021.0, 2022.0])}
)
reference = xr.DataArray(
    np.array([10.0, 30.0]), dims="time", coords={"time": np.array([2021.0, 2022.0])}
)
label = dict(sciresult.compare(actual, reference, rtol=0.0, atol=0.0).examples[0].labels)["time"]

label.kind            # <LabelKind.FLOAT: 'float'>
label.value           # 2022.0
label.dtype_unit      # None, because the coordinate is not temporal
label.truncated       # False
label.original_length # None, because nothing was truncated
```

Kinds are `INTEGER`, `FLOAT`, `DATETIME`, `TIMEDELTA`, `BYTES`, `TEXT`, and
`POSITION`. `POSITION` is what you get for a dimension with no coordinate
variable, where the label is the integer offset. The kind is kept separate from
the value so a rendered label is never mistaken for a parsed one. For a temporal
coordinate, `dtype_unit` carries the unit that is part of the structural
comparison; it is `None` for numeric and text labels.

## Duplicate labels

Duplicate coordinate labels are legal and never change a verdict. Integer
positions are always reported next to the labels, so two entries with
`time=2020.0` remain distinguishable:

```text
example[1] finite_tolerance: flat_index=1 index=(1,) {time=2020.0} ...
example[2] finite_tolerance: flat_index=2 index=(2,) {time=2020.0} ...
```

Reading positions from labels alone is not safe, and the library does not
pretend otherwise. Labels are read from the **actual** operand; since the
coordinates already compared equal, both operands carry the same labels at each
position, which is why a coordinate difference is a structural finding rather
than a label ambiguity.

## Truncation

Label values longer than 100 characters are truncated with a `...` suffix, and
the entry is marked `truncated=True`. Truncation is announced, never silent, and
it applies to display only: the values in `report.counts`,
`max_finite_absolute_error`, and every score are exact.

`max_examples` (default 5) caps the retained examples. It changes neither the
verdict nor the counts nor the maximum error, so a caller can raise it for a
detailed report without changing what the comparison means.

## Rendering

`report.format()` produces deterministic plain text. The report below is
constructed here, so the rendering that follows it is executed in the test suite
rather than trusted:

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
    np.array([10.0, 20.5, 30.0]),
    dims="time",
    coords={"time": np.array([2020.0, 2021.0, 2022.0])},
    name="run_a",
)
report = sciresult.compare(actual, reference, rtol=1e-7, atol=1e-12)
```

```text
sciresult comparison report
status: different (numerical difference)
actual: name=run_b dims=(time,) shape=(3,) dtype=<f8
reference: name=run_a dims=(time,) shape=(3,) dtype=<f8
options: rtol=1e-07 atol=1e-12 equal_nan=False max_examples=5

structural issues: 0
  structure is compatible

numerical scan: complete (every pair was classified and evaluated)
  counts: total=3 finite_pairs=3 finite_failures=1 nan_position_mismatches=0 matching_nan_pairs=0 rejected_nan_pairs=0 matching_infinity_pairs=0 infinity_mismatches=0
  total failures: 1
  finite population: 3 finite pair(s)
  max finite absolute error: 0.5 at flat index 1 index=(1,) {time="2021.0"}

examples: 1 shown, cap 5, 1 pair(s) failed
  failure kinds:
    finite_tolerance: finite tolerance failure (discrepancy exceeds allowance)
    infinity: infinity mismatch (signed infinities disagree or pair with finite)
    nan_policy: NaN-policy rejection (NaN on both sides, equal_nan=False)
    nan_position: NaN-position mismatch (NaN on exactly one side)
  example[1] finite_tolerance: flat_index=1 index=(1,) {time="2021.0"} actual=20.0 reference=20.5 absolute_error=0.5 allowed_error=2.0500009999999997e-06 score=243902.3200476488

No alignment, interpolation, unit conversion, or input mutation was performed.
```

`saturated=False` is omitted from an example line, and is shown only when the
score overflowed. The `failure kinds` legend is always printed in full, in a
fixed order, whether or not any of those kinds occurred in this report.

Line order is fixed, number formatting is fixed, and identical inputs produce
byte-identical output. Text fixtures in `tests/fixtures/` pin every line, and
`tests/test_docs.py` checks that this block matches what `format()` returns.

## Machine-readable output

`report.to_dict()` returns JSON-safe primitives: enums become strings, positions
become lists, NaN and infinity become tagged objects, and no NumPy scalar or
array survives. `report.to_json()` renders it.

Semantic comparison of two reports is done through that encoding, not through
`==`:

```python
def build_the_same_report_twice():
    """Return two independently computed reports over identical inputs."""
    left = xr.DataArray(np.array([np.nan]), dims="x")
    right = xr.DataArray(np.array([1.0]), dims="x")
    return (
        sciresult.compare(left, right, rtol=0.0, atol=0.0),
        sciresult.compare(left, right, rtol=0.0, atol=0.0),
    )


left, right = build_the_same_report_twice()

left.to_dict() == right.to_dict()   # True
left.format() == right.format()     # True
left == right                       # False
```

`left == right` is `False` whenever either report contains a NaN, because
dataclass equality compares floating-point payloads and `NaN != NaN`. Two reports
that came from separate `compare()` calls hold separate NaN objects, so identity
does not rescue the comparison. Comparing a report with *itself* does return
`True`, since CPython's tuple comparison short-circuits on identical objects;
that inconsistency is another reason not to rely on `==`.

This is a Python property, not a `sciresult` defect. Use the encoding, or
`format()`, whenever you want to know whether two reports mean the same thing.
The deterministic nature of the encoding also makes it the right basis for
snapshot tests.