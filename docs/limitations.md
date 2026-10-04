# Limitations

Read this before relying on the library. Several of these are deliberate design
decisions, and the refusal to guess is the point.

## The reference may be wrong

The library does not know which operand is correct. It has no notion of truth, no
source of record, and no way to tell an intentional change from an error.

`actual` and `reference` are labels, not verdicts on authority. If you pass the
same array twice, `sciresult` will not notice anything suspicious about it. If
the wrong file is your reference, a `DIFFERENT` report is still a `DIFFERENT`
report, and a `MATCH` report still tells you only that two inputs agreed.

It is also not a validation framework. It cannot check units, ranges, physical
consistency, monotonicity, or any domain rule.

## Tolerances remain your responsibility

`rtol` and `atol` are required and have no defaults, because a default would
decide how wrong your science may be. Choosing them is a scientific judgement
about your domain, your instrument, and your units. This library will not make it
for you, and it will not warn that your tolerance is loose.

`rtol` scales with `abs(reference)`, not with the magnitude of the difference. A
large reference and a large allowed error go together. If your accuracy scales
with absolute magnitude instead, `atol` carries the whole burden.

The relation is inclusive, so a discrepancy exactly equal to the allowance
passes. If you need strict inequality, use a tolerance one unit tighter.

A loose tolerance is accepted without comment, because choosing one is a
scientific judgement this library will not make for you:

```python
import numpy as np
import xarray as xr

import sciresult

actual = xr.DataArray(np.array([10.0, 20.0]), dims="x")
reference = xr.DataArray(np.array([100.0, 200.0]), dims="x")

# Every pair differs by 90% of the reference, and this call says MATCH.
# Nothing warns you: rtol=1e-7 there would say DIFFERENT.
permissive = sciresult.compare(actual, reference, rtol=1.0, atol=0.0)
assert permissive.status is sciresult.ComparisonStatus.MATCH

strict = sciresult.compare(actual, reference, rtol=1e-7, atol=0.0)
assert strict.status is sciresult.ComparisonStatus.DIFFERENT
```

## Pointwise comparison may be scientifically inappropriate

Position-by-position equality is the only relation implemented, and it is the
wrong question whenever a legitimate difference is distributed.

- **Different grids.** Comparing two model runs on different meshes is not
  pointwise. There is no sensible `rtol` between a value sampled at `0.25 m`
  and one sampled at `0.5 m`.
- **Different vertical coordinates.** A sigma level and a pressure level are
  different quantities even when their labels look alike.
- **Phase or lag shifts.** A waveform shifted by half a sample has no matching
  positions, though it is arguably the same signal.
- **Coordinate-dependent tolerances.** When acceptable error depends on the
  coordinate (higher near a boundary, lower elsewhere), this library cannot
  express it.
- **Statistical or ensemble comparison.** Two ensemble members are not compared
  pointwise to one another. Compare summary statistics first.

In every one of these cases the library will refuse rather than align,
interpolate, or regrid for you. That refusal is the intended behaviour.

## Eager-only support

Only exact `numpy.ndarray` and `numpy.memmap` backends are supported.

Not supported: Dask arrays, CuPy arrays, sparse arrays, masked arrays,
`np.matrix`, and any other `ndarray` subclass. This is not an oversight. Every
numerical guarantee here is about binary64 arithmetic, block-wise iteration, and
the absence of hidden copies; those cannot be guaranteed through an array
backend you did not design for.

If your data is lazy, materialise the slice you intend to compare and pass that.

Rejection happens before any comparison work, with a role-prefixed reason code:

```python
import numpy as np
import xarray as xr

import sciresult

integer_data = xr.DataArray(np.array([1, 2, 3]), dims="x")

try:
    sciresult.compare(integer_data, integer_data.copy(), rtol=0.0, atol=0.0)
except sciresult.UnsupportedInputError as error:
    print(error)
    # the message names the offending role, so it is never ambiguous which of
    # the two operands was refused
    # actual data dtype <i8 is not supported; only real binary32 and binary64
    # data are supported
```

## Unsupported dtypes and indexes

- **Data:** integer, boolean, complex, float16, float128, object, string,
  datetime, and timedelta are all rejected. Compare in binary64.
- **Auxiliary coordinates:** complex, float16, and object are rejected. float16
  never reaches the check: pandas casts a float16 index to float64 first, with a
  `FutureWarning`.
- **Indexes:** only `xarray.indexes.PandasIndex` is accepted. A custom or
  foreign index family is rejected rather than converted, because the structural
  comparison of index values has no meaning without knowing its rules.
- **Scalar arrays** and any **zero-length** dimension are rejected. An empty
  comparison has no verdict, so `EmptyComparisonError` is raised instead of a
  vacuous `MATCH`.

## Admission I/O caveat

Admission reads `array.data` and inspects its type, shape, and dtype. With a
`numpy.memmap` this touches the operating system page cache, so the first access
can block on disk I/O and can raise an OS-level error if the backing file is
unavailable.

Admission is not sandboxed and does not prompt. Do not pass a mapping over an
untrusted or user-supplied path expecting the library to ask first. It reads only
what it needs and never writes.

## No hard memory cap

The numerical scan works in fixed-size blocks and allocates per-block
temporaries, so peak memory tracks the block size rather than the array size.
`benchmarks/memory_probe.py` measures this directly.

That is a design property with measured evidence, not a guarantee. There is no
enforced cap, no global memory accounting, and no cancellation interface. On a
memory-constrained host, a caller is still responsible for the working set, and a
huge memmap will fault in as the scan walks it.

Time is linear in element count plus a bounded cost per failure, measured at
roughly 3.5 microseconds per failing pair on the development machine. Rows where
every pair fails are dominated by building one diagnostic object per failure, so
their cost tracks the failure count rather than the array size. Do not treat any
of these numbers as a prediction for your hardware.

## No physical-significance ranking

The ranking is deterministic and documented, and it is not physical significance.

Ranking is by category severity, then by the dimensionless score
`absolute_error / allowed_error`, then by integer position. It knows nothing
about uncertainty, correlation, downstream sensitivity, or which values matter to
your conclusion. Two errors with equal scores are ordered by position, which is
arbitrary with respect to your science.

If the worst-looking error is in a region you do not care about, the report will
still rank it first. Conversely, a systematic small error across every point can
produce no alarming example at all while being scientifically serious.

## No repair

The library will not align, reindex, sort, transpose, broadcast, interpolate,
convert units, promote dtypes, or round coordinates, and it never mutates inputs.
When two arrays are not directly comparable it reports that instead of making
them comparable. If you need any of those operations, perform them yourself and
check that they were correct.

## Report and traceback privacy

A report can contain your data.

It includes operand names, dimension names, shapes, `dtype.str`, the tolerance
options, up to `max_examples` examples with coordinate labels and both operand
values, the largest finite absolute error with its label, and structural messages
quoting coordinate names and guarded attribute values. Tracebacks from
`assert_close` embed the rendered report.

Before sharing a report in an issue, a log, or a ticket, review it. Label values
longer than 100 characters are truncated with an explicit marker, but truncated
is not redacted, and ordinary content is not sanitised for HTML or for a
particular log viewer.

The library performs no network access and collects no telemetry. There is no
analytics and no crash reporting.

## No market-demand claim

No user research, survey, or demand analysis exists for this package. Nothing in
this repository should be read as evidence that anyone needs strict diagnostic
comparison, or that this is the right tool for them. The feature set is derived
from a written specification, not from users. Adopting it is your decision.

Reports are also easy to misread as authoritative when they are not: a report is
an observation about two arrays at one tolerance, not a judgement about either of
them.

`docs/release.md` states what has been verified and what has not, separately, so
that neither is mistaken for the other.

## Unverified claims

Everything here was verified on the development machine and its dependency
versions only. Not verified: other operating systems, other NumPy or Xarray
versions, distributed or accelerator hardware, behaviour under memory pressure,
and performance on any other machine.

## See also

- [semantics.md](semantics.md) - what is actually compared
- [diagnostics.md](diagnostics.md) - how examples are ranked and labelled
- [release.md](release.md) - verification status and release procedure