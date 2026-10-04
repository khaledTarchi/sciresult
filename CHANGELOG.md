# Changelog

All notable changes to this project are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Categories are kept distinct because they have different consequences for you:

- **API changes** alter how you call the library.
- **Numerical semantic changes** alter what a verdict means. Read these twice.
- **Compatibility changes** alter which dependency versions work.
- **Formatting-only changes** alter rendered text and nothing else.
- **Bug fixes** correct behaviour that contradicted the documented rules.

## [Unreleased]

Nothing released yet. The version below is the development version.

### Added

- `compare()` returning an immutable `ComparisonReport`, and `assert_close()`
  raising `ComparisonMismatch` whose message is the rendered report.
- Strict structural comparison of dimension names and order, shape, data dtype,
  coordinate names, dimensions, shapes, dtypes and values, index presence, and
  index values, with one aggregate issue per difference category.
- Explicit classification of every pair into exactly one of: finite pair,
  NaN-position mismatch, matching NaN, rejected NaN pair, matching infinity, or
  infinity mismatch.
- Bounded, deterministic, coordinate-aware failure examples with a total
  ordering, an integer tie-break, and a `max_examples` cap that changes no
  verdict and no count.
- Plain-text rendering, strict JSON-safe encoding, and typed machine-readable
  error codes on every exception.
- Public record types (`ComparisonReport`, `PairCounts`, `StructuralIssue`,
  `FailureExample`, `LocatedMetric`, `LabelValue`) with validated invariants,
  frozen and slotted.
- `py.typed` marker and strict-mode type annotations across the package.
- Documentation: quickstart, semantics, diagnostics, limitations, release.
- `benchmarks/compare_arrays.py` and `benchmarks/memory_probe.py`.
- `tests/test_docs.py`, which executes every Python example in `README.md` and
  `docs/` and asserts the values those documents claim, so a hand-written number
  cannot drift away from the implementation.
- Test suite of unit, property-based, and differential tests, including an
  independent scalar oracle in `tests/oracles.py`.

### Numerical semantics

- `FailureExample.exceedance` renamed to `FailureExample.score`, which is the
  documented dimensionless ratio `absolute_error / allowed_error`. The name was
  changed because "exceedance" invites reading it as a multiple of the
  allowance, which it is not.
- A positive allowance whose quotient overflows binary64 reports
  `score=+inf` with `score_saturated=True`: the exact-real ratio is finite, so
  ranking by computed score is not ranking by exact-real ratio, and the report
  says so in a note.
- A zero allowance reports `score=+inf` with `score_saturated=False`. The
  exact-real ratio is genuinely infinite, so those examples tie in exact-real
  ranking exactly as they tie here. The report describes this infinity as
  intentional rather than as overflow.

### Fixed

- A NaN paired with an infinity was counted in both the NaN-position and the
  infinity category, so the categories summed to more than the number of pairs.
  Infinity classification now excludes any pair containing a NaN.
- `diagnostic_score` reported `score_saturated=True` for a zero allowance,
  which claimed a fidelity loss that does not occur.
- Report records accepted an infinite `score` while claiming its ranking was
  exact. The invariant now distinguishes overflow saturation from an intentional
  infinity.
- Structural differences in the data dtype or a guarded attribute no longer
  suppress the numerical scan, because neither changes which positions pair
  with which. Counts and examples are now reported for those inputs instead of
  being withheld.
- A non-numeric `name` attribute is reported with the role-prefixed code
  `actual_name_not_text` rather than a generic type error.
- Coordinate dtype fields report the exact `dtype.str`, including byte order,
  instead of a normalised form.
- `benchmarks/memory_probe.py` wrote its memmap fixtures into the working
  directory, so a run littered the checkout with `left.bin` and `right.bin` and
  both files were picked up into the sdist. The parent process now owns a
  temporary directory and removes it, and the sdist excludes `**/*.bin`.
- `tests/test_installation.py` located the repository root from the imported
  package, so every metadata test failed with a `FileNotFoundError` about
  `site-packages/pyproject.toml` when the suite was run against an installed
  wheel. The root now comes from the test file, and the metadata tests skip
  themselves outside a checkout.

### Known limitations

Recorded in [docs/limitations.md](docs/limitations.md) rather than here, so there
is one place to look. The most important: two reports built from separate
`compare()` calls are not `==` when either holds a NaN, because dataclass
equality compares floating-point payloads and `NaN != NaN`. Comparing a report
with itself does return `True`, because CPython short-circuits tuple comparison on
identical objects. Compare `report.to_dict()` or `report.format()` instead.

## [0.1.0] - 2026-10-03

First release. Strict, coordinate-aware comparison of two Xarray `DataArray`
values, with deterministic bounded diagnostics. Distributed under the MIT
licence, copyright (c) 2026 Khaled Tarchi.

### Compatibility

Verified on Python 3.12.10 with NumPy 2.0.2 and Xarray 2026.9.0. Declared
minimums are Python 3.11, NumPy 1.26, and Xarray 2024.10.0; see
[docs/release.md](docs/release.md) for exactly which combinations have been
executed. No licence was previously granted, so no earlier version exists, no
version is superseded, and there is nothing for existing users to reconcile.

[Unreleased]: https://github.com/khaledTarchi/sciresult/compare/main...HEAD
[0.1.0]: https://github.com/khaledTarchi/sciresult/releases/tag/0.1.0