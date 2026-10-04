"""Public orchestration for :mod:`sciresult`.

This module contains no numerical formula and no formatting formula. It
validates options, admits inputs, compares structure, stops before numerical
evaluation when the structure differs, otherwise scans numerically, builds
bounded diagnostics, and constructs the immutable report.
"""

from __future__ import annotations

import xarray as xr

from sciresult import numeric
from sciresult.diagnostics import (
    LabelResolver,
    build_examples,
    build_located_metric,
    label_availability_note,
)
from sciresult.errors import ComparisonMismatch
from sciresult.model import (
    ComparisonReport,
    ComparisonStatus,
    FailureExample,
    LabelValue,
    LocatedMetric,
)
from sciresult.numeric import ScanResult
from sciresult.structure import StructureResult, compare_structure
from sciresult.validation import Options, ValidatedPair, admit_pair, normalize_options

__all__ = ["assert_close", "compare"]


def compare(
    actual: xr.DataArray,
    reference: xr.DataArray,
    *,
    rtol: float,
    atol: float,
    equal_nan: bool = False,
    max_examples: int = 5,
) -> ComparisonReport:
    """Compare one supported ``DataArray`` with one reference ``DataArray``.

    Parameters
    ----------
    actual:
        The result under test. It is the first argument.
    reference:
        The reference the result is compared against. It is always the second
        argument, and the numerical relation is reference-relative.
    rtol:
        Relative tolerance. Mandatory; no default scientific tolerance is
        provided.
    atol:
        Absolute tolerance. Mandatory; no default scientific tolerance is
        provided.
    equal_nan:
        Whether a NaN on both sides counts as a match. Independent of coordinate
        NaN handling, which always treats corresponding coordinate NaNs as equal
        missing labels.
    max_examples:
        Number of retained failure examples, between 0 and 100 inclusive. It
        never affects structural totals, numerical counts, status, or whether
        scanning completes.

    Returns
    -------
    ComparisonReport
        An immutable, detached report.

    Raises
    ------
    InvalidComparisonArgument
        If an option is not normalizable.
    UnsupportedInputError
        If an input is outside the supported domain.
    EmptyComparisonError
        If an input contains zero elements.
    ComparisonArithmeticError
        If a finite arithmetic step overflows or becomes invalid. No partial
        report is produced.

    Notes
    -----
    ``MATCH`` means only that the supported comparison passed. The library
    verifies a declared relationship between two arrays that already exist. It
    makes no claim about scientific validity.

    Assure the caller that, for supported inputs and the documented
    finite-computation semantics, the declared labeled domains were compared and
    the supplied tolerance was evaluated without silently aligning,
    interpolating, converting, or mutating the inputs.
    """
    options = normalize_options(
        rtol=rtol, atol=atol, equal_nan=equal_nan, max_examples=max_examples
    )
    pair = admit_pair(actual, reference)
    structure = compare_structure(pair)
    if structure.issue_count and not structure.correspondence_preserved:
        return _structural_report(pair, options, structure)
    scan = numeric.scan_numeric(pair, options, block_size=numeric.block_size())
    return _numerical_report(pair, options, structure, scan)


def assert_close(
    actual: xr.DataArray,
    reference: xr.DataArray,
    *,
    rtol: float,
    atol: float,
    equal_nan: bool = False,
    max_examples: int = 5,
) -> None:
    """Assert that two supported arrays are equivalent.

    Calls :func:`compare` exactly once.

    Returns
    -------
    None
        When the report status is ``MATCH``.

    Raises
    ------
    ComparisonMismatch
        When the report status is ``DIFFERENT``. Derives from
        :class:`AssertionError` and carries the completed report in
        ``.report``. Its message is the rendered report.
    InvalidComparisonArgument, UnsupportedInputError, EmptyComparisonError,
    ComparisonArithmeticError
        Propagated unchanged. They are never converted into an ordinary
        assertion mismatch.
    """
    report = compare(
        actual,
        reference,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        max_examples=max_examples,
    )
    if report.status is ComparisonStatus.DIFFERENT:
        raise ComparisonMismatch(report)


def _structural_report(
    pair: ValidatedPair, options: Options, structure: StructureResult
) -> ComparisonReport:
    return ComparisonReport(
        status=ComparisonStatus.DIFFERENT,
        actual_name=pair.actual.name,
        reference_name=pair.reference.name,
        actual_dimensions=pair.actual.dimensions,
        reference_dimensions=pair.reference.dimensions,
        actual_shape=pair.actual.shape,
        reference_shape=pair.reference.shape,
        actual_dtype=pair.actual.dtype.str,
        reference_dtype=pair.reference.dtype.str,
        rtol=options.rtol,
        atol=options.atol,
        equal_nan=options.equal_nan,
        max_examples=options.max_examples,
        structural_issue_count=structure.issue_count,
        structural_issues=structure.issues,
        numeric_scan_complete=False,
        counts=None,
        examples=(),
        max_finite_absolute_error=None,
        notes=(
            "the numerical scan was not performed because the labeled comparison domains differ",
        ),
    )


def _numerical_report(
    pair: ValidatedPair,
    options: Options,
    structure: StructureResult,
    scan: ScanResult,
) -> ComparisonReport:
    if not structure.correspondence_preserved:  # pragma: no cover - guarded by the caller
        raise AssertionError("a numerical report requires preserved correspondence")

    resolver = LabelResolver(pair.actual)
    examples = build_examples(scan.candidates, resolver)
    metric = None
    if scan.max_absolute_value is not None and scan.max_absolute_index is not None:
        metric = build_located_metric(scan.max_absolute_value, scan.max_absolute_index, resolver)

    counts = scan.counts
    notes: list[str] = []
    if structure.issue_count:
        notes.append(
            "structural differences were established; positions still correspond "
            "one to one, so the numerical relation was evaluated and is reported "
            "independently of them"
        )
    if counts.finite_pairs == 0:
        notes.append(
            "zero finite pairs were compared; the numerical relation was not evaluated "
            "at any position"
        )
    nonfinite_population = counts.total - counts.finite_pairs
    if nonfinite_population:
        notes.append(
            f"{nonfinite_population} of {counts.total} pairs were nonfinite and were "
            "classified without evaluating the tolerance relation"
        )
    if any(example.score_saturated for example in examples):
        notes.append(
            "at least one diagnostic score saturated at positive infinity; "
            "computed-score ranking is not exact-real ranking at saturated positions"
        )
    if any(label.truncated for label in _iter_labels(examples, metric)):
        notes.append(
            "at least one displayed label was truncated to the display bound; "
            "comparison always used the original values"
        )
    availability = label_availability_note(resolver)
    if availability is not None:
        notes.append(availability)

    status = (
        ComparisonStatus.DIFFERENT
        if counts.total_failures or structure.issue_count
        else ComparisonStatus.MATCH
    )
    return ComparisonReport(
        status=status,
        actual_name=pair.actual.name,
        reference_name=pair.reference.name,
        actual_dimensions=pair.actual.dimensions,
        reference_dimensions=pair.reference.dimensions,
        actual_shape=pair.actual.shape,
        reference_shape=pair.reference.shape,
        actual_dtype=pair.actual.dtype.str,
        reference_dtype=pair.reference.dtype.str,
        rtol=options.rtol,
        atol=options.atol,
        equal_nan=options.equal_nan,
        max_examples=options.max_examples,
        structural_issue_count=structure.issue_count,
        structural_issues=structure.issues,
        numeric_scan_complete=True,
        counts=counts,
        examples=examples,
        max_finite_absolute_error=metric,
        notes=tuple(notes),
    )


def _iter_labels(
    examples: tuple[FailureExample, ...], metric: LocatedMetric | None
) -> tuple[LabelValue, ...]:
    labels = [label for example in examples for _, label in example.labels]
    if metric is not None:
        labels.extend(label for _, label in metric.labels)
    return tuple(labels)
