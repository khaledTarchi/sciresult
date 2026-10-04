"""Immutable public data model for :mod:`sciresult`.

Everything a report exposes is a frozen, slotted dataclass holding detached
values only: no NumPy arrays, no NumPy views, no
:class:`xarray.DataArray`, no mutable mappings, no callbacks, no closures and
no borrowed input objects.

Constructors validate their invariants eagerly. Validation is cheap and purely
structural; it never reads a caller's arrays.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

__all__ = [
    "CATEGORY_PRIORITY",
    "ComparisonReport",
    "ComparisonStatus",
    "FailureExample",
    "FailureKind",
    "LabelKind",
    "LabelValue",
    "LocatedMetric",
    "PairCounts",
    "StructuralCode",
    "StructuralIssue",
]

# Implementation and presentation controls. These are not scientific parameters
# and are deliberately not part of the version-one public API.
_MAX_STRUCTURAL_EXAMPLES: Final = 20
_MAX_EXAMPLES_LIMIT: Final = 100


class ComparisonStatus(StrEnum):
    """Outcome of a comparison that completed.

    Unsupported inputs, invalid arguments, empty inputs, and arithmetic errors
    are exceptions rather than statuses.
    """

    MATCH = "match"
    DIFFERENT = "different"


class FailureKind(StrEnum):
    """Category of an individual pair-level difference."""

    NAN_POSITION = "nan_position"
    NAN_POLICY = "nan_policy"
    INFINITY = "infinity"
    FINITE_TOLERANCE = "finite_tolerance"


class StructuralCode(StrEnum):
    """Category of an independent structural difference."""

    DIMENSIONS = "dimensions"
    SHAPE = "shape"
    DATA_DTYPE = "data_dtype"
    COORDINATE_NAMES = "coordinate_names"
    COORDINATE_DIMENSIONS = "coordinate_dimensions"
    COORDINATE_SHAPE = "coordinate_shape"
    COORDINATE_DTYPE = "coordinate_dtype"
    COORDINATE_VALUES = "coordinate_values"
    INDEX_PRESENCE = "index_presence"
    INDEX_VALUES = "index_values"
    GUARDED_ATTRIBUTE = "guarded_attribute"


class LabelKind(StrEnum):
    """Representation tag for a coordinate label snapshot."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BYTES = "bytes"
    DATETIME = "datetime"
    TIMEDELTA = "timedelta"
    POSITION = "position"


#: Presentation priority used to order retained failure examples. A larger
#: value is presented first. This is a presentation order, not a claim about
#: scientific importance.
CATEGORY_PRIORITY: Final[Mapping[FailureKind, int]] = {
    FailureKind.NAN_POSITION: 3,
    FailureKind.NAN_POLICY: 2,
    FailureKind.INFINITY: 1,
    FailureKind.FINITE_TOLERANCE: 0,
}


def element_count(shape: tuple[int, ...]) -> int:
    """Return the number of elements described by ``shape``."""
    return math.prod(shape)


def _require_plain_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_optional_text(value: object, name: str) -> None:
    if value is not None and type(value) is not str:
        raise TypeError(f"{name} must be a str or None, got {type(value).__name__}")


def _require_shape(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
    for axis, extent in enumerate(value):
        _require_plain_int(extent, f"{name}[{axis}]")
    return value


def _require_dimensions(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
    for axis, label in enumerate(value):
        if type(label) is not str:
            raise TypeError(f"{name}[{axis}] must be a str, got {type(label).__name__}")
    return value


def _require_index_tuple(value: object, shape: tuple[int, ...], name: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
    if len(value) != len(shape):
        raise ValueError(f"{name} must have {len(shape)} entries, got {len(value)}")
    for axis, (position, extent) in enumerate(zip(value, shape, strict=True)):
        _require_plain_int(position, f"{name}[{axis}]")
        if position >= extent:
            raise ValueError(f"{name}[{axis}]={position} is out of range for extent {extent}")
    return value


def _require_flat_index(value: object, size: int, name: str) -> int:
    index = _require_plain_int(value, name)
    if index >= size:
        raise ValueError(f"{name}={index} is out of range for {size} elements")
    return index


def _require_labels(value: object, name: str) -> tuple[tuple[str, LabelValue], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
    for entry, item in enumerate(value):
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(f"{name}[{entry}] must be a (name, LabelValue) pair")
        if type(item[0]) is not str:
            raise TypeError(f"{name}[{entry}][0] must be a str")
        if not isinstance(item[1], LabelValue):
            raise TypeError(f"{name}[{entry}][1] must be a LabelValue")
    return value


def _require_optional_float(value: object, name: str) -> None:
    if value is not None and type(value) is not float:
        raise TypeError(f"{name} must be a float or None, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class LabelValue:
    """A detached, bounded snapshot of one coordinate label.

    Attributes
    ----------
    kind:
        How to interpret :attr:`value`.
    value:
        A plain Python value. Temporal labels are stored as integer ticks in
        :attr:`dtype_unit`; byte-string labels are stored base64-encoded so the
        representation preserves bytes exactly.
    dtype_unit:
        Datetime/timedelta unit such as ``"ns"``; ``None`` for other kinds.
    truncated:
        ``True`` when the displayed text was shortened to the display bound.
    original_length:
        Length of the complete display text before truncation.
    """

    kind: LabelKind
    value: bool | int | float | str
    dtype_unit: str | None = None
    truncated: bool = False
    original_length: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LabelKind):
            raise TypeError(f"kind must be a LabelKind, got {type(self.kind).__name__}")
        if type(self.truncated) is not bool:
            raise TypeError(f"truncated must be a bool, got {type(self.truncated).__name__}")
        _text_kinds = (LabelKind.STRING, LabelKind.BYTES)
        _temporal_kinds = (LabelKind.DATETIME, LabelKind.TIMEDELTA)
        if self.kind is LabelKind.BOOLEAN:
            if type(self.value) is not bool:
                raise TypeError(f"value must be a bool for kind {self.kind.value}")
        elif self.kind in (LabelKind.INTEGER, LabelKind.POSITION):
            if type(self.value) is not int:
                raise TypeError(f"value must be an int for kind {self.kind.value}")
        elif self.kind is LabelKind.FLOAT:
            if type(self.value) is not float:
                raise TypeError(f"value must be a float for kind {self.kind.value}")
        elif self.kind in _text_kinds:
            if type(self.value) is not str:
                raise TypeError(f"value must be a str for kind {self.kind.value}")
        elif self.kind in _temporal_kinds:
            if type(self.value) is not int:
                raise TypeError(f"value must be an int tick count for kind {self.kind.value}")
            if type(self.dtype_unit) is not str or not self.dtype_unit:
                raise TypeError(f"dtype_unit is required for kind {self.kind.value}")
        else:  # pragma: no cover - exhaustive over LabelKind
            raise TypeError(f"unsupported label kind {self.kind!r}")

        if self.kind in _text_kinds:
            if self.dtype_unit is not None:
                raise TypeError(f"dtype_unit must be None for kind {self.kind.value}")
            self._validate_text_bounds()
        elif self.kind in _temporal_kinds:
            if self.truncated or self.original_length is not None:
                raise TypeError("temporal labels cannot be truncated")
        else:
            if self.dtype_unit is not None:
                raise TypeError(f"dtype_unit must be None for kind {self.kind.value}")
            if self.truncated or self.original_length is not None:
                raise TypeError("only text labels can be truncated")

    def _validate_text_bounds(self) -> None:
        if not self.truncated:
            if self.original_length is not None:
                raise ValueError("original_length requires truncated=True")
            return
        _require_plain_int(self.original_length, "original_length")
        assert self.original_length is not None
        if type(self.value) is not str:
            raise ValueError("only string labels can be truncated for display")
        if self.original_length <= len(self.value):
            raise ValueError("original_length must exceed the truncated display length")


@dataclass(frozen=True, slots=True)
class StructuralIssue:
    """One established structural difference.

    The message states facts that were established by the comparison. It does
    not speculate about causes.
    """

    code: StructuralCode
    coordinate_name: str | None
    message: str
    first_index: tuple[int, ...] | None = None
    actual: LabelValue | None = None
    reference: LabelValue | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, StructuralCode):
            raise TypeError(f"code must be a StructuralCode, got {type(self.code).__name__}")
        _require_optional_text(self.coordinate_name, "coordinate_name")
        if type(self.message) is not str:
            raise TypeError(f"message must be a str, got {type(self.message).__name__}")
        if not self.message:
            raise ValueError("message must not be empty")
        if self.first_index is not None:
            if type(self.first_index) is not tuple:
                raise TypeError("first_index must be a tuple or None")
            for axis, position in enumerate(self.first_index):
                _require_plain_int(position, f"first_index[{axis}]")
        for name, item in (("actual", self.actual), ("reference", self.reference)):
            if item is not None and not isinstance(item, LabelValue):
                raise TypeError(f"{name} must be a LabelValue or None")


@dataclass(frozen=True, slots=True)
class FailureExample:
    """One bounded, coordinate-aware failure diagnostic.

    Attributes
    ----------
    kind:
        Which pair category this diagnostic belongs to.
    flat_index:
        C-order logical position in the flattened data.
    index:
        Per-axis logical position, matching the declared dimension order.
    labels:
        Coordinate labels for ``index``, one entry per declared dimension.
    actual, reference:
        The two operand values at this position.
    absolute_error, allowed_error:
        ``abs(actual - reference)`` and ``atol + rtol * abs(reference)``. Both are
        ``None`` for nonfinite categories, which are never evaluated against the
        tolerance relation.
    score:
        Dimensionless diagnostic ``absolute_error / allowed_error``. Positive
        infinity when the allowance is zero, and also when the exact ratio
        exceeds binary64 range. ``None`` for nonfinite categories. Diagnostic
        only: it never affects pass or fail.
    score_saturated:
        ``True`` only when :attr:`score` is infinite because a positive
        allowance produced an unrepresentable quotient. Then the exact-real ratio
        is finite, so ranking by computed score is not ranking by exact-real
        ratio. ``False`` for a zero allowance, where the exact-real ratio is
        genuinely infinite and every such example ties either way, and ``False``
        whenever the score is finite or absent.
    """

    kind: FailureKind
    flat_index: int
    index: tuple[int, ...]
    labels: tuple[tuple[str, LabelValue], ...]
    actual: float
    reference: float
    absolute_error: float | None
    allowed_error: float | None
    score: float | None
    score_saturated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FailureKind):
            raise TypeError(f"kind must be a FailureKind, got {type(self.kind).__name__}")
        if type(self.flat_index) is not int:
            raise TypeError(
                f"flat_index must be a Python int, got {type(self.flat_index).__name__}"
            )
        if type(self.index) is not tuple:
            raise TypeError(f"index must be a tuple, got {type(self.index).__name__}")
        for axis, position in enumerate(self.index):
            _require_plain_int(position, f"index[{axis}]")
        _require_labels(self.labels, "labels")
        for name in ("actual", "reference"):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be a float, got {type(value).__name__}")
        _require_optional_float(self.absolute_error, "absolute_error")
        _require_optional_float(self.allowed_error, "allowed_error")
        _require_optional_float(self.score, "score")
        if type(self.score_saturated) is not bool:
            raise TypeError("score_saturated must be a bool")

        if self.kind is FailureKind.FINITE_TOLERANCE:
            if self.absolute_error is None or self.allowed_error is None or self.score is None:
                raise ValueError("finite tolerance examples require error magnitudes")
            if not math.isfinite(self.absolute_error) or self.absolute_error <= 0.0:
                raise ValueError("absolute_error must be positive and finite")
            if not math.isfinite(self.allowed_error) or self.allowed_error < 0.0:
                raise ValueError("allowed_error must be finite and nonnegative")
            if self.absolute_error <= self.allowed_error:
                raise ValueError("a finite tolerance example must exceed its allowance")
            if math.isnan(self.score) or self.score <= 0.0:
                raise ValueError("score must be positive")
        else:
            if self.absolute_error is not None:
                raise ValueError("nonfinite examples must not invent an absolute error")
            if self.allowed_error is not None:
                raise ValueError("nonfinite examples must not invent an allowance")
            if self.score is not None:
                raise ValueError("nonfinite examples must not invent a score")

        # Saturation claims the exact-real ratio is finite but unrepresentable,
        # which is a statement about a representable score, so it cannot be made
        # about a missing or finite one.
        if self.score_saturated and (self.score is None or not math.isinf(self.score)):
            raise ValueError("score_saturated requires an infinite score")


@dataclass(frozen=True, slots=True)
class LocatedMetric:
    """A scalar metric together with the position that attained it."""

    value: float
    flat_index: int
    index: tuple[int, ...]
    labels: tuple[tuple[str, LabelValue], ...]

    def __post_init__(self) -> None:
        if type(self.value) is not float:
            raise TypeError(f"value must be a float, got {type(self.value).__name__}")
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")
        if self.value < 0.0:
            raise ValueError("value must be nonnegative")
        if type(self.flat_index) is not int:
            raise TypeError(
                f"flat_index must be a Python int, got {type(self.flat_index).__name__}"
            )
        if type(self.index) is not tuple:
            raise TypeError(f"index must be a tuple, got {type(self.index).__name__}")
        _require_labels(self.labels, "labels")


@dataclass(frozen=True, slots=True)
class PairCounts:
    """Complete population counts for one completed numerical scan.

    The categories partition the population:

    ``total == finite_pairs + nan_position_mismatches + matching_nan_pairs
    + rejected_nan_pairs + matching_infinity_pairs + infinity_mismatches``
    """

    total: int
    finite_pairs: int
    finite_failures: int
    nan_position_mismatches: int
    matching_nan_pairs: int
    rejected_nan_pairs: int
    matching_infinity_pairs: int
    infinity_mismatches: int

    def __post_init__(self) -> None:
        for name in (
            "total",
            "finite_pairs",
            "finite_failures",
            "nan_position_mismatches",
            "matching_nan_pairs",
            "rejected_nan_pairs",
            "matching_infinity_pairs",
            "infinity_mismatches",
        ):
            _require_plain_int(getattr(self, name), name)
        if self.finite_failures > self.finite_pairs:
            raise ValueError("finite_failures cannot exceed finite_pairs")
        classified = (
            self.finite_pairs
            + self.nan_position_mismatches
            + self.matching_nan_pairs
            + self.rejected_nan_pairs
            + self.matching_infinity_pairs
            + self.infinity_mismatches
        )
        if classified != self.total:
            raise ValueError(
                f"pair categories sum to {classified}, which does not match total {self.total}"
            )

    @property
    def total_failures(self) -> int:
        """Number of pairs that failed for any reason."""
        return (
            self.finite_failures
            + self.nan_position_mismatches
            + self.rejected_nan_pairs
            + self.infinity_mismatches
        )


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Immutable result of :func:`sciresult.compare`.

    The report is fully detached: it holds no array, no view, no
    :class:`xarray.DataArray` and no borrowed input object.

    Notes
    -----
    ``MATCH`` means only that the supported comparison passed. It is not a
    statement about scientific validity.
    """

    status: ComparisonStatus

    actual_name: str | None
    reference_name: str | None

    actual_dimensions: tuple[str, ...]
    reference_dimensions: tuple[str, ...]

    actual_shape: tuple[int, ...]
    reference_shape: tuple[int, ...]

    actual_dtype: str
    reference_dtype: str

    rtol: float
    atol: float
    equal_nan: bool
    max_examples: int

    structural_issue_count: int
    structural_issues: tuple[StructuralIssue, ...]

    numeric_scan_complete: bool
    counts: PairCounts | None

    examples: tuple[FailureExample, ...]
    max_finite_absolute_error: LocatedMetric | None

    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, ComparisonStatus):
            raise TypeError(f"status must be a ComparisonStatus, got {type(self.status).__name__}")
        _require_optional_text(self.actual_name, "actual_name")
        _require_optional_text(self.reference_name, "reference_name")
        _require_dimensions(self.actual_dimensions, "actual_dimensions")
        _require_dimensions(self.reference_dimensions, "reference_dimensions")
        _require_shape(self.actual_shape, "actual_shape")
        _require_shape(self.reference_shape, "reference_shape")
        for name in ("actual_dtype", "reference_dtype"):
            if type(getattr(self, name)) is not str:
                raise TypeError(f"{name} must be a str")
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be a float, got {type(value).__name__}")
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.equal_nan) is not bool:
            raise TypeError(f"equal_nan must be a bool, got {type(self.equal_nan).__name__}")
        _require_plain_int(self.max_examples, "max_examples")
        if self.max_examples > _MAX_EXAMPLES_LIMIT:
            raise ValueError(f"max_examples must be <= {_MAX_EXAMPLES_LIMIT}")
        _require_plain_int(self.structural_issue_count, "structural_issue_count")
        if type(self.structural_issues) is not tuple:
            raise TypeError("structural_issues must be a tuple")
        for position, issue in enumerate(self.structural_issues):
            if not isinstance(issue, StructuralIssue):
                raise TypeError(f"structural_issues[{position}] must be a StructuralIssue")
        if len(self.structural_issues) > _MAX_STRUCTURAL_EXAMPLES:
            raise ValueError(
                f"at most {_MAX_STRUCTURAL_EXAMPLES} structural issues may be retained, "
                f"got {len(self.structural_issues)}"
            )
        if self.structural_issue_count < len(self.structural_issues):
            raise ValueError("structural_issue_count must cover every retained issue")
        if type(self.numeric_scan_complete) is not bool:
            raise TypeError("numeric_scan_complete must be a bool")
        if self.counts is not None and not isinstance(self.counts, PairCounts):
            raise TypeError("counts must be a PairCounts or None")
        if type(self.examples) is not tuple:
            raise TypeError("examples must be a tuple")
        for position, example in enumerate(self.examples):
            if not isinstance(example, FailureExample):
                raise TypeError(f"examples[{position}] must be a FailureExample")
        if type(self.notes) is not tuple:
            raise TypeError("notes must be a tuple")
        for position, note in enumerate(self.notes):
            if type(note) is not str or not note:
                raise TypeError(f"notes[{position}] must be a nonempty str")
        if self.max_finite_absolute_error is not None and not isinstance(
            self.max_finite_absolute_error, LocatedMetric
        ):
            raise TypeError("max_finite_absolute_error must be a LocatedMetric or None")
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        if len(self.examples) > self.max_examples:
            raise ValueError(
                f"{len(self.examples)} examples exceed max_examples={self.max_examples}"
            )

        for position in range(1, len(self.examples)):
            previous = self.examples[position - 1]
            current = self.examples[position]
            if _example_key(previous) <= _example_key(current):
                raise ValueError("examples must be stored in strictly descending key order")

        if self.numeric_scan_complete:
            if self.counts is None:
                raise ValueError("a completed numerical scan requires counts")
            if self.actual_shape != self.reference_shape:
                raise ValueError("a completed numerical scan requires identical shapes")
            if self.counts.total != element_count(self.actual_shape):
                raise ValueError("counts.total must match the number of compared elements")
            for position, example in enumerate(self.examples):
                size = element_count(self.actual_shape)
                _require_flat_index(example.flat_index, size, f"examples[{position}].flat_index")
                _require_index_tuple(
                    example.index, self.actual_shape, f"examples[{position}].index"
                )
            if self.max_finite_absolute_error is not None:
                size = element_count(self.actual_shape)
                _require_flat_index(
                    self.max_finite_absolute_error.flat_index,
                    size,
                    "max_finite_absolute_error.flat_index",
                )
                _require_index_tuple(
                    self.max_finite_absolute_error.index,
                    self.actual_shape,
                    "max_finite_absolute_error.index",
                )
        else:
            # The scan was withheld because the two sides do not label the same
            # quantities. Reporting partial numerical results would imply that
            # corresponding positions were compared, which they were not.
            if self.counts is not None:
                raise ValueError("a withheld numerical scan must not report counts")
            if self.max_finite_absolute_error is not None:
                raise ValueError("a withheld numerical scan must not report extrema")
            if self.examples:
                raise ValueError("a withheld numerical scan must not report examples")
            if self.structural_issue_count == 0:
                raise ValueError("an identical structure requires a completed numerical scan")

        if self.status is ComparisonStatus.MATCH and self.structural_issue_count:
            raise ValueError("MATCH requires a structure without structural issues")

        assert self.counts is not None or self.structural_issue_count
        total_failures = self.counts.total_failures if self.counts is not None else 0
        if self.status is ComparisonStatus.MATCH:
            if self.structural_issue_count:
                raise ValueError("MATCH requires a compatible structure")
            if not self.numeric_scan_complete:
                raise ValueError("MATCH requires a completed numerical scan")
            if total_failures:
                raise ValueError("MATCH requires zero failures")
        elif not self.structural_issue_count and not total_failures:
            raise ValueError("DIFFERENT requires at least one established failure")

    def format(self) -> str:
        """Render the report as plain text."""
        # Imported lazily: sciresult.formatting depends on this module.
        from sciresult.formatting import format_report  # noqa: PLC0415

        return format_report(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, strictly JSON-safe dictionary.

        The result contains only ``None``, ``bool``, ``int``, finite ``float``,
        ``str``, ``list``, and string-keyed ``dict``. Nonstandard values are
        encoded as tagged objects, so ``json.dumps(report.to_dict(),
        allow_nan=False)`` always succeeds.
        """
        # Imported lazily: sciresult.formatting depends on this module.
        from sciresult.formatting import report_to_dict  # noqa: PLC0415

        result: dict[str, Any] = report_to_dict(self)
        return result

    def __bool__(self) -> bool:
        raise TypeError("Inspect report.status explicitly")


def _example_key(example: FailureExample) -> tuple[int, float, int]:
    """Return the deterministic total ordering key of ``example``."""
    score = 0.0 if example.score is None else example.score
    return (CATEGORY_PRIORITY[example.kind], score, -example.flat_index)
