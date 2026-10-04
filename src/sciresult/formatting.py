"""Pure text rendering and strict JSON-safe dictionary encoding.

Both renderers receive a completed :class:`~sciresult.model.ComparisonReport`
and nothing else. They never inspect source arrays, read files, compute labels,
reevaluate tolerances, or change the report status.

``report_to_dict`` returns a newly allocated detached tree containing only
``None``, ``bool``, ``int``, finite ``float``, ``str``, ``list``, and
string-keyed ``dict``. Nonstandard values use tagged objects, so
``json.dumps(report.to_dict(), allow_nan=False)`` always succeeds.
"""

from __future__ import annotations

import math
from typing import Union

from sciresult.model import (
    ComparisonReport,
    ComparisonStatus,
    FailureExample,
    FailureKind,
    LabelKind,
    LabelValue,
    LocatedMetric,
    PairCounts,
    StructuralIssue,
)

__all__ = [
    "JsonValue",
    "encode_scalar",
    "escape_controls",
    "escape_value",
    "format_report",
    "report_to_dict",
    "safe_display",
]

JsonValue = Union[None, bool, int, float, str, list["JsonValue"], dict[str, "JsonValue"]]  # noqa: UP007

_MAX_DISPLAY_CHARACTERS = 256

_CONTROL_TRANSLATION = {
    code: f"\\x{code:02x}" for code in [*range(0x00, 0x20), 0x7F] if code != 0x0A
}
_VALUE_TRANSLATION = {code: f"\\x{code:02x}" for code in [*range(0x00, 0x20), 0x7F]}

_NONFINITE_NOUNS = "No alignment, interpolation, unit conversion, or input mutation was performed."

_KIND_HEADINGS = {
    FailureKind.NAN_POSITION: "NaN-position mismatch (NaN on exactly one side)",
    FailureKind.NAN_POLICY: "NaN-policy rejection (NaN on both sides, equal_nan=False)",
    FailureKind.INFINITY: "infinity mismatch (signed infinities disagree or pair with finite)",
    FailureKind.FINITE_TOLERANCE: "finite tolerance failure (discrepancy exceeds allowance)",
}


def escape_controls(text: str) -> str:
    """Escape control characters other than the line separator.

    Escaping is idempotent, so it is safe to apply to already-escaped text. The
    line separator is preserved so that the rendered report keeps its structure;
    line separators inside user-supplied values are escaped by
    :func:`escape_value` before they reach this pass.
    """
    return text.translate(_CONTROL_TRANSLATION)


def escape_value(text: str) -> str:
    """Escape every control character in a user-supplied value, newlines included."""
    return text.translate(_VALUE_TRANSLATION)


def safe_display(text: str, *, limit: int = _MAX_DISPLAY_CHARACTERS) -> str:
    """Return ``text`` shortened to ``limit`` characters with controls escaped.

    Truncation affects diagnostics only; comparisons always use original values.
    """
    return escape_value(text[:limit])


def format_report(report: ComparisonReport) -> str:
    """Render a completed report as deterministic plain text."""
    lines: list[str] = []
    lines.append("sciresult comparison report")
    lines.append(f"status: {_status_line(report)}")
    lines.append(
        f"actual: name={_optional(report.actual_name)} "
        f"dims={_dimensions(report.actual_dimensions)} shape={report.actual_shape} "
        f"dtype={report.actual_dtype}"
    )
    lines.append(
        f"reference: name={_optional(report.reference_name)} "
        f"dims={_dimensions(report.reference_dimensions)} shape={report.reference_shape} "
        f"dtype={report.reference_dtype}"
    )
    lines.append(
        f"options: rtol={_float(report.rtol)} atol={_float(report.atol)} "
        f"equal_nan={report.equal_nan} max_examples={report.max_examples}"
    )

    lines.append("")
    lines.append(f"structural issues: {report.structural_issue_count}")
    if report.structural_issues:
        for position, issue in enumerate(report.structural_issues, start=1):
            lines.append(f"  structural[{position}] {_format_structural_issue(issue)}")
        if report.structural_issue_count > len(report.structural_issues):
            omitted = report.structural_issue_count - len(report.structural_issues)
            lines.append(f"  ({omitted} further structural issue(s) not retained)")
    else:
        lines.append("  structure is compatible")

    lines.append("")
    lines.append(f"numerical scan: {_scan_line(report)}")
    if report.counts is None:
        lines.append("  counts: not available because no numerical scan was performed")
        lines.append("  max finite absolute error: not available")
    else:
        lines.append(f"  counts: {_format_counts(report.counts)}")
        lines.append(f"  total failures: {report.counts.total_failures}")
        if report.counts.finite_pairs == 0:
            lines.append(
                "  finite population: zero finite pairs; the tolerance relation was not evaluated"
            )
        else:
            lines.append(f"  finite population: {report.counts.finite_pairs} finite pair(s)")
        nonfinite = (
            report.counts.nan_position_mismatches
            + report.counts.matching_nan_pairs
            + report.counts.rejected_nan_pairs
            + report.counts.matching_infinity_pairs
            + report.counts.infinity_mismatches
        )
        if nonfinite:
            lines.append(f"  nonfinite population: {nonfinite} pair(s)")
        lines.append(
            "  max finite absolute error: " + _format_metric(report.max_finite_absolute_error)
        )

    lines.append("")
    lines.append(
        f"examples: {len(report.examples)} shown, cap {report.max_examples}, "
        f"{_finite_caption(report)}"
    )
    if not report.examples:
        lines.append("  no failure examples were retained")
    else:
        lines.append("  failure kinds:")
        for kind in (
            FailureKind.FINITE_TOLERANCE,
            FailureKind.INFINITY,
            FailureKind.NAN_POLICY,
            FailureKind.NAN_POSITION,
        ):
            lines.append(f"    {kind.value}: {_KIND_HEADINGS[kind]}")
    for position, example in enumerate(report.examples, start=1):
        lines.append(f"  example[{position}] {_format_example(example)}")

    if report.notes:
        lines.append("")
        lines.append("notes:")
        for note in report.notes:
            lines.append(f"  - {escape_value(note)}")

    lines.append("")
    lines.append(_NONFINITE_NOUNS)
    return escape_controls("\n".join(lines))


def _status_line(report: ComparisonReport) -> str:
    if report.status is ComparisonStatus.MATCH:
        return "match (the supported comparison passed)"
    if report.structural_issue_count:
        return "different (structural difference; numerical scanning was not performed)"
    return "different (numerical difference)"


def _scan_line(report: ComparisonReport) -> str:
    if report.numeric_scan_complete:
        return "complete (every pair was classified and evaluated)"
    return "not performed (structural difference)"


def _finite_caption(report: ComparisonReport) -> str:
    if report.counts is None:
        return "no numerical population was scanned"
    if report.counts.total_failures == 0:
        return "no pair failed"
    return f"{report.counts.total_failures} pair(s) failed"


def _format_counts(counts: PairCounts) -> str:
    return (
        f"total={counts.total} "
        f"finite_pairs={counts.finite_pairs} "
        f"finite_failures={counts.finite_failures} "
        f"nan_position_mismatches={counts.nan_position_mismatches} "
        f"matching_nan_pairs={counts.matching_nan_pairs} "
        f"rejected_nan_pairs={counts.rejected_nan_pairs} "
        f"matching_infinity_pairs={counts.matching_infinity_pairs} "
        f"infinity_mismatches={counts.infinity_mismatches}"
    )


def _format_structural_issue(issue: StructuralIssue) -> str:
    parts = [f"{issue.code.value}:", escape_value(issue.message)]
    if issue.actual is not None or issue.reference is not None:
        parts.append(
            f"actual={_format_label(issue.actual)} reference={_format_label(issue.reference)}"
        )
    return " ".join(parts)


def _format_metric(metric: LocatedMetric | None) -> str:
    if metric is None:
        return "none (no finite pairs)"
    return (
        f"{_float(metric.value)} at flat index {metric.flat_index} "
        f"index={metric.index} {_format_labels(metric.labels)}"
    )


def _format_example(example: FailureExample) -> str:
    parts = [
        f"{example.kind.value}:",
        f"flat_index={example.flat_index}",
        f"index={example.index}",
    ]
    if example.labels:
        parts.append(_format_labels(example.labels))
    parts.append(f"actual={_float(example.actual)}")
    parts.append(f"reference={_float(example.reference)}")
    if example.kind is FailureKind.FINITE_TOLERANCE:
        parts.append(f"absolute_error={_float(example.absolute_error)}")
        parts.append(f"allowed_error={_float(example.allowed_error)}")
        parts.append(f"score={_float(example.score)}")
        if example.score_saturated:
            parts.append("(score saturated at positive infinity)")
        if example.allowed_error == 0.0:
            parts.append("(allowance is zero, so the score is intentionally infinite)")
    return " ".join(parts)


def _format_labels(labels: tuple[tuple[str, LabelValue], ...]) -> str:
    if not labels:
        return "{}"
    body = ", ".join(f"{name}={_format_label(label)}" for name, label in labels)
    return "{" + body + "}"


def _format_label(label: LabelValue | None) -> str:
    if label is None:
        return "absent"
    value = label.value
    if label.kind is LabelKind.BYTES:
        rendered = f"base64:{safe_display(str(value))}"
    elif label.kind is LabelKind.STRING:
        rendered = safe_display(str(value))
    elif label.kind is LabelKind.FLOAT:
        rendered = _float(float(value))
    elif label.kind in (LabelKind.DATETIME, LabelKind.TIMEDELTA):
        rendered = f"{value} ({label.dtype_unit})"
    else:
        rendered = str(value)
    if label.truncated:
        rendered = f"{rendered} [truncated from {label.original_length} characters]"
    return f'"{rendered}"'


def _optional(name: str | None) -> str:
    return "None" if name is None else escape_value(safe_display(name))


def _dimensions(names: tuple[str, ...]) -> str:
    body = ", ".join(escape_value(safe_display(name)) for name in names)
    return "(" + body + ("," if len(names) == 1 else "") + ")"


def _float(value: float | None) -> str:
    """Render a float with a round-trip-capable, finite representation."""
    if value is None:
        return "n/a"
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "+inf"
    if value == -math.inf:
        return "-inf"
    return repr(value)


def encode_scalar(value: object) -> JsonValue:
    """Encode one value as a strictly JSON-safe representation."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"kind": "nan"}
        if value == math.inf:
            return {"kind": "positive_infinity"}
        if value == -math.inf:
            return {"kind": "negative_infinity"}
        return value
    if isinstance(value, str):
        return value
    raise TypeError(f"cannot encode {type(value).__name__} as a JSON-safe value")


def report_to_dict(report: ComparisonReport) -> dict[str, JsonValue]:
    """Return a detached, strictly JSON-safe dictionary for ``report``."""
    return {
        "status": report.status.value,
        "actual_name": report.actual_name,
        "reference_name": report.reference_name,
        "actual_dimensions": list(report.actual_dimensions),
        "reference_dimensions": list(report.reference_dimensions),
        "actual_shape": list(report.actual_shape),
        "reference_shape": list(report.reference_shape),
        "actual_dtype": report.actual_dtype,
        "reference_dtype": report.reference_dtype,
        "rtol": report.rtol,
        "atol": report.atol,
        "equal_nan": report.equal_nan,
        "max_examples": report.max_examples,
        "structural_issue_count": report.structural_issue_count,
        "structural_issues": [_issue_to_dict(issue) for issue in report.structural_issues],
        "numeric_scan_complete": report.numeric_scan_complete,
        "counts": None if report.counts is None else _counts_to_dict(report.counts),
        "examples": [_example_to_dict(example) for example in report.examples],
        "max_finite_absolute_error": (
            None
            if report.max_finite_absolute_error is None
            else _metric_to_dict(report.max_finite_absolute_error)
        ),
        "notes": list(report.notes),
        "guarantee": _NONFINITE_NOUNS,
    }


def _counts_to_dict(counts: PairCounts) -> dict[str, JsonValue]:
    return {
        "total": counts.total,
        "finite_pairs": counts.finite_pairs,
        "finite_failures": counts.finite_failures,
        "nan_position_mismatches": counts.nan_position_mismatches,
        "matching_nan_pairs": counts.matching_nan_pairs,
        "rejected_nan_pairs": counts.rejected_nan_pairs,
        "matching_infinity_pairs": counts.matching_infinity_pairs,
        "infinity_mismatches": counts.infinity_mismatches,
        "total_failures": counts.total_failures,
    }


def _issue_to_dict(issue: StructuralIssue) -> dict[str, JsonValue]:
    return {
        "code": issue.code.value,
        "coordinate_name": issue.coordinate_name,
        "message": issue.message,
        "first_index": None if issue.first_index is None else list(issue.first_index),
        "actual": None if issue.actual is None else _label_to_dict(issue.actual),
        "reference": None if issue.reference is None else _label_to_dict(issue.reference),
    }


def _example_to_dict(example: FailureExample) -> dict[str, JsonValue]:
    return {
        "kind": example.kind.value,
        "flat_index": example.flat_index,
        "index": list(example.index),
        "labels": {name: _label_to_dict(label) for name, label in example.labels},
        "actual": encode_scalar(example.actual),
        "reference": encode_scalar(example.reference),
        "absolute_error": encode_scalar(example.absolute_error),
        "allowed_error": encode_scalar(example.allowed_error),
        "score": encode_scalar(example.score),
        "score_saturated": example.score_saturated,
    }


def _metric_to_dict(metric: LocatedMetric) -> dict[str, JsonValue]:
    return {
        "value": metric.value,
        "flat_index": metric.flat_index,
        "index": list(metric.index),
        "labels": {name: _label_to_dict(label) for name, label in metric.labels},
    }


def _label_to_dict(label: LabelValue) -> dict[str, JsonValue]:
    if label.kind in (LabelKind.DATETIME, LabelKind.TIMEDELTA):
        return {"kind": label.kind.value, "ticks": label.value, "unit": label.dtype_unit}
    encoded: dict[str, JsonValue] = {"kind": label.kind.value}
    value = encode_scalar(label.value)
    if label.truncated:
        return {
            "kind": "truncated_string",
            "value": value,
            "original_length": label.original_length,
            "label_kind": label.kind.value,
        }
    encoded["value"] = value
    return encoded
