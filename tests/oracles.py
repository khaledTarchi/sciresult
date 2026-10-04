"""Independent scalar reference implementations for :mod:`sciresult`.

This module deliberately contains **no** private production helpers and no
NumPy vectorisation of the comparison. Every rule is re-implemented here from
the documented semantics using plain Python scalars, so agreement between these
functions and the production kernel is meaningful evidence rather than a
tautology.

The oracle models the documented finite binary64 operations, in the documented
order, using Python floats (IEEE-754 binary64 with the same round-to-nearest
semantics NumPy uses). It deliberately does *not* model exact real arithmetic:
it raises :class:`OracleArithmeticError` where the production kernel raises
:class:`sciresult.ComparisonArithmeticError`, and it saturates the diagnostic
score the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "FINITE_TOLERANCE",
    "INFINITY",
    "NAN_POLICY",
    "NAN_POSITION",
    "OracleArithmeticError",
    "OracleCandidate",
    "OracleResult",
    "coordinate_values_equal",
    "first_differing_index",
    "oracle_compare",
    "oracle_pair_counts",
    "unravel",
]

FINITE_TOLERANCE: Final = "finite_tolerance"
NAN_POLICY: Final = "nan_policy"
NAN_POSITION: Final = "nan_position"
INFINITY: Final = "infinity"

#: Mirrors ``sciresult.model.CATEGORY_PRIORITY``; larger ranks first.
_PRIORITY: Final[dict[str, int]] = {
    NAN_POSITION: 3,
    NAN_POLICY: 2,
    INFINITY: 1,
    FINITE_TOLERANCE: 0,
}


class OracleArithmeticError(AssertionError):
    """A finite binary64 operation overflowed or became invalid."""


def _is_nan(value: float) -> bool:
    # Self-inequality is the definition of NaN. Written out longhand on purpose so
    # this oracle shares no helper with the implementation under test.
    return value != value  # noqa: PLR0124


def _subtract(left: float, right: float) -> float:
    result = left - right
    if not math.isfinite(result):
        raise OracleArithmeticError("subtraction of finite operands is not finite")
    return result


def _multiply(left: float, right: float) -> float:
    result = left * right
    if not math.isfinite(result):
        raise OracleArithmeticError("multiplication of finite operands is not finite")
    return result


def _add(left: float, right: float) -> float:
    result = left + right
    if not math.isfinite(result):
        raise OracleArithmeticError("addition of finite operands is not finite")
    return result


def diagnostic_score(absolute_error: float, allowed_error: float) -> tuple[float, bool]:
    """Return ``(score, score_saturated)`` for one finite failure.

    Binary64 division of finite operands that overflows yields positive
    infinity without raising on the supported platforms, so a saturated score is
    detected by inspecting the result.
    """
    if allowed_error > 0.0:
        ratio = absolute_error / allowed_error
        if math.isinf(ratio):
            return math.inf, True
        return ratio, False
    return math.inf, False


@dataclass(frozen=True)
class OracleCandidate:
    """One failure diagnostic as computed by the oracle."""

    kind: str
    flat_index: int
    actual: float
    reference: float
    absolute_error: float | None
    allowed_error: float | None
    score: float
    score_saturated: bool

    @property
    def key(self) -> tuple[int, float, int]:
        return (_PRIORITY[self.kind], self.score, -self.flat_index)


@dataclass(frozen=True)
class OracleResult:
    """Complete oracle outcome for a pair of flat scalar sequences."""

    counts: dict[str, int]
    candidates: tuple[OracleCandidate, ...]
    max_absolute_value: float | None
    max_absolute_index: int | None

    @property
    def total_failures(self) -> int:
        return (
            self.counts["finite_failures"]
            + self.counts["nan_position_mismatches"]
            + self.counts["rejected_nan_pairs"]
            + self.counts["infinity_mismatches"]
        )

    def is_match(self) -> bool:
        return self.total_failures == 0


def oracle_pair_counts(
    actual: list[float],
    reference: list[float],
    *,
    rtol: float,
    atol: float,
    equal_nan: bool,
) -> OracleResult:
    """Classify every pair and count the whole population.

    ``reference`` is always the second input, matching the public API.
    """
    if len(actual) != len(reference):
        raise AssertionError("oracle expects equal-length inputs")

    counts = {
        "total": 0,
        "finite_pairs": 0,
        "finite_failures": 0,
        "nan_position_mismatches": 0,
        "matching_nan_pairs": 0,
        "rejected_nan_pairs": 0,
        "matching_infinity_pairs": 0,
        "infinity_mismatches": 0,
    }
    candidates: list[OracleCandidate] = []
    max_absolute_value: float | None = None
    max_absolute_index: int | None = None

    pairs = zip(actual, reference, strict=True)
    for flat_index, (actual_value, reference_value) in enumerate(pairs):
        counts["total"] += 1
        actual_nan = _is_nan(actual_value)
        reference_nan = _is_nan(reference_value)

        if actual_nan or reference_nan:
            if not (actual_nan and reference_nan):
                counts["nan_position_mismatches"] += 1
                candidates.append(
                    _nonfinite_candidate(NAN_POSITION, flat_index, actual_value, reference_value)
                )
            elif equal_nan:
                counts["matching_nan_pairs"] += 1
            else:
                counts["rejected_nan_pairs"] += 1
                candidates.append(
                    _nonfinite_candidate(NAN_POLICY, flat_index, actual_value, reference_value)
                )
            continue

        actual_infinite = math.isinf(actual_value)
        reference_infinite = math.isinf(reference_value)
        if actual_infinite or reference_infinite:
            if actual_infinite and reference_infinite and actual_value == reference_value:
                counts["matching_infinity_pairs"] += 1
            else:
                counts["infinity_mismatches"] += 1
                candidates.append(
                    _nonfinite_candidate(INFINITY, flat_index, actual_value, reference_value)
                )
            continue

        counts["finite_pairs"] += 1
        difference = _subtract(actual_value, reference_value)
        absolute_error = abs(difference)
        relative_part = _multiply(rtol, abs(reference_value))
        allowed_error = _add(atol, relative_part)
        if absolute_error <= allowed_error:
            pass
        else:
            counts["finite_failures"] += 1
            score, saturated = diagnostic_score(absolute_error, allowed_error)
            candidates.append(
                OracleCandidate(
                    kind=FINITE_TOLERANCE,
                    flat_index=flat_index,
                    actual=actual_value,
                    reference=reference_value,
                    absolute_error=absolute_error,
                    allowed_error=allowed_error,
                    score=score,
                    score_saturated=saturated,
                )
            )
        if max_absolute_value is None or absolute_error > max_absolute_value:
            max_absolute_value = absolute_error
            max_absolute_index = flat_index

    ordered = tuple(sorted(candidates, key=lambda item: item.key, reverse=True))
    return OracleResult(
        counts=counts,
        candidates=ordered,
        max_absolute_value=max_absolute_value,
        max_absolute_index=max_absolute_index,
    )


def oracle_compare(
    actual: list[float],
    reference: list[float],
    *,
    rtol: float,
    atol: float,
    equal_nan: bool,
) -> tuple[OracleResult, tuple[OracleCandidate, ...]]:
    """Return ``(full_result, top_examples)`` using a complete descending sort."""
    result = oracle_pair_counts(actual, reference, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return result, result.candidates


def _nonfinite_candidate(
    kind: str, flat_index: int, actual: float, reference: float
) -> OracleCandidate:
    return OracleCandidate(
        kind=kind,
        flat_index=flat_index,
        actual=actual,
        reference=reference,
        absolute_error=None,
        allowed_error=None,
        score=0.0,
        score_saturated=False,
    )


def unravel(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Convert a C-order flat position into a per-axis index tuple."""
    index = [0] * len(shape)
    remainder = flat_index
    for axis in range(len(shape) - 1, -1, -1):
        extent = shape[axis]
        index[axis] = remainder % extent
        remainder //= extent
    if remainder:
        raise AssertionError("flat index out of range for shape")
    return tuple(index)


def coordinate_values_equal(left: Any, right: Any) -> bool:
    """Return whether two admitted coordinate arrays are exactly equal.

    Booleans, integers, and fixed-width strings compare exactly. Floating
    values compare exactly with corresponding NaNs treated as equal. Temporal
    values compare exactly with corresponding NaTs treated as equal. Signed
    zeros compare equal. Datetime and timedelta units are assumed to match
    already, because representation is checked separately.
    """
    if tuple(left.shape) != tuple(right.shape):
        return False
    kind = left.dtype.kind
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    for flat_index in range(flat_left.size):
        left_item = flat_left[flat_index]
        right_item = flat_right[flat_index]
        if kind == "f":
            if math.isnan(float(left_item)) and math.isnan(float(right_item)):
                continue
            if float(left_item) != float(right_item):
                return False
        elif kind in ("M", "m"):
            if np_isnat(left_item) and np_isnat(right_item):
                continue
            if left_item != right_item:
                return False
        elif left_item != right_item:
            return False
    return True


def np_isnat(value: Any) -> bool:
    """Return whether a datetime64/timedelta64 scalar is NaT."""
    return int(value) == -(2**63)


def first_differing_index(left: Any, right: Any) -> tuple[int, ...] | None:
    """Return the first logical C-order position where two arrays differ."""
    if tuple(left.shape) != tuple(right.shape):
        raise AssertionError("shapes must match")
    if coordinate_values_equal(left, right):
        return None
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    for flat_index in range(flat_left.size):
        left_item = flat_left[flat_index]
        right_item = flat_right[flat_index]
        if (
            left.dtype.kind == "f"
            and math.isnan(float(left_item))
            and math.isnan(float(right_item))
        ):
            continue
        if left.dtype.kind in ("M", "m") and np_isnat(left_item) and np_isnat(right_item):
            continue
        if left_item != right_item:
            return unravel(flat_index, tuple(left.shape))
    raise AssertionError("arrays differ but no differing element was found")
