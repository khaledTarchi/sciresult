"""Bounded-memory numerical scan.

The scan walks both arrays once, in bounded blocks, in C logical order. For each
block it

1. classifies every pair into exactly one category,
2. evaluates the finite binary64 relation separately for the finite subset,
3. updates exact Python-integer counters,
4. tracks the maximum finite absolute discrepancy, and
5. offers a bounded number of candidates to a deterministic heap.

The public semantics are explicit binary64 operations:

.. code-block:: text

    difference_raw = binary64(actual - reference)
    absolute_error = abs(difference_raw)
    relative_part  = binary64(rtol * abs(reference))
    allowed_error  = binary64(atol + relative_part)
    pass           = absolute_error <= allowed_error

Subtraction, absolute value, multiplication, addition, and comparison are
separate NumPy operations. No fused operation, alternate algebraic formula, or
extended precision is used. This is a finite-computation specification, not an
exact-real theorem.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from sciresult.errors import ComparisonArithmeticError
from sciresult.model import CATEGORY_PRIORITY, FailureKind, PairCounts
from sciresult.validation import Options, ValidatedPair

__all__ = [
    "BlockClassification",
    "BlockCounts",
    "BlockPair",
    "Candidate",
    "FiniteErrors",
    "ScanAccumulator",
    "ScanResult",
    "block_size",
    "classify_block",
    "diagnostic_score",
    "finite_errors",
    "iter_pair_blocks",
    "scan_numeric",
]

#: Elements per buffered block. An implementation control, not a scientific
#: parameter, and not part of the public API. Tests may override it.
_NUMERIC_BLOCK_SIZE: Final = 65_536

Ordering = tuple[int, float, int]
HeapItem = tuple[Ordering, "Candidate"]

_EMPTY_POSITIONS: Final[NDArray[np.intp]] = np.empty(0, dtype=np.intp)


def block_size() -> int:
    """Return the block size used by the current scan."""
    return _NUMERIC_BLOCK_SIZE


@dataclass(frozen=True, slots=True)
class BlockPair:
    """One bounded, paired, binary64 block in C logical order."""

    start: int
    actual: NDArray[np.float64]
    reference: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BlockCounts:
    """Category counts for one block, as exact Python integers."""

    finite_pairs: int
    nan_position_mismatches: int
    matching_nan_pairs: int
    rejected_nan_pairs: int
    matching_infinity_pairs: int
    infinity_mismatches: int


@dataclass(frozen=True, slots=True)
class BlockClassification:
    """Mutually exclusive classification of one block.

    ``finite_positions`` and the three mismatch position arrays are ephemeral
    scratch: they never enter a :class:`ScanResult`.
    """

    counts: BlockCounts
    finite_mask: NDArray[np.bool_]
    finite_positions: NDArray[np.intp]
    nan_position_positions: NDArray[np.intp]
    nan_policy_positions: NDArray[np.intp]
    infinity_positions: NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class FiniteErrors:
    """Block-local finite error magnitudes and the failure mask."""

    absolute_error: NDArray[np.float64]
    allowed_error: NDArray[np.float64]
    failure_mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One detached failure candidate awaiting label resolution."""

    kind: FailureKind
    flat_index: int
    actual: float
    reference: float
    absolute_error: float | None
    allowed_error: float | None
    score: float
    score_saturated: bool

    @property
    def ordering(self) -> Ordering:
        """Return the deterministic total ordering key."""
        return (CATEGORY_PRIORITY[self.kind], self.score, -self.flat_index)


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Detached outcome of one completed scan."""

    counts: PairCounts
    candidates: tuple[Candidate, ...]
    max_absolute_value: float | None
    max_absolute_index: int | None


def iter_pair_blocks(
    actual: NDArray[Any],
    reference: NDArray[Any],
    *,
    block_size: int,
) -> Iterator[BlockPair]:
    """Yield paired binary64 blocks in C logical order.

    Iteration uses NumPy's buffered external loop with readonly operand flags,
    safe casting to binary64, an explicit buffer size, and no broadcasting. The
    whole-array ``flatten``, unconditional ``ravel``, whole-array
    ``astype(np.float64)``, full difference arrays, and full error arrays are
    never constructed.

    A scalar DataArray has shape ``()`` and yields exactly one block with
    ``start == 0`` and size one.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    iterator = np.nditer(
        [actual, reference],
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=[["readonly"], ["readonly"]],
        op_dtypes=["float64", "float64"],
        casting="safe",
        buffersize=block_size,
        order="C",
    )
    start = 0
    for actual_block, reference_block in iterator:
        size = int(actual_block.size)
        if size == 0:
            continue
        for offset in range(0, size, block_size):
            stop = min(offset + block_size, size)
            yield BlockPair(
                start=start + offset,
                actual=actual_block[offset:stop],
                reference=reference_block[offset:stop],
            )
        start += size


def classify_block(
    actual: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    equal_nan: bool,
) -> BlockClassification:
    """Classify every pair in a block into exactly one category.

    Categories are assigned in this order: NaN involvement, then remaining
    infinity involvement, then finite pair.
    """
    actual_nan = np.isnan(actual)
    reference_nan = np.isnan(reference)
    either_nan = actual_nan | reference_nan
    both_nan = actual_nan & reference_nan

    nan_position = either_nan & ~both_nan

    # Infinity involvement is only considered for pairs in which neither side is NaN.
    # Without this mask a (nan, inf) pair would be counted both as a NaN-position
    # mismatch and as an infinity mismatch, so the categories would not partition the
    # population.
    infinite = (np.isinf(actual) | np.isinf(reference)) & ~either_nan
    both_infinite = np.isinf(actual) & np.isinf(reference)
    matching_infinity = both_infinite & (actual == reference)
    infinity_mismatch = infinite & ~matching_infinity

    finite_mask = ~(either_nan | infinite)

    if equal_nan:
        matching_nan_pairs = int(np.count_nonzero(both_nan))
        rejected_nan_pairs = 0
        nan_policy_positions = _EMPTY_POSITIONS
    else:
        matching_nan_pairs = 0
        rejected_nan_pairs = int(np.count_nonzero(both_nan))
        nan_policy_positions = np.flatnonzero(both_nan)

    counts = BlockCounts(
        finite_pairs=int(np.count_nonzero(finite_mask)),
        nan_position_mismatches=int(np.count_nonzero(nan_position)),
        matching_nan_pairs=matching_nan_pairs,
        rejected_nan_pairs=rejected_nan_pairs,
        matching_infinity_pairs=int(np.count_nonzero(matching_infinity)),
        infinity_mismatches=int(np.count_nonzero(infinity_mismatch)),
    )
    return BlockClassification(
        counts=counts,
        finite_mask=finite_mask,
        finite_positions=np.flatnonzero(finite_mask),
        nan_position_positions=np.flatnonzero(nan_position),
        nan_policy_positions=nan_policy_positions,
        infinity_positions=np.flatnonzero(infinity_mismatch),
    )


def finite_errors(
    actual: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    rtol: float,
    atol: float,
) -> FiniteErrors:
    """Evaluate the finite relation for finite operands only.

    Raises
    ------
    ComparisonArithmeticError
        If a finite subtraction, multiplication, or addition overflows or
        becomes invalid.
    """
    try:
        difference_raw = actual - reference
        absolute_error = np.abs(difference_raw)
        relative_part = np.multiply(rtol, np.abs(reference))
        allowed_error = np.add(atol, relative_part)
        failure_mask = absolute_error > allowed_error
    except FloatingPointError as exc:
        raise ComparisonArithmeticError(
            "a finite arithmetic step overflowed or became invalid; "
            "the comparison was aborted without producing a report",
            code="finite_arithmetic_overflow",
        ) from exc
    return FiniteErrors(
        absolute_error=absolute_error,
        allowed_error=allowed_error,
        failure_mask=failure_mask,
    )


def diagnostic_score(absolute_error: float, allowed_error: float) -> tuple[float, bool]:
    """Return ``(score, score_saturated)`` for one finite failure.

    ``score = absolute_error / allowed_error`` when the allowance is positive,
    and positive infinity when the allowance is zero and the discrepancy is
    positive. The score is diagnostic only: it never affects pass or fail after
    the finite relation has been evaluated.

    The flag distinguishes the two ways a score can be infinite:

    * A positive allowance whose quotient overflows binary64 gives ``True``. The
      exact-real ratio is finite but unrepresentable, so computed-score ranking
      is not exact-real ranking and the report must say so.
    * A zero allowance gives ``False``. The exact-real ratio is genuinely
      infinite, so every such example ties in exact-real ranking exactly as it
      does here, and the infinity is intentional rather than an artefact of
      overflow.
    """
    if allowed_error > 0.0:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            ratio = float(np.divide(np.float64(absolute_error), np.float64(allowed_error)))
        if math.isinf(ratio):
            return math.inf, True
        return ratio, False
    return math.inf, False


class ScanAccumulator:
    """The only mutable numerical state during a scan.

    It holds Python-integer counters, an optional maximum, and a bounded
    candidate heap. It never retains an input array, a block, an
    :class:`xarray.DataArray`, or a coordinate array.
    """

    __slots__ = (
        "_finite_failures",
        "_finite_pairs",
        "_heap",
        "_infinity_mismatches",
        "_limit",
        "_matching_infinity_pairs",
        "_matching_nan_pairs",
        "_max_absolute_index",
        "_max_absolute_value",
        "_nan_position_mismatches",
        "_processed",
        "_rejected_nan_pairs",
    )

    def __init__(self, *, limit: int) -> None:
        self._limit = limit
        self._processed = 0
        self._finite_pairs = 0
        self._finite_failures = 0
        self._nan_position_mismatches = 0
        self._matching_nan_pairs = 0
        self._rejected_nan_pairs = 0
        self._matching_infinity_pairs = 0
        self._infinity_mismatches = 0
        self._max_absolute_value: float | None = None
        self._max_absolute_index: int | None = None
        self._heap: list[HeapItem] = []

    @property
    def processed(self) -> int:
        """Number of elements processed so far."""
        return self._processed

    def add_size(self, size: int) -> None:
        """Record the number of elements covered by one block."""
        self._processed += size

    def add_counts(self, counts: BlockCounts) -> None:
        """Add one block's category counts."""
        self._finite_pairs += counts.finite_pairs
        self._nan_position_mismatches += counts.nan_position_mismatches
        self._matching_nan_pairs += counts.matching_nan_pairs
        self._rejected_nan_pairs += counts.rejected_nan_pairs
        self._matching_infinity_pairs += counts.matching_infinity_pairs
        self._infinity_mismatches += counts.infinity_mismatches

    def add_finite_failures(self, failures: int) -> None:
        """Add the number of failing finite pairs in one block."""
        self._finite_failures += failures

    def offer_candidate(self, candidate: Candidate) -> None:
        """Offer one candidate to the bounded heap.

        When ``max_examples`` is zero no candidate object is created by the
        caller, and nothing is retained here.
        """
        if self._limit == 0:
            return
        item: HeapItem = (candidate.ordering, candidate)
        if len(self._heap) < self._limit:
            heapq.heappush(self._heap, item)
            return
        if item[0] > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def update_maximum(self, value: float, flat_index: int) -> None:
        """Update the maximum finite absolute discrepancy.

        On a tie the lower logical flat index is retained, which holds because
        blocks are processed in increasing logical order and replacement only
        happens for a strictly larger value.
        """
        if self._max_absolute_value is None or value > self._max_absolute_value:
            self._max_absolute_value = value
            self._max_absolute_index = flat_index

    def finalize(self, total_size: int) -> ScanResult:
        """Validate the accumulated state and return the detached result."""
        if self._processed != total_size:
            raise AssertionError(f"processed {self._processed} elements, expected {total_size}")
        counts = PairCounts(
            total=total_size,
            finite_pairs=self._finite_pairs,
            finite_failures=self._finite_failures,
            nan_position_mismatches=self._nan_position_mismatches,
            matching_nan_pairs=self._matching_nan_pairs,
            rejected_nan_pairs=self._rejected_nan_pairs,
            matching_infinity_pairs=self._matching_infinity_pairs,
            infinity_mismatches=self._infinity_mismatches,
        )
        classified = (
            counts.finite_pairs
            + counts.nan_position_mismatches
            + counts.matching_nan_pairs
            + counts.rejected_nan_pairs
            + counts.matching_infinity_pairs
            + counts.infinity_mismatches
        )
        if classified != total_size:
            raise AssertionError("pair categories do not partition the population")
        if not 0 <= counts.finite_failures <= counts.finite_pairs:
            raise AssertionError("finite failures are not a subset of finite pairs")
        if len(self._heap) > self._limit:
            raise AssertionError("retained candidates exceed the requested limit")

        orderings = [ordering for ordering, _ in self._heap]
        if len(set(orderings)) != len(orderings):
            raise AssertionError("retained candidate orderings are not unique")
        if (self._max_absolute_value is None) != (self._max_absolute_index is None):
            raise AssertionError("maximum value and index are inconsistent")
        if self._max_absolute_value is not None and counts.finite_pairs == 0:
            raise AssertionError("a finite maximum was recorded without finite pairs")

        candidates = tuple(
            candidate for _, candidate in sorted(self._heap, key=lambda item: item[0], reverse=True)
        )
        return ScanResult(
            counts=counts,
            candidates=candidates,
            max_absolute_value=self._max_absolute_value,
            max_absolute_index=self._max_absolute_index,
        )


def scan_numeric(
    pair: ValidatedPair,
    options: Options,
    *,
    block_size: int,
) -> ScanResult:
    """Perform one complete bounded-memory numerical scan.

    Every block is processed; there is no early exit on mismatch. A finite
    arithmetic error aborts the scan and propagates as
    :class:`~sciresult.errors.ComparisonArithmeticError`.
    """
    total_size = math.prod(pair.actual.shape)
    accumulator = ScanAccumulator(limit=options.max_examples)
    with np.errstate(over="raise", invalid="raise", under="ignore", divide="raise"):
        for block in iter_pair_blocks(pair.actual.data, pair.reference.data, block_size=block_size):
            _scan_block(block, options, accumulator)
    return accumulator.finalize(total_size)


def _scan_block(block: BlockPair, options: Options, accumulator: ScanAccumulator) -> None:
    size = int(block.actual.size)
    accumulator.add_size(size)
    classification = classify_block(block.actual, block.reference, equal_nan=options.equal_nan)
    accumulator.add_counts(classification.counts)

    retain_examples = options.max_examples > 0
    if retain_examples:
        for position in classification.nan_position_positions.tolist():
            accumulator.offer_candidate(
                _nonfinite_candidate(FailureKind.NAN_POSITION, block, position)
            )
        for position in classification.nan_policy_positions.tolist():
            accumulator.offer_candidate(
                _nonfinite_candidate(FailureKind.NAN_POLICY, block, position)
            )
        for position in classification.infinity_positions.tolist():
            accumulator.offer_candidate(_nonfinite_candidate(FailureKind.INFINITY, block, position))

    if classification.finite_positions.size == 0:
        return

    finite_actual = block.actual[classification.finite_mask]
    finite_reference = block.reference[classification.finite_mask]
    errors = finite_errors(finite_actual, finite_reference, rtol=options.rtol, atol=options.atol)
    accumulator.add_finite_failures(int(np.count_nonzero(errors.failure_mask)))

    maximum_position = int(classification.finite_positions[int(np.argmax(errors.absolute_error))])
    accumulator.update_maximum(float(errors.absolute_error.max()), block.start + maximum_position)

    if not retain_examples:
        return
    for local in np.flatnonzero(errors.failure_mask).tolist():
        absolute_error = float(errors.absolute_error[local])
        allowed_error = float(errors.allowed_error[local])
        score, saturated = diagnostic_score(absolute_error, allowed_error)
        accumulator.offer_candidate(
            Candidate(
                kind=FailureKind.FINITE_TOLERANCE,
                flat_index=block.start + int(classification.finite_positions[local]),
                actual=float(finite_actual[local]),
                reference=float(finite_reference[local]),
                absolute_error=absolute_error,
                allowed_error=allowed_error,
                score=score,
                score_saturated=saturated,
            )
        )


def _nonfinite_candidate(kind: FailureKind, block: BlockPair, position: int) -> Candidate:
    return Candidate(
        kind=kind,
        flat_index=block.start + position,
        actual=float(block.actual[position]),
        reference=float(block.reference[position]),
        absolute_error=None,
        allowed_error=None,
        score=0.0,
        score_saturated=False,
    )
