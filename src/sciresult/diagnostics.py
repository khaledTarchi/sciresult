"""Diagnostic selection, ordering, and coordinate-label resolution.

Candidates produced by the scan are already detached floats and integers. This
module turns the retained candidates into public, bounded, coordinate-aware
records, and it resolves labels only for the retained positions and for the
maximum finite absolute error.

Labels are never used as unique identifiers: diagnostics always carry the
integer logical position as well, so duplicate coordinate labels remain
unambiguous.
"""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from sciresult.model import (
    CATEGORY_PRIORITY,
    FailureExample,
    FailureKind,
    LabelKind,
    LabelValue,
    LocatedMetric,
)
from sciresult.numeric import Candidate
from sciresult.structure import snapshot_label, unravel_flat_index
from sciresult.validation import InputView

__all__ = [
    "LabelResolver",
    "build_examples",
    "build_located_metric",
    "candidate_ordering",
    "label_availability_note",
]

Ordering = tuple[int, float, int]


def candidate_ordering(candidate: Candidate) -> Ordering:
    """Return the deterministic total ordering key of ``candidate``.

    Nonfinite categories use score zero. With equal categories and equal
    scores, the lower logical flat index ranks first. This is a presentation
    order, not a claim about scientific importance.
    """
    return (CATEGORY_PRIORITY[candidate.kind], candidate.score, -candidate.flat_index)


class LabelResolver:
    """Resolve coordinates for the retained logical positions.

    A supported one-dimensional dimension coordinate supplies the label for its
    own dimension. Otherwise the integer position is reported and the dimension
    is recorded as having no label. Multidimensional auxiliary coordinates are
    not enumerated into failure examples in version one.
    """

    __slots__ = ("_dimension_coordinates", "_dimensions", "_shape", "_unlabelled")

    def __init__(self, view: InputView) -> None:
        self._dimensions = view.dimensions
        self._shape = view.shape
        self._dimension_coordinates = _dimension_coordinates(view)
        self._unlabelled = tuple(
            dimension
            for dimension in view.dimensions
            if dimension not in self._dimension_coordinates
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical shape the resolver was built for."""
        return self._shape

    @property
    def unlabelled_dimensions(self) -> tuple[str, ...]:
        """Dimensions reported by integer position because no label exists."""
        return self._unlabelled

    def labels(self, flat_index: int) -> tuple[tuple[str, LabelValue], ...]:
        """Return detached label snapshots for one logical position."""
        index = unravel_flat_index(flat_index, self._shape)
        entries: list[tuple[str, LabelValue]] = []
        for axis, dimension in enumerate(self._dimensions):
            coordinate = self._dimension_coordinates.get(dimension)
            if coordinate is None:
                entries.append((dimension, LabelValue(kind=LabelKind.POSITION, value=index[axis])))
                continue
            entries.append((dimension, snapshot_label(coordinate[index[axis]])))
        return tuple(entries)


def _dimension_coordinates(view: InputView) -> dict[str, NDArray[Any]]:
    found: dict[str, NDArray[Any]] = {}
    extents = dict(zip(view.dimensions, view.shape, strict=True))
    for coord in view.coordinates:
        if coord.dimensions != (coord.name,):
            continue
        if coord.shape[0] == extents[coord.name]:
            found[coord.name] = coord.data
    return found


def build_examples(
    candidates: tuple[Candidate, ...],
    resolver: LabelResolver,
) -> tuple[FailureExample, ...]:
    """Turn retained candidates into public, coordinate-aware examples.

    The candidates arrive in final descending order; the order is verified
    rather than assumed.
    """
    examples: list[FailureExample] = []
    previous: Ordering | None = None
    for candidate in candidates:
        ordering = candidate_ordering(candidate)
        if previous is not None and ordering >= previous:
            raise AssertionError("candidates are not in strictly descending order")
        previous = ordering
        finite_failure = candidate.kind is FailureKind.FINITE_TOLERANCE
        examples.append(
            FailureExample(
                kind=candidate.kind,
                flat_index=candidate.flat_index,
                index=unravel_flat_index(candidate.flat_index, resolver.shape),
                labels=resolver.labels(candidate.flat_index),
                actual=candidate.actual,
                reference=candidate.reference,
                absolute_error=candidate.absolute_error,
                allowed_error=candidate.allowed_error,
                score=candidate.score if finite_failure else None,
                score_saturated=candidate.score_saturated,
            )
        )
    return tuple(examples)


def build_located_metric(
    value: float,
    flat_index: int,
    resolver: LabelResolver,
) -> LocatedMetric:
    """Attach labels to the maximum finite absolute error."""
    return LocatedMetric(
        value=value,
        flat_index=flat_index,
        index=unravel_flat_index(flat_index, resolver.shape),
        labels=resolver.labels(flat_index),
    )


def label_availability_note(resolver: LabelResolver) -> str | None:
    """Return a note when some dimensions have no coordinate variable."""
    if not resolver.unlabelled_dimensions:
        return None
    listed = ", ".join(repr(dimension) for dimension in resolver.unlabelled_dimensions)
    return (
        f"dimension(s) {listed} have no coordinate variable; "
        "their positions are reported as integers"
    )
