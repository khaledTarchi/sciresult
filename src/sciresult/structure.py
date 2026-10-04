"""Deterministic structural comparison of two admitted inputs.

Structure is compared before any numerical evaluation, in a fixed rule order.
A structural difference stops the comparison: no numerical scan runs, and no
numerical counts or extrema are reported.

There is no broadcasting, transposition, alignment, reindexing, sorting,
interpolation, coordinate conversion, calendar conversion, or dtype promotion
for structural acceptance.

Byte order is *not* a structural difference: it does not change represented
values. What is compared is the data representation (kind and width, plus the
unit for datetime64 and timedelta64 coordinates), which is why a binary32
actual and a binary64 reference are structurally different.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from sciresult.model import LabelKind, LabelValue, StructuralCode, StructuralIssue
from sciresult.validation import CoordinateView, InputView, ValidatedPair

__all__ = [
    "StructureResult",
    "compare_coordinate",
    "compare_structure",
    "data_representation",
    "first_coordinate_difference",
    "unravel_flat_index",
]

_MAX_STRUCTURAL_EXAMPLES: Final = 20
_MAX_DISPLAY_CHARACTERS: Final = 256
_COORDINATE_BLOCK_SIZE: Final = 65_536

_LABEL_KINDS: Final[dict[str, LabelKind]] = {
    "b": LabelKind.BOOLEAN,
    "i": LabelKind.INTEGER,
    "u": LabelKind.INTEGER,
    "f": LabelKind.FLOAT,
    "U": LabelKind.STRING,
    "S": LabelKind.BYTES,
    "M": LabelKind.DATETIME,
    "m": LabelKind.TIMEDELTA,
}


@dataclass(frozen=True, slots=True)
class StructureResult:
    """Complete structural outcome.

    Attributes
    ----------
    issue_count:
        Number of established structural issues, including issues that were not
        retained as examples.
    issues:
        At most :data:`_MAX_STRUCTURAL_EXAMPLES` retained issues, in the fixed
        rule order.
    correspondence_breaks:
        Number of established issues that mean position *i* on the actual side
        and position *i* on the reference side denote different quantities.
        Tracked separately from ``issues`` because the retained example list is
        capped, and the numerical decision must not depend on that cap.
    """

    issue_count: int
    issues: tuple[StructuralIssue, ...]
    correspondence_breaks: int = 0

    @property
    def matches(self) -> bool:
        """Whether the two inputs are structurally equivalent."""
        return self.issue_count == 0

    @property
    def correspondence_preserved(self) -> bool:
        """Whether element *i* still denotes the same quantity on both sides.

        A declared width or guarded-attribute difference does not break the
        one-to-one correspondence between positions, so the numerical scan is
        still meaningful and is performed. Any difference in dimensions, shape,
        coordinate names, coordinate values, index presence, or index values
        means the two sides label different quantities, and comparing
        corresponding positions would silently align them.
        """
        return self.correspondence_breaks == 0


_CORRESPONDENCE_CODES: Final[frozenset[StructuralCode]] = frozenset(
    {
        StructuralCode.DIMENSIONS,
        StructuralCode.SHAPE,
        StructuralCode.COORDINATE_NAMES,
        StructuralCode.COORDINATE_VALUES,
        StructuralCode.INDEX_PRESENCE,
        StructuralCode.INDEX_VALUES,
    }
)


class _IssueCollector:
    """Collect issues while preserving the complete total.

    Only the first :data:`_MAX_STRUCTURAL_EXAMPLES` issues are retained, but the
    running total always reflects every established issue.
    """

    __slots__ = ("_breaks", "_count", "_issues")

    def __init__(self) -> None:
        self._count = 0
        self._breaks = 0
        self._issues: list[StructuralIssue] = []

    def add(self, issue: StructuralIssue) -> None:
        self._count += 1
        if issue.code in _CORRESPONDENCE_CODES:
            self._breaks += 1
        if len(self._issues) < _MAX_STRUCTURAL_EXAMPLES:
            self._issues.append(issue)

    def result(self) -> StructureResult:
        return StructureResult(
            issue_count=self._count,
            issues=tuple(self._issues),
            correspondence_breaks=self._breaks,
        )


def compare_structure(pair: ValidatedPair) -> StructureResult:
    """Compare the labeled comparison domains of two admitted inputs."""
    actual = pair.actual
    reference = pair.reference
    collector = _IssueCollector()

    _compare_dimensions(actual, reference, collector)
    _compare_shapes(actual, reference, collector)
    _compare_data_dtypes(actual, reference, collector)
    _compare_coordinate_names(actual, reference, collector)
    _compare_shared_coordinates(actual, reference, collector)
    _compare_indexes(actual, reference, collector)
    _compare_guarded_attributes(actual, reference, collector)

    return collector.result()


def _compare_dimensions(
    actual: InputView, reference: InputView, collector: _IssueCollector
) -> None:
    if actual.dimensions == reference.dimensions:
        return
    collector.add(
        StructuralIssue(
            code=StructuralCode.DIMENSIONS,
            coordinate_name=None,
            message=(
                f"actual dimensions {actual.dimensions} differ from reference dimensions "
                f"{reference.dimensions}; no transposition or renaming was performed"
            ),
        )
    )


def _compare_shapes(actual: InputView, reference: InputView, collector: _IssueCollector) -> None:
    if actual.shape == reference.shape:
        return
    collector.add(
        StructuralIssue(
            code=StructuralCode.SHAPE,
            coordinate_name=None,
            message=(
                f"actual shape {actual.shape} differs from reference shape {reference.shape}; "
                "no broadcasting was performed"
            ),
        )
    )


def _compare_data_dtypes(
    actual: InputView, reference: InputView, collector: _IssueCollector
) -> None:
    if data_representation(actual.dtype) == data_representation(reference.dtype):
        return
    collector.add(
        StructuralIssue(
            code=StructuralCode.DATA_DTYPE,
            coordinate_name=None,
            message=(
                f"actual data dtype {actual.dtype.str} is not the same representation as "
                f"reference data dtype {reference.dtype.str}; no dtype promotion was performed"
            ),
        )
    )


def _compare_coordinate_names(
    actual: InputView, reference: InputView, collector: _IssueCollector
) -> None:
    actual_names = {coord.name for coord in actual.coordinates}
    reference_names = {coord.name for coord in reference.coordinates}
    if actual_names == reference_names:
        return
    only_actual = sorted(actual_names - reference_names)
    only_reference = sorted(reference_names - actual_names)
    collector.add(
        StructuralIssue(
            code=StructuralCode.COORDINATE_NAMES,
            coordinate_name=None,
            message=(
                f"coordinate names differ: "
                f"actual only {only_actual}, reference only {only_reference}"
            ),
        )
    )


def _compare_shared_coordinates(
    actual: InputView, reference: InputView, collector: _IssueCollector
) -> None:
    reference_by_name = {coord.name: coord for coord in reference.coordinates}
    for coord in actual.coordinates:
        other = reference_by_name.get(coord.name)
        if other is None:
            continue
        for issue in compare_coordinate(coord, other):
            collector.add(issue)


def compare_coordinate(
    actual: CoordinateView,
    reference: CoordinateView,
) -> tuple[StructuralIssue, ...]:
    """Compare one coordinate pair, respecting rule dependencies.

    Dimensions are checked first. When they differ, shape, dtype, and values are
    not compared, because a value comparison would require a compatibility that
    has not been established.
    """
    if actual.dimensions != reference.dimensions:
        return (
            StructuralIssue(
                code=StructuralCode.COORDINATE_DIMENSIONS,
                coordinate_name=actual.name,
                message=(
                    f"coordinate {actual.name!r} actual dimensions {actual.dimensions} differ "
                    f"from reference dimensions {reference.dimensions}"
                ),
            ),
        )

    issues: list[StructuralIssue] = []
    if actual.shape != reference.shape:
        issues.append(
            StructuralIssue(
                code=StructuralCode.COORDINATE_SHAPE,
                coordinate_name=actual.name,
                message=(
                    f"coordinate {actual.name!r} actual shape {actual.shape} differs from "
                    f"reference shape {reference.shape}"
                ),
            )
        )

    if data_representation(actual.dtype) != data_representation(reference.dtype):
        issues.append(
            StructuralIssue(
                code=StructuralCode.COORDINATE_DTYPE,
                coordinate_name=actual.name,
                message=(
                    f"coordinate {actual.name!r} actual dtype {actual.dtype.str} is not the same "
                    f"representation as reference dtype {reference.dtype.str}"
                ),
            )
        )
        return tuple(issues)

    if actual.shape != reference.shape:
        return tuple(issues)

    first_difference = first_coordinate_difference(
        actual.data, reference.data, block_size=_COORDINATE_BLOCK_SIZE
    )
    if first_difference is not None:
        issues.append(
            StructuralIssue(
                code=StructuralCode.COORDINATE_VALUES,
                coordinate_name=actual.name,
                message=(
                    f"coordinate {actual.name!r} values differ at logical index "
                    f"{first_difference}; the original values are recorded alongside this message"
                ),
                first_index=first_difference,
                actual=snapshot_label(actual.data[first_difference]),
                reference=snapshot_label(reference.data[first_difference]),
            )
        )
    return tuple(issues)


def data_representation(dtype: np.dtype[Any]) -> tuple[str, int, str | None]:
    """Return the comparable representation of ``dtype``.

    Two arrays are representation-compatible when their kinds and widths agree
    and, for datetime64 and timedelta64, their units agree. Byte order is
    excluded because it does not change represented values.
    """
    unit: str | None = None
    if dtype.kind in ("M", "m"):
        name = dtype.name
        unit = name[name.index("[") + 1 : name.index("]")] if "[" in name else "generic"
    return (dtype.kind, dtype.itemsize, unit)


def first_coordinate_difference(
    actual: NDArray[Any],
    reference: NDArray[Any],
    *,
    block_size: int,
) -> tuple[int, ...] | None:
    """Return the first logical position where two coordinate arrays differ.

    The scan is block-based so that large coordinate arrays do not allocate
    full-size temporary arrays. Positions are C-order logical positions.
    """
    start = 0
    for actual_block, reference_block in _coordinate_blocks(actual, reference, block_size):
        positions = np.flatnonzero(_block_differs(actual_block, reference_block))
        if positions.size:
            return unravel_flat_index(start + int(positions[0]), actual.shape)
        start += int(actual_block.size)
    return None


def _coordinate_blocks(
    actual: NDArray[Any], reference: NDArray[Any], block_size: int
) -> Iterator[tuple[NDArray[Any], NDArray[Any]]]:
    iterator = np.nditer(
        [actual, reference],
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=[["readonly"], ["readonly"]],
        casting="no",
        buffersize=block_size,
        order="C",
    )
    for actual_block, reference_block in iterator:
        size = int(actual_block.size)
        if size == 0:
            continue
        if size <= block_size:
            yield actual_block, reference_block
            continue
        for offset in range(0, size, block_size):
            stop = offset + block_size
            yield actual_block[offset:stop], reference_block[offset:stop]


def _block_differs(actual: NDArray[Any], reference: NDArray[Any]) -> NDArray[Any]:
    """Return the elementwise difference mask for equal-representation blocks.

    Coordinate equality is exact. Corresponding NaNs and NaTs count as equal
    missing labels, and signed zeros compare equal. Coordinates are never
    subtracted, and integer coordinates are never cast to floating point.
    """
    kind = actual.dtype.kind
    if kind == "f":
        both_missing = np.isnan(actual) & np.isnan(reference)
        return cast("NDArray[Any]", (actual != reference) & ~both_missing)
    if kind in ("M", "m"):
        both_missing = np.isnat(actual) & np.isnat(reference)
        return cast("NDArray[Any]", (actual != reference) & ~both_missing)
    return cast("NDArray[Any]", (actual != reference).reshape(actual.shape))


def unravel_flat_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Convert a C-order logical flat position into a per-axis index tuple."""
    index = [0] * len(shape)
    remainder = flat_index
    for axis in range(len(shape) - 1, -1, -1):
        extent = shape[axis]
        index[axis] = remainder % extent
        remainder //= extent
    if remainder:
        raise AssertionError("flat index is out of range for the given shape")
    return tuple(index)


def snapshot_label(value: object) -> LabelValue:
    """Snapshot one admitted coordinate scalar as a detached label.

    The dtype is taken from the scalar itself, so this works for boolean,
    integer, floating, fixed-width text, datetime64, and timedelta64
    coordinates.

    Byte-string values are base64-encoded so the representation preserves bytes
    exactly; no arbitrary byte string is decoded with a lossy character codec.
    Text values longer than the display bound are truncated for display only;
    the comparison itself always uses the original values.
    """
    scalar = value if isinstance(value, np.generic) else np.asarray(value)[()]
    dtype = scalar.dtype
    kind = dtype.kind
    label_kind = _LABEL_KINDS[kind]
    if kind == "b":
        return LabelValue(kind=label_kind, value=bool(scalar))
    if kind in ("i", "u"):
        return LabelValue(kind=label_kind, value=int(cast(Any, scalar)))
    if kind == "f":
        return LabelValue(kind=label_kind, value=float(cast(Any, scalar)))
    if kind == "U":
        return _text_label(label_kind, str(scalar))
    if kind == "S":
        return _text_label(label_kind, base64.b64encode(bytes(scalar)).decode("ascii"))
    return LabelValue(
        kind=label_kind, value=int(scalar.astype(np.int64)), dtype_unit=_temporal_unit(dtype)
    )


def _text_label(kind: LabelKind, text: str) -> LabelValue:
    if len(text) <= _MAX_DISPLAY_CHARACTERS:
        return LabelValue(kind=kind, value=text)
    return LabelValue(
        kind=kind,
        value=text[:_MAX_DISPLAY_CHARACTERS],
        truncated=True,
        original_length=len(text),
    )


def _temporal_unit(dtype: np.dtype[Any]) -> str:
    name = dtype.name
    return name[name.index("[") + 1 : name.index("]")] if "[" in name else "generic"


def _compare_indexes(actual: InputView, reference: InputView, collector: _IssueCollector) -> None:
    actual_dims = {dimension for dimension, _ in actual.indexes}
    reference_dims = {dimension for dimension, _ in reference.indexes}
    if actual_dims != reference_dims:
        collector.add(
            StructuralIssue(
                code=StructuralCode.INDEX_PRESENCE,
                coordinate_name=None,
                message=(
                    f"index presence differs: actual indexed dimensions {sorted(actual_dims)}, "
                    f"reference indexed dimensions {sorted(reference_dims)}"
                ),
            )
        )

    reference_by_dimension = dict(reference.indexes)
    for dimension, actual_index in actual.indexes:
        reference_index = reference_by_dimension.get(dimension)
        if reference_index is None:
            continue
        if not actual_index.equals(reference_index):
            collector.add(
                StructuralIssue(
                    code=StructuralCode.INDEX_VALUES,
                    coordinate_name=dimension,
                    message=(
                        f"index values for dimension {dimension!r} differ according to the "
                        "supported public index equality operation"
                    ),
                )
            )


def _compare_guarded_attributes(
    actual: InputView, reference: InputView, collector: _IssueCollector
) -> None:
    reference_by_name = {coord.name: coord for coord in reference.coordinates}
    _compare_guarded_entry(
        None,
        (
            ("units", actual.units, reference.units),
            ("standard_name", actual.standard_name, reference.standard_name),
        ),
        collector,
    )
    for coord in actual.coordinates:
        other = reference_by_name.get(coord.name)
        if other is None:
            continue
        _compare_guarded_entry(
            coord.name,
            (
                ("units", coord.units, other.units),
                ("standard_name", coord.standard_name, other.standard_name),
            ),
            collector,
        )


def _compare_guarded_entry(
    coordinate_name: str | None,
    attributes: tuple[tuple[str, str | None, str | None], ...],
    collector: _IssueCollector,
) -> None:
    where = "data variable" if coordinate_name is None else f"coordinate {coordinate_name!r}"
    for attribute, actual_value, reference_value in attributes:
        _compare_guarded_value(
            attribute,
            where,
            actual_value,
            reference_value,
            collector,
            coordinate_name=coordinate_name,
        )


def _compare_guarded_value(
    attribute: str,
    where: str,
    actual_value: str | None,
    reference_value: str | None,
    collector: _IssueCollector,
    *,
    coordinate_name: str | None,
) -> None:
    if (actual_value is None) != (reference_value is None):
        collector.add(
            StructuralIssue(
                code=StructuralCode.GUARDED_ATTRIBUTE,
                coordinate_name=coordinate_name,
                message=(
                    f"guarded attribute {attribute!r} presence differs on the {where}: "
                    f"actual {'present' if actual_value is not None else 'absent'}, "
                    f"reference {'present' if reference_value is not None else 'absent'}"
                ),
                actual=(
                    None if actual_value is None else _text_label(LabelKind.STRING, actual_value)
                ),
                reference=(
                    None
                    if reference_value is None
                    else _text_label(LabelKind.STRING, reference_value)
                ),
            )
        )
        return
    if actual_value is None or reference_value is None:
        return
    if actual_value != reference_value:
        collector.add(
            StructuralIssue(
                code=StructuralCode.GUARDED_ATTRIBUTE,
                coordinate_name=coordinate_name,
                message=(
                    f"guarded attribute {attribute!r} differs on the {where}; "
                    "the strings are not interpreted or converted"
                ),
                actual=_text_label(LabelKind.STRING, actual_value),
                reference=_text_label(LabelKind.STRING, reference_value),
            )
        )
