"""Option normalization and supported-input admission.

This module answers two questions and nothing else:

1. Are the supplied options normalizable into a supported configuration?
2. Is each input a supported, eager, NumPy-backed
   :class:`xarray.DataArray`?

It never coerces an unsupported object into a supported one, never calls
``compute``/``load``, and never mutates an input. It reads the public
``DataArray.data`` property, which may itself materialize some file-backed
wrappers before admission can inspect the resulting array; see
``docs/limitations.md``.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from sciresult.errors import (
    EmptyComparisonError,
    InvalidComparisonArgument,
    UnsupportedInputError,
)

__all__ = [
    "GUARDED_ATTRIBUTES",
    "CoordinateView",
    "InputView",
    "Options",
    "ValidatedPair",
    "admit_data_array",
    "admit_pair",
    "normalize_options",
]

#: Attributes that are compared between actual and reference. Every other
#: attribute, encoding, storage path, and chunking hint is ignored.
GUARDED_ATTRIBUTES: Final[tuple[str, ...]] = ("units", "standard_name")

_MAX_EXAMPLES: Final = 100

#: Exact array types admitted as data or coordinate storage. Anything else,
#: including masked arrays, subclasses, Dask arrays, sparse arrays, GPU arrays,
#: and unit wrappers, is rejected.
_ADMITTED_ARRAY_TYPES: Final[tuple[type, ...]] = (np.ndarray, np.memmap)

#: Coordinate dtype kinds admitted. Data is restricted further.
_COORDINATE_KINDS: Final[frozenset[str]] = frozenset("biufUSMm")

#: Closed set of Xarray index families admitted in version one.
_ADMITTED_INDEX_TYPE: Final[type] = xr.indexes.PandasIndex

#: Closed set of pandas index classes admitted inside an admitted
#: :class:`xarray.indexes.PandasIndex`. Anything else, including
#: ``CategoricalIndex``, ``IntervalIndex``, ``MultiIndex``, and CFTime-backed
#: indexes, is rejected.
_ADMITTED_PANDAS_INDEX_TYPES: Final[tuple[type, ...]] = (
    pd.Index,
    pd.RangeIndex,
    pd.DatetimeIndex,
    pd.TimedeltaIndex,
)


@dataclass(frozen=True, slots=True)
class Options:
    """Validated, normalized comparison options.

    Attributes
    ----------
    rtol:
        Relative tolerance, a finite nonnegative binary64 float.
    atol:
        Absolute tolerance, a finite nonnegative binary64 float.
    equal_nan:
        Whether a NaN on both sides is accepted as matching.
    max_examples:
        Number of retained failure examples, between 0 and 100 inclusive.
    """

    rtol: float
    atol: float
    equal_nan: bool
    max_examples: int


@dataclass(frozen=True, slots=True)
class CoordinateView:
    """Borrowed, call-local view of one admitted coordinate.

    This record may reference caller data. It never escapes into a public
    report.
    """

    name: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    data: NDArray[Any]
    units: str | None
    standard_name: str | None


@dataclass(frozen=True, slots=True)
class InputView:
    """Borrowed, call-local view of one admitted input."""

    name: str | None
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    data: NDArray[Any]
    coordinates: tuple[CoordinateView, ...]
    indexes: tuple[tuple[str, xr.Index], ...]
    units: str | None
    standard_name: str | None


@dataclass(frozen=True, slots=True)
class ValidatedPair:
    """Two admitted inputs that are ready to be compared."""

    actual: InputView
    reference: InputView


def normalize_options(
    *,
    rtol: float,
    atol: float,
    equal_nan: bool,
    max_examples: int,
) -> Options:
    """Validate and normalize the public options.

    Raises
    ------
    InvalidComparisonArgument
        If a tolerance is not a supported numeric scalar, is negative, is not
        finite, lies outside binary64 range, or would silently normalize to
        zero; if ``equal_nan`` is not a real Boolean; or if ``max_examples`` is
        not an integer in ``0..100``.
    """
    normalized_rtol = _normalize_tolerance(rtol, "rtol")
    normalized_atol = _normalize_tolerance(atol, "atol")
    if type(equal_nan) is not bool:
        raise InvalidComparisonArgument(
            f"equal_nan must be a bool, got {type(equal_nan).__name__}",
            code="equal_nan_not_bool",
        )
    normalized_examples = _normalize_max_examples(max_examples)
    return Options(
        rtol=normalized_rtol,
        atol=normalized_atol,
        equal_nan=equal_nan,
        max_examples=normalized_examples,
    )


def _normalize_tolerance(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise InvalidComparisonArgument(
            f"{name} must be a real number, not a Boolean",
            code=f"{name}_boolean",
        )
    if isinstance(value, (int, float, np.integer, np.floating)):
        pass
    elif isinstance(value, (np.ndarray, np.generic)):
        raise InvalidComparisonArgument(
            f"{name} must be a real number scalar, got {type(value).__name__}",
            code=f"{name}_not_scalar",
        )
    else:
        raise InvalidComparisonArgument(
            f"{name} must be a real number, got {type(value).__name__}",
            code=f"{name}_not_real",
        )

    was_nonzero = bool(value != 0)
    try:
        converted = float(value)
    except OverflowError as exc:
        raise InvalidComparisonArgument(
            f"{name} is outside the binary64 range",
            code=f"{name}_out_of_range",
        ) from exc
    except (TypeError, ValueError) as exc:  # pragma: no cover - guarded above
        raise InvalidComparisonArgument(
            f"{name} must be a real number, got {type(value).__name__}",
            code=f"{name}_not_real",
        ) from exc

    if was_nonzero and converted == 0.0:
        raise InvalidComparisonArgument(
            f"{name} underflowed to zero during normalization",
            code=f"{name}_underflow",
        )
    if not math.isfinite(converted):
        if math.isnan(converted):
            raise InvalidComparisonArgument(f"{name} must not be NaN", code=f"{name}_nan")
        raise InvalidComparisonArgument(f"{name} must not be infinite", code=f"{name}_infinite")
    if converted < 0.0:
        raise InvalidComparisonArgument(
            f"{name} must be nonnegative, got a negative value", code=f"{name}_negative"
        )
    if converted == 0.0:
        return 0.0
    return converted


def _normalize_max_examples(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise InvalidComparisonArgument(
            "max_examples must be an integer, not a Boolean",
            code="max_examples_boolean",
        )
    if not isinstance(value, (int, np.integer)):
        raise InvalidComparisonArgument(
            f"max_examples must be an integer, got {type(value).__name__}",
            code="max_examples_not_integer",
        )
    converted = int(value)
    if not 0 <= converted <= _MAX_EXAMPLES:
        raise InvalidComparisonArgument(
            f"max_examples must be between 0 and {_MAX_EXAMPLES} inclusive, got {converted}",
            code="max_examples_out_of_range",
        )
    return converted


def admit_pair(actual: object, reference: object) -> ValidatedPair:
    """Admit both inputs, actual first and reference second."""
    return ValidatedPair(
        actual=admit_data_array(actual, role="actual"),
        reference=admit_data_array(reference, role="reference"),
    )


def admit_data_array(array: object, *, role: str) -> InputView:
    """Admit one :class:`xarray.DataArray` into a borrowed input view.

    Raises
    ------
    UnsupportedInputError
        If the object is not a ``DataArray``, or uses an unsupported name,
        dimension label, array backend, data dtype, coordinate dtype, index
        family, or guarded attribute value.
    EmptyComparisonError
        If the array contains zero elements.
    """
    if not isinstance(array, xr.DataArray):
        raise UnsupportedInputError(
            f"{role} must be an xarray.DataArray, got {type(array).__name__}",
            code=f"{role}_not_a_dataarray",
        )

    name = array.name
    if name is not None and type(name) is not str:
        raise UnsupportedInputError(
            f"{role} name must be a string or None, got {type(name).__name__}",
            code=f"{role}_name_not_text",
        )

    dimensions = array.dims
    for axis, dimension in enumerate(dimensions):
        if type(dimension) is not str:
            raise UnsupportedInputError(
                f"{role} dimension {axis} must be a string, got {type(dimension).__name__}",
                code=f"{role}_dimension_not_text",
            )

    shape = tuple(int(extent) for extent in array.shape)
    if math.prod(shape) == 0:
        raise EmptyComparisonError(
            f"{role} contains zero elements; empty inputs cannot be compared",
            code=f"{role}_empty",
        )

    data = _admit_numpy(array.data, role=role, target="data", code_prefix=f"{role}_data")
    _require_real_float_dtype(data.dtype, role=role, target="data")

    coordinates = tuple(
        _admit_coordinate(coord, role=role) for _, coord in _sorted_coords(array, role=role)
    )
    indexes = _admit_indexes(array, role=role)
    units, standard_name = _guarded_values(array.attrs, role=role, target="data variable")

    return InputView(
        name=name,
        dimensions=_text_dimensions(dimensions, role=role),
        shape=shape,
        dtype=data.dtype,
        data=data,
        coordinates=coordinates,
        indexes=indexes,
        units=units,
        standard_name=standard_name,
    )


def _admit_numpy(
    value: object,
    *,
    role: str,
    target: str,
    code_prefix: str,
) -> NDArray[Any]:
    if not isinstance(value, np.ndarray):
        raise UnsupportedInputError(
            f"{role} {target} must be a NumPy ndarray, got {type(value).__name__}; "
            "lazy, sparse, device, and masked backends are not supported",
            code=f"{code_prefix}_unsupported_backend",
        )
    if type(value) not in _ADMITTED_ARRAY_TYPES:
        raise UnsupportedInputError(
            f"{role} {target} must be a plain ndarray or np.memmap, got {type(value).__name__}",
            code=f"{code_prefix}_unsupported_subclass",
        )
    return value


def _require_real_float_dtype(dtype: np.dtype[Any], *, role: str, target: str) -> None:
    if dtype.kind != "f" or dtype.itemsize not in (4, 8):
        raise UnsupportedInputError(
            f"{role} {target} dtype {dtype.str} is not supported; "
            "only real binary32 and binary64 data are supported",
            code=f"{role}_{target}_dtype_unsupported",
        )


def _sorted_coords(array: xr.DataArray, *, role: str) -> list[tuple[str, xr.DataArray]]:
    """Return ``(name, coord)`` pairs sorted by name, rejecting non-text names."""
    entries: list[tuple[str, xr.DataArray]] = []
    for name, coord in array.coords.items():
        if type(name) is not str:
            raise UnsupportedInputError(
                f"{role} coordinate name must be a string, got {type(name).__name__}",
                code=f"{role}_coordinate_name_not_text",
            )
        entries.append((name, coord))
    entries.sort(key=lambda entry: entry[0])
    return entries


def _text_dimensions(dimensions: tuple[Hashable, ...], *, role: str) -> tuple[str, ...]:
    result: list[str] = []
    for axis, dimension in enumerate(dimensions):
        if type(dimension) is not str:
            raise UnsupportedInputError(
                f"{role} dimension {axis} must be a string, got {type(dimension).__name__}",
                code=f"{role}_dimension_not_text",
            )
        result.append(dimension)
    return tuple(result)


def _admit_coordinate(coord: xr.DataArray, *, role: str) -> CoordinateView:
    coord_name = coord.name
    if type(coord_name) is not str:
        raise UnsupportedInputError(
            f"{role} coordinate name must be a string, got {type(coord_name).__name__}",
            code=f"{role}_coordinate_name_not_text",
        )
    for axis, dimension in enumerate(coord.dims):
        if type(dimension) is not str:
            raise UnsupportedInputError(
                f"{role} coordinate {coord_name!r} dimension {axis} must be a string",
                code=f"{role}_coordinate_dimension_not_text",
            )
    data = _admit_numpy(
        coord.data, role=role, target=f"coordinate {coord_name!r}", code_prefix=f"{role}_coord"
    )
    if data.dtype.kind not in _COORDINATE_KINDS:
        raise UnsupportedInputError(
            f"{role} coordinate {coord_name!r} dtype {data.dtype.str} is not supported",
            code=f"{role}_coordinate_dtype_unsupported",
        )
    if data.dtype.kind == "f" and data.dtype.itemsize not in (4, 8):
        raise UnsupportedInputError(
            f"{role} coordinate {coord_name!r} dtype {data.dtype.str} is not supported; "
            "only binary32 and binary64 floating coordinates are supported",
            code=f"{role}_coordinate_dtype_unsupported",
        )
    if data.dtype.kind in ("M", "m") and _temporal_unit(data.dtype) is None:
        raise UnsupportedInputError(
            f"{role} coordinate {coord_name!r} dtype {data.dtype.str} has no concrete unit",
            code=f"{role}_coordinate_dtype_unsupported",
        )
    units, standard_name = _guarded_values(
        coord.attrs, role=role, target=f"coordinate {coord_name!r}"
    )
    return CoordinateView(
        name=coord_name,
        dimensions=_text_dimensions(coord.dims, role=role),
        shape=tuple(int(extent) for extent in coord.shape),
        dtype=data.dtype,
        data=data,
        units=units,
        standard_name=standard_name,
    )


def _temporal_unit(dtype: np.dtype[Any]) -> str | None:
    name = dtype.name
    if "[" not in name:
        return None
    return name[name.index("[") + 1 : name.index("]")]


def _admit_indexes(array: xr.DataArray, *, role: str) -> tuple[tuple[str, xr.Index], ...]:
    admitted: list[tuple[str, xr.Index]] = []
    for dimension in sorted(array.xindexes):
        index = array.xindexes[dimension]
        if type(index) is not _ADMITTED_INDEX_TYPE:
            raise UnsupportedInputError(
                f"{role} index for dimension {dimension!r} is a {type(index).__name__}; "
                f"only {_ADMITTED_INDEX_TYPE.__name__} is supported",
                code=f"{role}_index_family_unsupported",
            )
        represented = index.to_pandas_index()
        if type(represented) not in _ADMITTED_PANDAS_INDEX_TYPES:
            raise UnsupportedInputError(
                f"{role} index for dimension {dimension!r} represents a "
                f"{type(represented).__name__}; that index family is not supported",
                code=f"{role}_index_backend_unsupported",
            )
        admitted.append((dimension, index))
    return tuple(admitted)


def _guarded_values(
    attrs: Mapping[Any, Any], *, role: str, target: str
) -> tuple[str | None, str | None]:
    units = _guarded_value(attrs, "units", role=role, target=target)
    standard_name = _guarded_value(attrs, "standard_name", role=role, target=target)
    return units, standard_name


def _guarded_value(
    attrs: Mapping[Any, Any], attribute: str, *, role: str, target: str
) -> str | None:
    if attribute not in attrs:
        return None
    value = attrs[attribute]
    if type(value) is not str:
        raise UnsupportedInputError(
            f"{role} {target} attribute {attribute!r} must be a string, got {type(value).__name__}",
            code=f"{role}_guarded_attribute_not_text",
        )
    return value
