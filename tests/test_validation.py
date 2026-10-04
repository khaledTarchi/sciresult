"""Option normalization and supported-input admission tests."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import sciresult
from sciresult import compare
from sciresult.errors import (
    EmptyComparisonError,
    InvalidComparisonArgument,
    UnsupportedInputError,
)
from sciresult.validation import normalize_options


def _float_array(values: list[float]) -> xr.DataArray:
    return xr.DataArray(
        np.array(values, dtype=np.float64), dims=("x",), coords={"x": range(len(values))}
    )


class MaskedDataArray(xr.DataArray):
    """A DataArray whose public ``data`` property yields a masked array.

    Xarray itself drops a mask when a masked array is passed to the constructor,
    so this subclass is how a masked backend is presented to admission.
    """

    __slots__ = ()

    @property
    def data(self):  # type: ignore[override]
        return np.ma.masked_array(np.asarray(self.variable.data))


class TestToleranceNormalization:
    def test_valid_python_float_tolerance(self) -> None:
        options = normalize_options(rtol=1e-6, atol=2.5, equal_nan=False, max_examples=5)
        assert options.rtol == 1e-6
        assert options.atol == 2.5
        assert options.equal_nan is False
        assert options.max_examples == 5

    def test_valid_numpy_floating_tolerance(self) -> None:
        options = normalize_options(
            rtol=np.float32(1e-6), atol=np.float64(1e-12), equal_nan=True, max_examples=0
        )
        assert options.rtol == pytest.approx(1e-6)
        assert options.atol == 1e-12
        assert type(options.rtol) is float
        assert type(options.atol) is float

    def test_valid_numpy_integer_tolerance(self) -> None:
        options = normalize_options(
            rtol=np.int64(3), atol=np.uint8(0), equal_nan=False, max_examples=1
        )
        assert options.rtol == 3.0
        assert options.atol == 0.0
        assert type(options.atol) is float

    def test_valid_integer_zero_tolerance(self) -> None:
        options = normalize_options(rtol=0, atol=0, equal_nan=False, max_examples=5)
        assert options.rtol == 0.0
        assert options.atol == 0.0

    def test_negative_signed_zero_is_normalized_to_positive_zero(self) -> None:
        options = normalize_options(
            rtol=-0.0, atol=np.float64(-0.0), equal_nan=False, max_examples=5
        )
        assert str(options.rtol) == "0.0"
        assert str(options.atol) == "0.0"

    @pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
    def test_boolean_tolerance_is_rejected(self, value: object) -> None:
        with pytest.raises(InvalidComparisonArgument) as info:
            normalize_options(rtol=value, atol=0.0, equal_nan=False, max_examples=5)
        assert info.value.code in {"rtol_boolean", "atol_boolean"}

    @pytest.mark.parametrize("value", [-1.0, -1e-300, np.float64(-2.0), np.int64(-3)])
    def test_negative_tolerance_is_rejected(self, value: object) -> None:
        with pytest.raises(InvalidComparisonArgument) as info:
            normalize_options(rtol=value, atol=0.0, equal_nan=False, max_examples=5)
        assert info.value.code == "rtol_negative"

    @pytest.mark.parametrize(
        "value", [float("nan"), np.float64("nan"), float("inf"), -float("inf")]
    )
    def test_non_finite_tolerance_is_rejected(self, value: object) -> None:
        with pytest.raises(InvalidComparisonArgument) as info:
            normalize_options(rtol=value, atol=0.0, equal_nan=False, max_examples=5)
        assert info.value.code in {"rtol_nan", "rtol_infinite"}

    def test_out_of_range_tolerance_is_rejected(self) -> None:
        with pytest.raises(InvalidComparisonArgument) as info:
            normalize_options(rtol=10**400, atol=0.0, equal_nan=False, max_examples=5)
        assert info.value.code == "rtol_out_of_range"

    @pytest.mark.skipif(
        np.finfo(np.longdouble).maxexp <= np.finfo(np.float64).maxexp,
        reason=(
            "requires an accepted real scalar wider than binary64; on platforms where "
            "np.longdouble is binary64 (including 64-bit Windows) no accepted input can "
            "underflow when narrowed to binary64"
        ),
    )
    def test_underflowing_tolerance_is_rejected(self) -> None:
        tiny = np.longdouble("1e-400")
        assert tiny != 0
        assert float(tiny) == 0.0
        with pytest.raises(InvalidComparisonArgument) as info:
            normalize_options(rtol=tiny, atol=0.0, equal_nan=False, max_examples=5)
        assert info.value.code == "rtol_underflow"

    @pytest.mark.skipif(
        np.finfo(np.longdouble).maxexp > np.finfo(np.float64).maxexp,
        reason="platform has a real scalar wider than binary64",
    )
    def test_exact_zero_wider_scalar_is_never_an_underflow(self) -> None:
        options = normalize_options(
            rtol=np.longdouble(0.0), atol=np.longdouble(0.0), equal_nan=False, max_examples=5
        )
        assert options.rtol == 0.0
        assert options.atol == 0.0

    def test_smallest_representable_binary64_tolerance_is_accepted(self) -> None:
        options = normalize_options(rtol=5e-324, atol=5e-324, equal_nan=False, max_examples=5)
        assert options.rtol == 5e-324
        assert options.rtol != 0.0

    @pytest.mark.parametrize(
        "value",
        [np.array(1e-6), np.array([1e-6]), "1e-6", None, 1 + 2j, [1e-6], {"rtol": 1e-6}],
    )
    def test_unsupported_tolerance_types_are_rejected(self, value: object) -> None:
        with pytest.raises(InvalidComparisonArgument) as info:
            normalize_options(rtol=value, atol=0.0, equal_nan=False, max_examples=5)
        assert info.value.code in {"rtol_not_real", "rtol_not_scalar"}

    def test_atol_is_validated_independently(self) -> None:
        with pytest.raises(InvalidComparisonArgument) as info:
            normalize_options(rtol=0.0, atol=-1.0, equal_nan=False, max_examples=5)
        assert info.value.code == "atol_negative"


class TestEqualNanNormalization:
    @pytest.mark.parametrize("value", [1, 0, "true", None, np.bool_(True), 1.0])
    def test_equal_nan_must_be_a_real_boolean(self, value: object) -> None:
        with pytest.raises(InvalidComparisonArgument) as info:
            normalize_options(rtol=0.0, atol=0.0, equal_nan=value, max_examples=5)
        assert info.value.code == "equal_nan_not_bool"

    @pytest.mark.parametrize("value", [False, True])
    def test_equal_nan_accepts_real_booleans(self, value: bool) -> None:
        options = normalize_options(rtol=0.0, atol=0.0, equal_nan=value, max_examples=5)
        assert options.equal_nan is value


class TestMaxExamplesNormalization:
    def test_zero_example_cap_is_accepted(self) -> None:
        assert (
            normalize_options(rtol=0.0, atol=0.0, equal_nan=False, max_examples=0).max_examples == 0
        )

    def test_maximum_example_cap_is_accepted(self) -> None:
        assert (
            normalize_options(rtol=0.0, atol=0.0, equal_nan=False, max_examples=100).max_examples
            == 100
        )

    @pytest.mark.parametrize("value", [-1, 101, 1000])
    def test_out_of_range_example_cap_is_rejected(self, value: int) -> None:
        with pytest.raises(InvalidComparisonArgument) as info:
            normalize_options(rtol=0.0, atol=0.0, equal_nan=False, max_examples=value)
        assert info.value.code == "max_examples_out_of_range"

    @pytest.mark.parametrize("value", [True, False, np.bool_(False)])
    def test_boolean_example_cap_is_rejected(self, value: object) -> None:
        with pytest.raises(InvalidComparisonArgument) as info:
            normalize_options(rtol=0.0, atol=0.0, equal_nan=False, max_examples=value)
        assert info.value.code == "max_examples_boolean"

    @pytest.mark.parametrize("value", [1.0, "3", None, np.float64(3.0)])
    def test_non_integer_example_cap_is_rejected(self, value: object) -> None:
        with pytest.raises(InvalidComparisonArgument) as info:
            normalize_options(rtol=0.0, atol=0.0, equal_nan=False, max_examples=value)
        assert info.value.code == "max_examples_not_integer"

    def test_numpy_integer_example_cap_is_accepted(self) -> None:
        assert (
            normalize_options(
                rtol=0.0, atol=0.0, equal_nan=False, max_examples=np.int64(4)
            ).max_examples
            == 4
        )


class TestInputAdmission:
    def test_wrong_object_type_is_rejected(self) -> None:
        array = _float_array([1.0, 2.0])
        with pytest.raises(UnsupportedInputError) as info:
            compare(np.array([1.0, 2.0]), array, rtol=0.0, atol=0.0)
        assert info.value.code == "actual_not_a_dataarray"

    def test_reference_object_type_is_reported_by_role(self) -> None:
        array = _float_array([1.0, 2.0])
        with pytest.raises(UnsupportedInputError) as info:
            compare(array, [1.0, 2.0], rtol=0.0, atol=0.0)
        assert info.value.code == "reference_not_a_dataarray"

    def test_empty_dataarray_is_rejected(self) -> None:
        empty = xr.DataArray(np.array([], dtype=np.float64), dims=("x",))
        full = _float_array([1.0])
        with pytest.raises(EmptyComparisonError) as info:
            compare(empty, full, rtol=0.0, atol=0.0)
        assert info.value.code == "actual_empty"

    def test_empty_reference_is_rejected(self) -> None:
        empty = xr.DataArray(np.array([], dtype=np.float64), dims=("x",))
        full = _float_array([1.0])
        with pytest.raises(EmptyComparisonError) as info:
            compare(full, empty, rtol=0.0, atol=0.0)
        assert info.value.code == "reference_empty"

    def test_float64_data_is_supported(self) -> None:
        array = _float_array([1.0, 2.0])
        assert (
            compare(array, array.copy(), rtol=0.0, atol=0.0).status
            is sciresult.ComparisonStatus.MATCH
        )

    def test_float32_data_is_supported(self) -> None:
        array = xr.DataArray(np.array([1.0, 2.0], dtype=np.float32), dims=("x",))
        assert (
            compare(array, array.copy(), rtol=0.0, atol=0.0).status
            is sciresult.ComparisonStatus.MATCH
        )

    def test_scalar_dataarray_is_supported(self) -> None:
        array = xr.DataArray(np.array(3.5, dtype=np.float64))
        assert (
            compare(array, array.copy(), rtol=0.0, atol=0.0).status
            is sciresult.ComparisonStatus.MATCH
        )

    @pytest.mark.parametrize("byte_order", [">", "<"])
    def test_nonnative_byte_order_is_supported(self, byte_order: str) -> None:
        values = np.array([1.0, 2.0], dtype=np.dtype(f"{byte_order}f8"))
        array = xr.DataArray(values, dims=("x",))
        report = compare(array, array.copy(), rtol=0.0, atol=0.0)
        assert report.actual_dtype == values.dtype.str

    def test_byte_order_difference_is_not_structural(self) -> None:
        little = xr.DataArray(np.array([1.0, 2.0], dtype="<f8"), dims=("x",))
        big = xr.DataArray(np.array([1.0, 2.0], dtype=">f8"), dims=("x",))
        report = compare(little, big, rtol=0.0, atol=0.0)
        assert report.status is sciresult.ComparisonStatus.MATCH
        assert report.actual_dtype == "<f8"
        assert report.reference_dtype == ">f8"

    def test_memmap_data_is_supported(self, memmap_factory) -> None:  # type: ignore[no-untyped-def]
        mapped = memmap_factory(np.array([1.0, 2.0, 3.0]))
        array = xr.DataArray(mapped, dims=("x",))
        assert (
            compare(array, array.copy(), rtol=0.0, atol=0.0).status
            is sciresult.ComparisonStatus.MATCH
        )

    def test_memmap_big_endian_data_is_supported(self, memmap_factory) -> None:  # type: ignore[no-untyped-def]
        mapped = memmap_factory(np.array([1.0, 2.0], dtype=">f8"), dtype=">f8")
        array = xr.DataArray(mapped, dims=("x",))
        assert (
            compare(array, array.copy(), rtol=0.0, atol=0.0).status
            is sciresult.ComparisonStatus.MATCH
        )

    @pytest.mark.parametrize(
        ("dtype", "code"),
        [
            (np.int64, "actual_data_dtype_unsupported"),
            (np.uint8, "actual_data_dtype_unsupported"),
            (np.complex128, "actual_data_dtype_unsupported"),
            (np.bool_, "actual_data_dtype_unsupported"),
            (np.float16, "actual_data_dtype_unsupported"),
        ],
    )
    def test_unsupported_data_dtypes_are_rejected(self, dtype: type, code: str) -> None:
        values = np.array([1, 0], dtype=dtype)
        array = xr.DataArray(values, dims=("x",))
        good = xr.DataArray(np.array([1.0, 0.0]), dims=("x",))
        with pytest.raises(UnsupportedInputError) as info:
            compare(array, good, rtol=0.0, atol=0.0)
        assert info.value.code == code

    def test_object_data_is_rejected(self) -> None:
        values = np.array([{"a": 1}, {"b": 2}], dtype=object)
        array = xr.DataArray(values, dims=("x",))
        with pytest.raises(UnsupportedInputError) as info:
            compare(array, array, rtol=0.0, atol=0.0)
        assert info.value.code == "actual_data_dtype_unsupported"

    def test_masked_data_is_rejected(self) -> None:
        masked = MaskedDataArray(xr.DataArray(np.array([1.0, 2.0]), dims=("x",)))
        good = xr.DataArray(np.array([1.0, 2.0]), dims=("x",))
        with pytest.raises(UnsupportedInputError) as info:
            compare(masked, good, rtol=0.0, atol=0.0)
        assert info.value.code == "actual_data_unsupported_subclass"

    def test_unsupported_ndarray_subclass_is_rejected(self) -> None:
        class RowMajor(np.ndarray):
            pass

        values = np.zeros((2, 2)).view(RowMajor)
        array = xr.DataArray(values, dims=("x", "y"))
        with pytest.raises(UnsupportedInputError) as info:
            compare(array, array, rtol=0.0, atol=0.0)
        assert info.value.code == "actual_data_unsupported_subclass"

    def test_lazy_duck_array_is_not_explicitly_computed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dask_array = pytest.importorskip("dask.array")
        lazy = dask_array.zeros((3, 3), chunks=2)
        array = xr.DataArray(lazy, dims=("x", "y"))

        def explode(*args: object, **kwargs: object) -> None:
            raise AssertionError("lazy data was explicitly computed")

        monkeypatch.setattr(type(lazy), "compute", explode)
        monkeypatch.setattr(dask_array.Array, "compute", explode, raising=False)

        good = xr.DataArray(np.zeros((3, 3)), dims=("x", "y"))
        with pytest.raises(UnsupportedInputError) as info:
            compare(array, good, rtol=0.0, atol=0.0)
        assert info.value.code == "actual_data_unsupported_backend"

    @pytest.mark.parametrize(
        "values",
        [
            np.array([object(), object()], dtype=object),
            np.array([{"a"}, {"b"}], dtype=object),
        ],
    )
    def test_unsupported_coordinate_type_is_rejected(self, values: np.ndarray) -> None:
        array = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": values})
        with pytest.raises(UnsupportedInputError) as info:
            compare(array, array, rtol=0.0, atol=0.0)
        assert info.value.code == "actual_coordinate_dtype_unsupported"

    def test_float16_coordinate_is_rejected(self) -> None:
        # A float16 *dimension* coordinate cannot be built: pandas silently casts it to
        # float64. A two-dimensional auxiliary coordinate has no pandas index, so the
        # float16 dtype survives Xarray construction and reaches admission unchanged.
        auxiliary = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        array = xr.DataArray(
            np.zeros((2, 2)), dims=("x", "y"), coords={"aux": (("x", "y"), auxiliary)}
        )
        assert array.coords["aux"].dtype == np.float16
        with pytest.raises(UnsupportedInputError) as info:
            compare(array, array, rtol=0.0, atol=0.0)
        assert info.value.code == "actual_coordinate_dtype_unsupported"

    @pytest.mark.parametrize("attribute", ["units", "standard_name"])
    @pytest.mark.parametrize("value", [1.0, None, ["K"], {"a": 1}])
    def test_invalid_guarded_attribute_is_rejected(self, attribute: str, value: object) -> None:
        array = xr.DataArray(np.zeros(2), dims=("x",), attrs={attribute: value})
        with pytest.raises(UnsupportedInputError) as info:
            compare(array, array, rtol=0.0, atol=0.0)
        assert info.value.code == "actual_guarded_attribute_not_text"

    @pytest.mark.parametrize("attribute", ["units", "standard_name"])
    def test_string_guarded_attributes_are_supported(self, attribute: str) -> None:
        array = xr.DataArray(np.zeros(2), dims=("x",), attrs={attribute: "K"})
        other = xr.DataArray(np.zeros(2), dims=("x",), attrs={attribute: "K"})
        assert compare(array, other, rtol=0.0, atol=0.0).status is sciresult.ComparisonStatus.MATCH

    def test_custom_index_is_rejected(self) -> None:
        class CustomIndex(xr.indexes.Index):
            def __init__(self, variables: object) -> None:
                self.variables = dict(variables)  # type: ignore[call-overload]

            @classmethod
            def from_variables(cls, variables, *, options=None):  # type: ignore[no-untyped-def]
                return cls({v.dims[0]: v for v in variables})

            def to_pandas_index(self):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def isel(self, indexers):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def equals(self, other):  # type: ignore[no-untyped-def]
                return isinstance(other, CustomIndex)

            def create_variables(self, variables=None):  # type: ignore[no-untyped-def]
                if variables is not None:
                    return dict(variables)
                return dict(self.variables)

            def to_index_variables(self, variables=None):  # type: ignore[no-untyped-def]
                return self.create_variables(variables)

            def should_add_coord_to_array(self, name, var, dims) -> bool:  # type: ignore[no-untyped-def]
                return True

            def copy(self, deep: bool = True):  # type: ignore[no-untyped-def]
                return self

            def rename(self, name_dict, dims_dict):  # type: ignore[no-untyped-def]
                return self

        # Xarray has no supported public constructor that takes a foreign Index, but
        # xr.Coordinates(coords, indexes=...) is public and accepts any xr.indexes.Index.
        custom = CustomIndex.from_variables([xr.Variable("x", np.array([1.0, 2.0]))])
        coordinates = xr.Coordinates(custom.create_variables(), indexes={"x": custom})
        array = xr.DataArray(np.zeros(2), dims=("x",), coords=coordinates)
        assert type(array.xindexes["x"]).__name__ == "CustomIndex"

        good = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": [1.0, 2.0]})
        with pytest.raises(UnsupportedInputError) as info:
            compare(array, good, rtol=0.0, atol=0.0)
        assert info.value.code == "actual_index_family_unsupported"

    def test_non_text_coordinate_name_is_rejected(self) -> None:
        array = xr.DataArray(np.zeros(2), dims=("x",), coords={1: ("x", [1.0, 2.0])})
        assert 1 in array.coords
        with pytest.raises(UnsupportedInputError) as info:
            compare(array, array, rtol=0.0, atol=0.0)
        assert info.value.code == "actual_coordinate_name_not_text"

    def test_non_text_dimension_name_is_rejected(self) -> None:
        # Xarray requires every dimension name to be a string, so the guard is only
        # reachable through a DataArray that reports non-text dimensions.
        class NonTextDimensionArray(xr.DataArray):
            __slots__ = ()

            @property
            def dims(self) -> tuple[object, ...]:
                return (1,)

        array = NonTextDimensionArray(np.zeros(2), dims=("x",))
        assert array.dims == (1,)
        with pytest.raises(UnsupportedInputError) as info:
            compare(array, array, rtol=0.0, atol=0.0)
        assert info.value.code == "actual_dimension_not_text"

    def test_fortran_contiguous_data_is_supported(self) -> None:
        values = np.asfortranarray(np.arange(6.0).reshape(2, 3))
        array = xr.DataArray(values, dims=("x", "y"))
        assert (
            compare(array, array.copy(), rtol=0.0, atol=0.0).status
            is sciresult.ComparisonStatus.MATCH
        )

    def test_noncontiguous_view_is_supported(self) -> None:
        base = np.arange(10.0)
        view = base[::-1]
        assert not view.flags["C_CONTIGUOUS"]
        array = xr.DataArray(view, dims=("x",))
        assert (
            compare(array, array.copy(), rtol=0.0, atol=0.0).status
            is sciresult.ComparisonStatus.MATCH
        )
