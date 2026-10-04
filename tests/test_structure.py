"""Structural comparison semantics.

Every test here targets a rule that is decided *before* any numeric work:
dimension order, shape, coordinate sets, coordinate values, index families,
and guarded attributes. Numeric agreement is covered in ``test_numeric.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import sciresult
from sciresult.model import ComparisonStatus, StructuralCode

ZERO = {"rtol": 0.0, "atol": 0.0}


def _codes(report: sciresult.ComparisonReport) -> list[StructuralCode]:
    return [issue.code for issue in report.structural_issues]


def _codes_for(report: sciresult.ComparisonReport, name: str | None) -> list[StructuralCode]:
    return [issue.code for issue in report.structural_issues if issue.coordinate_name == name]


class TestDimensionOrder:
    def test_dimension_order_mismatch_is_structural(self) -> None:
        actual = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
        reference = xr.DataArray(np.zeros((2, 3)), dims=("y", "x"))
        report = sciresult.compare(actual, reference, **ZERO)
        assert report.status is ComparisonStatus.DIFFERENT
        assert StructuralCode.DIMENSIONS in _codes(report)

    def test_transposed_shape_is_a_shape_difference_not_a_dimension_difference(self) -> None:
        # Same dimension names, different order AND different shape. Both facts must be
        # reported, independently, because the library never transposes to "fix" either.
        actual = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
        reference = xr.DataArray(np.zeros((3, 2)), dims=("y", "x"))
        report = sciresult.compare(actual, reference, **ZERO)
        assert StructuralCode.DIMENSIONS in _codes(report)
        assert StructuralCode.SHAPE in _codes(report)

    def test_identical_dimension_order_matches(self) -> None:
        values = np.arange(6.0).reshape(2, 3)
        actual = xr.DataArray(values, dims=("x", "y"))
        reference = xr.DataArray(values.copy(), dims=("x", "y"))
        assert sciresult.compare(actual, reference, **ZERO).status is ComparisonStatus.MATCH

    def test_no_coordinate_name_is_normalized_before_comparison(self) -> None:
        # "time" and "t" denote the same physical axis, but the library compares names
        # literally: renaming a coordinate is a structural difference.
        actual = xr.DataArray(np.zeros(2), dims=("time",), coords={"time": [0.0, 1.0]})
        reference = xr.DataArray(np.zeros(2), dims=("t",), coords={"t": [0.0, 1.0]})
        report = sciresult.compare(actual, reference, **ZERO)
        assert report.status is ComparisonStatus.DIFFERENT
        assert StructuralCode.DIMENSIONS in _codes(report)
        assert StructuralCode.COORDINATE_NAMES in _codes(report)


class TestShape:
    def test_shape_mismatch_is_structural(self) -> None:
        actual = xr.DataArray(np.zeros(6), dims=("x",))
        reference = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
        report = sciresult.compare(actual, reference, **ZERO)
        assert StructuralCode.SHAPE in _codes(report)

    def test_data_is_never_broadcast_or_reshaped(self) -> None:
        actual = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"))
        reference = xr.DataArray(np.zeros(3), dims=("x",))
        report = sciresult.compare(actual, reference, **ZERO)
        assert report.status is ComparisonStatus.DIFFERENT
        assert StructuralCode.SHAPE in _codes(report)


class TestDataDtype:
    def test_float32_and_float64_are_structurally_distinct(self) -> None:
        actual = xr.DataArray(np.zeros(2, dtype=np.float32), dims=("x",))
        reference = xr.DataArray(np.zeros(2, dtype=np.float64), dims=("x",))
        report = sciresult.compare(actual, reference, **ZERO)
        assert StructuralCode.DATA_DTYPE in _codes(report)
        assert report.actual_dtype == "<f4"
        assert report.reference_dtype == "<f8"

    def test_equal_width_float_dtype_is_not_structural(self) -> None:
        actual = xr.DataArray(np.zeros(2, dtype=np.float64), dims=("x",))
        reference = xr.DataArray(np.zeros(2, dtype=np.longdouble).astype(np.float64), dims=("x",))
        assert sciresult.compare(actual, reference, **ZERO).status is ComparisonStatus.MATCH

    def test_promotion_is_never_performed(self) -> None:
        # The values are numerically identical; only the declared widths differ.
        actual = xr.DataArray(np.array([1.0, 2.0], dtype=np.float32), dims=("x",))
        reference = xr.DataArray(np.array([1.0, 2.0], dtype=np.float64), dims=("x",))
        report = sciresult.compare(actual, reference, rtol=1e-12, atol=1e-12)
        assert report.status is ComparisonStatus.DIFFERENT
        assert StructuralCode.DATA_DTYPE in _codes(report)
        # Positions still correspond one to one, so the scan runs and reports zero
        # pair-level failures: the only established difference is the declared width.
        assert report.counts is not None
        assert report.counts.total_failures == 0
        assert report.counts.finite_pairs == 2


class TestCoordinateNames:
    def test_missing_coordinate_on_actual_side_is_reported(self) -> None:
        actual = xr.DataArray(np.zeros(2), dims=("x",))
        reference = xr.DataArray(np.zeros(2), dims=("x",), coords={"lat": ("x", [1.0, 2.0])})
        report = sciresult.compare(actual, reference, **ZERO)
        issue = next(
            i for i in report.structural_issues if i.code is StructuralCode.COORDINATE_NAMES
        )
        assert "lat" in issue.message

    def test_extra_coordinate_on_actual_side_is_reported(self) -> None:
        actual = xr.DataArray(np.zeros(2), dims=("x",), coords={"lat": ("x", [1.0, 2.0])})
        reference = xr.DataArray(np.zeros(2), dims=("x",))
        report = sciresult.compare(actual, reference, **ZERO)
        issue = next(
            i for i in report.structural_issues if i.code is StructuralCode.COORDINATE_NAMES
        )
        assert "lat" in issue.message
        assert report.status is ComparisonStatus.DIFFERENT

    def test_identical_coordinate_sets_match(self) -> None:
        values = np.zeros(2)
        coords = {"lat": ("x", [1.0, 2.0]), "lon": ("x", [3.0, 4.0])}
        actual = xr.DataArray(values, dims=("x",), coords=coords)
        reference = xr.DataArray(values.copy(), dims=("x",), coords=dict(coords))
        assert sciresult.compare(actual, reference, **ZERO).status is ComparisonStatus.MATCH


class TestCoordinateValues:
    def test_shifted_coordinate_is_a_structural_difference(self) -> None:
        actual = xr.DataArray(np.zeros(3), dims=("t",), coords={"t": [0.0, 1.0, 2.0]})
        reference = xr.DataArray(np.zeros(3), dims=("t",), coords={"t": [1.0, 2.0, 3.0]})
        report = sciresult.compare(actual, reference, **ZERO)
        assert report.status is ComparisonStatus.DIFFERENT
        assert StructuralCode.COORDINATE_VALUES in _codes_for(report, "t")
        # The labelling differs, so comparing position 0 to position 0 would silently
        # align two different instants. The scan is withheld and the report says so.
        assert report.counts is None
        assert report.numeric_scan_complete is False

    def test_coordinate_tolerance_is_not_applied(self) -> None:
        # rtol/atol govern data pairs only. Coordinate equality is exact.
        actual = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": [0.0, 1.0]})
        reference = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": [0.0, 1.0 + 1e-12]})
        report = sciresult.compare(actual, reference, rtol=1e-6, atol=1e-6)
        assert StructuralCode.COORDINATE_VALUES in _codes_for(report, "x")

    def test_nan_coordinates_at_the_same_position_are_equal(self) -> None:
        actual = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": [np.nan, 1.0]})
        reference = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": [np.nan, 1.0]})
        report = sciresult.compare(actual, reference, **ZERO)
        assert StructuralCode.COORDINATE_VALUES not in _codes_for(report, "x")
        assert report.status is ComparisonStatus.MATCH

    def test_nan_against_a_number_is_a_coordinate_difference(self) -> None:
        actual = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": [np.nan, 1.0]})
        reference = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": [0.0, 1.0]})
        report = sciresult.compare(actual, reference, **ZERO)
        assert StructuralCode.COORDINATE_VALUES in _codes_for(report, "x")

    def test_signed_zero_coordinates_are_equal(self) -> None:
        actual = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": [-0.0, 1.0]})
        reference = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": [0.0, 1.0]})
        assert sciresult.compare(actual, reference, **ZERO).status is ComparisonStatus.MATCH

    def test_integer_coordinates_are_not_rounded(self) -> None:
        actual = xr.DataArray(
            np.zeros(2), dims=("x",), coords={"x": np.array([1, 2], dtype=np.int64)}
        )
        reference = xr.DataArray(
            np.zeros(2), dims=("x",), coords={"x": np.array([1, 3], dtype=np.int64)}
        )
        report = sciresult.compare(actual, reference, rtol=1e-9, atol=1e-9)
        assert StructuralCode.COORDINATE_VALUES in _codes_for(report, "x")

    def test_datetime_coordinates_compare_as_temporal_values(self) -> None:
        actual = xr.DataArray(
            np.zeros(2), dims=("t",), coords={"t": np.array(["2020-01-01", "2020-01-02"], "M8[ns]")}
        )
        reference = xr.DataArray(
            np.zeros(2), dims=("t",), coords={"t": np.array(["2020-01-01", "2020-01-03"], "M8[ns]")}
        )
        report = sciresult.compare(actual, reference, **ZERO)
        assert StructuralCode.COORDINATE_VALUES in _codes_for(report, "t")

    def test_matching_nan_t_coordinates_are_equal(self) -> None:
        actual = xr.DataArray(
            np.zeros(2), dims=("t",), coords={"t": np.array(["NaT", "2020-01-02"], "M8[ns]")}
        )
        reference = xr.DataArray(
            np.zeros(2), dims=("t",), coords={"t": np.array(["NaT", "2020-01-02"], "M8[ns]")}
        )
        assert sciresult.compare(actual, reference, **ZERO).status is ComparisonStatus.MATCH

    def test_auxiliary_coordinate_values_are_compared(self) -> None:
        actual = xr.DataArray(
            np.zeros((2, 2)),
            dims=("x", "y"),
            coords={"aux": (("x", "y"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        )
        reference = xr.DataArray(
            np.zeros((2, 2)),
            dims=("x", "y"),
            coords={"aux": (("x", "y"), np.array([[1.0, 2.0], [3.0, 9.0]]))},
        )
        report = sciresult.compare(actual, reference, **ZERO)
        assert StructuralCode.COORDINATE_VALUES in _codes_for(report, "aux")


class TestIndexes:
    def test_index_presence_difference_is_structural(self) -> None:
        actual = xr.DataArray(np.zeros(2), dims=("x",))
        reference = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": [1.0, 2.0]})
        report = sciresult.compare(actual, reference, **ZERO)
        assert StructuralCode.INDEX_PRESENCE in _codes(report)

    def test_dimension_without_index_matches_dimension_with_matching_index(self) -> None:
        # "x" on one side and [0, 1] on the other are not the same labelling, so this
        # must differ. The library never invents an index for the bare side.
        actual = xr.DataArray(np.zeros(2), dims=("x",))
        reference = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": [0, 1]})
        report = sciresult.compare(actual, reference, **ZERO)
        assert StructuralCode.INDEX_PRESENCE in _codes(report)

    def test_matching_indexes_match(self) -> None:
        actual = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": [0, 1]})
        reference = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": [0, 1]})
        assert sciresult.compare(actual, reference, **ZERO).status is ComparisonStatus.MATCH


class TestGuardedAttributes:
    @pytest.mark.parametrize("attribute", ["units", "standard_name"])
    def test_presence_difference_is_structural(self, attribute: str) -> None:
        actual = xr.DataArray(np.zeros(2), dims=("x",), attrs={attribute: "K"})
        reference = xr.DataArray(np.zeros(2), dims=("x",))
        report = sciresult.compare(actual, reference, **ZERO)
        assert StructuralCode.GUARDED_ATTRIBUTE in _codes(report)

    @pytest.mark.parametrize("attribute", ["units", "standard_name"])
    def test_value_difference_is_structural_and_not_converted(self, attribute: str) -> None:
        actual = xr.DataArray(np.zeros(2), dims=("x",), attrs={attribute: "K"})
        reference = xr.DataArray(np.zeros(2), dims=("x",), attrs={attribute: "kelvin"})
        report = sciresult.compare(actual, reference, **ZERO)
        assert StructuralCode.GUARDED_ATTRIBUTE in _codes(report)
        issue = next(
            i for i in report.structural_issues if i.code is StructuralCode.GUARDED_ATTRIBUTE
        )
        assert issue.actual is not None
        assert issue.reference is not None
        assert issue.actual.value == "K"
        assert issue.reference.value == "kelvin"

    def test_data_variable_attributes_are_checked(self) -> None:
        actual = xr.DataArray(np.zeros(2), dims=("x",), attrs={"units": "K"})
        reference = xr.DataArray(np.zeros(2), dims=("x",), attrs={"units": "K"})
        assert sciresult.compare(actual, reference, **ZERO).status is ComparisonStatus.MATCH

    def test_coordinate_attributes_are_checked(self) -> None:
        actual = xr.DataArray(
            np.zeros(2), dims=("x",), coords={"x": ("x", [0.0, 1.0], {"units": "m"})}
        )
        reference = xr.DataArray(
            np.zeros(2), dims=("x",), coords={"x": ("x", [0.0, 1.0], {"units": "s"})}
        )
        report = sciresult.compare(actual, reference, **ZERO)
        assert StructuralCode.GUARDED_ATTRIBUTE in _codes_for(report, "x")

    def test_unguarded_attributes_are_ignored(self) -> None:
        # Only `units` and `standard_name` are guarded. Everything else is out of scope
        # and must not influence the outcome.
        actual = xr.DataArray(np.zeros(2), dims=("x",), attrs={"history": "a"})
        reference = xr.DataArray(np.zeros(2), dims=("x",), attrs={"history": "b"})
        assert sciresult.compare(actual, reference, **ZERO).status is ComparisonStatus.MATCH


class TestStructuralOutcomeRules:
    def test_structure_only_difference_reports_no_pair_failures(self) -> None:
        actual = xr.DataArray(np.zeros(3), dims=("t",), coords={"t": [0.0, 1.0, 2.0]})
        reference = xr.DataArray(np.zeros(3), dims=("t",), coords={"t": [9.0, 9.0, 9.0]})
        report = sciresult.compare(actual, reference, **ZERO)
        assert report.status is ComparisonStatus.DIFFERENT
        assert report.structural_issue_count >= 1
        assert report.counts is None
        assert report.numeric_scan_complete is False
        assert any("not performed" in note for note in report.notes)

    def test_correspondence_breaking_difference_withholds_the_scan(self) -> None:
        # "x" and "y" name different axes, so position 0 here is not the same quantity.
        actual = xr.DataArray(np.array([1.0, 2.0]), dims=("x",))
        reference = xr.DataArray(np.array([1.0, 9.0]), dims=("y",))
        report = sciresult.compare(actual, reference, **ZERO)
        assert report.status is ComparisonStatus.DIFFERENT
        assert StructuralCode.DIMENSIONS in _codes(report)
        assert report.counts is None

    def test_guarded_attribute_difference_still_scans(self) -> None:
        # A unit-string difference does not move any element, so both facts are reported.
        actual = xr.DataArray(np.array([1.0, 2.0]), dims=("x",), attrs={"units": "K"})
        reference = xr.DataArray(np.array([1.0, 2.0]), dims=("x",), attrs={"units": "s"})
        report = sciresult.compare(actual, reference, **ZERO)
        assert report.status is ComparisonStatus.DIFFERENT
        assert StructuralCode.GUARDED_ATTRIBUTE in _codes(report)
        assert report.counts is not None
        assert report.counts.total_failures == 0
        assert report.numeric_scan_complete is True

    def test_structure_issue_count_is_capped(self) -> None:
        actual = xr.DataArray(np.zeros(1), dims=("x",))
        reference = xr.DataArray(
            np.zeros(1),
            dims=("x",),
            coords={f"c{index}": ("x", [0.0]) for index in range(40)},
        )
        report = sciresult.compare(actual, reference, **ZERO)
        assert report.structural_issue_count <= 20
        assert len(report.structural_issues) <= 20

    def test_identical_inputs_report_no_structural_issues(self) -> None:
        values = np.linspace(0.0, 1.0, 5)
        actual = xr.DataArray(values, dims=("x",), coords={"x": values})
        report = sciresult.compare(actual, actual.copy(), **ZERO)
        assert report.structural_issue_count == 0
        assert report.status is ComparisonStatus.MATCH
