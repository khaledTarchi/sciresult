"""End-to-end public API behaviour, including seeded scientific defects.

The seeded fixtures in :class:`TestSeededScientificDefects` are small, fixed
scenarios of the kind of mistake that silently corrupts a scientific result: a
shifted time coordinate, a single bad pixel, a slow drift, a moved missing
value, and so on. Each one documents the expected status, the expected category,
whether a numerical scan happens at all, the expected failure count, and the
leading diagnostic, so a regression names the science it broke.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import xarray as xr

import sciresult
from sciresult import ComparisonStatus, FailureKind, StructuralCode

TOL = {"rtol": 0.0, "atol": 0.0}


def _plain(values: object, dims: tuple[str, ...] = ("x",)) -> xr.DataArray:
    return xr.DataArray(np.asarray(values), dims=dims)


class TestSignatureAndOptions:
    def test_tolerances_are_keyword_only(self) -> None:
        array = _plain([1.0, 2.0])
        with pytest.raises(TypeError):
            sciresult.compare(array, array, 0.0, 0.0)  # type: ignore[misc]

    def test_tolerances_are_required(self) -> None:
        array = _plain([1.0, 2.0])
        with pytest.raises(TypeError):
            sciresult.compare(array, array)  # type: ignore[call-arg]

    def test_defaults_match_the_documented_contract(self) -> None:
        array = _plain([1.0, 2.0])
        report = sciresult.compare(array, array, rtol=0.0, atol=0.0)
        assert report.equal_nan is False
        assert report.max_examples == 5
        assert report.status is ComparisonStatus.MATCH

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
    def test_invalid_tolerances_are_rejected(self, bad: float) -> None:
        array = _plain([1.0, 2.0])
        with pytest.raises(sciresult.InvalidComparisonArgument) as info:
            sciresult.compare(array, array, rtol=bad, atol=0.0)
        assert info.value.code in ("rtol_nan", "rtol_infinite", "rtol_negative")

    @pytest.mark.parametrize("bad", [True, 1.5, "5", None])
    def test_max_examples_must_be_an_integer(self, bad: object) -> None:
        array = _plain([1.0, 2.0])
        with pytest.raises(sciresult.InvalidComparisonArgument):
            sciresult.compare(array, array, rtol=0.0, atol=0.0, max_examples=bad)  # type: ignore[arg-type]

    def test_integer_tolerances_are_accepted_as_floats(self) -> None:
        array = _plain([1.0, 2.0])
        report = sciresult.compare(array, array, rtol=0, atol=0)
        assert report.rtol == 0.0
        assert type(report.rtol) is float

    def test_equal_nan_must_be_a_bool(self) -> None:
        array = _plain([1.0, 2.0])
        with pytest.raises(sciresult.InvalidComparisonArgument):
            sciresult.compare(array, array, rtol=0.0, atol=0.0, equal_nan=1)  # type: ignore[arg-type]


class TestEqualNanPolicy:
    def test_nan_on_both_sides_fails_by_default(self) -> None:
        report = sciresult.compare(_plain([np.nan, 1.0]), _plain([np.nan, 1.0]), rtol=0.0, atol=0.0)
        assert report.status is ComparisonStatus.DIFFERENT
        assert report.examples[0].kind is FailureKind.NAN_POLICY

    def test_equal_nan_true_accepts_matched_nan(self) -> None:
        report = sciresult.compare(
            _plain([np.nan, 1.0]), _plain([np.nan, 1.0]), rtol=0.0, atol=0.0, equal_nan=True
        )
        assert report.status is ComparisonStatus.MATCH
        assert report.counts is not None
        assert report.counts.matching_nan_pairs == 1
        assert report.counts.rejected_nan_pairs == 0

    def test_equal_nan_true_does_not_hide_a_one_sided_nan(self) -> None:
        report = sciresult.compare(
            _plain([np.nan]), _plain([1.0]), rtol=0.0, atol=0.0, equal_nan=True
        )
        assert report.status is ComparisonStatus.DIFFERENT
        assert report.examples[0].kind is FailureKind.NAN_POSITION


class TestInfinityHandling:
    def test_matched_infinities_are_a_match(self) -> None:
        report = sciresult.compare(
            _plain([np.inf, -np.inf]), _plain([np.inf, -np.inf]), rtol=0.0, atol=0.0
        )
        assert report.status is ComparisonStatus.MATCH
        assert report.counts is not None
        assert report.counts.matching_infinity_pairs == 2

    def test_infinity_pairs_with_a_finite_value(self) -> None:
        report = sciresult.compare(_plain([np.inf]), _plain([1e308]), rtol=0.0, atol=1e308)
        assert report.status is ComparisonStatus.DIFFERENT
        assert report.examples[0].kind is FailureKind.INFINITY

    def test_signed_infinities_disagree(self) -> None:
        report = sciresult.compare(_plain([np.inf]), _plain([-np.inf]), rtol=0.0, atol=0.0)
        assert report.examples[0].kind is FailureKind.INFINITY

    def test_a_finite_tolerance_never_rescues_an_infinity(self) -> None:
        # A generous tolerance must not turn an infinite discrepancy into a pass.
        report = sciresult.compare(_plain([np.inf]), _plain([0.0]), rtol=1.0, atol=1e308)
        assert report.status is ComparisonStatus.DIFFERENT
        assert report.examples[0].kind is FailureKind.INFINITY


class TestAssertClose:
    def test_matching_input_returns_none(self) -> None:
        array = _plain([1.0, 2.0])
        assert sciresult.assert_close(array, array, rtol=0.0, atol=0.0) is None

    def test_a_mismatch_raises_with_the_report_attached(self) -> None:
        with pytest.raises(sciresult.ComparisonMismatch) as info:
            sciresult.assert_close(_plain([1.0, 9.0]), _plain([1.0, 0.0]), **TOL)
        report = info.value.report
        assert report.status is ComparisonStatus.DIFFERENT
        assert len(report.examples) == 1

    def test_the_message_is_the_rendered_report(self) -> None:
        with pytest.raises(sciresult.ComparisonMismatch) as info:
            sciresult.assert_close(_plain([1.0, 9.0]), _plain([1.0, 0.0]), **TOL)
        assert str(info.value) == report_text(info.value.report)

    def test_an_operational_problem_is_not_reported_as_a_mismatch(self) -> None:
        with pytest.raises(sciresult.UnsupportedInputError):
            sciresult.assert_close(_plain([1, 2]), _plain([1.0, 2.0]), **TOL)  # type: ignore[arg-type]

    def test_assert_close_does_not_mutate_its_inputs(self) -> None:
        actual_values = np.array([1.0, 2.0])
        reference_values = np.array([1.0, 3.0])
        actual = xr.DataArray(actual_values.copy(), dims=("x",))
        reference = xr.DataArray(reference_values.copy(), dims=("x",))
        with pytest.raises(sciresult.ComparisonMismatch):
            sciresult.assert_close(actual, reference, **TOL)
        assert np.array_equal(actual.data, actual_values)
        assert np.array_equal(reference.data, reference_values)


def report_text(report: sciresult.ComparisonReport) -> str:
    return sciresult.formatting.format_report(report)


class TestEmptyInput:
    def test_an_empty_comparison_is_refused(self) -> None:
        with pytest.raises(sciresult.EmptyComparisonError):
            sciresult.compare(_plain(np.zeros(0)), _plain(np.zeros(0)), **TOL)

    def test_a_zero_sized_axis_is_also_refused(self) -> None:
        with pytest.raises(sciresult.EmptyComparisonError):
            sciresult.compare(
                _plain(np.zeros((0, 3)), dims=("x", "y")),
                _plain(np.zeros((0, 3)), dims=("x", "y")),
                **TOL,
            )


class TestSeededScientificDefects:
    """Fixed scenarios of realistic mistakes, with documented expectations."""

    def test_shifted_time_coordinate(self) -> None:
        # Same values, same shape, time axis shifted by one step.
        actual = xr.DataArray(
            np.array([10.0, 20.0, 30.0]),
            dims=("t",),
            coords={"t": np.array([1.0, 2.0, 3.0])},
        )
        reference = xr.DataArray(
            np.array([10.0, 20.0, 30.0]),
            dims=("t",),
            coords={"t": np.array([2.0, 3.0, 4.0])},
        )
        report = sciresult.compare(actual, reference, **TOL)
        assert report.status is ComparisonStatus.DIFFERENT
        assert {issue.code for issue in report.structural_issues} >= {
            StructuralCode.COORDINATE_VALUES,
            StructuralCode.INDEX_VALUES,
        }
        # Positions are no longer trustworthy, so nothing numeric is claimed.
        assert report.numeric_scan_complete is False
        assert report.counts is None
        assert report.examples == ()

    def test_one_localized_numerical_spike(self) -> None:
        reference = xr.DataArray(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]), dims=("t",), coords={"t": np.arange(5.0)}
        )
        actual = reference.copy()
        actual[2] = 3.5
        report = sciresult.compare(actual, reference, rtol=0.0, atol=0.0)
        assert report.status is ComparisonStatus.DIFFERENT
        assert report.structural_issue_count == 0
        assert report.numeric_scan_complete is True
        assert report.counts is not None
        assert report.counts.total_failures == 1
        assert report.counts.finite_failures == 1
        leading = report.examples[0]
        assert leading.kind is FailureKind.FINITE_TOLERANCE
        assert leading.index == (2,)
        assert dict(leading.labels)["t"].value == 2.0
        assert report.max_finite_absolute_error is not None
        assert report.max_finite_absolute_error.index == (2,)

    def test_broad_low_amplitude_drift(self) -> None:
        # Every point is slightly wrong: many equal-score failures, so the report
        # must stay bounded while still counting every failure.
        size = 1000
        reference = xr.DataArray(np.zeros(size), dims=("t",))
        actual = xr.DataArray(np.full(size, 1e-3), dims=("t",))
        report = sciresult.compare(actual, reference, rtol=0.0, atol=1e-4, max_examples=3)
        assert report.status is ComparisonStatus.DIFFERENT
        assert report.counts is not None
        assert report.counts.total_failures == size
        assert len(report.examples) == 3
        assert all(example.score == 10.0 for example in report.examples)
        assert [example.flat_index for example in report.examples] == [0, 1, 2]

    def test_changed_missing_value_position(self) -> None:
        actual = xr.DataArray(np.array([1.0, np.nan, 3.0]), dims=("x",))
        reference = xr.DataArray(np.array([1.0, 2.0, np.nan]), dims=("x",))
        report = sciresult.compare(actual, reference, rtol=0.0, atol=0.0)
        assert report.status is ComparisonStatus.DIFFERENT
        assert report.counts is not None
        assert report.counts.nan_position_mismatches == 2
        # NaN-position outranks everything, so both positions are reported first.
        assert {example.index for example in report.examples} == {(1,), (2,)}
        assert all(example.kind is FailureKind.NAN_POSITION for example in report.examples)

    def test_changed_data_dtype(self) -> None:
        actual = xr.DataArray(np.array([1.0, 2.0], dtype=np.float64), dims=("x",))
        reference = xr.DataArray(np.array([1.0, 2.0], dtype=np.float32), dims=("x",))
        report = sciresult.compare(actual, reference, **TOL)
        # The values agree but the stored precision changed, which changes the
        # meaning of any later arithmetic, so it is reported as a difference.
        assert report.status is ComparisonStatus.DIFFERENT
        assert {issue.code for issue in report.structural_issues} == {StructuralCode.DATA_DTYPE}
        assert report.actual_dtype == "<f8"
        assert report.reference_dtype == "<f4"
        # A dtype change does not move positions, so values are still checked.
        assert report.numeric_scan_complete is True
        assert report.counts is not None
        assert report.counts.total == 2

    def test_changed_units_attribute(self) -> None:
        actual = xr.DataArray(np.array([1.0, 2.0]), dims=("x",), attrs={"units": "m/s"})
        reference = xr.DataArray(np.array([1.0, 2.0]), dims=("x",), attrs={"units": "km/h"})
        report = sciresult.compare(actual, reference, **TOL)
        # Identical values with different units are not the same scientific result,
        # so a guarded-attribute difference is a difference in its own right.
        assert report.status is ComparisonStatus.DIFFERENT
        assert {issue.code for issue in report.structural_issues} == {
            StructuralCode.GUARDED_ATTRIBUTE
        }
        issue = report.structural_issues[0]
        assert issue.coordinate_name is None
        assert "'units'" in issue.message
        assert "not interpreted or converted" in issue.message
        assert issue.actual is not None and issue.actual.value == "m/s"
        assert issue.reference is not None and issue.reference.value == "km/h"
        assert report.numeric_scan_complete is True
        assert report.counts is not None
        assert report.counts.total_failures == 0

    def test_changed_dimension_order(self) -> None:
        actual = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("x", "y"))
        reference = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("y", "x"))
        report = sciresult.compare(actual, reference, **TOL)
        assert report.status is ComparisonStatus.DIFFERENT
        assert StructuralCode.DIMENSIONS in {issue.code for issue in report.structural_issues}
        assert report.numeric_scan_complete is False
        assert report.counts is None
        assert report.examples == ()
        assert report.max_finite_absolute_error is None

    def test_duplicated_coordinate_labels(self) -> None:
        actual = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=("t",),
            coords={"t": np.array([0.0, 0.0, 1.0])},
        )
        reference = actual.copy()
        report = sciresult.compare(actual, reference, **TOL)
        # Duplicate labels are allowed inputs, so identical values must still match.
        assert report.status is ComparisonStatus.MATCH
        assert report.structural_issues == ()
        assert report.numeric_scan_complete is True

    def test_duplicate_labels_are_disambiguated_by_position(self) -> None:
        actual = xr.DataArray(
            np.array([1.0, 2.0, 9.0]), dims=("t",), coords={"t": np.array([0.0, 0.0, 1.0])}
        )
        reference = xr.DataArray(
            np.array([1.0, 2.0, 3.0]), dims=("t",), coords={"t": np.array([0.0, 0.0, 1.0])}
        )
        report = sciresult.compare(actual, reference, **TOL)
        # Two positions share the label 0.0, so the label alone is ambiguous and
        # the integer position is what identifies the failing element.
        example = report.examples[0]
        assert dict(example.labels)["t"].value == 1.0
        assert example.index == (2,)
        assert example.flat_index == 2

    def test_duplicate_labels_do_not_mask_a_coordinate_shift(self) -> None:
        actual = xr.DataArray(
            np.array([1.0, 2.0, 3.0]), dims=("t",), coords={"t": np.array([0.0, 0.0, 1.0])}
        )
        reference = xr.DataArray(
            np.array([1.0, 2.0, 3.0]), dims=("t",), coords={"t": np.array([0.0, 1.0, 1.0])}
        )
        report = sciresult.compare(actual, reference, **TOL)
        assert report.status is ComparisonStatus.DIFFERENT
        assert StructuralCode.INDEX_VALUES in {issue.code for issue in report.structural_issues}
        assert report.numeric_scan_complete is False

    def test_a_sign_error_is_localised_and_ranked_first(self) -> None:
        reference = xr.DataArray(np.array([1.0, -2.0, 3.0]), dims=("x",))
        actual = xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=("x",))
        report = sciresult.compare(actual, reference, rtol=0.0, atol=0.0)
        assert report.counts is not None and report.counts.total_failures == 1
        example = report.examples[0]
        assert example.index == (1,)
        assert example.actual == 2.0
        assert example.reference == -2.0
        assert example.absolute_error == 4.0

    def test_a_precision_change_is_caught_by_a_tight_tolerance(self) -> None:
        reference = xr.DataArray(np.array([0.1, 0.2, 0.3]), dims=("x",))
        actual = reference.astype(np.float32).astype(np.float64)
        loose = sciresult.compare(actual, reference, rtol=1e-3, atol=0.0)
        tight = sciresult.compare(actual, reference, rtol=1e-12, atol=0.0)
        assert loose.status is ComparisonStatus.MATCH
        assert tight.status is ComparisonStatus.DIFFERENT
        assert tight.counts is not None
        assert tight.counts.total_failures > 0


class TestReportDoesNotRetainInputs:
    def test_the_report_holds_no_array_reference(self) -> None:
        actual = xr.DataArray(np.arange(10.0), dims=("x",), coords={"x": np.arange(10.0)})
        reference = actual + 1.0
        report = sciresult.compare(actual, reference, **TOL)
        values = [getattr(report, name) for name in report.__slots__]
        for value in values:
            assert not isinstance(value, np.ndarray)
            assert not isinstance(value, xr.DataArray)
        assert not isinstance(report.max_finite_absolute_error, xr.DataArray)

    def test_mutating_an_input_afterwards_cannot_change_the_report(self) -> None:
        actual = xr.DataArray(np.array([1.0, 2.0]), dims=("x",), coords={"x": np.array([7.0, 8.0])})
        reference = xr.DataArray(
            np.array([1.0, 5.0]), dims=("x",), coords={"x": np.array([7.0, 8.0])}
        )
        report = sciresult.compare(actual, reference, **TOL)
        actual.data[1] = 99.0
        actual.coords["x"].data[1] = 0.0
        reference.data[1] = -1.0
        assert report.examples[0].actual == 2.0
        assert report.examples[0].reference == 5.0
        assert dict(report.examples[0].labels)["x"].value == 8.0


class TestBooleanRefusal:
    def test_a_report_cannot_be_used_as_a_condition(self) -> None:
        report = sciresult.compare(_plain([1.0]), _plain([1.0]), **TOL)
        with pytest.raises(TypeError):
            bool(report)

    def test_the_refusal_names_the_attribute_to_read(self) -> None:
        report = sciresult.compare(_plain([1.0]), _plain([1.0]), **TOL)
        with pytest.raises(TypeError, match="status"):
            bool(report)

    def test_reading_status_explicitly_works(self) -> None:
        report = sciresult.compare(_plain([1.0]), _plain([2.0]), **TOL)
        assert report.status is ComparisonStatus.DIFFERENT
        assert report.status.value == "different"


class TestFormatAndSerialise:
    def test_format_is_available_on_the_report(self) -> None:
        report = sciresult.compare(_plain([1.0, 2.0]), _plain([1.0, 3.0]), **TOL)
        assert report.format().startswith("sciresult comparison report")

    def test_to_dict_is_json_ready(self) -> None:
        report = sciresult.compare(_plain([1.0, 2.0]), _plain([1.0, 3.0]), **TOL)
        text = json.dumps(report.to_dict(), allow_nan=False)
        assert json.loads(text)["status"] == "different"

    def test_a_mismatch_error_exposes_the_same_payload(self) -> None:
        with pytest.raises(sciresult.ComparisonMismatch) as info:
            sciresult.assert_close(_plain([1.0, 2.0]), _plain([1.0, 3.0]), **TOL)
        assert info.value.report.to_dict()["status"] == "different"
        assert info.value.report.status is not ComparisonStatus.MATCH


class TestDeterministicAcrossCalls:
    def test_the_same_inputs_give_the_same_text(self) -> None:
        rng = np.random.default_rng(11)
        actual = rng.normal(size=200)
        reference = actual + rng.normal(scale=0.25, size=200)
        actual[7] = np.nan
        reference[7] = np.nan
        actual[8] = np.inf
        first = sciresult.compare(
            xr.DataArray(actual, dims=("x",)), xr.DataArray(reference, dims=("x",)), **TOL
        )
        second = sciresult.compare(
            xr.DataArray(actual, dims=("x",)), xr.DataArray(reference, dims=("x",)), **TOL
        )
        assert first.format() == second.format()
        assert first.counts == second.counts
        assert json.dumps(first.to_dict(), allow_nan=False) == json.dumps(
            second.to_dict(), allow_nan=False
        )

    def test_results_do_not_depend_on_block_size(self) -> None:
        rng = np.random.default_rng(12)
        actual = rng.normal(size=1000)
        reference = actual + rng.normal(scale=0.5, size=1000)
        full = sciresult.compare(
            xr.DataArray(actual, dims=("x",)), xr.DataArray(reference, dims=("x",)), **TOL
        )
        for size in (1, 7, 64, 997, 4096):
            sciresult.numeric.BLOCK_SIZE = size
            try:
                chunked = sciresult.compare(
                    xr.DataArray(actual, dims=("x",)),
                    xr.DataArray(reference, dims=("x",)),
                    **TOL,
                )
            finally:
                sciresult.numeric.BLOCK_SIZE = 65536
            assert chunked.counts == full.counts
            assert chunked.examples == full.examples


class TestUnrepresentableScore:
    def test_zero_allowance_is_an_intentional_infinity(self) -> None:
        report = sciresult.compare(_plain([1.0]), _plain([0.0]), **TOL)
        example = report.examples[0]
        assert example.allowed_error == 0.0
        assert example.score == math.inf
        # The exact-real ratio is genuinely infinite, not a finite value that
        # overflowed, so ranking fidelity is not in question.
        assert example.score_saturated is False
        assert not any("saturated" in note for note in report.notes)

    def test_overflowed_quotient_is_marked_as_saturated(self) -> None:
        report = sciresult.compare(_plain([1.0]), _plain([0.0]), rtol=0.0, atol=5e-324)
        example = report.examples[0]
        assert example.allowed_error == 5e-324
        assert example.score == math.inf
        assert example.score_saturated is True
        assert any("saturated" in note for note in report.notes)
