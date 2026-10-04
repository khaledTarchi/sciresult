"""The report model: invariants, immutability, and JSON safety.

A report is the library's product. These tests assert that it cannot be built in
an internally inconsistent state, that it cannot be mutated after the fact, and
that it serializes to strict JSON without leaking nonstandard Python objects.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import pickle

import numpy as np
import pytest
import xarray as xr

import sciresult
from sciresult import ComparisonStatus, FailureKind, LabelKind, StructuralCode

ZERO = {"rtol": 0.0, "atol": 0.0}


def _matching_report() -> sciresult.ComparisonReport:
    values = np.array([1.0, 2.0, 3.0])
    return sciresult.compare(
        xr.DataArray(values, dims=("x",)), xr.DataArray(values.copy(), dims=("x",)), **ZERO
    )


def _failing_report() -> sciresult.ComparisonReport:
    return sciresult.compare(
        xr.DataArray(np.array([1.0, 9.0]), dims=("x",)),
        xr.DataArray(np.array([1.0, 0.0]), dims=("x",)),
        **ZERO,
    )


def _structural_report() -> sciresult.ComparisonReport:
    return sciresult.compare(
        xr.DataArray(np.zeros(2), dims=("x",)),
        xr.DataArray(np.zeros(3), dims=("x",)),
        **ZERO,
    )


def _replace(report: sciresult.ComparisonReport, **changes: object) -> dict[str, object]:
    fields = {field.name for field in dataclasses.fields(report)}
    return {name: changes.get(name, getattr(report, name)) for name in fields}


def _rebuild(report: sciresult.ComparisonReport, **changes: object) -> object:
    return sciresult.ComparisonReport(**_replace(report, **changes))


class TestReportsAreFrozen:
    @pytest.mark.parametrize(
        "report_factory",
        [_matching_report, _failing_report, _structural_report],
    )
    def test_attribute_assignment_is_refused(self, report_factory: object) -> None:
        report = report_factory()  # type: ignore[operator]
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.status = ComparisonStatus.MATCH  # type: ignore[misc]

    @pytest.mark.parametrize(
        "report_factory",
        [_matching_report, _failing_report, _structural_report],
    )
    def test_deepcopy_is_equal(self, report_factory: object) -> None:
        report = report_factory()  # type: ignore[operator]
        assert copy.deepcopy(report) == report

    @pytest.mark.parametrize(
        "report_factory",
        [_matching_report, _failing_report, _structural_report],
    )
    def test_pickle_round_trip_is_equal(self, report_factory: object) -> None:
        report = report_factory()  # type: ignore[operator]
        assert pickle.loads(pickle.dumps(report)) == report

    def test_equality_is_structural(self) -> None:
        assert _matching_report() == _matching_report()
        assert _matching_report() != _failing_report()

    def test_bool_is_refused_so_the_status_cannot_be_missed(self) -> None:
        with pytest.raises(TypeError) as info:
            bool(_matching_report())
        assert "status" in str(info.value)

    def test_truthiness_cannot_be_faked_by_truthy_attributes(self) -> None:
        report = _failing_report()
        with pytest.raises(TypeError):
            if report:  # pragma: no cover - the guard is the assertion
                pytest.fail("a report must not be usable as a condition")


class TestReportInvariants:
    def test_status_must_be_an_enum_member(self) -> None:
        with pytest.raises(TypeError):
            _rebuild(_matching_report(), status="match")

    def test_tolerances_must_be_finite_and_nonnegative(self) -> None:
        for bad in (-1.0, math.inf, math.nan):
            with pytest.raises(ValueError):
                _rebuild(_matching_report(), rtol=bad)

    def test_counts_total_must_match_the_element_count(self) -> None:
        report = _failing_report()
        assert report.counts is not None
        with pytest.raises(ValueError, match="does not match total"):
            dataclasses.replace(report.counts, total=99)

    def test_a_withheld_scan_must_not_carry_counts(self) -> None:
        report = _structural_report()
        assert report.counts is None
        assert report.numeric_scan_complete is False
        matching = _matching_report()
        with pytest.raises(ValueError, match="withheld"):
            _rebuild(report, counts=matching.counts)

    def test_a_withheld_scan_must_not_carry_examples(self) -> None:
        report = _structural_report()
        failing = _failing_report()
        with pytest.raises(ValueError, match="withheld"):
            _rebuild(report, examples=failing.examples)

    def test_a_completed_scan_requires_a_count(self) -> None:
        report = _matching_report()
        with pytest.raises(ValueError, match="requires counts"):
            _rebuild(report, counts=None)

    def test_match_forbids_structural_issues(self) -> None:
        report = _structural_report()
        with pytest.raises(ValueError, match="MATCH"):
            _rebuild(report, status=ComparisonStatus.MATCH)

    def test_different_requires_an_established_finding(self) -> None:
        report = _matching_report()
        with pytest.raises(ValueError, match="DIFFERENT"):
            _rebuild(report, status=ComparisonStatus.DIFFERENT)

    def test_examples_may_not_exceed_max_examples(self) -> None:
        report = _failing_report()
        with pytest.raises(ValueError, match="exceed max_examples"):
            _rebuild(report, max_examples=0)

    def test_examples_must_be_in_strictly_descending_order(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([1.0, 3.0, 9.0, 27.0]), dims=("x",)),
            xr.DataArray(np.zeros(4), dims=("x",)),
            rtol=0.0,
            atol=1.0,
        )
        assert [example.score for example in report.examples] == [27.0, 9.0, 3.0]
        with pytest.raises(ValueError, match="descending"):
            _rebuild(report, examples=tuple(reversed(report.examples)))

    def test_example_flat_index_must_be_inside_the_data(self) -> None:
        report = _failing_report()
        out_of_range = dataclasses.replace(report.examples[0], flat_index=99)
        with pytest.raises(ValueError, match="out of range"):
            _rebuild(report, examples=(out_of_range,))

    def test_example_index_tuple_must_match_the_rank(self) -> None:
        report = _failing_report()
        wrong_rank = dataclasses.replace(report.examples[0], index=(0, 0))
        with pytest.raises((ValueError, TypeError)):
            _rebuild(report, examples=(wrong_rank,))


class TestFailureExampleInvariants:
    def test_finite_tolerance_examples_require_error_magnitudes(self) -> None:
        report = _failing_report()
        with pytest.raises(ValueError, match="require error magnitudes"):
            dataclasses.replace(report.examples[0], absolute_error=None)

    def test_absolute_error_must_be_positive_and_finite(self) -> None:
        report = _failing_report()
        for bad in (0.0, -1.0, math.inf, math.nan):
            with pytest.raises(ValueError, match="positive and finite"):
                dataclasses.replace(report.examples[0], absolute_error=bad)

    def test_nonfinite_examples_may_not_carry_error_magnitudes(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([np.nan]), dims=("x",)),
            xr.DataArray(np.array([1.0]), dims=("x",)),
            **ZERO,
        )
        example = report.examples[0]
        for field in ("absolute_error", "allowed_error", "score"):
            with pytest.raises(ValueError):
                dataclasses.replace(example, **{field: 1.0})

    def test_a_zero_allowance_infinity_is_not_saturation(self) -> None:
        # The exact-real ratio is infinite here, so the example still ties in
        # exact-real ranking and the report must not claim otherwise.
        report = sciresult.compare(
            xr.DataArray(np.array([1.0]), dims=("x",)),
            xr.DataArray(np.array([0.0]), dims=("x",)),
            **ZERO,
        )
        example = report.examples[0]
        assert example.score == math.inf
        assert example.score_saturated is False

    def test_saturation_requires_an_infinite_score(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([1.0, 9.0]), dims=("x",)),
            xr.DataArray(np.zeros(2), dims=("x",)),
            rtol=0.0,
            atol=1.0,
        )
        example = report.examples[0]
        assert math.isfinite(example.score)
        with pytest.raises(ValueError, match="score_saturated requires an infinite"):
            dataclasses.replace(example, score_saturated=True)

    def test_saturation_cannot_be_claimed_without_a_score(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([np.nan]), dims=("x",)),
            xr.DataArray(np.array([1.0]), dims=("x",)),
            **ZERO,
        )
        with pytest.raises(ValueError, match="score_saturated requires an infinite"):
            dataclasses.replace(report.examples[0], score_saturated=True)

    def test_kind_must_be_an_enum_member(self) -> None:
        with pytest.raises(TypeError):
            dataclasses.replace(_failing_report().examples[0], kind="finite_tolerance")

    def test_nonfinite_kinds_may_not_claim_a_finite_failure(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([np.inf]), dims=("x",)),
            xr.DataArray(np.array([-np.inf]), dims=("x",)),
            **ZERO,
        )
        example = report.examples[0]
        assert example.kind is FailureKind.INFINITY
        assert example.absolute_error is None


class TestLabelValueInvariants:
    def test_kind_must_be_an_enum_member(self) -> None:
        with pytest.raises(TypeError):
            sciresult.LabelValue(kind="float", value=1.0)

    def test_float_kind_requires_a_float_value(self) -> None:
        with pytest.raises(TypeError):
            sciresult.LabelValue(kind=LabelKind.FLOAT, value=1)

    def test_temporal_kinds_require_a_unit(self) -> None:
        with pytest.raises(TypeError, match="dtype_unit is required"):
            sciresult.LabelValue(kind=LabelKind.DATETIME, value=0)

    def test_non_temporal_kinds_forbid_a_unit(self) -> None:
        with pytest.raises(TypeError, match="dtype_unit must be None"):
            sciresult.LabelValue(kind=LabelKind.FLOAT, value=1.0, dtype_unit="ns")

    def test_original_length_requires_truncation(self) -> None:
        with pytest.raises(ValueError, match="truncated"):
            sciresult.LabelValue(kind=LabelKind.STRING, value="ab", original_length=9)

    def test_original_length_must_exceed_the_display_length(self) -> None:
        with pytest.raises(ValueError, match="exceed"):
            sciresult.LabelValue(
                kind=LabelKind.STRING, value="abcdef", truncated=True, original_length=3
            )

    def test_truncation_is_only_meaningful_for_text(self) -> None:
        with pytest.raises(TypeError, match="only text labels"):
            sciresult.LabelValue(kind=LabelKind.FLOAT, value=1.0, truncated=True, original_length=9)


class TestPairCountsInvariants:
    def test_fields_must_be_nonnegative_integers(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            sciresult.PairCounts(
                total=2,
                finite_pairs=-1,
                finite_failures=0,
                nan_position_mismatches=0,
                matching_nan_pairs=0,
                rejected_nan_pairs=0,
                matching_infinity_pairs=0,
                infinity_mismatches=0,
            )

    def test_subcategories_may_not_exceed_the_total(self) -> None:
        with pytest.raises(ValueError):
            sciresult.PairCounts(
                total=1,
                finite_pairs=2,
                finite_failures=0,
                nan_position_mismatches=0,
                matching_nan_pairs=0,
                rejected_nan_pairs=0,
                matching_infinity_pairs=0,
                infinity_mismatches=0,
            )

    def test_failures_may_not_exceed_the_pairs_they_came_from(self) -> None:
        with pytest.raises(ValueError):
            sciresult.PairCounts(
                total=2,
                finite_pairs=1,
                finite_failures=2,
                nan_position_mismatches=0,
                matching_nan_pairs=0,
                rejected_nan_pairs=0,
                matching_infinity_pairs=0,
                infinity_mismatches=0,
            )

    def test_total_failures_sums_every_category(self) -> None:
        counts = sciresult.PairCounts(
            total=10,
            finite_pairs=4,
            finite_failures=1,
            nan_position_mismatches=2,
            matching_nan_pairs=1,
            rejected_nan_pairs=0,
            matching_infinity_pairs=2,
            infinity_mismatches=1,
        )
        assert counts.total_failures == 1 + 2 + 0 + 1


class TestJsonSafety:
    @pytest.mark.parametrize(
        "report_factory",
        [_matching_report, _failing_report, _structural_report],
    )
    def test_to_dict_is_strictly_json_serializable(self, report_factory: object) -> None:
        report = report_factory()  # type: ignore[operator]
        text = json.dumps(report.to_dict(), allow_nan=False, sort_keys=True)
        assert json.loads(text) == report.to_dict()

    def test_to_dict_contains_no_numpy_scalars(self) -> None:
        report = _failing_report()

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
            else:
                assert not isinstance(node, np.generic), type(node)
                assert type(node) in (type(None), bool, int, float, str)

        walk(report.to_dict())

    def test_to_dict_is_detached_from_the_report(self) -> None:
        report = _failing_report()
        payload = report.to_dict()
        payload["counts"] = None
        assert report.counts is not None

    def test_status_is_encoded_as_a_plain_tag(self) -> None:
        payload = _failing_report().to_dict()
        assert payload["status"] == "different"
        assert isinstance(payload["status"], str)

    def test_enums_are_encoded_as_their_values(self) -> None:
        payload = _structural_report().to_dict()
        assert payload["structural_issues"][0]["code"] == StructuralCode.SHAPE.value

    def test_infinite_scores_survive_strict_json(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([1.0]), dims=("x",)),
            xr.DataArray(np.array([0.0]), dims=("x",)),
            **ZERO,
        )
        payload = report.to_dict()
        # An infinite diagnostic score cannot be encoded as a JSON number, so it is
        # tagged rather than silently turned into null or a bare Infinity token.
        example = payload["examples"][0]
        assert example["score"] != float("inf")
        assert json.dumps(payload, allow_nan=False)


class TestExceptionHierarchy:
    def test_every_operational_error_shares_one_base(self) -> None:
        for error_type in (
            sciresult.InvalidComparisonArgument,
            sciresult.UnsupportedInputError,
            sciresult.EmptyComparisonError,
            sciresult.ComparisonArithmeticError,
        ):
            assert issubclass(error_type, sciresult.SciResultError)

    def test_a_mismatch_is_a_result_rather_than_an_operational_error(self) -> None:
        # A failed comparison is an ordinary outcome for assert_close, so it must be
        # catchable as an AssertionError without being conflated with a broken call.
        assert issubclass(sciresult.ComparisonMismatch, AssertionError)
        assert not issubclass(sciresult.ComparisonMismatch, sciresult.SciResultError)

    def test_only_the_mismatch_is_an_assertion_error(self) -> None:
        assert issubclass(sciresult.ComparisonMismatch, AssertionError)
        for error_type in (
            sciresult.InvalidComparisonArgument,
            sciresult.UnsupportedInputError,
            sciresult.EmptyComparisonError,
            sciresult.ComparisonArithmeticError,
        ):
            assert not issubclass(error_type, AssertionError)

    def test_error_codes_are_stable_and_machine_readable(self) -> None:
        with pytest.raises(sciresult.UnsupportedInputError) as info:
            sciresult.compare(
                xr.DataArray(np.zeros(2, dtype=np.int32), dims=("x",)),
                xr.DataArray(np.zeros(2), dims=("x",)),
                **ZERO,
            )
        assert info.value.code == "actual_data_dtype_unsupported"
        assert info.value.REASON == "unsupported_input"

    def test_an_error_message_does_not_depend_on_object_reprs(self) -> None:
        with pytest.raises(sciresult.UnsupportedInputError) as info:
            sciresult.compare("not an array", "also not", **ZERO)  # type: ignore[arg-type]
        assert "<string" not in str(info.value)
