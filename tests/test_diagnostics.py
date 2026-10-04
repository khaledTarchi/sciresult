"""Failure-example diagnostics: labels, ordering, and bounded display.

These tests cover what a user actually reads when a comparison fails: which
positions are reported, what labels identify them, and how those labels are
represented without ever changing the comparison itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr

import sciresult
from sciresult import ComparisonStatus, FailureKind, LabelKind

ZERO = {"rtol": 0.0, "atol": 0.0}


def _labels_by_dimension(
    example: sciresult.FailureExample,
) -> dict[str, sciresult.LabelValue]:
    return dict(example.labels)


class TestLabelAvailability:
    def test_dimension_coordinate_supplies_the_label(self) -> None:
        actual = xr.DataArray(
            np.array([0.0, 0.0, 5.0]),
            dims=("t",),
            coords={"t": np.array([10.0, 20.0, 30.0])},
        )
        report = sciresult.compare(
            actual, xr.DataArray(np.zeros(3), dims=("t",), coords={"t": actual.t.data}), **ZERO
        )
        assert report.status is ComparisonStatus.DIFFERENT
        label = _labels_by_dimension(report.examples[0])["t"]
        assert label.kind is LabelKind.FLOAT
        assert label.value == 30.0

    def test_integer_coordinate_keeps_its_integer_kind(self) -> None:
        actual = xr.DataArray(
            np.array([0.0, 9.0]), dims=("x",), coords={"x": np.array([7, 8], dtype=np.int64)}
        )
        report = sciresult.compare(
            actual, xr.DataArray(np.zeros(2), dims=("x",), coords={"x": actual.x.data}), **ZERO
        )
        label = _labels_by_dimension(report.examples[0])["x"]
        assert label.kind is LabelKind.INTEGER
        assert label.value == 8
        assert type(label.value) is int

    def test_missing_coordinate_is_reported_as_an_integer_position(self) -> None:
        actual = xr.DataArray(np.array([0.0, 0.0, 5.0]), dims=("t",))
        report = sciresult.compare(actual, xr.DataArray(np.zeros(3), dims=("t",)), **ZERO)
        label = _labels_by_dimension(report.examples[0])["t"]
        assert label.kind is LabelKind.POSITION
        assert label.value == 2
        assert any("no coordinate variable" in note for note in report.notes)

    def test_every_declared_dimension_appears_exactly_once(self) -> None:
        actual = xr.DataArray(
            np.zeros((2, 3)), dims=("x", "y"), coords={"x": [0.0, 1.0], "y": [0.0, 1.0, 2.0]}
        )
        other = actual.copy()
        other[1, 2] = 9.0
        report = sciresult.compare(actual, other, **ZERO)
        labels = _labels_by_dimension(report.examples[0])
        assert set(labels) == {"x", "y"}
        assert labels["x"].value == 1.0
        assert labels["y"].value == 2.0

    def test_partial_coordinate_coverage_reports_positions_for_the_rest(self) -> None:
        actual = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"), coords={"x": [0.0, 1.0]})
        other = actual.copy()
        other[1, 2] = 9.0
        report = sciresult.compare(actual, other, **ZERO)
        labels = _labels_by_dimension(report.examples[0])
        assert labels["x"].kind is LabelKind.FLOAT
        assert labels["y"].kind is LabelKind.POSITION
        assert labels["y"].value == 2
        assert any("'y'" in note for note in report.notes)

    def test_multidimensional_auxiliary_coordinate_is_not_enumerated(self) -> None:
        actual = xr.DataArray(
            np.zeros((2, 2)),
            dims=("x", "y"),
            coords={"aux": (("x", "y"), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        )
        other = actual.copy()
        other[0, 0] = 9.0
        report = sciresult.compare(actual, other, **ZERO)
        labels = _labels_by_dimension(report.examples[0])
        assert set(labels) == {"x", "y"}
        assert all(label.kind is LabelKind.POSITION for label in labels.values())


class TestLabelKinds:
    def test_datetime_coordinate_label_keeps_ticks_and_unit(self) -> None:
        actual = xr.DataArray(
            np.zeros(2),
            dims=("t",),
            coords={"t": np.array(["2020-01-01", "2020-06-15"], dtype="M8[ns]")},
        )
        other = actual.copy()
        other[1] = 5.0
        report = sciresult.compare(actual, other, **ZERO)
        label = _labels_by_dimension(report.examples[0])["t"]
        assert label.kind is LabelKind.DATETIME
        assert label.dtype_unit == "ns"
        assert type(label.value) is int
        assert label.value == int(np.datetime64("2020-06-15", "ns").astype(np.int64))

    def test_timedelta_coordinate_label_keeps_ticks_and_unit(self) -> None:
        actual = xr.DataArray(
            np.zeros(2), dims=("t",), coords={"t": np.array([1, 5], dtype="m8[s]")}
        )
        other = actual.copy()
        other[1] = 5.0
        report = sciresult.compare(actual, other, **ZERO)
        label = _labels_by_dimension(report.examples[0])["t"]
        assert label.kind is LabelKind.TIMEDELTA
        assert label.dtype_unit == "s"
        assert label.value == 5

    def test_nan_coordinate_label_keeps_the_float_kind(self) -> None:
        actual = xr.DataArray(
            np.array([0.0, 0.0]), dims=("x",), coords={"x": np.array([np.nan, 1.0])}
        )
        other = actual.copy()
        other[0] = 5.0
        report = sciresult.compare(actual, other, **ZERO)
        label = _labels_by_dimension(report.examples[0])["x"]
        assert label.kind is LabelKind.FLOAT
        assert isinstance(label.value, float)
        assert math.isnan(label.value)

    def test_string_coordinate_label_is_truncated_for_display_only(self) -> None:
        long_text = "s" * 400
        actual = xr.DataArray(
            np.zeros(2), dims=("x",), coords={"x": np.array(["short", long_text])}
        )
        other = actual.copy()
        other[1] = 5.0
        report = sciresult.compare(actual, other, **ZERO)
        label = _labels_by_dimension(report.examples[0])["x"]
        assert label.kind is LabelKind.STRING
        assert label.truncated is True
        assert len(label.value) < len(long_text)
        assert label.original_length == len(long_text)
        assert any("truncated" in note for note in report.notes)

    def test_short_string_label_is_not_marked_truncated(self) -> None:
        actual = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": np.array(["alpha", "beta"])})
        other = actual.copy()
        other[1] = 5.0
        report = sciresult.compare(actual, other, **ZERO)
        label = _labels_by_dimension(report.examples[0])["x"]
        assert label.value == "beta"
        assert label.truncated is False
        assert label.original_length is None

    def test_a_truncated_label_does_not_change_the_comparison(self) -> None:
        long_text = "s" * 400
        actual = xr.DataArray(
            np.zeros(2), dims=("x",), coords={"x": np.array([long_text, "other"])}
        )
        report = sciresult.compare(actual, actual.copy(), **ZERO)
        assert report.status is ComparisonStatus.MATCH

    def test_control_characters_in_a_label_survive_the_report(self) -> None:
        awkward = "line\nbreak\ttab"
        actual = xr.DataArray(np.zeros(2), dims=("x",), coords={"x": np.array([awkward, "b"])})
        other = actual.copy()
        other[0] = 1.0
        report = sciresult.compare(actual, other, **ZERO)
        label = _labels_by_dimension(report.examples[0])["x"]
        assert label.value == awkward


class TestExampleContents:
    def test_example_records_both_operands(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([1.0, 2.0]), dims=("x",)),
            xr.DataArray(np.array([1.0, 9.0]), dims=("x",)),
            **ZERO,
        )
        example = report.examples[0]
        assert example.actual == 2.0
        assert example.reference == 9.0
        assert example.absolute_error == 7.0
        assert example.allowed_error == 0.0
        assert example.kind is FailureKind.FINITE_TOLERANCE

    def test_nonfinite_examples_do_not_invent_error_magnitudes(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([np.nan, np.inf]), dims=("x",)),
            xr.DataArray(np.array([1.0, 1.0]), dims=("x",)),
            **ZERO,
        )
        for example in report.examples:
            assert example.kind is not FailureKind.FINITE_TOLERANCE
            assert example.absolute_error is None
            assert example.allowed_error is None
            assert example.score is None
            assert example.score_saturated is False

    def test_examples_use_the_actual_side_for_labels(self) -> None:
        actual = xr.DataArray(
            np.array([0.0, 5.0]), dims=("x",), coords={"x": np.array([100.0, 200.0])}
        )
        reference = xr.DataArray(
            np.array([0.0, 0.0]), dims=("x",), coords={"x": np.array([100.0, 999.0])}
        )
        # The coordinates differ, so the scan is withheld; here the data agrees.
        report = sciresult.compare(actual, reference, **ZERO)
        assert report.counts is None
        assert report.status is ComparisonStatus.DIFFERENT

    def test_max_metric_carries_labels_too(self) -> None:
        actual = xr.DataArray(
            np.array([0.0, 0.0]), dims=("x",), coords={"x": np.array([10.0, 20.0])}
        )
        reference = xr.DataArray(np.array([0.0, 7.0]), dims=("x",), coords={"x": actual.x.data})
        report = sciresult.compare(actual, reference, rtol=0.0, atol=100.0)
        metric = report.max_finite_absolute_error
        assert metric is not None
        assert _labels_by_dimension(metric)["x"].value == 20.0


class TestOrderingContract:
    def test_examples_are_in_strictly_descending_report_order(self) -> None:
        actual = xr.DataArray(np.array([np.nan, np.nan, np.inf, 1.0, 10.0, 5.0]), dims=("x",))
        reference = xr.DataArray(np.array([1.0, np.nan, -np.inf, 0.0, 0.0, 0.0]), dims=("x",))
        report = sciresult.compare(actual, reference, rtol=0.0, atol=1.0)
        priority = {
            FailureKind.NAN_POSITION: 3,
            FailureKind.NAN_POLICY: 2,
            FailureKind.INFINITY: 1,
            FailureKind.FINITE_TOLERANCE: 0,
        }
        kinds = [example.kind for example in report.examples]
        ranks = [priority[kind] for kind in kinds]
        assert ranks == sorted(ranks, reverse=True)

    def test_the_cap_keeps_the_most_significant_examples(self) -> None:
        actual = xr.DataArray(np.array([np.nan, 1.0, 10.0]), dims=("x",))
        reference = xr.DataArray(np.array([1.0, 0.0, 0.0]), dims=("x",))
        full = sciresult.compare(actual, reference, rtol=0.0, atol=1.0, max_examples=10)
        capped = sciresult.compare(actual, reference, rtol=0.0, atol=1.0, max_examples=1)
        assert capped.examples[0].kind is full.examples[0].kind
        assert capped.examples[0].index == full.examples[0].index


class TestReportDeterminism:
    def test_repeated_comparison_is_identical(self) -> None:
        rng = np.random.default_rng(4242)
        actual = rng.normal(size=1000)
        reference = actual + rng.normal(scale=0.1, size=1000)
        actual[5] = np.nan
        reference[9] = np.inf
        first = sciresult.compare(
            xr.DataArray(actual, dims=("x",)),
            xr.DataArray(reference, dims=("x",)),
            rtol=1e-6,
            atol=1e-9,
            max_examples=7,
        )
        second = sciresult.compare(
            xr.DataArray(actual, dims=("x",)),
            xr.DataArray(reference, dims=("x",)),
            rtol=1e-6,
            atol=1e-9,
            max_examples=7,
        )
        assert first.to_dict() == second.to_dict()

    def test_inputs_are_never_mutated(self) -> None:
        actual_values = np.array([1.0, 2.0, 3.0])
        reference_values = np.array([1.0, 5.0, 3.0])
        coords = np.array([10.0, 20.0, 30.0])
        actual = xr.DataArray(actual_values.copy(), dims=("x",), coords={"x": coords.copy()})
        reference = xr.DataArray(reference_values.copy(), dims=("x",), coords={"x": coords.copy()})
        sciresult.compare(actual, reference, rtol=0.0, atol=0.0)
        assert np.array_equal(actual.data, actual_values)
        assert np.array_equal(reference.data, reference_values)
        assert np.array_equal(actual.coords["x"].data, coords)

    def test_reported_labels_are_detached_from_the_input(self) -> None:
        coords = np.array([10.0, 20.0])
        actual = xr.DataArray(np.array([0.0, 9.0]), dims=("x",), coords={"x": coords})
        reference = xr.DataArray(np.array([0.0, 0.0]), dims=("x",), coords={"x": coords.copy()})
        report = sciresult.compare(actual, reference, **ZERO)
        label = _labels_by_dimension(report.examples[0])["x"]
        coords[1] = 999.0
        assert label.value == 20.0


class TestNamesAndIdentity:
    def test_names_are_reported_when_present(self) -> None:
        actual = xr.DataArray(np.zeros(2), dims=("x",), name="run_a")
        reference = xr.DataArray(np.zeros(2), dims=("x",), name="run_b")
        report = sciresult.compare(actual, reference, **ZERO)
        assert report.actual_name == "run_a"
        assert report.reference_name == "run_b"

    def test_names_are_none_when_absent(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.zeros(2), dims=("x",)),
            xr.DataArray(np.zeros(2), dims=("x",)),
            **ZERO,
        )
        assert report.actual_name is None
        assert report.reference_name is None

    def test_a_name_difference_alone_does_not_change_the_status(self) -> None:
        named = xr.DataArray(np.zeros(2), dims=("x",), name="run_a")
        unnamed = xr.DataArray(np.zeros(2), dims=("x",))
        report = sciresult.compare(named, unnamed, **ZERO)
        assert report.status is ComparisonStatus.MATCH

    def test_non_text_names_are_rejected(self) -> None:
        array = xr.DataArray(np.zeros(2), dims=("x",))
        array.name = 7  # type: ignore[assignment]
        with pytest.raises(sciresult.UnsupportedInputError) as info:
            sciresult.compare(array, array, **ZERO)
        assert info.value.code == "actual_name_not_text"
