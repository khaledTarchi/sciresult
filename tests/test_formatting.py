"""Human-readable output: the plain-text report, JSON encoding, and escaping.

A comparison report is read by people debugging a scientific result, so the
text form is pinned with snapshots. The invariants that make it trustworthy are
asserted directly: an empty finite population is never described as a pass, and
an unrepresentable diagnostic score is never printed as a finite number.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import xarray as xr

import sciresult
from sciresult.formatting import format_report

ZERO = {"rtol": 0.0, "atol": 0.0}


class TestSnapshotReports:
    def test_matching_comparison(self, text_snapshot: object) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        report = sciresult.compare(
            xr.DataArray(values, dims=("t",), coords={"t": np.arange(4.0)}, name="run_a"),
            xr.DataArray(values.copy(), dims=("t",), coords={"t": np.arange(4.0)}, name="run_b"),
            rtol=1e-7,
            atol=1e-12,
        )
        text_snapshot("match", format_report(report))

    def test_numerical_difference(self, text_snapshot: object) -> None:
        actual = xr.DataArray(
            np.array([1.0, 2.0, 3.0000001, 4.0, 5.0]),
            dims=("t",),
            coords={"t": np.array([10.0, 20.0, 30.0, 40.0, 50.0])},
            name="model_v2",
        )
        reference = xr.DataArray(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            dims=("t",),
            coords={"t": np.array([10.0, 20.0, 30.0, 40.0, 50.0])},
            name="model_v1",
        )
        report = sciresult.compare(actual, reference, rtol=1e-9, atol=0.0)
        text_snapshot("numerical_difference", format_report(report))

    def test_mixed_nonfinite_categories(self, text_snapshot: object) -> None:
        actual = xr.DataArray(
            np.array([np.nan, np.nan, np.inf, -np.inf, 1.0, 2.0]),
            dims=("t",),
            coords={"t": np.arange(6.0)},
        )
        reference = xr.DataArray(
            np.array([1.0, np.nan, np.inf, np.inf, 1.0, 5.0]),
            dims=("t",),
            coords={"t": np.arange(6.0)},
        )
        report = sciresult.compare(actual, reference, rtol=0.0, atol=0.5)
        text_snapshot("nonfinite_categories", format_report(report))

    def test_structural_difference(self, text_snapshot: object) -> None:
        actual = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"), name="actual")
        reference = xr.DataArray(np.zeros((3, 2)), dims=("y", "x"), name="reference")
        report = sciresult.compare(actual, reference, **ZERO)
        text_snapshot("structural_difference", format_report(report))

    def test_nan_position_and_nan_policy(self, text_snapshot: object) -> None:
        actual = xr.DataArray(np.array([np.nan, np.nan, 1.0, 2.0]), dims=("x",))
        reference = xr.DataArray(np.array([1.0, np.nan, 1.0, 2.0]), dims=("x",))
        report = sciresult.compare(actual, reference, **ZERO)
        text_snapshot("nan_position_and_policy", format_report(report))

    def test_missing_population(self, text_snapshot: object) -> None:
        actual = xr.DataArray(np.array([np.nan, np.nan]), dims=("x",))
        reference = xr.DataArray(np.array([1.0, np.nan]), dims=("x",))
        report = sciresult.compare(actual, reference, **ZERO)
        assert report.counts is not None and report.counts.finite_pairs == 0
        text_snapshot("missing_population", format_report(report))

    def test_two_dimensional_labels(self, text_snapshot: object) -> None:
        actual = xr.DataArray(
            np.zeros((2, 3)),
            dims=("x", "y"),
            coords={"x": np.array([1.0, 2.0]), "y": np.array([0.0, 10.0, 20.0])},
            name="observed",
        )
        reference = actual.copy()
        reference[1, 1] = 4.0
        report = sciresult.compare(actual, reference, rtol=0.0, atol=0.0)
        text_snapshot("two_dimensional_labels", format_report(report))

    def test_no_labels_available(self, text_snapshot: object) -> None:
        actual = xr.DataArray(np.array([0.0, 0.0, 1.0]), dims=("x",))
        report = sciresult.compare(actual, xr.DataArray(np.zeros(3), dims=("x",)), **ZERO)
        text_snapshot("no_labels", format_report(report))


class TestReportTextInvariants:
    def test_a_missing_population_is_never_reported_as_a_pass(self) -> None:
        # Every pair is nonfinite, so the tolerance relation was never evaluated.
        actual = xr.DataArray(np.array([np.nan, np.inf]), dims=("x",))
        reference = xr.DataArray(np.array([1.0, 1.0]), dims=("x",))
        report = sciresult.compare(actual, reference, **ZERO)
        text = format_report(report)
        assert report.counts is not None and report.counts.finite_pairs == 0
        assert "zero finite pairs" in text
        assert "not evaluated" in text

    def test_a_withheld_scan_says_so_explicitly(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.zeros(2), dims=("x",)),
            xr.DataArray(np.zeros(3), dims=("x",)),
            **ZERO,
        )
        text = format_report(report)
        assert "numerical scan: not performed" in text
        assert "counts: not available" in text
        assert "not performed (structural difference)" in text

    def test_the_scan_is_never_silently_withheld(self) -> None:
        # A dtype difference keeps positions paired, so the scan must still run.
        report = sciresult.compare(
            xr.DataArray(np.array([1.0, 2.0]), dims=("x",)),
            xr.DataArray(np.array([1.0, 2.0], dtype=np.float32), dims=("x",)),
            **ZERO,
        )
        assert report.numeric_scan_complete is True
        assert "numerical scan: complete" in format_report(report)

    def test_a_zero_allowance_score_is_presented_as_intentional(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([1.0]), dims=("x",)),
            xr.DataArray(np.array([0.0]), dims=("x",)),
            **ZERO,
        )
        text = format_report(report)
        assert report.examples[0].score == math.inf
        assert report.examples[0].score_saturated is False
        example_line = next(
            line for line in text.splitlines() if line.strip().startswith("example[1]")
        )
        assert "allowance is zero" in example_line
        assert "score saturated" not in example_line
        assert not any("saturated" in note for note in report.notes)

    def test_an_overflowed_score_is_announced_as_saturated(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([1.0]), dims=("x",)),
            xr.DataArray(np.array([0.0]), dims=("x",)),
            rtol=0.0,
            atol=5e-324,
        )
        text = format_report(report)
        assert report.examples[0].score_saturated is True
        example_line = next(
            line for line in text.splitlines() if line.strip().startswith("example[1]")
        )
        assert "(score saturated at positive infinity)" in example_line
        assert any("not exact-real ranking" in note for note in report.notes)

    def test_many_coordinate_differences_aggregate_into_one_issue(self) -> None:
        # A category, not an element, is a structural issue: 25 differing labels
        # must not produce 25 issues or a 25-line report.
        size = 25
        actual = xr.DataArray(np.zeros(size), dims=("x",), coords={"x": np.arange(float(size))})
        shifted = np.arange(float(size))
        shifted[3] = 999.0
        reference = xr.DataArray(np.zeros(size), dims=("x",), coords={"x": shifted})
        report = sciresult.compare(actual, reference, **ZERO)
        assert report.structural_issue_count == len(report.structural_issues)
        assert {issue.code.value for issue in report.structural_issues} == {
            "coordinate_values",
            "index_values",
        }
        text = format_report(report)
        assert "further structural issue(s) not retained" not in text
        assert sum(
            1 for line in text.splitlines() if line.strip().startswith("structural[")
        ) == len(report.structural_issues)

    def test_the_cap_is_stated_alongside_the_examples(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.arange(10.0)), xr.DataArray(np.zeros(10)), **ZERO, max_examples=2
        )
        assert len(report.examples) == 2
        assert "cap 2" in format_report(report)

    def test_a_truncated_label_is_marked_in_the_text(self) -> None:
        long_text = "a" * 500
        actual = xr.DataArray(
            np.array([1.0, 0.0]), dims=("x",), coords={"x": np.array([long_text, "b"])}
        )
        reference = xr.DataArray(
            np.array([0.0, 0.0]), dims=("x",), coords={"x": np.array([long_text, "b"])}
        )
        report = sciresult.compare(actual, reference, **ZERO)
        label = dict(report.examples[0].labels)["x"]
        assert label.truncated is True
        assert label.original_length == len(long_text)
        assert "[truncated from 500 characters]" in format_report(report)
        assert any("truncated" in note for note in report.notes)

    def test_control_characters_are_escaped_not_emitted_raw(self) -> None:
        awkward = "line\nbreak"
        actual = xr.DataArray(
            np.array([0.0, 1.0]), dims=("x",), coords={"x": np.array(["b", awkward])}
        )
        reference = xr.DataArray(
            np.array([0.0, 0.0]), dims=("x",), coords={"x": np.array(["b", awkward])}
        )
        text = format_report(sciresult.compare(actual, reference, **ZERO))
        label = dict(sciresult.compare(actual, reference, **ZERO).examples[0].labels)["x"]
        assert label.value == awkward  # the report keeps the exact label value
        assert "\n" in text  # the report keeps its own line structure
        assert "line\\x0abreak" in text  # the payload's newline is escaped, not emitted
        assert len(text.splitlines()) < 60

    def test_every_line_is_deterministic_and_printable(self) -> None:
        rng = np.random.default_rng(7)
        actual = rng.normal(size=64)
        reference = actual + rng.normal(scale=0.5, size=64)
        report = sciresult.compare(
            xr.DataArray(actual, dims=("x",)), xr.DataArray(reference, dims=("x",)), **ZERO
        )
        text = format_report(report)
        assert text == format_report(report)
        assert not text.endswith("\n")
        assert text.rstrip().endswith("input mutation was performed.")
        assert "\r" not in text


class TestJsonEncoding:
    def test_json_form_is_strictly_serializable(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([np.nan, 1.0, 2.0]), dims=("x",)),
            xr.DataArray(np.array([0.0, 1.0, 5.0]), dims=("x",)),
            **ZERO,
        )
        payload = sciresult.formatting.report_to_dict(report)
        text = json.dumps(payload, allow_nan=False, sort_keys=True)
        assert json.loads(text) == payload

    def test_json_encodes_labels_with_their_kind_and_unit(self) -> None:
        actual = xr.DataArray(
            np.array([0.0, 1.0]),
            dims=("t",),
            coords={"t": np.array(["2020-01-01", "2020-06-15"], dtype="M8[ns]")},
        )
        reference = xr.DataArray(
            np.array([0.0, 0.0]),
            dims=("t",),
            coords={"t": np.array(["2020-01-01", "2020-06-15"], dtype="M8[ns]")},
        )
        payload = sciresult.formatting.report_to_dict(sciresult.compare(actual, reference, **ZERO))
        label = payload["examples"][0]["labels"]["t"]  # type: ignore[index]
        assert label == {
            "kind": "datetime",
            "ticks": int(np.datetime64("2020-06-15", "ns").astype(np.int64)),
            "unit": "ns",
        }

    def test_json_marks_a_truncated_label(self) -> None:
        long_text = "a" * 500
        actual = xr.DataArray(
            np.array([0.0, 1.0]), dims=("x",), coords={"x": np.array(["b", long_text])}
        )
        reference = xr.DataArray(
            np.array([0.0, 0.0]), dims=("x",), coords={"x": np.array(["b", long_text])}
        )
        payload = sciresult.formatting.report_to_dict(sciresult.compare(actual, reference, **ZERO))
        label = payload["examples"][0]["labels"]["x"]  # type: ignore[index]
        assert label["kind"] == "truncated_string"
        assert label["label_kind"] == "string"
        assert label["original_length"] == 500
        assert len(label["value"]) < 500

    def test_json_omits_the_max_metric_when_there_is_no_finite_pair(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([np.nan, np.inf]), dims=("x",)),
            xr.DataArray(np.array([1.0, -np.inf]), dims=("x",)),
            **ZERO,
        )
        payload = sciresult.formatting.report_to_dict(report)
        assert payload["max_finite_absolute_error"] is None
        assert report.max_finite_absolute_error is None

    def test_json_keeps_the_max_metric_with_its_label(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([1.0, 5.0, 2.0]), dims=("x",)),
            xr.DataArray(np.zeros(3), dims=("x",)),
            rtol=0.0,
            atol=10.0,
        )
        payload = sciresult.formatting.report_to_dict(report)
        metric = payload["max_finite_absolute_error"]
        assert metric is not None
        assert metric["value"] == 5.0
        assert metric["labels"]["x"] == {"kind": "position", "value": 1}

    def test_json_and_text_agree_on_the_status(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array([1.0, 2.0]), dims=("x",)),
            xr.DataArray(np.array([1.0, 3.0]), dims=("x",)),
            **ZERO,
        )
        payload = sciresult.formatting.report_to_dict(report)
        assert payload["status"] == "different"
        assert "status: different" in format_report(report)

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_json_dtype_fields_are_exact(self, dtype: type) -> None:
        report = sciresult.compare(
            xr.DataArray(np.zeros(2, dtype=dtype), dims=("x",)),
            xr.DataArray(np.zeros(2, dtype=dtype), dims=("x",)),
            **ZERO,
        )
        payload = sciresult.formatting.report_to_dict(report)
        assert payload["actual_dtype"] == np.dtype(dtype).str
        assert payload["actual_dtype"] == dtype().dtype.str
