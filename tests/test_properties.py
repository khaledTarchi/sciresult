"""Property-based and differential testing against an independent oracle.

Two independent implementations of the same specification are compared on
randomly generated inputs:

* :mod:`tests.oracles` re-derives every pair classification, count, candidate
  rank, and diagnostic score from the written rules using plain Python scalars;
* :mod:`sciresult` implements them with vectorised NumPy over blocks.

Any disagreement is a defect in one of the two. A case where one implementation
reports an arithmetic overflow and the other does not is treated the same as a
mismatched value: both must refuse the same inputs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import xarray as xr
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import sciresult
from oracles import OracleArithmeticError, oracle_compare

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

# The range is wide enough to exercise cancellation and near-boundary allowances
# while keeping a subtract of two such values representable, so that ordinary
# differential cases reach the comparison instead of the overflow guard. The
# overflow guard is tested separately.
floats = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e150, max_value=1e150, width=64
)
tolerances = st.floats(min_value=0.0, max_value=1e-3, allow_nan=False, allow_infinity=False)
nonfinite_values = st.sampled_from(
    [
        math.nan,
        math.inf,
        -math.inf,
        0.0,
        -0.0,
        1.0,
        -1.0,
        1e300,
        -1e300,
        5e-324,
        -5e-324,
        0.1,
        3.0,
    ]
)
pair_lists = st.lists(floats | nonfinite_values, min_size=1, max_size=24)


def _array(values: list[float], dims: tuple[str, ...] = ("x",)) -> xr.DataArray:
    return xr.DataArray(np.array(values, dtype=np.float64), dims=dims)


def _counts(report: sciresult.ComparisonReport) -> dict[str, int]:
    counts = report.counts
    assert counts is not None
    return {
        "total": counts.total,
        "finite_pairs": counts.finite_pairs,
        "finite_failures": counts.finite_failures,
        "nan_position_mismatches": counts.nan_position_mismatches,
        "matching_nan_pairs": counts.matching_nan_pairs,
        "rejected_nan_pairs": counts.rejected_nan_pairs,
        "matching_infinity_pairs": counts.matching_infinity_pairs,
        "infinity_mismatches": counts.infinity_mismatches,
    }


def _library_report(
    actual: list[float],
    reference: list[float],
    rtol: float,
    atol: float,
    equal_nan: bool,
    **extra: Any,
) -> sciresult.ComparisonReport | None:
    """Return the report, or ``None`` when the library refuses the arithmetic."""
    try:
        return sciresult.compare(
            _array(actual), _array(reference), rtol=rtol, atol=atol, equal_nan=equal_nan, **extra
        )
    except sciresult.ComparisonArithmeticError:
        return None


def _oracle_result(
    actual: list[float], reference: list[float], rtol: float, atol: float, equal_nan: bool
) -> tuple[Any, tuple[Any, ...]] | None:
    """Return ``(result, all_candidates)``, or ``None`` when the oracle refuses.

    The oracle deliberately keeps every candidate so it can be checked against a
    full sort; the library is allowed to retain only ``max_examples`` of them.
    """
    try:
        return oracle_compare(actual, reference, rtol=rtol, atol=atol, equal_nan=equal_nan)
    except OracleArithmeticError:
        return None


class TestDifferentialAgainstOracle:
    @given(
        data=st.data(), actual=pair_lists, rtol=tolerances, atol=tolerances, equal_nan=st.booleans()
    )
    @SETTINGS
    def test_counts_match_an_independent_implementation(
        self, data: Any, actual: list[float], rtol: float, atol: float, equal_nan: bool
    ) -> None:
        reference = data.draw(pair_lists, label="reference")
        assumed = min(len(actual), len(reference))
        actual, reference = actual[:assumed], reference[:assumed]
        report = _library_report(actual, reference, rtol, atol, equal_nan)
        outcome = _oracle_result(actual, reference, rtol, atol, equal_nan)
        assert (report is None) == (outcome is None)
        if report is None or outcome is None:
            return
        expected, _candidates = outcome
        observed = _counts(report)
        for name, value in expected.counts.items():
            assert observed[name] == value, f"{name}: {observed[name]} != {value}"

    @given(
        data=st.data(), actual=pair_lists, rtol=tolerances, atol=tolerances, equal_nan=st.booleans()
    )
    @SETTINGS
    def test_status_and_examples_match_an_independent_implementation(
        self, data: Any, actual: list[float], rtol: float, atol: float, equal_nan: bool
    ) -> None:
        reference = data.draw(pair_lists, label="reference")
        assumed = min(len(actual), len(reference))
        actual, reference = actual[:assumed], reference[:assumed]
        report = _library_report(actual, reference, rtol, atol, equal_nan)
        outcome = _oracle_result(actual, reference, rtol, atol, equal_nan)
        assert (report is None) == (outcome is None)
        if report is None or outcome is None:
            return
        expected, candidates = outcome
        total_failures = (
            expected.counts["finite_failures"]
            + expected.counts["nan_position_mismatches"]
            + expected.counts["rejected_nan_pairs"]
            + expected.counts["infinity_mismatches"]
        )
        assert report.status.value == ("different" if total_failures else "match")
        # The report may retain only its cap; the ranking must be the full-sort
        # prefix, not a different order of the same examples.
        assert [(example.kind.value, example.flat_index) for example in report.examples] == [
            (candidate.kind, candidate.flat_index)
            for candidate in candidates[: report.max_examples]
        ]

    @given(
        data=st.data(), actual=pair_lists, rtol=tolerances, atol=tolerances, equal_nan=st.booleans()
    )
    @SETTINGS
    def test_diagnostic_scores_match_an_independent_implementation(
        self, data: Any, actual: list[float], rtol: float, atol: float, equal_nan: bool
    ) -> None:
        reference = data.draw(pair_lists, label="reference")
        assumed = min(len(actual), len(reference))
        actual, reference = actual[:assumed], reference[:assumed]
        report = _library_report(actual, reference, rtol, atol, equal_nan)
        outcome = _oracle_result(actual, reference, rtol, atol, equal_nan)
        assert (report is None) == (outcome is None)
        if report is None or outcome is None:
            return
        by_index = {candidate.flat_index: candidate for candidate in outcome[1]}
        for example in report.examples:
            oracle = by_index[example.flat_index]
            if example.kind.value != "finite_tolerance":
                # A nonfinite category is never scored against the relation.
                assert example.score is None
                assert example.score_saturated is False
                continue
            assert example.score_saturated is oracle.score_saturated
            if example.score_saturated:
                assert example.score == math.inf
            else:
                assert example.score == pytest.approx(oracle.score, rel=1e-12, abs=0.0)

    @given(
        data=st.data(), actual=pair_lists, rtol=tolerances, atol=tolerances, equal_nan=st.booleans()
    )
    @SETTINGS
    def test_max_absolute_error_matches_an_independent_implementation(
        self, data: Any, actual: list[float], rtol: float, atol: float, equal_nan: bool
    ) -> None:
        reference = data.draw(pair_lists, label="reference")
        assumed = min(len(actual), len(reference))
        actual, reference = actual[:assumed], reference[:assumed]
        report = _library_report(actual, reference, rtol, atol, equal_nan)
        outcome = _oracle_result(actual, reference, rtol, atol, equal_nan)
        assert (report is None) == (outcome is None)
        if report is None or outcome is None:
            return
        expected, _candidates = outcome
        if expected.max_absolute_value is None:
            assert report.max_finite_absolute_error is None
        else:
            metric = report.max_finite_absolute_error
            assert metric is not None
            # The maximum spans passing pairs too, so it is the largest finite
            # discrepancy, not the largest retained failure.
            assert metric.value == pytest.approx(expected.max_absolute_value, rel=1e-12, abs=0.0)
            assert metric.flat_index == expected.max_absolute_index
            assert metric.index == (expected.max_absolute_index,)

    @given(
        data=st.data(),
        actual=pair_lists,
        rtol=tolerances,
        atol=tolerances,
        block_size=st.sampled_from([1, 2, 3, 8, 64, 1024]),
    )
    @SETTINGS
    def test_arithmetic_overflow_agrees_with_the_oracle(
        self, data: Any, actual: list[float], rtol: float, atol: float, block_size: int
    ) -> None:
        reference = data.draw(pair_lists, label="reference")
        assumed = min(len(actual), len(reference))
        actual, reference = actual[:assumed], reference[:assumed]
        sciresult.numeric.BLOCK_SIZE = block_size
        try:
            report = _library_report(actual, reference, rtol, atol, False)
        finally:
            sciresult.numeric.BLOCK_SIZE = 65536
        outcome = _oracle_result(actual, reference, rtol, atol, False)
        # Both implementations must refuse the same inputs, not merely agree on
        # the values they can represent.
        assert (report is None) == (outcome is None)
        if report is not None and outcome is not None:
            assert _counts(report) == outcome[0].counts


class TestInvariants:
    @given(
        data=st.data(), actual=pair_lists, rtol=tolerances, atol=tolerances, equal_nan=st.booleans()
    )
    @SETTINGS
    def test_every_pair_is_classified_exactly_once(
        self, data: Any, actual: list[float], rtol: float, atol: float, equal_nan: bool
    ) -> None:
        reference = data.draw(pair_lists, label="reference")
        assumed = min(len(actual), len(reference))
        actual, reference = actual[:assumed], reference[:assumed]
        report = _library_report(actual, reference, rtol, atol, equal_nan)
        assume(report is not None)
        assert report is not None and report.counts is not None
        counts = _counts(report)
        classified = (
            counts["finite_pairs"]
            + counts["nan_position_mismatches"]
            + counts["matching_nan_pairs"]
            + counts["rejected_nan_pairs"]
            + counts["matching_infinity_pairs"]
            + counts["infinity_mismatches"]
        )
        assert classified == counts["total"] == len(actual)

    @given(
        data=st.data(),
        actual=pair_lists,
        rtol=tolerances,
        atol=tolerances,
        max_examples=st.integers(min_value=1, max_value=8),
    )
    @SETTINGS
    def test_the_cap_bounds_storage_without_changing_the_verdict(
        self, data: Any, actual: list[float], rtol: float, atol: float, max_examples: int
    ) -> None:
        reference = data.draw(pair_lists, label="reference")
        assumed = min(len(actual), len(reference))
        actual, reference = actual[:assumed], reference[:assumed]
        bounded = _library_report(actual, reference, rtol, atol, False, max_examples=max_examples)
        unbounded = _library_report(actual, reference, rtol, atol, False, max_examples=100)
        assume(bounded is not None and unbounded is not None)
        assert bounded is not None and unbounded is not None
        assert bounded.status is unbounded.status
        assert _counts(bounded) == _counts(unbounded)
        assert len(bounded.examples) <= max_examples
        # Retention is a prefix of the full ranking, never a reshuffle.
        assert (
            bounded.to_dict()["examples"]
            == unbounded.to_dict()["examples"][: len(bounded.examples)]
        )

    @given(
        values=st.lists(floats, min_size=1, max_size=20),
        rtol=tolerances,
        atol=st.floats(min_value=1e-9, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @SETTINGS
    def test_the_tolerance_relation_is_inclusive_at_the_boundary(
        self, values: list[float], rtol: float, atol: float
    ) -> None:
        # Against a zero reference the allowance is exactly atol, so a discrepancy
        # of exactly atol must pass: the relation uses <=, not <.
        report = _library_report([*values, atol], [*values, 0.0], rtol, atol, False)
        assume(report is not None)
        assert report is not None and report.counts is not None
        assert report.counts.finite_failures == 0
        assert report.status.value == "match"

    @given(
        values=st.lists(floats, min_size=1, max_size=20),
        rtol=tolerances,
        atol=st.floats(min_value=1e-9, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @SETTINGS
    def test_a_discrepancy_just_past_the_boundary_fails(
        self, values: list[float], rtol: float, atol: float
    ) -> None:
        assume(atol > 1e-6)
        report = _library_report(
            [*values, math.nextafter(atol, math.inf)], [*values, 0.0], rtol, atol, False
        )
        assume(report is not None)
        assert report is not None and report.counts is not None
        assert report.counts.finite_failures == 1

    @given(
        values=st.lists(floats, min_size=1, max_size=20),
        rtol=tolerances,
        atol=tolerances,
        equal_nan=st.booleans(),
    )
    @SETTINGS
    def test_matched_nan_needs_the_nan_policy(
        self, values: list[float], rtol: float, atol: float, equal_nan: bool
    ) -> None:
        paired = [*values, math.nan]
        report = _library_report(paired, list(paired), rtol, atol, equal_nan)
        assume(report is not None)
        assert report is not None
        assert report.status.value == ("match" if equal_nan else "different")
        if not equal_nan:
            assert report.examples[0].kind.value == "nan_policy"

    @given(
        values=st.lists(floats, min_size=1, max_size=20),
        rtol=tolerances,
        atol=tolerances,
        equal_nan=st.booleans(),
    )
    @SETTINGS
    def test_signed_zero_is_never_a_failure(
        self, values: list[float], rtol: float, atol: float, equal_nan: bool
    ) -> None:
        report = _library_report([*values, 0.0], [*values, -0.0], rtol, atol, equal_nan)
        assume(report is not None)
        assert report is not None
        assert report.status.value == "match"

    @given(
        values=st.lists(floats, min_size=1, max_size=20),
        rtol=tolerances,
        atol=tolerances,
        equal_nan=st.booleans(),
    )
    @SETTINGS
    def test_identical_finite_input_matches(
        self, values: list[float], rtol: float, atol: float, equal_nan: bool
    ) -> None:
        report = _library_report(values, list(values), rtol, atol, equal_nan)
        assume(report is not None)
        assert report is not None
        assert report.status.value == "match"
        assert report.counts is not None
        assert report.counts.total_failures == 0
        assert report.examples == ()

    @given(values=st.lists(floats, min_size=1, max_size=16), rtol=tolerances, atol=tolerances)
    @SETTINGS
    def test_memory_layout_does_not_change_the_verdict(
        self, values: list[float], rtol: float, atol: float
    ) -> None:
        base = np.array(values)
        other = base + np.array(values)[::-1].copy() * 1e-6
        baseline = _library_report(values, other.tolist(), rtol, atol, False)
        assume(baseline is not None)
        assert baseline is not None
        baseline_counts = _counts(baseline)
        variants = {
            "fortran": (np.asfortranarray(base), np.asfortranarray(other)),
            "reversed": (base[::-1], other[::-1]),
            "noncontiguous": (base[::1], other[::1]),
        }
        for name, (left, right) in variants.items():
            shaped = sciresult.compare(
                xr.DataArray(left, dims=("x",)),
                xr.DataArray(right, dims=("x",)),
                rtol=rtol,
                atol=atol,
            )
            assert _counts(shaped) == baseline_counts, f"layout {name} changed the counts"

    @given(values=st.lists(floats, min_size=4, max_size=24), rtol=tolerances, atol=tolerances)
    @SETTINGS
    def test_a_strided_slice_agrees_with_a_contiguous_copy(
        self, values: list[float], rtol: float, atol: float
    ) -> None:
        assume(len(values) >= 4)
        left = np.array(values)[::2]
        right = (np.array(values) + 1e-6)[1::2]
        assume(left.shape == right.shape)
        strided = sciresult.compare(
            xr.DataArray(left, dims=("x",)), xr.DataArray(right, dims=("x",)), rtol=rtol, atol=atol
        )
        copied = sciresult.compare(
            xr.DataArray(np.ascontiguousarray(left), dims=("x",)),
            xr.DataArray(np.ascontiguousarray(right), dims=("x",)),
            rtol=rtol,
            atol=atol,
        )
        assert _counts(strided) == _counts(copied)


class TestMultiDimensionalPositions:
    @given(values=st.lists(floats, min_size=4, max_size=36), rtol=tolerances, atol=tolerances)
    @SETTINGS
    def test_positions_unravel_consistently(
        self, values: list[float], rtol: float, atol: float
    ) -> None:
        side = math.isqrt(len(values))
        assume(side * side == len(values))
        actual = np.array(values).reshape(side, side)
        reference = actual.copy()
        reference[side - 1, side - 1] = math.nextafter(float(actual[side - 1, side - 1]), math.inf)
        report = sciresult.compare(
            xr.DataArray(actual, dims=("y", "x")),
            xr.DataArray(reference, dims=("y", "x")),
            rtol=rtol,
            atol=atol,
        )
        if report.status.value == "match":
            return
        example = report.examples[0]
        assert example.index == (side - 1, side - 1)
        assert example.flat_index == len(values) - 1
        assert example.actual == float(actual[side - 1, side - 1])
        assert example.reference == float(reference[side - 1, side - 1])

    @given(values=st.lists(floats, min_size=6, max_size=36), rtol=tolerances, atol=tolerances)
    @SETTINGS
    def test_flat_index_is_row_major_over_the_declared_dimensions(
        self, values: list[float], rtol: float, atol: float
    ) -> None:
        assume(len(values) >= 6)
        actual = np.array(values[:6]).reshape(2, 3)
        reference = actual.copy()
        row, column = 1, 2
        reference[row, column] = math.nextafter(float(actual[row, column]), math.inf)
        report = sciresult.compare(
            xr.DataArray(actual, dims=("a", "b")),
            xr.DataArray(reference, dims=("a", "b")),
            rtol=rtol,
            atol=atol,
        )
        if report.status.value == "match":
            return
        example = report.examples[0]
        assert example.index == (row, column)
        # C order: the last element of a 2x3 array has flat index 5.
        assert example.flat_index == row * 3 + column
