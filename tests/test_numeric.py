"""Numerical relation semantics.

The relation under test is, for a finite pair ``(a, b)``:

    abs(a - b) <= atol + rtol * abs(b)

evaluated with separate binary64 operations under
``np.errstate(over="raise", invalid="raise", under="ignore")``. Nonfinite pairs
are classified *without* evaluating that relation, and the categories partition
the population exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

import oracles
import sciresult
from sciresult import ComparisonStatus, FailureKind
from sciresult.model import element_count

ZERO = {"rtol": 0.0, "atol": 0.0}


def _array(values: object, dtype: object = np.float64) -> xr.DataArray:
    return xr.DataArray(np.array(values, dtype=dtype), dims=("x",))


def _counts(report: sciresult.ComparisonReport) -> sciresult.PairCounts:
    assert report.counts is not None
    return report.counts


class TestToleranceRelation:
    def test_identical_arrays_match(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        report = sciresult.compare(_array(values), _array(values), **ZERO)
        assert report.status is ComparisonStatus.MATCH
        assert _counts(report).total_failures == 0
        assert _counts(report).finite_pairs == 3

    def test_absolute_tolerance_is_honored(self) -> None:
        report = sciresult.compare(_array([1.0, 1.05]), _array([1.0, 1.0]), rtol=0.0, atol=0.1)
        assert report.status is ComparisonStatus.MATCH

    def test_relative_tolerance_is_honored(self) -> None:
        # |1 - 100| = 99, allowance = 0 + 0.01 * 100 = 1 -> fails.
        report = sciresult.compare(_array([1.0]), _array([100.0]), rtol=0.01, atol=0.0)
        assert report.status is ComparisonStatus.DIFFERENT
        # |100 - 100| = 0 -> passes at any tolerance.
        report = sciresult.compare(_array([100.0]), _array([100.0]), rtol=0.01, atol=0.0)
        assert report.status is ComparisonStatus.MATCH

    def test_allowance_is_atol_plus_rtol_times_reference_magnitude(self) -> None:
        # 0.01 * 200 = 2, plus atol 1.5 = 3.5. A difference of exactly 3.5 passes.
        at_boundary = sciresult.compare(_array([203.5]), _array([200.0]), rtol=0.01, atol=1.5)
        assert at_boundary.status is ComparisonStatus.MATCH
        just_over = sciresult.compare(_array([203.5000001]), _array([200.0]), rtol=0.01, atol=1.5)
        assert just_over.status is ComparisonStatus.DIFFERENT

    def test_relative_term_uses_the_reference_not_the_actual(self) -> None:
        # allowance uses |b|. With a=1, b=1000, rtol=0.001: allowance 1, error 999.
        report = sciresult.compare(_array([1.0]), _array([1000.0]), rtol=0.001, atol=0.0)
        assert report.status is ComparisonStatus.DIFFERENT
        example = report.examples[0]
        assert example.allowed_error == pytest.approx(1.0)
        assert example.absolute_error == pytest.approx(999.0)

    def test_zero_tolerance_requires_exact_equality(self) -> None:
        # The next representable binary64 above 1.0.
        report = sciresult.compare(_array([1.0]), _array([float(np.nextafter(1.0, 2.0))]), **ZERO)
        assert report.status is ComparisonStatus.DIFFERENT

    def test_a_difference_below_the_ulp_is_not_a_difference(self) -> None:
        # 1.0 + 1e-300 is not representable and rounds back to 1.0. Reporting a
        # failure here would be a fabricated defect.
        report = sciresult.compare(_array([1.0]), _array([1.0 + 1e-300]), **ZERO)
        assert report.status is ComparisonStatus.MATCH

    def test_smallest_subnormal_difference_is_real(self) -> None:
        # The difference of two binary64 values is never a nonzero quantity below the
        # smallest subnormal, so 5e-324 is a genuine, reportable difference.
        report = sciresult.compare(_array([0.0]), _array([5e-324]), **ZERO)
        assert report.status is ComparisonStatus.DIFFERENT

    def test_signed_zeros_are_equal(self) -> None:
        report = sciresult.compare(_array([-0.0]), _array([0.0]), **ZERO)
        assert report.status is ComparisonStatus.MATCH
        assert _counts(report).finite_failures == 0

    def test_underflow_in_the_allowance_is_ignored_not_raised(self) -> None:
        # rtol * |b| = 1e-330 is below the smallest subnormal and rounds to zero.
        # Underflow is deliberately not an error, so this must succeed, not raise.
        report = sciresult.compare(_array([0.0]), _array([1e-30]), rtol=1e-300, atol=1e-30)
        assert report.status is ComparisonStatus.MATCH


class TestNonfiniteClassification:
    def test_nan_on_exactly_one_side_is_a_nan_position_mismatch(self) -> None:
        report = sciresult.compare(_array([np.nan, 1.0]), _array([1.0, 1.0]), rtol=0.0, atol=0.0)
        counts = _counts(report)
        assert counts.nan_position_mismatches == 1
        assert counts.finite_pairs == 1
        assert report.status is ComparisonStatus.DIFFERENT
        assert report.examples[0].kind is FailureKind.NAN_POSITION

    def test_nan_on_both_sides_is_a_policy_decision(self) -> None:
        rejecting = sciresult.compare(
            _array([np.nan]), _array([np.nan]), rtol=0.0, atol=0.0, equal_nan=False
        )
        assert _counts(rejecting).rejected_nan_pairs == 1
        assert _counts(rejecting).matching_nan_pairs == 0
        assert rejecting.examples[0].kind is FailureKind.NAN_POLICY

        accepting = sciresult.compare(
            _array([np.nan]), _array([np.nan]), rtol=0.0, atol=0.0, equal_nan=True
        )
        assert _counts(accepting).matching_nan_pairs == 1
        assert _counts(accepting).rejected_nan_pairs == 0
        assert accepting.status is ComparisonStatus.MATCH

    def test_matching_infinities_are_a_match(self) -> None:
        report = sciresult.compare(
            _array([np.inf, -np.inf]), _array([np.inf, -np.inf]), rtol=0.0, atol=0.0
        )
        counts = _counts(report)
        assert counts.matching_infinity_pairs == 2
        assert counts.infinity_mismatches == 0
        assert report.status is ComparisonStatus.MATCH

    def test_opposite_infinities_are_an_infinity_mismatch(self) -> None:
        report = sciresult.compare(_array([np.inf]), _array([-np.inf]), **ZERO)
        assert _counts(report).infinity_mismatches == 1
        assert report.examples[0].kind is FailureKind.INFINITY

    def test_infinity_against_finite_is_an_infinity_mismatch(self) -> None:
        report = sciresult.compare(_array([np.inf]), _array([1e308]), rtol=1.0, atol=1e308)
        assert _counts(report).infinity_mismatches == 1
        assert _counts(report).finite_pairs == 0

    @pytest.mark.parametrize("nan_side", ["actual", "reference"])
    def test_nan_against_infinity_is_classified_only_as_nan_position(self, nan_side: str) -> None:
        # Regression test: the NaN rule is evaluated first, so a pair that also
        # involves an infinity must be counted exactly once, not twice.
        actual = [np.nan, 0.0] if nan_side == "actual" else [np.inf, 0.0]
        reference = [np.inf, 0.0] if nan_side == "actual" else [np.nan, 0.0]
        report = sciresult.compare(_array(actual), _array(reference), **ZERO)
        counts = _counts(report)
        assert counts.nan_position_mismatches == 1
        assert counts.infinity_mismatches == 0
        assert counts.total == 2
        assert counts.total_failures == 1
        assert report.examples[0].kind is FailureKind.NAN_POSITION

    def test_nan_against_nan_with_equal_nan_true_is_not_an_infinity(self) -> None:
        report = sciresult.compare(
            _array([np.nan]), _array([np.nan]), rtol=0.0, atol=0.0, equal_nan=True
        )
        counts = _counts(report)
        assert counts.matching_nan_pairs == 1
        assert counts.infinity_mismatches == 0

    def test_nonfinite_pairs_are_never_evaluated_against_the_tolerance(self) -> None:
        # An enormous allowance must not rescue an infinity mismatch.
        report = sciresult.compare(
            _array([np.inf]),
            _array([0.0]),
            rtol=1e300,
            atol=1e300,
            equal_nan=True,
        )
        counts = _counts(report)
        assert counts.infinity_mismatches == 1
        assert counts.finite_pairs == 0
        assert counts.total == 1

    def test_categories_are_mutually_exclusive_and_exhaustive(self) -> None:
        actual = [np.nan, np.nan, np.inf, -np.inf, np.inf, 1.0, 2.0, 3.0, 4.0]
        reference = [np.nan, 1.0, np.inf, -np.inf, 1.0, 1.0, 2.0, 5.0, 4.0]
        report = sciresult.compare(
            _array(actual), _array(reference), rtol=0.0, atol=0.0, equal_nan=True
        )
        counts = _counts(report)
        assert counts.total == len(actual)
        accounted = (
            counts.finite_pairs
            + counts.nan_position_mismatches
            + counts.matching_nan_pairs
            + counts.rejected_nan_pairs
            + counts.matching_infinity_pairs
            + counts.infinity_mismatches
        )
        assert accounted == counts.total
        assert counts.finite_pairs == 4
        assert counts.finite_failures == 1
        assert counts.matching_nan_pairs == 1
        assert counts.nan_position_mismatches == 1
        assert counts.matching_infinity_pairs == 2
        assert counts.infinity_mismatches == 1
        assert counts.total_failures == 3


class TestArithmeticErrors:
    def test_overflowing_subtraction_raises(self) -> None:
        with pytest.raises(sciresult.ComparisonArithmeticError) as info:
            sciresult.compare(_array([1e308]), _array([-1e308]), **ZERO)
        assert info.value.code == "finite_arithmetic_overflow"

    def test_overflowing_allowance_raises(self) -> None:
        # rtol * |b| overflows even though both operands are finite.
        with pytest.raises(sciresult.ComparisonArithmeticError):
            sciresult.compare(_array([1e308]), _array([1e308]), rtol=1e300, atol=1e300)

    def test_arithmetic_error_is_not_an_assertion_mismatch(self) -> None:
        with pytest.raises(sciresult.ComparisonArithmeticError):
            sciresult.assert_close(_array([1e308]), _array([-1e308]), **ZERO)
        assert not issubclass(sciresult.ComparisonArithmeticError, AssertionError)

    def test_numpy_error_state_is_restored_after_an_arithmetic_failure(self) -> None:
        before = np.geterr()
        with pytest.raises(sciresult.ComparisonArithmeticError):
            sciresult.compare(_array([1e308]), _array([-1e308]), **ZERO)
        assert np.geterr() == before

    def test_numpy_error_state_is_restored_after_success(self) -> None:
        before = np.geterr()
        sciresult.compare(_array([1.0]), _array([1.0]), **ZERO)
        assert np.geterr() == before


class TestMaxAbsoluteError:
    def test_maximum_covers_passing_pairs_too(self) -> None:
        # The largest absolute error sits on a pair that PASSES under a loose tolerance.
        report = sciresult.compare(_array([1.0, 100.0]), _array([0.0, 0.0]), rtol=0.0, atol=200.0)
        assert report.status is ComparisonStatus.MATCH
        metric = report.max_finite_absolute_error
        assert metric is not None
        assert metric.value == pytest.approx(100.0)
        assert metric.index == (1,)

    def test_ties_are_broken_by_the_lowest_flat_index(self) -> None:
        report = sciresult.compare(
            _array([5.0, 5.0, 5.0]), _array([0.0, 0.0, 0.0]), rtol=0.0, atol=10.0
        )
        metric = report.max_finite_absolute_error
        assert metric is not None
        assert metric.index == (0,)

    def test_multi_dimensional_index_is_reported_per_axis(self) -> None:
        values = np.zeros((2, 3))
        other = values.copy()
        other[1, 2] = 7.0
        report = sciresult.compare(
            xr.DataArray(values, dims=("x", "y")),
            xr.DataArray(other, dims=("x", "y")),
            rtol=0.0,
            atol=100.0,
        )
        metric = report.max_finite_absolute_error
        assert metric is not None
        assert metric.index == (1, 2)
        assert metric.flat_index == 5

    def test_metric_is_none_when_there_are_no_finite_pairs(self) -> None:
        report = sciresult.compare(_array([np.nan]), _array([np.nan]), rtol=0.0, atol=0.0)
        assert report.max_finite_absolute_error is None


class TestDiagnosticRanking:
    def test_nan_position_outranks_nan_policy(self) -> None:
        report = sciresult.compare(
            _array([np.nan, np.nan]), _array([np.nan, 1.0]), rtol=0.0, atol=0.0
        )
        kinds = [example.kind for example in report.examples]
        assert kinds[0] is FailureKind.NAN_POSITION
        assert FailureKind.NAN_POLICY in kinds

    def test_nan_policy_outranks_infinity(self) -> None:
        report = sciresult.compare(
            _array([np.nan, np.inf]), _array([np.nan, -np.inf]), rtol=0.0, atol=0.0
        )
        kinds = [example.kind for example in report.examples]
        assert kinds.index(FailureKind.NAN_POLICY) < kinds.index(FailureKind.INFINITY)

    def test_infinity_outranks_finite_tolerance(self) -> None:
        report = sciresult.compare(_array([np.inf, 1.0]), _array([0.0, 2.0]), rtol=0.0, atol=0.0)
        kinds = [example.kind for example in report.examples]
        assert kinds.index(FailureKind.INFINITY) < kinds.index(FailureKind.FINITE_TOLERANCE)

    def test_larger_score_ranks_first_within_a_category(self) -> None:
        # atol = 1 keeps the scores finite so the ordering is the exact-real ordering.
        report = sciresult.compare(
            _array([1.0, 10.0, 5.0]), _array([0.0, 0.0, 0.0]), rtol=0.0, atol=1.0
        )
        scores = [example.score for example in report.examples]
        assert all(value is not None and math.isfinite(value) for value in scores)
        # Position 0 has an absolute error of 1 and an allowance of 1, so it passes and
        # is not an example. `score` is the dimensionless ratio error / allowance:
        # 10/1 and 5/1.
        assert scores == [10.0, 5.0]
        assert [example.index for example in report.examples] == [(1,), (2,)]
        assert _counts(report).finite_failures == 2
        assert _counts(report).finite_pairs == 3

    def test_equal_scores_rank_by_lowest_flat_index_first(self) -> None:
        report = sciresult.compare(_array([1.0, 1.0, 1.0]), _array([0.0, 0.0, 0.0]), **ZERO)
        indices = [example.index for example in report.examples]
        assert indices == [(0,), (1,), (2,)]

    def test_zero_allowance_gives_an_intentional_infinity(self) -> None:
        report = sciresult.compare(_array([1.0]), _array([0.0]), **ZERO)
        example = report.examples[0]
        assert example.score == math.inf
        assert example.allowed_error == 0.0
        # Not saturation: the exact-real ratio is infinite, so these examples tie
        # in exact-real ranking exactly as they tie here.
        assert example.score_saturated is False
        assert not any("saturated" in note for note in report.notes)

    def test_overflowed_quotient_saturates_the_score(self) -> None:
        report = sciresult.compare(_array([1.0]), _array([0.0]), rtol=0.0, atol=5e-324)
        example = report.examples[0]
        assert example.allowed_error == 5e-324
        assert example.score == math.inf
        assert example.score_saturated is True
        assert any("saturated" in note for note in report.notes)

    def test_max_examples_caps_examples_without_changing_counts(self) -> None:
        actual = _array(list(range(1, 21)))
        reference = _array([0.0] * 20)
        full = sciresult.compare(actual, reference, rtol=0.0, atol=0.0, max_examples=20)
        capped = sciresult.compare(actual, reference, rtol=0.0, atol=0.0, max_examples=3)
        assert len(full.examples) == 20
        assert len(capped.examples) == 3
        assert capped.counts == full.counts
        assert capped.status is full.status
        assert capped.max_finite_absolute_error == full.max_finite_absolute_error

    def test_max_examples_zero_still_reports_complete_counts(self) -> None:
        actual = _array(list(range(1, 21)))
        reference = _array([0.0] * 20)
        report = sciresult.compare(actual, reference, rtol=0.0, atol=0.0, max_examples=0)
        assert report.examples == ()
        assert _counts(report).finite_failures == 20


class TestBlockSizeInvariance:
    @pytest.mark.parametrize("block_size", [1, 2, 3, 7, 64, 65_536])
    def test_counts_are_identical_across_block_sizes(
        self, block_size: int, set_block_size: Callable[[int], None]
    ) -> None:
        rng = np.random.default_rng(20240607)
        actual = rng.normal(size=257)
        reference = actual + rng.normal(scale=1e-3, size=257)
        # Plant one rejected NaN pair, one NaN-position mismatch, one matching
        # infinity, and two infinity mismatches.
        actual[0] = np.nan
        reference[0] = np.nan
        actual[1] = np.nan
        reference[1] = 0.0
        actual[2] = -np.inf
        reference[2] = -np.inf
        actual[3] = np.inf
        reference[3] = 0.0
        actual[4] = np.inf
        reference[4] = 0.0

        set_block_size(block_size)
        report = sciresult.compare(
            _array(actual), _array(reference), rtol=1e-3, atol=1e-6, equal_nan=False
        )
        counts = _counts(report)
        assert counts.total == 257
        assert counts.total_failures > 0
        assert counts.rejected_nan_pairs == 1
        assert counts.nan_position_mismatches == 1
        assert counts.matching_infinity_pairs == 1
        assert counts.infinity_mismatches == 2
        accounted = (
            counts.finite_pairs
            + counts.nan_position_mismatches
            + counts.matching_nan_pairs
            + counts.rejected_nan_pairs
            + counts.matching_infinity_pairs
            + counts.infinity_mismatches
        )
        assert accounted == counts.total

    def test_ranking_is_identical_across_block_sizes(
        self, set_block_size: Callable[[int], None]
    ) -> None:
        values = np.array([1.0, 5.0, 1.0, 5.0, 3.0])
        reference = np.zeros(5)
        seen: set[tuple[tuple[int, ...], ...]] = set()
        for block_size in (1, 2, 3, 5, 64):
            set_block_size(block_size)
            report = sciresult.compare(_array(values), _array(reference), rtol=0.0, atol=0.0)
            seen.add(tuple(example.index for example in report.examples))
        assert len(seen) == 1


class TestMultiDimensionalAndScalar:
    def test_scalar_arrays_are_supported(self) -> None:
        report = sciresult.compare(
            xr.DataArray(np.array(1.0)),
            xr.DataArray(np.array(1.0 + 1e-12)),
            rtol=1e-6,
            atol=0.0,
        )
        assert report.status is ComparisonStatus.MATCH
        assert report.actual_shape == ()
        metric = report.max_finite_absolute_error
        assert metric is not None
        assert metric.index == ()

    def test_zero_dimensional_failure_reports_an_empty_index(self) -> None:
        report = sciresult.compare(xr.DataArray(np.array(1.0)), xr.DataArray(np.array(2.0)), **ZERO)
        assert report.status is ComparisonStatus.DIFFERENT
        assert report.examples[0].index == ()

    def test_counts_total_matches_element_count(self) -> None:
        values = np.zeros((3, 4, 5))
        report = sciresult.compare(
            xr.DataArray(values, dims=("a", "b", "c")),
            xr.DataArray(values + 1.0, dims=("a", "b", "c")),
            **ZERO,
        )
        assert _counts(report).total == element_count((3, 4, 5)) == 60


class TestFloat32Inputs:
    def test_float32_arrays_are_compared_in_binary64(self) -> None:
        report = sciresult.compare(
            _array([1.0, 2.0], dtype=np.float32),
            _array([1.0, 2.0], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
        assert report.status is ComparisonStatus.MATCH
        assert report.actual_dtype == "<f4"

    def test_float32_rounding_shows_up_as_a_real_difference(self) -> None:
        # The float32 literal and the binary64 literal are different numbers.
        report = sciresult.compare(
            _array([0.1], dtype=np.float32), _array([0.1], dtype=np.float64), **ZERO
        )
        assert report.status is ComparisonStatus.DIFFERENT


class TestDifferentialAgainstOracle:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_counts_match_the_scalar_oracle(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        # Large magnitudes stay below the point where a difference would overflow: an
        # overflow is a separate documented outcome, not a comparison result.
        pool = np.array(
            [0.0, -0.0, 1.0, -1.0, 1e200, -1e200, 5e-324, np.nan, np.inf, -np.inf],
            dtype=np.float64,
        )
        actual = rng.choice(pool, size=200)
        reference = rng.choice(pool, size=200)
        rtol, atol = 1e-9, 1e-12

        report = sciresult.compare(
            _array(actual), _array(reference), rtol=rtol, atol=atol, equal_nan=False
        )
        expected = oracles.oracle_pair_counts(
            [float(value) for value in actual],
            [float(value) for value in reference],
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
        counts = _counts(report)
        assert counts.total == expected.counts["total"]
        assert counts.finite_pairs == expected.counts["finite_pairs"]
        assert counts.finite_failures == expected.counts["finite_failures"]
        assert counts.nan_position_mismatches == expected.counts["nan_position_mismatches"]
        assert counts.rejected_nan_pairs == expected.counts["rejected_nan_pairs"]
        assert counts.matching_nan_pairs == expected.counts["matching_nan_pairs"]
        assert counts.matching_infinity_pairs == expected.counts["matching_infinity_pairs"]
        assert counts.infinity_mismatches == expected.counts["infinity_mismatches"]
        assert (report.status is ComparisonStatus.MATCH) == expected.is_match()

    @pytest.mark.parametrize("seed", [11, 12, 13])
    def test_example_ranking_matches_the_oracle(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        actual = rng.normal(size=300)
        reference = actual + rng.normal(scale=0.05, size=300)
        actual[rng.integers(0, 300, 10)] = np.nan
        reference[rng.integers(0, 300, 10)] = np.nan
        rtol, atol = 1e-6, 1e-9

        report = sciresult.compare(
            _array(actual),
            _array(reference),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            max_examples=10,
        )
        _, every_candidate = oracles.oracle_compare(
            [float(value) for value in actual],
            [float(value) for value in reference],
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        expected_top = every_candidate[: len(report.examples)]
        assert len(report.examples) == len(expected_top)
        for produced, expected in zip(report.examples, expected_top, strict=True):
            assert produced.flat_index == expected.flat_index
            assert produced.kind.value == expected.kind
            assert produced.score_saturated is expected.score_saturated
            if expected.kind == "finite_tolerance":
                assert produced.score == pytest.approx(expected.score, rel=1e-12)
                assert produced.absolute_error == pytest.approx(expected.absolute_error)
                assert produced.allowed_error == pytest.approx(expected.allowed_error)
            else:
                # A nonfinite category is never scored against the tolerance relation.
                assert produced.score is None
                assert produced.absolute_error is None
                assert produced.allowed_error is None

    def test_maximum_absolute_error_matches_the_oracle(self) -> None:
        rng = np.random.default_rng(99)
        actual = rng.normal(size=500) * 100.0
        reference = rng.normal(size=500) * 100.0
        report = sciresult.compare(_array(actual), _array(reference), rtol=0.0, atol=1e9)
        expected = oracles.oracle_pair_counts(
            [float(value) for value in actual],
            [float(value) for value in reference],
            rtol=0.0,
            atol=1e9,
            equal_nan=False,
        )
        metric = report.max_finite_absolute_error
        assert metric is not None
        assert expected.max_absolute_value is not None
        assert metric.value == pytest.approx(expected.max_absolute_value)
        assert metric.flat_index == expected.max_absolute_index
