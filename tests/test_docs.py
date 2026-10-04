"""Executability of every example in the documentation.

The specification requires that documentation examples be executable in tests, so
this module runs each ``python`` fence found in ``README.md`` and ``docs/`` and
then asserts the values those documents claim. A number written by hand in a
document is wrong sooner or later; a number that a test recomputes cannot drift
silently. This file already exists because the quickstart documented an
``allowed_error`` of 1.99999998e-06 where the library computes 2.0500009999999997e-06.

Blocks execute in document order in one shared namespace per file, because the
examples build on each other: the quickstart defines ``actual`` and ``reference``
once and then perturbs them. Blocks tagged ``console`` or ``text`` are not
executed, since they are shell transcripts and rendered output rather than
programs; the ``text`` block that reproduces a report is compared against
``report.format()`` instead.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

import sciresult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = [
    PROJECT_ROOT / "README.md",
    *sorted((PROJECT_ROOT / "docs").glob("*.md")),
]

_FENCE = re.compile(
    r"^```(?P<info>[A-Za-z0-9_+-]*)[^\S\n]*\n(?P<body>.*?)^```[^\S\n]*$",
    re.MULTILINE | re.DOTALL,
)
_REPORT_FENCE_START = "sciresult comparison report"


def fences(text: str, info: str) -> list[str]:
    """Return every code fence with the given info string, in document order."""
    return [m.group("body") for m in _FENCE.finditer(text) if m.group("info") == info]


@contextlib.contextmanager
def _repository_working_directory() -> Iterator[None]:
    """Run a snippet from the repository root.

    The documentation tells readers to run its commands from the repository root,
    so examples that walk up from the working directory must be exercised that
    way. The change is undone even when a snippet raises, so one failing document
    cannot leave the rest of the session in the wrong directory.
    """
    previous = Path.cwd()
    os.chdir(PROJECT_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def run_document(path: Path) -> dict[str, Any]:
    """Execute every Python example in one document and return its namespace."""
    blocks = fences(path.read_text(encoding="utf-8"), "python")
    assert blocks, f"{path.name} documents no Python examples, so none can be checked"
    namespace: dict[str, Any] = {"__name__": f"_docs_{path.stem}"}
    with _repository_working_directory():
        for number, block in enumerate(blocks, start=1):
            try:
                exec(compile(block, f"{path.name}:block{number}", "exec"), namespace)
            except Exception as error:  # pragma: no cover - failure path
                pytest.fail(
                    f"{path.name} python block {number} did not run: {error!r}\n\n{block}",
                    pytrace=False,
                )
    namespace["__document__"] = path.read_text(encoding="utf-8")
    return namespace


@pytest.fixture(scope="module")
def namespaces() -> dict[str, dict[str, Any]]:
    return {path.name: run_document(path) for path in DOCUMENTS}


def namespace_of(namespaces: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    return namespaces[name]


class TestEveryExampleExecutes:
    @pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: p.name)
    def test_all_python_examples_run(
        self, namespaces: dict[str, dict[str, Any]], path: Path
    ) -> None:
        assert namespaces[path.name]["__document__"]

    def test_no_documented_example_is_left_unexecuted(self) -> None:
        for path in DOCUMENTS:
            for block in fences(path.read_text(encoding="utf-8"), "python"):
                assert ">>>" not in block, f"{path.name} mixes shell syntax into a python block"
                assert "..." not in block.splitlines()[0], f"{path.name} block is a stub"


class TestReadmeClaims:
    def test_the_opening_example_matches(self, namespaces: dict[str, dict[str, Any]]) -> None:
        namespace = namespaces["README.md"]
        report = namespace["report"]
        assert report.status is sciresult.ComparisonStatus.MATCH
        assert report.counts is not None
        assert report.counts.total == 3
        assert report.examples == ()

    def test_comparison_report_has_no_truthiness(
        self, namespaces: dict[str, dict[str, Any]]
    ) -> None:
        report = namespaces["README.md"]["report"]
        with pytest.raises(TypeError):
            bool(report)

    def test_the_documented_names_differ_yet_still_match(
        self, namespaces: dict[str, dict[str, Any]]
    ) -> None:
        namespace = namespaces["README.md"]
        assert namespace["actual"].name == "run_b"
        assert namespace["reference"].name == "run_a"

    def test_the_failure_form_prints_a_report(self, namespaces: dict[str, dict[str, Any]]) -> None:
        namespace = namespaces["README.md"]
        perturbed = namespace["reference"].copy()
        perturbed[1] = 20.5
        report = sciresult.compare(namespace["actual"], perturbed, rtol=1e-7, atol=1e-12)
        assert "examples:" in report.format()

    def test_assert_close_message_is_the_rendered_report(
        self, namespaces: dict[str, dict[str, Any]]
    ) -> None:
        namespace = namespaces["README.md"]
        actual, reference = namespace["actual"], namespace["reference"]
        sciresult.assert_close(actual, reference, rtol=1e-7, atol=1e-12)

        perturbed = reference.copy()
        perturbed[0] = 99.0
        with pytest.raises(sciresult.ComparisonMismatch) as caught:
            sciresult.assert_close(actual, perturbed, rtol=1e-7, atol=1e-12)
        assert caught.value.report is not None
        assert str(caught.value).startswith("sciresult comparison report")


class TestQuickstartClaims:
    def test_the_selected_array_is_the_one_documented(
        self, namespaces: dict[str, dict[str, Any]]
    ) -> None:
        actual = namespaces["quickstart.md"]["actual"]
        assert actual.name == "run_b"
        assert actual.dims == ("time",)
        assert actual.attrs == {"units": "mm/day"}
        assert list(actual.coords["time"].values) == [2020.0, 2021.0, 2022.0]

    def test_the_passing_call(self, namespaces: dict[str, dict[str, Any]]) -> None:
        passing = namespaces["quickstart.md"]["passing"]
        assert passing.status is sciresult.ComparisonStatus.MATCH
        assert passing.counts.total == 3
        assert passing.examples == ()

    def test_the_structural_mismatch_withholds_the_scan(
        self, namespaces: dict[str, dict[str, Any]]
    ) -> None:
        structural = namespaces["quickstart.md"]["structural"]
        assert structural.status is sciresult.ComparisonStatus.DIFFERENT
        assert structural.structural_issue_count == 2
        assert [issue.code for issue in structural.structural_issues] == [
            sciresult.StructuralCode.COORDINATE_VALUES,
            sciresult.StructuralCode.INDEX_VALUES,
        ]
        assert structural.numeric_scan_complete is False
        assert structural.counts is None
        assert structural.examples == ()
        assert structural.max_finite_absolute_error is None

    def test_the_numerical_mismatch_counts(self, namespaces: dict[str, dict[str, Any]]) -> None:
        report = namespaces["quickstart.md"]["report"]
        assert report.status is sciresult.ComparisonStatus.DIFFERENT
        assert report.structural_issue_count == 0
        assert report.numeric_scan_complete is True
        assert report.counts is not None
        assert report.counts.total == 3
        assert report.counts.finite_pairs == 3
        assert report.counts.finite_failures == 1

    def test_the_documented_example_values(self, namespaces: dict[str, dict[str, Any]]) -> None:
        example = namespaces["quickstart.md"]["example"]
        assert example.kind is sciresult.FailureKind.FINITE_TOLERANCE
        assert example.index == (1,)
        assert example.actual == 20.0
        assert example.reference == 20.5
        assert example.absolute_error == 0.5
        assert example.allowed_error == 2.0500009999999997e-06
        assert example.score == 243902.3200476488
        assert example.score_saturated is False
        label = dict(example.labels)["time"]
        assert label.kind is sciresult.LabelKind.FLOAT
        assert label.value == 2021.0

    def test_the_documented_pytest_example_passes(
        self, namespaces: dict[str, dict[str, Any]]
    ) -> None:
        test = namespaces["quickstart.md"]["test_run_b_matches_run_a"]
        assert test() is None


class TestSemanticsClaims:
    def test_the_relation_is_inclusive(self, namespaces: dict[str, dict[str, Any]]) -> None:
        namespace = namespaces["semantics.md"]
        # Exactly on the boundary, so the inclusive comparison is what decides.
        actual = xr.DataArray(np.array([1.5]), dims="x")
        reference = xr.DataArray(np.array([1.0]), dims="x")
        on_the_boundary = sciresult.compare(actual, reference, rtol=0.0, atol=0.5)
        assert on_the_boundary.status is sciresult.ComparisonStatus.MATCH

        just_past = sciresult.compare(actual, reference, rtol=0.0, atol=np.nextafter(0.5, 0.0))
        assert just_past.status is sciresult.ComparisonStatus.DIFFERENT
        assert namespace["matching"].numeric_scan_complete is True

    def test_a_coordinate_difference_withholds_the_scan(self) -> None:
        left = xr.DataArray(np.array([1.0]), dims="x", coords={"x": [1.0]})
        right = xr.DataArray(np.array([1.0]), dims="x", coords={"x": [2.0]})
        report = sciresult.compare(left, right, rtol=0.0, atol=0.0)
        assert report.status is sciresult.ComparisonStatus.DIFFERENT
        assert report.counts is None
        assert report.numeric_scan_complete is False

    def test_a_dtype_difference_does_not_withhold_the_scan(self) -> None:
        left = xr.DataArray(np.array([1.0, 2.0], dtype=np.float32), dims="x")
        right = xr.DataArray(np.array([1.0, 3.0], dtype=np.float64), dims="x")
        report = sciresult.compare(left, right, rtol=0.0, atol=0.0)
        assert report.status is sciresult.ComparisonStatus.DIFFERENT
        assert report.numeric_scan_complete is True
        assert report.counts is not None
        assert report.counts.total == 2
        assert report.counts.finite_failures == 1
        assert [issue.code for issue in report.structural_issues] == [
            sciresult.StructuralCode.DATA_DTYPE
        ]

    def test_no_tolerance_rescues_an_infinity(self) -> None:
        left = xr.DataArray(np.array([np.inf]), dims="x")
        right = xr.DataArray(np.array([1e308]), dims="x")
        report = sciresult.compare(left, right, rtol=1.0, atol=1e308)
        assert report.status is sciresult.ComparisonStatus.DIFFERENT
        assert report.counts is not None
        assert report.counts.infinity_mismatches == 1

    def test_guarded_attributes_are_compared_and_others_are_not(self) -> None:
        left = xr.DataArray(np.array([1.0]), dims="x", attrs={"units": "m", "note": "a"})
        same_units = xr.DataArray(np.array([1.0]), dims="x", attrs={"units": "m", "note": "b"})
        assert (
            sciresult.compare(left, same_units, rtol=0.0, atol=0.0).status
            is sciresult.ComparisonStatus.MATCH
        )

        other_units = xr.DataArray(np.array([1.0]), dims="x", attrs={"units": "km"})
        report = sciresult.compare(left, other_units, rtol=0.0, atol=0.0)
        assert [issue.code for issue in report.structural_issues] == [
            sciresult.StructuralCode.GUARDED_ATTRIBUTE
        ]

    def test_duplicate_labels_are_allowed(self) -> None:
        values = xr.DataArray(np.array([1.0, 2.0]), dims="x", coords={"x": [5.0, 5.0]})
        report = sciresult.compare(values, values.copy(), rtol=0.0, atol=0.0)
        assert report.status is sciresult.ComparisonStatus.MATCH
        assert report.structural_issue_count == 0

    def test_equal_nan_does_not_affect_coordinates(self) -> None:
        left = xr.DataArray(np.array([1.0, 2.0]), dims="x", coords={"x": [1.0, np.nan]})
        right = xr.DataArray(np.array([1.0, 2.0]), dims="x", coords={"x": [1.0, np.nan]})
        for equal_nan in (True, False):
            report = sciresult.compare(left, right, rtol=0.0, atol=0.0, equal_nan=equal_nan)
            assert report.status is sciresult.ComparisonStatus.MATCH


class TestDiagnosticsClaims:
    def test_the_documented_rendering_matches_reality(
        self, namespaces: dict[str, dict[str, Any]]
    ) -> None:
        namespace = namespaces["diagnostics.md"]
        rendered = next(
            block
            for block in fences(namespace["__document__"], "text")
            if block.startswith(_REPORT_FENCE_START)
        )
        assert namespace["report"].format().rstrip("\n") == rendered.rstrip("\n")

    def test_zero_allowance_is_not_saturation(self) -> None:
        left = xr.DataArray(np.array([1.0, 1.0]), dims="x")
        right = xr.DataArray(np.array([np.nan, 2.0]), dims="x")
        report = sciresult.compare(left, right, rtol=0.0, atol=0.0)
        examples = {example.kind: example for example in report.examples}
        zero_allowance = examples[sciresult.FailureKind.FINITE_TOLERANCE]
        assert zero_allowance.allowed_error == 0.0
        assert zero_allowance.score == float("inf")
        assert zero_allowance.score_saturated is False

    def test_quotient_overflow_is_saturation(self) -> None:
        tiny = np.nextafter(0.0, 1.0)
        left = xr.DataArray(np.array([1.0]), dims="x")
        right = xr.DataArray(np.array([tiny]), dims="x")
        report = sciresult.compare(left, right, rtol=0.0, atol=tiny)
        example = report.examples[0]
        assert example.kind is sciresult.FailureKind.FINITE_TOLERANCE
        assert example.score == float("inf")
        assert example.score_saturated is True

    def test_nonfinite_examples_carry_no_finite_numbers(self) -> None:
        left = xr.DataArray(np.array([np.nan, np.inf]), dims="x")
        right = xr.DataArray(np.array([1.0, np.inf]), dims="x")
        report = sciresult.compare(left, right, rtol=0.0, atol=0.0)
        assert report.examples
        for example in report.examples:
            assert example.absolute_error is None
            assert example.allowed_error is None
            assert example.score is None
            assert example.score_saturated is False

    def test_saturation_is_marked_in_the_rendering(self) -> None:
        tiny = np.nextafter(0.0, 1.0)
        left = xr.DataArray(np.array([1.0]), dims="x")
        right = xr.DataArray(np.array([tiny]), dims="x")
        report = sciresult.compare(left, right, rtol=0.0, atol=tiny)
        rendered = report.format()
        assert "score=+inf (score saturated at positive infinity)" in rendered
        assert "not exact-real ranking" in rendered

    def test_an_unsaturated_score_carries_no_marker(self) -> None:
        left = xr.DataArray(np.array([1.0]), dims="x")
        right = xr.DataArray(np.array([2.0]), dims="x")
        report = sciresult.compare(left, right, rtol=0.0, atol=0.0)
        assert "saturated" not in report.format()

    def test_duplicate_labels_stay_distinguishable_by_position(self) -> None:
        left = xr.DataArray(np.array([1.0, 2.0]), dims="x", coords={"x": [5.0, 5.0]})
        right = xr.DataArray(np.array([9.0, 9.0]), dims="x", coords={"x": [5.0, 5.0]})
        report = sciresult.compare(left, right, rtol=0.0, atol=0.0)
        assert len(report.examples) == 2
        assert [example.index for example in report.examples] == [(0,), (1,)]
        assert all(dict(example.labels)["x"].value == 5.0 for example in report.examples)

    def test_max_examples_does_not_change_the_verdict_or_counts(self) -> None:
        left = xr.DataArray(np.array([1.0, 2.0, 3.0, 4.0]), dims="x")
        right = xr.DataArray(np.array([9.0, 9.0, 9.0, 9.0]), dims="x")
        small = sciresult.compare(left, right, rtol=0.0, atol=0.0, max_examples=1)
        large = sciresult.compare(left, right, rtol=0.0, atol=0.0, max_examples=5)
        assert small.status is large.status
        assert small.counts == large.counts
        assert len(small.examples) == 1
        assert len(large.examples) == 4

    def test_max_finite_absolute_error_includes_passing_pairs(self) -> None:
        left = xr.DataArray(np.array([1.0, 2.0]), dims="x")
        right = xr.DataArray(np.array([1.0, 2.5]), dims="x")
        report = sciresult.compare(left, right, rtol=0.0, atol=0.5)
        assert report.status is sciresult.ComparisonStatus.MATCH
        assert report.examples == ()
        assert report.max_finite_absolute_error is not None
        located = report.max_finite_absolute_error
        assert located.value == 0.5
        assert located.index == (1,)
        assert located.flat_index == 1

    def test_max_finite_absolute_error_is_none_with_no_finite_pairs(self) -> None:
        left = xr.DataArray(np.array([np.nan]), dims="x")
        right = xr.DataArray(np.array([1.0]), dims="x")
        report = sciresult.compare(left, right, rtol=0.0, atol=0.0)
        assert report.max_finite_absolute_error is None

    def test_two_independent_nan_reports_are_unequal(self) -> None:
        def build() -> sciresult.ComparisonReport:
            return sciresult.compare(
                xr.DataArray(np.array([np.nan]), dims="x"),
                xr.DataArray(np.array([1.0]), dims="x"),
                rtol=0.0,
                atol=0.0,
            )

        left, right = build(), build()
        assert left.to_dict() == right.to_dict()
        assert left.format() == right.format()
        assert left != right

    def test_the_documented_encoding_is_json_safe(
        self, namespaces: dict[str, dict[str, Any]]
    ) -> None:
        report = namespaces["diagnostics.md"]["report"]
        assert json.loads(json.dumps(report.to_dict())) == report.to_dict()


class TestLimitationsClaims:
    def test_the_documented_loose_tolerance_really_matches(
        self, namespaces: dict[str, dict[str, Any]]
    ) -> None:
        namespace = namespaces["limitations.md"]
        permissive = namespace["permissive"]
        assert permissive.status is sciresult.ComparisonStatus.MATCH
        assert permissive.examples == ()
        assert namespace["strict"].status is sciresult.ComparisonStatus.DIFFERENT

    def test_integer_data_is_refused_with_a_role_prefixed_code(
        self, namespaces: dict[str, dict[str, Any]]
    ) -> None:
        integers = namespaces["limitations.md"]["integer_data"]
        with pytest.raises(sciresult.UnsupportedInputError) as caught:
            sciresult.compare(integers, integers.copy(), rtol=0.0, atol=0.0)
        message = str(caught.value)
        assert "actual data dtype <i8 is not supported" in message
        assert "binary32 and binary64" in message

    def test_float16_data_is_refused(self) -> None:
        data = xr.DataArray(np.array([1.0], dtype=np.float16), dims="x")
        with pytest.raises(sciresult.UnsupportedInputError) as caught:
            sciresult.compare(data, data.copy(), rtol=0.0, atol=0.0)
        assert "<f2" in str(caught.value)

    def test_float16_coordinates_cannot_reach_admission(self) -> None:
        # pandas refuses to index float16 on the pinned version, so the documented
        # rejection is unreachable through the public path. Documented, tested, and
        # left as-is rather than papered over.
        with pytest.warns(FutureWarning, match="float16"):
            xr.DataArray(
                np.array([1.0, 2.0]), dims="x", coords={"x": np.array([1.0, 2.0], np.float16)}
            )

    def test_a_memmap_is_accepted_and_never_written(self, tmp_path: Path) -> None:
        path = tmp_path / "operand.bin"
        backing = np.array([1.0, 2.0, 3.0])
        backing.tofile(path)
        before = path.read_bytes()
        mapped = np.memmap(path, dtype=np.float64, mode="r", shape=(3,))
        report = sciresult.compare(
            xr.DataArray(mapped, dims="x"),
            xr.DataArray(backing, dims="x"),
            rtol=0.0,
            atol=0.0,
        )
        assert report.status is sciresult.ComparisonStatus.MATCH
        assert path.read_bytes() == before

    def test_operational_errors_are_not_assertion_failures(self) -> None:
        integers = xr.DataArray(np.array([1, 2]), dims="x")
        with pytest.raises(sciresult.SciResultError):
            sciresult.assert_close(integers, integers.copy(), rtol=0.0, atol=0.0)
        assert not issubclass(sciresult.SciResultError, AssertionError)
        assert issubclass(sciresult.ComparisonMismatch, AssertionError)
        assert not issubclass(sciresult.ComparisonMismatch, sciresult.SciResultError)

    def test_a_report_carries_the_data_and_labels_it_was_given(self) -> None:
        left = xr.DataArray(np.array([np.inf]), dims="station", coords={"station": [42.5]})
        right = xr.DataArray(np.array([1.0]), dims="station", coords={"station": [42.5]})
        report = sciresult.compare(left, right, rtol=0.0, atol=0.0)
        rendered = report.format()
        assert "42.5" in rendered
        assert "inf" in rendered

    def test_coordinate_names_appear_in_structural_messages(self) -> None:
        left = xr.DataArray(np.array([1.0]), dims="station", coords={"station": [1.0]})
        right = xr.DataArray(np.array([1.0]), dims="station", coords={"station": [2.0]})
        report = sciresult.compare(left, right, rtol=0.0, atol=0.0)
        assert "station" in report.format()


class TestReleaseDocumentClaims:
    def test_the_version_matches_the_package(self, namespaces: dict[str, dict[str, Any]]) -> None:
        namespace = namespaces["release.md"]
        assert namespace["declared"] == sciresult.__version__ == "0.1.0"

    def test_publication_is_authorised(self, namespaces: dict[str, dict[str, Any]]) -> None:
        namespace = namespaces["release.md"]
        assert namespace["project"]["license"] == "MIT"
        assert namespace["project"]["license-files"] == ["LICENSE"]
        assert namespace["project"]["authors"] == [
            {"name": "Khaled Tarchi", "email": "khaledTarchi@users.noreply.github.com"}
        ]

    def test_no_placeholder_survives_in_the_release_metadata(self) -> None:
        # release.md and release.yml are excluded because they legitimately name
        # the guard itself. A placeholder there is the safety check, not a gap.
        checked = [
            "pyproject.toml",
            "SECURITY.md",
            "README.md",
            "LICENSE",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "docs/quickstart.md",
            "docs/semantics.md",
            "docs/diagnostics.md",
            "docs/limitations.md",
        ]
        offenders = [
            name
            for name in checked
            if "PLACEHOLDER_" in (PROJECT_ROOT / name).read_text(encoding="utf-8")
        ]
        assert offenders == []

    def test_release_prose_carries_no_placeholder_outside_its_guard(self) -> None:
        # Strip the executable fences before searching: the guard assertion inside
        # them is supposed to mention the placeholder.
        text = (PROJECT_ROOT / "docs/release.md").read_text(encoding="utf-8")
        prose = _FENCE.sub("", text)
        assert "PLACEHOLDER_" not in prose

    def test_the_licence_file_is_real_mit(self) -> None:
        licence = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
        assert licence.startswith("MIT License")
        assert "Copyright (c) 2026 Khaled Tarchi" in licence
        assert "Permission is hereby granted, free of charge" in licence
        assert 'THE SOFTWARE IS PROVIDED "AS IS"' in licence
        assert "PLACEHOLDER" not in licence

    def test_the_changelog_records_a_dated_release(self) -> None:
        changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "## [0.1.0] - 2026-10-03" in changelog
        assert "## [0.1.0] - unreleased" not in changelog

    def test_every_project_url_is_real(self, namespaces: dict[str, dict[str, Any]]) -> None:
        urls = namespaces["release.md"]["project"]["urls"]
        assert urls
        for name, value in urls.items():
            assert value.startswith("https://github.com/khaledTarchi/sciresult"), (name, value)

    def test_every_required_file_exists(self) -> None:
        required = [
            "README.md",
            "LICENSE",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "docs/quickstart.md",
            "docs/semantics.md",
            "docs/diagnostics.md",
            "docs/limitations.md",
            "docs/release.md",
            "pyproject.toml",
            ".github/workflows/ci.yml",
            ".github/workflows/release.yml",
        ]
        missing = [name for name in required if not (PROJECT_ROOT / name).is_file()]
        assert missing == []

    def test_the_release_workflow_refuses_without_authorisation(self) -> None:
        workflow = (PROJECT_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        assert "workflow_dispatch" in workflow
        assert "authorization_confirmed" in workflow
        # The manual-dispatch gate and the placeholder guard must both survive
        # any future edit, so that publishing stays a deliberate act.
        assert "PLACEHOLDER_" in workflow
        assert "environment:" in workflow
        assert "name: pypi" in workflow

    def test_the_ci_workflow_covers_the_declared_support_matrix(self) -> None:
        ci = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        for version in ("3.11", "3.12", "3.13"):
            assert f'"{version}"' in ci
        for system in ("ubuntu-latest", "macos-latest", "windows-latest"):
            assert system in ci
        assert "numpy==1.26" in ci
        assert "xarray==2024.10.0" in ci
        assert "python -m build" in ci
        assert "--no-isolation" not in ci  # CI builds with isolation on purpose
