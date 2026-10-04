"""Packaging, public API surface, and resource-bound tests.

These tests assert the promises a downstream user relies on when the package is
installed rather than imported from a checkout: one public entry point, one
authoritative version, type information shipped, no command-line interface, no
pytest plugin, and no dependency on any optional acceleration library.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

import sciresult

PACKAGE_ROOT = Path(sciresult.__file__).resolve().parent
# The repository root is located from this test file, not from the imported
# package. Locating it from the package would resolve to site-packages when the
# suite runs against an installed wheel, and every metadata assertion below
# would fail with a FileNotFoundError about a pyproject.toml that is not there.
# These tests validate repository metadata, so they are skipped when the tests
# have been copied somewhere without it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
RUNNING_FROM_A_CHECKOUT = PYPROJECT.is_file()

requires_checkout = pytest.mark.skipif(
    not RUNNING_FROM_A_CHECKOUT,
    reason="repository metadata is unavailable; tests were run outside a checkout",
)

PUBLIC_NAMES = (
    "compare",
    "assert_close",
    "ComparisonReport",
    "ComparisonMismatch",
    "ComparisonStatus",
    "EmptyComparisonError",
    "FailureExample",
    "FailureKind",
    "InvalidComparisonArgument",
    "LabelKind",
    "LabelValue",
    "LocatedMetric",
    "PairCounts",
    "SciResultError",
    "StructuralCode",
    "StructuralIssue",
    "UnsupportedInputError",
    "ComparisonArithmeticError",
    "__version__",
)


def _pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter outside the pytest import hook."""
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    source_root = str(PACKAGE_ROOT.parent)
    environment["PYTHONPATH"] = f"{source_root}{os.pathsep}{existing}" if existing else source_root
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=PROJECT_ROOT,
        env=environment,
    )


class TestPublicSurface:
    @pytest.mark.parametrize("name", PUBLIC_NAMES)
    def test_public_name_is_importable(self, name: str) -> None:
        assert hasattr(sciresult, name)

    def test_dunder_all_matches_the_documented_surface(self) -> None:
        assert set(sciresult.__all__) == set(PUBLIC_NAMES)

    def test_everything_in_all_is_documented(self) -> None:
        undocumented = [name for name in sciresult.__all__ if not getattr(sciresult, name).__doc__]
        assert undocumented == []

    def test_only_compare_and_assert_close_are_callable_entry_points(self) -> None:
        public_callables = {
            name
            for name in sciresult.__all__
            if callable(getattr(sciresult, name)) and not isinstance(getattr(sciresult, name), type)
        }
        assert public_callables == {"compare", "assert_close"}

    @requires_checkout
    def test_the_package_exposes_no_command_line_interface(self) -> None:
        assert not hasattr(sciresult, "main")
        assert not hasattr(sciresult, "cli")
        config = _pyproject()
        assert "scripts" not in config["project"]
        assert "gui-scripts" not in config["project"]

    @requires_checkout
    def test_the_package_registers_no_entry_points(self) -> None:
        config = _pyproject()
        assert "entry-points" not in config["project"]
        assert "scripts" not in config["project"]

    def test_there_is_no_pytest_plugin(self) -> None:
        assert not any(
            name.startswith("pytest") for name in dir(sciresult) if not name.startswith("__")
        )
        try:
            distribution = metadata.distribution("sciresult")
        except metadata.PackageNotFoundError:
            distribution = None
        if distribution is not None:
            groups = {entry.group for entry in distribution.entry_points}
            assert not any("pytest" in group for group in groups)

    def test_no_module_imports_a_pytest_plugin(self) -> None:
        for module in (package for package in sys.modules if package.startswith("sciresult")):
            source = Path(sys.modules[module].__file__ or "").read_text(encoding="utf-8")
            assert "pytest11" not in source
            assert "import pytest" not in source


class TestVersioning:
    def test_the_version_is_a_release_string(self) -> None:
        assert sciresult.__version__ == "0.1.0"

    @requires_checkout
    def test_the_version_has_one_authoritative_source(self) -> None:
        config = _pyproject()
        assert "version" not in config["project"]
        assert config["project"]["dynamic"] == ["version"]
        # The version is read from the package source, so building and installing
        # never requires Git to be present or the tree to be a checkout.
        assert config["tool"]["hatch"]["version"] == {"path": "src/sciresult/__init__.py"}

    def test_a_fresh_interpreter_reports_the_same_version(self) -> None:
        completed = _run("import sciresult; print(sciresult.__version__)")
        assert completed.stdout.strip() == sciresult.__version__

    @requires_checkout
    def test_the_version_is_written_once_in_the_source_tree(self) -> None:
        occurrences = [
            path.name
            for path in PACKAGE_ROOT.glob("*.py")
            if "__version__" in path.read_text(encoding="utf-8")
        ]
        assert occurrences == ["__init__.py"]


class TestTypingMetadata:
    def test_py_typed_ships_with_the_package(self) -> None:
        marker = PACKAGE_ROOT / "py.typed"
        assert marker.is_file()
        assert marker.read_text(encoding="utf-8") == ""

    @requires_checkout
    def test_the_marker_is_included_by_the_build(self) -> None:
        config = _pyproject()
        assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/sciresult"]

    def test_annotations_are_resolvable_at_runtime(self) -> None:
        assert importlib.import_module("sciresult.api") is not None
        assert sciresult.compare.__annotations__


class TestDeclaredDependencies:
    @requires_checkout
    def test_only_required_dependencies_are_declared(self) -> None:
        dependencies = _pyproject()["project"]["dependencies"]
        names = {item.split(">")[0].split("=")[0].split("[")[0].strip() for item in dependencies}
        assert names == {"numpy", "xarray"}

    @requires_checkout
    def test_no_acceleration_library_is_required_at_runtime(self) -> None:
        for module in ("numba", "cython", "pyarrow", "dask"):
            assert module not in sys.modules or module not in {
                item.split(" ")[0] for item in _pyproject()["project"]["dependencies"]
            }

    @requires_checkout
    def test_the_test_dependencies_are_declared_in_the_optional_group(self) -> None:
        optional = _pyproject()["project"]["optional-dependencies"]
        assert "test" in optional
        names = " ".join(optional["test"])
        for package in ("pytest", "hypothesis"):
            assert package in names


class TestFreshInterpreterImport:
    def test_importing_the_package_pulls_in_no_test_dependency(self) -> None:
        code = (
            "import sys; import sciresult; "
            "assert 'pytest' not in sys.modules; "
            "assert 'hypothesis' not in sys.modules; "
            "print('ok')"
        )
        assert _run(code).stdout.strip() == "ok"

    def test_a_comparison_runs_in_a_fresh_interpreter(self) -> None:
        code = (
            "import numpy as np, xarray as xr, sciresult;"
            "a = xr.DataArray(np.array([1.0, 2.0]), dims=('x',));"
            "r = sciresult.compare(a, a, rtol=0.0, atol=0.0);"
            "print(r.status.value)"
        )
        assert _run(code).stdout.strip() == "match"


class TestResourceBounds:
    def test_a_large_comparison_does_not_allocate_a_whole_array_temporary(self) -> None:
        # A difference array would need one float64 per element. Tracking the
        # peak allocation of the scan itself is done in benchmarks/memory_probe.py;
        # here the bound is checked structurally: the scan is blocked.
        size = 1 << 16
        actual = np.zeros(size)
        reference = np.zeros(size)
        report = sciresult.compare(
            xr.DataArray(actual, dims=("x",)),
            xr.DataArray(reference, dims=("x",)),
            rtol=0.0,
            atol=0.0,
        )
        assert report.counts is not None
        assert report.counts.total == size
        assert report.examples == ()

    def test_diagnostic_storage_is_bounded_by_the_cap(self) -> None:
        size = 100_000
        actual = np.arange(float(size))
        reference = np.zeros(size)
        report = sciresult.compare(
            xr.DataArray(actual, dims=("x",)),
            xr.DataArray(reference, dims=("x",)),
            rtol=0.0,
            atol=0.0,
            max_examples=5,
        )
        assert report.counts is not None
        # Position 0 holds 0.0 on both sides, so it passes; everything else fails.
        assert report.counts.total_failures == size - 1
        assert len(report.examples) == 5

    def test_the_report_does_not_retain_the_inputs(self) -> None:
        size = 10_000
        actual = xr.DataArray(np.arange(float(size)), dims=("x",))
        reference = xr.DataArray(np.zeros(size), dims=("x",))
        report = sciresult.compare(actual, reference, rtol=0.0, atol=0.0, max_examples=3)
        retained = sum(sys.getsizeof(getattr(report, name)) for name in report.__slots__)
        assert retained < actual.nbytes // 10

    def test_a_noncontiguous_input_is_not_copied_whole(self) -> None:
        base = np.zeros(200_000)
        view = base[::2]
        report = sciresult.compare(
            xr.DataArray(view, dims=("x",)),
            xr.DataArray(np.zeros(view.size), dims=("x",)),
            rtol=0.0,
            atol=0.0,
        )
        assert report.counts is not None
        assert report.counts.total == view.size

    def test_a_memmap_input_is_read_in_place(self, memmap_factory: Any) -> None:
        make = memmap_factory
        left = make(np.arange(10_000.0), "left.bin")
        right = make(np.arange(10_000.0), "right.bin")
        report = sciresult.compare(
            xr.DataArray(left, dims=("x",)), xr.DataArray(right, dims=("x",)), rtol=0.0, atol=0.0
        )
        assert report.status.value == "match"
        assert isinstance(report.max_finite_absolute_error, sciresult.LocatedMetric)
