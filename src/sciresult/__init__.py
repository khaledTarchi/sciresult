"""sciresult: strict, coordinate-aware comparison of two Xarray ``DataArray`` values.

The library answers one question:

    Are these arrays defined over the same labeled comparison domain, and where
    does the actual result exceed the supplied tolerance?

Two evaluation functions are provided:

- :func:`compare` returns an immutable :class:`ComparisonReport`.
- :func:`assert_close` raises :class:`ComparisonMismatch` when the arrays differ.

Importing this package performs no comparison, registers no plugin, reads no
configuration file, and triggers no optional import or filesystem activity.

Assurance statement
-------------------
For supported inputs and the documented finite-computation semantics, the
library compares the declared labeled domains and evaluates the supplied
numerical tolerance without silently aligning, interpolating, converting, or
mutating the inputs. A ``MATCH`` means only that the supported comparison passed.
"""

from __future__ import annotations

from sciresult.api import assert_close, compare
from sciresult.errors import (
    ComparisonArithmeticError,
    ComparisonMismatch,
    EmptyComparisonError,
    InvalidComparisonArgument,
    SciResultError,
    UnsupportedInputError,
)
from sciresult.model import (
    ComparisonReport,
    ComparisonStatus,
    FailureExample,
    FailureKind,
    LabelKind,
    LabelValue,
    LocatedMetric,
    PairCounts,
    StructuralCode,
    StructuralIssue,
)

__version__ = "0.1.0"

__all__ = [
    "ComparisonArithmeticError",
    "ComparisonMismatch",
    "ComparisonReport",
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
    "__version__",
    "assert_close",
    "compare",
]
