"""Typed exception hierarchy for :mod:`sciresult`.

This module deliberately imports nothing from the rest of the package at
runtime so that it can be imported from anywhere without creating a cycle.
``ComparisonMismatch`` only needs the report type for typing, so that import is
guarded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from sciresult.model import ComparisonReport

__all__ = [
    "ComparisonArithmeticError",
    "ComparisonMismatch",
    "EmptyComparisonError",
    "InvalidComparisonArgument",
    "SciResultError",
    "UnsupportedInputError",
]


class SciResultError(Exception):
    """Base class for every operational error raised by :mod:`sciresult`.

    Parameters
    ----------
    message:
        A concise, already-safe description. Implementations must not embed the
        ``repr`` of a caller-supplied object.
    code:
        A stable, machine-readable reason code. It defaults to the class-level
        :attr:`REASON` and is refined per call site when useful.

    Attributes
    ----------
    message:
        The message the exception was constructed with.
    code:
        The stable reason code for this error instance.
    """

    REASON: ClassVar[str] = "sciresult_error"

    code: str = "unspecified"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class InvalidComparisonArgument(SciResultError):  # noqa: N818 - name fixed by the API contract
    """A supplied option failed validation.

    Raised for tolerances, ``equal_nan``, and ``max_examples`` that cannot be
    normalized into a supported configuration. Never raised for input objects.
    """

    REASON: ClassVar[str] = "invalid_comparison_argument"


class UnsupportedInputError(SciResultError):
    """An input is outside the supported domain.

    Raised for wrong object types, unsupported array backends, unsupported
    dtypes, unsupported coordinate types, unsupported index families, and
    non-string guarded attribute values. The library never coerces an
    unsupported object into a supported one.
    """

    REASON: ClassVar[str] = "unsupported_input"


class EmptyComparisonError(SciResultError):
    """An input contains zero elements.

    Empty inputs are rejected rather than reported as matching.
    """

    REASON: ClassVar[str] = "empty_comparison"


class ComparisonArithmeticError(SciResultError):
    """A finite arithmetic step overflowed or became invalid.

    Raised when binary64 subtraction, multiplication, or addition over finite
    operands overflows or produces an invalid value. No report is produced in
    that case: the evaluation is aborted instead of reporting a partial result.
    """

    REASON: ClassVar[str] = "comparison_arithmetic_error"


class ComparisonMismatch(AssertionError):  # noqa: N818 - name fixed by the API contract
    """Raised by :func:`sciresult.assert_close` when arrays are different.

    This is an :class:`AssertionError` so that ordinary test frameworks treat
    it as a failure rather than as an error.

    Parameters
    ----------
    report:
        The completed, detached report describing the difference. It is also
        available as :attr:`report`.

    Notes
    -----
    The exception message is the rendered report, so a failing test prints the
    same bounded, coordinate-aware diagnostics that :meth:`ComparisonReport.format`
    produces.
    """

    def __init__(self, report: ComparisonReport) -> None:
        super().__init__(report.format())
        self.report = report
