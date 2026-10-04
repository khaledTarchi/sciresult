# Contributing to sciresult

Thank you for your interest. This project is small and deliberately narrow, so
most contributions will be about correctness of the stated rules rather than new
features.

## Current state: not released

No licence has been selected and the maintainer identity is still a placeholder.
That blocks publication and blocks redistribution of any contribution you send.
Please raise that with the project owner before investing significant work.

## The one rule that matters

**The specification is the contract.** `docs/semantics.md` states exactly what a
comparison does. Code that disagrees with the specification is a bug even if it
looks reasonable, and a specification that is ambiguous is a defect to be fixed
before the code is.

If you change behaviour, change `docs/semantics.md` in the same commit and add a
`CHANGELOG.md` entry under the correct category. Numerical semantic changes are
reviewed most strictly of all.

## Getting set up

```console
$ python -m venv .venv
$ . .venv/bin/activate          # Windows: .venv\Scripts\activate
$ pip install -e ".[test,dev]"
$ python -m pytest
```

Requires Python 3.11 or newer. Minimum supported dependency versions are declared
in `pyproject.toml`; the versions actually exercised are listed in
`docs/release.md`.

## Checks that must pass

```console
$ python -m pytest
$ python -m ruff check src tests benchmarks
$ python -m ruff format --check src tests benchmarks
$ python -m mypy src
$ python -m build --no-isolation
$ python -m twine check dist/*
```

`pytest` runs with `-W error`, so a new warning fails the suite. Do not silence a
warning without understanding it.

## How the test suite is organised

- **Unit tests** pin documented behaviour: `tests/test_validation.py`,
  `test_structure.py`, `test_numeric.py`, `test_diagnostics.py`, `test_model.py`,
  `test_formatting.py`, `test_api.py`, `test_installation.py`.
- **Property and differential tests** in `tests/test_properties.py` compare the
  package against `tests/oracles.py`, an independent scalar implementation of
  the same rules. Disagreement on any generated input fails the suite.
- **Text fixtures** in `tests/fixtures/` pin the rendered report. Regenerate
  them deliberately:

  ```console
  $ set SCIRESULT_UPDATE_SNAPSHOTS=1        # Windows
  $ SCIRESULT_UPDATE_SNAPSHOTS=1 python -m pytest tests/test_formatting.py  # POSIX
  $ git diff tests/fixtures                  # review every change
  ```

  A formatting change that alters a fixture must be justified in the commit
  message. Formatting is part of the contract for anyone reading a report.

## Two implementations, on purpose

`src/sciresult/` is vectorised NumPy. `tests/oracles.py` is plain Python
scalars written directly from `docs/semantics.md`. Keeping them independent is
what makes the differential tests worth running.

When you find a disagreement, resist the urge to make one side match the other
immediately. First decide which side is right, and write or change a test that
says so. A "fix" that silences a differential failure without deciding who was
correct has destroyed the only thing that test was for.

## What a good pull request looks like

- One topic per change. Behaviour changes separate from refactors.
- A test that fails before the change and passes after it.
- No new optional runtime dependency. Diagnostics must not require an
  accelerator library.
- Benchmarks updated if you change the scan loop, with before and after numbers
  from `benchmarks/compare_arrays.py`.
- No new public name without a docstring, a `__all__` entry, and a test in
  `tests/test_installation.py`.

## Style

Ruff enforces formatting and lint. Beyond that: prefer explicit rule sequences
over clever abstractions, name the rule being applied, and comment the reasoning
rather than the mechanics. The numeric core is written to be read and checked
against a document, so it should look like both.

## Licence

By contributing you are agreeing that your contribution is licensed under
whatever licence the project owner ultimately selects. That licence is not
chosen yet, which is another reason publication is currently blocked.