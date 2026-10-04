# Release status and procedure

## Status: authorised, awaiting publication

| Item | Value |
| --- | --- |
| Version | 0.1.0 |
| Licence | MIT |
| Copyright holder | Khaled Tarchi |
| Maintainer | Khaled Tarchi (`khaledTarchi@users.noreply.github.com`) |
| Repository | <https://github.com/khaledTarchi/sciresult> |
| Security reporting | Private GitHub security advisory |
| Blockers | None |

The project owner selected MIT and named the copyright holder and maintainer.
Publication is authorised. What remains is mechanical: build, verify, tag, and
upload, which `.github/workflows/release.yml` performs.

### Outstanding cosmetic items

Neither blocks publication, and neither is a placeholder any more:

- The maintainer email is the GitHub `noreply` form, which is the convention for
  package metadata. It does not receive ordinary mail. Replace it in
  `pyproject.toml` if a real address is preferred.
- Security reports go through GitHub private vulnerability reporting rather than
  an inbox. `SECURITY.md` also lists the maintainer address for anyone who
  prefers direct contact.

## Support matrix

Verified on this machine only, on 2026-10-03.

| Component | Verified | Minimum declared |
| --- | --- | --- |
| Python | 3.12.10 (Windows) | 3.11 |
| NumPy | 2.0.2 | 1.26 |
| Xarray | 2026.9.0 | 2024.10.0 |
| pytest | 8.4.2 | test extra |
| Hypothesis | 6.168.3 | test extra |
| Ruff | 0.16.1 | dev extra |
| mypy | 2.3.0 | dev extra |
| Hatchling | 1.32.4 | build backend |

Declared minimums are **not** verified. Nothing was installed or executed
against NumPy 1.26 or Xarray 2024.10.0, because dependency installation requires
network access, which is unavailable in this environment. Treat the minimum
column as an unverified claim until CI runs the lower bound.

Not verified at all: Linux, macOS, non-x86-64 platforms, distributed arrays,
accelerator hardware, and behaviour under memory pressure.

## Commands

Run in order from the repository root. Every command below has been executed
here unless marked otherwise.

The version reported by the package and the declared metadata are checked like
this:

```python
import tomllib
from pathlib import Path

import sciresult


def find_repository_root() -> Path:
    """Walk up from the working directory until pyproject.toml appears."""
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise SystemExit("run this from inside the repository")


root = find_repository_root()
declared = sciresult.__version__
metadata_text = (root / "pyproject.toml").read_text(encoding="utf-8")
project = tomllib.loads(metadata_text)["project"]
licence_text = (root / "LICENSE").read_text(encoding="utf-8")

# These hold today. If any of them stops holding, this document and the
# procedure it describes are out of date.
assert declared == "0.1.0"
assert project["name"] == "sciresult"
assert project["license"] == "MIT"
assert project["license-files"] == ["LICENSE"]
assert "MIT License" in licence_text
assert "Khaled Tarchi" in licence_text
assert "PLACEHOLDER_" not in metadata_text
assert "PLACEHOLDER_" not in licence_text
```

```console
$ python -m pytest
$ python -m ruff check src tests benchmarks
$ python -m ruff format --check src tests benchmarks
$ python -m mypy src
$ python -m build --no-isolation
$ python -m twine check dist/*
```

Results at the time of writing, on 2026-10-03:

| Check | Result |
| --- | --- |
| `pytest` | **452 passed, 1 skipped** |
| skipped test | requires an accepted real scalar wider than binary64; on 64-bit Windows `np.longdouble` is binary64, so no accepted input can underflow when narrowed |
| `ruff check` | clean |
| `ruff format --check` | clean |
| `mypy src` | clean, strict mode |
| `build` (no isolation) | `sciresult-0.1.0.tar.gz` and `sciresult-0.1.0-py3-none-any.whl` |
| `twine check --strict` | both artifacts pass |
| wheel contents | package plus `py.typed`; no `tests/`, `benchmarks/`, or `docs/` |
| sdist contents | 44 entries: `src/`, all 5 `docs/`, 9 test modules with 8 fixtures, both benchmarks, root files |
| scratch files in artifacts | none; `**/*.bin` is excluded and the memory probe cleans up after itself |
| clean-environment wheel install | verified: `py.typed` present, fresh-interpreter `compare()` works, no `__bool__` |
| full suite against the installed wheel | 452 passed, 1 skipped, run from outside the source tree with `src/` off the path |

`--no-isolation` is required because there is no network access. A build with
isolation would be the preferable final check, and CI runs it.

### Artifact inspection

```console
$ python -m twine check --strict dist/*
$ python -m zipfile --list dist/sciresult-0.1.0-py3-none-any.whl
$ tar -tzf dist/sciresult-0.1.0.tar.gz
```

### Verifying the wheel, offline

A fresh virtual environment cannot install NumPy or Xarray without network
access, so the check below inherits the already-installed scientific stack and
installs only the wheel with `--no-deps`. That is enough to prove the wheel
supplies everything `sciresult` needs, which is the thing a source tree cannot
show.

```console
$ python -m venv --system-site-packages %TEMP%\wheel-check          # POSIX
$ python -m venv --system-site-packages "$env:TEMP\wheel-check"     # Windows
$ %TEMP%\wheel-check\Scripts\python -m pip install --no-index --no-deps dist\*.whl
$ cd %TEMP%                                                          # leave the checkout
$ %TEMP%\wheel-check\Scripts\python -m pytest <repo>\tests -q -o pythonpath=
452 passed, 1 skipped
```

The `-o pythonpath=` override is the important part. It clears the `pythonpath`
setting from `pyproject.toml`, so `import sciresult` resolves to site-packages
rather than to `src/`, and the suite genuinely exercises the built artifact.

## Artifact checks

Before any release, all of these must hold:

- `dist/` contains one `.whl` and one `.tar.gz`, and nothing else.
- The wheel is named for the normalised project name and carries the version
  from `src/sciresult/__init__.py`. `tests/test_installation.py` checks the
  version sources agree.
- `twine check` passes on both artifacts.
- The sdist contains `LICENSE`, all five files under `docs/`, the `src/` tree,
  the `tests/` tree including fixtures, and the `benchmarks/` tree.
- The wheel contains the `sciresult` package and `py.typed`, and does **not**
  contain `tests/`, `benchmarks/`, or `docs/`.
- The installed package imports in a fresh interpreter, outside the source
  checkout, so a stray `src/` on `sys.path` cannot mask a missing file.
- `sciresult.__version__` matches the wheel version.
- Benchmarks are run from the installed wheel and their output recorded, not
  just imported from `src/`.

`tests/test_installation.py` asserts the metadata, version, typing, and
resource-bound properties that can be checked without building. Those metadata
tests locate the repository from the test file rather than from the imported
package, so they keep working when the suite runs against an installed wheel, and
they skip themselves if the tests are copied somewhere without a `pyproject.toml`.

## Benchmarks

```console
$ python benchmarks/compare_arrays.py --size 300000
$ python benchmarks/memory_probe.py --size 1000000
```

Measured on this machine, 2026-10-03, NumPy 2.0.2 and Xarray 2026.9.0, minimum
of 3 runs. **These are local observations, not performance claims.**

| Scenario | Diagnostic (s) | Verdict-only (s) | Upstream (s) |
| --- | --- | --- | --- |
| successful scan | 0.0135 | 0.0063 | 0.0008 |
| sparse finite failures | 0.0137 | 0.0061 | not applicable, raises |
| dense finite failures | 1.0676 | 0.0066 | not applicable, raises |
| many equal-score failures | 1.2379 | 0.0084 | not applicable, raises |
| all NaN | 1.0811 | 0.0077 | 0.0011 |
| mixed nonfinite | 0.0171 | 0.0082 | not applicable, raises |
| zero allowance | 1.0537 | 0.0068 | not applicable, raises |
| Fortran layout | 0.0146 | 0.0080 | 0.0009 |
| reversed layout | 0.0151 | 0.0083 | 0.0010 |
| noncontiguous layout | 0.0079 | 0.0040 | 0.0006 |
| coordinate-heavy | 0.0102 | 0.0046 | 0.0006 |

The verdict-only column returns a Boolean and does no classification, ranking,
capping, label extraction, or rendering. It is expected to be faster, and the
library is not designed to beat it. The upstream column is
`xarray.testing.assert_equal`, which raises on any difference, so it is timed
only where the inputs match; the comparison against it is a sanity check on the
benchmark, not a claim of superiority.

Memory, from `memory_probe.py` in fresh subprocesses at 1,000,000 elements
(15.3 MiB of input per operand), using peak process memory:

| Scenario | Steady-state growth |
| --- | --- |
| successful scan | 0.0 to 0.9 MiB, under 6% of input |
| sparse failures | 0.0 to 0.9 MiB |
| dense failures | 0.0 to 0.9 MiB |
| all NaN | 0.0 to 0.9 MiB |

Growth tracks the block size rather than the array size. These figures bound
this implementation on this machine after warm-up, and they are not a guarantee
about any other host. See [limitations.md](limitations.md#no-hard-memory-cap).

## Publication authorisation

Granted by the project owner on 2026-10-03:

1. **Licence chosen:** MIT, recorded in `LICENSE` and as `license = "MIT"` in
   `pyproject.toml`.
2. **Copyright holder named:** Khaled Tarchi, in both `LICENSE` and
   `CHANGELOG.md`.
3. **Maintainer named:** Khaled Tarchi, with an email in `[project] authors`.
4. **Repository URL recorded** in `[project.urls]`, `README.md`, and the
   `CHANGELOG.md` link definitions.
5. **Security channel live:** private GitHub vulnerability reporting, documented
   in `SECURITY.md`.
6. **Changelog dated:** `## [0.1.0] - 2026-10-03`.
7. **CI matrix** defined in `.github/workflows/ci.yml`. It must pass on the tag
   before the upload job can run.

### Releasing

```console
$ git tag -a v0.1.0 -m "Release 0.1.0"
$ git push origin main --tags
```

Then on GitHub: **Actions → Release → Run workflow**, set `version` to `0.1.0`,
tick `authorization_confirmed`, and approve the `pypi` environment reviewer
prompt. Trusted publishing must already be configured on the PyPI side for this
repository and workflow.

The release workflow is manual on purpose, and its gates are asserted by
`tests/test_docs.py`, so a future edit cannot quietly turn a guarded publish into
an automatic one.

## Rollback

Before publication there is nothing to roll back. After a release:

1. **Yank the version** on the index, which prevents new installs without
   breaking existing ones:

   ```console
   $ python -m pip install twine requests
   $ python -c "import requests; requests.post('https://pypi.org/pypi/sciresult/0.1.0/json', auth=('__token__', '<token>'), json={'yanked': True, 'yanked_reason': 'reason'})"
   ```

   Prefer the PyPI web interface or the index's own API over a hand-written
   request, and use an API token rather than a password. A yanked version
   remains installable by existing users, which is the point: they need to be
   able to see why it was withdrawn.

2. **Publish a fixed version** rather than reusing the yanked one. Yanked
   versions cannot be reused, so there is no in-place repair: a bad `0.1.0`
   means `0.1.1`.

3. **Revert the offending commit** if the bug is in the code, and record the
   yank and the fix in `CHANGELOG.md` under the correct category.

4. **Do not delete the release.** Deleting loses the provenance that makes a
   yanked version useful to users who already depend on it.

If a release is found to be wrong about numerical semantics rather than about
packaging, that is a `API changes` or `Numerical semantic changes` entry, it
needs a version bump, and the changelog must state which previously reported
verdicts were affected.