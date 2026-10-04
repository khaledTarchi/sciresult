# Security policy

## Supported versions

`sciresult` has not been released. There is no released version to support.

| Version | Supported |
| --- | --- |
| 0.1.0 | Yes |

Each release receives security fixes for as long as it remains the current
version. Older versions are not patched.

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Report privately through GitHub's private vulnerability reporting:

    https://github.com/khaledTarchi/sciresult/security/advisories/new

That opens a private advisory visible only to the maintainer, and it does not
require a public issue. Please do not open a public issue for a security problem.

If you would rather not use GitHub, contact the maintainer directly at
`khaledTarchi@users.noreply.github.com`.

When reporting, include:

- the version or commit you tested,
- your Python, NumPy, and Xarray versions,
- a minimal reproduction, ideally as a `compare()` call,
- what you expected and what happened.

Expect an acknowledgement within a few days, and either a fix or an explanation
of why the report is not a vulnerability.

## Threat model

`sciresult` is a library that runs inside your process on data you already have.
It is not a service, it opens no sockets, spawns no processes, reads no
environment variables, and writes no files. The realistic risks are narrower than
for most packages:

- **Denial of service through unbounded work.** Input size is the caller's
  choice. See "Resource use" below.
- **Silent wrong answers.** The serious risk is not a crash but a verdict that is
  wrong in a way you do not notice. The design targets this: no repair, no
  alignment, no unit conversion, strict structural checks, and refusal rather
  than a guess.
- **Untrusted data in rendered reports.** Coordinate labels and attribute values
  are copied into the report and rendered as text. They are escaped, so they
  cannot inject terminal control sequences into the report, but a report is not
  sanitised for HTML or for inclusion in a log viewer.

## Trusted-input assumptions

`sciresult` assumes its inputs are well-formed scientific data arrays, not
hostile ones. Specifically:

- `DataArray.data` is expected to be an exact `numpy.ndarray` or `numpy.memmap`.
  Subclasses, masked arrays, and Dask or CuPy arrays are rejected during
  admission rather than coerced.
- Coordinate values are trusted to be ordinary numeric, datetime, or text data.
  The library reads them, bounds their display length, and escapes control
  characters, but it does not defend against adversarially constructed dtype
  metadata.
- Coordinate and attribute *names* are trusted to be text. A non-text name is
  rejected with an explicit error code.

The library performs no parsing of untrusted formats. If you need to compare
data from an untrusted source, validate and parse it in your own code first.

## No sandbox

`sciresult` is not a sandbox and must not be used as one. It executes no
user-supplied code, but it also cannot protect the calling process from a hostile
environment. If `numpy` or `xarray` themselves are compromised, this library
offers no containment.

## Report privacy

A report contains:

- the operand `name` attributes, dimension names, shapes, and `dtype.str`,
- the data dtype of each operand,
- the tolerance options you passed,
- up to `max_examples` failure examples, each holding coordinate labels and the
  two operand values at that position,
- the largest finite absolute error, with its label,
- structural issue messages, which quote coordinate names and the differing
  values of guarded attributes.

**A report can therefore contain your data and your coordinate labels.** Before
sharing a report in an issue, a log, or a ticket, check it for anything
sensitive. `report.format()` renders coordinate labels verbatim apart from
control-character escaping and display truncation, and truncation is announced
rather than silent.

The library performs no network access and collects no telemetry. There is no
analytics, no crash reporting, and no usage measurement anywhere in the package
or its tests.

## Resource use

Comparison time is linear in the number of array elements, plus a bounded cost
per failure. The numerical scan works through fixed-size blocks and allocates
per-block temporaries, so peak memory tracks the block size rather than the
array size; `benchmarks/memory_probe.py` measures this. Diagnostic storage is
bounded by `max_examples`.

There is no global time limit, no cancellation interface, and no hard
process-wide memory guarantee. A caller comparing an enormous array accepts the
time that implies.

## I/O caveats

`sciresult` performs no file or network I/O. If you pass a `numpy.memmap`, the
mapping is read through the operating system page cache and the library issues no
`flush`, no `fsync`, and no write of any kind. `np.memmap` in write mode is
accepted for reading; the library does not modify it.

## Errors are not secrets

Error messages are built from the library's own findings and the stable reason
codes documented in `docs/semantics.md`. The library does not call arbitrary
`repr()` on user objects while constructing an error, so an error message cannot
leak an object's memory layout or internal state through a formatting hook. It
can still contain coordinate names, attribute values, and dtype strings, which
are the same fields a report contains.