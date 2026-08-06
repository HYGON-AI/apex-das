# Contributing

Thank you for contributing to apex-das.

## Before submitting

1. Confirm that contributed code may be redistributed under the repository license.
2. Retain every upstream copyright, license header, and attribution notice.
3. Document copied or adapted code in THIRD_PARTY_NOTICES.md.
4. Do not include credentials, private keys, internal addresses, customer data, or confidential material.
5. Use HCU for user-visible hardware naming. Keep ROCm/HIP names only where required for API or toolchain compatibility.
6. Add or update tests for functional changes.

## Commit messages

Use the following format:

    type(scope): concise change

Recommended types are feat, fix, perf, refactor, docs, test, build, and chore.
Describe compatibility or performance impact in the body and reference an issue
when one exists.

Examples:

    feat(gds): add hipFile and POSIX fallback
    fix(transducer): use a single HCU warp for scalar backward

## License

By submitting a contribution, you confirm that you have the right to submit it
under the applicable repository and per-file licenses.

