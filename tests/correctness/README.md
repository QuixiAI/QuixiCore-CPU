# Correctness Tests

Contract correctness tests live here once kernel work starts. Each supported
operation is validated against the umbrella spec (`specs/kernels/`) and
tolerance registry (`registry/tolerances.yaml`):

- ISA variants are compared against the scalar reference (`_ref.cpp`) on the
  executing hardware, across dtypes, layouts, and edge shapes.
- References are compared against shared test vectors from the umbrella
  `test-vectors/` organization where they exist.

A kernel family cannot be claimed supported without coverage here.
