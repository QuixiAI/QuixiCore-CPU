# Memory

Allocation and layout utilities. Planned responsibilities:

- aligned allocation (cache-line and vector-width alignment) portable across
  the platform allocators,
- packed-weight buffer management for microkernel-friendly layouts,
- scratch/workspace arenas so kernels never allocate on the hot path.

Empty apart from this note until the first packing path lands.
