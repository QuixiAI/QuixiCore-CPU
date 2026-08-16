# Performance Working Directory

- `perf.md` — the operating guide: how to measure and decide.
- `findings.md` — established findings; read before proposing an experiment.
- `backlog.md` — the idea beam; pick the next experiment here.
- `optimization_status.md` — the append-only experiment notebook.
- `baseline_status.md` — environment, standing gates, latest snapshot.
- `harness/run_bench.sh` — the bench entrypoint (wraps `scripts/bench` with
  provenance capture, verdict/noise guards, and a pre-filled notebook entry).
- `baselines/` — committed curated baselines per host fingerprint.
- `results/` — raw run output (`run.json`, `results.jsonl`, `summary.md` per
  run; git-ignored).

Build the harness with `cmake --preset perf && cmake --build --preset perf`;
`scripts/bench` finds the binary under `build/perf/`. Every recorded run
carries CPU model, core count, ISA target, OS, compiler and flags, thread
count, command line, warmups, iterations, median, and variance or min/max —
see the umbrella's `docs/benchmarking.md` for the schema-1 reporting format.
