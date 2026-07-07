#pragma once

#include <functional>

namespace qcb {

struct TimingResult {
  double ms;      // median per-call latency
  double p20_ms;
  double p80_ms;
  double cv;      // population stdev / mean over samples
  int batch;      // calls per timed sample
};

// Times fn with the QuixiCore measurement discipline (see perf/perf.md):
// warmup by call count AND wall time, then `iters` samples of adaptively
// batched calls sized so each sample spans at least min_sample_ms.
TimingResult time_thunk(const std::function<void()>& fn, int warmup, int iters,
                        double min_sample_ms = 2.0);

}  // namespace qcb
