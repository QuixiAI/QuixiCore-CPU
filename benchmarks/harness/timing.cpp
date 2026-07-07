#include "harness/timing.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace qcb {
namespace {

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point t0, Clock::time_point t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

TimingResult time_thunk(const std::function<void()>& fn, int warmup, int iters,
                        double min_sample_ms) {
  // Warm up by call count AND wall time: re-touches pages, warms caches and
  // branch predictors, and lets the OS ramp clocks out of idle.
  const auto warm_start = Clock::now();
  int calls = 0;
  while (calls < warmup || ms_between(warm_start, Clock::now()) < 50.0) {
    fn();
    ++calls;
  }

  // Estimate one call to size the batch.
  const auto est_start = Clock::now();
  fn();
  const double est_ms = std::max(ms_between(est_start, Clock::now()), 1e-6);

  // Batch calls per sample so each sample spans at least min_sample_ms.
  // The Metal harness caps the batch at 64 to bound GPU wall time; on CPU the
  // batch exists to defeat clock-read and loop overhead for sub-microsecond
  // kernels, which needs a much larger cap.
  const int batch = std::clamp(
      static_cast<int>(std::ceil(min_sample_ms / est_ms)), 1, 4096);

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    const auto s0 = Clock::now();
    for (int j = 0; j < batch; ++j) {
      fn();
    }
    samples.push_back(ms_between(s0, Clock::now()) / batch);
  }

  std::sort(samples.begin(), samples.end());
  const int n = static_cast<int>(samples.size());
  const double median = (n % 2 == 1)
                            ? samples[static_cast<size_t>(n / 2)]
                            : 0.5 * (samples[static_cast<size_t>(n / 2 - 1)] +
                                     samples[static_cast<size_t>(n / 2)]);

  // p20/p80 use the Metal harness's exact index selection so percentile
  // fields stay comparable across backends.
  double p20 = median;
  double p80 = median;
  if (n > 1) {
    p20 = samples[static_cast<size_t>(
        std::max(0, static_cast<int>(0.20 * n) - 1))];
    p80 = samples[static_cast<size_t>(
        std::min(n - 1, static_cast<int>(0.80 * n)))];
  }

  double mean = 0.0;
  for (const double s : samples) {
    mean += s;
  }
  mean /= n;
  double var = 0.0;
  for (const double s : samples) {
    var += (s - mean) * (s - mean);
  }
  var /= n;
  const double cv = mean > 0.0 ? std::sqrt(var) / mean : 0.0;

  return {median, p20, p80, cv, batch};
}

}  // namespace qcb
