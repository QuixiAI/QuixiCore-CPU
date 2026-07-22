// Thread pool correctness: exact range coverage, small-count inlining,
// nesting fallback, and the bit-exactness guarantee — threaded kernels must
// produce identical outputs at any thread count.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/rms_norm.h"
#include "quixicore_cpu/threading.h"
#include "src/threading/thread_pool.h"

// Tests must fail in any build configuration, so no assert()/NDEBUG.
#define REQUIRE(cond)                                                     \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << "FAILED: " #cond " at " << __FILE__ << ":" << __LINE__ \
                << '\n';                                                  \
      return 1;                                                           \
    }                                                                     \
  } while (0)

namespace {

class Rng {
 public:
  explicit Rng(uint32_t seed) : state_(seed) {}
  float next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return static_cast<float>(state_ >> 8) * (2.0f / 16777216.0f) - 1.0f;
  }

 private:
  uint32_t state_;
};

}  // namespace

int main() {
  using quixicore_cpu::QuantFormat;
  using quixicore_cpu::Status;

  REQUIRE(quixicore_cpu::num_threads() == 1);
  quixicore_cpu::set_num_threads(0);  // clamps to 1
  REQUIRE(quixicore_cpu::num_threads() == 1);
  quixicore_cpu::set_num_threads(4);
  REQUIRE(quixicore_cpu::num_threads() == 4);

  // A freshly resized pool must not replay an old generation or skip the
  // first job while its worker threads are still starting.
  for (int repetition = 0; repetition < 32; ++repetition) {
    quixicore_cpu::set_num_threads(1);
    quixicore_cpu::set_num_threads(4);
    std::vector<int> hits(257, 0);
    quixicore_cpu::threading::parallel_ranges(
        257, 1, [&](long long begin, long long end, int) {
          for (long long i = begin; i < end; ++i) {
            hits[static_cast<std::size_t>(i)] += 1;
          }
        });
    for (const int hit : hits) {
      REQUIRE(hit == 1);
    }
  }

  // Every index covered exactly once; ranges are disjoint so plain writes
  // are race-free.
  for (const long long count : {0LL, 1LL, 3LL, 4LL, 1000LL, 1001LL}) {
    std::vector<int> hits(static_cast<size_t>(count), 0);
    quixicore_cpu::threading::parallel_ranges(
        count, 1, [&](long long b, long long e, int) {
          for (long long i = b; i < e; ++i) {
            hits[static_cast<size_t>(i)] += 1;
          }
        });
    for (const int h : hits) {
      REQUIRE(h == 1);
    }
  }

  // min_per_chunk keeps small counts inline (single worker index 0).
  {
    int calls = 0;
    int worker_seen = -1;
    long long begin_seen = -1;
    long long end_seen = -1;
    quixicore_cpu::threading::parallel_ranges(
        7, 8, [&](long long b, long long e, int w) {
          ++calls;
          worker_seen = w;
          begin_seen = b;
          end_seen = e;
        });
    REQUIRE(calls == 1);
    REQUIRE(worker_seen == 0);
    REQUIRE(begin_seen == 0);
    REQUIRE(end_seen == 7);
  }

  // Nested calls execute inline instead of deadlocking.
  {
    bool inner_ran = false;
    quixicore_cpu::threading::parallel_ranges(
        4, 1, [&](long long, long long, int) {
          quixicore_cpu::threading::parallel_ranges(
              2, 1, [&](long long, long long, int) { inner_ran = true; });
        });
    REQUIRE(inner_ran);
  }

  // Bit-exactness across thread counts: qgemv q8_0.
  {
    const long long n = 257;
    const long long k = 1024;
    Rng rng(0xA341316Cu);
    std::vector<float> w(static_cast<size_t>(n * k));
    std::vector<float> x(static_cast<size_t>(k));
    for (auto& v : w) {
      v = rng.next();
    }
    for (auto& v : x) {
      v = rng.next();
    }
    size_t size = 0;
    REQUIRE(quixicore_cpu::qgemv_packed_size(QuantFormat::kQ8_0, n, k,
                                                  &size) == Status::kOk);
    std::vector<uint8_t> packed(size);
    REQUIRE(quixicore_cpu::qgemv_pack(QuantFormat::kQ8_0, w.data(), n, k,
                                           packed.data()) == Status::kOk);

    std::vector<float> y1(static_cast<size_t>(n));
    std::vector<float> y4(static_cast<size_t>(n));
    quixicore_cpu::set_num_threads(1);
    REQUIRE(quixicore_cpu::qgemv(QuantFormat::kQ8_0, packed.data(),
                                      x.data(), y1.data(), n, k) ==
            Status::kOk);
    quixicore_cpu::set_num_threads(4);
    REQUIRE(quixicore_cpu::qgemv(QuantFormat::kQ8_0, packed.data(),
                                      x.data(), y4.data(), n, k) ==
            Status::kOk);
    REQUIRE(std::memcmp(y1.data(), y4.data(), n * sizeof(float)) == 0);
  }

  // Bit-exactness across thread counts: rms_norm.
  {
    const long long rows = 33;
    const long long hidden = 777;
    Rng rng(0xC2B2AE35u);
    std::vector<float> x(static_cast<size_t>(rows * hidden));
    std::vector<float> w(static_cast<size_t>(hidden));
    for (auto& v : x) {
      v = rng.next();
    }
    for (auto& v : w) {
      v = rng.next();
    }
    std::vector<float> y1(static_cast<size_t>(rows * hidden));
    std::vector<float> y4(static_cast<size_t>(rows * hidden));
    quixicore_cpu::set_num_threads(1);
    REQUIRE(quixicore_cpu::rms_norm(x.data(), w.data(), y1.data(), rows,
                                    hidden, 1e-5f) == Status::kOk);
    quixicore_cpu::set_num_threads(4);
    REQUIRE(quixicore_cpu::rms_norm(x.data(), w.data(), y4.data(), rows,
                                    hidden, 1e-5f) == Status::kOk);
    REQUIRE(std::memcmp(y1.data(), y4.data(),
                        static_cast<size_t>(rows * hidden) * sizeof(float)) ==
            0);
  }

  quixicore_cpu::set_num_threads(1);
  REQUIRE(quixicore_cpu::num_threads() == 1);
  return 0;
}
