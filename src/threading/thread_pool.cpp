#include "src/threading/thread_pool.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "quixicore_cpu/threading.h"

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace quixicore_cpu::threading {
namespace {

thread_local bool tls_in_parallel = false;

std::pair<long long, long long> chunk_range(long long count, int parts,
                                            int index) {
  const long long base = count / parts;
  const long long rem = count % parts;
  const long long begin =
      index * base + std::min<long long>(index, rem);
  return {begin, begin + base + (index < rem ? 1 : 0)};
}

class Pool {
 public:
  static Pool& instance() {
    static Pool pool;
    return pool;
  }

  int total() {
    std::lock_guard<std::mutex> control(control_m_);
    return total_;
  }

  void resize(int total) {
    total = std::clamp(total, 1, 512);
    std::lock_guard<std::mutex> control(control_m_);
    if (total == total_) {
      return;
    }
    stop_workers();
    total_ = total;
    workers_.reserve(static_cast<size_t>(total - 1));
    uint64_t initial_generation = 0;
    {
      std::lock_guard<std::mutex> lock(m_);
      initial_generation = generation_;
    }
    for (int index = 1; index < total; ++index) {
      workers_.emplace_back([this, index, initial_generation] {
        worker_loop(index, initial_generation);
      });
    }
  }

  void run(long long count, long long min_per_chunk, RangeFn fn, void* ctx) {
    if (count <= 0) {
      return;
    }
    min_per_chunk = std::max<long long>(min_per_chunk, 1);
    if (tls_in_parallel) {  // no nesting; execute inline
      fn(ctx, 0, count, 0);
      return;
    }
    std::lock_guard<std::mutex> control(control_m_);
    const int parts = static_cast<int>(std::clamp<long long>(
        count / min_per_chunk, 1, static_cast<long long>(total_)));
    if (parts == 1) {
      tls_in_parallel = true;
      fn(ctx, 0, count, 0);
      tls_in_parallel = false;
      return;
    }
    {
      std::lock_guard<std::mutex> lock(m_);
      job_fn_ = fn;
      job_ctx_ = ctx;
      job_count_ = count;
      job_parts_ = parts;
      pending_ = total_ - 1;
      ++generation_;
    }
    cv_start_.notify_all();

    const auto [begin, end] = chunk_range(count, parts, 0);
    tls_in_parallel = true;
    fn(ctx, begin, end, 0);
    tls_in_parallel = false;

    std::unique_lock<std::mutex> lock(m_);
    cv_done_.wait(lock, [this] { return pending_ == 0; });
    job_fn_ = nullptr;
  }

 private:
  ~Pool() {
    std::lock_guard<std::mutex> control(control_m_);
    stop_workers();
  }

  void stop_workers() {
    {
      std::lock_guard<std::mutex> lock(m_);
      stop_ = true;
      ++generation_;
    }
    cv_start_.notify_all();
    for (auto& worker : workers_) {
      worker.join();
    }
    workers_.clear();
    std::lock_guard<std::mutex> lock(m_);
    stop_ = false;
  }

  void worker_loop(int index, uint64_t initial_generation) {
#if defined(__APPLE__)
    // Bias workers toward performance cores alongside the caller.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
    // A worker can be created after the pool has already completed jobs. Use
    // the generation captured before thread creation: sampling it here could
    // race with the first job and cause the worker to skip that job.
    uint64_t seen = initial_generation;
    while (true) {
      RangeFn fn = nullptr;
      void* ctx = nullptr;
      long long count = 0;
      int parts = 0;
      {
        std::unique_lock<std::mutex> lock(m_);
        cv_start_.wait(lock,
                       [&] { return stop_ || generation_ != seen; });
        if (stop_) {
          return;
        }
        seen = generation_;
        fn = job_fn_;
        ctx = job_ctx_;
        count = job_count_;
        parts = job_parts_;
      }
      if (fn != nullptr && index < parts) {
        const auto [begin, end] = chunk_range(count, parts, index);
        tls_in_parallel = true;
        fn(ctx, begin, end, index);
        tls_in_parallel = false;
      }
      {
        std::lock_guard<std::mutex> lock(m_);
        if (--pending_ == 0) {
          cv_done_.notify_one();
        }
      }
    }
  }

  std::mutex control_m_;  // serializes resize and parallel regions
  int total_ = 1;
  std::vector<std::thread> workers_;

  std::mutex m_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  uint64_t generation_ = 0;
  int pending_ = 0;
  bool stop_ = false;
  RangeFn job_fn_ = nullptr;
  void* job_ctx_ = nullptr;
  long long job_count_ = 0;
  int job_parts_ = 0;
};

}  // namespace

void parallel_ranges_impl(long long count, long long min_per_chunk,
                          RangeFn fn, void* ctx) {
  Pool::instance().run(count, min_per_chunk, fn, ctx);
}

}  // namespace quixicore_cpu::threading

namespace quixicore_cpu {

void set_num_threads(int n) { threading::Pool::instance().resize(n); }

int num_threads() { return threading::Pool::instance().total(); }

}  // namespace quixicore_cpu
