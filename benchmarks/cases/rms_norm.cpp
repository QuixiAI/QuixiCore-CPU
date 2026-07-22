// rms_norm: contract-kernel benchmark case through the public rms_norm
// entry point.
//
// Shape provenance: umbrella decode_small family (batch {1,2,4} x hidden
// {2048,4096}, sequence 1 -> rows = batch), plus larger row counts as
// prefill-flavored stress shapes.
//
// Baseline per the three-baseline rule: ref_scalar (the library scalar
// reference, float64 accumulation). Memory-bound; judge gbps against the
// mem_triad in-cache/DRAM ladder for the matching working set.

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "harness/shapes.h"
#include "kernels/norms/rms_norm.h"
#include "quixicore_cpu/rms_norm.h"

namespace qcb {
namespace {

using quixicore_cpu::Status;

constexpr float kEps = 1e-5f;

class Rng {
 public:
  float next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return static_cast<float>(state_ >> 8) * (2.0f / 16777216.0f) - 1.0f;
  }

 private:
  uint32_t state_ = 0x7FEB352Du;
};

struct Buffers {
  AlignedBuffer<float> x;
  AlignedBuffer<float> w;
  AlignedBuffer<float> y;
  long long rows = 0;
  long long hidden = 0;
};

CaseDecl make_rms_norm_decl(long long rows, long long hidden) {
  CaseDecl decl;
  decl.kernel = "rms_norm";
  decl.variant = "R" + std::to_string(rows) + "_H" + std::to_string(hidden);
  decl.shape = {{"rows", rows}, {"hidden", hidden}};
  decl.dtype = "f32";
  decl.notes =
      std::string("public rms_norm, variant ") + quixicore_cpu::rms_norm_variant();
  // Conservative traffic: read x, write y, read weight once.
  decl.bytes_moved =
      4.0 * (2.0 * static_cast<double>(rows) * hidden + hidden);
  decl.make = [rows, hidden]() {
    auto bufs = std::make_shared<Buffers>();
    bufs->rows = rows;
    bufs->hidden = hidden;
    bufs->x = aligned_alloc_array<float>(rows * hidden);
    bufs->w = aligned_alloc_array<float>(hidden);
    bufs->y = aligned_alloc_array<float>(rows * hidden);
    Rng rng;
    for (long long i = 0; i < rows * hidden; ++i) {
      bufs->x.get()[i] = rng.next();
    }
    for (long long j = 0; j < hidden; ++j) {
      bufs->w.get()[j] = rng.next();
    }
    for (long long i = 0; i < rows * hidden; ++i) {
      bufs->y.get()[i] = 0.0f;
    }

    CaseBody body;
    body.target = [bufs]() {
      if (quixicore_cpu::rms_norm(bufs->x.get(), bufs->w.get(), bufs->y.get(),
                                  bufs->rows, bufs->hidden, kEps) !=
          Status::kOk) {
        throw std::runtime_error("rms_norm failed");
      }
      do_not_optimize(bufs->y.get());
    };
    body.baselines.emplace_back("ref_scalar", [bufs]() {
      quixicore_cpu::norms::rms_norm_ref(bufs->x.get(), bufs->w.get(),
                                         bufs->y.get(), bufs->rows,
                                         bufs->hidden, kEps);
      do_not_optimize(bufs->y.get());
    });
    body.check = [bufs]() {
      if (quixicore_cpu::rms_norm(bufs->x.get(), bufs->w.get(), bufs->y.get(),
                                  bufs->rows, bufs->hidden, kEps) !=
          Status::kOk) {
        throw std::runtime_error("rms_norm failed");
      }
      const float* x = bufs->x.get();
      const float* w = bufs->w.get();
      const float* y = bufs->y.get();
      CheckResult check;
      for (long long r = 0; r < bufs->rows; ++r) {
        double sumsq = 0.0;
        for (long long j = 0; j < bufs->hidden; ++j) {
          const double v = x[r * bufs->hidden + j];
          sumsq += v * v;
        }
        const double scale =
            1.0 / std::sqrt(sumsq / static_cast<double>(bufs->hidden) +
                            static_cast<double>(kEps));
        for (long long j = 0; j < bufs->hidden; ++j) {
          const double ref = static_cast<double>(x[r * bufs->hidden + j]) *
                             w[j] * scale;
          check_value(check, y[r * bufs->hidden + j], ref,
                      kFp32Tolerance);
        }
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_rms_norm_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  using Shape = std::pair<long long, long long>;  // (rows, hidden)
  static const std::vector<Shape> kSmoke = {{1, 4096}};
  static const std::vector<Shape> kQuick = {
      {1, 2048}, {1, 4096}, {4, 4096}, {512, 4096}};
  static const std::vector<Shape> kComprehensive = [] {
    std::vector<Shape> shapes;
    for (const long long rows : kDecodeSmallBatch) {
      for (const long long hidden : kDecodeSmallHidden) {
        shapes.emplace_back(rows, hidden);
      }
    }
    shapes.emplace_back(512, 2048);
    shapes.emplace_back(512, 4096);
    shapes.emplace_back(4096, 4096);
    return shapes;
  }();

  for (const auto& [rows, hidden] :
       pick(ctx.preset, kSmoke, kQuick, kComprehensive)) {
    out.push_back(make_rms_norm_decl(rows, hidden));
  }
}

}  // namespace qcb
