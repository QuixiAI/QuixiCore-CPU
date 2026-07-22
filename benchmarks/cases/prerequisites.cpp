// Focused evidence for CPU-private packed panels and retained workspaces.
// Panel preparation happens once during case construction. The target reuses
// both the prepared weights and caller-owned workspace; the baseline invokes
// the canonical qgemm path, which traverses the contract layout once per M row.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "harness/alloc.h"
#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/packed_weights.h"
#include "quixicore_cpu/qgemm.h"
#include "quixicore_cpu/qgemv.h"
#include "quixicore_cpu/workspace.h"

namespace qcb {
namespace {

using quixicore_cpu::CpuPackedWeights;
using quixicore_cpu::QuantFormat;
using quixicore_cpu::Status;
using quixicore_cpu::Workspace;

class Rng {
 public:
  float next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return static_cast<float>(state_ >> 8) * (2.0f / 16777216.0f) - 1.0f;
  }

 private:
  std::uint32_t state_ = 0xD1B54A35u;
};

struct Buffers {
  AlignedBuffer<std::uint8_t> contract_weights;
  AlignedBuffer<float> x;
  AlignedBuffer<float> target;
  AlignedBuffer<float> baseline;
  CpuPackedWeights weights;
  Workspace workspace;
  long long m = 0;
  long long n = 0;
  long long k = 0;
};

CaseDecl make_prepacked_qgemm(long long m, long long n, long long k) {
  CaseDecl decl;
  decl.kernel = "prerequisites";
  decl.variant = "prepacked_q4_0_M" + std::to_string(m) + "_N" +
                 std::to_string(n) + "_K" + std::to_string(k);
  decl.shape = {{"m", m}, {"n", n}, {"k", k}};
  decl.format = "q4_0";
  decl.notes =
      "runtime CPU row panel with a warmed caller-owned workspace versus "
      "canonical qgemm";
  decl.flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
               static_cast<double>(k);
  decl.make = [m, n, k]() {
    std::size_t contract_bytes = 0;
    if (quixicore_cpu::qgemv_packed_size(QuantFormat::kQ4_0, n, k,
                                        &contract_bytes) != Status::kOk) {
      throw std::runtime_error("q4_0 packed-size query failed");
    }

    auto buffers = std::make_shared<Buffers>();
    buffers->m = m;
    buffers->n = n;
    buffers->k = k;
    buffers->contract_weights = aligned_alloc_array<std::uint8_t>(
        static_cast<long long>(contract_bytes));
    buffers->x = aligned_alloc_array<float>(m * k);
    buffers->target = aligned_alloc_array<float>(m * n);
    buffers->baseline = aligned_alloc_array<float>(m * n);

    {
      auto dense = aligned_alloc_array<float>(n * k);
      Rng rng;
      for (long long index = 0; index < n * k; ++index) {
        dense.get()[index] = 0.25f * rng.next();
      }
      if (quixicore_cpu::qgemv_pack(
              QuantFormat::kQ4_0, dense.get(), n, k,
              buffers->contract_weights.get()) != Status::kOk) {
        throw std::runtime_error("q4_0 packing failed");
      }
    }
    Rng rng;
    for (long long index = 0; index < m * k; ++index) {
      buffers->x.get()[index] = rng.next();
    }
    if (buffers->weights.prepare(QuantFormat::kQ4_0,
                                 buffers->contract_weights.get(), n, k) !=
        Status::kOk) {
      throw std::runtime_error("CPU panel preparation failed");
    }

    auto target = [buffers]() {
      if (quixicore_cpu::qgemm_prepacked(
              buffers->weights, buffers->x.get(), buffers->target.get(),
              buffers->m, &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("qgemm_prepacked failed");
      }
      do_not_optimize(buffers->target.get());
    };
    auto baseline = [buffers]() {
      if (quixicore_cpu::qgemm(
              QuantFormat::kQ4_0, buffers->contract_weights.get(),
              buffers->x.get(), buffers->baseline.get(), buffers->m,
              buffers->n, buffers->k) != Status::kOk) {
        throw std::runtime_error("canonical qgemm baseline failed");
      }
      do_not_optimize(buffers->baseline.get());
    };

    CaseBody body;
    body.target = target;
    body.baselines.emplace_back("canonical_qgemm", baseline);
    body.check = [buffers, target, baseline]() {
      target();
      baseline();
      CheckResult check;
      for (long long index = 0; index < buffers->m * buffers->n; ++index) {
        check_value(check, buffers->target.get()[index],
                    buffers->baseline.get()[index],
                    Tolerance{3e-5, 3e-5});
      }
      return check;
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_prerequisites_cases(const BuildCtx& ctx,
                               std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_prepacked_qgemm(3, 64, 64));
    return;
  }
  out.push_back(make_prepacked_qgemm(16, 1024, 1408));
  if (ctx.preset == Preset::kComprehensive) {
    out.push_back(make_prepacked_qgemm(16, 4096, 4096));
  }
}

}  // namespace qcb
