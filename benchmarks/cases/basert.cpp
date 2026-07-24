// Live Metal BaseRT additions, exercised on BF16 storage with scalar oracles.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness/case.h"
#include "harness/donotopt.h"
#include "quixicore_cpu/float_storage.h"

namespace qcb {
namespace {

using quixicore_cpu::FloatStorageInput;
using quixicore_cpu::FloatStorageOutput;
using quixicore_cpu::FloatStorageType;
using quixicore_cpu::Status;

struct Buffers {
  std::vector<std::uint16_t> x, w, b, extra, out, reference;
  std::vector<int> ids0, ids1, mask_out, mask_reference;
  std::vector<float> running, maximum, out_f32, reference_f32;
  quixicore_cpu::FloatStorageWorkspace workspace;
};

float bf16(const std::vector<std::uint16_t>& values, long long index) {
  return quixicore_cpu::bf16_to_float(values[static_cast<std::size_t>(index)]);
}

void set_bf16(std::vector<std::uint16_t>& values, long long index,
              float value) {
  values[static_cast<std::size_t>(index)] =
      quixicore_cpu::float_to_bf16(value);
}

void fill_wave(std::vector<std::uint16_t>& values, float scale, float step,
               bool cosine = false) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    const float phase = step * static_cast<float>(i + 1);
    values[i] = quixicore_cpu::float_to_bf16(
        scale * (cosine ? std::cos(phase) : std::sin(phase)));
  }
}

CheckResult check_bf16(const Buffers& buffers) {
  CheckResult check;
  for (std::size_t i = 0; i < buffers.out.size(); ++i) {
    check_value(check, quixicore_cpu::bf16_to_float(buffers.out[i]),
                quixicore_cpu::bf16_to_float(buffers.reference[i]),
                kQuantizedTolerance);
  }
  return check;
}

FloatStorageInput in(const std::vector<std::uint16_t>& values) {
  return {values.data(), FloatStorageType::kBF16,
          static_cast<long long>(values.size())};
}

FloatStorageOutput out(std::vector<std::uint16_t>& values) {
  return {values.data(), FloatStorageType::kBF16,
          static_cast<long long>(values.size())};
}

void reserve(Buffers& buffers, std::size_t count) {
  if (buffers.workspace.reserve(count) != Status::kOk) {
    throw std::runtime_error("BaseRT benchmark workspace allocation failed");
  }
}

CaseDecl make_calibration(long long tokens, long long channels) {
  CaseDecl decl;
  decl.kernel = "basert_aux";
  decl.variant = "calibration_T" + std::to_string(tokens) + "_C" +
                 std::to_string(channels);
  decl.shape = {{"T", tokens}, {"C", channels}};
  decl.dtype = "bf16";
  decl.format = "channel_absmax";
  decl.bytes_moved = 2.0 * tokens * channels + 4.0 * channels;
  decl.notes = "per-channel calibration reduction with running-state merge";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(tokens * channels);
    buffers->maximum.resize(channels);
    buffers->running.resize(channels);
    fill_wave(buffers->x, 3.0f, 0.0017f);
    for (long long c = 0; c < channels; ++c) {
      buffers->running[c] = 0.2f + 0.0001f * static_cast<float>(c);
    }
    reserve(*buffers, buffers->x.size());
    auto reference = [=]() {
      for (long long c = 0; c < channels; ++c) {
        float maximum = std::fabs(buffers->running[c]);
        for (long long t = 0; t < tokens; ++t) {
          maximum = std::max(maximum, std::fabs(bf16(buffers->x, t * channels + c)));
        }
        buffers->maximum[c] = maximum;
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::calibration_absmax_storage(
              in(buffers->x), buffers->running.data(), buffers->maximum.data(),
              tokens, channels, &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("calibration target failed");
      }
      do_not_optimize(buffers->maximum.data());
    };
    body.baselines.emplace_back("scalar_channel_reduce", [=]() {
      reference();
      do_not_optimize(buffers->maximum.data());
    });
    body.check = [=]() {
      std::vector<float> target(channels);
      if (quixicore_cpu::calibration_absmax_storage(
              in(buffers->x), buffers->running.data(), target.data(), tokens,
              channels, &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("calibration check failed");
      }
      reference();
      CheckResult check;
      for (long long c = 0; c < channels; ++c) {
        check_value(check, target[c], buffers->maximum[c], kFp32Tolerance);
      }
      return check;
    };
    return body;
  };
  return decl;
}

CaseDecl make_softcap(long long tokens, long long vocab) {
  const long long count = tokens * vocab;
  CaseDecl decl;
  decl.kernel = "basert_aux";
  decl.variant = "softcap_T" + std::to_string(tokens) + "_V" +
                 std::to_string(vocab);
  decl.shape = {{"T", tokens}, {"V", vocab}};
  decl.dtype = "bf16";
  decl.format = "final_logit_softcap";
  decl.bytes_moved = 4.0 * count;
  decl.notes = "Gemma final-logit cap*tanh(logit/cap)";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(count);
    buffers->out.resize(count);
    buffers->reference.resize(count);
    fill_wave(buffers->x, 20.0f, 0.00031f);
    reserve(*buffers, 2 * buffers->x.size());
    auto reference = [=]() {
      for (long long i = 0; i < count; ++i) {
        set_bf16(buffers->reference, i, 30.0f * std::tanh(bf16(buffers->x, i) / 30.0f));
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::logits_softcap_storage(
              in(buffers->x), out(buffers->out), 30.0f,
              &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("softcap target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_mul_tanh_div", [=]() {
      reference();
      do_not_optimize(buffers->reference.data());
    });
    body.check = [=]() {
      body.target();
      reference();
      return check_bf16(*buffers);
    };
    return body;
  };
  return decl;
}

CaseDecl make_value_clip(long long tokens, long long channels) {
  const long long count = tokens * channels;
  CaseDecl decl;
  decl.kernel = "basert_aux";
  decl.variant = "value_clip_B1_T" + std::to_string(tokens) + "_C" +
                 std::to_string(channels);
  decl.shape = {{"B", 1}, {"T", tokens}, {"C", channels}};
  decl.dtype = "bf16";
  decl.format = "scalar_bounds_clip";
  decl.bytes_moved = 4.0 * count;
  decl.notes = "arbitrary-shape scalar clamp with infinite-bound semantics";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(count);
    buffers->out.resize(count);
    buffers->reference.resize(count);
    fill_wave(buffers->x, 8.0f, 0.00073f);
    reserve(*buffers, 2 * count);
    auto reference = [=]() {
      for (long long i = 0; i < count; ++i) {
        set_bf16(buffers->reference, i,
                 std::clamp(bf16(buffers->x, i), -3.0f, 5.0f));
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::value_clip_storage(
              in(buffers->x), out(buffers->out), -3.0f, 5.0f,
              &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("value clip target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_clamp", [=]() {
      reference();
      do_not_optimize(buffers->reference.data());
    });
    body.check = [=]() {
      body.target();
      reference();
      return check_bf16(*buffers);
    };
    return body;
  };
  return decl;
}

CaseDecl make_embedding(long long tokens, long long hidden) {
  constexpr long long token_vocab = 4096;
  constexpr long long type_vocab = 4;
  CaseDecl decl;
  decl.kernel = "basert_embedding";
  decl.variant = "types_T" + std::to_string(tokens) + "_D" +
                 std::to_string(hidden);
  decl.shape = {{"T", tokens}, {"D", hidden}};
  decl.dtype = "bf16";
  decl.format = "token_type";
  decl.bytes_moved = 6.0 * tokens * hidden;
  decl.notes = "bounds-checked token/type gather, scaled add";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(token_vocab * hidden);
    buffers->w.resize(type_vocab * hidden);
    buffers->out.resize(tokens * hidden);
    buffers->reference.resize(tokens * hidden);
    buffers->ids0.resize(tokens);
    buffers->ids1.resize(tokens);
    fill_wave(buffers->x, 0.15f, 0.00017f);
    fill_wave(buffers->w, 0.10f, 0.0031f, true);
    for (long long t = 0; t < tokens; ++t) {
      buffers->ids0[t] = static_cast<int>((t * 37 + 11) % token_vocab);
      buffers->ids1[t] = static_cast<int>(t % type_vocab);
    }
    reserve(*buffers, buffers->x.size() + buffers->w.size() +
                          buffers->out.size());
    auto reference = [=]() {
      for (long long t = 0; t < tokens; ++t) {
        for (long long d = 0; d < hidden; ++d) {
          set_bf16(buffers->reference, t * hidden + d,
                   1.25f * bf16(buffers->x, buffers->ids0[t] * hidden + d) +
                       bf16(buffers->w, buffers->ids1[t] * hidden + d));
        }
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::embedding_lookup_types_storage(
              buffers->ids0.data(), buffers->ids1.data(), in(buffers->x),
              in(buffers->w), out(buffers->out), token_vocab, type_vocab,
              tokens, hidden, 1.25f, &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("embedding target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_two_gather_add", [=]() {
      reference();
      do_not_optimize(buffers->reference.data());
    });
    body.check = [=]() {
      body.target();
      reference();
      return check_bf16(*buffers);
    };
    return body;
  };
  return decl;
}

CaseDecl make_masked_pool(long long batch, long long tokens, long long hidden) {
  CaseDecl decl;
  decl.kernel = "basert_embedding";
  decl.variant = "masked_pool_B" + std::to_string(batch) + "_T" +
                 std::to_string(tokens) + "_D" + std::to_string(hidden);
  decl.shape = {{"B", batch}, {"T", tokens}, {"D", hidden}};
  decl.dtype = "bf16";
  decl.format = "masked_rms_l2";
  decl.bytes_moved = 2.0 * (batch * tokens * hidden + batch * hidden);
  decl.notes = "mask-aware mean pool fused with RMSNorm and L2 normalize";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(batch * tokens * hidden);
    buffers->w.resize(hidden);
    buffers->ids0.resize(batch * tokens);
    buffers->out.resize(batch * hidden);
    buffers->reference.resize(batch * hidden);
    fill_wave(buffers->x, 0.2f, 0.00091f);
    fill_wave(buffers->w, 0.6f, 0.0037f, true);
    for (long long i = 0; i < batch * tokens; ++i) buffers->ids0[i] = i % 5 != 0;
    reserve(*buffers, buffers->x.size() + buffers->w.size() + buffers->out.size());
    auto reference = [=]() {
      for (long long b = 0; b < batch; ++b) {
        long long selected = 0;
        for (long long t = 0; t < tokens; ++t) selected += buffers->ids0[b * tokens + t] != 0;
        double mean_square = 0.0;
        for (long long d = 0; d < hidden; ++d) {
          double sum = 0.0;
          for (long long t = 0; t < tokens; ++t) if (buffers->ids0[b * tokens + t]) {
            sum += bf16(buffers->x, (b * tokens + t) * hidden + d);
          }
          const float value = selected == 0 ? 0.0f : static_cast<float>(sum / selected);
          buffers->reference[b * hidden + d] = quixicore_cpu::float_to_bf16(value);
          mean_square += double(value) * value;
        }
        const double inv_rms = 1.0 / std::sqrt(mean_square / hidden + 1.0e-5);
        double l2 = 0.0;
        for (long long d = 0; d < hidden; ++d) {
          const float value = bf16(buffers->reference, b * hidden + d) *
                              static_cast<float>(inv_rms) * bf16(buffers->w, d);
          buffers->reference[b * hidden + d] = quixicore_cpu::float_to_bf16(value);
          l2 += double(value) * value;
        }
        const double inv_l2 = 1.0 / std::sqrt(l2 + 1.0e-12);
        for (long long d = 0; d < hidden; ++d) {
          set_bf16(buffers->reference, b * hidden + d,
                   bf16(buffers->reference, b * hidden + d) * static_cast<float>(inv_l2));
        }
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::masked_mean_pool_rms_l2_storage(
              in(buffers->x), buffers->ids0.data(), in(buffers->w),
              out(buffers->out), batch, tokens, hidden, 1.0e-5f,
              &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("masked pool target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_pool_rms_l2", [=]() {
      reference();
      do_not_optimize(buffers->reference.data());
    });
    body.check = [=]() {
      body.target();
      reference();
      return check_bf16(*buffers);
    };
    return body;
  };
  return decl;
}

CaseDecl make_patch(long long height, long long width, long long channels,
                    long long patch) {
  const long long count = height * width * channels;
  CaseDecl decl;
  decl.kernel = "basert_vision";
  decl.variant = "patchify_H" + std::to_string(height) + "_W" +
                 std::to_string(width) + "_P" + std::to_string(patch);
  decl.shape = {{"B", 1}, {"H", height}, {"W", width}, {"C", channels},
                {"P", patch}};
  decl.dtype = "bf16";
  decl.format = "nhwc_patches";
  decl.bytes_moved = 4.0 * count;
  decl.notes = "NHWC image to transformer patch tokens";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(count);
    buffers->out.resize(count);
    buffers->reference.resize(count);
    fill_wave(buffers->x, 0.2f, 0.0013f);
    reserve(*buffers, 2 * count);
    auto reference = [=]() {
      long long destination = 0;
      for (long long oy = 0; oy < height / patch; ++oy) for (long long ox = 0; ox < width / patch; ++ox) {
        for (long long ky = 0; ky < patch; ++ky) for (long long kx = 0; kx < patch; ++kx) {
          const long long source = ((oy * patch + ky) * width + ox * patch + kx) * channels;
          for (long long c = 0; c < channels; ++c) buffers->reference[destination++] = buffers->x[source + c];
        }
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::extract_patches_2d_storage(
              in(buffers->x), out(buffers->out), 1, height, width, channels,
              patch, patch, patch, patch, 0, 0,
              &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("patch target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_nhwc_patchify", [=]() {
      reference();
      do_not_optimize(buffers->reference.data());
    });
    body.check = [=]() { body.target(); reference(); return check_bf16(*buffers); };
    return body;
  };
  return decl;
}

CaseDecl make_patch3d(long long frames, long long height, long long width,
                      long long channels, long long temporal,
                      long long patch) {
  const long long output_frames = frames / temporal;
  const long long output_height = height / patch;
  const long long output_width = width / patch;
  const long long patch_dim = temporal * patch * patch * channels;
  const long long rows = output_frames * output_height * output_width;
  CaseDecl decl;
  decl.kernel = "basert_vision";
  decl.variant = "patchify3d_T" + std::to_string(frames) + "_H" +
                 std::to_string(height) + "_W" + std::to_string(width) +
                 "_PT" + std::to_string(temporal) + "_P" +
                 std::to_string(patch);
  decl.shape = {{"B", 1}, {"T", frames}, {"H", height}, {"W", width},
                {"C", channels}, {"PT", temporal}, {"P", patch}};
  decl.dtype = "bf16";
  decl.format = "nthwc_patches";
  decl.bytes_moved = 4.0 * frames * height * width * channels;
  decl.notes = "NTHWC temporal/spatial patch extraction";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(frames * height * width * channels);
    buffers->out.resize(rows * patch_dim);
    buffers->reference.resize(rows * patch_dim);
    fill_wave(buffers->x, 0.2f, 0.00037f);
    reserve(*buffers, buffers->x.size() + buffers->out.size());
    auto reference = [=]() {
      long long destination = 0;
      for (long long ot = 0; ot < output_frames; ++ot) {
        for (long long oy = 0; oy < output_height; ++oy) {
          for (long long ox = 0; ox < output_width; ++ox) {
            for (long long kt = 0; kt < temporal; ++kt) {
              for (long long ky = 0; ky < patch; ++ky) {
                for (long long kx = 0; kx < patch; ++kx) {
                  const long long source =
                      ((((ot * temporal + kt) * height + oy * patch + ky) *
                            width +
                        ox * patch + kx) *
                       channels);
                  for (long long c = 0; c < channels; ++c) {
                    buffers->reference[destination++] =
                        buffers->x[source + c];
                  }
                }
              }
            }
          }
        }
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::extract_patches_3d_storage(
              in(buffers->x), out(buffers->out), 1, frames, height, width,
              channels, temporal, patch, patch, temporal, patch, patch, 0, 0,
              0, &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("3d patch target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_nthwc_patchify", [=]() {
      reference();
      do_not_optimize(buffers->reference.data());
    });
    body.check = [=]() {
      body.target();
      reference();
      return check_bf16(*buffers);
    };
    return body;
  };
  return decl;
}

CaseDecl make_interpolate(long long ih, long long iw, long long oh,
                          long long ow, long long channels) {
  CaseDecl decl;
  decl.kernel = "basert_vision";
  decl.variant = "pos_interp_" + std::to_string(ih) + "x" + std::to_string(iw) +
                 "_to_" + std::to_string(oh) + "x" + std::to_string(ow) +
                 "_D" + std::to_string(channels);
  decl.shape = {{"IH", ih}, {"IW", iw}, {"OH", oh}, {"OW", ow}, {"D", channels}};
  decl.dtype = "bf16";
  decl.format = "bilinear_half_pixel";
  decl.bytes_moved = 2.0 * (ih * iw + oh * ow) * channels;
  decl.notes = "bilinear learned-position resize with half-pixel centers";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(ih * iw * channels);
    buffers->out.resize(oh * ow * channels);
    buffers->reference.resize(buffers->out.size());
    fill_wave(buffers->x, 0.2f, 0.00071f);
    reserve(*buffers, buffers->x.size() + buffers->out.size());
    auto reference = [=]() {
      for (long long y = 0; y < oh; ++y) {
        const double fy = std::clamp((y + 0.5) * ih / double(oh) - 0.5, 0.0, double(ih - 1));
        const long long y0 = static_cast<long long>(std::floor(fy)), y1 = std::min(y0 + 1, ih - 1);
        for (long long x = 0; x < ow; ++x) {
          const double fx = std::clamp((x + 0.5) * iw / double(ow) - 0.5, 0.0, double(iw - 1));
          const long long x0 = static_cast<long long>(std::floor(fx)), x1 = std::min(x0 + 1, iw - 1);
          for (long long c = 0; c < channels; ++c) {
            const float a = bf16(buffers->x, (y0 * iw + x0) * channels + c);
            const float b = bf16(buffers->x, (y0 * iw + x1) * channels + c);
            const float d = bf16(buffers->x, (y1 * iw + x0) * channels + c);
            const float e = bf16(buffers->x, (y1 * iw + x1) * channels + c);
            const float top = a + float(fx - x0) * (b - a);
            const float bottom = d + float(fx - x0) * (e - d);
            set_bf16(buffers->reference, (y * ow + x) * channels + c,
                     top + float(fy - y0) * (bottom - top));
          }
        }
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::interpolate_position_2d_storage(
              in(buffers->x), out(buffers->out), ih, iw, oh, ow, channels,
              false, &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("interpolation target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_gather_lerp", [=]() { reference(); do_not_optimize(buffers->reference.data()); });
    body.check = [=]() { body.target(); reference(); return check_bf16(*buffers); };
    return body;
  };
  return decl;
}

CaseDecl make_avg_pool(long long height, long long width, long long channels,
                       long long kernel) {
  const long long oh = height / kernel, ow = width / kernel;
  CaseDecl decl;
  decl.kernel = "basert_vision";
  decl.variant = "avg_pool_H" + std::to_string(height) + "_W" +
                 std::to_string(width) + "_K" + std::to_string(kernel) +
                 "_D" + std::to_string(channels);
  decl.shape = {{"H", height}, {"W", width}, {"K", kernel}, {"D", channels}};
  decl.dtype = "bf16";
  decl.format = "nhwc_avg";
  decl.bytes_moved = 2.0 * (height * width + oh * ow) * channels;
  decl.notes = "channel-contiguous NHWC spatial average pool";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(height * width * channels);
    buffers->out.resize(oh * ow * channels);
    buffers->reference.resize(buffers->out.size());
    fill_wave(buffers->x, 0.2f, 0.00029f);
    reserve(*buffers, buffers->x.size() + buffers->out.size());
    auto reference = [=]() {
      for (long long y = 0; y < oh; ++y) for (long long x = 0; x < ow; ++x) for (long long c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (long long ky = 0; ky < kernel; ++ky) for (long long kx = 0; kx < kernel; ++kx) {
          sum += bf16(buffers->x, ((y * kernel + ky) * width + x * kernel + kx) * channels + c);
        }
        set_bf16(buffers->reference, (y * ow + x) * channels + c, sum / float(kernel * kernel));
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::avg_pool2d_tokens_storage(
              in(buffers->x), out(buffers->out), 1, height, width, channels,
              kernel, kernel, kernel, kernel, false,
              &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("average pool target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_nhwc_average", [=]() { reference(); do_not_optimize(buffers->reference.data()); });
    body.check = [=]() { body.target(); reference(); return check_bf16(*buffers); };
    return body;
  };
  return decl;
}

CaseDecl make_audio_conv(long long length, long long input_channels,
                         long long output_channels, long long stride) {
  constexpr long long kernel = 3;
  const long long output_length = (length + 2 - kernel) / stride + 1;
  CaseDecl decl;
  decl.kernel = "basert_audio";
  decl.variant = "conv_B1_T" + std::to_string(length) + "_C" +
                 std::to_string(input_channels) + "_O" +
                 std::to_string(output_channels) + "_K3_S" +
                 std::to_string(stride);
  decl.shape = {{"B", 1}, {"T", length}, {"C", input_channels},
                {"O", output_channels}, {"K", kernel}, {"S", stride}};
  decl.dtype = "bf16";
  decl.format = "nwc_conv";
  decl.flops = 2.0 * output_length * output_channels * kernel * input_channels;
  decl.notes = "Whisper NWC direct convolution with bias";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(length * input_channels);
    buffers->w.resize(output_channels * kernel * input_channels);
    buffers->b.resize(output_channels);
    buffers->out.resize(output_length * output_channels);
    buffers->reference.resize(buffers->out.size());
    fill_wave(buffers->x, 0.15f, 0.0017f);
    fill_wave(buffers->w, 0.08f, 0.00041f, true);
    fill_wave(buffers->b, 0.03f, 0.07f);
    reserve(*buffers, buffers->x.size() + buffers->w.size() + buffers->b.size() + buffers->out.size());
    auto reference = [=]() {
      for (long long t = 0; t < output_length; ++t) for (long long o = 0; o < output_channels; ++o) {
        float sum = bf16(buffers->b, o);
        for (long long k = 0; k < kernel; ++k) {
          const long long source_t = t * stride + k - 1;
          if (source_t < 0 || source_t >= length) continue;
          for (long long c = 0; c < input_channels; ++c) {
            sum += bf16(buffers->x, source_t * input_channels + c) *
                   bf16(buffers->w, (o * kernel + k) * input_channels + c);
          }
        }
        set_bf16(buffers->reference, t * output_channels + o, sum);
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::audio_conv1d_direct_storage(
              in(buffers->x), in(buffers->w), in(buffers->b), out(buffers->out),
              1, length, input_channels, output_channels, kernel, stride, 1, 1,
              &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("audio convolution target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_nwc_convolution", [=]() { reference(); do_not_optimize(buffers->reference.data()); });
    body.check = [=]() { body.target(); reference(); return check_bf16(*buffers); };
    return body;
  };
  return decl;
}

CaseDecl make_audio_depthwise(long long length, long long channels,
                              long long kernel) {
  CaseDecl decl;
  decl.kernel = "basert_audio";
  decl.variant = "depthwise_B1_T" + std::to_string(length) + "_C" +
                 std::to_string(channels) + "_K" + std::to_string(kernel);
  decl.shape = {{"B", 1}, {"T", length}, {"C", channels}, {"K", kernel}};
  decl.dtype = "bf16";
  decl.format = "lightconv_silu";
  decl.flops = 2.0 * length * channels * kernel;
  decl.notes = "Conformer depthwise LightConv with fused SiLU";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(length * channels);
    buffers->w.resize(channels * kernel);
    buffers->b.resize(channels);
    buffers->out.resize(length * channels);
    buffers->reference.resize(buffers->out.size());
    fill_wave(buffers->x, 0.15f, 0.00081f);
    fill_wave(buffers->w, 0.08f, 0.0013f, true);
    fill_wave(buffers->b, 0.03f, 0.011f);
    reserve(*buffers, buffers->x.size() + buffers->w.size() + buffers->b.size() + buffers->out.size());
    auto reference = [=]() {
      for (long long t = 0; t < length; ++t) for (long long c = 0; c < channels; ++c) {
        float sum = bf16(buffers->b, c);
        for (long long k = 0; k < kernel; ++k) {
          const long long source_t = t + k - kernel / 2;
          if (source_t >= 0 && source_t < length) {
            sum += bf16(buffers->x, source_t * channels + c) * bf16(buffers->w, c * kernel + k);
          }
        }
        set_bf16(buffers->reference, t * channels + c, sum / (1.0f + std::exp(-sum)));
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::audio_depthwise_conv1d_storage(
              in(buffers->x), in(buffers->w), in(buffers->b), out(buffers->out),
              1, length, channels, kernel, 1, kernel / 2, 1, true,
              &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("audio depthwise target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_depthwise_silu", [=]() { reference(); do_not_optimize(buffers->reference.data()); });
    body.check = [=]() { body.target(); reference(); return check_bf16(*buffers); };
    return body;
  };
  return decl;
}

CaseDecl make_vision_projection(long long height, long long width,
                                long long input_channels,
                                long long output_channels, long long patch) {
  const long long output_height = height / patch;
  const long long output_width = width / patch;
  const long long patch_dim = patch * patch * input_channels;
  const long long rows = output_height * output_width;
  CaseDecl decl;
  decl.kernel = "basert_vision";
  decl.variant = "patch_projection_H" + std::to_string(height) + "_W" +
                 std::to_string(width) + "_P" + std::to_string(patch) +
                 "_O" + std::to_string(output_channels);
  decl.shape = {{"B", 1}, {"H", height}, {"W", width},
                {"C", input_channels}, {"P", patch},
                {"O", output_channels}};
  decl.dtype = "bf16";
  decl.format = "nhwc_patch_projection";
  decl.flops = 2.0 * rows * output_channels * patch_dim;
  decl.notes = "patch extraction plus FP32 projection and bias";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(height * width * input_channels);
    buffers->w.resize(output_channels * patch_dim);
    buffers->b.resize(output_channels);
    buffers->out.resize(rows * output_channels);
    buffers->reference.resize(buffers->out.size());
    fill_wave(buffers->x, 0.2f, 0.0011f);
    fill_wave(buffers->w, 0.08f, 0.00037f, true);
    fill_wave(buffers->b, 0.03f, 0.017f);
    reserve(*buffers, buffers->x.size() + buffers->w.size() +
                          buffers->b.size() + buffers->out.size());
    auto reference = [=]() {
      for (long long oy = 0; oy < output_height; ++oy) {
        for (long long ox = 0; ox < output_width; ++ox) {
          const long long row = oy * output_width + ox;
          for (long long output_channel = 0;
               output_channel < output_channels; ++output_channel) {
            float sum = bf16(buffers->b, output_channel);
            long long feature = 0;
            for (long long ky = 0; ky < patch; ++ky) {
              for (long long kx = 0; kx < patch; ++kx) {
                const long long source =
                    ((oy * patch + ky) * width + ox * patch + kx) *
                    input_channels;
                for (long long channel = 0; channel < input_channels;
                     ++channel, ++feature) {
                  sum += bf16(buffers->x, source + channel) *
                         bf16(buffers->w,
                              output_channel * patch_dim + feature);
                }
              }
            }
            set_bf16(buffers->reference,
                     row * output_channels + output_channel, sum);
          }
        }
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::vision_patch_projection_storage(
              in(buffers->x), in(buffers->w), in(buffers->b),
              out(buffers->out), 1, height, width, input_channels,
              output_channels, patch, patch, patch, patch, 0, 0,
              &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("vision patch projection target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_direct_patch_projection", [=]() {
      reference();
      do_not_optimize(buffers->reference.data());
    });
    body.check = [=]() {
      body.target();
      reference();
      return check_bf16(*buffers);
    };
    return body;
  };
  return decl;
}

CaseDecl make_vision_projection3d(
    long long frames, long long height, long long width,
    long long input_channels, long long output_channels, long long temporal,
    long long patch) {
  const long long output_frames = frames / temporal;
  const long long output_height = height / patch;
  const long long output_width = width / patch;
  const long long rows = output_frames * output_height * output_width;
  const long long patch_dim = temporal * patch * patch * input_channels;
  CaseDecl decl;
  decl.kernel = "basert_vision";
  decl.variant = "patch_projection3d_T" + std::to_string(frames) + "_H" +
                 std::to_string(height) + "_W" + std::to_string(width) +
                 "_PT" + std::to_string(temporal) + "_P" +
                 std::to_string(patch) + "_O" +
                 std::to_string(output_channels);
  decl.shape = {{"B", 1}, {"T", frames}, {"H", height}, {"W", width},
                {"C", input_channels}, {"O", output_channels},
                {"PT", temporal}, {"P", patch}};
  decl.dtype = "bf16";
  decl.format = "nthwc_patch_projection";
  decl.flops = 2.0 * rows * output_channels * patch_dim;
  decl.notes = "fused temporal/spatial patch projection with FP32 sums";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(frames * height * width * input_channels);
    buffers->w.resize(output_channels * patch_dim);
    buffers->b.resize(output_channels);
    buffers->out.resize(rows * output_channels);
    buffers->reference.resize(rows * output_channels);
    fill_wave(buffers->x, 0.2f, 0.00071f);
    fill_wave(buffers->w, 0.08f, 0.00019f, true);
    fill_wave(buffers->b, 0.03f, 0.017f);
    reserve(*buffers, buffers->x.size() + buffers->w.size() +
                          buffers->b.size() + buffers->out.size());
    auto reference = [=]() {
      for (long long ot = 0; ot < output_frames; ++ot) {
        for (long long oy = 0; oy < output_height; ++oy) {
          for (long long ox = 0; ox < output_width; ++ox) {
            const long long row =
                (ot * output_height + oy) * output_width + ox;
            for (long long output_channel = 0;
                 output_channel < output_channels; ++output_channel) {
              float sum = bf16(buffers->b, output_channel);
              long long feature = 0;
              for (long long kt = 0; kt < temporal; ++kt) {
                for (long long ky = 0; ky < patch; ++ky) {
                  for (long long kx = 0; kx < patch; ++kx) {
                    const long long source =
                        ((((ot * temporal + kt) * height + oy * patch + ky) *
                              width +
                          ox * patch + kx) *
                         input_channels);
                    for (long long c = 0; c < input_channels;
                         ++c, ++feature) {
                      sum += bf16(buffers->x, source + c) *
                             bf16(buffers->w,
                                  output_channel * patch_dim + feature);
                    }
                  }
                }
              }
              set_bf16(buffers->reference,
                       row * output_channels + output_channel, sum);
            }
          }
        }
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::vision_patch_projection_3d_storage(
              in(buffers->x), in(buffers->w), in(buffers->b),
              out(buffers->out), 1, frames, height, width, input_channels,
              output_channels, temporal, patch, patch, temporal, patch, patch,
              0, 0, 0, &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("3d patch projection target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_direct_patch_projection3d", [=]() {
      reference();
      do_not_optimize(buffers->reference.data());
    });
    body.check = [=]() {
      body.target();
      reference();
      return check_bf16(*buffers);
    };
    return body;
  };
  return decl;
}

CaseDecl make_cross_attention(long long query_heads, long long kv_heads,
                              long long query_length, long long key_length,
                              long long head_dim) {
  const long long q_count = query_heads * query_length * head_dim;
  const long long kv_count = kv_heads * key_length * head_dim;
  CaseDecl decl;
  decl.kernel = "basert_audio";
  decl.variant = "cross_B1_H" + std::to_string(query_heads) + "_" +
                 std::to_string(kv_heads) + "_T" +
                 std::to_string(query_length) + "_" +
                 std::to_string(key_length) + "_D" +
                 std::to_string(head_dim);
  decl.shape = {{"B", 1}, {"Hq", query_heads}, {"Hkv", kv_heads},
                {"Tq", query_length}, {"Tk", key_length},
                {"D", head_dim}};
  decl.dtype = "bf16";
  decl.format = "cross_attention_softcap";
  decl.flops =
      4.0 * query_heads * query_length * key_length * head_dim;
  decl.notes = "independent-length GQA cross-attention with score softcap";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(q_count);
    buffers->w.resize(kv_count);
    buffers->b.resize(kv_count);
    buffers->ids0 = {static_cast<int>(key_length)};
    buffers->out.resize(q_count);
    buffers->reference.resize(q_count);
    fill_wave(buffers->x, 0.15f, 0.0017f);
    fill_wave(buffers->w, 0.15f, 0.00071f, true);
    fill_wave(buffers->b, 0.20f, 0.00091f);
    reserve(*buffers, 2 * q_count + 2 * kv_count);
    auto reference = [=]() {
      const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
      const long long group = query_heads / kv_heads;
      std::vector<double> scores(static_cast<std::size_t>(key_length));
      for (long long query_head = 0; query_head < query_heads; ++query_head) {
        const long long kv_head = query_head / group;
        for (long long query_position = 0; query_position < query_length;
             ++query_position) {
          const long long query_row =
              query_head * query_length + query_position;
          double maximum = -std::numeric_limits<double>::infinity();
          for (long long key_position = 0; key_position < key_length;
               ++key_position) {
            const long long kv_row = kv_head * key_length + key_position;
            double score = 0.0;
            for (long long feature = 0; feature < head_dim; ++feature) {
              score += bf16(buffers->x, query_row * head_dim + feature) *
                       bf16(buffers->w, kv_row * head_dim + feature);
            }
            score = 50.0 * std::tanh(score * scale / 50.0);
            scores[key_position] = score;
            maximum = std::max(maximum, score);
          }
          double denominator = 0.0;
          for (long long key_position = 0; key_position < key_length;
               ++key_position) {
            scores[key_position] =
                std::exp(scores[key_position] - maximum);
            denominator += scores[key_position];
          }
          for (long long feature = 0; feature < head_dim; ++feature) {
            double sum = 0.0;
            for (long long key_position = 0; key_position < key_length;
                 ++key_position) {
              const long long kv_row = kv_head * key_length + key_position;
              sum += scores[key_position] *
                     bf16(buffers->b, kv_row * head_dim + feature);
            }
            set_bf16(buffers->reference,
                     query_row * head_dim + feature,
                     static_cast<float>(sum / denominator));
          }
        }
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::cross_attention_storage(
              in(buffers->x), in(buffers->w), in(buffers->b),
              buffers->ids0.data(), nullptr, out(buffers->out), 1,
              query_heads, kv_heads, query_length, key_length, head_dim, 0.0f,
              50.0f, &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("cross attention target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_scores_softmax_value", [=]() {
      reference();
      do_not_optimize(buffers->reference.data());
    });
    body.check = [=]() {
      body.target();
      reference();
      return check_bf16(*buffers);
    };
    return body;
  };
  return decl;
}

CaseDecl make_factorized_position(long long tokens, long long positions,
                                  long long channels) {
  CaseDecl decl;
  decl.kernel = "basert_vision";
  decl.variant = "factor_pos_B1_N" + std::to_string(tokens) + "_P" +
                 std::to_string(positions) + "_D" +
                 std::to_string(channels);
  decl.shape = {{"B", 1}, {"N", tokens}, {"P", positions}, {"D", channels}};
  decl.dtype = "bf16";
  decl.format = "factorized_xy";
  decl.bytes_moved = 2.0 * (2 * positions + tokens) * channels;
  decl.notes = "bounds-checked sum of independent learned x/y positions";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(2 * positions * channels);
    buffers->ids0.resize(2 * tokens);
    buffers->ids1.resize(tokens, 1);
    buffers->out.resize(tokens * channels);
    buffers->reference.resize(tokens * channels);
    fill_wave(buffers->x, 0.10f, 0.00071f);
    for (long long token = 0; token < tokens; ++token) {
      buffers->ids0[token * 2] = static_cast<int>(token % positions);
      buffers->ids0[token * 2 + 1] =
          static_cast<int>((token / positions) % positions);
    }
    reserve(*buffers, buffers->x.size() + 2 * buffers->out.size());
    auto reference = [=]() {
      for (long long token = 0; token < tokens; ++token) {
        const long long px = buffers->ids0[token * 2];
        const long long py = buffers->ids0[token * 2 + 1];
        for (long long d = 0; d < channels; ++d) {
          set_bf16(buffers->reference, token * channels + d,
                   bf16(buffers->x, px * channels + d) +
                       bf16(buffers->x, (positions + py) * channels + d));
        }
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::factorized_position_2d_storage(
              buffers->ids0.data(), in(buffers->x), buffers->ids1.data(),
              out(buffers->out), 1, tokens, positions, channels,
              &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("factorized position target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_two_gathers", [=]() {
      reference();
      do_not_optimize(buffers->reference.data());
    });
    body.check = [=]() {
      body.target();
      reference();
      return check_bf16(*buffers);
    };
    return body;
  };
  return decl;
}

CaseDecl make_position_pool(long long tokens, long long channels,
                            long long kernel, long long source_width) {
  const long long source_height = tokens / source_width;
  const long long output_length =
      (source_width / kernel) * (source_height / kernel);
  CaseDecl decl;
  decl.kernel = "basert_vision";
  decl.variant = "position_pool_B1_N" + std::to_string(tokens) + "_L" +
                 std::to_string(output_length) + "_K" +
                 std::to_string(kernel) + "_D" + std::to_string(channels);
  decl.shape = {{"B", 1}, {"N", tokens}, {"L", output_length},
                {"K", kernel}, {"D", channels}};
  decl.dtype = "bf16";
  decl.format = "coordinate_pool_fp32";
  decl.bytes_moved = 4.0 * (tokens + output_length) * channels;
  decl.notes = "arbitrarily ordered coordinate scatter to FP32 pooled tokens";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(tokens * channels);
    buffers->ids0.resize(2 * tokens);
    buffers->ids1.resize(tokens, 1);
    buffers->out_f32.resize(output_length * channels);
    buffers->reference_f32.resize(output_length * channels);
    buffers->mask_out.resize(output_length);
    buffers->mask_reference.resize(output_length);
    fill_wave(buffers->x, 0.10f, 0.00029f);
    for (long long token = 0; token < tokens; ++token) {
      const long long spatial = (token * 37) % tokens;
      buffers->ids0[token * 2] = static_cast<int>(spatial % source_width);
      buffers->ids0[token * 2 + 1] =
          static_cast<int>(spatial / source_width);
    }
    reserve(*buffers, buffers->x.size() + buffers->out_f32.size());
    auto reference = [=]() {
      std::fill(buffers->reference_f32.begin(), buffers->reference_f32.end(),
                0.0f);
      std::fill(buffers->mask_reference.begin(),
                buffers->mask_reference.end(), 0);
      const long long pooled_width = source_width / kernel;
      const float scale = std::sqrt(static_cast<float>(channels)) /
                          static_cast<float>(kernel * kernel);
      for (long long token = 0; token < tokens; ++token) {
        const long long bucket = buffers->ids0[token * 2] / kernel +
                                 pooled_width *
                                     (buffers->ids0[token * 2 + 1] / kernel);
        buffers->mask_reference[bucket] = 1;
        for (long long d = 0; d < channels; ++d) {
          buffers->reference_f32[bucket * channels + d] +=
              bf16(buffers->x, token * channels + d) * scale;
        }
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::pool_tokens_by_position_storage(
              in(buffers->x), buffers->ids0.data(), buffers->ids1.data(),
              buffers->out_f32.data(), buffers->mask_out.data(), 1, tokens,
              channels, output_length, kernel, source_width,
              &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("position pool target failed");
      }
      do_not_optimize(buffers->out_f32.data());
    };
    body.baselines.emplace_back("scalar_coordinate_scatter", [=]() {
      reference();
      do_not_optimize(buffers->reference_f32.data());
    });
    body.check = [=]() {
      body.target();
      reference();
      CheckResult check;
      for (std::size_t i = 0; i < buffers->out_f32.size(); ++i) {
        check_value(check, buffers->out_f32[i], buffers->reference_f32[i],
                    kFp32Tolerance);
      }
      if (buffers->mask_out != buffers->mask_reference) check.passed = false;
      return check;
    };
    return body;
  };
  return decl;
}

CaseDecl make_vision_rope(long long heads, long long tokens,
                          long long head_dim, bool qwen = false) {
  const long long side = static_cast<long long>(std::ceil(std::sqrt(tokens)));
  const long long positions = std::max(8LL, side);
  const long long pairs = head_dim / 4;
  CaseDecl decl;
  decl.kernel = "basert_vision";
  decl.variant = std::string(qwen ? "qwen_rope2d_B1_H" : "rope2d_B1_H") +
                 std::to_string(heads) + "_N" +
                 std::to_string(tokens) + "_D" + std::to_string(head_dim);
  decl.shape = {{"B", 1}, {"H", heads}, {"N", tokens}, {"D", head_dim}};
  decl.dtype = "bf16";
  decl.format = qwen ? "xy_global_split_rope" : "xy_rope";
  decl.bytes_moved = 4.0 * heads * tokens * head_dim;
  decl.notes = qwen ? "Qwen global split-half x/y vision rotation"
                    : "two independent split-half axis rotations";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(heads * tokens * head_dim);
    buffers->w.resize(positions * pairs);
    buffers->b.resize(positions * pairs);
    buffers->ids0.resize(2 * tokens);
    buffers->out.resize(buffers->x.size());
    buffers->reference.resize(buffers->x.size());
    fill_wave(buffers->x, 0.10f, 0.00031f);
    for (long long i = 0; i < positions * pairs; ++i) {
      set_bf16(buffers->w, i, std::cos(0.0017f * static_cast<float>(i + 1)));
      set_bf16(buffers->b, i, std::sin(0.0017f * static_cast<float>(i + 1)));
    }
    for (long long token = 0; token < tokens; ++token) {
      buffers->ids0[token * 2] = static_cast<int>(token % side);
      buffers->ids0[token * 2 + 1] = static_cast<int>(token / side);
    }
    reserve(*buffers, buffers->x.size() + buffers->w.size() +
                          buffers->b.size() + buffers->out.size());
    auto reference = [=]() {
      for (long long h = 0; h < heads; ++h) {
        for (long long token = 0; token < tokens; ++token) {
          const long long row = h * tokens + token;
          const long long px = buffers->ids0[token * 2];
          const long long py = buffers->ids0[token * 2 + 1];
          for (long long pair = 0; pair < pairs; ++pair) {
            const float x0 = bf16(buffers->x, row * head_dim + pair);
            const float middle0 =
                bf16(buffers->x, row * head_dim + pairs + pair);
            const float middle1 =
                bf16(buffers->x, row * head_dim + 2 * pairs + pair);
            const float y1 = bf16(buffers->x, row * head_dim + 3 * pairs + pair);
            const float cx = bf16(buffers->w, px * pairs + pair);
            const float sx = bf16(buffers->b, px * pairs + pair);
            const float cy = bf16(buffers->w, py * pairs + pair);
            const float sy = bf16(buffers->b, py * pairs + pair);
            if (qwen) {
              set_bf16(buffers->reference, row * head_dim + pair,
                       x0 * cx - middle1 * sx);
              set_bf16(buffers->reference, row * head_dim + pairs + pair,
                       middle0 * cy - y1 * sy);
              set_bf16(buffers->reference, row * head_dim + 2 * pairs + pair,
                       x0 * sx + middle1 * cx);
              set_bf16(buffers->reference, row * head_dim + 3 * pairs + pair,
                       middle0 * sy + y1 * cy);
            } else {
              set_bf16(buffers->reference, row * head_dim + pair,
                       x0 * cx - middle0 * sx);
              set_bf16(buffers->reference, row * head_dim + pairs + pair,
                       x0 * sx + middle0 * cx);
              set_bf16(buffers->reference, row * head_dim + 2 * pairs + pair,
                       middle1 * cy - y1 * sy);
              set_bf16(buffers->reference, row * head_dim + 3 * pairs + pair,
                       middle1 * sy + y1 * cy);
            }
          }
        }
      }
    };
    CaseBody body;
    body.target = [=]() {
      const Status status =
          qwen ? quixicore_cpu::qwen_vision_rope_2d_storage(
                     in(buffers->x), in(buffers->w), in(buffers->b),
                     buffers->ids0.data(), out(buffers->out), 1, heads,
                     tokens, head_dim, positions, &buffers->workspace)
               : quixicore_cpu::vision_rope_2d_storage(
                     in(buffers->x), in(buffers->w), in(buffers->b),
                     buffers->ids0.data(), out(buffers->out), 1, heads,
                     tokens, head_dim, positions, &buffers->workspace);
      if (status != Status::kOk) {
        throw std::runtime_error("vision rope target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_axis_rotate", [=]() {
      reference();
      do_not_optimize(buffers->reference.data());
    });
    body.check = [=]() {
      body.target();
      reference();
      return check_bf16(*buffers);
    };
    return body;
  };
  return decl;
}

CaseDecl make_audio_causal_depthwise(long long length, long long channels,
                                     long long kernel) {
  CaseDecl decl;
  decl.kernel = "basert_audio";
  decl.variant = "causal_depthwise_B1_T" + std::to_string(length) + "_C" +
                 std::to_string(channels) + "_K" + std::to_string(kernel);
  decl.shape = {{"B", 1}, {"T", length}, {"C", channels}, {"K", kernel}};
  decl.dtype = "bf16";
  decl.format = "lightconv_causal";
  decl.flops = 2.0 * length * channels * kernel;
  decl.notes = "left-padded causal NWC depthwise convolution";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(length * channels);
    buffers->w.resize(channels * kernel);
    buffers->b.resize(channels);
    buffers->out.resize(length * channels);
    buffers->reference.resize(length * channels);
    fill_wave(buffers->x, 0.15f, 0.00081f);
    fill_wave(buffers->w, 0.08f, 0.0013f, true);
    fill_wave(buffers->b, 0.03f, 0.011f);
    reserve(*buffers, buffers->x.size() + buffers->w.size() +
                          buffers->b.size() + buffers->out.size());
    auto reference = [=]() {
      for (long long t = 0; t < length; ++t) {
        for (long long c = 0; c < channels; ++c) {
          float sum = bf16(buffers->b, c);
          for (long long k = 0; k < kernel; ++k) {
            const long long source_t = t + k - (kernel - 1);
            if (source_t >= 0) {
              sum += bf16(buffers->x, source_t * channels + c) *
                     bf16(buffers->w, c * kernel + k);
            }
          }
          set_bf16(buffers->reference, t * channels + c, sum);
        }
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::audio_causal_depthwise_conv1d_storage(
              in(buffers->x), in(buffers->w), in(buffers->b),
              out(buffers->out), 1, length, channels, kernel, 1,
              &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("causal depthwise target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_left_pad_depthwise", [=]() {
      reference();
      do_not_optimize(buffers->reference.data());
    });
    body.check = [=]() {
      body.target();
      reference();
      return check_bf16(*buffers);
    };
    return body;
  };
  return decl;
}

CaseDecl make_audio_relative(long long length, long long heads,
                             long long head_dim) {
  constexpr long long chunk = 12;
  constexpr long long left = 13;
  constexpr long long right = 0;
  constexpr long long relative_positions = 13;
  const long long count = length * heads * head_dim;
  const long long context_length = chunk + left - 1 + right;
  CaseDecl decl;
  decl.kernel = "basert_audio";
  decl.variant = "relative_B1_T" + std::to_string(length) + "_H" +
                 std::to_string(heads) + "_D" + std::to_string(head_dim);
  decl.shape = {{"B", 1}, {"T", length}, {"H", heads}, {"D", head_dim},
                {"chunk", chunk}, {"left", left}, {"right", right}};
  decl.dtype = "bf16";
  decl.format = "blocked_relative_softcap";
  decl.flops = 4.0 * length * heads * context_length * head_dim;
  decl.notes = "bounded-context relative attention with learned query scale";
  decl.make = [=]() {
    auto buffers = std::make_shared<Buffers>();
    buffers->x.resize(count);
    buffers->w.resize(count);
    buffers->b.resize(count);
    buffers->extra.resize(relative_positions * heads * head_dim);
    buffers->running.resize(head_dim);
    buffers->ids0 = {static_cast<int>(length)};
    buffers->out.resize(count);
    buffers->reference.resize(count);
    fill_wave(buffers->x, 0.08f, 0.00037f);
    fill_wave(buffers->w, 0.08f, 0.00041f, true);
    fill_wave(buffers->b, 0.20f, 0.00029f);
    fill_wave(buffers->extra, 0.08f, 0.00073f, true);
    for (long long d = 0; d < head_dim; ++d) {
      buffers->running[d] = 0.2f * std::sin(0.017f * static_cast<float>(d + 1));
    }
    reserve(*buffers, buffers->x.size() + buffers->w.size() +
                          buffers->b.size() + buffers->extra.size() +
                          buffers->out.size());
    auto reference = [=]() {
      const float log_two = std::log(2.0f);
      const float q_scale = 1.0f / (std::sqrt(static_cast<float>(head_dim)) * log_two);
      const float k_scale = 1.0f / log_two;
      std::vector<float> scaled(head_dim);
      std::vector<float> scores(context_length);
      std::vector<long long> rows(context_length);
      for (long long t = 0; t < length; ++t) {
        const long long qi = t % chunk;
        const long long start = (t / chunk) * chunk - (left - 1);
        for (long long h = 0; h < heads; ++h) {
          const long long output_row = t * heads + h;
          for (long long d = 0; d < head_dim; ++d) {
            const float raw = buffers->running[d];
            const float learned = std::max(raw, 0.0f) +
                                  std::log1p(std::exp(-std::fabs(raw)));
            scaled[d] = bf16(buffers->x, output_row * head_dim + d) *
                        q_scale * learned;
          }
          long long used = 0;
          float maximum = -std::numeric_limits<float>::infinity();
          for (long long ci = 0; ci < context_length; ++ci) {
            const long long key_t = start + ci;
            if (key_t < 0 || key_t >= length) continue;
            const long long kv_row = key_t * heads + h;
            const long long ri = ci - qi;
            float score = 0.0f;
            for (long long d = 0; d < head_dim; ++d) {
              score += scaled[d] * bf16(buffers->w, kv_row * head_dim + d) * k_scale;
              if (ri >= 0 && ri < relative_positions) {
                score += scaled[d] * bf16(
                    buffers->extra, (ri * heads + h) * head_dim + d);
              }
            }
            score = 50.0f * std::tanh(score / 50.0f);
            scores[used] = score;
            rows[used++] = kv_row;
            maximum = std::max(maximum, score);
          }
          float denominator = 0.0f;
          for (long long index = 0; index < used; ++index) {
            scores[index] = std::exp(scores[index] - maximum);
            denominator += scores[index];
          }
          for (long long d = 0; d < head_dim; ++d) {
            float sum = 0.0f;
            for (long long index = 0; index < used; ++index) {
              sum += scores[index] *
                     bf16(buffers->b, rows[index] * head_dim + d);
            }
            set_bf16(buffers->reference, output_row * head_dim + d,
                     sum / denominator);
          }
        }
      }
    };
    CaseBody body;
    body.target = [=]() {
      if (quixicore_cpu::audio_relative_attention_storage(
              in(buffers->x), in(buffers->w), in(buffers->b),
              in(buffers->extra), buffers->running.data(),
              buffers->ids0.data(), out(buffers->out), 1, length, heads,
              head_dim, relative_positions, chunk, left, right, 0.0f, 0.0f,
              50.0f, &buffers->workspace) != Status::kOk) {
        throw std::runtime_error("relative attention target failed");
      }
      do_not_optimize(buffers->out.data());
    };
    body.baselines.emplace_back("scalar_materialized_scores", [=]() {
      reference();
      do_not_optimize(buffers->reference.data());
    });
    body.check = [=]() {
      body.target();
      reference();
      return check_bf16(*buffers);
    };
    return body;
  };
  return decl;
}

}  // namespace

void build_basert_aux_cases(const BuildCtx& ctx, std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_calibration(128, 256));
    out.push_back(make_value_clip(64, 256));
    out.push_back(make_softcap(8, 1024));
    return;
  }
  out.push_back(make_calibration(512, 8192));
  out.push_back(make_calibration(8192, 4096));
  out.push_back(make_value_clip(1120, 768));
  out.push_back(make_value_clip(750, 1024));
  out.push_back(make_softcap(64, 128256));
  out.push_back(make_softcap(1024, 32000));
}

void build_basert_embedding_cases(const BuildCtx& ctx,
                                  std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_embedding(32, 256));
    out.push_back(make_masked_pool(2, 32, 256));
    return;
  }
  out.push_back(make_embedding(128, 768));
  out.push_back(make_embedding(512, 768));
  out.push_back(make_masked_pool(4, 128, 768));
  out.push_back(make_masked_pool(8, 512, 768));
}

void build_basert_vision_cases(const BuildCtx& ctx,
                               std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_patch(64, 64, 3, 8));
    out.push_back(make_patch3d(2, 64, 64, 3, 2, 8));
    out.push_back(make_interpolate(16, 16, 24, 24, 256));
    out.push_back(make_avg_pool(16, 16, 256, 2));
    out.push_back(make_vision_projection(32, 32, 3, 64, 8));
    out.push_back(make_vision_projection3d(2, 32, 32, 3, 64, 2, 8));
    out.push_back(make_factorized_position(256, 32, 256));
    out.push_back(make_position_pool(256, 256, 2, 16));
    out.push_back(make_vision_rope(4, 256, 64));
    out.push_back(make_vision_rope(4, 256, 64, true));
    return;
  }
  out.push_back(make_patch(224, 224, 3, 16));
  out.push_back(make_patch(896, 896, 3, 16));
  out.push_back(make_patch3d(2, 224, 224, 3, 2, 14));
  out.push_back(make_patch3d(8, 224, 224, 3, 2, 14));
  out.push_back(make_interpolate(16, 16, 56, 56, 768));
  out.push_back(make_interpolate(64, 64, 56, 56, 768));
  out.push_back(make_avg_pool(56, 56, 768, 4));
  out.push_back(make_avg_pool(64, 64, 1024, 4));
  out.push_back(make_vision_projection(224, 224, 3, 256, 16));
  out.push_back(make_vision_projection3d(2, 56, 56, 3, 128, 2, 14));
  out.push_back(make_factorized_position(1120, 64, 768));
  out.push_back(make_position_pool(1120, 768, 2, 40));
  out.push_back(make_vision_rope(8, 1120, 128));
  out.push_back(make_vision_rope(8, 1120, 128, true));
}

void build_basert_audio_cases(const BuildCtx& ctx,
                              std::vector<CaseDecl>& out) {
  if (ctx.preset == Preset::kSmoke) {
    out.push_back(make_audio_conv(128, 32, 64, 1));
    out.push_back(make_audio_depthwise(128, 256, 5));
    out.push_back(make_audio_causal_depthwise(128, 256, 5));
    out.push_back(make_cross_attention(4, 2, 2, 64, 64));
    out.push_back(make_audio_relative(64, 4, 64));
    return;
  }
  out.push_back(make_audio_conv(1500, 80, 384, 1));
  out.push_back(make_audio_conv(750, 384, 384, 2));
  out.push_back(make_audio_depthwise(750, 1024, 5));
  out.push_back(make_audio_causal_depthwise(750, 1024, 5));
  out.push_back(make_cross_attention(16, 16, 1, 1500, 64));
  out.push_back(make_cross_attention(8, 8, 12, 25, 128));
  out.push_back(make_audio_relative(750, 8, 128));
}

}  // namespace qcb
