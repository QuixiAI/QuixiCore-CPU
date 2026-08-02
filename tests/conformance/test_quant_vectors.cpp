// Check this backend's quant decoders against the umbrella test vectors.
//
// The vectors in QuixiCore/test-vectors/quant are derived from specs/formats/,
// not from any implementation, so this asks "does CPU agree with the spec"
// rather than "does CPU agree with itself". Across the four backends the same
// E8M0 byte currently decodes three different ways at the edges, which nothing
// could see because nothing compared them:
//
//   CPU    code 0 -> 2^-127, code 255 -> NaN     (conformant)
//   Metal  code 0 -> 2^-127, code 255 -> +Inf
//   CUDA   code 0 -> +0.0,   code 255 -> +Inf
//
// Comparison is on IEEE-754 bits. 2^-127 is subnormal, so a tolerance-based
// check accepts +0.0 for it and reports success.
//
// The vector directory is passed as argv[1] so the test does not assume a
// checkout layout; CMake supplies it.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

// The CPU backend's E8M0, mirrored from kernels/attention/attention_mxfp8.cpp.
// Duplicated deliberately: if that definition moves, this test should fail and
// be pointed at the new home rather than silently testing a stale copy.
float e8m0_decode(std::uint8_t code) {
  if (code == 255) return std::numeric_limits<float>::quiet_NaN();
  return std::ldexp(1.0f, static_cast<int>(code) - 127);
}

std::uint32_t as_bits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  return u;
}

std::string slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Pull every `"key": value` in document order. The vectors are flat enough
// that a JSON dependency would cost more than it is worth here.
void scan(const std::string& j, const char* key, std::vector<long>& nums,
          std::vector<std::string>& strs) {
  const std::string pat = std::string("\"") + key + "\":";
  std::size_t p = 0;
  while ((p = j.find(pat, p)) != std::string::npos) {
    std::size_t q = p + pat.size();
    while (q < j.size() && (j[q] == ' ' || j[q] == '\n')) ++q;
    if (j[q] == '"') {
      const std::size_t e = j.find('"', q + 1);
      strs.push_back(j.substr(q + 1, e - q - 1));
      nums.push_back(-1);
    } else if (j.compare(q, 4, "null") == 0) {
      strs.emplace_back("null");
      nums.push_back(-1);
    } else {
      strs.emplace_back();
      nums.push_back(std::strtol(j.c_str() + q, nullptr, 10));
    }
    p = q;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir =
      argc > 1 ? argv[1] : "../../../test-vectors/quant";

  const std::string j = slurp(dir + "/e8m0.json");
  std::vector<long> codes, ignored;
  std::vector<std::string> code_s, bits_s;
  scan(j, "code", codes, code_s);
  scan(j, "bits", ignored, bits_s);
  if (codes.empty() || codes.size() != bits_s.size()) {
    std::fprintf(stderr, "vector parse failed (%zu codes, %zu bits)\n",
                 codes.size(), bits_s.size());
    return 2;
  }

  int bad = 0;
  for (std::size_t i = 0; i < codes.size(); ++i) {
    const float got = e8m0_decode(static_cast<std::uint8_t>(codes[i]));
    const bool want_nan = bits_s[i] == "null";
    const bool ok =
        want_nan ? std::isnan(got)
                 : as_bits(got) == static_cast<std::uint32_t>(
                                       std::strtoul(bits_s[i].c_str(), nullptr, 16));
    if (!ok) {
      ++bad;
      std::printf("  code %-4ld expected %-12s got 0x%08x\n", codes[i],
                  want_nan ? "nan" : bits_s[i].c_str(), as_bits(got));
    }
  }

  std::printf("E8M0 vs %s/e8m0.json: %s (%zu/%zu codes)\n", dir.c_str(),
              bad ? "FAIL" : "conformant", codes.size() - bad, codes.size());
  return bad ? 1 : 0;
}
