#include "harness/env_info.h"

#include <cstdio>
#include <ctime>
#include <string>
#include <thread>

#include "quixicore_cpu/backend.h"
#include "quixicore_cpu/cpu_features.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/utsname.h>
#include <unistd.h>

#include <fstream>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace qcb {
namespace {

std::string run_cmd_first_line(const std::string& cmd) {
#if defined(_WIN32)
  FILE* pipe = _popen(cmd.c_str(), "r");
#else
  FILE* pipe = popen(cmd.c_str(), "r");
#endif
  if (pipe == nullptr) {
    return "";
  }
  char buf[512];
  std::string out;
  if (std::fgets(buf, sizeof buf, pipe) != nullptr) {
    out = buf;
  }
  // Drain so the child is not killed mid-write.
  while (std::fgets(buf, sizeof buf, pipe) != nullptr) {
  }
#if defined(_WIN32)
  _pclose(pipe);
#else
  pclose(pipe);
#endif
  while (!out.empty() &&
         (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
    out.pop_back();
  }
  return out;
}

std::string git_label() {
  // Runtime query (Metal harness parity) so the label reflects the tree as
  // run, not as configured.
  const std::string git =
      std::string("git -C \"") + QUIXICORE_CPU_BENCH_REPO_DIR + "\" ";
#if defined(_WIN32)
  const char* to_null = " 2>nul";
#else
  const char* to_null = " 2>/dev/null";
#endif
  const std::string sha =
      run_cmd_first_line(git + "rev-parse --short HEAD" + to_null);
  if (sha.empty()) {
    return "unknown";
  }
  const std::string dirty =
      run_cmd_first_line(git + "status --porcelain" + to_null);
  return dirty.empty() ? sha : sha + "-dirty";
}

#if defined(__APPLE__)
std::string sysctl_string(const char* name) {
  char buf[256];
  size_t size = sizeof buf;
  if (sysctlbyname(name, buf, &size, nullptr, 0) != 0) {
    return "";
  }
  return std::string(buf);
}

long long sysctl_int64(const char* name) {
  long long value = 0;
  size_t size = sizeof value;
  if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) {
    return -1;
  }
  return value;
}
#endif

std::string detect_os() {
#if defined(__APPLE__)
  const std::string version = sysctl_string("kern.osproductversion");
  return version.empty() ? "macOS" : "macOS " + version;
#elif defined(__linux__)
  utsname u{};
  if (uname(&u) == 0) {
    return std::string(u.sysname) + " " + u.release;
  }
  return "Linux";
#elif defined(_WIN32)
  return "Windows";
#else
  return "unknown";
#endif
}

std::string detect_cpu_model() {
#if defined(__APPLE__)
  const std::string brand = sysctl_string("machdep.cpu.brand_string");
  return brand.empty() ? "unknown" : brand;
#elif defined(__linux__)
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    // x86 exposes "model name"; many aarch64 SoCs expose only "Hardware".
    if (line.rfind("model name", 0) == 0 || line.rfind("Hardware", 0) == 0) {
      const auto colon = line.find(':');
      if (colon != std::string::npos) {
        auto value = line.substr(colon + 1);
        const auto start = value.find_first_not_of(" \t");
        return start == std::string::npos ? "unknown" : value.substr(start);
      }
    }
  }
  return "unknown";
#elif defined(_WIN32)
  char buf[256];
  DWORD size = sizeof buf;
  if (RegGetValueA(HKEY_LOCAL_MACHINE,
                   "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                   "ProcessorNameString", RRF_RT_REG_SZ, nullptr, buf,
                   &size) == ERROR_SUCCESS) {
    return std::string(buf);
  }
  return "unknown";
#else
  return "unknown";
#endif
}

long long detect_memory_bytes() {
#if defined(__APPLE__)
  return sysctl_int64("hw.memsize");
#elif defined(__linux__)
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages <= 0 || page_size <= 0) {
    return -1;
  }
  return static_cast<long long>(pages) * page_size;
#elif defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof status;
  if (GlobalMemoryStatusEx(&status) != 0) {
    return static_cast<long long>(status.ullTotalPhys);
  }
  return -1;
#else
  return -1;
#endif
}

std::string detect_compiler() {
#if defined(__clang__)
  return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
  return std::string("GCC ") + __VERSION__;
#elif defined(_MSC_VER)
  return "MSVC " + std::to_string(_MSC_FULL_VER);
#else
  return "unknown";
#endif
}

}  // namespace

EnvInfo collect_env_info() {
  EnvInfo info;
  info.git_label = git_label();
  info.os = detect_os();
#if defined(__x86_64__) || defined(_M_X64)
  info.arch = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
  info.arch = "aarch64";
#else
  info.arch = "unknown";
#endif
  info.cpu_model = detect_cpu_model();
  info.logical_cores =
      static_cast<long long>(std::thread::hardware_concurrency());
#if defined(__APPLE__)
  info.physical_cores = sysctl_int64("hw.physicalcpu");
  info.perf_cores = sysctl_int64("hw.perflevel0.physicalcpu");
  info.eff_cores = sysctl_int64("hw.perflevel1.physicalcpu");
#endif
  info.memory_bytes = detect_memory_bytes();
  info.compiler = detect_compiler();
  info.build_type = QUIXICORE_CPU_BENCH_BUILD_TYPE;

  for (const auto& hint : quixicore_cpu::runtime_feature_hints()) {
    if (hint.available) {
      info.cpu_features.emplace_back(hint.name);
    }
  }

  const auto metadata = quixicore_cpu::backend_metadata();
  info.backend = std::string(metadata.backend);
  info.repo = std::string(metadata.repo);
  info.contract = std::string(metadata.contract);
  return info;
}

std::string format_now(const char* fmt) {
  const std::time_t now = std::time(nullptr);
  std::tm tm_buf{};
#if defined(_WIN32)
  localtime_s(&tm_buf, &now);
#else
  localtime_r(&now, &tm_buf);
#endif
  char buf[64];
  if (std::strftime(buf, sizeof buf, fmt, &tm_buf) == 0) {
    return "";
  }
  return std::string(buf);
}

}  // namespace qcb
