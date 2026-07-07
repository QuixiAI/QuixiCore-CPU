#pragma once

#include <string>
#include <vector>

namespace qcb {

struct EnvInfo {
  std::string git_label;  // "<short-sha>[-dirty]" or "unknown"
  std::string os;
  std::string arch;
  std::string cpu_model;
  long long logical_cores = -1;
  long long physical_cores = -1;  // -1 when the platform has no cheap query
  long long perf_cores = -1;      // Apple hybrid topology when available
  long long eff_cores = -1;
  long long memory_bytes = -1;
  std::string compiler;
  std::string build_type;
  std::vector<std::string> cpu_features;  // runtime-detected, available only
  std::string backend;
  std::string repo;
  std::string contract;
};

EnvInfo collect_env_info();

// strftime of local time, e.g. format_now("%Y-%m-%dT%H:%M:%S").
std::string format_now(const char* fmt);

}  // namespace qcb
