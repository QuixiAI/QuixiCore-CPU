#pragma once

#include <string_view>
#include <vector>

namespace quixicore_cpu {

struct BackendMetadata {
  std::string_view backend;
  std::string_view name;
  std::string_view repo;
  std::string_view umbrella;
  std::string_view contract;
  std::string_view status;
  std::vector<std::string_view> targets;
  std::vector<std::string_view> integrations;
};

struct FeatureHint {
  std::string_view name;
  bool available;
  std::string_view source;
};

BackendMetadata backend_metadata();

std::vector<std::string_view> supported_kernel_families();

std::vector<std::string_view> planned_kernel_families();

std::vector<FeatureHint> compile_time_feature_hints();

bool is_kernel_family_supported(std::string_view family);

}  // namespace quixicore_cpu

