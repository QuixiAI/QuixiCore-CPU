#include <iostream>
#include <string_view>
#include <vector>

#include "quixicore_cpu/backend.h"
#include "quixicore_cpu/cpu_features.h"

namespace {

void print_list(std::string_view label,
                const std::vector<std::string_view>& values) {
  std::cout << label << ":\n";
  if (values.empty()) {
    std::cout << "  none\n";
    return;
  }

  for (const auto value : values) {
    std::cout << "  - " << value << '\n';
  }
}

}  // namespace

int main() {
  const auto metadata = quixicore_cpu::backend_metadata();

  std::cout << "backend: " << metadata.backend << '\n';
  std::cout << "name: " << metadata.name << '\n';
  std::cout << "repo: " << metadata.repo << '\n';
  std::cout << "umbrella: " << metadata.umbrella << '\n';
  std::cout << "contract: " << metadata.contract << '\n';
  std::cout << "status: " << metadata.status << '\n';

  print_list("targets", metadata.targets);
  print_list("integrations", metadata.integrations);
  print_list("supported_kernel_families",
             quixicore_cpu::supported_kernel_families());
  print_list("planned_kernel_families",
             quixicore_cpu::planned_kernel_families());

  std::cout << "compile_time_feature_hints:\n";
  for (const auto feature : quixicore_cpu::compile_time_feature_hints()) {
    std::cout << "  - " << feature.name << ": "
              << (feature.available ? "available" : "unavailable") << " ("
              << feature.source << ")\n";
  }

  std::cout << "runtime_features:\n";
  for (const auto feature : quixicore_cpu::runtime_feature_hints()) {
    std::cout << "  - " << feature.name << ": "
              << (feature.available ? "available" : "unavailable") << " ("
              << feature.source << ")\n";
  }

  return 0;
}

