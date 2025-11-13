#include "mopso.hpp"
#include <cmath>
#include <cstddef>
#include <fstream>
const double pi = std::numbers::pi;

auto cost_fcn(std::span<const double> variables) {
  mopso::Cost<2> result{};
  for (const auto &var : variables) {
    result.objective[0] +=
        std::sin(var * 5) +            // NOLINT(readability-magic-numbers,
                                       // cppcoreguidelines-avoid-magic-numbers)
        std::sin(var * 7) +            // NOLINT(readability-magic-numbers,
                                       // cppcoreguidelines-avoid-magic-numbers)
        std::sin(var * 11);            // NOLINT(readability-magic-numbers,
                                       // cppcoreguidelines-avoid-magic-numbers)
    for (size_t j = 1; j < 100; j++) { // NOLINT(readability-magic-numbers,
      // cppcoreguidelines-avoid-magic-numbers)
      result.objective[1] +=
          std::sin(var *
                   static_cast<double>(
                       (6 * j) + 1)) + // NOLINT(readability-magic-numbers,
                                       // cppcoreguidelines-avoid-magic-numbers)
          std::sin(var *
                   static_cast<double>(
                       (6 * j) - 1)); // NOLINT(readability-magic-numbers,
                                      // cppcoreguidelines-avoid-magic-numbers)
    }
    result.infeasiblity += std::sin(var);
  }
  result.objective[0] = std::abs(result.objective[0]);
  result.objective[1] = std::abs(result.objective[1]);
  result.infeasiblity =
      std::abs((result.infeasiblity / static_cast<double>(variables.size())) -
               0.7); // NOLINT(readability-magic-numbers,
                     // cppcoreguidelines-avoid-magic-numbers)
  return result;
}
int main() {
  const size_t swarm_size{200};
  const size_t repository_size{200};
  const size_t iteration_count{200};
  const mopso::Problem problem{cost_fcn};
  const mopso::Variables lower_bound{0.0, 0.0, 0.0, 0.0};
  const mopso::Variables upper_bound{pi / 2, pi / 2, pi / 2, pi / 2};
  const auto [repository, swarm] = mopso::mopso<swarm_size>(
      lower_bound, upper_bound, problem, iteration_count, repository_size);
  auto file = std::ofstream("./repository.csv");
  repository.export_csv(file);
  file = std::ofstream("./swarm.csv");
  swarm.export_csv(file);
  repository.SelectLeader().info();
  return 0;
}
