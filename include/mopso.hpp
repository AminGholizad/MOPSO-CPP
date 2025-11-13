#ifndef PSO_H
#define PSO_H
#include "particle.hpp"
#include "repository.hpp"
#include <cmath>
#include <cstddef>
namespace mopso {

struct Weight_range {
  double begin{0};
  double end{0};
};
constexpr Weight_range DEFAULT_WEIGHT_RANGE{.begin = 0.1, .end = 0.01};

template <size_t SWARM_SIZE, size_t NUM_VARS, size_t OBJECTIVES,
          size_t GRID_SIZE = DEFAULT_GRID_SIZE>
struct Solution {
  Repository<NUM_VARS, OBJECTIVES, GRID_SIZE> repository{};
  Swarm<SWARM_SIZE, NUM_VARS, OBJECTIVES, GRID_SIZE> swarm{};
};

template <size_t SWARM_SIZE, size_t NUM_VARS, size_t OBJECTIVES,
          size_t GRID_SIZE = DEFAULT_GRID_SIZE>
Solution<SWARM_SIZE, NUM_VARS, OBJECTIVES, GRID_SIZE>
mopso(const Variables<NUM_VARS> &lower_bound,
      const Variables<NUM_VARS> &upper_bound,
      const Problem<OBJECTIVES> &problem, const size_t max_iter = 1000,
      const size_t repository_size = 100, const double alpha = 0.1,
      const double beta = 2.0, const double gamma = 2.0,
      const Coefficient &coefficients = DEFAULT_COEFFICIENTS,
      const Weight_range &weight_range = DEFAULT_WEIGHT_RANGE,
      const double mu = 0.1) {
  auto calc_weight = [&](size_t iter) {
    return ((static_cast<double>(max_iter - iter) -
             (weight_range.begin - weight_range.end)) /
            static_cast<double>(max_iter)) +
           weight_range.end;
  };
  auto calc_mutation_propablity = [&](size_t iter) {
    const double den = max_iter > 1 ? static_cast<double>(max_iter) - 1.0 : 1.0;
    return std::pow(1.0 - (static_cast<double>(iter) / den), 1.0 / mu);
  };
  auto swarm = Swarm<SWARM_SIZE, NUM_VARS, OBJECTIVES, GRID_SIZE>(
      lower_bound, upper_bound, problem);

  auto repository = Repository<NUM_VARS, OBJECTIVES, GRID_SIZE>(
      swarm.particles, repository_size, alpha, beta, gamma);

  for (size_t i = 0; i < max_iter; i++) {
    const auto gBest = repository.SelectLeader();
    const auto curretn_weight = calc_weight(i);
    const auto current_mutation_probablity = calc_mutation_propablity(i);

    swarm.update_particles(gBest, problem, curretn_weight, coefficients,
                           current_mutation_probablity);
    repository.update(swarm.particles);
  }
  return {repository, swarm};
}
} // namespace mopso
#endif // PSO_H
