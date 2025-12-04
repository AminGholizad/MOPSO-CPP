#ifndef PARTICLE_H
#define PARTICLE_H
#include <Random-Helper.hpp>
#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
namespace mopso {
struct Coefficient {
  double personal;
  double global;
};

constexpr Coefficient DEFAULT_COEFFICIENTS{.personal = 0.2, .global = 0.2};
constexpr double DEFAULT_WEIGHT = 0.5;
constexpr double DEFAULT_MUTATION_PROBABLITY = 0.1;
constexpr double THRESHOLD = 0.5;
constexpr size_t DEFAULT_GRID_SIZE = 7;

template <size_t NUM_VARS> using Variables = std::array<double, NUM_VARS>;

template <size_t OBJECTIVES> struct Cost {
  Variables<OBJECTIVES> objective{0};
  double infeasiblity{0};
};

template <size_t OBJECTIVES>
using Problem = std::function<Cost<OBJECTIVES>(std::span<const double>)>;

template <size_t NUM_VARS, size_t OBJECTIVES, size_t GRID_SIZE>
class Repository; // Forward declaration

template <size_t NUM_VARS, size_t OBJECTIVES,
          size_t GRID_SIZE = DEFAULT_GRID_SIZE>
class Particle {
  using variables = Variables<NUM_VARS>;
  using Cost = Cost<OBJECTIVES>;
  using Problem = Problem<OBJECTIVES>;
  using Grid = std::array<std::array<double, GRID_SIZE + 1>, OBJECTIVES>;

public:
  constexpr Particle() = default;
  constexpr Particle(variables lower, variables upper, const Problem &problem)
      : lower_bound{std::move(lower)}, upper_bound{std::move(upper)} {
    for (size_t i = 0; i < NUM_VARS; i++) {
      position[i] = rnd::unifrnd(lower_bound[i], upper_bound[i]);
      velocity[i] = 0.0;
    }
    cost = problem(position);
    pBest_position = position;
    pBest_cost = cost;
  }
  constexpr static void update_grid_index(const Grid &grid,
                                          std::span<Particle> swarm) {
    for (auto &particle : swarm) {
      particle.update_grid_index(grid);
    }
  }
  constexpr void
  update(const Particle &gBest, const Problem &problem,
         const double weight = DEFAULT_WEIGHT,
         const Coefficient &coeficients = DEFAULT_COEFFICIENTS,
         const double mutation_probablity = DEFAULT_MUTATION_PROBABLITY) {
    updateV(gBest, weight, coeficients);
    updateX();
    cost = problem(position);
    Mutate(problem, mutation_probablity);
    updatePBest();
  }
  [[nodiscard]] constexpr bool dominates(const Particle &other) const {
    if (cost.infeasiblity < other.cost.infeasiblity) {
      return true;
    }
    if (cost.infeasiblity > other.cost.infeasiblity) {
      return false;
    }
    bool flag = false;
    for (size_t i = 0; i < OBJECTIVES; i++) {
      if (cost.objective[i] > other.cost.objective[i]) {
        return false;
      }
      if (!flag && (cost.objective[i] < other.cost.objective[i])) {
        flag = true;
      }
    }
    return flag;
  }

  constexpr void info(std::ostream &out = std::cout) const {
    out << "particle info:\n";
    out << "\tcost=(";
    for (size_t i = 0; i < OBJECTIVES - 1; i++) {
      out << cost.objective[i] << ", ";
    }
    out << cost.objective.back() << ")\n";
    out << "\tinfeasiblity = " << cost.infeasiblity << '\n';
    out << "\tx=(";
    for (size_t i = 0; i < NUM_VARS - 1; i++) {
      out << position[i] << ", ";
    }
    out << position.back() << ")\n";
    out << "\tv=(";
    for (size_t i = 0; i < NUM_VARS - 1; i++) {
      out << velocity[i] << ", ";
    }
    out << velocity.back() << ")\n";
    out << "\tpBest:" << '\n';
    out << "\t\tcost=(";
    for (size_t i = 0; i < OBJECTIVES - 1; i++) {
      out << pBest_cost.objective[i] << ", ";
    }
    out << pBest_cost.objective.back() << ")\n";
    out << "\t\tinfeasiblity = " << pBest_cost.infeasiblity << '\n';
    out << "\t\tx=(";
    for (size_t i = 0; i < NUM_VARS - 1; i++) {
      out << pBest_position[i] << ", ";
    }
    out << pBest_position.back() << ")\n";
    if (is_dominated) {
      out << "is dominated" << '\n';
    } else {
      out << "is not dominated" << '\n';
    }
    out << "grid index:" << grid_index << '\n';
  }
  constexpr void export_csv(std::ostream &out) const {
    out << '"';
    for (size_t i = 0; i < NUM_VARS - 1; i++) {
      out << position[i] << ',';
    }
    out << position.back() << "\",\"";
    for (size_t i = 0; i < OBJECTIVES - 1; i++) {
      out << cost.objective[i] << ',';
    }
    out << cost.objective.back() << "\",";
    out << cost.infeasiblity << ",\"";
    for (size_t i = 0; i < NUM_VARS - 1; i++) {
      out << pBest_position[i] << ',';
    }
    out << pBest_position.back() << "\",\"";
    for (size_t i = 0; i < OBJECTIVES - 1; i++) {
      out << pBest_cost.objective[i] << ",";
    }
    out << pBest_cost.objective.back() << "\",";
    out << pBest_cost.infeasiblity << ',';
    if (is_dominated) {
      out << "yes,";
    } else {
      out << "no,";
    }
    out << grid_index << '\n';
  }
  constexpr static void export_csv(std::ostream &out,
                                   std::span<const Particle> swarm) {
    if (swarm.empty()) {
      return;
    }
    out << "x,cost,infeasiblity,pBest,pBest_cost,pBest_infeasiblity,is_"
           "dominated,grid_index\n";
    for (const auto &particle : swarm) {
      particle.export_csv(out);
    }
  }
  constexpr static void update_domination(std::span<Particle> swarm) {
    for (size_t i = 0; i < swarm.size(); i++) {
      swarm[i].is_dominated = false;
      for (size_t j = 0; j < swarm.size(); j++) {
        if (i == j) {
          continue;
        }
        if (swarm[j].dominates(swarm[i])) {
          swarm[i].is_dominated = true;
          break;
        }
      }
    }
  }
  constexpr void update_grid_index(const Grid &grid) {
    size_t GridSubIndex = 1;
    grid_index = GridSubIndex; // after the first iteration = the GridSubIndex
    for (size_t i = 0; i < OBJECTIVES; i++) {
      for (size_t j = 0; j < grid[i].size(); j++) {
        if (cost.objective[i] < grid[i][j]) {
          GridSubIndex = j;
          break;
        }
      }
      grid_index = ((grid_index - 1) * grid[i].size()) + GridSubIndex;
    }
  }

private:
  constexpr void
  updateV(const Particle &gBest, const double weight = DEFAULT_WEIGHT,
          const Coefficient &coefficients = DEFAULT_COEFFICIENTS) {
    for (size_t i = 0; i < NUM_VARS; i++) {
      velocity[i] = (weight * velocity[i]) +
                    (coefficients.personal * rnd::rand() *
                     (pBest_position[i] - position[i])) +
                    (coefficients.global * rnd::rand() *
                     (gBest.position[i] - position[i]));
    }
  }
  constexpr void updateX() {
    for (size_t i = 0; i < NUM_VARS; i++) {
      position[i] += velocity[i];
      if (position[i] > upper_bound[i] || position[i] < lower_bound[i]) {
        velocity[i] *= -1;
        position[i] += 2 * velocity[i];
        while (position[i] > upper_bound[i] || position[i] < lower_bound[i]) {
          position[i] -= velocity[i];
          velocity[i] *= -0.5; // NOLINT(readability-magic-numbers,
                               // cppcoreguidelines-avoid-magic-numbers)
          position[i] += velocity[i];
        }
      }
    }
  }
  constexpr void updatePBest() {
    if (cost.infeasiblity < pBest_cost.infeasiblity) {
      pBest_position = position;
      pBest_cost = cost;
    } else if (cost.infeasiblity == pBest_cost.infeasiblity) {
      bool flag = false;
      for (size_t i = 0; i < OBJECTIVES; i++) {
        if (cost.objective[i] > pBest_cost.objective[i]) {
          return;
        }
        if (!flag && (cost.objective[i] < pBest_cost.objective[i])) {
          flag = true;
        }
      }
      if (flag) {
        pBest_position = position;
        pBest_cost = cost;
      }
    }
  }
  constexpr void
  Mutate(const Problem &problem,
         const double mutation_probablity = DEFAULT_MUTATION_PROBABLITY) {
    if (rnd::rand() > mutation_probablity) {
      return;
    }
    const auto candidate = rnd::unifrnd<size_t>(0, NUM_VARS - 1);
    const double delta_x = // NOLINT(cppcoreguidelines-init-variables)
        mutation_probablity * (upper_bound[candidate] - lower_bound[candidate]);
    const double new_lower_bound = // NOLINT(cppcoreguidelines-init-variables)
        std::max(position[candidate] - delta_x, lower_bound[candidate]);
    const double new_upper_bound = // NOLINT(cppcoreguidelines-init-variables)
        std::min(position[candidate] + delta_x, upper_bound[candidate]);
    auto new_position = position;
    new_position[candidate] = rnd::unifrnd(new_lower_bound, new_upper_bound);
    auto new_cost = problem(new_position);
    if (new_cost.infeasiblity <= cost.infeasiblity) {
      bool replace_flag = false;
      for (size_t i = 0; i < OBJECTIVES; i++) {
        if (new_cost.objective[i] > cost.objective[i]) {
          if (rnd::rand() < THRESHOLD) {
            position[candidate] = new_position[candidate];
            cost.objective = new_cost.objective;
            cost.infeasiblity = new_cost.infeasiblity;
          }
          return;
        }
        if (!replace_flag && (new_cost.objective[i] < cost.objective[i])) {
          replace_flag = true;
        }
      }
      if (replace_flag) {
        position[candidate] = new_position[candidate];
        cost.objective = new_cost.objective;
        cost.infeasiblity = new_cost.infeasiblity;
      }
    } else if (rnd::rand() < THRESHOLD) {
      position[candidate] = new_position[candidate];
      cost.objective = new_cost.objective;
      cost.infeasiblity = new_cost.infeasiblity;
    }
  }

  variables lower_bound{};
  variables upper_bound{};
  variables position{};
  variables velocity{};
  variables pBest_position{};
  Cost cost{};
  Cost pBest_cost{};
  size_t grid_index{0};
  bool is_dominated{false};

  friend class Repository<NUM_VARS, OBJECTIVES, GRID_SIZE>;
};

template <size_t SWARM_SIZE, size_t NUM_VARS, size_t OBJECTIVES,
          size_t GRID_SIZE = DEFAULT_GRID_SIZE>
struct Swarm {
  using Particle = Particle<NUM_VARS, OBJECTIVES, GRID_SIZE>;
  std::array<Particle, SWARM_SIZE> particles;

  constexpr Swarm() = default;
  constexpr Swarm(const Variables<NUM_VARS> &lower_bound,
                  const Variables<NUM_VARS> &upper_bound,
                  const Problem<OBJECTIVES> &problem) {
    for (auto &particle : particles) {
      particle = Particle(lower_bound, upper_bound, problem);
    }
  }

  constexpr void update_particles(const Particle &gBest,
                                  const Problem<OBJECTIVES> &problem,
                                  const double weight,
                                  const Coefficient &coefficients,
                                  const double mutation_probablity) {
    for (auto &particle : particles) {
      particle.update(gBest, problem, weight, coefficients,
                      mutation_probablity);
    }
  }

  constexpr void export_csv(std::ostream &out) const {
    Particle::export_csv(out, particles);
  }

  constexpr explicit operator std::span<Particle>() { return particles; }
  constexpr explicit operator std::span<const Particle>() const {
    return particles;
  }
};
} // namespace mopso
#endif // PARTICLE_H
