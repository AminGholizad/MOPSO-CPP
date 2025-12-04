#ifndef REP
#define REP
#include "particle.hpp"
#include <Random-Helper.hpp>
#include <map>
#include <vector>
namespace mopso {
template <size_t NUM_VARS, size_t OBJECTIVES,
          size_t GRID_SIZE = DEFAULT_GRID_SIZE>
class Repository {

  using Particle = Particle<NUM_VARS, OBJECTIVES, GRID_SIZE>;
  using Grid = std::array<std::array<double, GRID_SIZE + 1>, OBJECTIVES>;

public:
  constexpr Repository() = default;
  constexpr Repository(std::span<Particle> curr_swarm, size_t r_size,
                       double a_val, double b_val, double ga_val)
      : repository_size{r_size}, alpha{a_val}, beta{b_val}, gamma{ga_val} {
    Particle::update_domination(curr_swarm);
    extend_swarm(curr_swarm);
    update_grid();
    Particle::update_grid_index(grid, swarm);
  }

  [[nodiscard]] constexpr Particle SelectLeader() const {
    size_t selected = select_index(-beta);
    return swarm[selected];
  }

  constexpr void update(std::span<Particle> curr_swarm) {
    Particle::update_domination(curr_swarm);
    extend_swarm(curr_swarm);
    update_grid();
    Particle::update_grid_index(grid, swarm);
    delete_extra_rep_memebrs();
  }

  [[nodiscard]] constexpr size_t size() const { return swarm.size(); }

  constexpr void info(std::ostream &out = std::cout) const {
    for (const auto &particle : swarm) {
      particle.info(out);
      out << '\n';
    }
  }

  constexpr void export_csv(std::ostream &out) const {
    Particle::export_csv(out, swarm);
  }

private:
  std::vector<Particle> swarm{};
  Grid grid{};
  size_t repository_size{0};
  double alpha{0};
  double beta{0};
  double gamma{0};

  constexpr void extend_swarm(std::span<const Particle> curr_swarm) {
    for (auto particle : curr_swarm) {
      if (!particle.is_dominated) {
        swarm.push_back(std::move(particle));
      }
    }
  }
  constexpr void update_grid() {
    for (size_t i = 0; i < OBJECTIVES; i++) {
      const auto [mini, maxi] = std::minmax_element(
          swarm.cbegin(), swarm.cend(),
          [&](const auto &particle_a, const auto &particle_b) {
            return particle_a.cost.objective[i] < particle_b.cost.objective[i];
          });
      double cmini = mini->cost.objective[i];
      double cmaxi = maxi->cost.objective[i];

      double delta_c = cmaxi - cmini;
      cmini -= alpha * delta_c;
      cmaxi += alpha * delta_c;
      const double delta = (cmaxi - cmini) / static_cast<double>(GRID_SIZE - 1);

      double val = cmini;
      for (auto &grid_i : grid[i]) {
        grid_i = val;
        val += delta;
      }

      grid[i][GRID_SIZE] = (std::numeric_limits<double>::max());
    }
  }

  [[nodiscard]] constexpr size_t select_index(const double tau) const {
    std::map<size_t, size_t> mOC;
    for (const auto &particle : swarm) {
      if (mOC.find(particle.grid_index) != mOC.end()) {
        mOC[particle.grid_index]++;
      } else {
        mOC[particle.grid_index] = 1;
      }
    }
    std::map<size_t, double> probabilities;
    double sum = 0.0;
    for (const auto &[key, val] : mOC) {
      const double mag = std::exp(tau * static_cast<double>(val));
      sum += mag;
      probabilities[key] = mag;
    }
    for (auto &[key, val] : probabilities) {
      val /= sum;
    } // normalize

    double cumsum = 0.0;
    for (const auto &[key, val] : probabilities) {
      cumsum += val;
      probabilities[key] = cumsum;
    }
    // roulette
    size_t sci = 0;
    const double random_number = rnd::rand();
    for (const auto &[key, val] : mOC) {
      if (random_number <= probabilities[key]) {
        sci = key;
        break;
      }
    }
    // end_roulette
    auto smi = rnd::unifrnd<size_t>(0, mOC[sci] - 1);
    size_t GIi = 0;
    for (size_t i = 0; i < swarm.size(); i++) {
      if (smi == 0) {
        GIi = i;
        break;
      }
      if (swarm[i].grid_index == sci) {
        smi--;
      }
    }
    return GIi;
  }

  constexpr void delete_extra_rep_memebrs() {
    while (swarm.size() > repository_size) {
      const auto selected_index = select_index(gamma);
      auto iter = swarm.begin();
      std::advance(iter, selected_index);
      swarm.erase(iter);
    }
  }
};
} // namespace mopso
#endif
