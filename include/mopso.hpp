#ifndef PSO_H
#define PSO_H
#include <cmath>
#include "particle.hpp"
#include "repository.hpp"
namespace mopso {
  template<ull N,ull O,ull S=100>
  std::pair<Repository<N,O>,std::array<Particle<N,O>,S>> mopso(const vars<N>& lower_bound,const vars<N>& upper_bound,const Problem<N,O>& problem,const size_t max_iter=1000,size_t Rep_size=100,size_t grid_size=7,size_t alpha=0.1,size_t beta=2,size_t gamma=2,const std::array<double,2>& c={0.2,0.2},const std::array<double,2>& iw={0.1,0.01},const double mu=0.1){
    auto w = [&](size_t it){ return ((max_iter - it) - (iw[0] - iw[1]))/max_iter + iw[1];};
    auto pm = [&](size_t it){ return std::pow(1-it/(max_iter-1.),1/mu);};
    std::array<Particle<N,O>,S> swarm;
    for (size_t i = 0; i < S; i++)
      swarm[i]=Particle<N,O>(lower_bound,upper_bound,problem);
    Repository<N,O> rep(swarm,Rep_size,grid_size,alpha,beta,gamma);
    for (size_t i = 0; i < max_iter; i++) {
      Particle<N,O> gBest = rep.SelectLeader();
      auto wc = w(i);
      auto pc = pm(i);
      for (size_t j = 0; j < S; j++) {
        swarm[j].update(gBest,problem,wc,c,pc);
      }
      rep.update(swarm);
    }
    return std::make_pair(rep,swarm);
  }
} /* pso */
#endif //PSO_H
