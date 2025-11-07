#ifndef REP
#define REP
#include <vector>
#include <map>
#include <limits>
#include "rand.hpp"
#include "particle.hpp"
namespace mopso {
  template <ull N, ull O>
  class Repository {
    public:
      inline Repository() = default;
      inline Repository(Repository const&) = default;
      inline Repository(Repository&&) = default;
      inline Repository& operator=(Repository const&) = default;
      inline Repository& operator=(Repository&&) = default;
      template <ull S>
      inline Repository (std::array<Particle<N,O>,S> s,ull r,ull g,double a,double b,double ga)
            :rep_size{r},grid_size{g},alpha{a},beta{b},gamma{ga}{
              Particle<N,O>::update_domination(s);
              for (size_t i = 0; i < S; i++)
                if (!s[i].is_dominated)
                  swarm.push_back(s[i]);
              get_grid();
              for (size_t i = 0; i < swarm.size(); i++)
                swarm[i].update_grid_index(Grid);
            }
      inline Particle<N,O> SelectLeader()const&{
        size_t sm = select_index(-beta);
        return swarm[sm];
      }
      template <ull S>
      inline void update(std::array<Particle<N,O>,S>& s){
        Particle<N,O>::update_domination(s);
        for (size_t i = 0; i < S; i++)
          if (!s[i].is_dominated)
            swarm.push_back(s[i]);
        get_grid();
        for (size_t i = 0; i < swarm.size(); i++)
          swarm[i].update_grid_index(Grid);
        while (swarm.size() > rep_size) DeleteOneRepMemebr();
      }
      inline size_t size()const&{
        return swarm.size();
      }
      inline void info(std::ostream& out=std::cout){
        for (auto p : swarm){
          p.info(out);
          out << '\n';
        }
      }
      inline void csv_out(std::ostream& out){
        Particle<N,O>::csv_out(out,swarm);
      }
    private:
      ull rep_size;
      std::vector<Particle<N,O>> swarm;
      GRD<O> Grid;
      ull grid_size;
      double alpha;
      double beta;
      double gamma;
      inline void get_grid(){
        for (size_t i = 0; i < O; i++) {
          Grid[i].clear();
          double cmini=std::numeric_limits<double>::max(),cmaxi=-cmini;
          for (size_t j = 0; j < swarm.size(); j++) {
            if (swarm[j].cost[i]>cmaxi)
              cmaxi=swarm[j].cost[i];
            if (swarm[j].cost[i]<cmini)
              cmini=swarm[j].cost[i];
          }
          double dc=cmaxi-cmini;
          cmini-=alpha*dc;
          cmaxi+=alpha*dc;
          double delta = (cmaxi - cmini) / (grid_size-1);
          for(size_t k=0; k < grid_size-1; k++){
            Grid[i].push_back(cmini + delta*k);
          }
          Grid[i].push_back(cmaxi);
          Grid[i].push_back(std::numeric_limits<double>::max());
        }
      }
      inline size_t select_index(double tau)const&{
        std::map<size_t, size_t> mOC;
        for (size_t i = 0; i < swarm.size(); i++) {
          if (mOC.find(swarm[i].grid_index)!=mOC.end()){
            mOC[swarm[i].grid_index]++;
          } else {
            mOC[swarm[i].grid_index]=1;
          }
        }
        std::map<size_t,double> P;
        double s=0.;
        for (auto i : mOC) {
          double m = std::exp(tau*i.second);
          s+=m;
          P[i.first]=m;
        }
        for (auto i : mOC) P[i.first]/=s;//normalize
        //cumsum:
        for (auto i = mOC.begin(); i != mOC.end(); i++){
          auto m = P[i->first];
          std::advance(i,1);
          P[i->first]+=m;
          std::advance(i,-1);
        }
        //roulette
        double sci=0;
        double r=rnd::rand();
        for (auto i : mOC)
          if (r<=P[i.first]) {
            sci = i.first;
            break;
          }
        //end_roulette
        size_t sm = rnd::unifrnd<size_t>(0,mOC[sci]-1);
        size_t GIi = 0;
        for (size_t i = 0; i < swarm.size(); i++){
          if (sm==0){
            GIi = i;
            break;
          }
          if (swarm[i].grid_index==sci) sm--;
        }
        return GIi;
      }
      inline void DeleteOneRepMemebr(){
        size_t sm = select_index(gamma);
        swarm.erase(swarm.begin()+sm);
      }
  };
} /* mopso */
#endif
