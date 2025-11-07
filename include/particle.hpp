#ifndef PARTICLE_H
#define PARTICLE_H
#include <iostream>
#include <array>
#include <functional>
#include "rand.hpp"
namespace mopso {
  using ull=unsigned long long;
  template<ull N>
  using vars=std::array<double,N>;
  template<ull N,ull O>
  using Problem=std::function<std::pair<vars<O>,double>(vars<N>)>;
  template <ull N, ull O> class Repository;
  template<ull O> using GRD = std::array<std::vector<double>,O>;
  template<ull N,ull O>
  class Particle {
    public:
      inline Particle() = default;
      inline Particle(Particle const&) = default;
      inline Particle(Particle&&) = default;
      inline Particle& operator=(Particle const&) = default;
      inline Particle& operator=(Particle&&) = default;
      inline Particle(const vars<N>& l,const vars<N>& u,const Problem<N,O>& problem):l{l},u{u} {
        for (size_t i = 0; i < N; i++){
          x[i]=rnd::unifrnd(l[i],u[i]);
          v[i]=0.;
        }
        std::tie(cost,infeasiblity)=problem(x);
        pBest=x;
        pBest_cost=cost;
        pBest_infeasiblity=infeasiblity;
        is_dominated=false;
      }
      inline void update(const Particle& gBest,const Problem<N,O>& problem,const double w=0.5,const std::array<double,2>& c={0.2,0.2},const double pm=0.1){
        updateV(gBest,w,c);
        updateX();
        std::tie(cost,infeasiblity) = problem(x);
        Mutate(problem,pm);
        updatePBest();
      }
      inline bool dominates(const Particle& b)const&{
        if (infeasiblity<b.infeasiblity) return true;
        if (infeasiblity>b.infeasiblity) return false;
        bool f = false;
        for (size_t i = 0; i < O; i++){
          if (cost[i]>b.cost[i]) return false;
          if (!f && (cost[i]<b.cost[i])) f = true;
        }
        return f;
      }
      template <ull S>
      inline static Particle get_Best(const std::array<Particle,S>& swarm){
        return *std::min_element(swarm.begin(),swarm.end(),
                                [](const auto& a,const auto& b){
                                  return a.dominates(b);
                                });
      }
      template <ull S>
      inline static void update_domination(std::array<Particle,S>& swarm){
        for (size_t i = 0; i < S; i++) {
          swarm[i].is_dominated=false;
          for (size_t j = 0; j < S; j++) {
            if (i==j) continue;
            if (swarm[j].dominates(swarm[i])){
              swarm[i].is_dominated=true;
              break;
            }
          }
        }
      }
      inline void info(std::ostream& out=std::cout) const&{
        out << "particle info:\n";
        out << "\tcost=(";
        for (size_t i = 0; i < O-1; i++)
          out << cost[i] << ", ";
        out << cost.back() << ")\n";
        out << "\tinfeasiblity = " << infeasiblity << '\n';
        out << "\tx=(";
        for (size_t i = 0; i < N-1; i++)
          out << x[i] << ", ";
        out << x.back() << ")\n";
        out << "\tv=(";
        for (size_t i = 0; i < N-1; i++)
          out << v[i] << ", ";
        out << v.back() << ")\n";
        out << "\tpBest:" << '\n';
        out << "\t\tcost=(";
        for (size_t i = 0; i < O-1; i++)
          out << pBest_cost[i] << ", ";
        out << pBest_cost.back() << ")\n";
        out << "\t\tinfeasiblity = " << pBest_infeasiblity << '\n';
        out << "\t\tx=(";
        for (size_t i = 0; i < N-1; i++)
          out << pBest[i] << ", ";
        out << pBest.back() << ")\n";
        if (is_dominated) out << "is dominated:yes" << '\n';
        else out << "is dominated:no" << '\n';
        out << "grid index:" << grid_index << '\n';
      }
      inline void csv_out(std::ostream& out,bool header=true) const&{
        if (header) out<< "x,cost,infeasiblity,pBest,pBest_cost,pBest_infeasiblity,is_dominated,grid_index\n";
        out << '"';
        for (size_t i = 0; i < N-1; i++) out << x[i] << ',';
        out << x.back() << "\",\"";
        for (size_t i = 0; i < O-1; i++) out << cost[i] << ',';
        out << cost.back() << "\",";
        out << infeasiblity << ",\"";
        for (size_t i = 0; i < N-1; i++)
          out << pBest[i] << ',';
        out << pBest.back() << "\",\"";
        for (size_t i = 0; i < O-1; i++)
          out << pBest_cost[i] << ",";
        out << pBest_cost.back() << "\",";
        out << pBest_infeasiblity << ',';
        if (is_dominated) out << "yes,";
        else out << "no,";
        out << grid_index << '\n';
      }
      inline static void csv_out(std::ostream& out,std::vector<Particle>& swarm){
        swarm[0].csv_out(out,true);
        for (size_t i = 1; i < swarm.size(); i++)
          swarm[i].csv_out(out,false);
      }
      template<ull T>
      inline static void csv_out(std::ostream& out,std::array<Particle,T>& swarm){
        swarm[0].csv_out(out,true);
        for (size_t i = 1; i < T; i++)
          swarm[i].csv_out(out,false);
      }
      inline void update_grid_index(const GRD<O>& g){
        ull nGrid = g[0].size();
        ull GridSubIndex=0;
        for (size_t j = 0; j < nGrid; j++) {
          if (cost[0]<g[0][j]) {
            GridSubIndex=j;
            break;
          }
        }
        grid_index = GridSubIndex;
        for (size_t i = 1; i < O; i++) {
          for (size_t j = 0; j < nGrid; j++) {
            if (cost[i]<g[i][j]) {
              GridSubIndex=j;
              break;
            }
          }
          grid_index--;
          grid_index*=nGrid;
          grid_index+=GridSubIndex;
        }
      }
    private:
      inline void updateV(const Particle& gBest,const double w=0.5,const std::array<double,2>& c={0.2,0.2}){
        for (size_t i = 0; i < N; i++)
          v[i]=w*v[i]+c[0]*rnd::rand()*(pBest[i]-x[i])+c[1]*rnd::rand()*(gBest.x[i]-x[i]);
      }
      inline void updateX(){
        for (size_t i = 0; i < N; i++) {
          x[i]+=v[i];
          if (x[i]>u[i] || x[i]<l[i]){
            v[i]*=-1;
            x[i]+=2*v[i];
            if (x[i]>u[i] || x[i]<l[i]){
              do {
                x[i]-=v[i];
                v[i]*=-0.5;
                x[i]+=v[i];
              } while(x[i]>u[i] || x[i]<l[i]);
            }
          }
        }
      }
      inline void updatePBest(){
        if (infeasiblity<pBest_infeasiblity) {
          pBest=x;
          pBest_cost=cost;
          pBest_infeasiblity=infeasiblity;
        } else if (infeasiblity==pBest_infeasiblity){
          bool f = false;
          for (size_t i = 0; i < O; i++){
            if (cost[i]>pBest_cost[i]) return;
            if (!f && (cost[i]<pBest_cost[i])) f = true;
          }
          if(f){
            pBest=x;
            pBest_cost=cost;
            pBest_infeasiblity=infeasiblity;
          }
        }
      }
      inline void Mutate(const Problem<N,O>& problem,const double pm=0.1){
        if (rnd::rand()>pm) return;
        size_t j = rnd::unifrnd<size_t>(0,N);
        double dx=pm*(u[j]-l[j]);
        double lb=std::max(x[j]-dx,l[j]);
        double ub=std::min(x[j]+dx,u[j]);
        auto X = x;
        X[j]=rnd::unifrnd(lb,ub);
        auto [c,i] = problem(X);
        if (i <= infeasiblity){
          bool f = false;
          for (size_t i = 0; i < O; i++){
            if (c[i]>cost[i]) {
              if (rnd::rand()<0.5){
                x[j]=X[j];
                cost = c;
                infeasiblity = i;
              }
              return;
            }
            if (!f && (c[i]<cost[i])) f = true;
          }
          if(f){
            x[j]=X[j];
            cost = c;
            infeasiblity = i;
          }
        } else if(rnd::rand()<0.5) {
          x[j]=X[j];
          cost = c;
          infeasiblity = i;
        }
      }
      vars<N> l;
      vars<N> u;
      vars<N> x;
      vars<N> v;
      vars<O> cost;
      double infeasiblity;
      vars<N> pBest;
      vars<O> pBest_cost;
      double pBest_infeasiblity;
      bool is_dominated;
      ull grid_index;
      friend class Repository<N,O>;
  };
} /* pso */
#endif //PARTICLE_H
