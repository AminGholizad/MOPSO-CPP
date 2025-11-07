#include <iostream>
#include <fstream>
#include "mopso.hpp"
#include <cmath>
const double pi = 3.1415926535897932385;
template<mopso::ull N, mopso::ull O>
std::pair<mopso::vars<O>,double>cost_fcn(mopso::vars<N> x){
  double s1 = 0.;
  double s2 = 0.;
  double c = 0.;
  for (size_t i = 0; i < N; i++) {
    s1+=std::sin(x[i]*5)+std::sin(x[i]*7)+std::sin(x[i]*11);
    for (size_t j = 1; j < 100; j++) {
      s2+=std::sin(x[i]*(6*j+1))+std::sin(x[i]*(6*j-1));
    }
    c+=std::sin(x[i]);
  }
  s1=std::abs(s1);
  s2=std::abs(s2);
  c=std::abs(c/N-0.7);
  mopso::vars<O> s{s1,s2};
  return std::make_pair(s,c);
}
int main(){
  const mopso::ull Nvars{4};
  const mopso::ull Nobjs{2};
  const mopso::ull Swarm_size{200};
  const mopso::ull Rep_size{200};
  const mopso::ull max_iter{200};
  mopso::vars<Nvars> l{0,0,0,0};
  mopso::vars<Nvars> u{pi/2,pi/2,pi/2,pi/2};
  auto [rep,swarm] = mopso::mopso<Nvars,Nobjs,Swarm_size>(l,u,cost_fcn<Nvars,Nobjs>,max_iter,Rep_size);
  std::ofstream f("./repository.csv");
  rep.csv_out(f);
  f=std::ofstream ("./swarm.csv");
  mopso::Particle<Nvars,Nobjs>::csv_out(f,swarm);
  rep.SelectLeader().info();
  return 0;
}
