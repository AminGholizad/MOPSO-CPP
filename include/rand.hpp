#ifndef RAND
#define RAND
#include <chrono>
#include <random>
namespace rnd {
[[nodiscard]] constexpr std::mt19937 &Generator() {
  static std::random_device random_device;
  static unsigned seed =
      (random_device.entropy() == 0)
          ? static_cast<unsigned>(
                std::chrono::system_clock::now().time_since_epoch().count())
          : random_device();
  static std::mt19937 gen(seed);
  return gen;
}

template <typename T>
concept NumericType = std::integral<T> || std::floating_point<T>;

template <NumericType T>
[[nodiscard]] constexpr T unifrnd(T min_val, T max_val) {
  if constexpr (std::is_integral_v<T>) {
    std::uniform_int_distribution<T> dis(min_val, max_val);
    return dis(Generator());
  } else {
    std::uniform_real_distribution<T> dis(min_val, max_val);
    return dis(Generator());
  }
}
[[nodiscard]] constexpr double rand() { return unifrnd(0.0, 1.0); }
} // namespace rnd
#endif // RAND
