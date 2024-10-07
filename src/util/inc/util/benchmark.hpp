#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace quxflux
{
  namespace detail
  {
    void do_not_optimize(const volatile void*);
  }

  template<typename T>
  decltype(auto) do_not_optimize(T&& t)
  {
    detail::do_not_optimize(&t);
    return std::forward<T>(t);
  }

  template<typename Duration = std::chrono::duration<double>>
  auto measure_exec_time(const auto& f)
  {
    using clock = std::chrono::steady_clock;

    const auto start = do_not_optimize(clock::now());
    f();
    const auto end = do_not_optimize(clock::now());

    return std::chrono::duration_cast<Duration>(end - start);
  }

  auto benchmark(const auto& f, const auto& before_measurement = [] {}, const auto& after_measurement = [] {}, size_t n = 10)
  {
    std::vector<std::chrono::duration<double>> durations;

    for (size_t i = 0; i < n; ++i)
    {
      before_measurement();
      durations.push_back(quxflux::measure_exec_time([&] { f(); }));
      after_measurement();
    }

    std::ranges::nth_element(durations, std::ranges::next(std::ranges::begin(durations), std::ranges::ssize(durations) / 2));
    return *std::ranges::next(std::ranges::begin(durations), std::ranges::ssize(durations) / 2);
  }
}  // namespace quxflux
