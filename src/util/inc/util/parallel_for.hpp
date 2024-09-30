#pragma once

#include <algorithm>
#include <thread>
#include <type_traits>
#include <vector>

namespace quxflux
{
  namespace detail
  {
    template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    constexpr T int_div_ceil(const T x, const T y)
    {
      return x == 0 ? 0 : 1 + ((x - 1) / y);
    }
  }  // namespace detail

  void parallel_for_naive(const size_t n, const auto& f)
  {
    std::vector<std::jthread> threads;

    const auto num_threads = std::max(std::thread::hardware_concurrency(), 1u);
    const size_t per_thread = detail::int_div_ceil(n, size_t{num_threads});

    for (size_t t_idx = 0; t_idx < num_threads; ++t_idx)
    {
      threads.emplace_back([=] {
        for (size_t i = per_thread * t_idx, max = std::min(per_thread * (t_idx + 1), n); i < max; ++i)
          f(i);
      });
    }
  }
}  // namespace quxflux