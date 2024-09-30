#include <util/benchmark.hpp>
#include <util/parallel_for.hpp>
#include <util/sycl_devices.hpp>

#include <array>
#include <ranges>
#include <random>

#include <sycl/sycl.hpp>

namespace
{
  template<typename T>
  constexpr T interpolate_bezier_cubic(const std::span<const T, 4> control_points, const typename T::value_type t)
  {
    const auto one_minus_t = typename T::value_type{1} - t;
    return {one_minus_t * one_minus_t * one_minus_t * control_points[0] +  //
            3 * one_minus_t * one_minus_t * t * control_points[1] +        //
            3 * one_minus_t * t * t * control_points[2] +                  //
            t * t * t * control_points[3]};                                //
  }

  struct point3f : std::array<float, 3>
  {};

  constexpr point3f operator*(const float t, const point3f& p)
  {
    return {t * p[0], t * p[1], t * p[2]};
  }

  constexpr point3f operator+(const point3f& a, const point3f& b)
  {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
  }

  using impl = std::function<void(std::span<point3f>, std::span<const point3f, 4>, std::span<const float>)>;
  void benchmark(const std::span<point3f> result, const std::span<const point3f, 4> control_points, const std::span<const float> ts, const impl& f, const std::string& impl_name)
  {
    std::cout << "Benchmarking " << impl_name << "...\n";
    std::vector<std::chrono::duration<double>> timing;

    for (size_t i = 0; i < 5; ++i)
    {
      std::ranges::fill(result, point3f{});

      timing.push_back(quxflux::measure_exec_time([&] { f(result, control_points, ts); }));
      quxflux::do_not_optimize(result);
    }

    std::ranges::nth_element(timing, std::ranges::next(std::ranges::begin(timing), std::ranges::ssize(timing) / 2));
    const auto median_dur = *std::ranges::next(std::ranges::begin(timing), std::ranges::ssize(timing) / 2);

    std::cout << "Median duration: " << median_dur.count() << " s, rate: " << (static_cast<double>(result.size()) / double{std::mega::num}) / median_dur.count() << " mPts/s\n";
  }

  void calculate_native(const std::span<point3f> result, const std::span<const point3f, 4> control_points, const std::span<const float> ts)
  {
    for (size_t i = 0, n = std::min(result.size(), ts.size()); i < n; ++i)
      result[i] = interpolate_bezier_cubic(control_points, ts[i]);
  }

  void calculate_native_parallel(const std::span<point3f> result, const std::span<const point3f, 4> control_points, const std::span<const float> ts)
  {
    quxflux::parallel_for_naive(std::min(result.size(), ts.size()), [&](const size_t i) { result[i] = interpolate_bezier_cubic(control_points, ts[i]); });
  }

  class interpolate_bezier_cubic_kernel;
  auto make_sycl_implementation(const sycl::device& device)
  {
    return [queue = sycl::queue{device}](const std::span<point3f> result, const std::span<const point3f, 4> control_points, const std::span<const float> ts) mutable {
      const auto n = std::min(result.size(), ts.size());

      sycl::buffer<point3f> results_buffer(result);
      sycl::buffer<const float> ts_buffer(ts);

      std::array<point3f, 4> cps{};
      std::ranges::copy(control_points, cps.begin());

      queue.submit([&](sycl::handler& handler) {
        sycl::accessor device_results{results_buffer, sycl::write_only};
        sycl::accessor device_ts{ts_buffer, sycl::read_only};

        handler.require(device_results);
        handler.require(device_ts);
        handler.parallel_for<interpolate_bezier_cubic_kernel>(sycl::range{n}, [=](const sycl::item<1> idx) { device_results[idx] = interpolate_bezier_cubic(std::span{cps}, device_ts[idx]); });
      });
    };
  }

  auto make_sycl_implementations()
  {
    std::vector<std::pair<std::string, impl>> impls;

    for (const auto& device : quxflux::get_supported_devices<interpolate_bezier_cubic_kernel>())
      impls.emplace_back(quxflux::get_name(device), make_sycl_implementation(device));

    return impls;
  }
}  // namespace

int main()
{
  static constexpr auto control_points = std::to_array<point3f>({
    {0.f, 0.f, 0.f},
    {1.f, 0.f, .2f},
    {2.f, 0.f, -.8f},
    {3.f, 0.f, .1f},
  });

  auto implementations = make_sycl_implementations();
  implementations.insert(implementations.begin(), std::pair{"host chunked parallel", &calculate_native_parallel});
  implementations.insert(implementations.begin(), std::pair{"host", &calculate_native});

  constexpr size_t num_points = 10'000'000;

  const auto independent_variables = [] {
    std::vector<float> t(num_points);
    std::ranges::generate(t, [gen = std::mt19937{42}, dis = std::uniform_real_distribution{0.f, 1.f}]() mutable { return dis(gen); });
    return t;
  }();

  std::vector<point3f> interpolants(num_points);
  for (const auto& [name, f] : implementations)
  {
    std::cout << "benchmarking implementation: " << name << '\n';

    const auto median_duration =quxflux::benchmark([&] { f(interpolants, control_points, independent_variables); }, 10, [&] { std::ranges::fill(interpolants, point3f{}); },
                       [&] {
                         // make sure compiler does not outsmart us and put out a rudimentary checksum
                         quxflux::do_not_optimize(interpolants);
                         int32_t checksum = 0;
                         for (const auto& p : interpolants)
                           for (const auto& c : p)
                             checksum += static_cast<int32_t>(c * 1000);

                         std::clog << "checksum: " << checksum << '\n';
                       });

    std::cout << "median exec time:" << median_duration.count() << " s, "  //
              << "processing speed: " << static_cast<double>(interpolants.size()) / double{std::mega::num} / median_duration.count() << " mio. points/s\n";
  }
}
