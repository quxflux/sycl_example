#include <util/benchmark.hpp>
#include <util/parallel_for.hpp>
#include <util/sycl_devices.hpp>

#include <sorting_network_cpp/sorting_network.h>

#include <array>
#include <ranges>
#include <random>

#include <sycl/sycl.hpp>

namespace
{
  using data_set = std::array<uint8_t, 5 * 5>;
  using sorting_net = quxflux::sorting_net::sorting_network<std::tuple_size_v<data_set>>;

  void calculate_native(const std::span<data_set> data)
  {
    for (auto& t : data)
      sorting_net{}(t.begin());
  }

  void calculate_native_parallel(const std::span<data_set> data)
  {
    quxflux::parallel_for_naive(data.size(), [&](const size_t i) { sorting_net{}(data[i].begin()); });
  }

  class sorting_net_kernel;
  auto make_sycl_implementation(const sycl::device& device)
  {
    return [queue = sycl::queue{device}](const std::span<data_set> data) mutable {
      sycl::buffer<data_set> data_buffer(data);

      queue.submit([&](sycl::handler& handler) {
        sycl::accessor device_data{data_buffer, sycl::read_write};

        handler.require(device_data);
        handler.parallel_for<sorting_net_kernel>(data_buffer.get_range(), [=](const sycl::item<1> idx) { sorting_net{}(device_data[idx].begin()); });
      });
    };
  }

  auto make_sycl_implementations()
  {
    std::vector<std::pair<std::string, std::function<void(std::span<data_set>)>>> impls;

    for (const auto& device : quxflux::get_supported_devices<sorting_net_kernel>())
      impls.emplace_back(quxflux::get_name(device), make_sycl_implementation(device));

    return impls;
  }
}  // namespace

int main()
{
  auto implementations = make_sycl_implementations();
  implementations.insert(implementations.begin(), std::pair{"host chunked parallel", &calculate_native_parallel});
  implementations.insert(implementations.begin(), std::pair{"host", &calculate_native});

  std::vector<data_set> data(10'000'000);

  std::mt19937 gen{42};
  std::uniform_int_distribution<int32_t> dis;

  for (auto& d : data)
    std::ranges::generate(d, [&] { return static_cast<data_set::value_type>(dis(gen)); });

  std::cout << data.size() << " arrays with each " << data_set{}.size() << " items\n";

  for (const auto& [name, f] : implementations)
  {
    std::cout << "benchmarking implementation: " << name << '\n';

    std::vector data_cpy = data;
    const auto median_duration = quxflux::benchmark([&] { f(data_cpy); }, 10,
                                                    // every run should process the same data
                                                    [&] { std::ranges::copy(data, data_cpy.begin()); },
                                                    [&] {
                                                      // make sure compiler does not outsmart us and put out a rudimentary checksum
                                                      quxflux::do_not_optimize(data_cpy);
                                                      uint32_t checksum = 0;
                                                      for (const auto& d : data_cpy)
                                                        for (const auto t : d)
                                                          checksum += t;

                                                      std::clog << "checksum: " << checksum << '\n';
                                                    });

    std::cout << "median exec time:" << median_duration.count() << " s, "  //
              << "sorting speed: " << static_cast<double>(data.size() * data_set{}.size()) / double{std::mega::num} / median_duration.count() << " mio. items/s\n";
  }
}
