#include <util/benchmark.hpp>
#include <util/parallel_for.hpp>
#include <util/sycl_devices.hpp>

#include <array>
#include <ranges>
#include <random>

#include <Eigen/Dense>
#include <sycl/sycl.hpp>

namespace
{
  constexpr size_t dim = 4;
  using matrix_storage = std::array<float, dim * dim>;

  void invert(matrix_storage& storage)
  {
    Eigen::Map<Eigen::Matrix<float, dim, dim, Eigen::RowMajor>> mat(storage.data());
    mat = mat.inverse().eval();
  }

  void calculate_native(const std::span<matrix_storage> data)
  {
    for (auto& storage : data)
      invert(storage);
  }

  void calculate_native_parallel(const std::span<matrix_storage> data)
  {
    quxflux::parallel_for_naive(data.size(), [=](const size_t i) { invert(data[i]); });
  }

  class invert_matrix_kernel;
  auto make_sycl_implementation(const sycl::device& device)
  {
    return [queue = sycl::queue{device}](const std::span<matrix_storage> data) mutable {
      sycl::buffer<matrix_storage> matrix_buffer(data);

      queue.submit([&](sycl::handler& handler) {
        sycl::accessor device_matrices{matrix_buffer, sycl::read_write};

        handler.require(device_matrices);
        handler.parallel_for<invert_matrix_kernel>(matrix_buffer.get_range(), [=](const sycl::item<1> idx) { invert(device_matrices[idx]); });
      });
    };
  }

  auto make_sycl_implementations()
  {
    std::vector<std::pair<std::string, std::function<void(std::span<matrix_storage>)>>> impls;

    for (const auto& device : quxflux::get_supported_devices<invert_matrix_kernel>())
      impls.emplace_back(quxflux::get_name(device), make_sycl_implementation(device));

    return impls;
  }
}  // namespace

int main()
{
  auto implementations = make_sycl_implementations();
  implementations.insert(implementations.begin(), std::pair{"host chunked parallel", &calculate_native_parallel});
  implementations.insert(implementations.begin(), std::pair{"host", &calculate_native});

  constexpr size_t num_matrices = 1'000'000;

  const auto matrices = [] {
    std::vector<matrix_storage> t(num_matrices);

    auto generator = [gen = std::mt19937{42}, dis = std::uniform_real_distribution{0.f, 1.f}]() mutable {
      return dis(gen);
    };

    for (auto& storage : t)
      std::ranges::generate(storage, generator);

    return t;
  }();

  std::vector<matrix_storage> result(num_matrices);
  for (const auto& [name, f] : implementations)
  {
    std::cout << "benchmarking implementation: " << name << '\n';

    const auto median_duration = quxflux::benchmark([&] { f(result); }, 10,  //
                                                    [&] { std::ranges::copy(matrices, result.begin()); },
                                                    [&] {
                                                      // make sure compiler does not outsmart us and put out a rudimentary checksum
                                                      quxflux::do_not_optimize(result);
                                                      int32_t checksum = 0;
                                                      for (const auto& m : result)
                                                        for (const auto& c : m)
                                                          checksum += static_cast<int32_t>(c * 100000);

                                                      std::clog << "checksum: " << checksum << '\n';
                                                    });

    std::cout << "median exec time:" << median_duration.count() << " s, "  //
              << "processing speed: " << static_cast<double>(matrices.size()) / double{std::mega::num} / median_duration.count() << " mio. matrices/s\n";
  }
}
