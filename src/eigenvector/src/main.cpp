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
  constexpr size_t dim = 3;

  using scalar = float;
  using matrix_storage = std::array<scalar, dim * dim>;
  using vector_storage = std::array<scalar, dim>;

  vector_storage calculate_eigenvector_with_largest_eigenvalue(const matrix_storage& storage)
  {
    using matrix = Eigen::Matrix<scalar, dim, dim, Eigen::RowMajor>;
    using vector = Eigen::Matrix<scalar, 1, dim, Eigen::RowMajor>;

    vector_storage result;

    const Eigen::SelfAdjointEigenSolver<matrix> solver(Eigen::Map<const matrix>(storage.data()));
    Eigen::Map<vector>(result.data()) = solver.eigenvectors().col(0).normalized();

    return result;
  }

  void calculate_native(const std::span<const matrix_storage> matrices, const std::span<vector_storage> vectors)
  {
    for (size_t i = 0, n = std::min(matrices.size(), vectors.size()); i < n; ++i)
      vectors[i] = calculate_eigenvector_with_largest_eigenvalue(matrices[i]);
  }

  void calculate_native_parallel(const std::span<const matrix_storage> matrices, const std::span<vector_storage> vectors)
  {
    quxflux::parallel_for_naive(std::min(matrices.size(), vectors.size()), [=](const size_t i) { vectors[i] = calculate_eigenvector_with_largest_eigenvalue(matrices[i]); });
  }

  class calculate_eigenvector;
  auto make_sycl_implementation(const sycl::device& device)
  {
    return [queue = sycl::queue{device}](const std::span<const matrix_storage> matrices, const std::span<vector_storage> vectors) mutable {
      sycl::buffer<matrix_storage> matrix_buffer(matrices);
      sycl::buffer<vector_storage> vectors_buffer(vectors);

      queue.submit([&](sycl::handler& handler) {
        const sycl::accessor device_matrices{matrix_buffer, sycl::read_only};
        const sycl::accessor device_vectors{vectors_buffer, sycl::write_only};

        handler.require(device_matrices);
        handler.require(device_vectors);
        handler.parallel_for<calculate_eigenvector>(matrix_buffer.get_range(),
                                                   [=](const sycl::item<1> idx) { device_vectors[idx] = calculate_eigenvector_with_largest_eigenvalue(device_matrices[idx]); });
      });
    };
  }

  auto make_sycl_implementations()
  {
    std::vector<std::pair<std::string, std::function<void(std::span<const matrix_storage>, std::span<vector_storage>)>>> impls;

    for (const auto& device : quxflux::get_supported_devices<calculate_eigenvector>())
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

    auto generator = [gen = std::mt19937{42}, dis = std::uniform_real_distribution{}]() mutable {
      return dis(gen);
    };

    for (auto& storage : t)
      std::ranges::generate(storage, generator);

    return t;
  }();

  std::vector<vector_storage> result(num_matrices);
  for (const auto& [name, f] : implementations)
  {
    std::cout << "benchmarking implementation: " << name << '\n';

    const auto median_duration = quxflux::benchmark([&] { f(matrices, result); }, 10,  //
                                                    [&] { std::ranges::fill(result, vector_storage{}); },
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
