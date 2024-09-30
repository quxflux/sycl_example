#pragma once

#include <vector>

#include <sycl/sycl.hpp>

namespace quxflux
{
  namespace detail
  {
    [[nodiscard]] inline std::string remove_trailing_whitespace(std::string s)
    {
      while (!s.empty() && s.back() == ' ')
        s.pop_back();

      return s;
    }
  }  // namespace detail

  [[nodiscard]] inline std::string get_name(const sycl::device& device)
  {
    return detail::remove_trailing_whitespace(device.get_platform().get_info<sycl::info::platform::name>()) + " | " + detail::remove_trailing_whitespace(device.get_info<sycl::info::device::name>());
  }

  template<typename Kernel>
  [[nodiscard]] std::vector<sycl::device> get_supported_devices()
  {
    std::vector<sycl::device> devices;

    for (const auto& platform : sycl::platform::get_platforms())
      for (const auto& device : platform.get_devices())
      {
        if (!sycl::is_compatible<Kernel>(device))
        {
          std::clog << "Device " << get_name(device) << " is not compatible with the kernel, skipping.\n";
          continue;
        }

        if (device.get_info<sycl::info::device::name>().find("Emulation Device") != std::string::npos)
        {
          // the intel dpc++ SYCL implementation comes with an FPGA emulation backend, which we don't want to use
          continue;
        }

        devices.push_back(device);
      }

    return devices;
  }
}  // namespace quxflux
