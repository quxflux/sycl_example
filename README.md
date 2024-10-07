# sycl_starter project

This repo contains toy examples demonstrating the usage of SYCL.

# Notes 

This code was implemented using [Intel's DPC++ compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) and the [codeplay CUDA backend](https://developer.codeplay.com/products/oneapi/nvidia/2023.0.0/guides/get-started-guide-nvidia) running under Ubuntu 22.04.

The CUDA backend is by default disabled. Use the `SYCL_STARTER_USE_CUDA_BACKEND` CMake option to control CUDA backend usage.

# Contained projects

## bezier_curve

Calculates interpolated points using [cubic Bezier curves](https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Higher-order_curves).

## inverse_matrix

Calculates the inverse of small matrices using the [Eigen library](https://eigen.tuxfamily.org/).

## sorting_net

Sorts small arrays using a [sorting network](https://en.wikipedia.org/wiki/Sorting_network) (sorting network implementation taken from [sorting_network_cpp](https://github.com/quxflux/sorting_network_cpp)).
