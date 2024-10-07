# sycl_starter project

This repo contains toy examples demonstrating the usage of SYCL.

# Notes

This code was implemented
using [Intel's DPC++ compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
and
the [codeplay CUDA backend](https://developer.codeplay.com/products/oneapi/nvidia/2023.0.0/guides/get-started-guide-nvidia)
running under Ubuntu 22.04.

The CUDA backend is by default disabled. Use the `SYCL_STARTER_USE_CUDA_BACKEND` CMake option to control CUDA backend
usage.

# Contained projects

## bezier_curve

Calculates interpolated points
using [cubic Bezier curves](https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Higher-order_curves). This calculation could
be part of a visualization or point cloud processing pipeline.

## eigenvector

Calculates the eigenvectors associated with the largest eigenvalue of 3x3 symmetric matrices using
the [Eigen library](https://eigen.tuxfamily.org/).
This calculation is part of a processing pipeline
when [estimating normals for (3d) point clouds](https://pointclouds.org/documentation/tutorials/normal_estimation.html).

## sorting_net

Sorts small arrays using a [sorting network](https://en.wikipedia.org/wiki/Sorting_network) (sorting network
implementation taken from [sorting_network_cpp](https://github.com/quxflux/sorting_network_cpp)).
Sorting many small arrays can be used for the implementation of
a [median filter](https://en.wikipedia.org/wiki/Median_filter), when size of the arrays (i.e., the filter radius) is
small.