# RobustPCLReconstruction

## Overview

This repository contains the solvers based on our paper [Robust Point Cloud Based Reconstruction of Large-Scale Outdoor Scenes](https://arxiv.org/abs/1905.09634) in CVPR2019.

This is a c++ implementation based on [strasdat/Sophus](https://github.com/strasdat/Sophus). The solvers' cpp files are in the **test/ceres/** directory.

## Dependency

  - [Eigen 3.3.0](http://eigen.tuxfamily.org/index.php?title=Main_Page)
  - [Google ceres](http://ceres-solver.org/)

## Quickstart
Assume the current directory is the root of this repository.

> Compile
```sh
$ chmod +x ./scripts/run_cpp_tests.sh
$ ./scripts/run_cpp_tests.sh
```

> Run
```sh
$ chmod +x ./scripts/run_robust_pcl_reconstruction_example.sh
$ ./scripts/run_robust_pcl_reconstruction_example.sh
```

## About current release
The solver based on the Gaussian-Uniform mixture model is released.
Please stay tuned for the solver based on the Cauchy-Uniform mixture model.
