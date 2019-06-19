# RobustPCLReconstruction

## Demo video for an application in UAV-UGV collabrative SLAM
[![Watch the video](https://github.com/ziquan111/RobustPCLReconstruction/raw/master/figure_coslam_demo.png)](https://www.youtube.com/watch?v=ZZQT_REkItU)

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

## Common problems
**1** . For Ubuntu 16.04 LTS, the default version of Eigen is 3.2.9. 
[Click to download Eigen 3.3.7](http://bitbucket.org/eigen/eigen/get/3.3.7.zip) or download other Eigen 3.3.*, then (re)-compile ceres with the correct Eigen.

**2** . **ccache** may not be installed by default. Simply install it.
```sh
$ sudo apt-get install ccache
```

## About current release
The solver based on the Gaussian-Uniform mixture model is released.
Please stay tuned for the solver based on the Cauchy-Uniform mixture model.

