# Bayesian Code Diffusion Artifacts [MLSys 2025 Under Review]

- This repository contains an implementation of Bayesian Code Diffusion.
- The code has been tested using the Docker image `nvcr.io/nvidia/pytorch:23.04-py3`.
- It is also based on Apache TVM ([GitHub link](https://github.com/apache/tvm)).


## Introduction

![Overview of the BayesianCodeDiffusion](https://github.com/ijgit/CodeDiffusion/blob/main/overview.pdf)

- Bayesian Code Diffusion optimize deep learning programs with reduced compilation time by clustering subgraphs and diffusing reusable optimization parameters.
- The detailed implementation can be found in `tvm-codediffusion/python/tvm/auto_scheduler/task_scheduler.py` (subgraphs clustering) and
- `tvm-codediffusion/src/auto_scheduler/search_policy/sketch_policy_rules.cc` (code diffusion).

## Install

- The build files for Ansor are located in the `tvm-ansor` folder, and the build files for Bayesian Code Diffusion are in the `tvm-codediffusion` folder.
- In each folder, run the following commands:

    ```bash
    # Build Ansor
    cd tvm-ansor
    mkdir build
    cp config.cmake ./build
    cp compile.sh ./build
    cd build
    bash compile.sh

    # Build Code Diffusion
    cd tvm-codediffusion
    mkdir build
    cp config.cmake ./build
    cp compile.sh ./build
    cd build
    bash compile.sh
    ```

## Run

- We provide an example of running BayesianCodeDiffusion for SqueezeNet on CPU/GPU.
- Execute the `script.sh` file located in the `experiments` directory.

    ```
    cd experiments
    bash script.sh
    ```
