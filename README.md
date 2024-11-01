# Bayesian Code Diffusion Artifacts [MLSys 2025 Under Review]

- This code has been tested on the Docker image `nvcr.io/nvidia/pytorch:23.04-py3`.
- It is also based on Apache TVM ([GitHub link](https://github.com/apache/tvm)).

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

- Execute the `script.sh` file located in the `experiments` directory.

```
cd experiments
bash script.sh
```