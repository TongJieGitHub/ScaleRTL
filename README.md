# ScaleRTL


## Introductions

ScaleRTL is a cycle-accurate software RTL simulation tool designed for deep learning accelerators. It provides a scalable and unified code generation flow that automatically produces optimized, parallel RTL simulation code. ScaleRTL leverages structural parallelism in accelerator designs to eliminate redundant code and generate simulators targeting both CPU and GPU backends using MLIR. By integrating CUDA Graph, it reduces kernel launch overhead, further accelerating GPU-based RTL simulation.

## Installation

Setup depedencies:

```bash
mkdir install
export INSTALL_PREFIX=$PWD/install
git submodule update --init --recursive
cd third_party
./setup-circt.sh
./setup-lemon.sh
./setup-kahypar.sh
cd ..
```

Build ScaleRTL:

```bash
mkdir build && cd build
cmake -G Ninja .. \
    -DCMAKE_INSTALL_PREFIX=~/ScaleRTL/install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ 
ninja scaleRTL scaleRTL-opt
ninja install
cd ..
```

Build other tools:

```bash
cd third_party/tools
make -j$(nproc) prepare
make -j$(nproc) setup
cd ../..
```

## Usage

```bash
# Environment setup:
. ./env.sh

# Build the simulation executable:
cd runs
make -j$(nproc) build
cd ..

# Run the simulation executable:
python3 benchmark.py
```

## Citation

If you use ScaleRTL in your work, please cite our paper:

```bibtex
@inproceedings{tong2025scalrtl,
  author    = {Tong, Jie and Lee, Wan-Luan and Ogras, Umit Yusuf and Huang, Tsung-Wei},
  title     = {Scalable Code Generation for RTL Simulation of Deep Learning Accelerators with MLIR},
  booktitle = {Proceedings of the International European Conference on Parallel and Distributed Computing (Euro-Par)},
  year      = {2025}
}
```

We also gratefully acknowledge that ScaleRTL builds upon prior work, Khronos:

```bibtex
@inproceedings{Khronos,
  author = {Kexing, Zhou and Yun, Liang and Yibo, Lin and Runsheng, Wang and Ru, Huang},
  title = {Khronos: Fusing Memory Access for Improved Hardware RTL Simulation},
  booktitle = {MICRO '23: 55th IEEE/ACM International Symposium on Microarchitecture},
  publisher = {ACM},
  year = {2023},
  url = {https://doi.org/10.1145/3613424.3614301},
  doi = {10.1145/3613424.3614301}
}
```
