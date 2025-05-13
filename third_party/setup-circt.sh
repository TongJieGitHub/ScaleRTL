#!/bin/bash

if [[ "$INSTALL_PREFIX" == "" ]]; then
  INSTALL_PREFIX="~/ScaleRTL/install"
  [ -d $INSTALL_PREFIX ] || mkdir $INSTALL_PREFIX
fi

cd circt
git apply ../patch-circt.patch
pushd llvm

[ -d build ] || mkdir build
cd build
cmake -G Ninja ../llvm \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++
ninja install

popd

[ -d build ] || mkdir build
cd build
cmake  -G Ninja .. \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
  -DESI_COSIM=OFF -DESI_CAPN=OFF -DCAPNP_DISABLE=ON \
  -DVERILATOR_DISABLE=ON \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++
ninja install