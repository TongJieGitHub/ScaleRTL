#!/bin/bash

design=$1
. ../common.sh
prepare_target_dir $design

# 1. run firtool to translate firrtl into circt HW dialect
timeit firtool --ir-hw --disable-all-randomization $design.fir -o $design.mlir 2>&1 | tee compile.log

# 2. analyze, count and extract instance under top module
output=$(timeit scaleRTL-opt $design.mlir --ksim-count-instance-under-top -o $design-ExtractedModule.mlir 2>&1 | tee -a compile.log)
num_instances=$(echo "$output" | grep "Total direct instances under top module:" | sed 's/Total direct instances under top module: \([0-9]*\)/\1/')
echo "Extracted $num_instances instances" | tee -a compile.log
# Verify we got a valid number
if [[ -z "$num_instances" || ! "$num_instances" =~ ^[0-9]+$ ]]; then
    echo "Error: Could not extract valid instance count" | tee -a compile.log
    exit 1
fi
# timeit scaleRTL-opt $design.mlir --ksim-count-instance-under-top -o $design-ExtractedModule.mlir 2>&1 | tee -a compile.log

# 3. generate simulation code
timeit scaleRTL $design-ExtractedModule.mlir -o $design-ExtractedModule-llvm.mlir --tol=0 --batchSize=1 --parallel 0 --out=llvm 2>&1 | tee -a compile.log

# 4. convert global variables to struct
timeit scaleRTL-opt $design-ExtractedModule-llvm.mlir -ksim-global-to-struct="header=EvalContext.h" -o $design-ExtractedModule-struct.mlir 2>&1 | tee -a compile.log

# 5.llvm dialect to llvmir
timeit mlir-translate $design-ExtractedModule-struct.mlir --mlir-to-llvmir -o $design-ExtractedModule-struct.ll 2>&1 | tee -a compile.log

# 6. use llc to compile llvm code into `obj` file
timeit llc --relocation-model=dynamic-no-pic -O2 -filetype=obj $design-ExtractedModule-struct.ll -o transformed.o 2>&1 | tee -a compile.log

# 7. generate benchmark c++ file
python $BASE_DIR/benchmark_generator.py -t 4 -c $num_instances

# 8. use clang to compile testbench and link `obj` file.
timeit clang++ -std=c++17 -O2 -fopenmp benchmark.cpp transformed.o -o $design | tee -a compile.log

provide_exe_file $design $design
