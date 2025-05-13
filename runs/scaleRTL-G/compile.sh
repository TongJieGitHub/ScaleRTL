#!/bin/bash

design=$1
. ../common.sh
prepare_target_dir $design

# 1. run firtool to translate firrtl into circt HW dialect
timeit firtool --ir-hw --disable-all-randomization $design.fir -o $design.mlir 2>&1 | tee compile.log

# 2. analyze and count instance under top module
scaleRTL-opt $design.mlir --ksim-count-instance -o $design-ExtractedModule.mlir &> modules.txt

# 3. 
while IFS=, read -r moduleName moduleNumber; do
    timeit scaleRTL-opt $design.mlir --ksim-extract-instance-by-name="moduleName=${moduleName}" -o "${design}-ExtractedModule_${moduleName}.mlir" 2>&1 | tee -a compile.log
    timeit scaleRTL $design-ExtractedModule_${moduleName}.mlir -o $design-ExtractedModule_${moduleName}-llvm.mlir --tol=0 --batchSize=1 --parallel 0 --out=llvm 2>&1 | tee -a compile.log
    timeit scaleRTL-opt $design-ExtractedModule_${moduleName}-llvm.mlir -ksim-global-to-struct-nvgpu="header=EvalContext_${moduleName}.h structName=EvalContext_${moduleName}" -o $design-ExtractedModule_${moduleName}-struct.mlir 2>&1 | tee -a compile.log
    timeit mlir-opt $design-ExtractedModule_${moduleName}-struct.mlir --convert-scf-to-cf --convert-arith-to-llvm --convert-func-to-llvm --convert-index-to-llvm -o kernel_${moduleName}.llvm.mlir 2>&1 | tee -a compile.log
    timeit mlir-translate --mlir-to-llvmir kernel_${moduleName}.llvm.mlir -o kernel_${moduleName}.ll 2>&1 | tee -a compile.log
    timeit llc kernel_${moduleName}.ll -march=nvptx64 -mcpu=sm_86 -o kernel_${moduleName}.ptx 2>&1 | tee -a compile.log
    sed -i 's/.visible .func/.visible .entry/' kernel_${moduleName}.ptx 
    timeit ptxas -arch=sm_86 -o kernel_${moduleName}.cubin kernel_${moduleName}.ptx 2>&1 | tee -a compile.log
    timeit fatbinary --create=kernel_${moduleName}.fatbin --image=profile=sm_86,file=kernel_${moduleName}.cubin --image=profile=compute_86,file=kernel_${moduleName}.ptx 2>&1 | tee -a compile.log
    python3 $BASE_DIR/convert_fatbin.py kernel_${moduleName}.fatbin kernel_${moduleName}_fatbin.h ${moduleName}
done < modules.txt

# 4.
python3 $BASE_DIR/generate_benchmark_graph_v2.py 

# 5.
timeit nvcc -lcuda -o $design benchmark_graph_v2.cu 2>&1 | tee -a compile.log

provide_exe_file $design $design

