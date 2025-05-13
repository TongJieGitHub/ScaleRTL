#ifndef KSIM_PASSES_TD
#define KSIM_PASSES_TD

include "mlir/Pass/PassBase.td"

def RemoveSV : Pass<"ksim-remove-sv", "circt::ModuleOp"> {
  let summary = "Remove SV Dialect";
  let description = summary;
  let constructor = "ksim::createRemoveSVPass()";
}

def IsolateTopModule : Pass<"ksim-isolate-top-module"> {
  let summary = "Isolate top module";
  let description = summary;
  let constructor = "ksim::createIsolateTopModulePass()";
  let options = [
    Option<"topModuleName", "top", "std::string", "", "The top module name">
  ];
}

def SlicePropagation : Pass<"ksim-slice-propagation", "circt::hw::HWModuleOp"> {
  let summary = "Slice propagation pass";
  let description = [{
    for operations like "slice(f(x))", if f is element wise operation, rewrite it to "f(slice(x))"
  }];
  let constructor = "ksim::createSlicePropagationPass()";
}

def AddZeroReset : Pass<"ksim-add-reset", "circt::hw::HWModuleOp"> {
  let summary = "Add zero reset to registers without reset value";
  let description = summary;
  let constructor = "ksim::createAddZeroResetPass()";
}

def Flatten : Pass<"ksim-flatten", "circt::ModuleOp"> {
  let summary = "Flatten all module";
  let description = "Flatten all module into top module.";
  let constructor = "ksim::createFlattenPass()";
  let statistics = [
    Statistic<"numEdges", "num-edges", "number of edges after flatten">,
    Statistic<"numOps", "num-ops", "number of ops after flatten">
  ];
}

def DCE : Pass<"ksim-dce", "circt::hw::HWModuleOp"> {
  let summary = "dead code elimination";
  let description = summary;
  let constructor = "ksim::createDCEPass()";
}

def DumpComb : Pass<"ksim-dump-comb"> {
  let summary = "dump combnational logic graph to csv/dot";
  let description = summary;
  let constructor = "ksim::createDumpCombPass()";
  let options = [
    Option<"file", "file", "std::string", "", "dump file">
  ];
}

def MiliToMoore : Pass<"ksim-mili-to-moore", "circt::hw::HWModuleOp"> {
  let summary = "Convert small mili state machine to moore state machine";
  let description = [{
    Convert state machine like: [s', out] = f(s, in) into 
    $$
    \begin{aligned}
      s' = f_{s}(s, in)
      out = f_{out}(s, in)
    \end{aligned}
    $$
  }];
  let constructor = "ksim::createMiliToMoorePass()";
}

def LoadFIRRTL : Pass<"ksim-load-firrtl", "circt::ModuleOp"> {
  let summary = "Load firrtl object to ksim ir";
  let description = [{
    This pass will load firrtl compiler generated module,
    user defined FIRRTL, etc.
  }];
  let constructor = "ksim::createLoadFIRRTLPass()";
}

def CleanDesign : Pass<"ksim-clean-design", "circt::ModuleOp"> {
  let summary = "Clean chipyard design";
  let description = [{
    This pass will remove plusarg reader, replace EICG_wrapper into compreg.ce
  }];
  let constructor = "ksim::createCleanDesignPass()";
}

def TemporalFusion : Pass<"ksim-temporal-fusion", "circt::hw::HWModuleOp"> {
  let summary = "Temporal Fusion Algorithm for KSim";
  let description = summary;
  let constructor = "ksim::createTemporalFusionPass()";
  let options = [
    Option<"tol", "tol", "int", "10", "Maximum iteration to solve the programming problem">,
    Option<"verbose", "verbose", "bool", "false", "Verbose optimization log">,
    Option<"graphOut", "graph-out", "std::string", "", "output graph dot visualization">,
    Option<"disableOptimization", "disable-optimization", "bool", "false", "disable optimization">,
    Option<"disableClockGate", "disable-clock-gate", "bool", "false", "disable clock gate optimization">
  ];
}

def TemporalFusion2 : Pass<"ksim-temporal-fusion2", "circt::hw::HWModuleOp"> {
  let summary = "Temporal Fusion Algorithm for KSim";
  let description = summary;
  let constructor = "ksim::createTemporalFusion2Pass()";
  let options = [
    Option<"tol", "tol", "int", "10", "Maximum iteration to solve the programming problem">,
    Option<"verbose", "verbose", "bool", "false", "Verbose optimization log">,
    Option<"graphOut", "graph-out", "std::string", "", "output graph dot visualization">,
    Option<"disableOptimization", "disable-optimization", "bool", "false", "disable optimization">,
    Option<"disableClockGate", "disable-clock-gate", "bool", "false", "disable clock gate optimization">
  ];
}

def LowerState : Pass<"ksim-lower-state", "circt::ModuleOp"> {
  let summary = "Lower state component to ksim.low.queue op";
  let description = summary;
  let constructor = "ksim::createLowerStatePass()";
  let options = [
    Option<"headerFile", "header", "std::string", "", "output header file">,
    Option<"driverFile", "driver", "std::string", "", "output driver file">,
    Option<"emitComb", "emitComb", "bool", "false", "emit combinational logic">,
    Option<"prefix", "prefix", "std::string", "", "variable prefix">,
    Option<"disableOptimization", "disable-optimization", "bool", "false", "disable optimization">
  ];
  let statistics = [
    Statistic<"fusedClockGate", "fused-clock-gate", "number of fused clock gate">,
    Statistic<"allRegs", "all-regs", "number of all registers">
  ];
}


def Batch : Pass<"ksim-batch", "circt::ModuleOp"> {
  let summary = "Convert single cycle to multiple cycles in one iteration for KSim";
  let description = summary;
  let constructor = "ksim::createBatchPass()";
  let options = [
    Option<"batchSize", "batchSize", "int", "2", "batch size">,
  ];
}

def Partition : Pass<"ksim-partition", "circt::ModuleOp"> {
  let summary = "Partition for parallel simulation";
  let description = summary;
  let constructor = "ksim::createPartitionPass()";
  let options = [
    Option<"components", "k", "int", "4", "partition components">,
    Option<"kahypar", "kahypar", "std::string", "\"KaHyPar\"", "kahypar executable">,
    Option<"driverFileOpenMP", "driverFileOpenMP", "std::string", "\"\"", "emit simulation OpenMP code">,
    Option<"driverFileTaskflowAsync", "driverFileTaskflowAsync", "std::string", "\"\"", "emit simulation Taskflow async code">,
    Option<"driverFileTaskflowCond", "driverFileTaskflowCond", "std::string", "\"\"", "emit simulation Taskflow cond code">,
  ];
}

def PartitionV2 : Pass<"ksim-partitionV2", "circt::ModuleOp"> {
  let summary = "Partition for parallel simulation";
  let description = summary;
  let constructor = "ksim::createPartitionV2Pass()";
  let options = [
    Option<"components", "k", "int", "4", "partition components">,
    Option<"kahypar", "kahypar", "std::string", "\"KaHyPar\"", "kahypar executable">,
    Option<"driverFileOpenMP", "driverFileOpenMP", "std::string", "\"\"", "emit simulation OpenMP code">,
    Option<"driverFileTaskflowAsync", "driverFileTaskflowAsync", "std::string", "\"\"", "emit simulation Taskflow async code">,
    Option<"driverFileTaskflowCond", "driverFileTaskflowCond", "std::string", "\"\"", "emit simulation Taskflow cond code">,
  ];
}

def BatchFlattenFunc : Pass<"ksim-batch-flatten-func", "circt::ModuleOp"> {
  let summary = "Flatten all func into top func";
  let description = summary;
  let constructor = "ksim::createBatchFlattenFuncPass()";
}

def LowerToLLVM : Pass<"ksim-lower-to-llvm", "circt::ModuleOp"> {
  let summary = "Lower sequential logic to llvm dialect";
  let description = summary;
  let constructor = "ksim::createLowerToLLVMPass()";
  let options = [
    Option<"disableOptimization", "disable-optimization", "bool", "false", "disable optimization">
  ];
}

def CombToLLVM : Pass<"ksim-comb-to-llvm", "circt::ModuleOp"> {
  let summary = "Lower combinational logic to llvm dialect without changing module";
  let description = summary;
  let constructor = "ksim::createCombToLLVMPass()";
}

def AddLLVMDebugInfo : Pass<"ksim-add-debug-info", "mlir::LLVM::LLVMFuncOp"> {
  let summary = "Add LLVM debug info";
  let description = summary;
  let constructor = "ksim::createAddLLVMDebugInfoPass()";
  let options = [];
}

def CountInstance : Pass<"ksim-count-instance", "circt::ModuleOp"> {
  let summary = "Count instance and print a DOT graph of the module hierarchy.";
  let constructor =  "ksim::createCountInstancePass()";
}

def CountInstanceUnderTop : Pass<"ksim-count-instance-under-top", "circt::ModuleOp"> {
  let summary = "Count instance under the top module";
  let constructor =  "ksim::createCountInstanceUnderTopPass()";
}

def ExtractInstanceByName : Pass<"ksim-extract-instance-by-name", "circt::ModuleOp"> {
  let summary = "Extract instance by the module name";
  let constructor =  "ksim::createExtractInstanceByNamePass()";
  let options = [
    Option<"moduleName", "moduleName", "std::string", "\"\"", "module name">,
  ];
}

def GlobalToStruct : Pass<"ksim-global-to-struct", "mlir::ModuleOp"> {
  let summary = "Convert globals to struct in LLVM dialect.";
  let constructor =  "ksim::createGlobalToStructPass()";
  let options = [
    Option<"headerFile", "header", "std::string", "", "output header file">
  ];
}

def NVGPUGlobalToStruct : Pass<"ksim-global-to-struct-nvgpu", "mlir::ModuleOp"> {
  let summary = "Convert globals to struct in LLVM dialect for NVGPU.";
  let constructor =  "ksim::createNVGPUGlobalToStructPass()";
  let options = [
    Option<"headerFile", "header", "std::string", "", "output header file">,
    Option<"structName", "structName", "std::string", "\"EvalContext\"", "struct name">
  ];
}

#endif
