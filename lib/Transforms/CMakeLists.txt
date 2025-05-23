set(LLVM_TARGET_DEFINITIONS SlicePropagation.td)
mlir_tablegen(SlicePropagation.cpp.inc -gen-rewriters)
add_public_tablegen_target(KSimSlicePropagationIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_circt_dialect_library(
  KSimTransforms

  CleanDesign.cpp
  IsolateTopModule.cpp
  AddZeroReset.cpp
  AddDebugInfo.cpp
  CombToLLVM.cpp
  Flatten.cpp
  LoadFIRRTL.cpp
  TemporalFusion.cpp
  TemporalFusion2.cpp
  RemoveSV.cpp
  LowerToLLVM.cpp
  LowerState.cpp
  SlicePropagation.cpp
  DCE.cpp
  MiliToMoore.cpp
  DumpComb.cpp
  Partition.cpp
  PartitionV2.cpp
  Batch.cpp
  BatchFlattenFunc.cpp
  CountInstance.cpp
  CountInstanceUnderTop.cpp
  ExtractInstanceByName.cpp
  GlobalToStructPass.cpp
  NVGPUGlobalToStruct.cpp

  DEPENDS
  KSimTransformsIncGen
  KSimSlicePropagationIncGen

  LINK_LIBS PUBLIC

  KSim
  KSimLLVMQueue

  CIRCTFIRRTL
  CIRCTHW
  CIRCTSeq
  CIRCTComb
  CIRCTCombToArith
  CIRCTHWToLLVM
  CIRCTCombToLLVM

  MLIRPass
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRArithToLLVM
  MLIRReconcileUnrealizedCasts

  lemon
)
