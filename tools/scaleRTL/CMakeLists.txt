add_llvm_tool(scaleRTL scaleRTL.cpp)
llvm_update_compile_flags(scaleRTL)
target_link_libraries(scaleRTL PRIVATE 
  KSim
  KSimTransforms

  CIRCTHW
  CIRCTComb
  CIRCTSeq

  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRTransforms
  MLIRSupport
  MLIRBytecodeWriter
  MLIRBytecodeReader
  MLIRTargetLLVMIRExport
  MLIRLLVMIRTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)
