add_llvm_tool(scaleRTL-opt scaleRTL-opt.cpp)
llvm_update_compile_flags(scaleRTL-opt)
target_link_libraries(scaleRTL-opt PRIVATE 
  KSim
  KSimTransforms

  CIRCTHW
  CIRCTComb
  CIRCTSeq

  MLIRIR
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRParser
  MLIRTransforms
  MLIRSupport
  MLIROptLib
  MLIRLLVMIRTransforms
)
