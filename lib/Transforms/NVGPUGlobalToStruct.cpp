//===- NVGPUGlobalToStructPass.cpp - Convert globals to struct fields for NVIDIA GPU ------===//
// This MLIR pass transforms global variables into fields of a context struct.
// Functions/kernels are updated to take a pointer to an array of these structs
// as an argument, where each NVIDIA GPU thread accesses its own struct.
//===-------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#define GEN_PASS_DEF_NVGPUGLOBALTOSTRUCT 
#include "PassDetails.h"

using namespace ksim;
using namespace mlir;

// Structure to store information about global variables
struct GlobalInfo {
  StringRef name;
  LLVM::GlobalOp globalOp;
  Type type;
  unsigned fieldIndex;
  uint64_t byteOffset;  // Byte offset
  bool isFrequentlyAccessed;  // For GPU memory coalescing optimization
};

struct NVGPUGlobalToStructPass : ksim::impl::NVGPUGlobalToStructBase<NVGPUGlobalToStructPass> {
  using ksim::impl::NVGPUGlobalToStructBase<NVGPUGlobalToStructPass>::NVGPUGlobalToStructBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NVVM::NVVMDialect>();    // <- This is the correct way to declare NVVM dependence
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    context->loadDialect<NVVM::NVVMDialect>();
    
    // Collect all global variables
    llvm::SmallVector<GlobalInfo, 8> globals;
    llvm::DenseMap<StringRef, unsigned> globalNameToFieldIdx;
    llvm::DenseMap<StringRef, uint64_t> globalNameToByteOffset;
    unsigned fieldIdx = 0;
    uint64_t currentOffset = 0;
    
    moduleOp.walk([&](LLVM::GlobalOp globalOp) {
      GlobalInfo info;
      info.name = globalOp.getSymName();
      info.globalOp = globalOp;
      info.type = globalOp.getType();
      info.fieldIndex = fieldIdx++;
      info.isFrequentlyAccessed = false; // Default
      
      // Mark scalars as frequently accessed for better coalescing
      if (info.type.isIntOrFloat()) {
        info.isFrequentlyAccessed = true;
      }
      
      // Calculate byte offset for this field
      info.byteOffset = currentOffset;
      
      // Calculate size of this type to update the offset for the next field
      uint64_t typeSize = 0;
      
      // Handle different types to calculate size
      if (auto arrayType = info.type.dyn_cast<LLVM::LLVMArrayType>()) {
        Type elementType = arrayType.getElementType();
        uint64_t elemSize = 0;
        
        // Calculate element size based on type
        if (elementType.isIntOrFloat()) {
          elemSize = elementType.getIntOrFloatBitWidth() / 8;
        } else {
          // Use a default size for complex types or emit a warning
          elemSize = 1;
          emitWarning(globalOp.getLoc(), "Couldn't determine exact size for array element type");
        }
        
        typeSize = elemSize * arrayType.getNumElements();
      } else if (auto intType = info.type.dyn_cast<IntegerType>()) {
        typeSize = intType.getWidth() / 8;
        if (intType.getWidth() % 8 != 0) {
          typeSize++; // Round up for non-byte-aligned types
        }
      } else if (info.type.isF32()) {
        typeSize = 4;
      } else if (info.type.isF64()) {
        typeSize = 8;
      } else {
        // Default size for unknown types
        typeSize = 8; // Pointer size on most platforms
        emitWarning(globalOp.getLoc(), "Couldn't determine exact size for type, using default");
      }
      
      // NVIDIA GPU-specific alignment - Use 16-byte alignment for better coalescing
      const uint64_t gpuAlignment = 16;
      
      // Round up the current offset to GPU alignment
      currentOffset = (currentOffset + gpuAlignment - 1) & ~(gpuAlignment - 1);
      
      // Update offset for this field
      info.byteOffset = currentOffset;
      
      // Move offset for next field
      currentOffset += typeSize;
      
      globals.push_back(info);
      globalNameToFieldIdx[globalOp.getSymName()] = info.fieldIndex;
      globalNameToByteOffset[globalOp.getSymName()] = info.byteOffset;
    });
    
    if (globals.empty())
      return;
    
    // Optimize struct layout for GPU - sort fields for coalesced access
    optimizeStructLayoutForGPU(globals, globalNameToByteOffset);
    
    // Create struct type for the context
    OpBuilder builder(context);
    builder.setInsertionPointToStart(moduleOp.getBody());
    
    // Collect field types for the struct
    llvm::SmallVector<Type, 8> fieldTypes;
    for (const auto &globalInfo : globals) {
      fieldTypes.push_back(globalInfo.type);
    }
    
    // Create the struct type
    auto namedStructType = LLVM::LLVMStructType::getIdentified(context, structName);
    namedStructType.setBody(fieldTypes, /*isPacked=*/false);
    
    // Create addressOf replacements
    llvm::DenseMap<Operation *, std::pair<Value, uint64_t>> addressOfReplacements;
    
    // Find all addressof operations and prepare replacements
    moduleOp.walk([&](LLVM::AddressOfOp addressOfOp) {
      StringRef globalName = addressOfOp.getGlobalName();
      auto it = globalNameToByteOffset.find(globalName);
      if (it != globalNameToByteOffset.end()) {
        addressOfReplacements[addressOfOp.getOperation()] = 
            std::make_pair(nullptr, it->second);
      }
    });
    
    // Find all functions to transform - assume all are kernels
    llvm::SmallVector<LLVM::LLVMFuncOp, 4> targetFunctions;
    
    moduleOp.walk([&](LLVM::LLVMFuncOp funcOp) {
      // Skip any external or declaration-only functions
      if (funcOp.isExternal() || funcOp.empty())
        return;
        
      // Check if this function uses any of the globals we captured
      bool usesGlobals = false;
      funcOp.walk([&](LLVM::AddressOfOp addressOfOp) {
        if (globalNameToByteOffset.count(addressOfOp.getGlobalName()) > 0) {
          usesGlobals = true;
        }
      });
      
      if (usesGlobals) {
        targetFunctions.push_back(funcOp);
      }
    });
    
    // If we didn't find any target functions, we're done
    if (targetFunctions.empty()) {
      emitWarning(moduleOp.getLoc(), "Could not find any functions that use globals");
      return;
    }
    
    // Store function names for header generation
    llvm::SmallVector<StringRef, 4> transformedFunctionNames;
    
    // Transform all functions as NVIDIA GPU kernels
    for (LLVM::LLVMFuncOp funcOp : targetFunctions) {
      StringRef funcName = funcOp.getSymName();
      transformedFunctionNames.push_back(funcName);
      
      // Treat all functions as GPU kernels
      transformNVGPUKernel(funcOp, globals, globalNameToByteOffset, addressOfReplacements);
    }
    
    // Erase the replaced addressof operations
    for (auto &pair : addressOfReplacements) {
      if (pair.second.first) {
        pair.first->erase();
      }
    }
    
    // Erase the global variables
    for (const auto &globalInfo : globals) {
      globalInfo.globalOp->erase();
    }
    
    // Generate the header file with the list of transformed functions
    if (!headerFile.empty()) {
      generateNVGPUHeaderFile(globals, namedStructType, transformedFunctionNames);
    }
  }

private:
  // Optimize struct layout for NVIDIA GPU memory coalescing
  void optimizeStructLayoutForGPU(
      llvm::SmallVector<GlobalInfo, 8> &globals,
      llvm::DenseMap<StringRef, uint64_t> &globalNameToByteOffset) {
    
    // Sort fields based on frequency of access (for memory coalescing)
    std::stable_sort(globals.begin(), globals.end(), 
                    [](const GlobalInfo &a, const GlobalInfo &b) {
                      // Prioritize frequently accessed fields first
                      return a.isFrequentlyAccessed > b.isFrequentlyAccessed;
                    });
    
    // Recalculate field indices and byte offsets with GPU alignment
    const uint64_t gpuAlignment = 16; // Align to 16 bytes for coalesced access
    uint64_t currentOffset = 0;
    
    for (unsigned i = 0; i < globals.size(); i++) {
      globals[i].fieldIndex = i;
      
      // Align to GPU alignment
      currentOffset = (currentOffset + gpuAlignment - 1) & ~(gpuAlignment - 1);
      globals[i].byteOffset = currentOffset;
      
      // Update the offset map
      globalNameToByteOffset[globals[i].name] = currentOffset;
      
      // Calculate size of this type
      uint64_t typeSize = 0;
      Type type = globals[i].type;
      
      if (auto arrayType = type.dyn_cast<LLVM::LLVMArrayType>()) {
        Type elementType = arrayType.getElementType();
        uint64_t elemSize = elementType.isIntOrFloat() ? 
                           (elementType.getIntOrFloatBitWidth() / 8) : 1;
        typeSize = elemSize * arrayType.getNumElements();
      } else if (auto intType = type.dyn_cast<IntegerType>()) {
        typeSize = (intType.getWidth() + 7) / 8; // Round up to bytes
      } else if (type.isF32()) {
        typeSize = 4;
      } else if (type.isF64()) {
        typeSize = 8;
      } else {
        typeSize = 8; // Default to pointer size
      }
      
      // Update current offset
      currentOffset += typeSize;
    }
  }
  
  // Detect if a function is an NVIDIA GPU kernel - always returns true
  bool detectNVGPUKernel(LLVM::LLVMFuncOp funcOp) {
    // Always assume functions are GPU kernels
    return true;
  }
  
  // Transform NVIDIA GPU kernel - assumes first arg is pointer to struct array
  void transformNVGPUKernel(
      LLVM::LLVMFuncOp kernelFunc,
      const llvm::SmallVector<GlobalInfo, 8> &globals,
      const llvm::DenseMap<StringRef, uint64_t> &globalNameToByteOffset,
      llvm::DenseMap<Operation *, std::pair<Value, uint64_t>> &addressOfReplacements) {
      
    MLIRContext *context = kernelFunc.getContext();
    OpBuilder builder(context);
    
    // Check if kernel already has arguments
    if (kernelFunc.getNumArguments() == 0) {
      // Need to add a parameter for the struct array
      auto structPtrType = LLVM::LLVMPointerType::get(builder.getI8Type());
      
      // Update function type
      auto funcType = kernelFunc.getFunctionType();
      llvm::SmallVector<Type, 4> inputTypes = {structPtrType};
      Type resultType = funcType.getReturnType();
      auto newFuncType = LLVM::LLVMFunctionType::get(resultType, inputTypes);
      kernelFunc.setFunctionType(newFuncType);
      
      // Update function body
      Block &entryBlock = kernelFunc.getBody().front();
      entryBlock.addArgument(structPtrType, kernelFunc.getLoc());
    }
    
    // Now the kernel has at least one parameter (the struct array pointer)
    Block &entryBlock = kernelFunc.getBody().front();
    Value structArrayPtr = entryBlock.getArgument(0);
    
    // Generate thread index calculation
    builder.setInsertionPointToStart(&entryBlock);
    Value threadIndex = generateNVThreadIndexCalculation(builder, kernelFunc.getLoc());
    
    // Get struct size (for striding the array)
    uint64_t structSize = 0;
    for (const auto &globalInfo : globals) {
      uint64_t fieldEnd = globalInfo.byteOffset;
      
      // Calculate size of this field
      if (auto arrayType = globalInfo.type.dyn_cast<LLVM::LLVMArrayType>()) {
        Type elementType = arrayType.getElementType();
        uint64_t elemSize = elementType.isIntOrFloat() ? 
                           (elementType.getIntOrFloatBitWidth() / 8) : 1;
        fieldEnd += elemSize * arrayType.getNumElements();
      } else if (auto intType = globalInfo.type.dyn_cast<IntegerType>()) {
        fieldEnd += (intType.getWidth() + 7) / 8;
      } else if (globalInfo.type.isF32()) {
        fieldEnd += 4;
      } else if (globalInfo.type.isF64()) {
        fieldEnd += 8;
      } else {
        fieldEnd += 8;
      }
      
      structSize = std::max(structSize, fieldEnd);
    }
    
    // Round up to alignment
    structSize = (structSize + 15) & ~15;
    
    // Calculate offset to this thread's struct
    Value structSizeVal = builder.create<LLVM::ConstantOp>(
        kernelFunc.getLoc(), 
        builder.getI64Type(),
        builder.getI64IntegerAttr(structSize));
        
    Value threadOffset = builder.create<LLVM::MulOp>(
        kernelFunc.getLoc(),
        builder.getI64Type(),
        threadIndex,
        structSizeVal);
        
    // Get pointer to this thread's struct
    Value thisThreadStructPtr = builder.create<LLVM::GEPOp>(
        kernelFunc.getLoc(),
        LLVM::LLVMPointerType::get(builder.getI8Type()),
        structArrayPtr,
        ValueRange{threadOffset});
    
    // Now replace addressof operations in this kernel
    for (auto &pair : addressOfReplacements) {
      Operation *op = pair.first;
      uint64_t byteOffset = pair.second.second;
      
      // Only process operations in this function
      if (op->getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>() != kernelFunc)
        continue;
        
      // Get the target type
      auto addressOfOp = cast<LLVM::AddressOfOp>(op);
      auto resultType = addressOfOp.getResult().getType();
      
      // Field offset from this thread's struct
      auto fieldOffsetVal = builder.create<LLVM::ConstantOp>(
          op->getLoc(), 
          builder.getI64Type(), 
          builder.getI64IntegerAttr(byteOffset));
      
      // Create GEP with byte offset from this thread's struct
      auto gepOp = builder.create<LLVM::GEPOp>(
          op->getLoc(),
          LLVM::LLVMPointerType::get(builder.getI8Type()),
          thisThreadStructPtr,
          ValueRange{fieldOffsetVal});
          
      // Cast to original pointer type if needed
      if (gepOp.getType() != resultType) {
        auto castOp = builder.create<LLVM::BitcastOp>(
            op->getLoc(), 
            resultType, 
            gepOp);
            
        op->replaceAllUsesWith(castOp);
        addressOfReplacements[op].first = castOp;
      } else {
        op->replaceAllUsesWith(gepOp);
        addressOfReplacements[op].first = gepOp;
      }
    }
  }
  
  // Generate NVIDIA thread index calculation using NVVM dialect directly
  Value generateNVThreadIndexCalculation(OpBuilder &builder, Location loc) {
    // Check if GPU dialect is available
    // if (builder.getContext()->isRegistered<gpu::GPUDialect>()) {
    //   // Use GPU dialect for thread indexing
    //   auto blockIdxX = builder.create<gpu::BlockIdOp>(loc, builder.getIndexType(), 
    //                                               builder.getStringAttr("x"));
    //   auto blockDimX = builder.create<gpu::BlockDimOp>(loc, builder.getIndexType(), 
    //                                                 builder.getStringAttr("x"));
    //   auto threadIdxX = builder.create<gpu::ThreadIdOp>(loc, builder.getIndexType(), 
    //                                                  builder.getStringAttr("x"));
      
    //   // Calculate global ID: blockIdx.x * blockDim.x + threadIdx.x
    //   auto blockOffset = builder.create<LLVM::MulOp>(loc, blockIdxX, blockDimX);
    //   auto globalId = builder.create<LLVM::AddOp>(loc, blockOffset, threadIdxX);
      
    //   // Convert to integer
    //   return builder.create<LLVM::IndexCastOp>(loc, builder.getI64Type(), globalId);
    // } else 
    {
      // Use NVVM intrinsics directly
      // Use specific NVVM operations for thread ID, block ID, and block dimensions
      auto i32Type = builder.getI32Type();
      auto i64Type = builder.getI64Type();

      // Thread, block ID, and dimensions using specific NVVM operations
      auto threadIdX = builder.create<NVVM::ThreadIdXOp>(loc, i32Type);
      auto blockIdX = builder.create<NVVM::BlockIdXOp>(loc, i32Type);
      auto blockDimX = builder.create<NVVM::BlockDimXOp>(loc, i32Type);

      // Convert to i64
      auto threadIdX64 = builder.create<LLVM::ZExtOp>(loc, i64Type, threadIdX);
      auto blockIdX64 = builder.create<LLVM::ZExtOp>(loc, i64Type, blockIdX);
      auto blockDimX64 = builder.create<LLVM::ZExtOp>(loc, i64Type, blockDimX);
      

      // Calculate global ID: blockIdx.x * blockDim.x + threadIdx.x
      auto blockOffset = builder.create<LLVM::MulOp>(loc, blockIdX64, blockDimX64);
      auto globalId = builder.create<LLVM::AddOp>(loc, blockOffset, threadIdX64);

      return globalId;
          }
  }
  
  // Generate NVIDIA GPU-specific header file
  void generateNVGPUHeaderFile(
      const llvm::SmallVector<GlobalInfo, 8> &globals, 
      LLVM::LLVMStructType structType,
      const llvm::SmallVector<StringRef, 4> &transformedFunctions) {
      
    std::error_code ec;
    llvm::raw_fd_ostream header(headerFile, ec);
    assert(!ec);

    // Use #pragma once
    header << "#pragma once\n\n";
    
    
    // Add C++ compatibility
    header << "#ifdef __cplusplus\n";
    header << "#include <cstdint>\n";
    header << "extern \"C\" {\n";
    header << "#else\n";
    header << "#include <stdint.h>\n";
    header << "#endif\n\n";
    
    // Generate documentation comment
    header << "/**\n";
    header << " * "<< structName <<" struct - Generated by MLIR NVGPUGlobalToStruct pass\n";
    header << " * Contains all global variables from the original MLIR module\n";
    header << " * Each CUDA thread will access its own instance of this struct\n";
    header << " */\n";
    
    // Generate struct with original field names
    header << "typedef struct "<< structName <<" {\n";
    
    // Sort globals by byte offset to maintain proper order in the struct
    llvm::SmallVector<GlobalInfo, 8> sortedGlobals(globals);
    std::sort(sortedGlobals.begin(), sortedGlobals.end(), 
              [](const GlobalInfo &a, const GlobalInfo &b) {
                return a.byteOffset < b.byteOffset;
              });
    
    uint64_t currentOffset = 0;
    
    // Add each field with comments
    for (const auto &global : sortedGlobals) {
      // Add padding if needed
      if (global.byteOffset > currentOffset) {
        uint64_t paddingSize = global.byteOffset - currentOffset;
        header << "    /* Padding for CUDA memory alignment */\n";
        header << "    uint8_t _padding" << currentOffset << "[" << paddingSize << "];\n";
        currentOffset += paddingSize;
      }
      
      StringRef originalName = global.name;
      std::string processedName = originalName.str();
      std::replace(processedName.begin(), processedName.end(), '.', '_');

      header << "    /* Field " << global.fieldIndex << " - Original global: @" 
         << processedName << " - Byte offset: " << global.byteOffset << " */\n";
      
      // Convert MLIR type to C type and update current offset
      if (auto arrayType = global.type.dyn_cast<LLVM::LLVMArrayType>()) {
        Type elementType = arrayType.getElementType();
        uint64_t numElements = arrayType.getNumElements();
        
        if (elementType.isInteger(8)) {
          header << "    uint8_t " << processedName << "[" << numElements << "];\n";
          currentOffset += numElements;
        } else if (elementType.isInteger(16)) {
          header << "    uint16_t " << processedName << "[" << numElements << "];\n";
          currentOffset += numElements * 2;
        } else if (elementType.isInteger(32)) {
          header << "    uint32_t " << processedName << "[" << numElements << "];\n";
          currentOffset += numElements * 4;
        } else if (elementType.isInteger(64)) {
          header << "    uint64_t " << processedName << "[" << numElements << "];\n";
          currentOffset += numElements * 8;
        } else {
          // Generic fallback
          header << "    char " << processedName << "[" << numElements << "];\n";
          currentOffset += numElements;
        }
      } else if (auto intType = global.type.dyn_cast<IntegerType>()) {
        unsigned width = intType.getWidth();
        if (width <= 8) {
          header << "    uint8_t " << processedName << ";\n";
          currentOffset += 1;
        } else if (width <= 16) {
          header << "    uint16_t " << processedName << ";\n";
          currentOffset += 2;
        } else if (width <= 32) {
          header << "    uint32_t " << processedName << ";\n";
          currentOffset += 4;
        } else if (width <= 64) {
          header << "    uint64_t " << processedName << ";\n";
          currentOffset += 8;
        }
      } else if (global.type.isa<FloatType>()) {
        // Handle float types
        if (global.type.isF32()) {
          header << "    float " << processedName << ";\n";
          currentOffset += 4;
        } else if (global.type.isF64()) {
          header << "    double " << processedName << ";\n";
          currentOffset += 8;
        }
      } else {
        // Generic fallback
        header << "    void *" << processedName << "; /* Unrecognized type */\n";
        currentOffset += 8; // Assume pointer size
      }
    }
    
    header << "} " << structName << ";\n\n";
    
    // Add byte offset constants for direct access
    header << "/* Byte offset constants for direct memory access */\n";
    for (const auto &global : globals) {
      StringRef originalName = global.name;
      std::string processedName = originalName.str();
      std::replace(processedName.begin(), processedName.end(), '.', '_');
      header << "#define " << StringRef(processedName).upper() << "_BYTE_OFFSET " << global.byteOffset << "\n";
    }

    header << "\n";
    if (!transformedFunctions.empty()) {
      StringRef mainFunc = transformedFunctions[0];
      header << "#define EvalFunc " << "\"" << mainFunc << "\"" << "\n";
    }
    
    // Close the C++ extern "C" block
    header << "#ifdef __cplusplus\n";
    header << "}\n";
    header << "#endif\n";
  }
};

// Factory function to create an instance of the pass
std::unique_ptr<mlir::Pass> ksim::createNVGPUGlobalToStructPass(NVGPUGlobalToStructOptions options) {
  return std::make_unique<NVGPUGlobalToStructPass>(options);
}