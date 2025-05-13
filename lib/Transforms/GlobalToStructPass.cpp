//===- GlobalToStructPass.cpp - Convert globals to struct fields ------===//
// This MLIR pass transforms global variables into fields of a context struct.
// Functions accessing globals are updated to take a pointer to the struct
// as an argument and access fields via GEP instead of global loads/stores.
//===------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#define GEN_PASS_DEF_GLOBALTOSTRUCT 
#include "PassDetails.h"

using namespace ksim;
using namespace mlir;

// namespace {

// Structure to store information about global variables
struct GlobalInfo {
  StringRef name;
  LLVM::GlobalOp globalOp;
  Type type;
  unsigned fieldIndex;
  uint64_t byteOffset;  // Added byte offset
};

struct GlobalToStructPass : ksim::impl::GlobalToStructBase<GlobalToStructPass> {
  using ksim::impl::GlobalToStructBase<GlobalToStructPass>::GlobalToStructBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    
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
      
      // Account for alignment
      // For simplicity, we align everything to their natural size (up to 8 bytes)
      uint64_t alignment = std::min(typeSize, (uint64_t)8);
      if (alignment > 0) {
        // Round up the current offset to the next multiple of alignment
        currentOffset = (currentOffset + alignment - 1) & ~(alignment - 1);
      }
      
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
    
    // Create struct type for the context
    OpBuilder builder(context);
    builder.setInsertionPointToStart(moduleOp.getBody());
    
    // Collect field types for the struct
    llvm::SmallVector<Type, 8> fieldTypes;
    for (const auto &globalInfo : globals) {
      fieldTypes.push_back(globalInfo.type);
    }
    
    // Create the struct type
    auto namedStructType = LLVM::LLVMStructType::getIdentified(context, "EvalContext");
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
    
    // Find the target functions to transform
    llvm::SmallVector<LLVM::LLVMFuncOp, 4> targetFunctions;
    
    moduleOp.walk([&](LLVM::LLVMFuncOp funcOp) {
      // Skip any external or declaration-only functions
      if (funcOp.isExternal() || funcOp.empty())
        return;
        
      // Skip functions that are already using the context parameter
      if (funcOp.getNumArguments() > 0)
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
    
    // Transform each target function
    for (LLVM::LLVMFuncOp funcOp : targetFunctions) {
      StringRef funcName = funcOp.getSymName();
      transformedFunctionNames.push_back(funcName);
      
      // Update function signature to include context parameter
      // Use i8* (void*) as the context parameter type for consistent byte addressing
      auto contextPtrType = LLVM::LLVMPointerType::get(builder.getI8Type());
      
      // Create new function type with context parameter
      auto funcType = funcOp.getFunctionType();
      llvm::SmallVector<Type, 4> inputTypes = {contextPtrType};
      
      // Preserve any existing parameters (although in your case there likely aren't any)
      for (unsigned i = 0; i < funcType.getNumParams(); i++) {
        inputTypes.push_back(funcType.getParamType(i));
      }
      
      Type resultType = funcType.getReturnType();
      
      auto newFuncType = LLVM::LLVMFunctionType::get(resultType, inputTypes);
      funcOp.setFunctionType(newFuncType);
      
      // Update function body
      Block &entryBlock = funcOp.getBody().front();
      
      // Add context parameter to entry block
      entryBlock.addArgument(contextPtrType, funcOp.getLoc());
      
      // Shift existing arguments if any
      if (funcType.getNumParams() > 0) {
        for (unsigned i = 0; i < funcType.getNumParams(); i++) {
          // The new arguments will be after the context parameter
          auto oldArg = entryBlock.getArgument(i + 1);
          auto newArg = entryBlock.getArgument(i + 1);
          oldArg.replaceAllUsesWith(newArg);
        }
      }
      
      Value contextPtr = entryBlock.getArgument(0);
      
      // Replace addressof operations
      builder.setInsertionPointToStart(&entryBlock);
      
      for (auto &pair : addressOfReplacements) {
        Operation *op = pair.first;
        uint64_t byteOffset = pair.second.second;
        
        // Only process operations in this function
        if (op->getParentRegion()->getParentOfType<LLVM::LLVMFuncOp>() != funcOp)
          continue;
        
        // Get the target type
        auto addressOfOp = cast<LLVM::AddressOfOp>(op);
        auto resultType = addressOfOp.getResult().getType();
        
        // Create a GEP with byte offset (using i8* base pointer)
        auto i8PtrType = LLVM::LLVMPointerType::get(builder.getI8Type());
        auto offsetVal = builder.create<LLVM::ConstantOp>(
            op->getLoc(), 
            builder.getI64Type(), 
            builder.getI64IntegerAttr(byteOffset));
        
        // Create the GEP using byte addressing
        auto gepOp = builder.create<LLVM::GEPOp>(
            op->getLoc(),
            i8PtrType,
            contextPtr,
            ValueRange{offsetVal});
            
        // Cast to the original pointer type if needed
        if (i8PtrType != resultType) {
          auto castOp = builder.create<LLVM::BitcastOp>(
              op->getLoc(), 
              resultType, 
              gepOp);
          
          // Replace all uses of the addressof with the cast
          op->replaceAllUsesWith(castOp);
          addressOfReplacements[op].first = castOp;
        } else {
          // Replace all uses of the addressof with the GEP
          op->replaceAllUsesWith(gepOp);
          addressOfReplacements[op].first = gepOp;
        }
      }
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
      generateHeaderFile(globals, namedStructType, transformedFunctionNames);
    }
  }

private:
  void generateHeaderFile(
      const llvm::SmallVector<GlobalInfo, 8> &globals, 
      LLVM::LLVMStructType structType,
      const llvm::SmallVector<StringRef, 4> &transformedFunctions) {
      
    std::error_code ec;
    llvm::raw_fd_ostream header(headerFile, ec);
    assert(!ec);

    // Use #pragma once instead of traditional header guards
    header << "#pragma once\n\n";
    
    // Add C++ compatibility
    header << "#ifdef __cplusplus\n";
    header << "#include<cstdint>\n";
    header << "extern \"C\" {\n";
    header << "#else\n";
    header << "#include <stdint.h>\n";
    header << "#endif\n\n";
    
    // Generate documentation comment
    header << "/**\n";
    header << " * EvalContext struct - Generated by MLIR GlobalToStruct pass\n";
    header << " * Contains all global variables from the original MLIR module\n";
    header << " */\n";
    
    // Generate typedefed struct with original field names
    header << "typedef struct EvalContext {\n";
    
    // Sort globals by byte offset to maintain proper order in the struct
    llvm::SmallVector<GlobalInfo, 8> sortedGlobals(globals.begin(), globals.end());
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
        header << "    /* Padding to align field */\n";
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
    
    header << "} EvalContext;\n\n";
    
    // Add byte offset constants for direct access
    header << "/* Byte offset constants for direct memory access */\n";
    for (const auto &global : globals) {
      StringRef originalName = global.name;
      std::string processedName = originalName.str();
      std::replace(processedName.begin(), processedName.end(), '.', '_');
      header << "#define " << StringRef(processedName).upper() << "_BYTE_OFFSET " << global.byteOffset << "\n";
    }
    header << "\n";
    
    // Generate function declarations for all transformed functions
    if (!transformedFunctions.empty()) {
      header << "/* Transformed function declarations */\n";
      for (StringRef funcName : transformedFunctions) {
        header << "/**\n";
        header << " * " << funcName << " - Generated from MLIR function\n";
        header << " * \n";
        header << " * This function was originally defined without parameters, accessing globals directly.\n";
        header << " * It has been transformed to take a pointer to the EvalContext struct that contains\n";
        header << " * all the required global variables.\n";
        header << " *\n";
        header << " * @param ctx Pointer to the EvalContext struct\n";
        header << " */\n";
        header << "void " << funcName << "(EvalContext* ctx);\n\n";
      }
      
      // Add macros for the first function as the entry point
      if (!transformedFunctions.empty()) {
        StringRef mainFunc = transformedFunctions[0];
        
        header << "/* Entry function constants */\n";
        header << "#define " << mainFunc << "_output_ahead 0\n";
        header << "#define " << mainFunc << "_reset_ahead 0\n\n";
        
        header << "#ifndef PARALLEL\n";
        header << "/* Static context for simplified API */\n";
        header << "static EvalContext __eval_context;\n\n";
        header << "#define InitFunc() memset(&__eval_context, 0, sizeof(EvalContext))\n";
        header << "#define EvalFunc " << mainFunc << "\n";
        header << "#define StopFunc()\n";
        header << "#endif\n\n";
      }
    }
    
    // Close the C++ extern "C" block
    header << "#ifdef __cplusplus\n";
    header << "}\n";
    header << "#endif\n";
  }
};

std::unique_ptr<mlir::Pass> ksim::createGlobalToStructPass(GlobalToStructOptions options) {
  return std::make_unique<GlobalToStructPass>(options);
}