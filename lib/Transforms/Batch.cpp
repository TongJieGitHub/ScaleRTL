#include "ksim/KSimDialect.h"
#include "ksim/KSimOps.h"
#include "ksim/Utils/RegInfo.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Support/Namespace.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <system_error>
#include <utility>

#define GEN_PASS_DEF_BATCH
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm; 

struct BatchPass : ksim::impl::BatchBase<BatchPass> {
  using ksim::impl::BatchBase<BatchPass>::BatchBase;
  void runOnOperation() final {
    // errs() << "\nbatch size = " << batchSize << "\n";
    auto modlist = getOperation();
    auto builder = OpBuilder(modlist->getContext());
    builder.setInsertionPointToStart(modlist.getBody());
    auto block = modlist.getBody();
    SmallVector<Operation*> toCloneOps;
    SmallVector<Operation*> toDeleteOps;
    llvm::DenseSet<llvm::StringRef> nameSet;
    // duplicate DefQueueOp and DefMemOp, add suffix "_t"
    // skip clock, reset and private
    for (int batchCount = 0; batchCount < batchSize; ++batchCount) {
      for(auto & op: *block) {
        if (auto defQueueOp = dyn_cast<ksim::DefQueueOp>(op)) {
          if (defQueueOp.isPrivate() || defQueueOp.getSymName() == "reset" || defQueueOp.getSymName() == "clock" || defQueueOp.getSymName().contains("valid")) {
            continue;
          }
          nameSet.insert(defQueueOp.getSymName());
          auto clone = defQueueOp.clone();
          clone.setSymName((defQueueOp.getSymName() + "_t" + std::to_string(batchCount)).str());
          toCloneOps.push_back(clone);
        } else if (auto defMemOp = dyn_cast<ksim::DefMemOp>(op)) {
          if (defMemOp.isPrivate()) {
            continue;
          }
          nameSet.insert(defMemOp.getSymName());
          auto clone = defMemOp.clone();
          clone.setSymName((defMemOp.getSymName() + "_t" + std::to_string(batchCount)).str());
          toCloneOps.push_back(clone);
        }
      }
    }
    for(auto & op: *block) {
        if (auto defQueueOp = dyn_cast<ksim::DefQueueOp>(op)) {
          if (defQueueOp.isPrivate() || defQueueOp.getSymName() == "reset" || defQueueOp.getSymName() == "clock" || defQueueOp.getSymName().contains("valid")) {
            continue;
          }
          nameSet.insert(defQueueOp.getSymName());
          toDeleteOps.push_back(defQueueOp);
        } else if (auto defMemOp = dyn_cast<ksim::DefMemOp>(op)) {
          if (defMemOp.isPrivate()) {
            continue;
          }
          nameSet.insert(defMemOp.getSymName());
          toDeleteOps.push_back(defMemOp);
        }
    }
    for(auto op: toCloneOps) {
      builder.insert(op);
    }
    for(auto op: toDeleteOps) {
      op->erase();
    }

    // duplicate func.func
    toCloneOps.clear();
    toDeleteOps.clear();
    auto funcOps = to_vector(modlist.getOps<func::FuncOp>());
    for (auto funcOp : funcOps) {
      builder = OpBuilder(funcOp);
      for (int batchCount = 0; batchCount < batchSize; ++batchCount) {
        IRMapping mapping;
        auto clone = funcOp.clone(mapping);
        clone.setSymName((funcOp.getSymName() + "_t" + std::to_string(batchCount)).str());
        clone->walk([&](Operation * op) {
            if (auto getQueueOp = dyn_cast<ksim::GetQueueOp>(op)) {
              if (nameSet.count(getQueueOp.getQueue()))
                getQueueOp.setQueue(getQueueOp.getQueue().str() + "_t" + std::to_string(batchCount));
            }
            else if (auto pushQueueOp = dyn_cast<ksim::PushQueueOp>(op)) {
              if (nameSet.count(pushQueueOp.getQueue()))
                pushQueueOp.setQueue(pushQueueOp.getQueue().str() + "_t" + std::to_string(batchCount));
            }
            else if (auto lowReadMemOp = dyn_cast<ksim::LowReadMemOp>(op)) {
                if (nameSet.count(lowReadMemOp.getMem()))
                lowReadMemOp.setMem(lowReadMemOp.getMem().str() + "_t" + std::to_string(batchCount));
            }
            else if (auto lowWriteMemOp = dyn_cast<ksim::LowWriteMemOp>(op)) {
                if (nameSet.count(lowWriteMemOp.getMem()))
                lowWriteMemOp.setMem(lowWriteMemOp.getMem().str() + "_t" + std::to_string(batchCount));
            }
        });
        toCloneOps.push_back(clone);
      }
      for(auto op: toCloneOps) {
        builder.insert(op);
      }
      funcOp.walk([&](Operation* op) {
        if(isa<func::FuncOp>(op)) return;
        if(isa<func::ReturnOp>(op)) return;
        // errs() << "\ndebug begin\n" << *op << "\ndebug end\n";
        toDeleteOps.push_back(op);
      });
      // must reverse otherwise crush
      for(auto op: reverse(toDeleteOps)) {
        op->erase();
      }
      // builder.setInsertionPoint(clockOp);
      builder.setInsertionPointToStart(&(funcOp.getRegion().getBlocks().front()));
      for(auto op: toCloneOps) {
        // builder.create<func::CallOp>(builder.getUnknownLoc(), TypeRange{}, SymbolRefAttr::get(builder.getContext(), dyn_cast<func::FuncOp>(op).getName()), ValueRange{});
        SmallVector<Value> operands;
        auto currentFuncOp = dyn_cast<func::FuncOp>(op);
        currentFuncOp.setPrivate();
        builder.create<func::CallOp>(funcOp.getLoc(), currentFuncOp , operands);
      }
      toCloneOps.clear();
      toDeleteOps.clear();
    }    
  }
};

std::unique_ptr<mlir::Pass> ksim::createBatchPass(BatchOptions options) {
  return std::make_unique<BatchPass>(options);
}