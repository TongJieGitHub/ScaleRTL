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

#define GEN_PASS_DEF_BATCHFLATTENFUNC
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm; 

struct BatchFlattenFuncPass : ksim::impl::BatchFlattenFuncBase<BatchFlattenFuncPass> {
  using ksim::impl::BatchFlattenFuncBase<BatchFlattenFuncPass>::BatchFlattenFuncBase;
  void runOnOperation() final {
    auto modlist = getOperation();
    // auto & instGraph = getAnalysis<hw::Instanceg>();
  }
};

std::unique_ptr<mlir::Pass> ksim::createBatchFlattenFuncPass() {
  return std::make_unique<BatchFlattenFuncPass>();
}