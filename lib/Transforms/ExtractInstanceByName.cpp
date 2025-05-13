#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"

#define GEN_PASS_DEF_EXTRACTINSTANCEBYNAME
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

struct ExtractInstanceByNamePass
    : public ksim::impl::ExtractInstanceByNameBase<ExtractInstanceByNamePass> {
  ExtractInstanceByNamePass(raw_ostream &os) : os(os) {}


  void filterModuleToOneChild(mlir::ModuleOp root, hw::InstanceGraph &instanceGraph, std::string targetModuleName) {
    // Find the node for the target module.
    hw::InstanceGraphNode *targetNode = nullptr;
    for (hw::InstanceGraphNode *node : instanceGraph) {
        if (node->getModule().getModuleName().str() == targetModuleName) {
            targetNode = node;
            break;
        }
    }

    if (!targetNode) {
        llvm::errs() << "Error: Target module '" << targetModuleName << "' not found.\n";
        return;
    }

    hw::HWModuleOp targetModule = cast<hw::HWModuleOp>(targetNode->getModule().getOperation());

    // Step 2: Collect all dependencies recursively (starting from target module)
    llvm::DenseSet<Operation *> neededModules;

    std::function<void(Operation *)> collectDependencies = [&](Operation *moduleOp) {
        if (neededModules.contains(moduleOp))
            return;

        neededModules.insert(moduleOp);

        if (auto hwModule = dyn_cast<hw::HWModuleOp>(moduleOp)) {
            for (auto &op : hwModule.getBody().getOps()) {
                if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
                    auto childModule = inst.getReferencedModule();
                    collectDependencies(childModule);
                }
            }
        }
    };

    collectDependencies(targetModule.getOperation());

    // Step 3: Walk through all modules in the root ModuleOp.
    //         - If not needed, delete it.
    //         - If needed but not the target module, mark private.
    SmallVector<Operation *, 8> toErase;

    for (Operation &op : root.getBody()->getOperations()) {
        if (isa<hw::HWModuleOp>(op) || isa<hw::HWModuleExternOp>(op)) {
            if (!neededModules.contains(&op)) {
                toErase.push_back(&op);
            } else {
                // This is a needed module — decide if it should be private.
                if (&op != targetModule.getOperation()) {
                    op.setAttr("sym_visibility", StringAttr::get(root.getContext(), "private"));
                } else {
                    op.removeAttr("sym_visibility"); // Ensure public for the main target module
                }
            }
        }
    }

    for (auto *op : toErase) {
        op->erase();
    }

    llvm::errs() << "Filtered to module '" << targetModuleName << "' and its dependencies.\n";
  }



  void runOnOperation() override {
    hw::InstanceGraph &instanceGraph = getAnalysis<hw::InstanceGraph>();

    if (!moduleName.empty()) {
        filterModuleToOneChild(getOperation(), instanceGraph, moduleName);
    }


    markAllAnalysesPreserved();
}


  

raw_ostream &os;
};

std::unique_ptr<mlir::Pass> ksim::createExtractInstanceByNamePass(ExtractInstanceByNameOptions options) {
  return std::make_unique<ExtractInstanceByNamePass>(llvm::errs());
}
