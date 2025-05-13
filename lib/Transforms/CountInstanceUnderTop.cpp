#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"

#define GEN_PASS_DEF_COUNTINSTANCEUNDERTOP
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

struct CountInstanceUnderTopPass
    : public ksim::impl::CountInstanceUnderTopBase<CountInstanceUnderTopPass> {
  CountInstanceUnderTopPass(raw_ostream &os) : os(os) {}


//   void extractAndSaveModule(hw::InstanceGraph &instanceGraph, StringRef moduleName) {
//     MLIRContext *context;

//     // Find the graph node for `moduleName`
//     hw::InstanceGraphNode *targetNode = nullptr;

//     for (hw::InstanceGraphNode *node : instanceGraph) {
//         if (node->getModule().getModuleName() == moduleName) {
//             targetNode = node;
//             context = node->getModule().getContext();
//             break;
//         }
//     }

//     if (!targetNode) {
//         llvm::errs() << "Error: Module '" << moduleName << "' not found in instance graph.\n";
//         return;
//     }

//     hw::HWModuleOp targetModule = cast<hw::HWModuleOp>(targetNode->getModule().getOperation());

//     llvm::errs() << "Extracting module: " << moduleName << "\n";

//     // Step 2: Collect all dependencies recursively
//     llvm::DenseSet<Operation *> neededModules;

//     std::function<void(Operation *)> collectDependencies = [&](Operation *moduleOp) {
//         if (neededModules.contains(moduleOp))
//             return;

//         neededModules.insert(moduleOp);

//         if (auto hwModule = dyn_cast<hw::HWModuleOp>(moduleOp)) {
//             for (auto &op : hwModule.getBody().getOps()) {
//                 if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
//                     auto childModule = inst.getReferencedModule();
//                     collectDependencies(childModule);
//                 }
//             }
//         }
//     };

//     collectDependencies(targetModule.getOperation());

//     // Step 3: Create a new standalone MLIR module with only these dependencies
//     OwningOpRef<mlir::ModuleOp> extractedModule = mlir::ModuleOp::create(mlir::UnknownLoc::get(context));

//     for (Operation *op : neededModules) {
//         auto cloned = op->clone();

//         if (cloned != targetModule.getOperation()) {
//             // Submodules become private
//             cloned->setAttr("sym_visibility", StringAttr::get(context, "private"));
//         } else {
//             // The directly referenced module stays public
//             cloned->removeAttr("sym_visibility");
//         }

//         extractedModule->push_back(cloned);
//     }

//     // Step 4: Write to file
//     std::error_code ec;
//     llvm::raw_fd_ostream outFile("ExtractedModule.mlir", ec, llvm::sys::fs::OF_Text);
//     if (ec) {
//         llvm::errs() << "Failed to open file for writing: " << ec.message() << "\n";
//         return;
//     }

//     extractedModule->print(outFile);
//     llvm::errs() << "Saved extracted module to 'ExtractedModule.mlir'\n";
//   }


  void filterModuleToOneChild(mlir::ModuleOp root, hw::InstanceGraph &instanceGraph, StringRef targetModuleName) {
    // Find the node for the target module.
    hw::InstanceGraphNode *targetNode = nullptr;
    for (hw::InstanceGraphNode *node : instanceGraph) {
        if (node->getModule().getModuleName() == targetModuleName) {
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

    // Find the top module using the "use with no instance" heuristic.
    hw::InstanceGraphNode *topNode = nullptr;

    for (hw::InstanceGraphNode *node : instanceGraph) {
        for (hw::InstanceRecord *use : node->uses()) {
            if (!use->getInstance() && node->getNumUses() == 1) {
                topNode = node;
                break;
            }
        }
        if (topNode) break;
    }

    if (!topNode) {
        os << "Error: No top module found.\n";
        return;
    }

    auto topModuleName = topNode->getModule().getModuleName();
    os << "Top module: " << topModuleName << "\n";

    // Scan all uses across all nodes to find instances *inside* the top module.
    int directInstanceCount = 0;
    StringRef childModuleName;

    for (hw::InstanceGraphNode *node : instanceGraph) {
        for (hw::InstanceRecord *use : node->uses()) {
            if (!use->getInstance())
                continue;

            // Find which module contains this instance.
            hw::InstanceGraphNode *parentNode = use->getParent();
            if (parentNode != topNode)
                continue;  // Skip instances not directly inside the top module.

            auto instance = use->getInstance();
            childModuleName = node->getModule().getModuleName();

            os << "  Direct instance: " << instance.getInstanceName()
               << " (Module type: " << childModuleName << ")\n";

            directInstanceCount++;
        }
    }

    if (!childModuleName.empty()) {
        //extractAndSaveModule(instanceGraph, childModuleName);
        filterModuleToOneChild(getOperation(), instanceGraph, childModuleName);
    }

    os << "Total direct instances under top module: " << directInstanceCount << "\n";

    markAllAnalysesPreserved();
}


  

raw_ostream &os;
};

std::unique_ptr<mlir::Pass> ksim::createCountInstanceUnderTopPass() {
  return std::make_unique<CountInstanceUnderTopPass>(llvm::errs());
}
