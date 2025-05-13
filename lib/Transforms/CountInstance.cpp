#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"

#define GEN_PASS_DEF_COUNTINSTANCE
#include "PassDetails.h"

using namespace ksim;
using namespace circt;
using namespace mlir;
using namespace llvm;

struct CountInstancePass
    : public ksim::impl::CountInstanceBase<CountInstancePass> {
  CountInstancePass(raw_ostream &os) : os(os) {}


    /// Recursive function to compute flattened instance count.
  int computeFlattenedInstanceCount(hw::InstanceGraphNode *node, 
    llvm::DenseMap<hw::InstanceGraphNode *, int> &flatCountCache) {
    // If already computed, return cached value
    if (flatCountCache.count(node))
      return flatCountCache[node];

    int totalInstances = 0; // Total instance count of this module

    // Iterate over all instances of this module
    for (hw::InstanceRecord *use : node->uses()) {
      if (use->getInstance()) {
        int parentCount = computeFlattenedInstanceCount(use->getParent(), flatCountCache);
        totalInstances += parentCount; // Accumulate instances based on parent count
      }
    }

    // If this module is not instantiated anywhere (top module), count as 1
    if (totalInstances == 0)
      totalInstances = 1;

    // Store result in cache
    flatCountCache[node] = totalInstances;
    return totalInstances;
  }


void runOnOperation() override {
  hw::InstanceGraph &instanceGraph = getAnalysis<hw::InstanceGraph>();

  llvm::DenseMap<hw::InstanceGraphNode *, int> flatInstanceCount;
  llvm::DenseMap<hw::InstanceGraphNode *, int> directInstanceCount;
  
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
  // os << "Top module: " << topModuleName << "\n";

  // Scan all uses across all nodes to find instances *inside* the top module.
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

          // os << "  Direct instance: " << instance.getInstanceName()
          //    << " (Module type: " << childModuleName << ")\n";
      }
  }





  // Step 1: Compute direct instantiation count
  for (hw::InstanceGraphNode *node : instanceGraph) {
    int count = 0;
    for (hw::InstanceRecord *use : node->uses()) {
      count++;
    }
    directInstanceCount[node] = count;
  }

  // Step 2: Compute flattened instance counts recursively
  for (hw::InstanceGraphNode *node : instanceGraph) {
    computeFlattenedInstanceCount(node, flatInstanceCount);
  }

  // Step 3: Print results
  // os << "Flattened Instance Graph:\n";
  // for (hw::InstanceGraphNode *node : instanceGraph) {
  //   auto mod = node->getModule();
  //   os << "Module: " << mod.getModuleName()
  //      << " (Direct Instantiated " << directInstanceCount[node] << " times, "
  //      << "Flattened Count: " << flatInstanceCount[node] << ")\n";

  //   for (hw::InstanceRecord *use : node->uses()) {
  //     if (use->getInstance()) {
  //       os << "  ├── Instance: " << use->getInstance().getInstanceName() 
  //       << " in Module: " << use->getParent()->getModule().getModuleName()
  //       << "\n";
  //     } 
  //   }
  // }

  // another step3:
  for (hw::InstanceGraphNode *node : instanceGraph) {
    auto mod = node->getModule();
    for (hw::InstanceRecord *use : node->uses()) {
      if (use->getInstance() && use->getParent()->getModule().getModuleName().str() == childModuleName.str()) {
        os 
          // << childModuleName << "," 
          << mod.getModuleName() << "," 
          << flatInstanceCount[node] << "\n";
        break;
      } 
    }
  }

  markAllAnalysesPreserved();
}

raw_ostream &os;
};

std::unique_ptr<mlir::Pass> ksim::createCountInstancePass() {
  return std::make_unique<CountInstancePass>(llvm::errs());
}
