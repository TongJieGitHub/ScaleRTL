import math

# Default block size (for larger modules)
DEFAULT_BLOCK_SIZE = 32

# How many iterations to batch into one graph (parameterized!)
BATCH_SIZE = 10   # You can change this easily or pass it via argparse if you want.

# Input file with module list (name, size)
modules_file = "modules.txt"

# Read modules.txt and compute per-module blockSize and gridSize
modules = []
with open(modules_file, "r") as f:
    for line in f:
        name, size = line.strip().split(",")
        size = int(size)
        blockSize = size if size <= 32 else DEFAULT_BLOCK_SIZE
        gridSize = math.ceil(size / blockSize)
        modules.append((name, size, blockSize, gridSize))

# Generate CUDA code
code = f"""
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda.h>

constexpr int BATCH_SIZE = {BATCH_SIZE};  // Batched iteration count
"""

# Include all headers for the modules
for mod, _, _, _ in modules:
    code += f'#include "EvalContext_{mod}.h"\n'
    code += f'#include "kernel_{mod}_fatbin.h"\n'

code += """
void checkCuResult(CUresult result, const char* msg) {
    if (result != CUDA_SUCCESS) {
        const char* errorStr;
        cuGetErrorString(result, &errorStr);
        std::cerr << msg << ": " << errorStr << std::endl;
        exit(1);
    }
}

struct ModuleInfo {
    const char* name;
    int size;
    int blockSize;
    int gridSize;
};

struct ModuleResources {
    CUmodule cuModule;
    CUfunction kernelFunction;
    CUdeviceptr deviceContext;
    CUstream stream;
    CUgraph graph;
    CUgraphExec graphExec;
};

std::vector<ModuleInfo> moduleList = {
"""
for mod, size, blockSize, gridSize in modules:
    code += f'    {{"{mod}", {size}, {blockSize}, {gridSize}}},\n'
code += "};\n"

# Main function
code += """
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <total_iteration_count>" << std::endl;
        return 1;
    }
    int totalIterations = std::stoi(argv[1]);
    if (totalIterations % BATCH_SIZE != 0) {
        std::cerr << "Total iterations must be divisible by BATCH_SIZE = " << BATCH_SIZE << std::endl;
        return 1;
    }
    int graphLaunchCount = totalIterations / BATCH_SIZE;

    checkCuResult(cuInit(0), "Failed to initialize CUDA Driver API");

    CUdevice cuDevice;
    checkCuResult(cuDeviceGet(&cuDevice, 0), "Failed to get device");

    CUcontext cuContext;
    checkCuResult(cuCtxCreate(&cuContext, 0, cuDevice), "Failed to create CUDA context");

    std::vector<ModuleResources> resources(moduleList.size());

    auto allocStart = std::chrono::high_resolution_clock::now();
"""

# Setup for each module
for idx, (mod, size, blockSize, gridSize) in enumerate(modules):
    code += f"""
    {{
        dim3 blockDim({blockSize});
        dim3 gridDim({gridSize});

        checkCuResult(cuModuleLoadData(&resources[{idx}].cuModule, kernelFatbin_{mod}), "Failed to load {mod} fatbin");
        checkCuResult(cuModuleGetFunction(&resources[{idx}].kernelFunction, resources[{idx}].cuModule, "{mod}"), "Failed to get kernel function for {mod}");

        checkCuResult(cuMemAlloc(&resources[{idx}].deviceContext, sizeof(EvalContext_{mod}) * ({size} * 2)), "Failed to allocate memory for {mod}");
        checkCuResult(cuMemsetD8(resources[{idx}].deviceContext, 0, sizeof(EvalContext_{mod}) * ({size} * 2)), "Failed to memset for {mod}");

        checkCuResult(cuStreamCreate(&resources[{idx}].stream, CU_STREAM_DEFAULT), "Failed to create stream for {mod}");

        checkCuResult(cuGraphCreate(&resources[{idx}].graph, 0), "Failed to create graph for {mod}");

        CUgraphNode lastNode = nullptr;
        for (int iter = 0; iter < BATCH_SIZE; ++iter) {{
            void* args[] = {{ &resources[{idx}].deviceContext }};
            CUDA_KERNEL_NODE_PARAMS nodeParams = {{}};
            nodeParams.func = resources[{idx}].kernelFunction;
            nodeParams.gridDimX = gridDim.x;
            nodeParams.gridDimY = 1;
            nodeParams.gridDimZ = 1;
            nodeParams.blockDimX = blockDim.x;
            nodeParams.blockDimY = 1;
            nodeParams.blockDimZ = 1;
            nodeParams.sharedMemBytes = 0;
            nodeParams.kernelParams = args;
            nodeParams.extra = nullptr;

            CUgraphNode kernelNode;
            checkCuResult(cuGraphAddKernelNode(&kernelNode, resources[{idx}].graph, lastNode ? &lastNode : nullptr, lastNode ? 1 : 0, &nodeParams), "Failed to add kernel node for {mod}");
            lastNode = kernelNode;
        }}

        checkCuResult(cuGraphInstantiateWithFlags(&resources[{idx}].graphExec, resources[{idx}].graph, 0), "Failed to instantiate graph for {mod}");
    }}
    """

code += """
    checkCuResult(cuCtxSynchronize(), "Failed to synchronize after setup");

    auto allocEnd = std::chrono::high_resolution_clock::now();

    auto kernelStart = std::chrono::high_resolution_clock::now();

    for (int batch = 0; batch < graphLaunchCount; ++batch) {
        for (size_t j = 0; j < moduleList.size(); ++j) {
            checkCuResult(cuGraphLaunch(resources[j].graphExec, resources[j].stream), "Failed to launch graph for module");
        }
        for (size_t j = 0; j < moduleList.size(); ++j) {
            checkCuResult(cuStreamSynchronize(resources[j].stream), "Failed to synchronize stream for module");
        }
    }

    auto kernelEnd = std::chrono::high_resolution_clock::now();

    auto freeStart = std::chrono::high_resolution_clock::now();

    for (size_t j = 0; j < moduleList.size(); ++j) {
        checkCuResult(cuMemFree(resources[j].deviceContext), "Failed to free device memory");
        checkCuResult(cuModuleUnload(resources[j].cuModule), "Failed to unload module");
        checkCuResult(cuGraphExecDestroy(resources[j].graphExec), "Failed to destroy graphExec");
        checkCuResult(cuGraphDestroy(resources[j].graph), "Failed to destroy graph");
        checkCuResult(cuStreamDestroy(resources[j].stream), "Failed to destroy stream");
    }

    checkCuResult(cuCtxDestroy(cuContext), "Failed to destroy CUDA context");

    auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(kernelEnd - allocStart).count();
    std::cout << totalTime << std::endl;

    return 0;
}
"""

with open("benchmark_graph_v2.cu", "w") as f:
    f.write(code)

print(f"Generated batched graph CUDA code in 'benchmark_graph_v2.cu' with BATCH_SIZE = {BATCH_SIZE}")
