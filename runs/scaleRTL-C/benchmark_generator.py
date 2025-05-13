#!/usr/bin/env python3
"""
Benchmark generator for the EvalFunc performance testing.
This script generates C++ benchmark code with hardcoded thread and context counts.
"""

import argparse
import os
import sys

# Template for the benchmark code - note the doubled curly braces {{ }} for C++ code blocks
# Only {num_threads} and {num_contexts} will be replaced
BENCHMARK_TEMPLATE = """#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <omp.h>

// Include the generated header
#include "EvalContext.h"

int main(int argc, char* argv[]) {{
    // Check command line arguments
    if (argc < 2) {{
        std::cerr << "Usage: " << argv[0] << " <iteration_count>" << std::endl;
        return 1;
    }}
    
    // Parse iteration count from command line
    int cnt = atoi(argv[1]);
    
    // Number of contexts and threads (hardcoded)
    const int NUM_THREADS = {num_threads};
    const int NUM_CONTEXTS = {num_contexts};
    
    // Set the number of OpenMP threads
    omp_set_num_threads(NUM_THREADS);
    
    // Create array of EvalContext instances
    std::vector<EvalContext> contexts(NUM_CONTEXTS);
    
    // Initialize all contexts (just zero them out)
    for (int i = 0; i < NUM_CONTEXTS; ++i) {{
        std::memset(&contexts[i], 0, sizeof(EvalContext));
    }}
    
    // Start timing the entire program
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run the evaluation loop with the specified count
    for (int iter = 0; iter < cnt; ++iter) {{
    #pragma omp parallel for
    for (int i = 0; i < NUM_THREADS; ++i) {{
        // Calculate the range of contexts this thread will handle
        int contexts_per_thread = NUM_CONTEXTS / NUM_THREADS;
        int start_idx = i * contexts_per_thread;
        int end_idx = (i == NUM_THREADS - 1) ? NUM_CONTEXTS : (i + 1) * contexts_per_thread;
        
        // Each thread processes its assigned contexts for the specified number of iterations
            for (int j = start_idx; j < end_idx; ++j) {{
                // Call the benchmark function with each context
                EvalFunc(&contexts[j]);
            }}
        }}
    }}
    
    // Stop timing
    auto stop = std::chrono::high_resolution_clock::now();
    
    // Output only the total duration in microseconds
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << std::endl;
    
    return 0;
}}
"""

def generate_benchmark(num_threads, num_contexts, output_file):
    """
    Generate a benchmark file with the specified parameters
    """
    # Fill in the template with the provided values
    benchmark_code = BENCHMARK_TEMPLATE.format(
        num_threads=num_threads,
        num_contexts=num_contexts
    )
    
    # Write the generated code to the output file
    with open(output_file, 'w') as f:
        f.write(benchmark_code)
    
    print(f"Generated benchmark file: {output_file}")
    print(f"Threads: {num_threads}, Contexts: {num_contexts}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate benchmark code for EvalFunc testing')
    parser.add_argument('-t', '--threads', type=int, default=2, 
                        help='Number of threads to use (default: 2)')
    parser.add_argument('-c', '--contexts', type=int, default=8, 
                        help='Number of context instances to create (default: 8)')
    parser.add_argument('-o', '--output', type=str, default='benchmark.cpp',
                        help='Output file name (default: benchmark.cpp)')
    
    args = parser.parse_args()
    
    # Generate the benchmark file
    generate_benchmark(args.threads, args.contexts, args.output)
    
    # Print compilation instructions
    print("\nCompilation command:")
    print(f"clang++ -std=c++17 -O2 -fopenmp {args.output} transformed.o -o benchmark_cpu")
    
    # Print execution instructions
    print("\nExecution command:")
    print("./benchmark_cpu <iteration_count>")

if __name__ == "__main__":
    main()