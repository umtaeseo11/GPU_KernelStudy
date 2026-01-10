# Skills

## write-cuda-kernel

**Description**: Implements a CUDA kernel following the 100-day challenge curriculum and code style conventions.

**Instructions**:

You are tasked with implementing a CUDA kernel for this repository's learning challenge. Follow these steps:

1. **Understand the task**: Read the day's requirements from README.md to understand what kernel needs to be implemented.

2. **Create directory structure**: Create a `dayN/` directory for the implementation.

3. **Implement following the established pattern**:
   - Use camelCase for kernel functions (e.g., `vectorAdd`, `vectorSub`)
   - Use snake_case for host wrapper functions (e.g., `vector_sub_cuda`)
   - Default block size: 256 threads
   - Default test array size: 1,000,000 elements
   - Include proper CUDA memory management with `cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`, `cudaFree`

4. **Code structure**:
   ```cpp
   #include <cuda_runtime.h>
   #include <iostream>
   #include <vector>

   // Personal notes/references as comments

   __global__ void kernelName(...) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) {
           // kernel logic
       }
   }

   void kernel_name_cuda(...) {
       // Memory allocation
       // Memory copy H2D
       // Kernel launch with 256 threads per block
       // cudaDeviceSynchronize()
       // Memory copy D2H
       // cudaFree cleanup
   }

   int main() {
       // Initialize inputs
       // "Sanity checking purposes" verification
       // Call CUDA wrapper
       // "Output verification" with informal messages
       // "....what is happening" for failures
       // "Good I guess" for success
   }
   ```

5. **Verification**: Include input sanity checks and output verification with the informal, learning-focused messaging style.

6. **After implementation**: Offer to run the `/optimize-cuda-kernel` skill to apply performance optimizations to the kernel.

**Example usage**:
- User: "Implement day 5"
- User: "Write the ReLU kernel for day 5"

**Can call**: optimize-cuda-kernel (after implementing a kernel)

---

## optimize-cuda-kernel

**Description**: Optimizes an existing CUDA kernel with performance improvements like memory coalescing, shared memory usage, vectorized loads, and warp-level primitives.

**Instructions**:

You are tasked with optimizing an existing CUDA kernel for better performance. Follow these steps:

1. **Read the current implementation**: Use the Read tool to examine the existing `.cu` file.

2. **Identify optimization opportunities** based on the kernel type:
   - **Memory optimizations**:
     - Coalesced memory access patterns
     - Vectorized loads using `float4` or `float2`
     - Shared memory usage for data reuse
   - **Warp-level optimizations**:
     - Warp shuffle operations for reductions
     - Avoid warp divergence
   - **Block-level optimizations**:
     - Shared memory with proper bank conflict avoidance
     - Block-level reductions
   - **Arithmetic optimizations**:
     - Fused multiply-add (FMA) operations
     - Minimize register pressure
     - Loop unrolling where beneficial

3. **Apply optimizations** while maintaining:
   - The informal, learning-focused comment style
   - Proper verification and error checking
   - The same output format and messaging

4. **Add optimization notes**: Include comments explaining what optimizations were applied and why (e.g., "using float4 for vectorized loads - 4x throughput").

5. **Suggest compilation flags**: Recommend appropriate nvcc flags like:
   - `-O3` for optimization level
   - `-arch=sm_80` or appropriate compute capability
   - `--use_fast_math` when applicable

6. **After optimization**: Offer to re-run `/write-cuda-kernel` if the user wants to implement a new kernel next.

**Example usage**:
- User: "Optimize the day 3 kernel"
- User: "/optimize-cuda-kernel day5/relu.cu"

**Can call**: write-cuda-kernel (to implement the next day's kernel after optimization)

---
