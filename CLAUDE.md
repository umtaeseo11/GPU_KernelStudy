# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **100-day CUDA & Triton kernel implementation challenge**. Each day focuses on implementing a specific high-performance kernel, progressing from basic element-wise operations to advanced LLM kernels like FlashAttention and MoE.

## Code Organization

- **Structure**: `dayN/` directories containing individual `.cu` files
- **Day Format**: Each day has a single CUDA kernel file (e.g., `day1/vector_add.cu`, `day2/vector_scale.cu`)
- **Progression**: Follows the 5-phase curriculum defined in README.md:
  - Phase 1 (Days 1-20): Element-wise operations & memory foundation
  - Phase 2 (Days 21-40): Reductions & parallel algorithms
  - Phase 3 (Days 41-60): Normalization & memory layout
  - Phase 4 (Days 61-80): Matrix multiplication (GEMM)
  - Phase 5 (Days 81-100): LLM & SOTA specialized kernels

## Building and Running

Each `.cu` file is standalone and can be compiled directly:

```bash
nvcc -o vector_add day1/vector_add.cu
./vector_add
```

For newer kernels with optimizations:
```bash
nvcc -O3 -arch=sm_80 -o kernel dayN/kernel_name.cu
./kernel
```

## Code Style Conventions

Based on existing implementations (day1, day2):

- **Kernel naming**: camelCase for kernel functions (e.g., `vectorAdd`, `vectorScale`)
- **Wrapper naming**: snake_case for host wrapper functions (e.g., `vector_add_cuda`)
- **Block size**: Default to 256 threads
- **Array size**: Default to 1,000,000 elements for testing
- **Verification**: Include "Sanity checking purposes" for inputs and "Output verification" for results
- **Error messages**: Use informal messages like "....what is happening" for failures, "Good I guess" for success
- **Memory pattern**: Always include `cudaDeviceSynchronize()` after kernel launch, proper `cudaFree` cleanup
- **Comments**: Include personal notes and references (e.g., "remember c ain't c++...don't get confused with syntax")

## Implementation Requirements

When implementing a new day's kernel:

1. Create `dayN/` directory
2. Implement the kernel matching the exact task from README.md curriculum
3. Follow the established pattern:
   - `__global__` kernel function
   - Host wrapper with CUDA memory management
   - `main()` with initialization, verification, and error checking
4. Match the informal, learning-focused code style
5. Include comments that reflect learning process
