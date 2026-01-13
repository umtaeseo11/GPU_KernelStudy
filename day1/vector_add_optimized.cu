#include <cuda_runtime.h>
#include <iostream>
#include <vector>

//remember c ain't c++...don't get confused with syntax, look back at 392?
// Optimized version with float4 vectorization + better error handling

// Error checking macro - way cleaner than checking every call manually
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Optimized: Use float4 for vectorized memory access
// Loads 128 bits at once instead of 32 bits = 4x memory throughput (theoretically)
__global__ void vectorAddOptimized(float *a, float *b, float *c, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // Vectorized load/store when we have 4 elements available
    if (idx + 3 < n) {
        float4 a4 = reinterpret_cast<float4*>(a)[idx / 4];
        float4 b4 = reinterpret_cast<float4*>(b)[idx / 4];
        float4 c4;

        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;

        reinterpret_cast<float4*>(c)[idx / 4] = c4;
    } else {
        // Handle tail elements that don't fit in float4
        for (int i = idx; i < n && i < idx + 4; i++) {
            c[i] = a[i] + b[i];
        }
    }
}

// Original simple version - kept for comparison
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void vector_add_cuda(std::vector<float>& host_a, std::vector<float>& host_b,
                     std::vector<float>& host_c, int n, bool use_optimized = true) {
    float *d_a, *d_b, *d_c;

    // Better error checking with macro
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, host_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, host_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    if (use_optimized) {
        // Each thread handles 4 elements, so divide grid by 4
        gridSize = (n + blockSize * 4 - 1) / (blockSize * 4);

        CUDA_CHECK(cudaEventRecord(start));
        vectorAddOptimized<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(stop));
    } else {
        gridSize = (n + blockSize - 1) / blockSize;

        CUDA_CHECK(cudaEventRecord(start));
        vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
        CUDA_CHECK(cudaEventRecord(stop));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << (use_optimized ? "Optimized" : "Original")
              << " kernel time: " << milliseconds << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(host_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    int n = 1000000; // max num of elements, should store enough but try lower
    std::vector<float> a(n), b(n), c_opt(n), c_orig(n);

    // Clear initialization
    for(int i = 0; i < n; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    // Verify inputs
    std::cout << "Sanity checking purposes: " << std::endl;
    std::cout << "a[0] = " << a[0] << ", b[0] = " << b[0] << std::endl;
    std::cout << "a[1] = " << a[1] << ", b[1] = " << b[1] << std::endl;
    std::cout << a[2] << std::endl;

    // Run both versions for comparison
    std::cout << "\n=== Running original kernel ===" << std::endl;
    vector_add_cuda(a, b, c_orig, n, false);

    std::cout << "\n=== Running optimized kernel ===" << std::endl;
    vector_add_cuda(a, b, c_opt, n, true);

    // Verify results (should be 0, 3, 6, 9, 12, ...)
    std::cout << "\nOutput verification (optimized):" << std::endl;
    bool all_good = true;
    for(int i = 0; i < 5; i++) {
        std::cout << "c[" << i << "] = " << c_opt[i]
                  << " (expected " << static_cast<float>(i * 3) << ")" << std::endl;
        if (std::abs(c_opt[i] - static_cast<float>(i * 3)) > 1e-5) {
            std::cout << "  ....what is happening" << std::endl;
            all_good = false;
        }
    }

    // Check last few elements too (tail case)
    std::cout << "\nTail verification:" << std::endl;
    for(int i = n - 3; i < n; i++) {
        std::cout << "c[" << i << "] = " << c_opt[i]
                  << " (expected " << static_cast<float>(i * 3) << ")" << std::endl;
        if (std::abs(c_opt[i] - static_cast<float>(i * 3)) > 1e-5) {
            std::cout << "  ....what is happening in tail" << std::endl;
            all_good = false;
        }
    }

    if (all_good) {
        std::cout << "\nGood I guess - vectorized version works!" << std::endl;
    }

    return 0;
}
