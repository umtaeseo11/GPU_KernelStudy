#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Boundary Handling: what happens when N isn't a nice multiple of block_size?
// Real data is almost never perfectly aligned. Need to handle the leftover threads
// that go past the end of the array without reading/writing garbage memory.
//
// The key insight: the grid might launch MORE threads than we have elements.
// Those extra threads must do nothing (bounds check with idx < n).
// This is the same pattern from day1 but now we're being explicit about it.
//
// Testing with weird sizes like primes, powers of 2 +/- 1, etc.

__global__ void boundaryAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // This is THE boundary check - without it we'd read/write past array bounds
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Also demonstrating a kernel that processes multiple elements per thread
// (grid-stride loop) - another way to handle arbitrary sizes
__global__ void boundaryAddStride(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // Each thread handles multiple elements, stepping by total thread count
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

void boundary_add_cuda(std::vector<float>& host_a, std::vector<float>& host_b,
                       std::vector<float>& host_c, int n, bool use_stride) {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    cudaMemcpy(d_a, host_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, host_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;

    if (use_stride) {
        // Intentionally use fewer blocks than needed - stride loop handles the rest
        int gridSize = 32; // way less than needed
        boundaryAddStride<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    } else {
        int gridSize = (n + blockSize - 1) / blockSize;
        boundaryAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(host_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    // Test a bunch of awkward sizes that don't align with block_size=256
    int test_sizes[] = {1, 7, 255, 257, 1023, 1025, 100003, 999999};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    bool all_good = true;

    for (int t = 0; t < num_tests; t++) {
        int n = test_sizes[t];
        std::vector<float> a(n), b(n), c(n);

        for (int i = 0; i < n; i++) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i * 2);
        }

        // Test both approaches
        for (int method = 0; method < 2; method++) {
            bool use_stride = (method == 1);
            boundary_add_cuda(a, b, c, n, use_stride);

            std::string method_name = use_stride ? "grid-stride" : "basic bounds check";

            // Check first, last, and some middle elements
            int check_indices[] = {0, n/2, n-1};
            for (int idx : check_indices) {
                float expected = static_cast<float>(idx * 3);
                if (std::abs(c[idx] - expected) > 1e-5) {
                    std::cout << "N=" << n << " [" << method_name << "] c[" << idx
                              << "] = " << c[idx] << " expected " << expected
                              << " ....what is happening" << std::endl;
                    all_good = false;
                }
            }
        }
    }

    std::cout << "Sanity checking purposes:" << std::endl;
    std::cout << "Tested sizes: ";
    for (int t = 0; t < num_tests; t++) {
        std::cout << test_sizes[t];
        if (t < num_tests - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    std::cout << "Both basic bounds check and grid-stride loop tested" << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (all_good) {
        std::cout << "\nGood I guess" << std::endl;
    }
    return 0;
}
