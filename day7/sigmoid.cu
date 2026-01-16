#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Sigmoid: 1 / (1 + exp(-x))
// Numerical stability is the key here!
// For large positive x: exp(-x) underflows to 0, sigmoid -> 1 (ok)
// For large negative x: exp(-x) overflows -> infinity (bad!)
//
// Solution: use two different formulas based on sign of x
// x >= 0: 1 / (1 + exp(-x))
// x < 0:  exp(x) / (1 + exp(x))
// This way we always compute exp of a negative number, which won't overflow

__global__ void sigmoidKernel(float *a, float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        if (x >= 0.0f) {
            // standard formula - exp(-x) won't overflow since x >= 0
            b[idx] = 1.0f / (1.0f + expf(-x));
        } else {
            // equivalent formula - exp(x) won't overflow since x < 0
            float exp_x = expf(x);
            b[idx] = exp_x / (1.0f + exp_x);
        }
    }
}

void sigmoid_cuda(std::vector<float>& host_a, std::vector<float>& host_b, int n) {
    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMemcpy(d_a, host_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    sigmoidKernel<<<gridSize, blockSize>>>(d_a, d_b, n);
    cudaDeviceSynchronize();

    cudaMemcpy(host_b.data(), d_b, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
}

// CPU reference for verification (also numerically stable)
float sigmoid_cpu(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        float exp_x = expf(x);
        return exp_x / (1.0f + exp_x);
    }
}

int main() {
    int n = 1000000;
    std::vector<float> a(n), b(n);

    // Initialize input with a range that tests numerical stability
    // includes very negative, zero, and very positive values
    for(int i = 0; i < n; i++) {
        // range from -100 to 100 to test stability at extremes
        a[i] = static_cast<float>(i - 500000) / 5000.0f;
    }

    // Verify inputs
    std::cout << "Sanity checking purposes: " << std::endl;
    std::cout << "a[0] = " << a[0] << " (very negative)" << std::endl;
    std::cout << "a[250000] = " << a[250000] << " (moderately negative)" << std::endl;
    std::cout << "a[500000] = " << a[500000] << " (should be 0)" << std::endl;
    std::cout << "a[750000] = " << a[750000] << " (moderately positive)" << std::endl;
    std::cout << "a[999999] = " << a[999999] << " (very positive)" << std::endl;

    sigmoid_cuda(a, b, n);

    // Verify results
    std::cout << "\nOutput verification:" << std::endl;

    // Test indices covering the full range
    int test_indices[] = {0, 250000, 500000, 750000, 999999};
    bool all_good = true;

    for(int i : test_indices) {
        float expected = sigmoid_cpu(a[i]);
        std::cout << "sigmoid(" << a[i] << ") = " << b[i]
                  << " (expected " << expected << ")" << std::endl;
        if (std::abs(b[i] - expected) > 1e-5) {
            std::cout << "  ....what is happening" << std::endl;
            all_good = false;
        }
    }

    // Extra check: sigmoid should always be in (0, 1)
    std::cout << "\nBounds check (should be in (0, 1)):" << std::endl;
    std::cout << "sigmoid(-100) = " << b[0] << " (should be close to 0)" << std::endl;
    std::cout << "sigmoid(0) = " << b[500000] << " (should be 0.5)" << std::endl;
    std::cout << "sigmoid(100) = " << b[999999] << " (should be close to 1)" << std::endl;

    if (b[0] <= 0.0f || b[0] >= 1.0f ||
        b[999999] <= 0.0f || b[999999] >= 1.0f) {
        std::cout << "  bounds violated....what is happening" << std::endl;
        all_good = false;
    }

    // Check for CUDA errors
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
