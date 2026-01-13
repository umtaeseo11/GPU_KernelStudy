#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// ReLU activation: max(0, x)
// pretty simple but important for neural nets obviously

__global__ void reluKernel(float *a, float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = a[idx] > 0.0f ? a[idx] : 0.0f;  // max(0, x)
    }
}

void relu_cuda(std::vector<float>& host_a, std::vector<float>& host_b, int n) {
    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMemcpy(d_a, host_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    reluKernel<<<gridSize, blockSize>>>(d_a, d_b, n);
    cudaDeviceSynchronize();

    cudaMemcpy(host_b.data(), d_b, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
}

int main() {
    int n = 1000000;
    std::vector<float> a(n), b(n);

    // Initialize input with mix of positive and negative values
    for(int i = 0; i < n; i++) {
        a[i] = static_cast<float>(i - 500000);  // will have negative and positive values
    }

    // Verify inputs
    std::cout << "Sanity checking purposes: " << std::endl;
    std::cout << "a[0] = " << a[0] << " (should be negative)" << std::endl;
    std::cout << "a[500000] = " << a[500000] << " (should be 0)" << std::endl;
    std::cout << "a[500001] = " << a[500001] << " (should be positive)" << std::endl;

    relu_cuda(a, b, n);

    // Verify results
    std::cout << "\nOutput verification:" << std::endl;
    // Check some negative inputs (should become 0)
    for(int i = 0; i < 3; i++) {
        float expected = a[i] > 0.0f ? a[i] : 0.0f;
        std::cout << "b[" << i << "] = " << b[i]
                  << " (expected " << expected << ")" << std::endl;
        if (std::abs(b[i] - expected) > 1e-5) {
            std::cout << "  ....what is happening" << std::endl;
        }
    }

    // Check zero and positive inputs
    for(int i = 500000; i < 500003; i++) {
        float expected = a[i] > 0.0f ? a[i] : 0.0f;
        std::cout << "b[" << i << "] = " << b[i]
                  << " (expected " << expected << ")" << std::endl;
        if (std::abs(b[i] - expected) > 1e-5) {
            std::cout << "  ....what is happening" << std::endl;
        }
    }

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "\nGood I guess" << std::endl;
    return 0;
}
