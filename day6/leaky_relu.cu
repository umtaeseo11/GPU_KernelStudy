#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Leaky ReLU: x < 0 ? alpha * x : x
// basically ReLU but with a small slope for negative values instead of zeroing them out
// helps with the "dying ReLU" problem i think

__global__ void leakyReluKernel(float *a, float *b, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = a[idx] < 0.0f ? alpha * a[idx] : a[idx];
    }
}

void leaky_relu_cuda(std::vector<float>& host_a, std::vector<float>& host_b, float alpha, int n) {
    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMemcpy(d_a, host_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    leakyReluKernel<<<gridSize, blockSize>>>(d_a, d_b, alpha, n);
    cudaDeviceSynchronize();

    cudaMemcpy(host_b.data(), d_b, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
}

int main() {
    int n = 1000000;
    float alpha = 0.01f;  // typical value for leaky relu
    std::vector<float> a(n), b(n);

    // Initialize input with mix of positive and negative values
    for(int i = 0; i < n; i++) {
        a[i] = static_cast<float>(i - 500000);
    }

    // Verify inputs
    std::cout << "Sanity checking purposes: " << std::endl;
    std::cout << "Alpha value: " << alpha << std::endl;
    std::cout << "a[0] = " << a[0] << " (should be negative)" << std::endl;
    std::cout << "a[500000] = " << a[500000] << " (should be 0)" << std::endl;
    std::cout << "a[500001] = " << a[500001] << " (should be positive)" << std::endl;

    leaky_relu_cuda(a, b, alpha, n);

    // Verify results
    std::cout << "\nOutput verification:" << std::endl;
    // Check some negative inputs (should be multiplied by alpha)
    for(int i = 0; i < 3; i++) {
        float expected = a[i] < 0.0f ? alpha * a[i] : a[i];
        std::cout << "b[" << i << "] = " << b[i]
                  << " (expected " << expected << ")" << std::endl;
        if (std::abs(b[i] - expected) > 1e-3) {  // slightly larger tolerance for float multiplication
            std::cout << "  ....what is happening" << std::endl;
        }
    }

    // Check zero and positive inputs (should stay the same)
    for(int i = 500000; i < 500003; i++) {
        float expected = a[i] < 0.0f ? alpha * a[i] : a[i];
        std::cout << "b[" << i << "] = " << b[i]
                  << " (expected " << expected << ")" << std::endl;
        if (std::abs(b[i] - expected) > 1e-3) {
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
