#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// AXPBY: y = alpha * x + beta * y
// This is a fused operation - instead of doing two separate kernels (scale x, scale y, add),
// we do it all in one pass. Load/Store optimization ftw

__global__ void axpby(float *x, float *y, float alpha, float beta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Read both x and y, compute, write back to y
        y[idx] = alpha * x[idx] + beta * y[idx];
    }
}

void axpby_cuda(std::vector<float>& host_x, std::vector<float>& host_y,
                float alpha, float beta, int n) {
    float *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_x, host_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, host_y.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    axpby<<<gridSize, blockSize>>>(d_x, d_y, alpha, beta, n);
    cudaDeviceSynchronize();

    // Only copy back y since that's where the result is stored
    cudaMemcpy(host_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int n = 1000000;
    float alpha = 2.0f;
    float beta = 3.0f;
    std::vector<float> x(n), y(n);

    // Initialize inputs
    for(int i = 0; i < n; i++) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i * 2);
    }

    // Verify inputs
    std::cout << "Sanity checking purposes: " << std::endl;
    std::cout << "x[0] = " << x[0] << ", y[0] = " << y[0] << std::endl;
    std::cout << "x[1] = " << x[1] << ", y[1] = " << y[1] << std::endl;
    std::cout << "alpha = " << alpha << ", beta = " << beta << std::endl;

    axpby_cuda(x, y, alpha, beta, n);

    // Verify results
    // For i=0: y = 2.0 * 0 + 3.0 * 0 = 0
    // For i=1: y = 2.0 * 1 + 3.0 * 2 = 2 + 6 = 8
    // For i=2: y = 2.0 * 2 + 3.0 * 4 = 4 + 12 = 16
    std::cout << "\nOutput verification:" << std::endl;
    for(int i = 0; i < 5; i++) {
        float expected = alpha * static_cast<float>(i) + beta * static_cast<float>(i * 2);
        std::cout << "y[" << i << "] = " << y[i]
                  << " (expected " << expected << ")" << std::endl;
        if (std::abs(y[i] - expected) > 1e-5) {
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
