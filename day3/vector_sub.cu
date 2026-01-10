#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// element-wise subtraction, similar to add but with minus operator

__global__ void vectorSub(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

void vector_sub_cuda(std::vector<float>& host_a, std::vector<float>& host_b,
                     std::vector<float>& host_c, int n) {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    cudaMemcpy(d_a, host_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, host_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorSub<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(host_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    int n = 1000000;
    std::vector<float> a(n), b(n), c(n);

    // Initialize inputs
    for(int i = 0; i < n; i++) {
        a[i] = static_cast<float>(i * 3);
        b[i] = static_cast<float>(i);
    }

    // Verify inputs
    std::cout << "Sanity checking purposes: " << std::endl;
    std::cout << "a[0] = " << a[0] << ", b[0] = " << b[0] << std::endl;
    std::cout << "a[1] = " << a[1] << ", b[1] = " << b[1] << std::endl;

    vector_sub_cuda(a, b, c, n);

    // Verify results (should be 0, 2, 4, 6, 8, ...)
    std::cout << "\nOutput verification:" << std::endl;
    for(int i = 0; i < 5; i++) {
        std::cout << "c[" << i << "] = " << c[i]
                  << " (expected " << static_cast<float>(i * 2) << ")" << std::endl; // 3x - x = 2x
        if (std::abs(c[i] - static_cast<float>(i * 2)) > 1e-5) {
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
