#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void vectorScale(float *a, float *b, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = a[idx] * scale;
    }
}

void vector_scale_cuda(std::vector<float>& host_a, std::vector<float>& host_b,
                       float scale, int n) {
    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMemcpy(d_a, host_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorScale<<<gridSize, blockSize>>>(d_a, d_b, scale, n);
    cudaDeviceSynchronize();

    cudaMemcpy(host_b.data(), d_b, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
}

int main() {
    int n = 1000000;
    float scale = 2.5f;
    std::vector<float> a(n), b(n);

    // Initialize input
    for(int i = 0; i < n; i++) {
        a[i] = static_cast<float>(i);
    }

    // Verify inputs
    std::cout << "Sanity checking purposes: " << std::endl;
    std::cout << "a[0] = " << a[0] << ", scale = " << scale << std::endl;
    std::cout << "a[1] = " << a[1] << ", scale = " << scale << std::endl;

    vector_scale_cuda(a, b, scale, n);

    // Verify results
    std::cout << "\nOutput verification:" << std::endl;
    for(int i = 0; i < 5; i++) {
        std::cout << "b[" << i << "] = " << b[i]
                  << " (expected " << static_cast<float>(i) * scale << ")" << std::endl;
        if (std::abs(b[i] - static_cast<float>(i) * scale) > 1e-5) {
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
