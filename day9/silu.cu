#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void silu(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        y[i] = val * sigmoid(val);
    }
}

float silu_ref(float x) {
    return x / (1.0f + expf(-x));
}

int main() {
    constexpr int N = 1000000;
    constexpr int BLOCK = 256;

    std::vector<float> h_x(N), h_y(N);
    for (int i = 0; i < N; i++)
        h_x[i] = (i - N/2) / 50000.0f;

    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    silu<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = true;
    for (int i = 0; i < N; i += N/10) {
        float expected = silu_ref(h_x[i]);
        if (std::abs(h_y[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": " << h_y[i] << " vs " << expected << "\n";
            pass = false;
        }
    }

    std::cout << (pass ? "PASS" : "FAIL") << "\n";

    cudaFree(d_x);
    cudaFree(d_y);
    return pass ? 0 : 1;
}
