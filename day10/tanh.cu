#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void tanh_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = tanhf(x[i]);
    }
}

int main() {
    constexpr int N = 1000000;
    constexpr int BLOCK = 256;

    std::vector<float> h_x(N), h_y(N);
    for (int i = 0; i < N; i++)
        h_x[i] = (i - N/2) / 100000.0f;

    float *d_x, *d_y;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    tanh_kernel<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = true;
    for (int i = 0; i < N; i += N/10) {
        float expected = tanhf(h_x[i]);
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
