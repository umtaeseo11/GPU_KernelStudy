#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

constexpr float SQRT_2_PI = 0.7978845608f;
constexpr float GELU_COEF = 0.044715f;

__global__ void gelu(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        float cube = val * val * val;
        float inner = SQRT_2_PI * (val + GELU_COEF * cube);
        y[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

float gelu_ref(float x) {
    float cube = x * x * x;
    float inner = SQRT_2_PI * (x + GELU_COEF * cube);
    return 0.5f * x * (1.0f + tanhf(inner));
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

    gelu<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = true;
    for (int i = 0; i < N; i += N/10) {
        float expected = gelu_ref(h_x[i]);
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
