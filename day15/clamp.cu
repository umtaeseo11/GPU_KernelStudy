#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Clamp/Clip: limit each element x to be within [min_val, max_val]
// clamp(x) = min(max(x, min_val), max_val)
//
// Super common in neural nets - gradient clipping, value clamping for stability, etc.
// Element-wise so this is straightforward like the activation functions

__global__ void clampKernel(const float *x, float *y, float min_val, float max_val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        // clamp: if val < min_val use min_val, if val > max_val use max_val, else val
        val = fmaxf(val, min_val);
        val = fminf(val, max_val);
        y[idx] = val;
    }
}

void clamp_cuda(std::vector<float>& host_x, std::vector<float>& host_y,
                float min_val, float max_val, int n) {
    float *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_x, host_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    clampKernel<<<gridSize, blockSize>>>(d_x, d_y, min_val, max_val, n);
    cudaDeviceSynchronize();

    cudaMemcpy(host_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}

float clamp_cpu(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

int main() {
    int n = 1000000;
    std::vector<float> x(n), y(n);

    // Range from -10 to 10
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<float>(i - n/2) / static_cast<float>(n/20);
    }

    float min_val = -2.0f;
    float max_val = 3.0f;

    std::cout << "Sanity checking purposes:" << std::endl;
    std::cout << "Clamping to [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "x[0] = " << x[0] << " (below min)" << std::endl;
    std::cout << "x[" << n/2 << "] = " << x[n/2] << " (near zero)" << std::endl;
    std::cout << "x[" << n-1 << "] = " << x[n-1] << " (above max)" << std::endl;

    clamp_cuda(x, y, min_val, max_val, n);

    std::cout << "\nOutput verification:" << std::endl;
    bool all_good = true;

    int test_indices[] = {0, n/4, n/2, 3*n/4, n-1};
    for (int i : test_indices) {
        float expected = clamp_cpu(x[i], min_val, max_val);
        std::cout << "clamp(" << x[i] << ") = " << y[i]
                  << " (expected " << expected << ")" << std::endl;
        if (std::abs(y[i] - expected) > 1e-5) {
            std::cout << "  ....what is happening" << std::endl;
            all_good = false;
        }
    }

    // Check that ALL values are within bounds
    for (int i = 0; i < n; i++) {
        if (y[i] < min_val - 1e-5 || y[i] > max_val + 1e-5) {
            std::cout << "  out of bounds at " << i << ": " << y[i]
                      << " ....what is happening" << std::endl;
            all_good = false;
            break;
        }
    }

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
