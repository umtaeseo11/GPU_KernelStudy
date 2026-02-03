#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>

// Dropout (RNG): generate the dropout mask on-the-fly on the GPU
// Instead of passing in a pre-computed mask, each thread generates its own
// random number using cuRAND's Philox RNG.
//
// Philox is nice for GPU because:
// - Each thread can independently generate random numbers given (seed, sequence, offset)
// - No shared state between threads = perfect parallelism
// - Deterministic given the same seed (good for reproducibility)
//
// This is how real frameworks do dropout - way faster than generating mask on CPU
// and transferring it over PCIe

__global__ void dropoutRngKernel(const float *x, float *y, float p, float scale,
                                 unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Initialize Philox RNG state for this thread
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);

        // Generate uniform random number in (0, 1]
        float rand_val = curand_uniform(&state);

        // Drop if rand_val < p, keep and scale otherwise
        y[idx] = (rand_val >= p) ? x[idx] * scale : 0.0f;
    }
}

void dropout_rng_cuda(std::vector<float>& host_x, std::vector<float>& host_y,
                      float p, unsigned long long seed, int n) {
    float *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_x, host_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    float scale = 1.0f / (1.0f - p);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    dropoutRngKernel<<<gridSize, blockSize>>>(d_x, d_y, p, scale, seed, n);
    cudaDeviceSynchronize();

    cudaMemcpy(host_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int n = 1000000;
    float p = 0.3f;
    unsigned long long seed = 42;

    std::vector<float> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f; // all ones makes it easy to check scaling
    }

    std::cout << "Sanity checking purposes:" << std::endl;
    std::cout << "dropout p = " << p << ", seed = " << seed << std::endl;
    std::cout << "All inputs are 1.0" << std::endl;

    dropout_rng_cuda(x, y, p, seed, n);

    // Count zeros and non-zeros
    int zero_count = 0;
    int nonzero_count = 0;
    float scale = 1.0f / (1.0f - p);

    for (int i = 0; i < n; i++) {
        if (y[i] == 0.0f) {
            zero_count++;
        } else {
            nonzero_count++;
            // Non-zero elements should be scaled by 1/(1-p)
            if (std::abs(y[i] - scale) > 1e-4) {
                std::cout << "  bad scaling at " << i << ": " << y[i]
                          << " expected " << scale << " ....what is happening" << std::endl;
                break;
            }
        }
    }

    float actual_drop_rate = static_cast<float>(zero_count) / n;
    std::cout << "\nOutput verification:" << std::endl;
    std::cout << "Zeros (dropped): " << zero_count << " (" << actual_drop_rate * 100 << "%)" << std::endl;
    std::cout << "Non-zeros (kept): " << nonzero_count << std::endl;
    std::cout << "Expected drop rate: " << p * 100 << "%" << std::endl;
    std::cout << "Scale factor: " << scale << std::endl;

    bool all_good = true;

    // Drop rate should be roughly p (within 1% for 1M elements)
    if (std::abs(actual_drop_rate - p) > 0.01f) {
        std::cout << "  drop rate too far off ....what is happening" << std::endl;
        all_good = false;
    }

    // Determinism check: same seed should give same result
    std::vector<float> y2(n);
    dropout_rng_cuda(x, y2, p, seed, n);
    bool deterministic = true;
    for (int i = 0; i < n; i++) {
        if (y[i] != y2[i]) {
            deterministic = false;
            break;
        }
    }
    std::cout << "Deterministic (same seed): " << (deterministic ? "yes" : "no") << std::endl;
    if (!deterministic) {
        std::cout << "  ....what is happening (Philox should be deterministic!)" << std::endl;
        all_good = false;
    }

    // Different seed should give different result
    std::vector<float> y3(n);
    dropout_rng_cuda(x, y3, p, seed + 1, n);
    bool different = false;
    for (int i = 0; i < 100; i++) {
        if (y[i] != y3[i]) {
            different = true;
            break;
        }
    }
    std::cout << "Different seed gives different output: " << (different ? "yes" : "no") << std::endl;

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
