#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// Dropout (Fixed Mask): zero out elements based on a pre-generated binary mask
// During training, dropout randomly sets some activations to 0 to prevent overfitting.
// The remaining activations get scaled by 1/(1-p) to keep expected values the same.
//
// "Fixed" means the mask is generated on CPU and passed in (as opposed to day17
// where we generate it on-the-fly with RNG on the GPU).
//
// dropout(x, mask) = mask[i] ? x[i] / (1 - p) : 0

__global__ void dropoutKernel(const float *x, const int *mask, float *y,
                              float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // mask[i] = 1 means keep, 0 means drop
        y[idx] = mask[idx] ? x[idx] * scale : 0.0f;
    }
}

void dropout_cuda(std::vector<float>& host_x, std::vector<int>& host_mask,
                  std::vector<float>& host_y, float p, int n) {
    float *d_x, *d_y;
    int *d_mask;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_mask, n * sizeof(int));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMemcpy(d_x, host_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, host_mask.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    float scale = 1.0f / (1.0f - p); // inverted dropout scaling

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    dropoutKernel<<<gridSize, blockSize>>>(d_x, d_mask, d_y, scale, n);
    cudaDeviceSynchronize();

    cudaMemcpy(host_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_mask);
    cudaFree(d_y);
}

int main() {
    int n = 1000000;
    float p = 0.3f; // dropout probability (30% of elements get dropped)
    std::vector<float> x(n), y(n);
    std::vector<int> mask(n);

    srand(42); // fixed seed for reproducibility

    for (int i = 0; i < n; i++) {
        x[i] = static_cast<float>(i + 1) / 100.0f; // positive values
    }

    // Generate mask on CPU: 1 = keep, 0 = drop
    int drop_count = 0;
    for (int i = 0; i < n; i++) {
        mask[i] = (static_cast<float>(rand()) / RAND_MAX) >= p ? 1 : 0;
        if (mask[i] == 0) drop_count++;
    }

    std::cout << "Sanity checking purposes:" << std::endl;
    std::cout << "dropout p = " << p << std::endl;
    std::cout << "Elements dropped: " << drop_count << " / " << n
              << " (" << 100.0f * drop_count / n << "%)" << std::endl;
    std::cout << "x[0] = " << x[0] << ", mask[0] = " << mask[0] << std::endl;

    dropout_cuda(x, mask, y, p, n);

    float scale = 1.0f / (1.0f - p);

    std::cout << "\nOutput verification:" << std::endl;
    bool all_good = true;

    for (int i = 0; i < 10; i++) {
        float expected = mask[i] ? x[i] * scale : 0.0f;
        std::cout << "x=" << x[i] << " mask=" << mask[i]
                  << " -> y=" << y[i] << " (expected " << expected << ")" << std::endl;
        if (std::abs(y[i] - expected) > 1e-4) {
            std::cout << "  ....what is happening" << std::endl;
            all_good = false;
        }
    }

    // Verify all dropped elements are exactly 0
    for (int i = 0; i < n; i++) {
        if (mask[i] == 0 && y[i] != 0.0f) {
            std::cout << "  dropped element not zero at " << i << " ....what is happening" << std::endl;
            all_good = false;
            break;
        }
    }

    // Check that kept elements are scaled correctly
    for (int i = 0; i < n; i++) {
        if (mask[i] == 1) {
            float expected = x[i] * scale;
            if (std::abs(y[i] - expected) > 1e-4) {
                std::cout << "  scaling wrong at " << i << " ....what is happening" << std::endl;
                all_good = false;
                break;
            }
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
