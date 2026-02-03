#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cfloat>
#include <algorithm>

// Int8 Quantization: FP32 -> Int8 with scaling and clipping
//
// Quantization maps floating-point values to a smaller integer range [-128, 127]
// This is how models get compressed for inference (smaller memory, faster compute)
//
// The basic idea:
// 1. Find the scale factor: scale = max(|x|) / 127
// 2. Quantize: q = clamp(round(x / scale), -128, 127)
//
// This is "symmetric" quantization (zero maps to zero).
// The scale factor is computed per-tensor here (could also be per-channel).

// Step 1: find the absolute maximum (for computing scale)
__global__ void absMaxKernel(const float *x, float *block_maxes, int n) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = (idx < n) ? fabsf(x[idx]) : 0.0f;
    __syncthreads();

    // Reduction to find max in block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_maxes[blockIdx.x] = sdata[0];
    }
}

// Step 2: quantize FP32 -> Int8
__global__ void quantizeKernel(const float *x, int8_t *q, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Round to nearest, then clamp to [-128, 127]
        float val = roundf(x[idx] / scale);
        val = fmaxf(val, -128.0f);
        val = fminf(val, 127.0f);
        q[idx] = static_cast<int8_t>(val);
    }
}

void int8_quant_cuda(std::vector<float>& host_x, std::vector<int8_t>& host_q,
                     float& out_scale, int n) {
    float *d_x;
    int8_t *d_q;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_q, n * sizeof(int8_t));
    cudaMemcpy(d_x, host_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Find absolute max using reduction
    float *d_block_maxes;
    cudaMalloc(&d_block_maxes, gridSize * sizeof(float));

    absMaxKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_x, d_block_maxes, n);
    cudaDeviceSynchronize();

    // Copy block maxes to CPU and find global max
    // (could do another kernel reduction but for learning purposes CPU is fine)
    std::vector<float> block_maxes(gridSize);
    cudaMemcpy(block_maxes.data(), d_block_maxes, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    float abs_max = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        abs_max = std::max(abs_max, block_maxes[i]);
    }

    // Compute scale (avoid division by zero)
    float scale = (abs_max > 0.0f) ? abs_max / 127.0f : 1.0f;
    out_scale = scale;

    // Quantize
    quantizeKernel<<<gridSize, blockSize>>>(d_x, d_q, scale, n);
    cudaDeviceSynchronize();

    cudaMemcpy(host_q.data(), d_q, n * sizeof(int8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_q);
    cudaFree(d_block_maxes);
}

// CPU reference for verification
int8_t quantize_cpu(float x, float scale) {
    float val = roundf(x / scale);
    val = std::max(val, -128.0f);
    val = std::min(val, 127.0f);
    return static_cast<int8_t>(val);
}

int main() {
    int n = 1000000;
    std::vector<float> x(n);
    std::vector<int8_t> q(n);

    // Generate values in a realistic range (like neural net activations)
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<float>(i - n/2) / static_cast<float>(n/10);
    }

    std::cout << "Sanity checking purposes:" << std::endl;
    std::cout << "x range: [" << x[0] << ", " << x[n-1] << "]" << std::endl;
    std::cout << "x[0] = " << x[0] << std::endl;
    std::cout << "x[" << n/2 << "] = " << x[n/2] << std::endl;
    std::cout << "x[" << n-1 << "] = " << x[n-1] << std::endl;

    float scale;
    int8_quant_cuda(x, q, scale, n);

    std::cout << "\nQuantization scale = " << scale << std::endl;

    std::cout << "\nOutput verification:" << std::endl;
    bool all_good = true;

    int test_indices[] = {0, n/4, n/2, 3*n/4, n-1};
    for (int i : test_indices) {
        int8_t expected = quantize_cpu(x[i], scale);
        std::cout << "x=" << x[i] << " -> q=" << (int)q[i]
                  << " (expected " << (int)expected << ")"
                  << " dequant=" << q[i] * scale << std::endl;
        if (q[i] != expected) {
            std::cout << "  ....what is happening" << std::endl;
            all_good = false;
        }
    }

    // Verify all quantized values are in [-128, 127]
    for (int i = 0; i < n; i++) {
        if (q[i] < -128 || q[i] > 127) {
            // This technically can't happen with int8_t but good to check the logic
            std::cout << "  out of range at " << i << " ....what is happening" << std::endl;
            all_good = false;
            break;
        }
    }

    // Check quantization error
    double total_error = 0.0;
    for (int i = 0; i < n; i++) {
        float dequant = static_cast<float>(q[i]) * scale;
        total_error += std::abs(x[i] - dequant);
    }
    double mean_error = total_error / n;
    std::cout << "\nMean quantization error: " << mean_error << std::endl;

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
