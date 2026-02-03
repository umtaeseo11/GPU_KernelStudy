#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// L2 Norm: sqrt(sum(x^2))
// Also known as the Euclidean norm / magnitude of a vector.
// Similar to mean (day13) - it's a reduction but we square each element first.
//
// Steps:
// 1. Each thread squares its element
// 2. Block-level reduction using shared memory
// 3. atomicAdd partial sums to global
// 4. sqrt on CPU (or a final kernel, but overkill here)

__global__ void sumSquaresKernel(const float *x, float *sum_sq, int n) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load squared values into shared memory
    sdata[tid] = (idx < n) ? x[idx] * x[idx] : 0.0f;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_sq, sdata[0]);
    }
}

float l2_norm_cuda(std::vector<float>& host_x, int n) {
    float *d_x, *d_sum_sq;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_sum_sq, sizeof(float));
    cudaMemcpy(d_x, host_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum_sq, 0, sizeof(float));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    sumSquaresKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_x, d_sum_sq, n);
    cudaDeviceSynchronize();

    float sum_sq;
    cudaMemcpy(&sum_sq, d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_sum_sq);

    return sqrtf(sum_sq);
}

// CPU reference
float l2_norm_cpu(std::vector<float>& x, int n) {
    double sum = 0.0; // use double for better precision on CPU side
    for (int i = 0; i < n; i++) {
        sum += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    return static_cast<float>(sqrt(sum));
}

int main() {
    int n = 1000000;
    std::vector<float> x(n);

    // Use small-ish values to avoid float precision blowup
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<float>(i) / static_cast<float>(n);
    }

    std::cout << "Sanity checking purposes:" << std::endl;
    std::cout << "x[0] = " << x[0] << std::endl;
    std::cout << "x[" << n/2 << "] = " << x[n/2] << std::endl;
    std::cout << "x[" << n-1 << "] = " << x[n-1] << std::endl;

    float gpu_norm = l2_norm_cuda(x, n);
    float cpu_norm = l2_norm_cpu(x, n);

    std::cout << "\nOutput verification:" << std::endl;
    std::cout << "GPU L2 norm = " << gpu_norm << std::endl;
    std::cout << "CPU L2 norm = " << cpu_norm << std::endl;

    bool all_good = true;
    // Float atomicAdd accumulation loses precision, so allow some tolerance
    float rel_error = std::abs(gpu_norm - cpu_norm) / cpu_norm;
    std::cout << "Relative error = " << rel_error << std::endl;
    if (rel_error > 0.01f) {
        std::cout << "  ....what is happening" << std::endl;
        all_good = false;
    }

    // Quick sanity: norm of [1, 0, 0, ...] should be 1
    std::vector<float> unit(100, 0.0f);
    unit[0] = 1.0f;
    float unit_norm = l2_norm_cuda(unit, 100);
    std::cout << "\nUnit vector norm = " << unit_norm << " (expected 1.0)" << std::endl;
    if (std::abs(unit_norm - 1.0f) > 1e-5) {
        std::cout << "  ....what is happening" << std::endl;
        all_good = false;
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
