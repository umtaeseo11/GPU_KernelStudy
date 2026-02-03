#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// Vector Mean: compute the average of a 1D array
// This is a reduction problem - need to sum everything then divide by n.
// Using atomicAdd for the global sum since we haven't done shared memory reductions yet.
// (that's coming in phase 2)
//
// Two-step approach:
// 1. Kernel to compute partial sums (each thread adds its element atomically)
// 2. Divide by n on CPU (or could do a second kernel but why)

__global__ void sumKernel(const float *x, float *sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(sum, x[idx]);
    }
}

// Slightly better version: each block does a local sum first using shared memory
// then atomicAdd only once per block (less contention)
__global__ void sumKernelShared(const float *x, float *sum, int n) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load into shared memory
    sdata[tid] = (idx < n) ? x[idx] : 0.0f;
    __syncthreads();

    // Reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block does the atomic add
    if (tid == 0) {
        atomicAdd(sum, sdata[0]);
    }
}

float vector_mean_cuda(std::vector<float>& host_x, int n, bool use_shared) {
    float *d_x, *d_sum;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemcpy(d_x, host_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float)); // zero out the sum

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    if (use_shared) {
        sumKernelShared<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_x, d_sum, n);
    } else {
        sumKernel<<<gridSize, blockSize>>>(d_x, d_sum, n);
    }
    cudaDeviceSynchronize();

    float total_sum;
    cudaMemcpy(&total_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_sum);

    return total_sum / static_cast<float>(n);
}

int main() {
    int n = 1000000;
    std::vector<float> x(n);

    // Fill with known values: 0, 1, 2, ..., n-1
    // Mean should be (n-1)/2 = 499999.5
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<float>(i);
    }

    std::cout << "Sanity checking purposes:" << std::endl;
    std::cout << "x[0] = " << x[0] << ", x[" << n-1 << "] = " << x[n-1] << std::endl;

    float expected_mean = static_cast<float>(n - 1) / 2.0f;

    // Test naive atomicAdd version
    float mean_naive = vector_mean_cuda(x, n, false);
    std::cout << "\nOutput verification (naive atomicAdd):" << std::endl;
    std::cout << "mean = " << mean_naive << " (expected " << expected_mean << ")" << std::endl;

    // Test shared memory version
    float mean_shared = vector_mean_cuda(x, n, true);
    std::cout << "\nOutput verification (shared memory reduction):" << std::endl;
    std::cout << "mean = " << mean_shared << " (expected " << expected_mean << ")" << std::endl;

    bool all_good = true;
    // atomicAdd on floats has some precision issues with large sums, allow more tolerance
    if (std::abs(mean_naive - expected_mean) > 1.0f) {
        std::cout << "  naive: ....what is happening" << std::endl;
        all_good = false;
    }
    if (std::abs(mean_shared - expected_mean) > 1.0f) {
        std::cout << "  shared: ....what is happening" << std::endl;
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
