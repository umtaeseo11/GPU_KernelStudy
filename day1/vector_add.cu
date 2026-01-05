#include <cuda_runtime.h>
#include <iostream>
#include <vector>
 
//remember c ain't c++...don't get confused with syntax, look back at 392?
 
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
 
void vector_add_cuda(std::vector<float>& host_a, std::vector<float>& host_b, 
                     std::vector<float>& host_c, int n) {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    cudaMemcpy(d_a, host_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, host_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();  // Added for error checking
    cudaMemcpy(host_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
 
int main() {
    int n = 1000000; // max num of elements, should store enough but try lower
    std::vector<float> a(n), b(n), c(n);
    // Clear initialization
    for(int i = 0; i < n; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }
    // Verify inputs
    std::cout << "Sanity checking purposes: " << std::endl;
    std::cout << "a[0] = " << a[0] << ", b[0] = " << b[0] << std::endl;
    std::cout << "a[1] = " << a[1] << ", b[1] = " << b[1] << std::endl;
 
 
    std::cout << a[2] << std::endl;
    vector_add_cuda(a, b, c, n);
    // Verify results (should be 0, 3, 6, 9, 12, ...)
    std::cout << "\nOutput verification:" << std::endl;
    for(int i = 0; i < 5; i++) {
        std::cout << "c[" << i << "] = " << c[i] 
<< " (expected " << static_cast<float>(i * 3) << ")" << std::endl; // basically x + 2x = 3x
        // try writing an error checking without AI, look back at undergrad notes if needed
        if (std::abs(c[i] - static_cast<float>(i * 3)) > 1e-5) {
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
