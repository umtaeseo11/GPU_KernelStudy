#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>

// Bitwise Ops for low-bit quantization
// In quantization we often need to pack/unpack values into fewer bits.
// For example, packing two 4-bit values into a single byte (int8),
// or extracting bit fields from packed representations.
//
// Key operations:
// - Bit shifting (<<, >>): move bits left/right
// - Bitwise AND (&): mask out bits
// - Bitwise OR (|): combine bit fields
//
// Use case: pack two 4-bit integers (0-15) into one uint8_t
// Upper nibble: value1 << 4
// Lower nibble: value2 & 0x0F
// Combined: (value1 << 4) | (value2 & 0x0F)

// Pack two 4-bit values into one byte
__global__ void packInt4(const uint8_t *a, const uint8_t *b, uint8_t *packed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // a goes in upper 4 bits, b goes in lower 4 bits
        packed[idx] = (a[idx] << 4) | (b[idx] & 0x0F);
    }
}

// Unpack one byte back into two 4-bit values
__global__ void unpackInt4(const uint8_t *packed, uint8_t *a, uint8_t *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = (packed[idx] >> 4) & 0x0F; // upper nibble
        b[idx] = packed[idx] & 0x0F;         // lower nibble
    }
}

// Pack four 2-bit values into one byte
__global__ void packInt2(const uint8_t *vals, uint8_t *packed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Pack 4 values from vals[idx*4 .. idx*4+3] into packed[idx]
        int base = idx * 4;
        packed[idx] = ((vals[base]     & 0x03) << 6) |
                      ((vals[base + 1] & 0x03) << 4) |
                      ((vals[base + 2] & 0x03) << 2) |
                      ((vals[base + 3] & 0x03));
    }
}

// Unpack one byte into four 2-bit values
__global__ void unpackInt2(const uint8_t *packed, uint8_t *vals, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int base = idx * 4;
        vals[base]     = (packed[idx] >> 6) & 0x03;
        vals[base + 1] = (packed[idx] >> 4) & 0x03;
        vals[base + 2] = (packed[idx] >> 2) & 0x03;
        vals[base + 3] = packed[idx] & 0x03;
    }
}

int main() {
    int n = 1000000;

    // === Test 4-bit packing ===
    std::cout << "=== INT4 Pack/Unpack ===" << std::endl;
    {
        std::vector<uint8_t> a(n), b(n), packed(n), a_out(n), b_out(n);
        for (int i = 0; i < n; i++) {
            a[i] = i % 16;       // 0-15 fits in 4 bits
            b[i] = (i * 3) % 16;
        }

        uint8_t *d_a, *d_b, *d_packed, *d_a_out, *d_b_out;
        cudaMalloc(&d_a, n); cudaMalloc(&d_b, n);
        cudaMalloc(&d_packed, n);
        cudaMalloc(&d_a_out, n); cudaMalloc(&d_b_out, n);
        cudaMemcpy(d_a, a.data(), n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        packInt4<<<gridSize, blockSize>>>(d_a, d_b, d_packed, n);
        cudaDeviceSynchronize();
        unpackInt4<<<gridSize, blockSize>>>(d_packed, d_a_out, d_b_out, n);
        cudaDeviceSynchronize();

        cudaMemcpy(a_out.data(), d_a_out, n, cudaMemcpyDeviceToHost);
        cudaMemcpy(b_out.data(), d_b_out, n, cudaMemcpyDeviceToHost);

        std::cout << "Sanity checking purposes:" << std::endl;
        bool all_good = true;
        for (int i = 0; i < 5; i++) {
            std::cout << "a=" << (int)a[i] << " b=" << (int)b[i]
                      << " -> unpacked a=" << (int)a_out[i] << " b=" << (int)b_out[i] << std::endl;
        }
        for (int i = 0; i < n; i++) {
            if (a_out[i] != a[i] || b_out[i] != b[i]) {
                std::cout << "  int4 round-trip failed at " << i << " ....what is happening" << std::endl;
                all_good = false;
                break;
            }
        }
        if (all_good) std::cout << "INT4 round-trip: passed" << std::endl;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_packed);
        cudaFree(d_a_out); cudaFree(d_b_out);
    }

    // === Test 2-bit packing ===
    std::cout << "\n=== INT2 Pack/Unpack ===" << std::endl;
    {
        int num_vals = n * 4; // 4 values per packed byte
        int num_packed = n;
        std::vector<uint8_t> vals(num_vals), packed(num_packed), vals_out(num_vals);

        for (int i = 0; i < num_vals; i++) {
            vals[i] = i % 4; // 0-3 fits in 2 bits
        }

        uint8_t *d_vals, *d_packed, *d_vals_out;
        cudaMalloc(&d_vals, num_vals);
        cudaMalloc(&d_packed, num_packed);
        cudaMalloc(&d_vals_out, num_vals);
        cudaMemcpy(d_vals, vals.data(), num_vals, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int gridSize = (num_packed + blockSize - 1) / blockSize;

        packInt2<<<gridSize, blockSize>>>(d_vals, d_packed, num_packed);
        cudaDeviceSynchronize();
        unpackInt2<<<gridSize, blockSize>>>(d_packed, d_vals_out, num_packed);
        cudaDeviceSynchronize();

        cudaMemcpy(vals_out.data(), d_vals_out, num_vals, cudaMemcpyDeviceToHost);

        std::cout << "Sanity checking purposes:" << std::endl;
        bool all_good = true;
        for (int i = 0; i < 8; i++) {
            std::cout << "val=" << (int)vals[i] << " -> unpacked=" << (int)vals_out[i] << std::endl;
        }
        for (int i = 0; i < num_vals; i++) {
            if (vals_out[i] != vals[i]) {
                std::cout << "  int2 round-trip failed at " << i << " ....what is happening" << std::endl;
                all_good = false;
                break;
            }
        }
        if (all_good) std::cout << "INT2 round-trip: passed" << std::endl;

        cudaFree(d_vals); cudaFree(d_packed); cudaFree(d_vals_out);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "\nGood I guess" << std::endl;
    return 0;
}
