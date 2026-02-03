#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// 2D Strided Load: Row-major to Column-major layout conversion (transpose-ish)
// Row-major: element (r, c) is at index r * cols + c
// Col-major: element (r, c) is at index c * rows + r
//
// This matters because memory access patterns hugely affect performance.
// Row-major is C-style, Col-major is Fortran/BLAS/cuBLAS style.
// When interfacing with libraries we often need to convert between them.

__global__ void rowToColMajor(const float *row_major, float *col_major, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int r = idx / cols;
        int c = idx % cols;
        // Read from row-major position, write to col-major position
        col_major[c * rows + r] = row_major[r * cols + c];
    }
}

__global__ void colToRowMajor(const float *col_major, float *row_major, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int r = idx / cols;
        int c = idx % cols;
        // Read from col-major position, write to row-major position
        row_major[r * cols + c] = col_major[c * rows + r];
    }
}

void layout_convert_cuda(std::vector<float>& host_in, std::vector<float>& host_out,
                         int rows, int cols, bool row_to_col) {
    int total = rows * cols;
    float *d_in, *d_out;
    cudaMalloc(&d_in, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMemcpy(d_in, host_in.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    if (row_to_col) {
        rowToColMajor<<<gridSize, blockSize>>>(d_in, d_out, rows, cols);
    } else {
        colToRowMajor<<<gridSize, blockSize>>>(d_in, d_out, rows, cols);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(host_out.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    int rows = 4;
    int cols = 3;
    int total = rows * cols;

    // Fill row-major: [[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
    std::vector<float> row_major(total);
    for (int i = 0; i < total; i++) {
        row_major[i] = static_cast<float>(i);
    }

    std::cout << "Sanity checking purposes:" << std::endl;
    std::cout << "Row-major (" << rows << "x" << cols << "):" << std::endl;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            std::cout << row_major[r * cols + c] << " ";
        }
        std::cout << std::endl;
    }

    // Convert row-major -> col-major
    std::vector<float> col_major(total);
    layout_convert_cuda(row_major, col_major, rows, cols, true);

    std::cout << "\nCol-major (stored linearly): ";
    for (int i = 0; i < total; i++) {
        std::cout << col_major[i] << " ";
    }
    std::cout << std::endl;

    // Col-major should read as columns stacked: [0,3,6,9, 1,4,7,10, 2,5,8,11]
    std::cout << "\nOutput verification (col-major reading):" << std::endl;
    bool all_good = true;
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            float val = col_major[c * rows + r];
            float expected = static_cast<float>(r * cols + c);
            if (std::abs(val - expected) > 1e-5) {
                std::cout << "  ....what is happening at (" << r << "," << c << ")" << std::endl;
                all_good = false;
            }
        }
    }

    // Now convert back: col-major -> row-major
    std::vector<float> back_to_row(total);
    layout_convert_cuda(col_major, back_to_row, rows, cols, false);

    std::cout << "Round-trip check:" << std::endl;
    for (int i = 0; i < total; i++) {
        if (std::abs(back_to_row[i] - row_major[i]) > 1e-5) {
            std::cout << "  round-trip failed at " << i << " ....what is happening" << std::endl;
            all_good = false;
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
