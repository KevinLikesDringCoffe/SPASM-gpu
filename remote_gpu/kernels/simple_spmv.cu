#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

__global__ void csr_spmv_kernel(
    const int num_rows,
    const float* values,
    const int* col_indices,
    const int* row_ptr,
    const float* x,
    float* y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int i = row_start; i < row_end; i++) {
            sum += values[i] * x[col_indices[i]];
        }

        y[row] = sum;
    }
}

}
