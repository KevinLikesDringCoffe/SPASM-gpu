#include <cuda_runtime.h>
#include <cstdint>

__constant__ uint16_t c_templateMasks[16];

__device__ inline void process_pattern_0x000f(const float* vals, const float* x, float* y) {
    y[0] += vals[0] * x[0] + vals[1] * x[1] + vals[2] * x[2] + vals[3] * x[3];
}

__device__ inline void process_pattern_0x00f0(const float* vals, const float* x, float* y) {
    y[1] += vals[0] * x[0] + vals[1] * x[1] + vals[2] * x[2] + vals[3] * x[3];
}

__device__ inline void process_pattern_0x0f00(const float* vals, const float* x, float* y) {
    y[2] += vals[0] * x[0] + vals[1] * x[1] + vals[2] * x[2] + vals[3] * x[3];
}

__device__ inline void process_pattern_0xf000(const float* vals, const float* x, float* y) {
    y[3] += vals[0] * x[0] + vals[1] * x[1] + vals[2] * x[2] + vals[3] * x[3];
}

__device__ inline void process_pattern_0x1111(const float* vals, const float* x, float* y) {
    float x0 = x[0];
    y[0] += vals[0] * x0;
    y[1] += vals[1] * x0;
    y[2] += vals[2] * x0;
    y[3] += vals[3] * x0;
}

__device__ inline void process_pattern_0x2222(const float* vals, const float* x, float* y) {
    float x1 = x[1];
    y[0] += vals[0] * x1;
    y[1] += vals[1] * x1;
    y[2] += vals[2] * x1;
    y[3] += vals[3] * x1;
}

__device__ inline void process_pattern_0x4444(const float* vals, const float* x, float* y) {
    float x2 = x[2];
    y[0] += vals[0] * x2;
    y[1] += vals[1] * x2;
    y[2] += vals[2] * x2;
    y[3] += vals[3] * x2;
}

__device__ inline void process_pattern_0x8888(const float* vals, const float* x, float* y) {
    float x3 = x[3];
    y[0] += vals[0] * x3;
    y[1] += vals[1] * x3;
    y[2] += vals[2] * x3;
    y[3] += vals[3] * x3;
}

__device__ inline void process_pattern_0x0033(const float* vals, const float* x, float* y) {
    y[0] += vals[0] * x[0] + vals[1] * x[1];
    y[1] += vals[2] * x[0] + vals[3] * x[1];
}

__device__ inline void process_pattern_0x00cc(const float* vals, const float* x, float* y) {
    y[0] += vals[0] * x[2] + vals[1] * x[3];
    y[1] += vals[2] * x[2] + vals[3] * x[3];
}

__device__ inline void process_pattern_0x3300(const float* vals, const float* x, float* y) {
    y[2] += vals[0] * x[0] + vals[1] * x[1];
    y[3] += vals[2] * x[0] + vals[3] * x[1];
}

__device__ inline void process_pattern_0xcc00(const float* vals, const float* x, float* y) {
    y[2] += vals[0] * x[2] + vals[1] * x[3];
    y[3] += vals[2] * x[2] + vals[3] * x[3];
}

__device__ inline void process_pattern_0x8421(const float* vals, const float* x, float* y) {
    y[0] += vals[0] * x[0];
    y[1] += vals[1] * x[1];
    y[2] += vals[2] * x[2];
    y[3] += vals[3] * x[3];
}

__device__ inline void process_pattern_0x4218(const float* vals, const float* x, float* y) {
    y[0] += vals[0] * x[3];
    y[1] += vals[1] * x[0];
    y[2] += vals[2] * x[1];
    y[3] += vals[3] * x[2];
}

__device__ inline void process_pattern_0x2184(const float* vals, const float* x, float* y) {
    y[0] += vals[0] * x[2];
    y[1] += vals[1] * x[3];
    y[2] += vals[2] * x[0];
    y[3] += vals[3] * x[1];
}

__device__ inline void process_pattern_0x1842(const float* vals, const float* x, float* y) {
    y[0] += vals[0] * x[1];
    y[1] += vals[1] * x[2];
    y[2] += vals[2] * x[3];
    y[3] += vals[3] * x[0];
}

__device__ inline void process_pattern_generic(uint16_t pattern, const float* vals, const float* x, float* y) {
    int val_idx = 0;
    #pragma unroll
    for (int row = 0; row < 4; row++) {
        #pragma unroll
        for (int col = 0; col < 4; col++) {
            int bit_pos = row * 4 + col;
            if (pattern & (1 << bit_pos)) {
                y[row] += vals[val_idx] * x[col];
                val_idx++;
            }
        }
    }
}

__device__ inline void process_4x4_block(uint32_t templateId, const float* vals, const float* x, float* y) {
    uint16_t pattern = c_templateMasks[templateId];

    switch(pattern) {
        case 0x000F: process_pattern_0x000f(vals, x, y); break;
        case 0x00F0: process_pattern_0x00f0(vals, x, y); break;
        case 0x0F00: process_pattern_0x0f00(vals, x, y); break;
        case 0xF000: process_pattern_0xf000(vals, x, y); break;
        case 0x1111: process_pattern_0x1111(vals, x, y); break;
        case 0x2222: process_pattern_0x2222(vals, x, y); break;
        case 0x4444: process_pattern_0x4444(vals, x, y); break;
        case 0x8888: process_pattern_0x8888(vals, x, y); break;
        case 0x0033: process_pattern_0x0033(vals, x, y); break;
        case 0x00CC: process_pattern_0x00cc(vals, x, y); break;
        case 0x3300: process_pattern_0x3300(vals, x, y); break;
        case 0xCC00: process_pattern_0xcc00(vals, x, y); break;
        case 0x8421: process_pattern_0x8421(vals, x, y); break;
        case 0x4218: process_pattern_0x4218(vals, x, y); break;
        case 0x2184: process_pattern_0x2184(vals, x, y); break;
        case 0x1842: process_pattern_0x1842(vals, x, y); break;
        default: process_pattern_generic(pattern, vals, x, y); break;
    }
}

__global__ void spasm_spmv_kernel(
    const uint32_t* __restrict__ tilePositions,
    const uint32_t* __restrict__ tileBlockRanges,
    const uint32_t* __restrict__ positionEncodings,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t numTiles,
    uint32_t tileSize,
    uint32_t maxCols)
{
    uint32_t tileIdx = blockIdx.x;
    if (tileIdx >= numTiles) return;

    uint32_t tileRow = tilePositions[tileIdx * 2];
    uint32_t tileCol = tilePositions[tileIdx * 2 + 1];
    uint32_t blockStart = tileBlockRanges[tileIdx * 2];
    uint32_t blockEnd = tileBlockRanges[tileIdx * 2 + 1];

    extern __shared__ float smem[];
    float* s_x = smem;

    uint32_t tileColStart = tileCol * tileSize;
    uint32_t tileColEnd = min(tileColStart + tileSize, maxCols);

    for (uint32_t i = threadIdx.x; i < (tileColEnd - tileColStart); i += blockDim.x) {
        s_x[i] = x[tileColStart + i];
    }
    __syncthreads();

    for (uint32_t posIdx = blockStart + threadIdx.x; posIdx < blockEnd; posIdx += blockDim.x) {
        uint32_t pos = positionEncodings[posIdx];

        uint32_t c_idx = pos & 0x1FFF;
        uint32_t r_idx = (pos >> 14) & 0x1FFF;
        uint32_t t_id = (pos >> 28) & 0xF;

        uint32_t globalRow = tileRow * tileSize + r_idx * 4;
        uint32_t localCol = c_idx * 4;

        const float* blockValues = &values[posIdx * 4];

        float localX[4];
        float localY[4] = {0, 0, 0, 0};

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (localCol + i < (tileColEnd - tileColStart)) {
                localX[i] = s_x[localCol + i];
            } else {
                localX[i] = 0.0f;
            }
        }

        process_4x4_block(t_id, blockValues, localX, localY);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (localY[i] != 0.0f) {
                atomicAdd(&y[globalRow + i], localY[i]);
            }
        }
    }
}

void launchSpasmSpmvKernel(
    const uint32_t* d_tilePositions,
    const uint32_t* d_tileBlockRanges,
    const uint32_t* d_positionEncodings,
    const float* d_values,
    const float* d_x,
    float* d_y,
    uint32_t numTiles,
    uint32_t tileSize,
    uint32_t maxCols)
{
    int blockSize = 256;
    int gridSize = numTiles;
    size_t sharedMemSize = tileSize * sizeof(float);

    spasm_spmv_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_tilePositions,
        d_tileBlockRanges,
        d_positionEncodings,
        d_values,
        d_x,
        d_y,
        numTiles,
        tileSize,
        maxCols
    );
}

void copyTemplateMasksToConstant(const uint16_t* h_masks, uint32_t numMasks) {
    cudaMemcpyToSymbol(c_templateMasks, h_masks, numMasks * sizeof(uint16_t));
}
