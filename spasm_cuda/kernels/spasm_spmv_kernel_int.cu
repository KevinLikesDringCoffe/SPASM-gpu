#include <cuda_runtime.h>
#include <cstdint>

__device__ inline uint32_t getColumnIndex_int(uint32_t pos) {
    return pos & 0x1FFF;
}

__device__ inline uint32_t getRowIndex_int(uint32_t pos) {
    return (pos >> 14) & 0x1FFF;
}

__device__ inline uint32_t getTemplateId_int(uint32_t pos) {
    return (pos >> 28) & 0xF;
}

__device__ void processBlock_int_0x000f(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

__device__ void processBlock_int_0x00f0(const int* values, const int* x, int* y) {
    y[1] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

__device__ void processBlock_int_0x0f00(const int* values, const int* x, int* y) {
    y[2] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

__device__ void processBlock_int_0xf000(const int* values, const int* x, int* y) {
    y[3] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

__device__ void processBlock_int_0x1111(const int* values, const int* x, int* y) {
    int x0 = x[0];
    y[0] += values[0] * x0;
    y[1] += values[1] * x0;
    y[2] += values[2] * x0;
    y[3] += values[3] * x0;
}

__device__ void processBlock_int_0x2222(const int* values, const int* x, int* y) {
    int x1 = x[1];
    y[0] += values[0] * x1;
    y[1] += values[1] * x1;
    y[2] += values[2] * x1;
    y[3] += values[3] * x1;
}

__device__ void processBlock_int_0x4444(const int* values, const int* x, int* y) {
    int x2 = x[2];
    y[0] += values[0] * x2;
    y[1] += values[1] * x2;
    y[2] += values[2] * x2;
    y[3] += values[3] * x2;
}

__device__ void processBlock_int_0x8888(const int* values, const int* x, int* y) {
    int x3 = x[3];
    y[0] += values[0] * x3;
    y[1] += values[1] * x3;
    y[2] += values[2] * x3;
    y[3] += values[3] * x3;
}

__device__ void processBlock_int_0x0033(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[0] + values[1] * x[1];
    y[1] += values[2] * x[0] + values[3] * x[1];
}

__device__ void processBlock_int_0x00cc(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[2] + values[1] * x[3];
    y[1] += values[2] * x[2] + values[3] * x[3];
}

__device__ void processBlock_int_0x3300(const int* values, const int* x, int* y) {
    y[2] += values[0] * x[0] + values[1] * x[1];
    y[3] += values[2] * x[0] + values[3] * x[1];
}

__device__ void processBlock_int_0xcc00(const int* values, const int* x, int* y) {
    y[2] += values[0] * x[2] + values[1] * x[3];
    y[3] += values[2] * x[2] + values[3] * x[3];
}

__device__ void processBlock_int_0x8421(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[0];
    y[1] += values[1] * x[1];
    y[2] += values[2] * x[2];
    y[3] += values[3] * x[3];
}

__device__ void processBlock_int_0x4218(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[3];
    y[1] += values[1] * x[0];
    y[2] += values[2] * x[1];
    y[3] += values[3] * x[2];
}

__device__ void processBlock_int_0x2184(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[2];
    y[1] += values[1] * x[3];
    y[2] += values[2] * x[0];
    y[3] += values[3] * x[1];
}

__device__ void processBlock_int_0x1842(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[1];
    y[1] += values[1] * x[2];
    y[2] += values[2] * x[3];
    y[3] += values[3] * x[0];
}

__device__ void processBlock_int_generic(uint16_t pattern, const int* values, const int* x, int* y) {
    int val_idx = 0;
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            int bit_pos = row * 4 + col;
            if (pattern & (1 << bit_pos)) {
                y[row] += values[val_idx] * x[col];
                val_idx++;
            }
        }
    }
}

__global__ void spmvKernel_int(
    const uint32_t* __restrict__ tilePositions,
    const uint32_t* __restrict__ tileBlockRanges,
    const uint32_t* __restrict__ positionEncodings,
    const int* __restrict__ values,
    const uint16_t* __restrict__ templatePatterns,
    const int* __restrict__ x,
    int* __restrict__ y,
    uint32_t rows,
    uint32_t cols,
    uint32_t tileSize,
    uint32_t numTiles)
{
    uint32_t tileIdx = blockIdx.x;
    if (tileIdx >= numTiles) return;

    uint32_t tileRow = tilePositions[tileIdx * 2];
    uint32_t tileCol = tilePositions[tileIdx * 2 + 1];

    uint32_t blockStart = tileBlockRanges[tileIdx * 2];
    uint32_t blockEnd = tileBlockRanges[tileIdx * 2 + 1];

    uint32_t numBlocks = blockEnd - blockStart;

    for (uint32_t blockIdx_local = threadIdx.x; blockIdx_local < numBlocks; blockIdx_local += blockDim.x) {
        uint32_t posIdx = blockStart + blockIdx_local;
        uint32_t pos = positionEncodings[posIdx];

        uint32_t blockRow = getRowIndex_int(pos);
        uint32_t blockCol = getColumnIndex_int(pos);
        uint32_t templateId = getTemplateId_int(pos);

        uint32_t globalRow = tileRow * tileSize + blockRow * 4;
        uint32_t globalCol = tileCol * tileSize + blockCol * 4;

        int localX[4];
        int localY[4] = {0, 0, 0, 0};

        for (int i = 0; i < 4; i++) {
            uint32_t colIdx = globalCol + i;
            localX[i] = (colIdx < cols) ? x[colIdx] : 0;
        }

        const int* blockValues = &values[posIdx * 4];

        switch (templateId) {
            case 0: processBlock_int_0x000f(blockValues, localX, localY); break;
            case 1: processBlock_int_0x00f0(blockValues, localX, localY); break;
            case 2: processBlock_int_0x0f00(blockValues, localX, localY); break;
            case 3: processBlock_int_0xf000(blockValues, localX, localY); break;
            case 4: processBlock_int_0x1111(blockValues, localX, localY); break;
            case 5: processBlock_int_0x2222(blockValues, localX, localY); break;
            case 6: processBlock_int_0x4444(blockValues, localX, localY); break;
            case 7: processBlock_int_0x8888(blockValues, localX, localY); break;
            case 8: processBlock_int_0x0033(blockValues, localX, localY); break;
            case 9: processBlock_int_0x00cc(blockValues, localX, localY); break;
            case 10: processBlock_int_0x3300(blockValues, localX, localY); break;
            case 11: processBlock_int_0xcc00(blockValues, localX, localY); break;
            case 12: processBlock_int_0x8421(blockValues, localX, localY); break;
            case 13: processBlock_int_0x4218(blockValues, localX, localY); break;
            case 14: processBlock_int_0x2184(blockValues, localX, localY); break;
            case 15: processBlock_int_0x1842(blockValues, localX, localY); break;
            default: processBlock_int_generic(templatePatterns[templateId], blockValues, localX, localY); break;
        }

        for (int i = 0; i < 4; i++) {
            uint32_t rowIdx = globalRow + i;
            if (rowIdx < rows && localY[i] != 0) {
                atomicAdd(&y[rowIdx], localY[i]);
            }
        }
    }
}

extern "C" void launchSPMVKernel_int(
    const uint32_t* tilePositions,
    const uint32_t* tileBlockRanges,
    const uint32_t* positionEncodings,
    const int* values,
    const uint16_t* templatePatterns,
    const int* x,
    int* y,
    uint32_t rows,
    uint32_t cols,
    uint32_t tileSize,
    uint32_t numTiles,
    uint32_t threadsPerBlock)
{
    dim3 grid(numTiles);
    dim3 block(threadsPerBlock);

    spmvKernel_int<<<grid, block>>>(
        tilePositions,
        tileBlockRanges,
        positionEncodings,
        values,
        templatePatterns,
        x,
        y,
        rows,
        cols,
        tileSize,
        numTiles
    );
}
