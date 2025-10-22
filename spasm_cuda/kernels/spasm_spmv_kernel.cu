#include <cuda_runtime.h>
#include <cstdint>

__device__ inline uint32_t getColumnIndex(uint32_t pos) {
    return pos & 0x1FFF;
}

__device__ inline uint32_t getRowIndex(uint32_t pos) {
    return (pos >> 14) & 0x1FFF;
}

__device__ inline uint32_t getTemplateId(uint32_t pos) {
    return (pos >> 28) & 0xF;
}

__device__ void processBlock_0x000f(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

__device__ void processBlock_0x00f0(const float* values, const float* x, float* y) {
    y[1] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

__device__ void processBlock_0x0f00(const float* values, const float* x, float* y) {
    y[2] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

__device__ void processBlock_0xf000(const float* values, const float* x, float* y) {
    y[3] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

__device__ void processBlock_0x1111(const float* values, const float* x, float* y) {
    float x0 = x[0];
    y[0] += values[0] * x0;
    y[1] += values[1] * x0;
    y[2] += values[2] * x0;
    y[3] += values[3] * x0;
}

__device__ void processBlock_0x2222(const float* values, const float* x, float* y) {
    float x1 = x[1];
    y[0] += values[0] * x1;
    y[1] += values[1] * x1;
    y[2] += values[2] * x1;
    y[3] += values[3] * x1;
}

__device__ void processBlock_0x4444(const float* values, const float* x, float* y) {
    float x2 = x[2];
    y[0] += values[0] * x2;
    y[1] += values[1] * x2;
    y[2] += values[2] * x2;
    y[3] += values[3] * x2;
}

__device__ void processBlock_0x8888(const float* values, const float* x, float* y) {
    float x3 = x[3];
    y[0] += values[0] * x3;
    y[1] += values[1] * x3;
    y[2] += values[2] * x3;
    y[3] += values[3] * x3;
}

__device__ void processBlock_0x0033(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[0] + values[1] * x[1];
    y[1] += values[2] * x[0] + values[3] * x[1];
}

__device__ void processBlock_0x00cc(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[2] + values[1] * x[3];
    y[1] += values[2] * x[2] + values[3] * x[3];
}

__device__ void processBlock_0x3300(const float* values, const float* x, float* y) {
    y[2] += values[0] * x[0] + values[1] * x[1];
    y[3] += values[2] * x[0] + values[3] * x[1];
}

__device__ void processBlock_0xcc00(const float* values, const float* x, float* y) {
    y[2] += values[0] * x[2] + values[1] * x[3];
    y[3] += values[2] * x[2] + values[3] * x[3];
}

__device__ void processBlock_0x8421(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[0];
    y[1] += values[1] * x[1];
    y[2] += values[2] * x[2];
    y[3] += values[3] * x[3];
}

__device__ void processBlock_0x4218(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[3];
    y[1] += values[1] * x[0];
    y[2] += values[2] * x[1];
    y[3] += values[3] * x[2];
}

__device__ void processBlock_0x2184(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[2];
    y[1] += values[1] * x[3];
    y[2] += values[2] * x[0];
    y[3] += values[3] * x[1];
}

__device__ void processBlock_0x1842(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[1];
    y[1] += values[1] * x[2];
    y[2] += values[2] * x[3];
    y[3] += values[3] * x[0];
}

__device__ void processBlock_generic(uint16_t pattern, const float* values, const float* x, float* y) {
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

__global__ void spmvKernel(
    const uint32_t* __restrict__ tilePositions,
    const uint32_t* __restrict__ tileBlockRanges,
    const uint32_t* __restrict__ positionEncodings,
    const float* __restrict__ values,
    const uint16_t* __restrict__ templatePatterns,
    const float* __restrict__ x,
    float* __restrict__ y,
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

        uint32_t blockRow = getRowIndex(pos);
        uint32_t blockCol = getColumnIndex(pos);
        uint32_t templateId = getTemplateId(pos);

        uint32_t globalRow = tileRow * tileSize + blockRow * 4;
        uint32_t globalCol = tileCol * tileSize + blockCol * 4;

        float localX[4];
        float localY[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int i = 0; i < 4; i++) {
            uint32_t colIdx = globalCol + i;
            localX[i] = (colIdx < cols) ? x[colIdx] : 0.0f;
        }

        const float* blockValues = &values[posIdx * 4];

        switch (templateId) {
            case 0: processBlock_0x000f(blockValues, localX, localY); break;
            case 1: processBlock_0x00f0(blockValues, localX, localY); break;
            case 2: processBlock_0x0f00(blockValues, localX, localY); break;
            case 3: processBlock_0xf000(blockValues, localX, localY); break;
            case 4: processBlock_0x1111(blockValues, localX, localY); break;
            case 5: processBlock_0x2222(blockValues, localX, localY); break;
            case 6: processBlock_0x4444(blockValues, localX, localY); break;
            case 7: processBlock_0x8888(blockValues, localX, localY); break;
            case 8: processBlock_0x0033(blockValues, localX, localY); break;
            case 9: processBlock_0x00cc(blockValues, localX, localY); break;
            case 10: processBlock_0x3300(blockValues, localX, localY); break;
            case 11: processBlock_0xcc00(blockValues, localX, localY); break;
            case 12: processBlock_0x8421(blockValues, localX, localY); break;
            case 13: processBlock_0x4218(blockValues, localX, localY); break;
            case 14: processBlock_0x2184(blockValues, localX, localY); break;
            case 15: processBlock_0x1842(blockValues, localX, localY); break;
            default: processBlock_generic(templatePatterns[templateId], blockValues, localX, localY); break;
        }

        for (int i = 0; i < 4; i++) {
            uint32_t rowIdx = globalRow + i;
            if (rowIdx < rows) {
                atomicAdd(&y[rowIdx], localY[i]);
            }
        }
    }
}

extern "C" void launchSPMVKernel(
    const uint32_t* tilePositions,
    const uint32_t* tileBlockRanges,
    const uint32_t* positionEncodings,
    const float* values,
    const uint16_t* templatePatterns,
    const float* x,
    float* y,
    uint32_t rows,
    uint32_t cols,
    uint32_t tileSize,
    uint32_t numTiles,
    uint32_t maxBlocksPerTile)
{
    dim3 grid(numTiles);
    dim3 block(256);

    spmvKernel<<<grid, block>>>(
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
