#include "../include/spasm_cuda.h"
#include <cstring>
#include <iostream>

static inline uint32_t getColumnIndex_int(uint32_t pos) {
    return pos & 0x1FFF;
}

static inline uint32_t getRowIndex_int(uint32_t pos) {
    return (pos >> 14) & 0x1FFF;
}

static inline uint32_t getTemplateId_int(uint32_t pos) {
    return (pos >> 28) & 0xF;
}

static void processBlock_int_0x000f(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

static void processBlock_int_0x00f0(const int* values, const int* x, int* y) {
    y[1] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

static void processBlock_int_0x0f00(const int* values, const int* x, int* y) {
    y[2] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

static void processBlock_int_0xf000(const int* values, const int* x, int* y) {
    y[3] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

static void processBlock_int_0x1111(const int* values, const int* x, int* y) {
    int x0 = x[0];
    y[0] += values[0] * x0;
    y[1] += values[1] * x0;
    y[2] += values[2] * x0;
    y[3] += values[3] * x0;
}

static void processBlock_int_0x2222(const int* values, const int* x, int* y) {
    int x1 = x[1];
    y[0] += values[0] * x1;
    y[1] += values[1] * x1;
    y[2] += values[2] * x1;
    y[3] += values[3] * x1;
}

static void processBlock_int_0x4444(const int* values, const int* x, int* y) {
    int x2 = x[2];
    y[0] += values[0] * x2;
    y[1] += values[1] * x2;
    y[2] += values[2] * x2;
    y[3] += values[3] * x2;
}

static void processBlock_int_0x8888(const int* values, const int* x, int* y) {
    int x3 = x[3];
    y[0] += values[0] * x3;
    y[1] += values[1] * x3;
    y[2] += values[2] * x3;
    y[3] += values[3] * x3;
}

static void processBlock_int_0x0033(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[0] + values[1] * x[1];
    y[1] += values[2] * x[0] + values[3] * x[1];
}

static void processBlock_int_0x00cc(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[2] + values[1] * x[3];
    y[1] += values[2] * x[2] + values[3] * x[3];
}

static void processBlock_int_0x3300(const int* values, const int* x, int* y) {
    y[2] += values[0] * x[0] + values[1] * x[1];
    y[3] += values[2] * x[0] + values[3] * x[1];
}

static void processBlock_int_0xcc00(const int* values, const int* x, int* y) {
    y[2] += values[0] * x[2] + values[1] * x[3];
    y[3] += values[2] * x[2] + values[3] * x[3];
}

static void processBlock_int_0x8421(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[0];
    y[1] += values[1] * x[1];
    y[2] += values[2] * x[2];
    y[3] += values[3] * x[3];
}

static void processBlock_int_0x4218(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[3];
    y[1] += values[1] * x[0];
    y[2] += values[2] * x[1];
    y[3] += values[3] * x[2];
}

static void processBlock_int_0x2184(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[2];
    y[1] += values[1] * x[3];
    y[2] += values[2] * x[0];
    y[3] += values[3] * x[1];
}

static void processBlock_int_0x1842(const int* values, const int* x, int* y) {
    y[0] += values[0] * x[1];
    y[1] += values[1] * x[2];
    y[2] += values[2] * x[3];
    y[3] += values[3] * x[0];
}

static void processBlock_int_generic(uint16_t pattern, const int* values, const int* x, int* y) {
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

void spmvCPU_int(const SPASMMatrixHost& A, const std::vector<int>& values_int,
                 const std::vector<int>& x, std::vector<int>& y) {
    y.resize(A.rows);
    std::fill(y.begin(), y.end(), 0);

    for (uint32_t tileIdx = 0; tileIdx < A.numTiles; tileIdx++) {
        uint32_t tileRow = A.tilePositions[tileIdx * 2];
        uint32_t tileCol = A.tilePositions[tileIdx * 2 + 1];

        uint32_t blockStart = A.tileBlockRanges[tileIdx * 2];
        uint32_t blockEnd = A.tileBlockRanges[tileIdx * 2 + 1];

        for (uint32_t posIdx = blockStart; posIdx < blockEnd; posIdx++) {
            uint32_t pos = A.positionEncodings[posIdx];

            uint32_t blockRow = getRowIndex_int(pos);
            uint32_t blockCol = getColumnIndex_int(pos);
            uint32_t templateId = getTemplateId_int(pos);

            uint32_t globalRow = tileRow * A.tileSize + blockRow * 4;
            uint32_t globalCol = tileCol * A.tileSize + blockCol * 4;

            int localX[4] = {0};
            int localY[4] = {0};

            for (int i = 0; i < 4; i++) {
                if (globalCol + i < A.cols) {
                    localX[i] = x[globalCol + i];
                }
            }

            const int* blockValues = &values_int[posIdx * 4];

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
                default:
                    if (templateId < A.numTemplates) {
                        processBlock_int_generic(A.templatePatterns[templateId], blockValues, localX, localY);
                    }
                    break;
            }

            for (int i = 0; i < 4; i++) {
                if (globalRow + i < A.rows) {
                    y[globalRow + i] += localY[i];
                }
            }
        }
    }
}
