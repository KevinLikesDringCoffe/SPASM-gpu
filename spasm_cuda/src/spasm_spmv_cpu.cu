#include "../include/spasm_cuda.h"
#include <cstring>

inline uint32_t getColumnIndex_cpu(uint32_t pos) {
    return pos & 0x1FFF;
}

inline uint32_t getRowIndex_cpu(uint32_t pos) {
    return (pos >> 14) & 0x1FFF;
}

inline uint32_t getTemplateId_cpu(uint32_t pos) {
    return (pos >> 28) & 0xF;
}

void processBlock_cpu_0x000f(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

void processBlock_cpu_0x00f0(const float* values, const float* x, float* y) {
    y[1] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

void processBlock_cpu_0x0f00(const float* values, const float* x, float* y) {
    y[2] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

void processBlock_cpu_0xf000(const float* values, const float* x, float* y) {
    y[3] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

void processBlock_cpu_0x1111(const float* values, const float* x, float* y) {
    float x0 = x[0];
    y[0] += values[0] * x0;
    y[1] += values[1] * x0;
    y[2] += values[2] * x0;
    y[3] += values[3] * x0;
}

void processBlock_cpu_0x2222(const float* values, const float* x, float* y) {
    float x1 = x[1];
    y[0] += values[0] * x1;
    y[1] += values[1] * x1;
    y[2] += values[2] * x1;
    y[3] += values[3] * x1;
}

void processBlock_cpu_0x4444(const float* values, const float* x, float* y) {
    float x2 = x[2];
    y[0] += values[0] * x2;
    y[1] += values[1] * x2;
    y[2] += values[2] * x2;
    y[3] += values[3] * x2;
}

void processBlock_cpu_0x8888(const float* values, const float* x, float* y) {
    float x3 = x[3];
    y[0] += values[0] * x3;
    y[1] += values[1] * x3;
    y[2] += values[2] * x3;
    y[3] += values[3] * x3;
}

void processBlock_cpu_0x0033(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[0] + values[1] * x[1];
    y[1] += values[2] * x[0] + values[3] * x[1];
}

void processBlock_cpu_0x00cc(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[2] + values[1] * x[3];
    y[1] += values[2] * x[2] + values[3] * x[3];
}

void processBlock_cpu_0x3300(const float* values, const float* x, float* y) {
    y[2] += values[0] * x[0] + values[1] * x[1];
    y[3] += values[2] * x[0] + values[3] * x[1];
}

void processBlock_cpu_0xcc00(const float* values, const float* x, float* y) {
    y[2] += values[0] * x[2] + values[1] * x[3];
    y[3] += values[2] * x[2] + values[3] * x[3];
}

void processBlock_cpu_0x8421(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[0];
    y[1] += values[1] * x[1];
    y[2] += values[2] * x[2];
    y[3] += values[3] * x[3];
}

void processBlock_cpu_0x4218(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[3];
    y[1] += values[1] * x[0];
    y[2] += values[2] * x[1];
    y[3] += values[3] * x[2];
}

void processBlock_cpu_0x2184(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[2];
    y[1] += values[1] * x[3];
    y[2] += values[2] * x[0];
    y[3] += values[3] * x[1];
}

void processBlock_cpu_0x1842(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[1];
    y[1] += values[1] * x[2];
    y[2] += values[2] * x[3];
    y[3] += values[3] * x[0];
}

void processBlock_cpu_generic(uint16_t pattern, const float* values, const float* x, float* y) {
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

void spmvCPU(const SPASMMatrixHost& A, const std::vector<float>& x, std::vector<float>& y) {
    y.resize(A.rows);
    std::fill(y.begin(), y.end(), 0.0f);

    for (uint32_t tileIdx = 0; tileIdx < A.numTiles; tileIdx++) {
        uint32_t tileRow = A.tilePositions[tileIdx * 2];
        uint32_t tileCol = A.tilePositions[tileIdx * 2 + 1];

        uint32_t blockStart = A.tileBlockRanges[tileIdx * 2];
        uint32_t blockEnd = A.tileBlockRanges[tileIdx * 2 + 1];

        for (uint32_t posIdx = blockStart; posIdx < blockEnd; posIdx++) {
            uint32_t pos = A.positionEncodings[posIdx];

            uint32_t blockRow = getRowIndex_cpu(pos);
            uint32_t blockCol = getColumnIndex_cpu(pos);
            uint32_t templateId = getTemplateId_cpu(pos);

            uint32_t globalRow = tileRow * A.tileSize + blockRow * 4;
            uint32_t globalCol = tileCol * A.tileSize + blockCol * 4;

            float localX[4] = {0.0f};
            float localY[4] = {0.0f};

            for (int i = 0; i < 4; i++) {
                if (globalCol + i < A.cols) {
                    localX[i] = x[globalCol + i];
                }
            }

            const float* blockValues = &A.values[posIdx * 4];

            switch (templateId) {
                case 0: processBlock_cpu_0x000f(blockValues, localX, localY); break;
                case 1: processBlock_cpu_0x00f0(blockValues, localX, localY); break;
                case 2: processBlock_cpu_0x0f00(blockValues, localX, localY); break;
                case 3: processBlock_cpu_0xf000(blockValues, localX, localY); break;
                case 4: processBlock_cpu_0x1111(blockValues, localX, localY); break;
                case 5: processBlock_cpu_0x2222(blockValues, localX, localY); break;
                case 6: processBlock_cpu_0x4444(blockValues, localX, localY); break;
                case 7: processBlock_cpu_0x8888(blockValues, localX, localY); break;
                case 8: processBlock_cpu_0x0033(blockValues, localX, localY); break;
                case 9: processBlock_cpu_0x00cc(blockValues, localX, localY); break;
                case 10: processBlock_cpu_0x3300(blockValues, localX, localY); break;
                case 11: processBlock_cpu_0xcc00(blockValues, localX, localY); break;
                case 12: processBlock_cpu_0x8421(blockValues, localX, localY); break;
                case 13: processBlock_cpu_0x4218(blockValues, localX, localY); break;
                case 14: processBlock_cpu_0x2184(blockValues, localX, localY); break;
                case 15: processBlock_cpu_0x1842(blockValues, localX, localY); break;
                default:
                    if (templateId < A.numTemplates) {
                        processBlock_cpu_generic(A.templatePatterns[templateId], blockValues, localX, localY);
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
