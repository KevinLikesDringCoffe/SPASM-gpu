#include <vector>
#include <cstdint>
#include <iostream>
#include <chrono>
#include "../../spasm_converter/include/core/format.h"
#include "../../spasm_converter/include/core/types.h"
#include "../../spasm_converter/include/io/spasm_io.h"

namespace cpu_ref {

inline void process_4x4_block(uint16_t pattern, const float* values,
                               const float* x_local, float* y_local) {
    int val_idx = 0;
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            int bit_pos = row * 4 + col;
            if (pattern & (1 << bit_pos)) {
                y_local[row] += values[val_idx] * x_local[col];
                val_idx++;
            }
        }
    }
}

void spmv_spasm(const spasm::SPASMMatrix& A, const std::vector<float>& x, std::vector<float>& y) {
    std::fill(y.begin(), y.end(), 0.0f);

    std::vector<int> blockToTile(A.getNumPositions(), -1);

    if (!A.tilePositions.empty() && !A.tileBlockRanges.empty()) {
        for (size_t tileIdx = 0; tileIdx < A.tilePositions.size(); tileIdx++) {
            const auto& range = A.tileBlockRanges[tileIdx];
            for (uint32_t blockIdx = range.blockStart; blockIdx < range.blockEnd; blockIdx++) {
                if (blockIdx < blockToTile.size()) {
                    blockToTile[blockIdx] = tileIdx;
                }
            }
        }
    }

    for (size_t posIdx = 0; posIdx < A.getNumPositions(); posIdx++) {
        spasm::PositionEncoding pos = A.getPosition(posIdx);

        uint32_t blockRow = spasm::getRowIndex(pos);
        uint32_t blockCol = spasm::getColumnIndex(pos);
        uint32_t templateId = spasm::getTemplateId(pos);

        uint32_t tileRow = 0, tileCol = 0;
        int tileIdx = blockToTile[posIdx];

        if (tileIdx >= 0 && tileIdx < (int)A.tilePositions.size()) {
            tileRow = A.tilePositions[tileIdx].tileRowIdx;
            tileCol = A.tilePositions[tileIdx].tileColIdx;
        }

        uint32_t globalRow = tileRow * A.tileSize + blockRow * 4;
        uint32_t globalCol = tileCol * A.tileSize + blockCol * 4;

        const float* blockValues = &A.values[posIdx * 4];

        float localX[4] = {0};
        float localY[4] = {0};

        for (int i = 0; i < 4; i++) {
            if (globalCol + i < x.size()) {
                localX[i] = x[globalCol + i];
            }
        }

        if (templateId < A.templatePatterns.size()) {
            uint16_t pattern = A.templatePatterns[templateId].mask;
            process_4x4_block(pattern, blockValues, localX, localY);
        }

        for (int i = 0; i < 4; i++) {
            if (globalRow + i < y.size()) {
                y[globalRow + i] += localY[i];
            }
        }
    }
}

void spmv_spasm_timed(const spasm::SPASMMatrix& A, const std::vector<float>& x,
                      std::vector<float>& y, int num_runs, double& avg_time_ms) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_runs; i++) {
        spmv_spasm(A, x, y);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    avg_time_ms = elapsed.count() / num_runs;
}

}
