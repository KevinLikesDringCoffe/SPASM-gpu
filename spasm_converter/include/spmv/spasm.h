#ifndef SPASM_SPMV_SPASM_H
#define SPASM_SPMV_SPASM_H

#include <vector>
#include <cstdint>
#include <functional>
#include <array>
#include "../core/format.h"
#include "../core/template_patterns.h"

namespace spmv {

// Function pointer type for block SpMV operations
using BlockSpMVFunc = std::function<void(const float*, const float*, float*)>;

// Block SpMV implementations for each template pattern
// Each function computes: y_block += A_block * x_block
// where A_block is a 4x4 block represented by the template pattern and values

// Template 0: 0x000f - Row 0: [* * * *]
inline void blockSpMV_0x000f(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

// Template 1: 0x00f0 - Row 1: [* * * *]
inline void blockSpMV_0x00f0(const float* values, const float* x, float* y) {
    y[1] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

// Template 2: 0x0f00 - Row 2: [* * * *]
inline void blockSpMV_0x0f00(const float* values, const float* x, float* y) {
    y[2] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

// Template 3: 0xf000 - Row 3: [* * * *]
inline void blockSpMV_0xf000(const float* values, const float* x, float* y) {
    y[3] += values[0] * x[0] + values[1] * x[1] + values[2] * x[2] + values[3] * x[3];
}

// Template 4: 0x1111 - Column 0: [* * * *]^T
inline void blockSpMV_0x1111(const float* values, const float* x, float* y) {
    float x0 = x[0];
    y[0] += values[0] * x0;
    y[1] += values[1] * x0;
    y[2] += values[2] * x0;
    y[3] += values[3] * x0;
}

// Template 5: 0x2222 - Column 1: [* * * *]^T
inline void blockSpMV_0x2222(const float* values, const float* x, float* y) {
    float x1 = x[1];
    y[0] += values[0] * x1;
    y[1] += values[1] * x1;
    y[2] += values[2] * x1;
    y[3] += values[3] * x1;
}

// Template 6: 0x4444 - Column 2: [* * * *]^T
inline void blockSpMV_0x4444(const float* values, const float* x, float* y) {
    float x2 = x[2];
    y[0] += values[0] * x2;
    y[1] += values[1] * x2;
    y[2] += values[2] * x2;
    y[3] += values[3] * x2;
}

// Template 7: 0x8888 - Column 3: [* * * *]^T
inline void blockSpMV_0x8888(const float* values, const float* x, float* y) {
    float x3 = x[3];
    y[0] += values[0] * x3;
    y[1] += values[1] * x3;
    y[2] += values[2] * x3;
    y[3] += values[3] * x3;
}

// Template 8: 0x0033 - 2x2 block top-left
inline void blockSpMV_0x0033(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[0] + values[1] * x[1];
    y[1] += values[2] * x[0] + values[3] * x[1];
}

// Template 9: 0x00cc - 2x2 block top-right
inline void blockSpMV_0x00cc(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[2] + values[1] * x[3];
    y[1] += values[2] * x[2] + values[3] * x[3];
}

// Template 10: 0x3300 - 2x2 block bottom-left
inline void blockSpMV_0x3300(const float* values, const float* x, float* y) {
    y[2] += values[0] * x[0] + values[1] * x[1];
    y[3] += values[2] * x[0] + values[3] * x[1];
}

// Template 11: 0xcc00 - 2x2 block bottom-right
inline void blockSpMV_0xcc00(const float* values, const float* x, float* y) {
    y[2] += values[0] * x[2] + values[1] * x[3];
    y[3] += values[2] * x[2] + values[3] * x[3];
}

// Template 12: 0x8421 - Main diagonal
inline void blockSpMV_0x8421(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[0];
    y[1] += values[1] * x[1];
    y[2] += values[2] * x[2];
    y[3] += values[3] * x[3];
}

// Template 13: 0x4218 - Anti-diagonal wrap
inline void blockSpMV_0x4218(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[3];
    y[1] += values[1] * x[0];
    y[2] += values[2] * x[1];
    y[3] += values[3] * x[2];
}

// Template 14: 0x2184 - Anti-diagonal wrap variant
inline void blockSpMV_0x2184(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[2];
    y[1] += values[1] * x[3];
    y[2] += values[2] * x[0];
    y[3] += values[3] * x[1];
}

// Template 15: 0x1842 - Anti-diagonal wrap variant
inline void blockSpMV_0x1842(const float* values, const float* x, float* y) {
    y[0] += values[0] * x[1];
    y[1] += values[1] * x[2];
    y[2] += values[2] * x[3];
    y[3] += values[3] * x[0];
}

// Generic block SpMV for arbitrary patterns (fallback)
inline void blockSpMV_generic(spasm::PatternMask pattern, const float* values, const float* x, float* y) {
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

// SPASM SpMV implementation
class SPASMSpMV {
private:
    // Lookup table for block SpMV functions
    std::array<BlockSpMVFunc, 16> blockSpMVFuncs;

    // Initialize the lookup table
    void initBlockSpMVFuncs() {
        blockSpMVFuncs[0] = [](const float* v, const float* x, float* y) { blockSpMV_0x000f(v, x, y); };
        blockSpMVFuncs[1] = [](const float* v, const float* x, float* y) { blockSpMV_0x00f0(v, x, y); };
        blockSpMVFuncs[2] = [](const float* v, const float* x, float* y) { blockSpMV_0x0f00(v, x, y); };
        blockSpMVFuncs[3] = [](const float* v, const float* x, float* y) { blockSpMV_0xf000(v, x, y); };
        blockSpMVFuncs[4] = [](const float* v, const float* x, float* y) { blockSpMV_0x1111(v, x, y); };
        blockSpMVFuncs[5] = [](const float* v, const float* x, float* y) { blockSpMV_0x2222(v, x, y); };
        blockSpMVFuncs[6] = [](const float* v, const float* x, float* y) { blockSpMV_0x4444(v, x, y); };
        blockSpMVFuncs[7] = [](const float* v, const float* x, float* y) { blockSpMV_0x8888(v, x, y); };
        blockSpMVFuncs[8] = [](const float* v, const float* x, float* y) { blockSpMV_0x0033(v, x, y); };
        blockSpMVFuncs[9] = [](const float* v, const float* x, float* y) { blockSpMV_0x00cc(v, x, y); };
        blockSpMVFuncs[10] = [](const float* v, const float* x, float* y) { blockSpMV_0x3300(v, x, y); };
        blockSpMVFuncs[11] = [](const float* v, const float* x, float* y) { blockSpMV_0xcc00(v, x, y); };
        blockSpMVFuncs[12] = [](const float* v, const float* x, float* y) { blockSpMV_0x8421(v, x, y); };
        blockSpMVFuncs[13] = [](const float* v, const float* x, float* y) { blockSpMV_0x4218(v, x, y); };
        blockSpMVFuncs[14] = [](const float* v, const float* x, float* y) { blockSpMV_0x2184(v, x, y); };
        blockSpMVFuncs[15] = [](const float* v, const float* x, float* y) { blockSpMV_0x1842(v, x, y); };
    }

public:
    SPASMSpMV() {
        initBlockSpMVFuncs();
    }

    // Perform SpMV: y = A * x
    void spmv(const spasm::SPASMMatrix& A, const std::vector<float>& x, std::vector<float>& y) {
        // Initialize output vector to zero
        std::fill(y.begin(), y.end(), 0.0f);

        // Build tile index map for efficient lookup
        std::vector<int> blockToTile(A.getNumPositions(), -1);

        if (!A.tilePositions.empty() && !A.tileBlockRanges.empty()) {
            // Map each block to its tile
            for (size_t tileIdx = 0; tileIdx < A.tilePositions.size(); tileIdx++) {
                const auto& range = A.tileBlockRanges[tileIdx];
                for (uint32_t blockIdx = range.blockStart; blockIdx < range.blockEnd; blockIdx++) {
                    if (blockIdx < blockToTile.size()) {
                        blockToTile[blockIdx] = tileIdx;
                    }
                }
            }
        }

        // Process all blocks
        for (size_t posIdx = 0; posIdx < A.getNumPositions(); posIdx++) {
            spasm::PositionEncoding pos = A.getPosition(posIdx);

            // Extract position information
            uint32_t blockRow = spasm::getRowIndex(pos);
            uint32_t blockCol = spasm::getColumnIndex(pos);
            uint32_t templateId = spasm::getTemplateId(pos);

            // Determine tile position
            uint32_t tileRow = 0, tileCol = 0;
            int tileIdx = blockToTile[posIdx];

            if (tileIdx >= 0 && tileIdx < (int)A.tilePositions.size()) {
                // Block belongs to a tile
                tileRow = A.tilePositions[tileIdx].tileRowIdx;
                tileCol = A.tilePositions[tileIdx].tileColIdx;
            }
            // else: tileRow = 0, tileCol = 0 (for blocks not in any tile)

            // Calculate global indices for the 4x4 block
            uint32_t globalRow = tileRow * A.tileSize + blockRow * 4;
            uint32_t globalCol = tileCol * A.tileSize + blockCol * 4;

            // Get values for this block (4 values)
            const float* blockValues = &A.values[posIdx * 4];

            // Prepare local input and output vectors
            float localX[4] = {0};
            float localY[4] = {0};

            // Copy input vector values (with bounds checking)
            for (int i = 0; i < 4; i++) {
                if (globalCol + i < x.size()) {
                    localX[i] = x[globalCol + i];
                }
            }

            // Perform block SpMV using template-specific function
            if (templateId < 16 && templateId < A.templatePatterns.size()) {
                blockSpMVFuncs[templateId](blockValues, localX, localY);
            } else {
                // Fallback to generic implementation
                spasm::PatternMask pattern = A.templatePatterns[templateId].mask;
                blockSpMV_generic(pattern, blockValues, localX, localY);
            }

            // Accumulate results to global output vector (with bounds checking)
            for (int i = 0; i < 4; i++) {
                if (globalRow + i < y.size()) {
                    y[globalRow + i] += localY[i];
                }
            }
        }
    }
};

} // namespace spmv

#endif // SPASM_SPMV_SPASM_H