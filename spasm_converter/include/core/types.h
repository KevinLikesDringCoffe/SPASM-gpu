#ifndef SPASM_CORE_TYPES_H
#define SPASM_CORE_TYPES_H

#include <cstdint>
#include <vector>

namespace spasm {

// Template pattern representation (16-bit bitmask for 4x4 pattern)
// Each bit represents whether a position has a non-zero value
// Bit mapping: bit_i corresponds to position (i/4, i%4) in 4x4 submatrix
using PatternMask = uint16_t;

// Position encoding packed into single uint32_t
// According to the paper Figure 5:
// Bits 0-12:  c_idx (13 bits) - Column index within tile
// Bit 13:     CE (1 bit) - Column end flag
// Bits 14-26: r_idx (13 bits) - Row index within tile
// Bit 27:     RE (1 bit) - Row end flag
// Bits 28-31: t_id (4 bits) - Template pattern ID
using PositionEncoding = uint32_t;

// Helper functions for position encoding
inline PositionEncoding packPosition(uint32_t c_idx, bool CE, uint32_t r_idx, bool RE, uint32_t t_id) {
    return (c_idx & 0x1FFF) |           // bits 0-12
           ((CE ? 1 : 0) << 13) |        // bit 13
           ((r_idx & 0x1FFF) << 14) |    // bits 14-26
           ((RE ? 1 : 0) << 27) |        // bit 27
           ((t_id & 0xF) << 28);         // bits 28-31
}

inline uint32_t getColumnIndex(PositionEncoding pos) {
    return pos & 0x1FFF;
}

inline bool getColumnEnd(PositionEncoding pos) {
    return (pos >> 13) & 1;
}

inline uint32_t getRowIndex(PositionEncoding pos) {
    return (pos >> 14) & 0x1FFF;
}

inline bool getRowEnd(PositionEncoding pos) {
    return (pos >> 27) & 1;
}

inline uint32_t getTemplateId(PositionEncoding pos) {
    return (pos >> 28) & 0xF;
}

// Tile block range - stores the start and end position of block data for a tile
struct TileBlockRange {
    uint32_t blockStart;    // Starting index in positionEncodings array
    uint32_t blockEnd;      // Ending index (exclusive) in positionEncodings array

    TileBlockRange() : blockStart(0), blockEnd(0) {}
    TileBlockRange(uint32_t start, uint32_t end) : blockStart(start), blockEnd(end) {}

    // Get number of blocks in this tile
    uint32_t getNumBlocks() const {
        return blockEnd - blockStart;
    }
};

// Global composition - tile-level COO format
struct TilePosition {
    uint32_t tileRowIdx;
    uint32_t tileColIdx;

    TilePosition() : tileRowIdx(0), tileColIdx(0) {}
    TilePosition(uint32_t row, uint32_t col) : tileRowIdx(row), tileColIdx(col) {}

    // Comparison operator for sorting
    bool operator<(const TilePosition& other) const {
        if (tileRowIdx != other.tileRowIdx) return tileRowIdx < other.tileRowIdx;
        return tileColIdx < other.tileColIdx;
    }

    bool operator==(const TilePosition& other) const {
        return tileRowIdx == other.tileRowIdx && tileColIdx == other.tileColIdx;
    }
};

// Template pattern definition
struct TemplatePattern {
    PatternMask mask;           // 16-bit pattern mask
    std::string description;    // Optional description

    TemplatePattern() : mask(0) {}
    TemplatePattern(PatternMask m, const std::string& desc = "")
        : mask(m), description(desc) {}

    // Get number of non-zero elements in pattern
    int getNumNonZeros() const {
        return __builtin_popcount(mask);
    }

    // Check if position (row, col) is non-zero in 4x4 pattern
    bool isNonZero(int row, int col) const {
        if (row < 0 || row >= 4 || col < 0 || col >= 4) return false;
        int bit_pos = row * 4 + col;
        return (mask >> bit_pos) & 1;
    }
};

// Constants based on paper
constexpr int SUBMATRIX_SIZE = 4;      // 4x4 submatrices
constexpr int MAX_TEMPLATES = 16;      // Maximum 16 template patterns
constexpr int VALUES_PER_POSITION = 4; // 4 values share one position encoding

} // namespace spasm

#endif // SPASM_CORE_TYPES_H