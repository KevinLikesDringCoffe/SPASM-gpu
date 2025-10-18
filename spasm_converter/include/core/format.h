#ifndef SPASM_CORE_FORMAT_H
#define SPASM_CORE_FORMAT_H

#include <vector>
#include <cstdint>
#include <cassert>
#include "types.h"

namespace spasm {

// SPASM sparse matrix format
// Data layout according to the paper:
// - Global composition: tile positions in COO format
// - Local patterns: continuous position encodings followed by continuous values
class SPASMMatrix {
public:
    // ========== Global Composition (Tile Level) ==========
    std::vector<TilePosition> tilePositions;  // COO format for non-empty tiles
    std::vector<TileBlockRange> tileBlockRanges;  // Block data ranges for each tile
    uint32_t tileSize;                        // Size of each tile (e.g., 1024)

    // ========== Local Patterns (Continuous Storage) ==========
    // Position encodings - stored continuously
    // Each uint32_t contains: c_idx(13) | CE(1) | r_idx(13) | RE(1) | t_id(4)
    std::vector<PositionEncoding> positionEncodings;

    // Values - stored continuously
    // Every 4 values correspond to one position encoding
    std::vector<float> values;

    // ========== Template Patterns ==========
    std::vector<TemplatePattern> templatePatterns;  // Up to 16 templates

    // ========== Matrix Metadata ==========
    uint32_t rows;           // Original matrix rows
    uint32_t cols;           // Original matrix columns
    uint32_t nnz;            // Total non-zeros (including padding)
    uint32_t originalNnz;    // Original non-zeros (before padding)
    uint32_t numPaddings;    // Number of zero paddings added

    // ========== Constructors ==========
    SPASMMatrix() : tileSize(1024), rows(0), cols(0), nnz(0), originalNnz(0), numPaddings(0) {}

    SPASMMatrix(uint32_t r, uint32_t c, uint32_t tile_sz = 1024)
        : tileSize(tile_sz), rows(r), cols(c), nnz(0), originalNnz(0), numPaddings(0) {}

    // ========== Data Access Methods ==========

    // Get position encoding at index
    PositionEncoding getPosition(size_t index) const {
        assert(index < positionEncodings.size());
        return positionEncodings[index];
    }

    // Get values for a position encoding at index (returns 4 values)
    std::vector<float> getValuesForPosition(size_t posIndex) const {
        assert(posIndex < positionEncodings.size());
        size_t valueStart = posIndex * VALUES_PER_POSITION;
        assert(valueStart + VALUES_PER_POSITION <= values.size());

        return std::vector<float>(
            values.begin() + valueStart,
            values.begin() + valueStart + VALUES_PER_POSITION
        );
    }

    // Add a position encoding with its 4 values
    void addPositionWithValues(PositionEncoding pos, const float vals[4]) {
        positionEncodings.push_back(pos);
        for (int i = 0; i < VALUES_PER_POSITION; i++) {
            values.push_back(vals[i]);
        }
    }

    // Get block range for a specific tile index
    TileBlockRange getTileBlockRange(size_t tileIndex) const {
        assert(tileIndex < tileBlockRanges.size());
        return tileBlockRanges[tileIndex];
    }

    // Get all position encodings for a specific tile
    std::vector<PositionEncoding> getTilePositionEncodings(size_t tileIndex) const {
        assert(tileIndex < tileBlockRanges.size());
        const auto& range = tileBlockRanges[tileIndex];
        return std::vector<PositionEncoding>(
            positionEncodings.begin() + range.blockStart,
            positionEncodings.begin() + range.blockEnd
        );
    }

    // Get all values for a specific tile
    std::vector<float> getTileValues(size_t tileIndex) const {
        assert(tileIndex < tileBlockRanges.size());
        const auto& range = tileBlockRanges[tileIndex];
        size_t valueStart = range.blockStart * VALUES_PER_POSITION;
        size_t valueEnd = range.blockEnd * VALUES_PER_POSITION;
        return std::vector<float>(
            values.begin() + valueStart,
            values.begin() + valueEnd
        );
    }

    // ========== Statistics Methods ==========

    // Get storage size in bytes
    size_t getStorageSizeBytes() const {
        size_t size = 0;

        // Global composition (tile positions)
        size += tilePositions.size() * sizeof(TilePosition);

        // Tile block ranges
        size += tileBlockRanges.size() * sizeof(TileBlockRange);

        // Position encodings (continuous storage)
        size += positionEncodings.size() * sizeof(PositionEncoding);

        // Values (continuous storage)
        size += values.size() * sizeof(float);

        // Template patterns
        size += templatePatterns.size() * sizeof(PatternMask);

        // Metadata
        size += sizeof(tileSize) + sizeof(rows) + sizeof(cols) + sizeof(nnz);

        return size;
    }

    // Get average bytes per non-zero
    float getBytesPerNonZero() const {
        if (originalNnz == 0) return 0.0f;
        return static_cast<float>(getStorageSizeBytes()) / static_cast<float>(originalNnz);
    }

    // Get compression ratio compared to COO format
    float getCompressionRatio() const {
        if (originalNnz == 0) return 0.0f;
        // COO format: (row_idx + col_idx + value) per non-zero
        size_t coo_size = originalNnz * (2 * sizeof(uint32_t) + sizeof(float));
        size_t spasm_size = getStorageSizeBytes();
        return static_cast<float>(coo_size) / static_cast<float>(spasm_size);
    }

    // Get padding rate
    float getPaddingRate() const {
        if (nnz == 0) return 0.0f;
        return static_cast<float>(numPaddings) / static_cast<float>(nnz);
    }

    // ========== Validation ==========

    bool validate() const {
        // Check template patterns count
        if (templatePatterns.size() > MAX_TEMPLATES) {
            return false;
        }

        // Check that values count matches position encodings
        if (values.size() != positionEncodings.size() * VALUES_PER_POSITION) {
            return false;
        }

        // Check tile size is multiple of submatrix size
        if (tileSize == 0 || tileSize % SUBMATRIX_SIZE != 0) {
            return false;
        }

        // Validate position encodings
        for (const auto& pos : positionEncodings) {
            uint32_t t_id = getTemplateId(pos);
            if (t_id >= templatePatterns.size()) {
                return false;
            }
        }

        return true;
    }

    // Clear all data
    void clear() {
        tilePositions.clear();
        tileBlockRanges.clear();
        positionEncodings.clear();
        values.clear();
        templatePatterns.clear();
        rows = cols = nnz = originalNnz = numPaddings = 0;
    }

    // Get number of position encodings
    size_t getNumPositions() const {
        return positionEncodings.size();
    }

    // Get number of tiles
    size_t getNumTiles() const {
        return tilePositions.size();
    }
};

} // namespace spasm

#endif // SPASM_CORE_FORMAT_H