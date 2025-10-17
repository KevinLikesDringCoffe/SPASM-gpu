#ifndef OPTIMIZED_CONVERTER_V2_H
#define OPTIMIZED_CONVERTER_V2_H

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include "spasm_format.h"
#include "cached_decomposer.h"
#include "mtx_reader.h"

namespace spasm {

// Optimized converter v2 with integrated pattern analysis and cached decomposition
class OptimizedConverterV2 {
public:
    // Extended block entry with value information
    struct BlockEntryWithValue {
        uint32_t tileRow;      // Tile row index
        uint32_t tileCol;      // Tile column index
        uint32_t blockRow;     // 4x4 block row within tile
        uint32_t blockCol;     // 4x4 block column within tile
        uint32_t localRow;     // Position within 4x4 block (0-3)
        uint32_t localCol;     // Position within 4x4 block (0-3)
        float value;

        // Comparison for sorting
        bool operator<(const BlockEntryWithValue& other) const {
            // Sort by tile first
            if (tileRow != other.tileRow) return tileRow < other.tileRow;
            if (tileCol != other.tileCol) return tileCol < other.tileCol;
            // Then by block within tile
            if (blockRow != other.blockRow) return blockRow < other.blockRow;
            if (blockCol != other.blockCol) return blockCol < other.blockCol;
            // Then by position within block
            if (localRow != other.localRow) return localRow < other.localRow;
            return localCol < other.localCol;
        }

        uint64_t getTileId() const {
            return ((uint64_t)tileRow << 32) | tileCol;
        }

        uint64_t getBlockId() const {
            return ((uint64_t)tileRow << 48) | ((uint64_t)tileCol << 32) |
                   ((uint64_t)blockRow << 16) | blockCol;
        }
    };

    // Combined pattern analysis and conversion result
    struct AnalysisAndConversionResult {
        std::vector<PatternFrequency> patterns;
        uint32_t uniquePatternCount;
        uint32_t totalPatternOccurrences;
    };

    // Optimized matrix conversion with integrated pattern analysis
    static AnalysisAndConversionResult convertAndAnalyze(
        SPASMMatrix& spasm,
        const COOMatrix& input,
        const std::vector<PatternMask>& templates,
        uint32_t tileSize,
        bool verbose = false) {

        AnalysisAndConversionResult result;
        uint32_t submatricesPerTile = tileSize / SUBMATRIX_SIZE;

        // Step 1: Convert all entries to block format and sort (single pass)
        if (verbose) {
            std::cout << "  Sorting entries by blocks...\n";
        }

        std::vector<BlockEntryWithValue> sortedEntries;
        sortedEntries.reserve(input.entries.size());

        for (const auto& entry : input.entries) {
            BlockEntryWithValue be;

            // Calculate tile indices
            be.tileRow = entry.row / tileSize;
            be.tileCol = entry.col / tileSize;

            // Calculate position within tile
            uint32_t rowInTile = entry.row % tileSize;
            uint32_t colInTile = entry.col % tileSize;

            // Calculate 4x4 block within tile
            be.blockRow = rowInTile / SUBMATRIX_SIZE;
            be.blockCol = colInTile / SUBMATRIX_SIZE;

            // Calculate position within 4x4 block
            be.localRow = rowInTile % SUBMATRIX_SIZE;
            be.localCol = colInTile % SUBMATRIX_SIZE;

            be.value = entry.value;
            sortedEntries.push_back(be);
        }

        // Sort entries - this is the key optimization
        std::sort(sortedEntries.begin(), sortedEntries.end());

        // Step 2: Process sorted entries linearly for both pattern analysis and conversion
        if (sortedEntries.empty()) {
            return result;
        }

        // Pattern counting for analysis
        std::unordered_map<PatternMask, uint32_t> patternCount;

        // Track current tile and block
        uint64_t currentTileId = sortedEntries[0].getTileId();
        uint64_t currentBlockId = sortedEntries[0].getBlockId();

        // Current block data
        PatternMask currentPattern = 0;
        float currentValues[16] = {0};
        uint32_t currentBlockRow = sortedEntries[0].blockRow;
        uint32_t currentBlockCol = sortedEntries[0].blockCol;

        // Track unique tiles
        std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> uniqueTiles;

        // Clear decomposition cache for new conversion
        CachedPatternDecomposer::clearCache();

        if (verbose) {
            std::cout << "  Processing " << sortedEntries.size() << " entries...\n";
        }

        int blocksProcessed = 0;

        for (const auto& entry : sortedEntries) {
            uint64_t tileId = entry.getTileId();
            uint64_t blockId = entry.getBlockId();

            // Check if we moved to a new block
            if (blockId != currentBlockId) {
                // Process the previous block
                if (currentPattern != 0) {
                    // Count pattern for analysis
                    patternCount[currentPattern]++;

                    // Convert block to SPASM format
                    processBlock(spasm, currentPattern, currentValues, templates,
                               currentBlockRow, currentBlockCol, submatricesPerTile);
                    blocksProcessed++;
                }

                // Check if we moved to a new tile
                if (tileId != currentTileId) {
                    // Record this tile
                    uniqueTiles[tileId] = {entry.tileRow, entry.tileCol};
                    currentTileId = tileId;
                }

                // Start new block
                currentBlockId = blockId;
                currentPattern = 0;
                std::fill(currentValues, currentValues + 16, 0.0f);
                currentBlockRow = entry.blockRow;
                currentBlockCol = entry.blockCol;
            }

            // Add this entry to current block
            int bitPos = entry.localRow * SUBMATRIX_SIZE + entry.localCol;
            currentPattern |= (1 << bitPos);
            currentValues[bitPos] = entry.value;
        }

        // Don't forget the last block
        if (currentPattern != 0) {
            patternCount[currentPattern]++;
            processBlock(spasm, currentPattern, currentValues, templates,
                       currentBlockRow, currentBlockCol, submatricesPerTile);
            blocksProcessed++;
        }

        // Step 3: Add tile positions to SPASM matrix
        for (const auto& [tileId, tilePos] : uniqueTiles) {
            spasm.tilePositions.emplace_back(tilePos.first, tilePos.second);
        }

        // Sort tile positions for better access pattern
        std::sort(spasm.tilePositions.begin(), spasm.tilePositions.end());

        // Step 4: Convert pattern counts to sorted vector for analysis
        result.patterns.reserve(patternCount.size());
        result.uniquePatternCount = patternCount.size();
        result.totalPatternOccurrences = 0;

        for (const auto& [pattern, count] : patternCount) {
            result.patterns.push_back({pattern, count});
            result.totalPatternOccurrences += count;
        }

        // Sort by frequency (descending)
        std::sort(result.patterns.begin(), result.patterns.end(), std::greater<PatternFrequency>());

        if (verbose) {
            std::cout << "  Processed " << uniqueTiles.size() << " non-empty tiles\n";
            std::cout << "  Total blocks processed: " << blocksProcessed << "\n";
            std::cout << "  Unique patterns found: " << result.uniquePatternCount << "\n";
            std::cout << "  Cache size: " << CachedPatternDecomposer::getCacheSize() << " entries\n";
        }

        return result;
    }

    // Visualize top-k patterns (no additional sorting needed)
    static void visualizeTopPatterns(const AnalysisAndConversionResult& analysisResult,
                                    int topK = 8,
                                    bool verbose = false) {

        const auto& patterns = analysisResult.patterns;
        uint32_t totalOccurrences = analysisResult.totalPatternOccurrences;

        std::cout << "\nTop-" << topK << " Patterns Visualization:\n";
        std::cout << std::string(50, '=') << "\n";
        std::cout << "Total unique patterns: " << analysisResult.uniquePatternCount << "\n";
        std::cout << "Total pattern occurrences: " << totalOccurrences << "\n\n";

        // Display top-k patterns
        int displayCount = std::min(topK, (int)patterns.size());
        uint32_t topKCoverage = 0;

        for (int i = 0; i < displayCount; i++) {
            PatternMask pattern = patterns[i].pattern;
            uint32_t freq = patterns[i].frequency;
            float percentage = (freq * 100.0f) / totalOccurrences;
            topKCoverage += freq;

            std::cout << "Pattern " << (i + 1) << ": ";
            std::cout << std::setw(6) << freq << " occurrences ";
            std::cout << "(" << std::fixed << std::setprecision(2) << percentage << "%)\n";

            if (verbose) {
                std::cout << "  Binary: " << std::bitset<16>(pattern) << "\n";
                std::cout << "  Non-zeros: " << __builtin_popcount(pattern) << "\n";
            }

            // Print pattern visualization
            for (int row = 0; row < 4; row++) {
                std::cout << "    ";
                for (int col = 0; col < 4; col++) {
                    int pos = row * 4 + col;
                    if (pattern & (1 << pos)) {
                        std::cout << "* ";
                    } else {
                        std::cout << ". ";
                    }
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        // Show coverage statistics
        float topKCoveragePercent = (topKCoverage * 100.0f) / totalOccurrences;
        std::cout << "Top-" << topK << " coverage: " << topKCoverage << " / " << totalOccurrences;
        std::cout << " (" << std::fixed << std::setprecision(2) << topKCoveragePercent << "%)\n";
        std::cout << std::string(50, '=') << "\n";
    }

private:
    static void processBlock(SPASMMatrix& spasm,
                            PatternMask pattern,
                            const float values[16],
                            const std::vector<PatternMask>& templates,
                            uint32_t blockRow,
                            uint32_t blockCol,
                            uint32_t submatricesPerTile) {

        // Use cached decomposition
        auto decomp = CachedPatternDecomposer::findBestDecomposition(pattern, templates);

        // Get templates used in decomposition
        auto usedTemplates = PatternDecomposer::getUsedTemplates(
            decomp.templateCombination, templates.size());

        // Track which positions have been covered by previous templates
        PatternMask covered = 0;

        // For each used template, create position encoding and values
        for (int t_id : usedTemplates) {
            PatternMask template_mask = templates[t_id];

            // Create position encoding
            bool CE = (blockCol == submatricesPerTile - 1);
            bool RE = (blockRow == submatricesPerTile - 1);
            PositionEncoding pos = packPosition(blockCol, CE, blockRow, RE, t_id);

            // Extract 4 values for this template
            float template_values[4] = {0};
            int value_idx = 0;

            // Get values in template pattern order
            for (int bit = 0; bit < 16 && value_idx < 4; bit++) {
                if (template_mask & (1 << bit)) {
                    // Check if this position is actually in the original pattern
                    // and hasn't been covered yet
                    if ((pattern & (1 << bit)) && !(covered & (1 << bit))) {
                        template_values[value_idx] = values[bit];
                    } else {
                        // This is either padding or an overlapped position
                        template_values[value_idx] = 0;
                    }
                    value_idx++;
                }
            }

            // Update covered positions
            covered |= (template_mask & pattern);

            // Pad with zeros if template has less than 4 non-zeros
            while (value_idx < 4) {
                template_values[value_idx] = 0;
                value_idx++;
            }

            // Add to SPASM matrix
            spasm.addPositionWithValues(pos, template_values);
        }
    }
};

} // namespace spasm

#endif // OPTIMIZED_CONVERTER_V2_H