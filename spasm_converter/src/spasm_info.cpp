#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <bitset>
#include "../include/io/spasm_io.h"
#include "../include/core/template_patterns.h"

using namespace spasm;

// Helper to format float values for display
std::string formatValue(float val) {
    if (val == 0.0f) {
        return "    .    ";
    }
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    if (val >= 0) ss << " ";
    if (std::abs(val) < 10) ss << " ";
    if (std::abs(val) < 100) ss << " ";
    ss << val;
    // Ensure fixed width
    std::string result = ss.str();
    if (result.length() > 9) {
        result = result.substr(0, 9);
    }
    while (result.length() < 9) {
        result += " ";
    }
    return result;
}

void printPattern(PatternMask pattern) {
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
}

// Print pattern with actual values
void printPatternWithValues(PatternMask pattern, const float values[4]) {
    // Map template positions to values
    std::vector<float> mappedValues(16, 0.0f);
    int valueIdx = 0;
    for (int i = 0; i < 16 && valueIdx < 4; i++) {
        if (pattern & (1 << i)) {
            mappedValues[i] = values[valueIdx++];
        }
    }

    // Print the pattern with values
    for (int row = 0; row < 4; row++) {
        std::cout << "    ";
        for (int col = 0; col < 4; col++) {
            int pos = row * 4 + col;
            std::cout << formatValue(mappedValues[pos]) << " ";
        }
        std::cout << "\n";
    }
}

// Decode and display position encoding details
void displayPositionEncoding(PositionEncoding pos) {
    uint32_t colIdx = getColumnIndex(pos);
    bool colEnd = getColumnEnd(pos);
    uint32_t rowIdx = getRowIndex(pos);
    bool rowEnd = getRowEnd(pos);
    uint32_t templateId = getTemplateId(pos);

    std::cout << "    Raw: 0x" << std::hex << std::setw(8) << std::setfill('0') << pos << std::dec;
    std::cout << " | Col:" << std::setw(4) << colIdx;
    std::cout << (colEnd ? "[E]" : "   ");
    std::cout << " | Row:" << std::setw(4) << rowIdx;
    std::cout << (rowEnd ? "[E]" : "   ");
    std::cout << " | Template:" << std::setw(2) << templateId;
}

// Visualize a range of blocks with their patterns and values
void visualizeBlocks(const SPASMMatrix& spasm, int start, int count) {
    std::cout << "\n========================================\n";
    std::cout << "Block-by-Block Visualization\n";
    std::cout << "========================================\n\n";

    if (start >= (int)spasm.getNumPositions()) {
        std::cout << "Error: Start position " << start << " exceeds total positions "
                  << spasm.getNumPositions() << "\n";
        return;
    }

    int end = std::min((int)spasm.getNumPositions(), start + count);
    std::cout << "Showing blocks " << start << " to " << (end - 1)
              << " (Total: " << spasm.getNumPositions() << ")\n\n";

    for (int i = start; i < end; i++) {
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ Block #" << std::setw(6) << i
                  << std::string(50, ' ') << "║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

        // Get position encoding
        PositionEncoding pos = spasm.getPosition(i);

        // Display position encoding details
        std::cout << "  Position Encoding:\n";
        displayPositionEncoding(pos);
        std::cout << "\n\n";

        // Get template information
        uint32_t templateId = getTemplateId(pos);
        if (templateId >= spasm.templatePatterns.size()) {
            std::cout << "  Error: Invalid template ID " << templateId << "\n\n";
            continue;
        }

        PatternMask templatePattern = spasm.templatePatterns[templateId].mask;

        // Get values for this position
        auto values = spasm.getValuesForPosition(i);
        if (values.size() != 4) {
            std::cout << "  Warning: Expected 4 values, got " << values.size() << "\n\n";
            continue;
        }

        // Display template pattern info
        std::cout << "  Template #" << templateId << " (";
        std::cout << __builtin_popcount(templatePattern) << " non-zeros, ";
        std::cout << "Binary: " << std::bitset<16>(templatePattern) << "):\n\n";

        // Display pattern with symbols
        std::cout << "  Pattern Mask:\n";
        printPattern(templatePattern);
        std::cout << "\n";

        // Display pattern with actual values
        std::cout << "  Pattern with Values:\n";
        float valArray[4] = {values[0], values[1], values[2], values[3]};
        printPatternWithValues(templatePattern, valArray);
        std::cout << "\n";

        // Display raw value array
        std::cout << "  Raw Values: [";
        for (size_t j = 0; j < values.size(); j++) {
            if (j > 0) std::cout << ", ";
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) << values[j];
        }
        std::cout << "]\n";

        // Display block position in matrix
        uint32_t colIdx = getColumnIndex(pos);
        uint32_t rowIdx = getRowIndex(pos);
        std::cout << "  Block Position in Tile: (" << rowIdx << ", " << colIdx << ")";
        if (getRowEnd(pos)) std::cout << " [Row End]";
        if (getColumnEnd(pos)) std::cout << " [Col End]";
        std::cout << "\n";

        std::cout << "\n" << std::string(68, '-') << "\n\n";
    }
}

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " <file.spasm> [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -v              : Verbose output (show templates and tiles)\n";
    std::cout << "  -samples        : Show sample position encodings\n";
    std::cout << "  -blocks <start> <count> : Visualize specific blocks with patterns and values\n";
    std::cout << "                    start: Starting block index (0-based)\n";
    std::cout << "                    count: Number of blocks to display\n";
    std::cout << "Example:\n";
    std::cout << "  " << progName << " matrix.spasm -blocks 0 5  # Show first 5 blocks\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string filename = argv[1];
    bool verbose = false;
    bool showSamples = false;
    bool showBlocks = false;
    int blockStart = 0;
    int blockCount = 10;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-v") {
            verbose = true;
        } else if (arg == "-samples") {
            showSamples = true;
        } else if (arg == "-blocks" && i + 2 < argc) {
            showBlocks = true;
            blockStart = std::stoi(argv[i + 1]);
            blockCount = std::stoi(argv[i + 2]);
            i += 2;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    try {
        std::cout << "\n========================================\n";
        std::cout << "SPASM File Information\n";
        std::cout << "========================================\n\n";
        std::cout << "File: " << filename << "\n\n";

        // Read SPASM file
        SPASMMatrix spasm = SPASMReader::readFromFile(filename);

        // Basic information
        std::cout << "Matrix Properties:\n";
        std::cout << "  Dimensions: " << spasm.rows << " x " << spasm.cols << "\n";
        std::cout << "  Tile size: " << spasm.tileSize << "\n";
        std::cout << "  Original non-zeros: " << spasm.originalNnz << "\n";
        std::cout << "  Padded non-zeros: " << spasm.nnz << "\n";
        std::cout << "  Padding count: " << spasm.numPaddings << "\n";
        std::cout << "  Padding rate: " << std::fixed << std::setprecision(2)
                  << (spasm.getPaddingRate() * 100) << "%\n\n";

        std::cout << "Storage Information:\n";
        std::cout << "  Number of tiles: " << spasm.getNumTiles() << "\n";
        std::cout << "  Number of templates: " << spasm.templatePatterns.size() << "\n";
        std::cout << "  Position encodings: " << spasm.getNumPositions() << "\n";
        std::cout << "  Values stored: " << spasm.values.size() << "\n";
        std::cout << "  Values per position: " << (spasm.values.size() / (double)spasm.getNumPositions()) << "\n\n";

        std::cout << "Compression Metrics:\n";
        uint64_t fileSize = getSPASMFileSize(spasm);
        std::cout << "  SPASM file size: " << (fileSize / 1024.0) << " KB\n";
        std::cout << "  Bytes per non-zero: " << spasm.getBytesPerNonZero() << "\n";
        std::cout << "  Compression vs COO: " << spasm.getCompressionRatio() << "x\n";

        uint64_t cooSize = spasm.originalNnz * (sizeof(uint32_t) * 2 + sizeof(float));
        std::cout << "  Estimated COO size: " << (cooSize / 1024.0) << " KB\n";
        std::cout << "  File compression ratio: " << (double)cooSize / fileSize << "x\n\n";

        // Template patterns
        if (verbose && !spasm.templatePatterns.empty()) {
            std::cout << "Template Patterns (" << spasm.templatePatterns.size() << "):\n";
            std::cout << std::string(40, '-') << "\n";
            for (size_t i = 0; i < spasm.templatePatterns.size(); i++) {
                auto pattern = spasm.templatePatterns[i].mask;
                int nnz = __builtin_popcount(pattern);
                std::cout << "Template " << std::setw(2) << i
                          << " (0x" << std::hex << std::setw(4) << std::setfill('0') << pattern
                          << std::dec << ") - " << nnz << " non-zeros:\n";
                printPattern(pattern);
            }
            std::cout << "\n";
        }

        // Sample position encodings
        if (showSamples && spasm.getNumPositions() > 0) {
            std::cout << "Sample Position Encodings:\n";
            std::cout << std::string(40, '-') << "\n";

            int numSamples = std::min(10UL, spasm.getNumPositions());
            for (int i = 0; i < numSamples; i++) {
                PositionEncoding pos = spasm.getPosition(i);
                std::cout << "Position " << i << ":\n";
                displayPositionEncoding(pos);
                std::cout << "\n";

                // Show associated values
                auto vals = spasm.getValuesForPosition(i);
                std::cout << "  Values (" << vals.size() << "): [";
                for (size_t j = 0; j < vals.size(); j++) {
                    if (j > 0) std::cout << ", ";
                    std::cout << vals[j];
                }
                std::cout << "]\n\n";
            }

            if (spasm.getNumPositions() > numSamples) {
                std::cout << "  ... (" << (spasm.getNumPositions() - numSamples)
                          << " more positions)\n\n";
            }
        }

        // Visualize specific blocks if requested
        if (showBlocks) {
            visualizeBlocks(spasm, blockStart, blockCount);
        }

        // Tile distribution
        if (verbose && !spasm.tilePositions.empty()) {
            std::cout << "Tile Distribution:\n";
            std::cout << std::string(40, '-') << "\n";

            // Find tile bounds
            uint32_t maxTileRow = 0, maxTileCol = 0;
            for (const auto& [row, col] : spasm.tilePositions) {
                maxTileRow = std::max(maxTileRow, row);
                maxTileCol = std::max(maxTileCol, col);
            }

            std::cout << "  Tile grid: " << (maxTileRow + 1) << " x " << (maxTileCol + 1) << "\n";
            std::cout << "  Non-empty tiles: " << spasm.tilePositions.size() << "\n";

            // Display tile block ranges
            if (!spasm.tileBlockRanges.empty()) {
                std::cout << "\n  Tile Block Ranges:\n";
                int maxTilesToShow = std::min(10, (int)spasm.tileBlockRanges.size());
                for (int i = 0; i < maxTilesToShow; i++) {
                    const auto& pos = spasm.tilePositions[i];
                    const auto& range = spasm.tileBlockRanges[i];
                    std::cout << "    Tile " << i << " (" << pos.tileRowIdx << ", " << pos.tileColIdx << "): ";
                    std::cout << "blocks [" << range.blockStart << ", " << range.blockEnd << ") ";
                    std::cout << "(" << range.getNumBlocks() << " blocks)\n";
                }
                if (spasm.tileBlockRanges.size() > maxTilesToShow) {
                    std::cout << "    ... (" << (spasm.tileBlockRanges.size() - maxTilesToShow)
                              << " more tiles)\n";
                }
            }

            // Show tile map (if small enough)
            if (maxTileRow < 10 && maxTileCol < 10) {
                std::cout << "\n  Tile occupancy map (* = non-empty):\n";
                std::cout << "    ";
                for (uint32_t c = 0; c <= maxTileCol; c++) {
                    std::cout << c << " ";
                }
                std::cout << "\n";

                for (uint32_t r = 0; r <= maxTileRow; r++) {
                    std::cout << "  " << r << " ";
                    for (uint32_t c = 0; c <= maxTileCol; c++) {
                        bool found = false;
                        for (const auto& [tr, tc] : spasm.tilePositions) {
                            if (tr == r && tc == c) {
                                found = true;
                                break;
                            }
                        }
                        std::cout << (found ? "* " : ". ");
                    }
                    std::cout << "\n";
                }
            }
            std::cout << "\n";
        }

        std::cout << "========================================\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}