#include <iostream>
#include <iomanip>
#include <string>
#include "../include/spasm_io.h"
#include "../include/template_selection.h"

using namespace spasm;

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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <file.spasm> [options]\n";
        std::cout << "Options:\n";
        std::cout << "  -v        : Verbose output (show templates)\n";
        std::cout << "  -samples  : Show sample position encodings\n";
        return 1;
    }

    std::string filename = argv[1];
    bool verbose = false;
    bool showSamples = false;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-v") {
            verbose = true;
        } else if (arg == "-samples") {
            showSamples = true;
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
        std::cout << "  Padding rate: " << (spasm.getPaddingRate() * 100) << "%\n\n";

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
                std::cout << "  Raw encoding: 0x" << std::hex << pos << std::dec << "\n";
                std::cout << "  Column index: " << getColumnIndex(pos) << "\n";
                std::cout << "  Column end: " << (getColumnEnd(pos) ? "true" : "false") << "\n";
                std::cout << "  Row index: " << getRowIndex(pos) << "\n";
                std::cout << "  Row end: " << (getRowEnd(pos) ? "true" : "false") << "\n";
                std::cout << "  Template ID: " << getTemplateId(pos) << "\n";

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