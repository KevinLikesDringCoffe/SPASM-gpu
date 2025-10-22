#include <iostream>
#include <vector>
#include <cstdint>
#include "../spasm_converter/include/core/format.h"
#include "../spasm_converter/include/io/spasm_io.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <spasm_file>\n";
        return 1;
    }

    spasm::SPASMMatrix matrix = spasm::SPASMReader::readFromFile(argv[1]);

    std::cout << "Verifying tile block ranges:\n";
    std::cout << "  Num tiles: " << matrix.getNumTiles() << "\n";
    std::cout << "  Num blocks: " << matrix.getNumPositions() << "\n\n";

    uint32_t totalBlocks = 0;
    std::vector<bool> covered(matrix.getNumPositions(), false);

    for (size_t i = 0; i < matrix.tileBlockRanges.size(); i++) {
        const auto& range = matrix.tileBlockRanges[i];
        uint32_t count = range.blockEnd - range.blockStart;
        totalBlocks += count;

        std::cout << "Tile " << i << ": blockStart=" << range.blockStart
                  << " blockEnd=" << range.blockEnd
                  << " count=" << count << "\n";

        // Mark covered blocks
        for (uint32_t idx = range.blockStart; idx < range.blockEnd; idx++) {
            if (idx < covered.size()) {
                if (covered[idx]) {
                    std::cout << "  WARNING: Block " << idx << " covered by multiple tiles!\n";
                }
                covered[idx] = true;
            } else {
                std::cout << "  ERROR: Block index " << idx << " out of range!\n";
            }
        }
    }

    // Check for uncovered blocks
    uint32_t uncovered = 0;
    for (size_t i = 0; i < covered.size(); i++) {
        if (!covered[i]) {
            if (uncovered < 10) {
                std::cout << "  Uncovered block: " << i << "\n";
            }
            uncovered++;
        }
    }

    std::cout << "\nSummary:\n";
    std::cout << "  Total blocks from ranges: " << totalBlocks << "\n";
    std::cout << "  Expected blocks: " << matrix.getNumPositions() << "\n";
    std::cout << "  Uncovered blocks: " << uncovered << "\n";
    std::cout << "  Match: " << (totalBlocks == matrix.getNumPositions() && uncovered == 0 ? "YES" : "NO") << "\n";

    return 0;
}
