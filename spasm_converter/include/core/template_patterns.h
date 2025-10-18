#ifndef SPASM_CORE_TEMPLATE_PATTERNS_H
#define SPASM_CORE_TEMPLATE_PATTERNS_H

#include <vector>
#include <algorithm>
#include <iostream>
#include "types.h"

namespace spasm {

// Pattern frequency structure (needed for compatibility)
struct PatternFrequency {
    PatternMask pattern;
    uint32_t frequency;

    PatternFrequency(PatternMask p = 0, uint32_t f = 0) : pattern(p), frequency(f) {}

    bool operator>(const PatternFrequency& other) const {
        return frequency > other.frequency;
    }
};

// Implements Algorithm 3 from paper
class TemplateSelector {
public:
    // All 20 base template patterns (constant array)
    static constexpr PatternMask BASE_TEMPLATES[20] = {
        // Row patterns (4) - indices 0-3
        0x000F,  // Row 0: 0000000000001111
        0x00F0,  // Row 1: 0000000011110000
        0x0F00,  // Row 2: 0000111100000000
        0xF000,  // Row 3: 1111000000000000

        // Column patterns (4) - indices 4-7
        0x1111,  // Col 0: 0001000100010001
        0x2222,  // Col 1: 0010001000100010
        0x4444,  // Col 2: 0100010001000100
        0x8888,  // Col 3: 1000100010001000

        // 2x2 Block patterns (4) - indices 8-11
        0x0033,  // Top-left:    0000000000110011
        0x00CC,  // Top-right:   0000000011001100
        0x3300,  // Bottom-left: 0011001100000000
        0xCC00,  // Bottom-right:1100110000000000

        // Main diagonal patterns (4) - indices 12-15
        0x8421,  // Main diagonal:        1000010000100001
        0x4218,  // Shifted main diag:    0100001000011000
        0x2184,  // Shifted main diag:    0010000110000100
        0x1842,  // Shifted main diag:    0001100001000010

        // Anti-diagonal patterns (4) - indices 16-19
        0x1248,  // Anti-diagonal:        0001001001001000
        0x2481,  // Shifted anti-diag:    0010010010000001
        0x4812,  // Shifted anti-diag:    0100100000010010
        0x8124   // Shifted anti-diag:    1000000100100100
    };

    static constexpr int NUM_BASE_TEMPLATES = 20;

    // Generate predefined template patterns (Table V from paper)
    static std::vector<std::vector<PatternMask>> getPredefinedTemplateSets() {
        std::vector<std::vector<PatternMask>> sets;

        // Set 0: 4 RW, 4 CW, 4 BW, 4 main diagonal (original paper set)
        sets.push_back(generateSet0());

        // Set 1: 4 RW, 4 CW, 4 BW, 4 anti-diagonal
        sets.push_back(generateSet1());

        // Set 2: Mixed patterns without blocks
        sets.push_back(generateSet2());

        // Set 3: Block-focused set
        sets.push_back(generateSet3());

        // Set 4: Diagonal-focused set
        sets.push_back(generateSet4());

        // Set 5: Balanced set
        sets.push_back(generateSet5());

        return sets;
    }

private:
    // Helper function to select templates from BASE_TEMPLATES by indices
    static std::vector<PatternMask> selectTemplates(const std::vector<int>& indices) {
        std::vector<PatternMask> templates;
        for (int idx : indices) {
            if (idx >= 0 && idx < NUM_BASE_TEMPLATES) {
                templates.push_back(BASE_TEMPLATES[idx]);
            }
        }
        return templates;
    }

    // Set 0: 4 row, 4 column, 4 block, 4 main diagonal (original paper set)
    static std::vector<PatternMask> generateSet0() {
        // Select: rows(0-3), columns(4-7), blocks(8-11), main diagonals(12-15)
        return selectTemplates({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    }

    // Set 1: 4 row, 4 column, 4 block, 4 anti-diagonal
    static std::vector<PatternMask> generateSet1() {
        // Select: rows(0-3), columns(4-7), blocks(8-11), anti-diagonals(16-19)
        return selectTemplates({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19});
    }

    // Set 2: Mixed patterns - rows, columns, all diagonal types
    static std::vector<PatternMask> generateSet2() {
        // Select: rows(0-3), columns(4-7), main diagonals(12-15), anti-diagonals(16-19)
        // This gives good coverage without blocks
        return selectTemplates({0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19});
    }

    // Set 3: Block-focused - all blocks, selected rows/columns and diagonals
    static std::vector<PatternMask> generateSet3() {
        // Select: first 2 rows(0-1), last 2 columns(6-7), all blocks(8-11), first 4 main diagonals(12-15)
        return selectTemplates({0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 3, 4, 5});
    }

    // Additional pattern sets can be defined here
    // Set 4: Diagonal-focused
    static std::vector<PatternMask> generateSet4() {
        // Select: 2 rows(0,2), 2 columns(5,7), 2 blocks(8,11), all diagonals(12-19)
        return selectTemplates({0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 3});
    }

    // Set 5: Balanced set
    static std::vector<PatternMask> generateSet5() {
        // Select evenly: 3 rows, 3 columns, 5 blocks, 5 diagonals
        return selectTemplates({0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 3, 12, 13, 16, 17, 18});
    }
};

} // namespace spasm

#endif // SPASM_CORE_TEMPLATE_PATTERNS_H