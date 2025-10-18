#ifndef SPASM_CONVERTER_PATTERN_ANALYZER_H
#define SPASM_CONVERTER_PATTERN_ANALYZER_H

#include <vector>
#include <limits>
#include <cstdint>
#include "../core/types.h"

namespace spasm {

struct DecompositionResult {
    uint16_t templateCombination;  // Bit flags for templates used
    uint32_t numPaddings;          // Number of paddings

    DecompositionResult() : templateCombination(0), numPaddings(UINT32_MAX) {}
};

// Implements pattern decomposition from paper Listing 1
class PatternDecomposer {
public:
    // Find best decomposition using brute force search
    static DecompositionResult findBestDecomposition(
        PatternMask pattern,
        const std::vector<PatternMask>& templates) {

        if (pattern == 0 || templates.empty()) {
            return DecompositionResult();
        }

        int n = std::min(static_cast<int>(templates.size()), MAX_TEMPLATES);
        DecompositionResult best;

        // Try all possible combinations (2^n - 1, excluding empty set)
        uint32_t max_combination = (1U << n);

        for (uint32_t combination = 1; combination < max_combination; combination++) {
            uint32_t num_paddings = calculatePaddingsForCombination(
                pattern, templates, combination, n);

            if (num_paddings < best.numPaddings) {
                best.templateCombination = combination;
                best.numPaddings = num_paddings;
            }
        }

        return best;
    }

    // Calculate total paddings for pattern set
    static uint64_t calculateTotalPaddings(
        const std::vector<PatternFrequency>& patterns,
        const std::vector<PatternMask>& templates) {

        uint64_t total_paddings = 0;

        for (const auto& pf : patterns) {
            auto decomp = findBestDecomposition(pf.pattern, templates);
            if (decomp.numPaddings != UINT32_MAX) {
                total_paddings += static_cast<uint64_t>(decomp.numPaddings) * pf.frequency;
            }
        }

        return total_paddings;
    }

    // Get which templates are used in decomposition
    static std::vector<int> getUsedTemplates(uint16_t combination, int num_templates) {
        std::vector<int> used;
        for (int i = 0; i < num_templates && i < MAX_TEMPLATES; i++) {
            if (combination & (1 << i)) {
                used.push_back(i);
            }
        }
        return used;
    }

private:
    // Calculate paddings for a specific template combination
    static uint32_t calculatePaddingsForCombination(
        PatternMask pattern,
        const std::vector<PatternMask>& templates,
        uint32_t combination,
        int num_templates) {

        PatternMask remain = pattern;
        PatternMask overlap = 0;
        uint32_t num_paddings = 0;

        // Process each template in combination
        for (int t_id = 0; t_id < num_templates; t_id++) {
            if (combination & (1U << t_id)) {
                PatternMask template_mask = templates[t_id];

                // Padding = positions in template but not in remaining pattern or already covered
                PatternMask padding = (~remain | overlap) & template_mask;

                // Update overlap (union of all used templates)
                overlap |= template_mask;

                // Remove covered positions from remaining
                remain &= ~template_mask;

                // Count padding bits
                num_paddings += __builtin_popcount(padding);
            }
        }

        // If pattern not fully covered, return max
        if (remain != 0) {
            return UINT32_MAX;
        }

        return num_paddings;
    }
};

} // namespace spasm

#endif // SPASM_CONVERTER_PATTERN_ANALYZER_H