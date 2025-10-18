#ifndef SPASM_CONVERTER_BLOCK_DECOMPOSER_H
#define SPASM_CONVERTER_BLOCK_DECOMPOSER_H

#include <vector>
#include <unordered_map>
#include <limits>
#include <cstdint>
#include "../core/types.h"
#include "pattern_analyzer.h"

namespace spasm {

// Cached pattern decomposer that memoizes decomposition results
class CachedPatternDecomposer {
private:
    // Cache key: combines pattern and template set size
    struct CacheKey {
        PatternMask pattern;
        size_t templateCount;

        bool operator==(const CacheKey& other) const {
            return pattern == other.pattern && templateCount == other.templateCount;
        }
    };

    // Hash function for cache key
    struct CacheKeyHash {
        size_t operator()(const CacheKey& key) const {
            return std::hash<uint32_t>()((uint32_t(key.pattern) << 8) | key.templateCount);
        }
    };

    // Thread-local cache to avoid synchronization overhead
    thread_local static std::unordered_map<CacheKey, DecompositionResult, CacheKeyHash> decompositionCache;

public:
    // Find best decomposition with caching
    static DecompositionResult findBestDecomposition(
        PatternMask pattern,
        const std::vector<PatternMask>& templates) {

        if (pattern == 0 || templates.empty()) {
            return DecompositionResult();
        }

        // Create cache key
        CacheKey key{pattern, templates.size()};

        // Check cache first
        auto it = decompositionCache.find(key);
        if (it != decompositionCache.end()) {
            return it->second;
        }

        // Not in cache, perform actual decomposition
        DecompositionResult result = PatternDecomposer::findBestDecomposition(pattern, templates);

        // Store in cache for future use
        decompositionCache[key] = result;

        return result;
    }

    // Clear cache (useful between different matrices)
    static void clearCache() {
        decompositionCache.clear();
    }

    // Get cache statistics
    static size_t getCacheSize() {
        return decompositionCache.size();
    }

    // Pre-populate cache with known patterns
    static void precomputePatterns(
        const std::vector<PatternFrequency>& patterns,
        const std::vector<PatternMask>& templates,
        int topK = 100) {

        int count = 0;
        for (const auto& pf : patterns) {
            if (count >= topK) break;

            CacheKey key{pf.pattern, templates.size()};
            if (decompositionCache.find(key) == decompositionCache.end()) {
                decompositionCache[key] = PatternDecomposer::findBestDecomposition(pf.pattern, templates);
            }
            count++;
        }
    }
};

// Define thread-local cache
thread_local std::unordered_map<CachedPatternDecomposer::CacheKey,
                                DecompositionResult,
                                CachedPatternDecomposer::CacheKeyHash>
    CachedPatternDecomposer::decompositionCache;

} // namespace spasm

#endif // SPASM_CONVERTER_BLOCK_DECOMPOSER_H