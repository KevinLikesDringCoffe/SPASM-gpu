#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <map>
#include <bitset>
#include <algorithm>
#include <sstream>
#include "../include/io/mtx_reader.h"
#include "../include/io/spasm_io.h"
#include "../include/core/template_patterns.h"
#include "../include/converter/converter.h"

namespace fs = std::filesystem;
using namespace spasm;
using namespace std::chrono;

// Enhanced formatting utilities
class Format {
public:
    static void printHeader(const std::string& title) {
        const int width = 70;
        std::cout << "\n╔" << std::string(width, '=') << "╗\n";
        int padding = (width - title.length()) / 2;
        std::cout << "║" << std::string(padding, ' ') << title
                  << std::string(width - padding - title.length(), ' ') << "║\n";
        std::cout << "╚" << std::string(width, '=') << "╝\n\n";
    }

    static void printSection(const std::string& title, char borderChar = '-') {
        const int width = 70;
        std::cout << "\n┌" << std::string(width, borderChar) << "┐\n";
        std::cout << "│ " << std::left << std::setw(width - 2) << title << " │\n";
        std::cout << "└" << std::string(width, borderChar) << "┘\n";
    }

    static void printSubSection(const std::string& title) {
        std::cout << "\n▶ " << title << "\n";
        std::cout << "  " << std::string(title.length(), '-') << "\n";
    }

    static void printKeyValue(const std::string& key, const std::string& value, int indent = 2) {
        std::cout << std::string(indent, ' ') << "• " << std::left << std::setw(25)
                  << key + ":" << value << "\n";
    }

    static void printProgress(const std::string& task, bool start = true) {
        if (start) {
            std::cout << "  ► " << task << "... ";
            std::cout.flush();
        } else {
            std::cout << "✓\n";
        }
    }

    static void printSeparator(char c = '-', int width = 70) {
        std::cout << std::string(width, c) << "\n";
    }

    static std::string formatNumber(uint64_t n) {
        std::stringstream ss;
        ss.imbue(std::locale(""));
        ss << std::fixed << n;
        return ss.str();
    }

    static std::string formatSize(uint64_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB"};
        int unit = 0;
        double size = bytes;
        while (size > 1024 && unit < 3) {
            size /= 1024;
            unit++;
        }
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << size << " " << units[unit];
        return ss.str();
    }

    static std::string formatTime(uint64_t ms) {
        if (ms < 1000) {
            return std::to_string(ms) + " ms";
        } else if (ms < 60000) {
            return std::to_string(ms / 1000) + "." + std::to_string((ms % 1000) / 100) + " s";
        } else {
            return std::to_string(ms / 60000) + "m " + std::to_string((ms % 60000) / 1000) + "s";
        }
    }
};

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

void visualizeTopPatterns(const OptimizedConverterV2::AnalysisAndConversionResult& analysisResult,
                          int topK = 8, bool verbose = false) {
    const auto& patterns = analysisResult.patterns;
    uint32_t totalOccurrences = analysisResult.totalPatternOccurrences;

    Format::printSection("Pattern Analysis Results");

    Format::printKeyValue("Unique patterns", std::to_string(analysisResult.uniquePatternCount));
    Format::printKeyValue("Total occurrences", Format::formatNumber(totalOccurrences));

    std::cout << "\n  Top-" << topK << " Most Frequent Patterns:\n";
    Format::printSeparator('.', 60);

    // Display top-k patterns
    int displayCount = std::min(topK, (int)patterns.size());
    uint32_t topKCoverage = 0;

    for (int i = 0; i < displayCount; i++) {
        PatternMask pattern = patterns[i].pattern;
        uint32_t freq = patterns[i].frequency;
        float percentage = (freq * 100.0f) / totalOccurrences;
        topKCoverage += freq;

        std::cout << "\n  Pattern #" << (i + 1) << ": ";
        std::cout << freq << " occurrences ";
        std::cout << "(" << std::fixed << std::setprecision(1) << percentage << "%)\n";

        if (verbose) {
            std::cout << "    Binary: " << std::bitset<16>(pattern) << "\n";
            std::cout << "    Non-zeros: " << __builtin_popcount(pattern) << "\n";
        }

        printPattern(pattern);
    }

    Format::printSeparator('.', 60);
    float topKCoveragePercent = (topKCoverage * 100.0f) / totalOccurrences;
    std::cout << "  Coverage: Top-" << topK << " patterns cover "
              << std::fixed << std::setprecision(1) << topKCoveragePercent
              << "% of all blocks\n";
}

void printUsage(const char* progName) {
    Format::printHeader("SPASM Converter Usage");
    std::cout << "Usage: " << progName << " <input.mtx> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -o <output.spasm>          : Output file (default: input.spasm)\n";
    std::cout << "  --tile-size <N>            : Tile size (default: 1024)\n";
    std::cout << "  --template-set <0-5>       : Template set ID (default: 0)\n";
    std::cout << "  --show-patterns <N>        : Display top N patterns (default: 8)\n";
    std::cout << "  -t <N>                     : Short for --tile-size\n";
    std::cout << "  -s <0-5>                   : Short for --template-set\n";
    std::cout << "  -k <N>                     : Short for --show-patterns\n";
    std::cout << "  -v, --verbose              : Verbose output\n";
    std::cout << "  --verify                   : Verify conversion\n";
    std::cout << "  -h, --help                 : Show this help\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    // Check for help flag first
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Parse arguments
    std::string inputFile = argv[1];
    std::string outputFile;
    uint32_t tileSize = 1024;
    int templateSetId = 0;
    int topK = 8;
    bool verbose = false;
    bool verify = false;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if ((arg == "-t" || arg == "--tile-size") && i + 1 < argc) {
            tileSize = std::stoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--template-set") && i + 1 < argc) {
            templateSetId = std::stoi(argv[++i]);
        } else if ((arg == "-k" || arg == "--show-patterns") && i + 1 < argc) {
            topK = std::stoi(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--verify") {
            verify = true;
        }
    }

    // Default output file
    if (outputFile.empty()) {
        fs::path inputPath(inputFile);
        outputFile = inputPath.stem().string() + ".spasm";
    }

    Format::printHeader("SPASM Matrix Converter");

    // Print configuration
    Format::printSubSection("Configuration");
    Format::printKeyValue("Input file", inputFile);
    Format::printKeyValue("Output file", outputFile);
    Format::printKeyValue("Tile size", std::to_string(tileSize));
    Format::printKeyValue("Template set", "Set " + std::to_string(templateSetId));
    Format::printKeyValue("Pattern visualization", "Top-" + std::to_string(topK));

    try {
        // Step 1: Read MTX file
        Format::printSection("Step 1: Reading MTX File");
        Format::printProgress("Loading matrix data");

        auto start = high_resolution_clock::now();
        COOMatrix input = MTXReader::readFile(inputFile);
        auto end = high_resolution_clock::now();
        auto readTime = duration_cast<milliseconds>(end - start).count();

        Format::printProgress("Loading matrix data", false);

        Format::printKeyValue("Matrix dimensions",
            std::to_string(input.rows) + " × " + std::to_string(input.cols));
        Format::printKeyValue("Non-zero elements", Format::formatNumber(input.nnz));
        Format::printKeyValue("Sparsity",
            std::to_string(100.0 - (100.0 * input.nnz) / (input.rows * input.cols)) + "%");
        Format::printKeyValue("Read time", Format::formatTime(readTime));

        // Step 2: Convert to SPASM format
        Format::printSection("Step 2: Matrix Conversion & Pattern Analysis");

        // Get template set
        auto templateSets = TemplateSelector::getPredefinedTemplateSets();
        std::vector<PatternMask> templates;
        if (templateSetId >= 0 && templateSetId < (int)templateSets.size()) {
            templates = templateSets[templateSetId];
            Format::printKeyValue("Template set", "Set " + std::to_string(templateSetId) +
                                 " (" + std::to_string(templates.size()) + " templates)");
        } else {
            templates = templateSets[0];
            Format::printKeyValue("Template set", "Default Set 0 (" +
                                 std::to_string(templates.size()) + " templates)");
        }

        // Create SPASM matrix
        Format::printProgress("Converting to SPASM format");

        start = high_resolution_clock::now();
        SPASMMatrix spasm(input.rows, input.cols, tileSize);
        spasm.originalNnz = input.nnz;

        // Store templates
        for (const auto& mask : templates) {
            spasm.templatePatterns.emplace_back(mask);
        }

        // Convert with integrated pattern analysis
        auto analysisResult = OptimizedConverterV2::convertAndAnalyze(
            spasm, input, templates, tileSize, verbose);

        // Calculate statistics
        spasm.nnz = spasm.values.size();
        spasm.numPaddings = spasm.nnz - spasm.originalNnz;

        end = high_resolution_clock::now();
        auto convertTime = duration_cast<milliseconds>(end - start).count();

        Format::printProgress("Converting to SPASM format", false);
        Format::printKeyValue("Conversion time", Format::formatTime(convertTime));

        // Visualize patterns
        visualizeTopPatterns(analysisResult, topK, verbose);

        // Step 3: Write SPASM file
        Format::printSection("Step 3: Writing Output");
        Format::printProgress("Writing SPASM file");

        start = high_resolution_clock::now();
        bool success = SPASMWriter::writeToFile(spasm, outputFile);
        end = high_resolution_clock::now();
        auto writeTime = duration_cast<milliseconds>(end - start).count();

        if (!success) {
            std::cerr << "\n✗ Error: Failed to write SPASM file\n";
            return 1;
        }

        Format::printProgress("Writing SPASM file", false);

        uint64_t fileSize = getSPASMFileSize(spasm);
        Format::printKeyValue("Output file size", Format::formatSize(fileSize));
        Format::printKeyValue("Write time", Format::formatTime(writeTime));

        // Step 4: Verify if requested
        if (verify) {
            Format::printSection("Step 4: Verification");
            Format::printProgress("Verifying conversion");

            start = high_resolution_clock::now();
            SPASMMatrix loaded = SPASMReader::readFromFile(outputFile);
            end = high_resolution_clock::now();
            auto loadTime = duration_cast<milliseconds>(end - start).count();

            bool valid = (loaded.rows == spasm.rows &&
                         loaded.cols == spasm.cols &&
                         loaded.originalNnz == spasm.originalNnz &&
                         loaded.nnz == spasm.nnz &&
                         loaded.positionEncodings.size() == spasm.positionEncodings.size() &&
                         loaded.values.size() == spasm.values.size());

            Format::printProgress("Verifying conversion", false);

            if (valid) {
                std::cout << "  ✓ Verification successful\n";
            } else {
                std::cout << "  ✗ Verification failed\n";
            }
            Format::printKeyValue("Verification time", Format::formatTime(loadTime));
        }

        // Print final summary
        Format::printSection("Conversion Summary", '=');

        // Matrix info
        Format::printSubSection("Matrix Statistics");
        Format::printKeyValue("Original non-zeros", Format::formatNumber(spasm.originalNnz));
        Format::printKeyValue("After padding", Format::formatNumber(spasm.nnz));
        Format::printKeyValue("Padding elements", Format::formatNumber(spasm.numPaddings));
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << (spasm.getPaddingRate() * 100) << "%";
        Format::printKeyValue("Padding rate", ss.str());

        // Storage info
        Format::printSubSection("Storage Information");
        Format::printKeyValue("Number of tiles", std::to_string(spasm.getNumTiles()));
        Format::printKeyValue("Position encodings", Format::formatNumber(spasm.getNumPositions()));
        Format::printKeyValue("Compression ratio",
            std::to_string(spasm.getCompressionRatio()).substr(0, 4) + "×");

        uint64_t cooSize = input.nnz * (sizeof(uint32_t) * 2 + sizeof(float));
        Format::printKeyValue("COO format size", Format::formatSize(cooSize));
        Format::printKeyValue("SPASM format size", Format::formatSize(fileSize));
        Format::printKeyValue("Size reduction",
            std::to_string(int((1 - (double)fileSize/cooSize) * 100)) + "%");

        // Timing
        Format::printSubSection("Performance");
        uint64_t totalTime = readTime + convertTime + writeTime;
        Format::printKeyValue("Total time", Format::formatTime(totalTime));

        Format::printSeparator('=');
        std::cout << "\n  ✓ Conversion completed successfully!\n";
        std::cout << "  Output saved to: " << outputFile << "\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}