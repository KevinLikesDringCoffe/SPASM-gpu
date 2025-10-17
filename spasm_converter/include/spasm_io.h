#ifndef SPASM_IO_H
#define SPASM_IO_H

#include <fstream>
#include <string>
#include <vector>
#include "spasm_format.h"

namespace spasm {

// SPASM file format structure (binary):
// Header:
//   - Magic number (4 bytes): 'SPAS'
//   - Version (4 bytes): format version
//   - Matrix rows (4 bytes)
//   - Matrix cols (4 bytes)
//   - Tile size (4 bytes)
//   - Original nnz (8 bytes)
//   - Padded nnz (8 bytes)
//   - Num tiles (4 bytes)
//   - Num positions (4 bytes)
//   - Num templates (4 bytes)
// Template patterns:
//   - Template masks (2 bytes each * num_templates)
// Tile positions:
//   - Row, Col pairs (4 bytes each * num_tiles * 2)
// Position encodings:
//   - Position encodings (4 bytes each * num_positions)
// Values:
//   - Float values (4 bytes each * padded_nnz)

class SPASMWriter {
public:
    static bool writeToFile(const SPASMMatrix& matrix, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing\n";
            return false;
        }

        // Write header
        const char magic[4] = {'S', 'P', 'A', 'S'};
        file.write(magic, 4);

        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));

        uint32_t rows = matrix.rows;
        uint32_t cols = matrix.cols;
        uint32_t tileSize = matrix.tileSize;
        file.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&tileSize), sizeof(uint32_t));

        uint64_t originalNnz = matrix.originalNnz;
        uint64_t paddedNnz = matrix.nnz;
        file.write(reinterpret_cast<const char*>(&originalNnz), sizeof(uint64_t));
        file.write(reinterpret_cast<const char*>(&paddedNnz), sizeof(uint64_t));

        uint32_t numTiles = matrix.tilePositions.size();
        uint32_t numPositions = matrix.positionEncodings.size();
        uint32_t numTemplates = matrix.templatePatterns.size();
        file.write(reinterpret_cast<const char*>(&numTiles), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&numPositions), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&numTemplates), sizeof(uint32_t));

        // Write template patterns
        for (const auto& tmpl : matrix.templatePatterns) {
            uint16_t mask = tmpl.mask;
            file.write(reinterpret_cast<const char*>(&mask), sizeof(uint16_t));
        }

        // Write tile positions
        for (const auto& [tileRow, tileCol] : matrix.tilePositions) {
            file.write(reinterpret_cast<const char*>(&tileRow), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&tileCol), sizeof(uint32_t));
        }

        // Write position encodings
        file.write(reinterpret_cast<const char*>(matrix.positionEncodings.data()),
                   matrix.positionEncodings.size() * sizeof(PositionEncoding));

        // Write values
        file.write(reinterpret_cast<const char*>(matrix.values.data()),
                   matrix.values.size() * sizeof(float));

        file.close();
        return true;
    }
};

class SPASMReader {
public:
    static SPASMMatrix readFromFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file " + filename + " for reading");
        }

        // Read and verify magic number
        char magic[4];
        file.read(magic, 4);
        if (std::string(magic, 4) != "SPAS") {
            throw std::runtime_error("Invalid SPASM file format - bad magic number");
        }

        // Read version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
        if (version != 1) {
            throw std::runtime_error("Unsupported SPASM format version: " + std::to_string(version));
        }

        // Read header
        uint32_t rows, cols, tileSize;
        file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&tileSize), sizeof(uint32_t));

        SPASMMatrix matrix(rows, cols, tileSize);

        uint64_t originalNnz, paddedNnz;
        file.read(reinterpret_cast<char*>(&originalNnz), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&paddedNnz), sizeof(uint64_t));
        matrix.originalNnz = originalNnz;
        matrix.nnz = paddedNnz;

        uint32_t numTiles, numPositions, numTemplates;
        file.read(reinterpret_cast<char*>(&numTiles), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&numPositions), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&numTemplates), sizeof(uint32_t));

        // Read template patterns
        matrix.templatePatterns.reserve(numTemplates);
        for (uint32_t i = 0; i < numTemplates; i++) {
            uint16_t mask;
            file.read(reinterpret_cast<char*>(&mask), sizeof(uint16_t));
            matrix.templatePatterns.emplace_back(mask);
        }

        // Read tile positions
        matrix.tilePositions.reserve(numTiles);
        for (uint32_t i = 0; i < numTiles; i++) {
            uint32_t tileRow, tileCol;
            file.read(reinterpret_cast<char*>(&tileRow), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&tileCol), sizeof(uint32_t));
            matrix.tilePositions.emplace_back(tileRow, tileCol);
        }

        // Read position encodings
        matrix.positionEncodings.resize(numPositions);
        file.read(reinterpret_cast<char*>(matrix.positionEncodings.data()),
                  numPositions * sizeof(PositionEncoding));

        // Read values
        matrix.values.resize(paddedNnz);
        file.read(reinterpret_cast<char*>(matrix.values.data()),
                  paddedNnz * sizeof(float));

        file.close();

        // Calculate derived values
        matrix.numPaddings = matrix.nnz - matrix.originalNnz;

        return matrix;
    }
};

// Helper functions for file size
inline uint64_t getSPASMFileSize(const SPASMMatrix& matrix) {
    uint64_t size = 0;
    size += 4;  // Magic number
    size += sizeof(uint32_t);  // Version
    size += 3 * sizeof(uint32_t);  // rows, cols, tileSize
    size += 2 * sizeof(uint64_t);  // originalNnz, paddedNnz
    size += 3 * sizeof(uint32_t);  // numTiles, numPositions, numTemplates
    size += matrix.templatePatterns.size() * sizeof(uint16_t);  // Template patterns
    size += matrix.tilePositions.size() * 2 * sizeof(uint32_t);  // Tile positions
    size += matrix.positionEncodings.size() * sizeof(PositionEncoding);  // Position encodings
    size += matrix.values.size() * sizeof(float);  // Values
    return size;
}

} // namespace spasm

#endif // SPASM_IO_H