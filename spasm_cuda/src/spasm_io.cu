#include "../include/spasm_cuda.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>

SPASMMatrixHost loadSPASMFromFile(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Could not open file: ") + filename);
    }

    SPASMMatrixHost matrix;

    char magic[4];
    file.read(magic, 4);
    if (std::string(magic, 4) != "SPAS") {
        throw std::runtime_error("Invalid SPASM file format");
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (version != 1 && version != 2) {
        throw std::runtime_error("Unsupported SPASM format version");
    }

    file.read(reinterpret_cast<char*>(&matrix.rows), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&matrix.cols), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&matrix.tileSize), sizeof(uint32_t));

    uint64_t originalNnz;
    file.read(reinterpret_cast<char*>(&originalNnz), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&matrix.nnz), sizeof(uint64_t));

    file.read(reinterpret_cast<char*>(&matrix.numTiles), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&matrix.numPositions), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&matrix.numTemplates), sizeof(uint32_t));

    matrix.templatePatterns.resize(matrix.numTemplates);
    for (uint32_t i = 0; i < matrix.numTemplates; i++) {
        file.read(reinterpret_cast<char*>(&matrix.templatePatterns[i]), sizeof(uint16_t));
    }

    matrix.tilePositions.resize(matrix.numTiles * 2);
    for (uint32_t i = 0; i < matrix.numTiles; i++) {
        file.read(reinterpret_cast<char*>(&matrix.tilePositions[i * 2]), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&matrix.tilePositions[i * 2 + 1]), sizeof(uint32_t));
    }

    if (version >= 2) {
        matrix.tileBlockRanges.resize(matrix.numTiles * 2);
        for (uint32_t i = 0; i < matrix.numTiles; i++) {
            file.read(reinterpret_cast<char*>(&matrix.tileBlockRanges[i * 2]), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&matrix.tileBlockRanges[i * 2 + 1]), sizeof(uint32_t));
        }
    } else {
        matrix.tileBlockRanges.resize(matrix.numTiles * 2);
        uint32_t blocksPerTile = matrix.numPositions / matrix.numTiles;
        uint32_t remainder = matrix.numPositions % matrix.numTiles;
        uint32_t currentStart = 0;
        for (uint32_t i = 0; i < matrix.numTiles; i++) {
            uint32_t blockCount = blocksPerTile + (i < remainder ? 1 : 0);
            matrix.tileBlockRanges[i * 2] = currentStart;
            matrix.tileBlockRanges[i * 2 + 1] = currentStart + blockCount;
            currentStart += blockCount;
        }
    }

    matrix.positionEncodings.resize(matrix.numPositions);
    file.read(reinterpret_cast<char*>(matrix.positionEncodings.data()),
              matrix.numPositions * sizeof(uint32_t));

    matrix.values.resize(matrix.nnz);
    file.read(reinterpret_cast<char*>(matrix.values.data()),
              matrix.nnz * sizeof(float));

    file.close();

    std::cout << "Loaded SPASM matrix: " << matrix.rows << "x" << matrix.cols
              << ", nnz=" << matrix.nnz
              << ", tiles=" << matrix.numTiles
              << ", positions=" << matrix.numPositions << std::endl;

    return matrix;
}
