#ifndef SPASM_CUDA_H
#define SPASM_CUDA_H

#include <cstdint>
#include <vector>

struct SPASMMatrixCUDA {
    uint32_t* d_tilePositions;
    uint32_t* d_tileBlockRanges;
    uint32_t* d_positionEncodings;
    float* d_values;
    uint16_t* d_templatePatterns;

    uint32_t rows;
    uint32_t cols;
    uint32_t tileSize;
    uint32_t numTiles;
    uint32_t numPositions;
    uint32_t numTemplates;
    uint64_t nnz;

    float* d_x;
    float* d_y;
};

struct SPASMMatrixHost {
    std::vector<uint32_t> tilePositions;
    std::vector<uint32_t> tileBlockRanges;
    std::vector<uint32_t> positionEncodings;
    std::vector<float> values;
    std::vector<uint16_t> templatePatterns;

    uint32_t rows;
    uint32_t cols;
    uint32_t tileSize;
    uint32_t numTiles;
    uint32_t numPositions;
    uint32_t numTemplates;
    uint64_t nnz;
};

SPASMMatrixHost loadSPASMFromFile(const char* filename);

void allocateSPASMCUDA(const SPASMMatrixHost& host, SPASMMatrixCUDA& cuda);
void copySPASMToDevice(const SPASMMatrixHost& host, SPASMMatrixCUDA& cuda);
void freeSPASMCUDA(SPASMMatrixCUDA& cuda);

void spmvCPU(const SPASMMatrixHost& A, const std::vector<float>& x, std::vector<float>& y);

void spmvCUDA(const SPASMMatrixCUDA& A, int numIterations = 1);

bool verifyResults(const std::vector<float>& cpu_result,
                  const std::vector<float>& gpu_result,
                  float tolerance = 1e-4f);

#endif
