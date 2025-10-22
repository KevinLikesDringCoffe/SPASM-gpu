#include "../include/spasm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

extern "C" void launchSPMVKernel_int(
    const uint32_t* tilePositions,
    const uint32_t* tileBlockRanges,
    const uint32_t* positionEncodings,
    const int* values,
    const uint16_t* templatePatterns,
    const int* x,
    int* y,
    uint32_t rows,
    uint32_t cols,
    uint32_t tileSize,
    uint32_t numTiles,
    uint32_t threadsPerBlock);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

struct SPASMMatrixCUDA_int {
    uint32_t* d_tilePositions;
    uint32_t* d_tileBlockRanges;
    uint32_t* d_positionEncodings;
    int* d_values;
    uint16_t* d_templatePatterns;

    uint32_t rows;
    uint32_t cols;
    uint32_t tileSize;
    uint32_t numTiles;
    uint32_t numPositions;
    uint32_t numTemplates;
    uint64_t nnz;

    int* d_x;
    int* d_y;
};

void allocateSPASMCUDA_int(const SPASMMatrixHost& host, SPASMMatrixCUDA_int& cuda) {
    cuda.rows = host.rows;
    cuda.cols = host.cols;
    cuda.tileSize = host.tileSize;
    cuda.numTiles = host.numTiles;
    cuda.numPositions = host.numPositions;
    cuda.numTemplates = host.numTemplates;
    cuda.nnz = host.nnz;

    CUDA_CHECK(cudaMalloc(&cuda.d_tilePositions, host.numTiles * 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&cuda.d_tileBlockRanges, host.numTiles * 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&cuda.d_positionEncodings, host.numPositions * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&cuda.d_values, host.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cuda.d_templatePatterns, host.numTemplates * sizeof(uint16_t)));

    CUDA_CHECK(cudaMalloc(&cuda.d_x, host.cols * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cuda.d_y, host.rows * sizeof(int)));
}

void copySPASMToDevice_int(const SPASMMatrixHost& host, const std::vector<int>& values_int, SPASMMatrixCUDA_int& cuda) {
    CUDA_CHECK(cudaMemcpy(cuda.d_tilePositions, host.tilePositions.data(),
                         host.numTiles * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda.d_tileBlockRanges, host.tileBlockRanges.data(),
                         host.numTiles * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda.d_positionEncodings, host.positionEncodings.data(),
                         host.numPositions * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda.d_values, values_int.data(),
                         host.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda.d_templatePatterns, host.templatePatterns.data(),
                         host.numTemplates * sizeof(uint16_t), cudaMemcpyHostToDevice));
}

void freeSPASMCUDA_int(SPASMMatrixCUDA_int& cuda) {
    if (cuda.d_tilePositions) cudaFree(cuda.d_tilePositions);
    if (cuda.d_tileBlockRanges) cudaFree(cuda.d_tileBlockRanges);
    if (cuda.d_positionEncodings) cudaFree(cuda.d_positionEncodings);
    if (cuda.d_values) cudaFree(cuda.d_values);
    if (cuda.d_templatePatterns) cudaFree(cuda.d_templatePatterns);
    if (cuda.d_x) cudaFree(cuda.d_x);
    if (cuda.d_y) cudaFree(cuda.d_y);
}

void spmvCUDA_int(const SPASMMatrixCUDA_int& A) {
    const uint32_t threadsPerBlock = 1024;

    std::vector<uint32_t> hostBlockRanges(A.numTiles * 2);
    CUDA_CHECK(cudaMemcpy(hostBlockRanges.data(), A.d_tileBlockRanges,
                         A.numTiles * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint32_t maxBlocksPerTile = 0;
    for (uint32_t i = 0; i < A.numTiles; i++) {
        uint32_t numBlocks = hostBlockRanges[i * 2 + 1] - hostBlockRanges[i * 2];
        maxBlocksPerTile = std::max(maxBlocksPerTile, numBlocks);
    }

    std::cout << "INT kernel configuration: " << A.numTiles << " blocks, "
              << threadsPerBlock << " threads per block" << std::endl;
    std::cout << "Max 4x4 blocks per tile: " << maxBlocksPerTile << std::endl;

    CUDA_CHECK(cudaMemset(A.d_y, 0, A.rows * sizeof(int)));

    launchSPMVKernel_int(
        A.d_tilePositions,
        A.d_tileBlockRanges,
        A.d_positionEncodings,
        A.d_values,
        A.d_templatePatterns,
        A.d_x,
        A.d_y,
        A.rows,
        A.cols,
        A.tileSize,
        A.numTiles,
        threadsPerBlock
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}

bool verifyResults_int(const std::vector<int>& cpu_result,
                       const std::vector<int>& gpu_result) {
    if (cpu_result.size() != gpu_result.size()) {
        std::cerr << "Result size mismatch: CPU=" << cpu_result.size()
                  << ", GPU=" << gpu_result.size() << std::endl;
        return false;
    }

    int errors = 0;
    long long maxError = 0;
    int maxErrorIdx = -1;

    for (size_t i = 0; i < cpu_result.size(); i++) {
        long long diff = std::abs((long long)cpu_result[i] - (long long)gpu_result[i]);

        if (diff != 0) {
            if (errors < 10) {
                std::cerr << "Mismatch at index " << i << ": CPU=" << cpu_result[i]
                          << ", GPU=" << gpu_result[i]
                          << ", diff=" << diff << std::endl;
            }
            errors++;
        }

        if (diff > maxError) {
            maxError = diff;
            maxErrorIdx = i;
        }
    }

    if (errors > 0) {
        std::cerr << "INT Version - Total errors: " << errors << " / " << cpu_result.size() << std::endl;
        std::cerr << "Max error: " << maxError << " at index " << maxErrorIdx << std::endl;
        std::cerr << "  CPU[" << maxErrorIdx << "]=" << cpu_result[maxErrorIdx]
                  << ", GPU[" << maxErrorIdx << "]=" << gpu_result[maxErrorIdx] << std::endl;
        return false;
    }

    std::cout << "INT Version - Verification PASSED! Results are identical." << std::endl;
    return true;
}
