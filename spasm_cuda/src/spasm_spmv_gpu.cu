#include "../include/spasm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

extern "C" void launchSPMVKernel(
    const uint32_t* tilePositions,
    const uint32_t* tileBlockRanges,
    const uint32_t* positionEncodings,
    const float* values,
    const uint16_t* templatePatterns,
    const float* x,
    float* y,
    uint32_t rows,
    uint32_t cols,
    uint32_t tileSize,
    uint32_t numTiles,
    uint32_t maxBlocksPerTile);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void allocateSPASMCUDA(const SPASMMatrixHost& host, SPASMMatrixCUDA& cuda) {
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
    CUDA_CHECK(cudaMalloc(&cuda.d_values, host.nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda.d_templatePatterns, host.numTemplates * sizeof(uint16_t)));

    CUDA_CHECK(cudaMalloc(&cuda.d_x, host.cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda.d_y, host.rows * sizeof(float)));
}

void copySPASMToDevice(const SPASMMatrixHost& host, SPASMMatrixCUDA& cuda) {
    CUDA_CHECK(cudaMemcpy(cuda.d_tilePositions, host.tilePositions.data(),
                         host.numTiles * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda.d_tileBlockRanges, host.tileBlockRanges.data(),
                         host.numTiles * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda.d_positionEncodings, host.positionEncodings.data(),
                         host.numPositions * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda.d_values, host.values.data(),
                         host.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda.d_templatePatterns, host.templatePatterns.data(),
                         host.numTemplates * sizeof(uint16_t), cudaMemcpyHostToDevice));
}

void freeSPASMCUDA(SPASMMatrixCUDA& cuda) {
    if (cuda.d_tilePositions) cudaFree(cuda.d_tilePositions);
    if (cuda.d_tileBlockRanges) cudaFree(cuda.d_tileBlockRanges);
    if (cuda.d_positionEncodings) cudaFree(cuda.d_positionEncodings);
    if (cuda.d_values) cudaFree(cuda.d_values);
    if (cuda.d_templatePatterns) cudaFree(cuda.d_templatePatterns);
    if (cuda.d_x) cudaFree(cuda.d_x);
    if (cuda.d_y) cudaFree(cuda.d_y);
}

void spmvCUDA(const SPASMMatrixCUDA& A, int numIterations) {
    const uint32_t threadsPerBlock = 1024;

    std::vector<uint32_t> hostBlockRanges(A.numTiles * 2);
    CUDA_CHECK(cudaMemcpy(hostBlockRanges.data(), A.d_tileBlockRanges,
                         A.numTiles * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint32_t maxBlocksPerTile = 0;
    for (uint32_t i = 0; i < A.numTiles; i++) {
        uint32_t numBlocks = hostBlockRanges[i * 2 + 1] - hostBlockRanges[i * 2];
        maxBlocksPerTile = std::max(maxBlocksPerTile, numBlocks);
    }

    std::cout << "Kernel configuration: " << A.numTiles << " blocks, "
              << threadsPerBlock << " threads per block" << std::endl;
    std::cout << "Max 4x4 blocks per tile: " << maxBlocksPerTile << std::endl;

    CUDA_CHECK(cudaMemset(A.d_y, 0, A.rows * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < numIterations; iter++) {
        if (iter > 0) {
            CUDA_CHECK(cudaMemset(A.d_y, 0, A.rows * sizeof(float)));
        }

        launchSPMVKernel(
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
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    float avgTime = milliseconds / numIterations;
    double gflops = (2.0 * A.nnz * numIterations) / (milliseconds * 1e6);

    std::cout << "GPU SpMV Performance:" << std::endl;
    std::cout << "  Total time: " << milliseconds << " ms (" << numIterations << " iterations)" << std::endl;
    std::cout << "  Average time: " << avgTime << " ms per iteration" << std::endl;
    std::cout << "  Performance: " << gflops << " GFLOP/s" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

bool verifyResults(const std::vector<float>& cpu_result,
                  const std::vector<float>& gpu_result,
                  float tolerance) {
    if (cpu_result.size() != gpu_result.size()) {
        std::cerr << "Result size mismatch: CPU=" << cpu_result.size()
                  << ", GPU=" << gpu_result.size() << std::endl;
        return false;
    }

    int errors = 0;
    float maxError = 0.0f;
    int maxErrorIdx = -1;

    for (size_t i = 0; i < cpu_result.size(); i++) {
        float diff = std::abs(cpu_result[i] - gpu_result[i]);
        float relError = (cpu_result[i] != 0.0f) ? diff / std::abs(cpu_result[i]) : diff;

        if (relError > tolerance) {
            if (errors < 10) {
                std::cerr << "Mismatch at index " << i << ": CPU=" << cpu_result[i]
                          << ", GPU=" << gpu_result[i] << ", error=" << relError << std::endl;
            }
            errors++;
        }

        if (diff > maxError) {
            maxError = diff;
            maxErrorIdx = i;
        }
    }

    if (errors > 0) {
        std::cerr << "Total errors: " << errors << " / " << cpu_result.size() << std::endl;
        std::cerr << "Max error: " << maxError << " at index " << maxErrorIdx << std::endl;
        return false;
    }

    std::cout << "Verification PASSED! Max error: " << maxError << std::endl;
    return true;
}
