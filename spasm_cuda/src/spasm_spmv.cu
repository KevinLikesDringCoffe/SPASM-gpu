#include "../include/spasm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>

extern void launchSpasmSpmvKernel(
    const uint32_t* d_tilePositions,
    const uint32_t* d_tileBlockRanges,
    const uint32_t* d_positionEncodings,
    const float* d_values,
    const float* d_x,
    float* d_y,
    uint32_t numTiles,
    uint32_t tileSize,
    uint32_t maxCols,
    uint32_t maxRows);

extern void copyTemplateMasksToConstant(const uint16_t* h_masks, uint32_t numMasks);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void spasmCudaMalloc(SPASMDeviceData& devData,
                     uint32_t numTiles,
                     uint32_t numPositions,
                     uint32_t numTemplates,
                     uint32_t tileSize,
                     uint32_t rows,
                     uint32_t cols)
{
    devData.numTiles = numTiles;
    devData.numPositions = numPositions;
    devData.numTemplates = numTemplates;
    devData.tileSize = tileSize;
    devData.rows = rows;
    devData.cols = cols;

    CUDA_CHECK(cudaMalloc(&devData.d_tilePositions, numTiles * 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&devData.d_tileBlockRanges, numTiles * 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&devData.d_positionEncodings, numPositions * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&devData.d_values, numPositions * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devData.d_templateMasks, numTemplates * sizeof(uint16_t)));
}

void spasmCudaCopy(SPASMDeviceData& devData,
                   const uint32_t* h_tilePositions,
                   const uint32_t* h_tileBlockRanges,
                   const uint32_t* h_positionEncodings,
                   const float* h_values,
                   const uint16_t* h_templateMasks)
{
    CUDA_CHECK(cudaMemcpy(devData.d_tilePositions, h_tilePositions,
                          devData.numTiles * 2 * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(devData.d_tileBlockRanges, h_tileBlockRanges,
                          devData.numTiles * 2 * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(devData.d_positionEncodings, h_positionEncodings,
                          devData.numPositions * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(devData.d_values, h_values,
                          devData.numPositions * 4 * sizeof(float),
                          cudaMemcpyHostToDevice));

    copyTemplateMasksToConstant(h_templateMasks, devData.numTemplates);
}

void spasmCudaSpmv(const SPASMDeviceData& devData,
                   const float* d_x,
                   float* d_y)
{
    CUDA_CHECK(cudaMemset(d_y, 0, devData.rows * sizeof(float)));

    launchSpasmSpmvKernel(
        devData.d_tilePositions,
        devData.d_tileBlockRanges,
        devData.d_positionEncodings,
        devData.d_values,
        d_x,
        d_y,
        devData.numTiles,
        devData.tileSize,
        devData.cols,
        devData.rows
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}

void spasmCudaFree(SPASMDeviceData& devData)
{
    CUDA_CHECK(cudaFree(devData.d_tilePositions));
    CUDA_CHECK(cudaFree(devData.d_tileBlockRanges));
    CUDA_CHECK(cudaFree(devData.d_positionEncodings));
    CUDA_CHECK(cudaFree(devData.d_values));
    CUDA_CHECK(cudaFree(devData.d_templateMasks));
}
