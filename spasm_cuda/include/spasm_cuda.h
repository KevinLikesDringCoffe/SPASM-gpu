#ifndef SPASM_CUDA_H
#define SPASM_CUDA_H

#include <cstdint>

typedef uint16_t PatternMask;
typedef uint32_t PositionEncoding;

struct SPASMDeviceData {
    uint32_t* d_tilePositions;
    uint32_t* d_tileBlockRanges;
    uint32_t* d_positionEncodings;
    float* d_values;
    uint16_t* d_templateMasks;

    uint32_t numTiles;
    uint32_t numPositions;
    uint32_t numTemplates;
    uint32_t tileSize;
    uint32_t rows;
    uint32_t cols;
};

void spasmCudaMalloc(SPASMDeviceData& devData,
                     uint32_t numTiles,
                     uint32_t numPositions,
                     uint32_t numTemplates,
                     uint32_t tileSize,
                     uint32_t rows,
                     uint32_t cols);

void spasmCudaCopy(SPASMDeviceData& devData,
                   const uint32_t* h_tilePositions,
                   const uint32_t* h_tileBlockRanges,
                   const uint32_t* h_positionEncodings,
                   const float* h_values,
                   const uint16_t* h_templateMasks);

void spasmCudaSpmv(const SPASMDeviceData& devData,
                   const float* d_x,
                   float* d_y);

void spasmCudaFree(SPASMDeviceData& devData);

#endif
