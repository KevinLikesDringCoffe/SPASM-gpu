#include <iostream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../spasm_converter/include/core/format.h"
#include "../spasm_converter/include/io/spasm_io.h"

__global__ void count_blocks_kernel(
    const uint32_t* tileBlockRanges,
    uint32_t numTiles,
    uint32_t* blockCount)
{
    uint32_t tileIdx = blockIdx.x;
    if (tileIdx >= numTiles) return;

    uint32_t blockStart = tileBlockRanges[tileIdx * 2];
    uint32_t blockEnd = tileBlockRanges[tileIdx * 2 + 1];

    uint32_t localCount = 0;
    for (uint32_t posIdx = blockStart + threadIdx.x; posIdx < blockEnd; posIdx += blockDim.x) {
        localCount++;
    }

    atomicAdd(blockCount, localCount);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <spasm_file>\n";
        return 1;
    }

    spasm::SPASMMatrix matrix = spasm::SPASMReader::readFromFile(argv[1]);

    std::cout << "Matrix info:\n";
    std::cout << "  Num tiles: " << matrix.getNumTiles() << "\n";
    std::cout << "  Num blocks (positions): " << matrix.getNumPositions() << "\n";

    // Prepare data
    std::vector<uint32_t> h_ranges;
    for (const auto& r : matrix.tileBlockRanges) {
        h_ranges.push_back(r.blockStart);
        h_ranges.push_back(r.blockEnd);
    }

    uint32_t *d_ranges, *d_count;
    cudaMalloc(&d_ranges, h_ranges.size() * sizeof(uint32_t));
    cudaMalloc(&d_count, sizeof(uint32_t));

    cudaMemcpy(d_ranges, h_ranges.data(), h_ranges.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(uint32_t));

    int blockSize = 256;
    int gridSize = matrix.getNumTiles();

    count_blocks_kernel<<<gridSize, blockSize>>>(d_ranges, matrix.getNumTiles(), d_count);

    uint32_t gpu_count;
    cudaMemcpy(&gpu_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::cout << "\nBlock processing count:\n";
    std::cout << "  Expected (CPU): " << matrix.getNumPositions() << "\n";
    std::cout << "  GPU counted:    " << gpu_count << "\n";
    std::cout << "  Match: " << (gpu_count == matrix.getNumPositions() ? "YES" : "NO") << "\n";

    cudaFree(d_ranges);
    cudaFree(d_count);

    return 0;
}
