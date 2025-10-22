#include <iostream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../spasm_converter/include/core/format.h"
#include "../spasm_converter/include/io/spasm_io.h"
#include "../spasm_converter/include/spmv/spasm.h"

__device__ inline void process_pattern_generic(uint16_t pattern, const float* vals, const float* x, float* y) {
    int val_idx = 0;
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            int bit_pos = row * 4 + col;
            if (pattern & (1 << bit_pos)) {
                y[row] += vals[val_idx] * x[col];
                val_idx++;
            }
        }
    }
}

__global__ void debug_single_block(
    uint32_t tileRow, uint32_t tileCol, uint32_t tileSize,
    uint32_t r_idx, uint32_t c_idx, uint32_t t_id,
    const float* values, const uint16_t* templateMasks,
    const float* x, float* y_out, float* debug_info)
{
    uint32_t globalRow = tileRow * tileSize + r_idx * 4;
    uint32_t globalCol = tileCol * tileSize + c_idx * 4;

    float localX[4];
    float localY[4] = {0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        localX[i] = x[globalCol + i];
    }

    uint16_t pattern = templateMasks[t_id];
    process_pattern_generic(pattern, values, localX, localY);

    for (int i = 0; i < 4; i++) {
        y_out[i] = localY[i];
    }

    // Store debug info
    debug_info[0] = globalRow;
    debug_info[1] = globalCol;
    debug_info[2] = pattern;
    debug_info[3] = t_id;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <spasm_file>\n";
        return 1;
    }

    spasm::SPASMMatrix matrix = spasm::SPASMReader::readFromFile(argv[1]);

    // Test first block
    if (matrix.getNumPositions() == 0) {
        std::cerr << "No blocks in matrix\n";
        return 1;
    }

    spasm::PositionEncoding pos0 = matrix.getPosition(0);
    uint32_t c_idx = pos0 & 0x1FFF;
    uint32_t r_idx = (pos0 >> 14) & 0x1FFF;
    uint32_t t_id = (pos0 >> 28) & 0xF;

    uint32_t tileRow = matrix.tilePositions[0].tileRowIdx;
    uint32_t tileCol = matrix.tilePositions[0].tileColIdx;

    std::cout << "Testing first block:\n";
    std::cout << "  Tile: (" << tileRow << ", " << tileCol << ")\n";
    std::cout << "  Block indices: r_idx=" << r_idx << " c_idx=" << c_idx << "\n";
    std::cout << "  Template ID: " << t_id << "\n";
    std::cout << "  Pattern mask: 0x" << std::hex << matrix.templatePatterns[t_id].mask << std::dec << "\n";

    // Create input vector
    std::vector<float> x(matrix.cols, 1.0f);

    // CPU reference
    float cpu_y[4] = {0, 0, 0, 0};
    float localX[4];
    uint32_t globalCol = tileCol * matrix.tileSize + c_idx * 4;
    for (int i = 0; i < 4; i++) {
        localX[i] = x[globalCol + i];
    }

    spmv::blockSpMV_generic(matrix.templatePatterns[t_id].mask,
                            &matrix.values[0], localX, cpu_y);

    std::cout << "\nCPU result: [" << cpu_y[0] << ", " << cpu_y[1] << ", "
              << cpu_y[2] << ", " << cpu_y[3] << "]\n";

    // GPU test
    float *d_x, *d_values, *d_y_out, *d_debug;
    uint16_t *d_masks;

    cudaMalloc(&d_x, matrix.cols * sizeof(float));
    cudaMalloc(&d_values, 4 * sizeof(float));
    cudaMalloc(&d_masks, 16 * sizeof(uint16_t));
    cudaMalloc(&d_y_out, 4 * sizeof(float));
    cudaMalloc(&d_debug, 4 * sizeof(float));

    cudaMemcpy(d_x, x.data(), matrix.cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, &matrix.values[0], 4 * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<uint16_t> h_masks;
    for (const auto& t : matrix.templatePatterns) {
        h_masks.push_back(t.mask);
    }
    cudaMemcpy(d_masks, h_masks.data(), h_masks.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    debug_single_block<<<1, 1>>>(
        tileRow, tileCol, matrix.tileSize,
        r_idx, c_idx, t_id,
        d_values, d_masks, d_x, d_y_out, d_debug
    );

    float gpu_y[4];
    float debug_info[4];
    cudaMemcpy(gpu_y, d_y_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_info, d_debug, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "GPU result: [" << gpu_y[0] << ", " << gpu_y[1] << ", "
              << gpu_y[2] << ", " << gpu_y[3] << "]\n";
    std::cout << "Debug info: globalRow=" << debug_info[0] << " globalCol=" << debug_info[1]
              << " pattern=" << std::hex << (int)debug_info[2] << std::dec
              << " t_id=" << (int)debug_info[3] << "\n";

    bool match = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(cpu_y[i] - gpu_y[i]) > 1e-5) {
            match = false;
            break;
        }
    }

    std::cout << "\nResult: " << (match ? "PASS" : "FAIL") << "\n";

    cudaFree(d_x);
    cudaFree(d_values);
    cudaFree(d_masks);
    cudaFree(d_y_out);
    cudaFree(d_debug);

    return match ? 0 : 1;
}
