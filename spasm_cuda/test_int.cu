#include "include/spasm_cuda.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>

extern void allocateSPASMCUDA_int(const SPASMMatrixHost& host, struct SPASMMatrixCUDA_int& cuda);
extern void copySPASMToDevice_int(const SPASMMatrixHost& host, const std::vector<int>& values_int, struct SPASMMatrixCUDA_int& cuda);
extern void freeSPASMCUDA_int(struct SPASMMatrixCUDA_int& cuda);
extern void spmvCUDA_int(const struct SPASMMatrixCUDA_int& A);

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

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " <spasm_file>" << std::endl;
    std::cout << "This program tests integer SpMV to verify correctness without floating-point errors." << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    const char* filename = argv[1];

    std::cout << "========================================" << std::endl;
    std::cout << "SPASM Integer SpMV Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "File: " << filename << std::endl;
    std::cout << std::endl;

    try {
        std::cout << "Loading SPASM matrix..." << std::endl;
        SPASMMatrixHost hostMatrix = loadSPASMFromFile(filename);
        std::cout << std::endl;

        std::cout << "Converting float values to integers..." << std::endl;
        std::vector<int> values_int(hostMatrix.values.size());
        for (size_t i = 0; i < hostMatrix.values.size(); i++) {
            values_int[i] = static_cast<int>(std::round(hostMatrix.values[i] * 1000.0f));
        }
        std::cout << "Scaling factor: 1000 (float * 1000 -> int)" << std::endl;
        std::cout << std::endl;

        std::vector<int> x(hostMatrix.cols, 1);
        std::vector<int> y_cpu;

        std::cout << "Running CPU integer SpMV..." << std::endl;
        spmvCPU_int(hostMatrix, values_int, x, y_cpu);
        std::cout << "CPU SpMV completed" << std::endl;
        std::cout << std::endl;

        std::cout << "Allocating GPU memory..." << std::endl;
        SPASMMatrixCUDA_int cudaMatrix;
        allocateSPASMCUDA_int(hostMatrix, cudaMatrix);

        std::cout << "Copying data to GPU..." << std::endl;
        copySPASMToDevice_int(hostMatrix, values_int, cudaMatrix);

        cudaMemcpy(cudaMatrix.d_x, x.data(), hostMatrix.cols * sizeof(int), cudaMemcpyHostToDevice);

        std::cout << std::endl;
        std::cout << "Running GPU integer SpMV..." << std::endl;
        spmvCUDA_int(cudaMatrix);
        std::cout << "GPU SpMV completed" << std::endl;
        std::cout << std::endl;

        std::vector<int> y_gpu(hostMatrix.rows);
        cudaMemcpy(y_gpu.data(), cudaMatrix.d_y, hostMatrix.rows * sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Verifying results..." << std::endl;
        bool correct = verifyResults_int(y_cpu, y_gpu);
        std::cout << std::endl;

        if (correct) {
            std::cout << "========================================" << std::endl;
            std::cout << "SUCCESS: GPU integer results match CPU!" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << std::endl;
            std::cout << "This confirms the kernel logic is correct." << std::endl;
            std::cout << "Float version errors are due to floating-point" << std::endl;
            std::cout << "rounding differences in atomic operations." << std::endl;
        } else {
            std::cout << "========================================" << std::endl;
            std::cout << "FAILURE: GPU integer results differ!" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << std::endl;
            std::cout << "This indicates a bug in the kernel logic," << std::endl;
            std::cout << "not just floating-point precision issues." << std::endl;
        }

        freeSPASMCUDA_int(cudaMatrix);

        return correct ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
