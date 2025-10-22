#include "include/spasm_cuda.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " <spasm_file> [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -n <iterations>  Number of iterations for GPU benchmark (default: 100)" << std::endl;
    std::cout << "  -h, --help       Print this help message" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    const char* filename = nullptr;
    int numIterations = 100;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-n") == 0) {
            if (i + 1 < argc) {
                numIterations = atoi(argv[++i]);
            } else {
                std::cerr << "Error: -n requires an argument" << std::endl;
                return 1;
            }
        } else if (filename == nullptr) {
            filename = argv[i];
        }
    }

    if (filename == nullptr) {
        std::cerr << "Error: No SPASM file specified" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "SPASM SpMV Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "File: " << filename << std::endl;
    std::cout << "GPU iterations: " << numIterations << std::endl;
    std::cout << std::endl;

    try {
        std::cout << "Loading SPASM matrix..." << std::endl;
        SPASMMatrixHost hostMatrix = loadSPASMFromFile(filename);
        std::cout << std::endl;

        std::vector<float> x(hostMatrix.cols, 1.0f);
        std::vector<float> y_cpu;

        std::cout << "Running CPU SpMV..." << std::endl;
        auto cpuStart = std::chrono::high_resolution_clock::now();
        spmvCPU(hostMatrix, x, y_cpu);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        auto cpuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart);

        std::cout << "CPU SpMV Performance:" << std::endl;
        std::cout << "  Time: " << cpuDuration.count() / 1000.0 << " ms" << std::endl;
        double cpuGflops = (2.0 * hostMatrix.nnz) / (cpuDuration.count() * 1e3);
        std::cout << "  Performance: " << cpuGflops << " GFLOP/s" << std::endl;
        std::cout << std::endl;

        std::cout << "Allocating GPU memory..." << std::endl;
        SPASMMatrixCUDA cudaMatrix;
        allocateSPASMCUDA(hostMatrix, cudaMatrix);

        std::cout << "Copying data to GPU..." << std::endl;
        copySPASMToDevice(hostMatrix, cudaMatrix);

        cudaMemcpy(cudaMatrix.d_x, x.data(), hostMatrix.cols * sizeof(float), cudaMemcpyHostToDevice);

        std::cout << std::endl;
        std::cout << "Running GPU SpMV..." << std::endl;
        spmvCUDA(cudaMatrix, numIterations);
        std::cout << std::endl;

        std::vector<float> y_gpu(hostMatrix.rows);
        cudaMemcpy(y_gpu.data(), cudaMatrix.d_y, hostMatrix.rows * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "Verifying results..." << std::endl;
        bool correct = verifyResults(y_cpu, y_gpu);
        std::cout << std::endl;

        if (correct) {
            std::cout << "========================================" << std::endl;
            std::cout << "SUCCESS: GPU results match CPU results!" << std::endl;
            std::cout << "========================================" << std::endl;

            double speedup = (cpuDuration.count() / 1000.0) /
                           ((cpuGflops > 0) ? (2.0 * hostMatrix.nnz) / (cpuGflops * 1e9) * 1000.0 : 1.0);
            if (cpuGflops > 0) {
                std::cout << "Speedup: " << speedup << "x" << std::endl;
            }
        } else {
            std::cout << "========================================" << std::endl;
            std::cout << "FAILURE: GPU results do NOT match CPU!" << std::endl;
            std::cout << "========================================" << std::endl;
        }

        freeSPASMCUDA(cudaMatrix);

        return correct ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
