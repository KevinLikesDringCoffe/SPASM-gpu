#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

#include "include/spasm_cuda.h"
#include "utils/timer.h"
#include "../spasm_converter/include/core/format.h"
#include "../spasm_converter/include/io/spasm_io.h"
#include "../spasm_converter/include/spmv/spasm.h"

std::vector<float> generateRandomVector(size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> vec(size);
    for (size_t i = 0; i < size; i++) {
        vec[i] = dist(gen);
    }
    return vec;
}

bool compareResults(const std::vector<float>& y_cpu, const std::vector<float>& y_gpu,
                    float tolerance = 1e-3f) {
    if (y_cpu.size() != y_gpu.size()) {
        std::cerr << "Size mismatch: CPU=" << y_cpu.size() << " GPU=" << y_gpu.size() << "\n";
        return false;
    }

    double maxError = 0.0;
    double avgError = 0.0;
    size_t numErrors = 0;

    for (size_t i = 0; i < y_cpu.size(); i++) {
        double diff = std::abs(y_cpu[i] - y_gpu[i]);
        avgError += diff;
        if (diff > maxError) maxError = diff;

        double relError = (std::abs(y_cpu[i]) > 1e-6) ? diff / std::abs(y_cpu[i]) : diff;
        if (relError > tolerance) {
            numErrors++;
            if (numErrors <= 10) {
                std::cout << "  Error at index " << i << ": CPU=" << y_cpu[i]
                          << " GPU=" << y_gpu[i] << " diff=" << diff << "\n";
            }
        }
    }

    avgError /= y_cpu.size();
    std::cout << "\n[Verification Results]\n";
    std::cout << "  Max error:     " << maxError << "\n";
    std::cout << "  Avg error:     " << avgError << "\n";
    std::cout << "  Num errors:    " << numErrors << " / " << y_cpu.size() << "\n";

    return numErrors == 0;
}

void printMatrixInfo(const spasm::SPASMMatrix& matrix) {
    std::cout << "\n[Matrix Information]\n";
    std::cout << "  Dimensions:       " << matrix.rows << " x " << matrix.cols << "\n";
    std::cout << "  Original NNZ:     " << matrix.originalNnz << "\n";
    std::cout << "  Padded NNZ:       " << matrix.nnz << "\n";
    std::cout << "  Num tiles:        " << matrix.getNumTiles() << "\n";
    std::cout << "  Num blocks:       " << matrix.getNumPositions() << "\n";
    std::cout << "  Num templates:    " << matrix.templatePatterns.size() << "\n";
    std::cout << "  Tile size:        " << matrix.tileSize << "\n";
    std::cout << "  Storage (MB):     " << matrix.getStorageSizeBytes() / (1024.0 * 1024.0) << "\n";
    std::cout << "  Bytes/NNZ:        " << matrix.getBytesPerNonZero() << "\n";
    std::cout << "  Compression:      " << matrix.getCompressionRatio() << "x\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <spasm_file> [num_runs]\n";
        return 1;
    }

    std::string spasmFile = argv[1];
    int numRuns = (argc >= 3) ? std::atoi(argv[2]) : 100;

    std::cout << "========================================\n";
    std::cout << "  SPASM CUDA SpMV Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "SPASM file:  " << spasmFile << "\n";
    std::cout << "Num runs:    " << numRuns << "\n";

    spasm::SPASMMatrix matrix = spasm::SPASMReader::readFromFile(spasmFile);
    printMatrixInfo(matrix);

    std::vector<float> x = generateRandomVector(matrix.cols);
    std::vector<float> y_cpu(matrix.rows, 0.0f);
    std::vector<float> y_gpu(matrix.rows, 0.0f);

    std::cout << "\n[CPU Reference]\n";
    spmv::SPASMSpMV cpu_spmv;
    CPUTimer cpu_timer;
    cpu_timer.start();
    for (int i = 0; i < numRuns; i++) {
        cpu_spmv.spmv(matrix, x, y_cpu);
    }
    cpu_timer.stop();
    double cpu_time_ms = cpu_timer.elapsed_ms() / numRuns;
    double cpu_gflops = (2.0 * matrix.originalNnz) / (cpu_time_ms * 1e6);
    double cpu_throughput = (matrix.originalNnz / 1e9) / (cpu_time_ms / 1000.0);
    std::cout << "  Time per run:     " << std::fixed << std::setprecision(4)
              << cpu_time_ms << " ms\n";
    std::cout << "  Performance:      " << cpu_gflops << " GFLOPS\n";
    std::cout << "  Throughput:       " << cpu_throughput << " GNNZ/s\n";

    std::cout << "\n[GPU Setup]\n";
    std::vector<uint32_t> h_tilePositions;
    std::vector<uint32_t> h_blockToTile(matrix.getNumPositions());
    std::vector<uint16_t> h_templateMasks;

    for (const auto& tp : matrix.tilePositions) {
        h_tilePositions.push_back(tp.tileRowIdx);
        h_tilePositions.push_back(tp.tileColIdx);
    }

    // Build blockToTile mapping
    for (size_t tileIdx = 0; tileIdx < matrix.tileBlockRanges.size(); tileIdx++) {
        const auto& range = matrix.tileBlockRanges[tileIdx];
        for (uint32_t blockIdx = range.blockStart; blockIdx < range.blockEnd; blockIdx++) {
            h_blockToTile[blockIdx] = tileIdx;
        }
    }

    for (const auto& tmpl : matrix.templatePatterns) {
        h_templateMasks.push_back(tmpl.mask);
    }

    SPASMDeviceData devData;
    spasmCudaMalloc(devData,
                    matrix.getNumTiles(),
                    matrix.getNumPositions(),
                    matrix.templatePatterns.size(),
                    matrix.tileSize,
                    matrix.rows,
                    matrix.cols);

    spasmCudaCopy(devData,
                  h_tilePositions.data(),
                  h_blockToTile.data(),
                  matrix.positionEncodings.data(),
                  matrix.values.data(),
                  h_templateMasks.data());

    float *d_x, *d_y;
    cudaMalloc(&d_x, matrix.cols * sizeof(float));
    cudaMalloc(&d_y, matrix.rows * sizeof(float));
    cudaMemcpy(d_x, x.data(), matrix.cols * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "  Memory allocated and copied\n";

    std::cout << "\n[GPU Warmup]\n";
    for (int i = 0; i < 5; i++) {
        spasmCudaSpmv(devData, d_x, d_y);
    }
    std::cout << "  Warmup completed\n";

    std::cout << "\n[GPU Benchmark]\n";
    CUDATimer timer;
    timer.start();
    for (int i = 0; i < numRuns; i++) {
        spasmCudaSpmv(devData, d_x, d_y);
    }
    timer.stop();

    float gpu_time_ms = timer.elapsed_ms() / numRuns;
    double gpu_gflops = (2.0 * matrix.originalNnz) / (gpu_time_ms * 1e6);
    double gpu_throughput = (matrix.originalNnz / 1e9) / (gpu_time_ms / 1000.0);

    std::cout << "  Time per run:     " << std::fixed << std::setprecision(4)
              << gpu_time_ms << " ms\n";
    std::cout << "  Performance:      " << gpu_gflops << " GFLOPS\n";
    std::cout << "  Throughput:       " << gpu_throughput << " GNNZ/s\n";
    std::cout << "  Speedup vs CPU:   " << (cpu_time_ms / gpu_time_ms) << "x\n";

    cudaMemcpy(y_gpu.data(), d_y, matrix.rows * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = compareResults(y_cpu, y_gpu);
    std::cout << "\n[Final Result]\n";
    std::cout << "  Correctness:      " << (correct ? "PASS" : "FAIL") << "\n";

    cudaFree(d_x);
    cudaFree(d_y);
    spasmCudaFree(devData);

    std::cout << "\n========================================\n";

    return correct ? 0 : 1;
}
