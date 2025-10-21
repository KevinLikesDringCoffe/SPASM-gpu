#ifndef SPASM_CUDA_TIMER_H
#define SPASM_CUDA_TIMER_H

#include <chrono>
#include <cuda_runtime.h>

class CPUTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        return elapsed.count();
    }
};

class CUDATimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

public:
    CUDATimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~CUDATimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        cudaEventRecord(start_event, 0);
    }

    void stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
    }

    float elapsed_ms() const {
        float elapsed;
        cudaEventElapsedTime(&elapsed, start_event, stop_event);
        return elapsed;
    }
};

#endif
