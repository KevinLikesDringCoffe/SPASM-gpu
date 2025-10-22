#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void test_pattern_kernel(const uint16_t* input, uint16_t* output) {
    for (int i = 0; i < 16; i++) {
        output[i] = input[i];
    }
}

int main() {
    uint16_t h_masks[16] = {
        0x000F, 0x00F0, 0x0F00, 0xF000,
        0x1111, 0x2222, 0x4444, 0x8888,
        0x0033, 0x00CC, 0x3300, 0xCC00,
        0x8421, 0x4218, 0x2184, 0x1842
    };

    uint16_t* d_input;
    uint16_t* d_output;
    uint16_t h_output[16];

    cudaMalloc(&d_input, 16 * sizeof(uint16_t));
    cudaMalloc(&d_output, 16 * sizeof(uint16_t));

    cudaMemcpy(d_input, h_masks, 16 * sizeof(uint16_t), cudaMemcpyHostToDevice);

    test_pattern_kernel<<<1, 1>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, 16 * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    std::cout << "Testing template masks transfer:\n";
    bool all_correct = true;
    for (int i = 0; i < 16; i++) {
        bool correct = (h_output[i] == h_masks[i]);
        std::cout << "  Template " << i << ": input=0x" << std::hex << h_masks[i]
                  << " output=0x" << h_output[i] << std::dec
                  << " " << (correct ? "OK" : "FAIL") << "\n";
        if (!correct) all_correct = false;
    }

    std::cout << "\nResult: " << (all_correct ? "PASS" : "FAIL") << "\n";

    cudaFree(d_input);
    cudaFree(d_output);
    return all_correct ? 0 : 1;
}
