#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "../include/io/mtx_reader.h"
#include "../include/io/spasm_io.h"
#include "../include/spmv/coo.h"
#include "../include/spmv/spasm.h"

using namespace std;
using namespace chrono;
using namespace spasm;
using namespace spmv;

// Generate random vector
vector<float> generateRandomVector(size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(min_val, max_val);

    vector<float> vec(size);
    for (size_t i = 0; i < size; i++) {
        vec[i] = dist(gen);
    }
    return vec;
}

// Compare two vectors and return metrics
struct ComparisonResult {
    double maxAbsError;
    double avgAbsError;
    double relativeError;
    size_t numDifferences;
    bool isEqual;
};

ComparisonResult compareVectors(const vector<float>& v1, const vector<float>& v2, float tolerance = 1e-2f) {
    ComparisonResult result = {0, 0, 0, 0, true};

    if (v1.size() != v2.size()) {
        cout << "Error: Vector sizes don't match (" << v1.size() << " vs " << v2.size() << ")\n";
        result.isEqual = false;
        return result;
    }

    double sumAbsError = 0;
    double sumSquaredV1 = 0;
    double sumSquaredDiff = 0;

    for (size_t i = 0; i < v1.size(); i++) {
        double diff = abs(v1[i] - v2[i]);
        sumAbsError += diff;
        sumSquaredV1 += v1[i] * v1[i];
        sumSquaredDiff += diff * diff;

        if (diff > result.maxAbsError) {
            result.maxAbsError = diff;
        }

        // Use relative error for large values, absolute error for small values
        double relDiff = 0;
        if (abs(v1[i]) > 1e-6) {
            relDiff = diff / abs(v1[i]);
        } else {
            relDiff = diff;
        }

        if (relDiff > tolerance) {
            result.numDifferences++;
            result.isEqual = false;
        }
    }

    result.avgAbsError = sumAbsError / v1.size();
    if (sumSquaredV1 > 0) {
        result.relativeError = sqrt(sumSquaredDiff) / sqrt(sumSquaredV1);
    }

    return result;
}

// Perform SpMV comparison
void compareSpMV(const string& mtxFile, const string& spasmFile) {
    cout << "\n";
    cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    cout << "║                      SpMV Comparison Benchmark                       ║\n";
    cout << "╚══════════════════════════════════════════════════════════════════════╝\n";
    cout << "\n";

    // Step 1: Read MTX file
    cout << "▶ Input Files\n";
    cout << "  ───────────\n";
    cout << "  • MTX file:    " << mtxFile << "\n";
    cout << "  • SPASM file:  " << spasmFile << "\n\n";

    spasm::COOMatrix spasmCooData;
    try {
        spasmCooData = spasm::MTXReader::readFile(mtxFile);
    } catch (const exception& e) {
        cerr << "Error: Failed to read MTX file: " << e.what() << "\n";
        return;
    }

    if (spasmCooData.entries.empty()) {
        cerr << "Error: Empty matrix\n";
        return;
    }

    uint32_t rows = spasmCooData.rows;
    uint32_t cols = spasmCooData.cols;

    // Step 2: Read SPASM file
    SPASMMatrix spasmMatrix;
    try {
        spasmMatrix = SPASMReader::readFromFile(spasmFile);
    } catch (const exception& e) {
        cerr << "Error reading SPASM file: " << e.what() << "\n";
        return;
    }

    // Print matrix information
    cout << "┌──────────────────────────────────────────────────────────────────────┐\n";
    cout << "│ Matrix Properties                                                     │\n";
    cout << "└──────────────────────────────────────────────────────────────────────┘\n";
    cout << "  • Dimensions:        " << rows << " × " << cols << "\n";
    cout << "  • Non-zeros:         " << spasmCooData.nnz << "\n";
    cout << "  • Sparsity:          " << fixed << setprecision(2)
         << (100.0 - (spasmCooData.nnz * 100.0) / (rows * cols)) << "%\n";
    cout << "  • Tiles:             " << spasmMatrix.getNumTiles() << "\n";
    cout << "  • Position encodings: " << spasmMatrix.getNumPositions() << "\n";
    cout << "  • Templates:         " << spasmMatrix.templatePatterns.size() << "\n\n";

    // Step 3: Create matrices and generate input vector
    COOMatrixSpmv cooMatrix = createCOOFromSpasmCOO(spasmCooData);
    vector<float> x = generateRandomVector(cols);

    // Step 4: Perform COO SpMV
    cout << "┌──────────────────────────────────────────────────────────────────────┐\n";
    cout << "│ Running SpMV Benchmarks                                               │\n";
    cout << "└──────────────────────────────────────────────────────────────────────┘\n";

    vector<float> y_coo(rows, 0.0f);
    auto start = high_resolution_clock::now();
    cooSpMV(cooMatrix, x, y_coo);
    auto end = high_resolution_clock::now();
    auto coo_time = duration_cast<microseconds>(end - start).count();

    cout << "  ► COO format SpMV... ";
    if (coo_time == 0) {
        cout << "< 1 μs\n";
    } else if (coo_time < 1000) {
        cout << coo_time << " μs\n";
    } else {
        cout << fixed << setprecision(2) << coo_time / 1000.0 << " ms\n";
    }

    // Step 5: Perform SPASM SpMV
    vector<float> y_spasm(rows, 0.0f);
    SPASMSpMV spasmSpmv;

    start = high_resolution_clock::now();
    spasmSpmv.spmv(spasmMatrix, x, y_spasm);
    end = high_resolution_clock::now();
    auto spasm_time = duration_cast<microseconds>(end - start).count();

    cout << "  ► SPASM format SpMV... ";
    if (spasm_time == 0) {
        cout << "< 1 μs\n";
    } else if (spasm_time < 1000) {
        cout << spasm_time << " μs\n";
    } else {
        cout << fixed << setprecision(2) << spasm_time / 1000.0 << " ms\n";
    }
    cout << "\n";

    // Step 6: Compare results
    auto comparison = compareVectors(y_coo, y_spasm);

    cout << "┌──────────────────────────────────────────────────────────────────────┐\n";
    cout << "│ Accuracy Verification                                                 │\n";
    cout << "└──────────────────────────────────────────────────────────────────────┘\n";
    cout << "  • Max absolute error:  " << scientific << setprecision(3)
         << comparison.maxAbsError << "\n";
    cout << "  • Avg absolute error:  " << comparison.avgAbsError << "\n";
    cout << "  • Relative error:      " << comparison.relativeError << "\n";
    cout << "  • Result:              " << (comparison.isEqual ? "✓ MATCH" : "✗ MISMATCH") << "\n\n";

    // Step 7: Performance summary
    cout << "┌──────────────────────────────────────────────────────────────────────┐\n";
    cout << "│ Performance Summary                                                   │\n";
    cout << "└──────────────────────────────────────────────────────────────────────┘\n";

    cout << fixed << setprecision(2);
    if (coo_time < 1000 && spasm_time < 1000) {
        cout << "  • COO format:          " << setw(8) << coo_time << " μs\n";
        cout << "  • SPASM format:        " << setw(8) << spasm_time << " μs\n";
    } else {
        cout << "  • COO format:          " << setw(8) << coo_time / 1000.0 << " ms\n";
        cout << "  • SPASM format:        " << setw(8) << spasm_time / 1000.0 << " ms\n";
    }

    if (spasm_time > 0) {
        double speedup = (double)coo_time / spasm_time;
        cout << "  • Speedup:             " << setw(8) << speedup << "x\n";
    }
    cout << "\n";

    // Final result
    cout << "══════════════════════════════════════════════════════════════════════\n";
    if (comparison.isEqual) {
        cout << "  ✓ SUCCESS: SpMV results match!\n";
    } else {
        cout << "  ✗ FAILURE: SpMV results don't match!\n";
    }
    cout << "══════════════════════════════════════════════════════════════════════\n\n";
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <matrix.mtx> <matrix.spasm>\n";
        cout << "Compare SpMV results between COO (MTX) and SPASM formats\n";
        return 1;
    }

    string mtxFile = argv[1];
    string spasmFile = argv[2];

    compareSpMV(mtxFile, spasmFile);

    return 0;
}