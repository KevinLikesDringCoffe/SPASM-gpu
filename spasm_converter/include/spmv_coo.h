#ifndef SPMV_COO_H
#define SPMV_COO_H

#include <vector>
#include <cstdint>
#include <algorithm>
#include "mtx_reader.h"

namespace spmv {

// COO format structure for SpMV (different from spasm::COOMatrix)
struct COOMatrixSpmv {
    uint32_t rows;
    uint32_t cols;
    uint32_t nnz;
    std::vector<uint32_t> row_indices;
    std::vector<uint32_t> col_indices;
    std::vector<float> values;

    COOMatrixSpmv(uint32_t r, uint32_t c) : rows(r), cols(c), nnz(0) {}
};

// COO SpMV implementation
// y = A * x
inline void cooSpMV(const COOMatrixSpmv& A, const std::vector<float>& x, std::vector<float>& y) {
    // Initialize output vector to zero
    std::fill(y.begin(), y.end(), 0.0f);

    // Perform SpMV
    for (uint32_t i = 0; i < A.nnz; i++) {
        uint32_t row = A.row_indices[i];
        uint32_t col = A.col_indices[i];
        float val = A.values[i];

        y[row] += val * x[col];
    }
}

// Convert spasm::COOMatrix to COOMatrixSpmv format
inline COOMatrixSpmv createCOOFromSpasmCOO(const spasm::COOMatrix& spasmCoo) {
    COOMatrixSpmv coo(spasmCoo.rows, spasmCoo.cols);
    coo.nnz = spasmCoo.nnz;

    coo.row_indices.reserve(coo.nnz);
    coo.col_indices.reserve(coo.nnz);
    coo.values.reserve(coo.nnz);

    for (const auto& entry : spasmCoo.entries) {
        coo.row_indices.push_back(entry.row);
        coo.col_indices.push_back(entry.col);
        coo.values.push_back(entry.value);
    }

    return coo;
}

} // namespace spmv

#endif // SPMV_COO_H