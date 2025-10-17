#ifndef MTX_READER_H
#define MTX_READER_H

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdio>

extern "C" {
    #include "../mmio.h"
}

namespace spasm {

// COO format entry
struct COOEntry {
    uint32_t row;
    uint32_t col;
    float value;

    COOEntry() : row(0), col(0), value(0.0f) {}
    COOEntry(uint32_t r, uint32_t c, float v) : row(r), col(c), value(v) {}

    bool operator<(const COOEntry& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

// COO format sparse matrix
class COOMatrix {
public:
    std::vector<COOEntry> entries;
    uint32_t rows;
    uint32_t cols;
    uint32_t nnz;

    COOMatrix() : rows(0), cols(0), nnz(0) {}
    COOMatrix(uint32_t r, uint32_t c) : rows(r), cols(c), nnz(0) {}

    void addEntry(uint32_t row, uint32_t col, float value) {
        entries.emplace_back(row, col, value);
        nnz = entries.size();
    }

    void sortByRowMajor() {
        std::sort(entries.begin(), entries.end());
    }

    // Get all entries in a submatrix region
    std::vector<COOEntry> getEntriesInSubmatrix(uint32_t row_start, uint32_t row_end,
                                                 uint32_t col_start, uint32_t col_end) const {
        std::vector<COOEntry> result;
        for (const auto& entry : entries) {
            if (entry.row >= row_start && entry.row < row_end &&
                entry.col >= col_start && entry.col < col_end) {
                result.push_back(entry);
            }
        }
        return result;
    }

    bool hasNonZeroInTile(uint32_t tile_row, uint32_t tile_col, uint32_t tile_size) const {
        uint32_t row_start = tile_row * tile_size;
        uint32_t row_end = std::min(row_start + tile_size, rows);
        uint32_t col_start = tile_col * tile_size;
        uint32_t col_end = std::min(col_start + tile_size, cols);

        for (const auto& entry : entries) {
            if (entry.row >= row_start && entry.row < row_end &&
                entry.col >= col_start && entry.col < col_end) {
                return true;
            }
        }
        return false;
    }
};

// MTX file reader using MMIO
class MTXReader {
public:
    static COOMatrix readFile(const std::string& filename) {
        FILE* f = fopen(filename.c_str(), "r");
        if (!f) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        MM_typecode matcode;
        int M, N, nz;

        // Read MTX header
        if (mm_read_banner(f, &matcode) != 0) {
            fclose(f);
            throw std::runtime_error("Could not process Matrix Market banner");
        }

        // Check matrix type
        if (!mm_is_coordinate(matcode)) {
            fclose(f);
            throw std::runtime_error("Only coordinate format is supported");
        }

        if (!mm_is_real(matcode) && !mm_is_integer(matcode)) {
            fclose(f);
            throw std::runtime_error("Only real and integer matrices are supported");
        }

        // Read matrix size
        if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
            fclose(f);
            throw std::runtime_error("Could not read matrix size");
        }

        COOMatrix matrix(M, N);
        matrix.entries.reserve(nz);

        // Read entries
        for (int i = 0; i < nz; i++) {
            int row, col;
            double val;
            if (fscanf(f, "%d %d %lg\n", &row, &col, &val) != 3) {
                fclose(f);
                throw std::runtime_error("Error reading matrix entry");
            }
            // Convert to 0-based indexing
            matrix.addEntry(row - 1, col - 1, static_cast<float>(val));
        }

        fclose(f);

        // Sort entries for better access pattern
        matrix.sortByRowMajor();

        std::cout << "Read matrix: " << matrix.rows << "x" << matrix.cols
                  << " with " << matrix.nnz << " non-zeros\n";

        return matrix;
    }
};

} // namespace spasm

#endif // MTX_READER_H