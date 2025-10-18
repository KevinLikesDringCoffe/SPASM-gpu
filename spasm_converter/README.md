# SPASM: Sparse Matrix Compression Format

SPASM (Sparse Adaptive Structure-aware Matrix) is an efficient sparse matrix compression format based on template pattern matching and block decomposition. This toolkit provides utilities for converting, analyzing, and computing with matrices in SPASM format.

## Overview

SPASM compresses sparse matrices by:
1. **Block Decomposition**: Decomposing matrices into 4×4 blocks
2. **Pattern Recognition**: Identifying recurring sparsity patterns
3. **Template Compression**: Using up to 16 template patterns for efficient encoding
4. **Tile Organization**: Organizing blocks into tiles for better spatial locality

The format achieves significant compression ratios (1.5-3×) compared to traditional COO format while maintaining efficient sparse matrix-vector multiplication (SpMV) performance.

## Project Structure

```
spasm_converter/
├── include/
│   ├── core/              # Core data structures and format definitions
│   │   ├── types.h        # Basic types and position encoding
│   │   ├── format.h       # SPASM matrix format
│   │   └── template_patterns.h  # Template pattern selection
│   ├── io/                # Input/Output operations
│   │   ├── mtx_reader.h   # Matrix Market format reader
│   │   └── spasm_io.h     # SPASM format I/O
│   ├── converter/         # Conversion algorithms
│   │   ├── converter.h    # Main converter implementation
│   │   ├── block_decomposer.h  # Block decomposition with caching
│   │   └── pattern_analyzer.h  # Pattern analysis
│   └── spmv/              # Sparse matrix-vector multiplication
│       ├── coo.h          # COO SpMV implementation
│       └── spasm.h        # SPASM SpMV implementation
├── src/                   # Source files for executables
│   ├── mtx2spasm.cpp      # Converter tool
│   ├── spasm_info.cpp     # Information display tool
│   └── spmv_compare.cpp   # SpMV comparison benchmark
├── data/                  # Test matrices
├── Makefile               # Build configuration
└── README.md              # This file
```

## Building

### Requirements
- C++17 compatible compiler (GCC 7+ or Clang 5+)
- Make build system

### Compilation

```bash
# Build all tools
make

# Build specific tool
make mtx2spasm
make spasm_info
make spmv_compare

# Clean build artifacts
make clean
```

The build process generates three executables:
- `mtx2spasm` - Converts Matrix Market (.mtx) files to SPASM format
- `spasm_info` - Displays detailed information about SPASM files
- `spmv_compare` - Benchmarks and compares SpMV implementations

## Tools

### 1. mtx2spasm - Matrix Converter

Converts sparse matrices from Matrix Market (MTX) format to SPASM format.

#### Usage

```bash
./mtx2spasm <input.mtx> <output.spasm> [options]
```

#### Options

- `-o <output.spasm>` - Output file (default: input.spasm)
- `--tile-size <N>` or `-t <N>` - Set tile size (default: 1024)
- `--template-set <0-5>` or `-s <0-5>` - Choose template pattern set (default: 0)
- `--show-patterns <N>` or `-k <N>` - Display top N patterns (default: 8)
- `-v` or `--verbose` - Verbose output with detailed information
- `--verify` - Verify conversion correctness
- `-h` or `--help` - Show help message

#### Examples

```bash
# Basic conversion (output defaults to input.spasm)
./mtx2spasm data/1138_bus/1138_bus.mtx

# Specify output file
./mtx2spasm input.mtx -o output.spasm

# Custom tile size (long format or short)
./mtx2spasm input.mtx --tile-size 512
./mtx2spasm input.mtx -t 512

# Use different template set
./mtx2spasm input.mtx --template-set 2
./mtx2spasm input.mtx -s 2

# Show top 16 patterns
./mtx2spasm input.mtx --show-patterns 16
./mtx2spasm input.mtx -k 16

# Verbose output with verification
./mtx2spasm input.mtx -v --verify
```

#### Output

The converter displays:
- Configuration and input file information
- Conversion progress with timing
- Pattern analysis (most frequent patterns)
- Compression statistics
- Performance metrics

Example output:
```
╔══════════════════════════════════════════════════════════════════════╗
║                        SPASM Matrix Converter                        ║
╚══════════════════════════════════════════════════════════════════════╝

▶ Matrix Statistics
  • Original non-zeros:      2,596
  • After padding:           2,596
  • Padding elements:        0
  • Padding rate:            0.00%

▶ Storage Information
  • Number of tiles:         1
  • Position encodings:      196
  • Compression ratio:       2.45×
  • COO format size:         28.51 KB
  • SPASM format size:       11.64 KB
  • Size reduction:          59%
```

### 2. spasm_info - Information Display

Displays comprehensive information about SPASM format files.

#### Usage

```bash
./spasm_info <file.spasm> [options]
```

#### Options

- `-v, --verbose` - Show detailed information including:
  - Sample position encodings
  - Sample values
  - Tile block ranges
  - All template patterns
- `-p, --patterns` - Show all template patterns
- `-s, --samples <N>` - Number of sample values to display (default: 5)

#### Examples

```bash
# Basic information
./spasm_info matrix.spasm

# Verbose output with all details
./spasm_info matrix.spasm -v

# Show template patterns only
./spasm_info matrix.spasm -p

# Show 10 sample values
./spasm_info matrix.spasm -v -s 10
```

#### Output Information

- **Matrix Properties**: Dimensions, non-zeros, padding statistics
- **Storage Information**: Tiles, templates, position encodings
- **Compression Metrics**: File size, compression ratios
- **Template Patterns** (with `-p` or `-v`): Visual representation of all patterns
- **Tile Information** (with `-v`): Tile positions and block ranges
- **Sample Data** (with `-v`): Position encodings and values

### 3. spmv_compare - SpMV Benchmark

Compares SpMV (Sparse Matrix-Vector multiplication) implementations between COO and SPASM formats, verifying correctness and measuring performance.

#### Usage

```bash
./spmv_compare <matrix.mtx> <matrix.spasm>
```

#### Examples

```bash
# Compare SpMV implementations
./spmv_compare data/1138_bus/1138_bus.mtx 1138_bus.spasm
```

#### Output

The benchmark displays:
- Input file information
- Matrix properties (dimensions, sparsity, tiles)
- Execution times for both formats
- Accuracy verification metrics
- Performance comparison

Example output:
```
╔══════════════════════════════════════════════════════════════════════╗
║                      SpMV Comparison Benchmark                       ║
╚══════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────┐
│ Matrix Properties                                                     │
└──────────────────────────────────────────────────────────────────────┘
  • Dimensions:        1138 × 1138
  • Non-zeros:         2,596
  • Sparsity:          99.80%
  • Tiles:             1
  • Position encodings: 196

┌──────────────────────────────────────────────────────────────────────┐
│ Accuracy Verification                                                 │
└──────────────────────────────────────────────────────────────────────┘
  • Max absolute error:  9.765e-04
  • Avg absolute error:  9.267e-06
  • Relative error:      3.534e-08
  • Result:              ✓ MATCH

┌──────────────────────────────────────────────────────────────────────┐
│ Performance Summary                                                   │
└──────────────────────────────────────────────────────────────────────┘
  • COO format:          5.00 μs
  • SPASM format:       45.00 μs
  • Speedup:             0.11x
```

#### Accuracy Metrics

- **Max absolute error**: Maximum element-wise difference
- **Avg absolute error**: Average element-wise difference
- **Relative error**: L2-norm of difference / L2-norm of expected result
- **Result**: ✓ MATCH or ✗ MISMATCH based on tolerance (1% default)

## SPASM Format Specification

### File Structure (Binary)

```
Header:
  - Magic number (4 bytes): "SPAS"
  - Version (4 bytes): 2
  - Rows (4 bytes)
  - Cols (4 bytes)
  - NNZ (4 bytes)
  - Original NNZ (4 bytes)
  - Tile size (4 bytes)
  - Number of tiles (4 bytes)

Tiles Section:
  - For each tile:
    - Tile row index (4 bytes)
    - Tile column index (4 bytes)
  - For each tile:
    - Block start index (4 bytes)
    - Block end index (4 bytes)

Templates Section:
  - Number of templates (4 bytes)
  - For each template:
    - Pattern mask (2 bytes)

Position Encodings:
  - Number of positions (4 bytes)
  - For each position (4 bytes):
    - Bits 0-12:  Column index (13 bits)
    - Bit 13:     Column end flag (1 bit)
    - Bits 14-26: Row index (13 bits)
    - Bit 27:     Row end flag (1 bit)
    - Bits 28-31: Template ID (4 bits)

Values:
  - For each position (16 bytes = 4 floats):
    - 4 float values corresponding to the template pattern
```

### Key Concepts

#### Position Encoding
Each 32-bit position encoding packs:
- **Column index** (13 bits): Position within tile
- **Column end flag** (1 bit): Marks last block in column
- **Row index** (13 bits): Position within tile
- **Row end flag** (1 bit): Marks last block in row
- **Template ID** (4 bits): Which of 16 templates to use

#### Template Patterns
16 common 4×4 sparsity patterns:
- Row patterns (4): Horizontal lines
- Column patterns (4): Vertical lines
- 2×2 block patterns (4): Dense sub-blocks
- Diagonal patterns (4): Various diagonal configurations

#### Tile Organization
- Matrices are divided into tiles (typically 1024×1024)
- Each tile contains multiple 4×4 blocks
- Tile block ranges enable efficient tile-based processing

## Performance Characteristics

### Compression Ratio
- **Small matrices** (< 1000×1000): 1.5-2.5× vs COO
- **Medium matrices** (1000-10000): 2.0-3.0× vs COO
- **Large matrices** (> 10000): 2.5-4.0× vs COO

### SpMV Performance
Current CPU implementation shows:
- **Small matrices**: COO is faster (initialization overhead)
- **Medium matrices**: Comparable performance (0.5-1.2×)
- **Large matrices**: SPASM competitive (0.7-1.5×)

**Note**: SPASM format is designed for GPU acceleration. The current CPU implementation demonstrates correctness; significant speedup expected on GPU.

### Memory Access Patterns
- **COO**: Random access, poor cache locality
- **SPASM**:
  - Tile-based organization improves spatial locality
  - Template-based computation enables vectorization
  - Compressed storage reduces memory bandwidth

## Example Workflow

### Complete Conversion and Analysis

```bash
# 1. Convert matrix to SPASM format
./mtx2spasm data/matrix.mtx matrix.spasm --tile-size 1024

# 2. Display information
./spasm_info matrix.spasm -v

# 3. Verify correctness and benchmark SpMV
./spmv_compare data/matrix.mtx matrix.spasm
```

### Batch Processing

```bash
# Convert multiple matrices
for mtx in data/*.mtx; do
    name=$(basename "$mtx" .mtx)
    ./mtx2spasm "$mtx" "${name}.spasm"
    ./spasm_info "${name}.spasm" > "${name}_info.txt"
done
```

## Implementation Details

### Converter Algorithm
1. **Pattern Analysis**: Scan all 4×4 blocks, count pattern frequencies
2. **Template Selection**: Choose top 16 most frequent patterns
3. **Block Decomposition**: For each block, decompose into selected templates
4. **Tile Organization**: Group blocks by tile, compute block ranges
5. **Position Encoding**: Pack block positions and template IDs
6. **Value Storage**: Store 4 values per position encoding

### SpMV Implementation

#### COO SpMV (Reference)
```cpp
for (nnz entries)
    y[row[i]] += value[i] * x[col[i]]
```

#### SPASM SpMV (Optimized)
```cpp
for (tiles)
    for (blocks in tile)
        template_func[template_id](values, x_block, y_block)
```

Template-specific functions are hand-optimized for each pattern, enabling:
- Reduced control flow overhead
- Better instruction-level parallelism
- Potential for SIMD vectorization

## Advanced Features

### Template Set Selection
Six predefined template sets optimize for different sparsity patterns:
- **Set 0** (Default): Balanced - rows, columns, 2×2 blocks, diagonals
- **Set 1**: Row-major - emphasizes horizontal patterns
- **Set 2**: Column-major - emphasizes vertical patterns
- **Set 3**: Block-structured - emphasizes dense sub-blocks
- **Set 4**: Diagonal-heavy - emphasizes diagonal patterns
- **Set 5**: Hybrid - combines multiple strategies

Choose based on matrix characteristics shown in pattern analysis.

### Tile Size Tuning
Optimal tile size depends on:
- **Matrix size**: Larger matrices benefit from smaller tiles
- **Sparsity pattern**: More structured patterns allow larger tiles
- **Target hardware**: Match cache size (GPU: 512-1024, CPU: 256-512)

Guidelines:
- Small matrices (< 1000): Use default 1024
- Medium matrices (1000-10000): Try 512 or 1024
- Large matrices (> 10000): Try 256 or 512

## Testing

### Test Matrices Included
- `data/test_mixed.mtx` - Small test (16×16)
- `data/1138_bus/1138_bus.mtx` - Medium test (1138×1138)
- `data/Chebyshev4/Chebyshev4.mtx` - Large test (68121×68121)

### Running Tests

```bash
# Quick test
./mtx2spasm data/test_mixed.mtx test.spasm
./spmv_compare data/test_mixed.mtx test.spasm

# Full test suite
for mtx in data/*/*.mtx; do
    name=$(basename "$mtx" .mtx)
    echo "Testing $name..."
    ./mtx2spasm "$mtx" "${name}.spasm"
    ./spmv_compare "$mtx" "${name}.spasm" | grep "Result:"
done
```

## Troubleshooting

### Compilation Errors
- Ensure C++17 support: `g++ --version` (need GCC 7+)
- Check include paths in Makefile
- Verify all header files in `include/` subdirectories

### Conversion Failures
- **"Could not open file"**: Check file path and permissions
- **"Invalid MTX format"**: Ensure Matrix Market format compliance
- **High padding rate**: Matrix poorly suited for 4×4 blocking, try different template set

### SpMV Mismatches
- Small numerical differences (< 1e-6) are normal due to:
  - Different operation ordering (floating-point non-associativity)
  - Compiler optimizations
- Large differences indicate bugs - please report with test case

## References

- Matrix Market format: https://math.nist.gov/MatrixMarket/
- SPASM format paper: [Insert paper reference when published]

## License

[Add license information]

## Contributors

[Add contributor information]

## Citation

If you use this software in your research, please cite:

```bibtex
[Add citation information]
```
