# SPASM Converter - Project Structure

## Directory Layout

```
spasm_converter/
│
├── include/                    # Header files (organized by functionality)
│   ├── core/                   # Core format and data structures
│   │   ├── types.h            # Basic types, position encoding, tile structures
│   │   ├── format.h           # SPASMMatrix class definition
│   │   └── template_patterns.h # Template pattern selection algorithm
│   │
│   ├── io/                     # Input/Output operations
│   │   ├── mtx_reader.h       # Matrix Market (MTX) format reader
│   │   └── spasm_io.h         # SPASM format reader/writer
│   │
│   ├── converter/              # Conversion algorithms
│   │   ├── converter.h        # Main MTX to SPASM converter
│   │   ├── block_decomposer.h # Block decomposition with caching
│   │   └── pattern_analyzer.h # Pattern analysis and decomposition
│   │
│   └── spmv/                   # Sparse matrix-vector multiplication
│       ├── coo.h              # COO format SpMV (reference)
│       └── spasm.h            # SPASM format SpMV (optimized)
│
├── src/                        # Executable source files
│   ├── mtx2spasm.cpp          # Converter tool (MTX → SPASM)
│   ├── spasm_info.cpp         # Information display tool
│   └── spmv_compare.cpp       # SpMV benchmark and verification
│
├── data/                       # Test matrices (Matrix Market format)
│   ├── test_mixed/            # Small test matrix (16×16)
│   ├── 1138_bus/              # Medium test matrix (1138×1138)
│   └── Chebyshev4/            # Large test matrix (68121×68121)
│
├── mmio.h / mmio.c            # External: Matrix Market I/O library
├── Makefile                   # Build configuration
├── README.md                  # Comprehensive user documentation
└── STRUCTURE.md               # This file - project structure overview
```

## Module Descriptions

### Core Module (`include/core/`)

**Purpose**: Fundamental data structures and algorithms for SPASM format

- **types.h**: Basic types and encoding
  - `PatternMask`: 16-bit mask for 4×4 sparsity patterns
  - `PositionEncoding`: 32-bit packed encoding (row, col, template ID, flags)
  - `TilePosition`: Tile-level COO coordinates
  - `TileBlockRange`: Block data range for each tile
  - Helper functions for encoding/decoding positions

- **format.h**: SPASM matrix representation
  - `SPASMMatrix`: Main data structure
    - Tile positions (COO format)
    - Tile block ranges
    - Position encodings (continuous)
    - Values (continuous, 4 per encoding)
    - Template patterns
    - Metadata (dimensions, NNZ, etc.)

- **template_patterns.h**: Template selection algorithm
  - `TemplateSelector`: Implements Algorithm 3 from paper
  - 20 predefined base template patterns
  - 6 template sets for different matrix types
  - Frequency-based pattern selection

### I/O Module (`include/io/`)

**Purpose**: Reading and writing matrix formats

- **mtx_reader.h**: Matrix Market format support
  - `COOMatrix`: Internal COO representation
  - `MTXReader`: Reads .mtx files
  - Handles symmetric/hermitian matrices
  - Converts to zero-based indexing

- **spasm_io.h**: SPASM format I/O
  - `SPASMWriter`: Binary format writer (version 2)
  - `SPASMReader`: Binary format reader
  - Backward compatibility with version 1
  - Efficient binary serialization

### Converter Module (`include/converter/`)

**Purpose**: Converting matrices to SPASM format

- **converter.h**: Main conversion algorithm
  - `OptimizedConverter`: MTX → SPASM conversion
  - Pattern analysis and counting
  - Template selection
  - Block decomposition and encoding
  - Tile organization and block range computation

- **block_decomposer.h**: Block decomposition
  - `CachedPatternDecomposer`: Cached decomposition
  - Thread-local caching for performance
  - Decomposes blocks into selected templates
  - Minimizes padding

- **pattern_analyzer.h**: Pattern analysis
  - `PatternDecomposer`: Core decomposition algorithm
  - Finds best template match for 4×4 blocks
  - Computes padding requirements
  - Extracts values according to pattern

### SpMV Module (`include/spmv/`)

**Purpose**: Sparse matrix-vector multiplication implementations

- **coo.h**: COO SpMV (reference)
  - `COOMatrixSpmv`: Simple COO structure
  - `cooSpMV()`: Standard COO algorithm
  - Used for correctness verification

- **spasm.h**: SPASM SpMV (optimized)
  - `SPASMSpMV`: Template-based SpMV
  - 16 hand-optimized template functions
  - Block-to-tile mapping for O(1) lookup
  - Efficient tile-based processing

## Executables

### mtx2spasm

**Purpose**: Convert Matrix Market files to SPASM format

**Features**:
- Automatic pattern analysis
- Template selection (6 predefined sets)
- Tile organization
- Compression statistics
- Pattern visualization

**Typical usage**:
```bash
./mtx2spasm input.mtx output.spasm --tile-size 1024
```

### spasm_info

**Purpose**: Display information about SPASM files

**Features**:
- Matrix properties and dimensions
- Storage and compression metrics
- Template pattern visualization
- Tile information (verbose mode)
- Sample data display (verbose mode)

**Typical usage**:
```bash
./spasm_info matrix.spasm -v
```

### spmv_compare

**Purpose**: Verify correctness and benchmark SpMV

**Features**:
- Compares COO vs SPASM SpMV
- Random vector generation
- Numerical accuracy verification
- Performance timing
- Detailed error analysis

**Typical usage**:
```bash
./spmv_compare matrix.mtx matrix.spasm
```

## Data Flow

### Conversion Flow (mtx2spasm)

```
MTX File
  ↓
MTXReader → COOMatrix
  ↓
Pattern Analysis → Frequency Count
  ↓
Template Selection → 16 Templates
  ↓
Block Decomposition → Position Encodings + Values
  ↓
Tile Organization → Tile Positions + Block Ranges
  ↓
SPASMWriter → SPASM File
```

### SpMV Flow (spmv_compare)

```
SPASM File → SPASMReader → SPASMMatrix
MTX File → MTXReader → COOMatrix
  ↓                       ↓
SPASM SpMV            COO SpMV
  ↓                       ↓
  └─────── Compare ───────┘
           ↓
    Accuracy Metrics + Performance
```

## Key Algorithms

### 1. Position Encoding (core/types.h)

Packs 32 bits:
```
[31:28] Template ID (4 bits)
[27]    Row end flag (1 bit)
[26:14] Row index (13 bits)
[13]    Column end flag (1 bit)
[12:0]  Column index (13 bits)
```

### 2. Template Selection (core/template_patterns.h)

1. Scan all 4×4 blocks
2. Count pattern frequencies
3. Sort by frequency (descending)
4. Select top 16 patterns
5. Map patterns to template IDs

### 3. Block Decomposition (converter/pattern_analyzer.h)

For each 4×4 block:
1. Extract pattern mask
2. Find best matching template
3. Compute padding if needed
4. Extract values in row-major order
5. Return decomposition result

### 4. SPASM SpMV (spmv/spasm.h)

```cpp
1. Build block-to-tile mapping table
2. For each position encoding:
   a. Decode row, col, template ID
   b. Lookup tile via mapping table
   c. Compute global position
   d. Execute template-specific function
   e. Accumulate to output vector
```

## Dependencies

### Internal Dependencies

- `core/` - Foundation, no dependencies
- `io/` - Depends on `core/`
- `converter/` - Depends on `core/`, `io/`
- `spmv/` - Depends on `core/`, `io/`

### External Dependencies

- **mmio**: Matrix Market I/O library (included)
- **C++ Standard Library**: C++17 features
  - `<vector>`, `<array>`, `<unordered_map>`
  - `<filesystem>` (C++17)
  - `<chrono>` for timing

## Coding Conventions

### Naming

- **Files**: `snake_case.h`, `snake_case.cpp`
- **Classes**: `PascalCase` (e.g., `SPASMMatrix`)
- **Functions**: `camelCase` (e.g., `getPosition`)
- **Variables**: `camelCase` or `snake_case`
- **Constants**: `UPPER_SNAKE_CASE` or `camelCase`
- **Namespaces**: `lowercase` (e.g., `spasm`, `spmv`)

### Header Guards

Format: `SPASM_<MODULE>_<FILE>_H`
- Example: `SPASM_CORE_TYPES_H`
- Example: `SPASM_SPMV_COO_H`

### Documentation

- Public APIs documented with comments
- Complex algorithms include explanation
- File headers describe module purpose

## Build System

### Makefile Targets

- `make` or `make all` - Build all tools
- `make mtx2spasm` - Build converter only
- `make spasm_info` - Build info tool only
- `make spmv_compare` - Build benchmark only
- `make clean` - Remove all build artifacts
- `make install` - Build and show usage info

### Compiler Flags

- `-std=c++17` - C++17 standard
- `-Wall` - All warnings
- `-O3` - Maximum optimization
- `-march=native` - CPU-specific optimization
- `-I./include` - Include directory

## Testing Strategy

### Test Matrices

1. **test_mixed** (16×16, 44 NNZ)
   - Quick sanity test
   - Mixed patterns

2. **1138_bus** (1138×1138, 2,596 NNZ)
   - Medium-sized real matrix
   - Power network data

3. **Chebyshev4** (68,121×68,121, 5,377,761 NNZ)
   - Large-scale matrix
   - Spectral element method

### Verification

- SpMV correctness: Relative error < 1e-6
- Compression: File size reduction verification
- Performance: Timing comparison COO vs SPASM

## Extension Points

### Adding New Template Sets

1. Define patterns in `core/template_patterns.h`
2. Add to `TemplateSelector::selectTemplates()`
3. Update `mtx2spasm.cpp` help text

### Supporting New Input Formats

1. Create reader in `io/` directory
2. Output to `COOMatrix` format
3. Converter handles rest automatically

### Custom SpMV Kernels

1. Add new file in `spmv/` directory
2. Implement SpMV function
3. Add comparison in `spmv_compare.cpp`

## Performance Considerations

### Memory Layout

- **Continuous storage**: Position encodings and values stored continuously
- **Cache-friendly**: Tile-based organization improves locality
- **Compressed**: Reduces memory bandwidth requirements

### Optimization Opportunities

1. **SIMD vectorization**: Template functions can use SSE/AVX
2. **Parallel processing**: Tiles can be processed in parallel
3. **GPU acceleration**: Format designed for GPU (future work)
4. **Persistent mapping**: Cache block-to-tile mapping

## Future Work

- GPU implementation (CUDA/HIP)
- Multi-threaded CPU SpMV
- SIMD-optimized template functions
- Dynamic template selection
- Support for other input formats (HB, Rutherford-Boeing)
- Performance profiling tools
