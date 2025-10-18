# SPASM Converter - Quick Start Guide

## Installation

### Prerequisites
- GCC 7+ or Clang 5+ (C++17 support)
- Make

### Build
```bash
cd spasm_converter
make
```

This creates three executables:
- `mtx2spasm` - Convert matrices to SPASM format
- `spasm_info` - View SPASM file information
- `spmv_compare` - Benchmark and verify SpMV

## Basic Usage

### 1. Convert a Matrix

```bash
./mtx2spasm input.mtx output.spasm
```

Example:
```bash
./mtx2spasm data/1138_bus/1138_bus.mtx my_matrix.spasm
```

### 2. View Matrix Information

```bash
./spasm_info output.spasm
```

For detailed information:
```bash
./spasm_info output.spasm -v
```

### 3. Verify and Benchmark

```bash
./spmv_compare input.mtx output.spasm
```

## Complete Example

```bash
# Convert
./mtx2spasm data/1138_bus/1138_bus.mtx bus.spasm

# View info
./spasm_info bus.spasm

# Benchmark
./spmv_compare data/1138_bus/1138_bus.mtx bus.spasm
```

## Command Reference

### mtx2spasm Options

```bash
./mtx2spasm <input.mtx> [options]

Options:
  -o <output.spasm>          Output file (default: input.spasm)
  --tile-size <N>            Tile size (default: 1024)
  --template-set <0-5>       Template set ID (default: 0)
  --show-patterns <N>        Display top N patterns (default: 8)
  -t <N>                     Short for --tile-size
  -s <0-5>                   Short for --template-set
  -k <N>                     Short for --show-patterns
  -v, --verbose              Verbose output
  --verify                   Verify conversion
  -h, --help                 Show help
```

### spasm_info Options

```bash
./spasm_info <file.spasm> [options]

Options:
  -v, --verbose        Show detailed information
  -p, --patterns       Show all template patterns
  -s, --samples <N>    Number of samples to display (default: 5)
```

### spmv_compare Usage

```bash
./spmv_compare <matrix.mtx> <matrix.spasm>
```

## Tips

1. **Choosing Tile Size**
   - Small matrices (< 1000): Use default 1024
   - Large matrices (> 10000): Try 512 or smaller

2. **Template Sets**
   - Set 0 (default): Best for general matrices
   - Set 1: Row-major patterns
   - Set 2: Column-major patterns
   - Set 3: Block-structured matrices

3. **Performance**
   - SPASM excels on large, structured matrices
   - Small matrices may show COO faster (overhead)
   - Format designed for GPU (future work)

## Troubleshooting

### Build Fails
- Check GCC version: `gcc --version` (need 7+)
- Install build tools: `sudo apt install build-essential`

### Cannot Open File
- Check file path is correct
- Verify file permissions: `ls -l file.mtx`

### SpMV Mismatch
- Small errors (< 1e-6) are normal (floating-point)
- Large errors indicate a bug - please report

## Getting Help

- Full documentation: See `README.md`
- Project structure: See `STRUCTURE.md`
- SpMV details: See `SPMV_README.md`

## Next Steps

1. Try different template sets
2. Experiment with tile sizes
3. Test on your own matrices
4. Profile performance characteristics

For detailed documentation, see `README.md`.
