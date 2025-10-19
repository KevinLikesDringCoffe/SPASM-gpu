# GPU Architecture & CUDA Programming Tutorial

## Table of Contents
1. [GPU Hardware Architecture](#part-1-gpu-hardware-architecture)
2. [CUDA Programming Model](#part-2-cuda-programming-model)
3. [SPASM GPU Optimization Strategy](#part-3-spasm-gpu-optimization-strategy)
4. [Learning Resources](#learning-resources)

---

# Part 1: GPU Hardware Architecture

## 1.1 CPU vs GPU: Design Philosophy

### CPU Design Philosophy: Powerful Single-Core Performance
```
┌─────────────────────────┐
│  Control  │  ALU  │ ALU │  ← Complex control units
├───────────┴───────┴─────┤
│   Large Cache (MB)      │  ← Large cache to reduce latency
├─────────────────────────┤
│     DRAM (Memory)       │
└─────────────────────────┘

Characteristics:
- 4-64 cores
- Each core is very powerful
- Lots of transistors for control and cache
- Good for: Complex logic, branchy code
```

### GPU Design Philosophy: Massive Parallelism
```
┌───────────────────────────────────┐
│ SM│SM│SM│SM│SM│SM│SM│SM│SM│...   │  ← Dozens to hundreds of SMs
│ ││ ││ ││ ││ ││ ││ ││ ││ ││        │    (Streaming Multiprocessors)
├───────────────────────────────────┤
│    Small Cache (KB)               │  ← Small cache
├───────────────────────────────────┤
│   High Bandwidth Memory (HBM)    │  ← Ultra-high bandwidth memory
└───────────────────────────────────┘

Characteristics:
- Thousands to tens of thousands of simple cores
- Each core is simple but efficient
- Massive parallelism
- Good for: Data-parallel, regular computations
```

### Performance Comparison (Intel Xeon 8380 vs NVIDIA A100)

| Metric | CPU (Intel Xeon 8380) | GPU (NVIDIA A100) |
|--------|----------------------|-------------------|
| Cores | 40 | 6912 CUDA cores |
| Clock Frequency | 2.3 GHz | 1.4 GHz |
| Peak Performance | ~3 TFLOPS | ~19.5 TFLOPS (FP32) |
| Memory Bandwidth | ~200 GB/s | ~1555 GB/s |
| L2 Cache | 60 MB | 40 MB |

**Why GPU is suitable for SpMV?**
- SpMV is **memory-bound** (limited by memory bandwidth)
- GPU memory bandwidth is **7-8x** faster than CPU!

---

## 1.2 GPU Hardware Hierarchy

```
GPU Chip
│
├─ SM (Streaming Multiprocessor) × 108 (A100 example)
│  │
│  ├─ CUDA Core × 64              ← Execute FP/INT operations
│  ├─ Tensor Core × 4             ← Execute matrix operations (FP16/INT8)
│  ├─ Load/Store Units × 32       ← Memory access units
│  ├─ Special Function Units      ← Special functions (sin, cos, sqrt, etc.)
│  │
│  ├─ Warp Scheduler × 4          ← Schedule 32 threads as a warp
│  │
│  ├─ Register File (64K × 32bit) ← Registers (per SM)
│  ├─ L1 Cache/Shared Memory      ← 192 KB (configurable)
│  │   └─ Can configure L1:Shared ratio:
│  │       - 0:192 KB
│  │       - 64:128 KB
│  │       - 128:64 KB
│  └─ L1 Instruction Cache        ← 32 KB
│
├─ L2 Cache (40 MB) - Shared by all SMs
│
└─ HBM2 Memory (40/80 GB, 1555 GB/s)
```

### Key Concepts

**1. SM (Streaming Multiprocessor)** = GPU's "core"
- Independent compute unit
- Has its own registers and shared memory
- Can run multiple thread blocks simultaneously

**2. CUDA Core** = Execution unit
- 64 per SM (A100)
- Execute floating-point or integer operations
- **NOT** an independent processor!

**3. Warp** = Execution unit of 32 threads
- GPU's **minimum execution unit**
- 32 threads in a warp **must execute the same instruction** (SIMT)
- This is why branches hurt performance!

---

## 1.3 Memory Hierarchy

```
[Fast] ←─────────────────────→ [Large Capacity]

Register (per thread)
  - Private to each thread
  - Access latency: ~1 cycle
  - Capacity: 255 per thread (limited!)
  - Bandwidth: ~20 TB/s

  ↓

Shared Memory (per block)
  - Shared by all threads in a block
  - Access latency: ~30 cycles
  - Capacity: up to 192 KB/SM
  - Bandwidth: ~15 TB/s
  - Explicitly controlled by programmer
  - **Most important optimization tool!**

  ↓

L1 Cache (per SM)
  - Private to each SM
  - Access latency: ~30 cycles
  - Capacity: configurable (shared with Shared Memory)
  - Bandwidth: ~15 TB/s
  - Automatically managed by hardware

  ↓

L2 Cache (shared)
  - Shared by all SMs
  - Access latency: ~200 cycles
  - Capacity: 40 MB (A100)
  - Bandwidth: ~8 TB/s

  ↓

Global Memory (device memory)
  - Accessible by all threads
  - Access latency: ~500 cycles
  - Capacity: 40-80 GB (A100)
  - Bandwidth: ~1.5 TB/s
  - **Main bottleneck!**

  ↓

Host Memory (CPU memory)
  - Accessed via PCIe
  - Latency: ~10,000 cycles
  - Bandwidth: ~32 GB/s (PCIe 4.0 x16)
  - **Should be avoided!**
```

**Optimization Key**:
- Prefer: Register > Shared Memory > L1/L2 > Global Memory
- Reduce global memory access count
- Use coalesced access to reduce memory transactions

---

# Part 2: CUDA Programming Model

## 2.1 Thread Hierarchy

CUDA uses **three-level** parallelism hierarchy:

```
Grid (entire GPU task)
│
├─ Block 0        Block 1        Block 2    ... Block N
│  │              │              │              │
│  ├─ Thread 0    ├─ Thread 0    ├─ Thread 0   ...
│  ├─ Thread 1    ├─ Thread 1    ├─ Thread 1
│  ├─ Thread 2    ├─ Thread 2    ├─ Thread 2
│  └─ ...         └─ ...         └─ ...
│
└─ Each block can have up to 1024 threads (hardware limit)
   Each SM can run multiple blocks simultaneously


3D View Example:

Grid:
  gridDim.x = 4, gridDim.y = 2, gridDim.z = 1

┌─────────┬─────────┬─────────┬─────────┐
│Block(0,0)│Block(1,0)│Block(2,0)│Block(3,0)│
├─────────┼─────────┼─────────┼─────────┤
│Block(0,1)│Block(1,1)│Block(2,1)│Block(3,1)│
└─────────┴─────────┴─────────┴─────────┘

Each Block:
  blockDim.x = 16, blockDim.y = 16, blockDim.z = 1

┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
...
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
   256 threads per block
```

### Built-in Variables

```cpp
// Grid dimensions
dim3 gridDim;     // Grid size (gridDim.x, .y, .z)
dim3 blockIdx;    // Current block index in grid

// Block dimensions
dim3 blockDim;    // Block size (blockDim.x, .y, .z)
dim3 threadIdx;   // Current thread index in block

// Compute global thread ID (most common)
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

---

## 2.2 First CUDA Program

**Vector Addition: C = A + B**

```cpp
// ========== CPU Version ==========
void vector_add_cpu(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// ========== GPU Kernel ==========
__global__ void vector_add_gpu(float* A, float* B, float* C, int N) {
    // Compute index for current thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// ========== Main Function ==========
int main() {
    int N = 1000000;  // 1 million elements
    size_t bytes = N * sizeof(float);

    // 1. Allocate CPU memory
    float *h_A, *h_B, *h_C;  // h = host
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // 2. Allocate GPU memory
    float *d_A, *d_B, *d_C;  // d = device
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 3. Copy data to GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 4. Launch kernel
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    vector_add_gpu<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);

    // 5. Copy result back to CPU
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 6. Verify result
    for (int i = 0; i < N; i++) {
        assert(abs(h_C[i] - (h_A[i] + h_B[i])) < 1e-5);
    }

    // 7. Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    printf("Success!\n");
    return 0;
}
```

### Key Syntax Explanation

1. **`__global__`**: Marks a GPU kernel function
   - Called from CPU, executed on GPU
   - Must return void

2. **`<<<grid, block>>>`**: Kernel launch syntax
   ```cpp
   kernel<<<grid_dim, block_dim, shared_mem_bytes, stream>>>(args);

   // Examples
   kernel<<<100, 256>>>();              // 100 blocks, 256 threads each
   kernel<<<dim3(10,10), 256>>>();      // 10×10 2D grid
   ```

3. **`cudaMalloc/cudaFree`**: GPU memory allocation/deallocation
   - Similar to CPU's malloc/free
   - Returns GPU memory address

4. **`cudaMemcpy`**: CPU↔GPU data transfer
   - This is a **synchronous** operation, blocks CPU
   - **Very slow**! Minimize calls

---

## 2.3 Warp and SIMT Execution Model

**Warp = Group of 32 threads**

```
A Block (256 threads) is divided into warps:

Block (256 threads)
├─ Warp 0: Threads 0-31
├─ Warp 1: Threads 32-63
├─ Warp 2: Threads 64-95
...
└─ Warp 7: Threads 224-255


SIMT Execution (Single Instruction, Multiple Threads):

32 threads in a warp must execute the same instruction!

Cycle 1: All threads execute: int tid = ...;
Cycle 2: All threads execute: if (tid < N)
Cycle 3: All threads execute: C[tid] = A[tid] + B[tid];
```

### Warp Divergence Problem

```cpp
__global__ void bad_kernel(int* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid % 2 == 0) {
        // Branch A: even threads
        data[tid] *= 2;
    } else {
        // Branch B: odd threads
        data[tid] *= 3;
    }
}

Execution:
Warp 0 (threads 0-31):
  - Even threads (0,2,4,...,30) execute branch A, odd threads idle
  - Odd threads (1,3,5,...,31) execute branch B, even threads idle
  → Actual execution time = Branch A time + Branch B time (serial!)
  → Only 50% efficiency!


Optimization: Make threads in same warp take same branch
__global__ void good_kernel(int* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;

    if (warp_id % 2 == 0) {
        data[tid] *= 2;  // Entire warp executes this
    } else {
        data[tid] *= 3;  // Entire warp executes this
    }
}
```

**Implication for SPASM**:
- 16 template patterns → severe warp divergence!
- Solution: Sort by template ID, group same patterns together

---

## 2.4 Memory Access Patterns

### Coalesced Access

```cpp
// ✅ Good pattern - Coalesced
__global__ void coalesced_read(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[tid];  // Sequential access

    // Warp 0 accesses: data[0], data[1], ..., data[31]
    // → 1 memory transaction reads 128 bytes (32 floats)
}

// ❌ Bad pattern - Strided
__global__ void strided_read(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[tid * 32];  // Strided access

    // Warp 0 accesses: data[0], data[32], data[64], ...
    // → 32 memory transactions!
    // → 32x performance degradation!
}

// ❌ Worst pattern - Random
__global__ void random_read(float* data, int* indices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[indices[tid]];  // Random access

    // → Up to 32 memory transactions
    // → Cache almost useless
}
```

**Memory Transaction Granularity**:
- GPU accesses memory in **32-byte segments**
- Reading 32 floats (128 bytes) in a warp requires:
  - Coalesced: 4 transactions (optimal)
  - Strided: 32 transactions (worst)

**Alignment Requirements**:
- Addresses should align to 32/64/128 byte boundaries
- `cudaMalloc` automatically satisfies alignment

---

## 2.5 Shared Memory Usage

### Why Shared Memory?
- Global Memory is slow (~500 cycles)
- Shared Memory is fast (~30 cycles)
- Can significantly improve performance when data is reused

### Matrix Multiplication Example

```cpp
// ❌ Naive version - Repeated global memory reads
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
        // Every iteration reads from global memory!
    }
    C[row * N + col] = sum;
}

// ✅ Shared Memory optimized version
#define TILE_SIZE 16

__global__ void matmul_shared(float* A, float* B, float* C, int N) {
    // Allocate shared memory
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Tiled computation
    for (int tile = 0; tile < N / TILE_SIZE; tile++) {
        // 1. Cooperatively load tile to shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];

        __syncthreads();  // Wait for all threads to finish loading

        // 2. Compute using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            // Read from shared memory - very fast!
        }

        __syncthreads();  // Wait for all threads to finish computing
    }

    C[row * N + col] = sum;
}
```

**Shared Memory Performance Comparison**:
- Naive version: 2N global memory reads per element
- Shared version: 2N/TILE_SIZE global memory reads per element
- Speedup: ~TILE_SIZE× (theoretical)

### Shared Memory Bank Conflicts

```
Shared Memory is organized into 32 banks:

Bank 0:  [0]  [32]  [64]  [96] ...
Bank 1:  [1]  [33]  [65]  [97] ...
Bank 2:  [2]  [34]  [66]  [98] ...
...
Bank 31: [31] [63]  [95] [127] ...

// ✅ No conflict
__shared__ float data[256];
float val = data[threadIdx.x];  // Each thread accesses different bank

// ❌ Bank Conflict
__shared__ float data[256];
float val = data[threadIdx.x * 2];  // Multiple threads access same bank
→ 2x performance degradation
```

---

# Part 3: SPASM GPU Optimization Strategy

## 3.1 Why SPASM is Suitable for GPU

```
Traditional Sparse Format (CSR) Problems:
┌────────────────────────────────┐
│ Row Ptr: [0, 3, 7, 9, ...]    │  ← Irregular access
│ Col Idx: [1,5,7,2,4,6,8,...]  │  ← Random access!
│ Values:  [...]                 │
└────────────────────────────────┘
→ Completely random memory access
→ Coalesced access nearly impossible
→ GPU efficiency <10%


SPASM Format Advantages:
┌────────────────────────────────┐
│ Tiles: COO format (regular)    │
│ Blocks: 4x4 regular structure  │  ← Regular access!
│ Position: continuous storage   │  ← Can vectorized load
│ Values: continuous (4/block)   │  ← Perfect coalesced access
└────────────────────────────────┘
→ Regular memory access
→ Coalesced access
→ GPU efficiency can reach 60-80%
```

## 3.2 Original Optimization Proposal Analysis

### Proposed Strategy
1. **Tile-level vector prefetch to Shared Memory**: Cache input vector chunks for each tile
2. **4x4 Block ensures Coalesced Access**: Utilize 4x4 block structure for merged access
3. **Distribute tiles to different SMs**: Tiles as parallel units assigned to different SMs
4. **Load balancing**: Avoid workload imbalance to maximize parallelism

### Advantages
✅ **Memory bandwidth optimization**: SPASM compression → reduce DRAM transfer → effectively increase bandwidth
✅ **Coalesced Access**: 4x4 block structure naturally suits GPU coalesced access
✅ **Shared Memory reuse**: Multiple blocks in a tile share same input vector segment

### Problems and Improvements

#### Problem 1: Tile Granularity Too Coarse, Low SM Utilization

**Current Status**:
- Default tile size = 1024×1024
- If #tiles < #SMs (e.g., A100 has 108 SMs), many SMs idle
- Number of blocks per tile varies hugely (depends on sparsity pattern)

**Improved Solution**:
```
Hierarchical parallelism:
1. Tile level: Assign to different Blocks (not SMs)
2. Block level (4x4): Assign to different Warps
3. Each Warp processes multiple 4x4 blocks

Grid: (num_tiles, 1, 1)
Block: (WARP_SIZE * NUM_WARPS_PER_BLOCK, 1, 1)  // e.g., 256 threads
```

#### Problem 2: Shared Memory Size Limitation

**Analysis**:
- Tile = 1024 → shared memory needs 1024×4 bytes = 4KB (input vector x)
- Also need 1024×4 bytes = 4KB (output vector y for atomic accumulation)
- **Total 8KB - feasible!**

**But consider**:
- Multiple blocks may access different row regions of same tile
- Output vector y atomic operations become bottleneck

**Improved Solution**:
```cuda
__shared__ float tile_x[TILE_SIZE];           // Input vector cache
__shared__ float tile_y_local[TILE_SIZE];     // Local output per block

// 1. Cooperatively load tile_x (coalesced)
for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
    tile_x[i] = x[tileCol * TILE_SIZE + i];
}
__syncthreads();

// 2. Computation phase: write to tile_y_local
// 3. Final atomic add to global y (reduce atomic operations)
```

#### Problem 3: Load Balancing Strategy

Number of blocks per tile varies dramatically!

**Solution A: Dynamic Task Scheduling (Recommended)**
```cuda
// Use global task queue
__device__ int global_block_counter = 0;

__global__ void spasm_spmv_dynamic() {
    while (true) {
        int block_id = atomicAdd(&global_block_counter, 1);
        if (block_id >= total_blocks) break;

        // Process 4x4 block at block_id
        process_block(block_id);
    }
}
```

**Solution B: Tile Sorting + Merge Small Tiles (Static Optimization)**
```cpp
// Preprocessing phase
1. Sort tiles by block count
2. Merge small tiles for one CUDA block to process
3. Large tiles get dedicated CUDA block
```

#### Problem 4: Position Encoding Decode Overhead

**Current**:
```cpp
// Every block must decode position encoding
uint32_t blockRow = (pos >> 14) & 0x1FFF;
uint32_t blockCol = pos & 0x1FFF;
uint32_t templateId = (pos >> 28) & 0xF;
```

**Improvement**:
```cuda
// Pre-decode: decode in CPU or GPU preprocessing kernel
struct DecodedBlock {
    uint16_t globalRow;  // = tileRow*1024 + blockRow*4
    uint16_t globalCol;
    uint8_t templateId;
    uint8_t padding;
};

// GPU just reads directly, no bit operations needed
```

#### Problem 5: Template Pattern Execution Branches

**Current**: 16 template patterns → lots of branches

**Improvements**:
```cuda
// Method 1: Function pointer table (Compute Capability >= 7.0)
__device__ void (*block_kernels[16])(float*, float*, float*) = {...};
block_kernels[templateId](values, x, y);

// Method 2: Sort by template, reduce warp divergence
// Pre-process: group blocks with same template together

// Method 3: Template specialization + unroll most common patterns
if (templateId == 12) {  // 0x8421 diagonal (most common)
    // Direct unroll
} else if (templateId == 0) {  // 0x000f row
    // Direct unroll
} else {
    // Generic handling
}
```

#### Problem 6: Coalesced Access Implementation Details

4x4 coalesced access requires careful attention:

**Key point**:
```cuda
// ✅ Correct coalesced access
// 32 threads in warp process 8 4x4 blocks, each thread handles one element
int lane = threadIdx.x % 32;
int block_in_warp = lane / 4;  // 0-7
int elem_in_block = lane % 4;  // 0-3

// Read x vector (coalesced)
float x_val = tile_x[blockCol * 4 + elem_in_block];

// ❌ Wrong: each thread independently processes one 4x4 block
// Causes lots of repeated reads and non-coalesced access
```

#### Problem 7: Value Storage Continuity

**SPASM Format Advantage**:
```
values array is continuous, every 4 values correspond to one position encoding
→ Naturally suited for vectorized load!
```

**Optimization**:
```cuda
// Use float4 vectorized load
float4* values_vec = (float4*)values;
float4 v = values_vec[blockIdx];  // Read 4 floats at once

// Or use LDGSTS instruction (Ampere+)
// Directly load from global memory to shared memory
```

---

## 3.3 Complete GPU Kernel Design Proposal

```cuda
__global__ void spasm_spmv_kernel(
    const uint32_t* __restrict__ position_encodings,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    const TilePosition* __restrict__ tile_positions,
    const TileBlockRange* __restrict__ tile_ranges,
    const PatternMask* __restrict__ patterns,
    int num_tiles, int tile_size
) {
    int tile_id = blockIdx.x;
    if (tile_id >= num_tiles) return;

    // Shared memory
    extern __shared__ float smem[];
    float* tile_x = smem;
    float* tile_y = smem + tile_size;

    // 1. Load input vector to shared memory (coalesced)
    int tileCol = tile_positions[tile_id].tileColIdx;
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        tile_x[i] = x[tileCol * tile_size + i];
    }

    // Initialize local output
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        tile_y[i] = 0.0f;
    }
    __syncthreads();

    // 2. Process all blocks in this tile
    TileBlockRange range = tile_ranges[tile_id];
    int num_blocks = range.blockEnd - range.blockStart;
    int tileRow = tile_positions[tile_id].tileRowIdx;

    // Each warp processes one block
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int num_warps = blockDim.x / 32;

    for (int b = warp_id; b < num_blocks; b += num_warps) {
        int block_idx = range.blockStart + b;

        // Decode position (consider pre-decoding)
        uint32_t pos = position_encodings[block_idx];
        int blockRow = (pos >> 14) & 0x1FFF;
        int blockCol = pos & 0x1FFF;
        int templateId = (pos >> 28) & 0xF;

        // Vectorized load values (4 values per block)
        const float* block_values = &values[block_idx * 4];

        // Execute SpMV based on template (warp-level parallelism)
        // Each thread processes one element or row
        execute_template_spmv(
            templateId, patterns[templateId],
            block_values, tile_x + blockCol * 4,
            tile_y + blockRow * 4, lane
        );
    }
    __syncthreads();

    // 3. Write back to global memory (atomic or reduction)
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        if (tile_y[i] != 0.0f) {
            atomicAdd(&y[tileRow * tile_size + i], tile_y[i]);
        }
    }
}
```

### Kernel Configuration

```cpp
// Option A: One Block per Tile (for many tiles)
dim3 grid(num_tiles, 1, 1);
dim3 block(256, 1, 1);  // Multiple blocks per SM
spasm_spmv_kernel_v1<<<grid, block>>>();

// Option B: Multiple Blocks per Tile (for very large tiles)
int blocks_per_tile = (blocks_in_tile + 255) / 256;
dim3 grid(num_tiles * blocks_per_tile, 1, 1);
dim3 block(256, 1, 1);
spasm_spmv_kernel_v2<<<grid, block>>>();

// Option C: Dynamic load balancing
dim3 grid(108 * 2, 1, 1);  // 2x SM count
dim3 block(256, 1, 1);
spasm_spmv_dynamic<<<grid, block>>>();
```

---

## 3.4 Performance Analysis Tools

### NVIDIA Nsight Compute
Analyze single kernel performance:

```bash
# Compile with line info
nvcc -lineinfo -o spmv spmv.cu

# Profile
ncu --set full -o report ./spmv

# Check key metrics in report:
# - Achieved Occupancy
# - Memory Throughput
# - Warp Execution Efficiency
# - Bank Conflicts
# - Branch Divergence
```

### NVIDIA Nsight Systems
Analyze overall program performance:

```bash
nsys profile -o timeline ./spmv

# View:
# - Kernel launch overhead
# - CPU-GPU data transfer time
# - Time distribution of different kernels
```

---

## 3.5 Performance Expectations

| Optimization | Expected Speedup | Memory Overhead | Complexity |
|--------------|------------------|-----------------|------------|
| Shared memory caching | 2-3x | 8KB/block | Low |
| Dynamic load balancing | 1.5-2x | Global counter | Medium |
| Pre-decode position | 1.2-1.5x | +8 bytes/block | Low |
| Vectorized load | 1.1-1.3x | None | Low |
| Template sorting | 1.1-1.2x | Preprocessing time | Medium |

**Overall Expected**: **5-10x** speedup vs naive GPU implementation
**vs CPU version**: **50-100x** speedup (depends on matrix characteristics)

### Important Considerations

1. **Atomic operation bottleneck**: Multiple tiles writing to same output row compete
   - Solution: Use two-phase reduction or block-level buffer

2. **Tile size selection**: 1024 may not be optimal
   - Suggestion: Support 256/512/1024 configurable

3. **Small matrix performance**: Kernel launch overhead may dominate
   - Solution: Multi-matrix batching

---

# Learning Resources

## 1. Official NVIDIA Documentation
- CUDA C Programming Guide
- CUDA Best Practices Guide
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

## 2. Online Courses
- Coursera: "Heterogeneous Parallel Programming" (UIUC)
- Udacity: "Intro to Parallel Programming"

## 3. Books
- "Programming Massively Parallel Processors" (classic textbook)
- "CUDA by Example"
- "Professional CUDA C Programming"

## 4. Practice
- NVIDIA CUDA Samples (included with CUDA Toolkit)
- Start with simple vector addition
- Gradually implement matrix multiplication, reduction, etc.

---

# Recommended Learning Path

Based on the SPASM SpMV project:

## Week 1-2: Basics
- Implement vector addition
- Implement matrix multiplication (naive + shared memory versions)
- Understand coalesced access

## Week 3-4: Advanced
- Implement CSR format SpMV (understand why it's hard to optimize)
- Implement simplified SPASM SpMV (single tile)
- Use Nsight Compute to analyze performance

## Week 5-6: Complete Implementation
- Implement multi-tile SPASM SpMV
- Dynamic load balancing
- Performance tuning

---

# Next Steps

Would you like a **complete SPASM GPU SpMV implementation tutorial**?
Including multiple versions from simple to complex, each with detailed comments and performance analysis.

Key features:
1. Version 1: Naive implementation (baseline)
2. Version 2: Shared memory optimization
3. Version 3: Warp-level optimization
4. Version 4: Dynamic load balancing
5. Version 5: Full optimization with all techniques

Each version includes:
- Complete source code with detailed comments
- Performance analysis and comparison
- Profiling results
- Optimization explanations
