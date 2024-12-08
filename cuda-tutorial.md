# Comprehensive CUDA Programming Tutorial

## Table of Contents
1. Introduction to CUDA
2. CUDA Architecture Fundamentals
3. Memory Hierarchy
4. Programming Model
5. Basic CUDA Programming
6. Advanced Concepts
7. Optimization Techniques
8. Practical Examples

## 1. Introduction to CUDA

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model that enables dramatic increases in computing performance by harnessing the power of GPUs (Graphics Processing Units).

### Why CUDA?
- Massive parallelism capabilities
- High memory bandwidth
- Cost-effective computing solution
- Ideal for data-parallel computations

## 2. CUDA Architecture Fundamentals

### Hardware Architecture
- **Streaming Multiprocessors (SMs)**: Basic processing units
- **CUDA Cores**: Individual processing elements within SMs
- **Warp**: Group of 32 threads that execute simultaneously
- **Thread Block**: Group of threads that can cooperate
- **Grid**: Collection of thread blocks

### Thread Hierarchy
1. **Thread**: Smallest execution unit
2. **Block**: Collection of threads
3. **Grid**: Collection of blocks

## 3. Memory Hierarchy

### Types of Memory
1. **Global Memory**
   - Accessible by all threads
   - Highest latency
   - Largest capacity

2. **Shared Memory**
   - Shared within thread block
   - Low latency
   - Limited size

3. **Registers**
   - Per-thread
   - Fastest access
   - Very limited

4. **Constant Memory**
   - Read-only
   - Cached
   - Limited size

5. **Texture Memory**
   - Optimized for 2D spatial locality
   - Read-only
   - Cached

## 4. Programming Model

### Basic Concepts
- **Kernel**: Function that runs on the GPU
- **Host**: CPU and its memory
- **Device**: GPU and its memory
- **Thread Block Synchronization**: `__syncthreads()`
- **Memory Transfer**: Between host and device

### Example of Basic Kernel Structure
```cuda
__global__ void myKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}
```

## 5. CUDA Programming Fundamentals and Syntax

### 5.1 Understanding CUDA Function Declarations

Let's break down CUDA function types in detail:

1. `__global__` Functions:
   - These run on the GPU
   - Can be called from CPU (or GPU in newer CUDA versions)
   - Must return void
   - Are asynchronous when called from CPU
   
2. `__device__` Functions:
   - These run on the GPU
   - Can only be called from other GPU functions
   - Like helper functions for your GPU code
   
3. `__host__` Functions:
   - These run on the CPU
   - Regular C++ functions
   - Can be combined with __device__ to work on both

Here's a practical example with explanation:
CUDA uses special function declarations to specify where the code runs:

- `__global__`: Runs on GPU, called from CPU or GPU
- `__device__`: Runs on GPU, called only from GPU
- `__host__`: Runs on CPU, called from CPU (default)

```cuda
// Function that runs on GPU, called from CPU
__global__ void gpuFunction(int* data) {
    // GPU code here
}

// Function that runs on GPU, called only from GPU
__device__ void deviceHelper(int* data) {
    // GPU helper function
}

// Function that can run on both CPU and GPU
__host__ __device__ void bothFunction(int* data) {
    // Code for both CPU and GPU
}
```

### 5.2 Thread and Block Organization

CUDA provides built-in variables to identify threads:
- `threadIdx`: Thread index within block (x, y, z)
- `blockIdx`: Block index within grid (x, y, z)
- `blockDim`: Size of block (x, y, z)
- `gridDim`: Size of grid (x, y, z)

```cuda
__global__ void threadExample() {
    // Get 1D index for thread
    int tid = threadIdx.x;
    
    // Get 2D indices for thread
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Calculate global index for 1D array
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate global indices for 2D array
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdx2D = globalRow * gridDim.x * blockDim.x + globalCol;
}
```

### 5.3 Basic Programming Examples

### Example 1: Array Initialization with Detailed Explanation

Let's understand a simple array initialization example step by step:

```cuda
// This kernel initializes an array on the GPU
__global__ void initArray(int* array, int n) {
    // Calculate unique index for this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread should work on the array
    if (idx < n) {
        array[idx] = idx;  // Set value to thread index
    }
}

int main() {
    int n = 1000;           // Size of our array
    int *h_array;          // Pointer for CPU (host) array
    int *d_array;          // Pointer for GPU (device) array
    
    // Allocate memory on CPU
    h_array = (int*)malloc(n * sizeof(int));
    
    // Allocate memory on GPU
    // cudaMalloc takes a pointer to a pointer
    cudaMalloc(&d_array, n * sizeof(int));
    
    // Calculate how to split our data across GPU threads
    int threadsPerBlock = 256;  // We'll use 256 threads per block
    
    // Calculate how many blocks we need
    // We use ceiling division to ensure we have enough blocks
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch our kernel
    // <<<blocks, threads>>> is CUDA's special syntax for kernel launch
    initArray<<<blocksPerGrid, threadsPerBlock>>>(d_array, n);
    
    // Copy result back to CPU (if needed)
    cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Clean up
    free(h_array);
    cudaFree(d_array);
    return 0;
}
```

Let's break this down further:

1. **Thread Index Calculation**:
   ```cuda
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   ```
   - `threadIdx.x`: Position of thread within its block (0 to 255 in our case)
   - `blockIdx.x`: Which block this thread is in
   - `blockDim.x`: Size of each block (256 in our case)
   - Example: For thread 5 in block 2 with 256 threads per block:
     - `threadIdx.x = 5`
     - `blockIdx.x = 2`
     - `blockDim.x = 256`
     - `idx = 2 * 256 + 5 = 517`

2. **Kernel Launch Parameters**:
   ```cuda
   int threadsPerBlock = 256;
   int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
   ```
   - If n = 1000 and threadsPerBlock = 256:
   - blocksPerGrid = (1000 + 256 - 1) / 256 = 4
   - This gives us 4 blocks to cover all 1000 elements

3. **Memory Management**:
   ```cuda
   cudaMalloc(&d_array, n * sizeof(int));  // Allocate GPU memory
   cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);  // Copy back to CPU
   ```
   - Memory must be explicitly allocated on GPU using cudaMalloc
   - Data must be explicitly copied between CPU and GPU using cudaMemcpy
   - Direction of copy is specified by last parameter (DeviceToHost or HostToDevice)
```cuda
// Initialize array with thread indices
__global__ void initArray(int* array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        array[idx] = idx;  // Each element gets its index as value
    }
}

int main() {
    int n = 1000;
    int *d_array;
    
    // Allocate device memory
    cudaMalloc(&d_array, n * sizeof(int));
    
    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    initArray<<<blocksPerGrid, threadsPerBlock>>>(d_array, n);
    
    // Clean up
    cudaFree(d_array);
    return 0;
}
```

### Example 2: Element-wise Operations with Detailed Explanation

Let's understand how to perform element-wise operations in CUDA with a simple vector addition example:

```cuda
// Kernel to add two vectors
__global__ void vectorAdd(float* a, float* b, float* result, int n) {
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (id < n) {
        result[id] = a[id] + b[id];
    }
}

int main() {
    // Size of vectors
    int n = 1000;
    size_t size = n * sizeof(float);
    
    // Host vectors (CPU memory)
    float *h_a, *h_b, *h_result;
    
    // Device vectors (GPU memory)
    float *d_a, *d_b, *d_result;
    
    // Allocate memory for host vectors
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_result = (float*)malloc(size);
    
    // Initialize host vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;  // Random numbers between 0 and 1
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate memory for device vectors
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, size);
    
    // Copy inputs to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Number of threads per block (can be tuned for performance)
    int threadsPerBlock = 256;
    
    // Number of blocks in grid
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the Vector Add kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, n);
    
    // Copy result back to host
    cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
    
    // Verify the result (first few elements)
    for (int i = 0; i < 5; i++) {
        printf("%.2f + %.2f = %.2f\n", h_a[i], h_b[i], h_result[i]);
    }
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_result);
    
    return 0;
}
```

Let's break down the key concepts:

1. **Memory Management Pattern**:
   ```cuda
   // CPU (Host) memory allocation
   float *h_a = (float*)malloc(size);
   
   // GPU (Device) memory allocation
   cudaMalloc(&d_a, size);
   
   // Copy from CPU to GPU
   cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
   ```
   This pattern is very common in CUDA:
   - Allocate host memory
   - Allocate device memory
   - Copy data to device
   - Perform computation
   - Copy results back
   - Free memory

2. **Thread Organization**:
   ```cuda
   int threadsPerBlock = 256;
   int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
   ```
   - Each block has 256 threads
   - If n = 1000:
     - We need ceiling(1000/256) = 4 blocks
     - Total threads = 4 * 256 = 1024 threads
     - Extra threads will be filtered out by the if(id < n) check

3. **Kernel Function**:
   ```cuda
   __global__ void vectorAdd(float* a, float* b, float* result, int n) {
       int id = blockIdx.x * blockDim.x + threadIdx.x;
       if (id < n) {
           result[id] = a[id] + b[id];
       }
   }
   ```
   - Each thread handles one addition operation
   - The if check prevents accessing beyond array bounds
   - This is embarrassingly parallel - no thread needs data from other threads
```cuda
// Multiply each element by a scalar
__global__ void scalarMultiply(float* array, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        array[idx] *= scalar;
    }
}

// Add two arrays element-wise
__global__ void arrayAdd(float* a, float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000;
    float *d_a, *d_b, *d_result;
    float scalar = 2.0f;
    
    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_result, n * sizeof(float));
    
    // Set grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernels
    scalarMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_a, scalar, n);
    arrayAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, n);
    
    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    return 0;
}
```

### Example 3: Simple Array Reduction
```cuda
// Sum all elements in an array using reduction
__global__ void reduceSum(float* input, float* output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    int n = 1000;
    float *d_input, *d_output;
    
    // Allocate device memory
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, ((n+255)/256) * sizeof(float));
    
    // Launch kernel
    reduceSum<<<(n+255)/256, 256>>>(d_input, d_output, n);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
```

### Example 4: 2D Array Processing
```cuda
// Process a 2D array (e.g., image processing)
__global__ void process2DArray(float* input, float* output, 
                             int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int idx = row * width + col;
        
        // Example: Simple blur (average of current pixel and neighbors)
        float sum = 0.0f;
        int count = 0;
        
        // Check neighboring pixels
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int newRow = row + i;
                int newCol = col + j;
                
                if (newRow >= 0 && newRow < height && 
                    newCol >= 0 && newCol < width) {
                    sum += input[newRow * width + newCol];
                    count++;
                }
            }
        }
        
        output[idx] = sum / count;
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    float *d_input, *d_output;
    
    // Allocate device memory
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));
    
    // Define block and grid dimensions for 2D
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    process2DArray<<<gridSize, blockSize>>>(d_input, d_output, 
                                          width, height);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
```

### Example 5: Error Handling
```cuda
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", \
               __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    float *d_data;
    int size = 1000;
    
    // Allocate with error checking
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    
    // Launch kernel
    someKernel<<<gridSize, blockSize>>>(d_data);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Check for async errors
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
```

### Hello World Example
```cuda
#include <stdio.h>

__global__ void helloWorld() {
    printf("Hello from thread %d in block %d\n", 
           threadIdx.x, blockIdx.x);
}

int main() {
    // Launch kernel with 2 blocks of 4 threads each
    helloWorld<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### Vector Addition Example
```cuda
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000;
    size_t size = n * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // Initialize arrays
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

## 6. Advanced Concepts

### Atomic Operations
```cuda
__global__ void atomicExample(int* counter) {
    atomicAdd(counter, 1);
}
```

### Shared Memory Usage
```cuda
__global__ void sharedMemExample(float* input, float* output, int n) {
    __shared__ float sharedData[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        sharedData[threadIdx.x] = input[idx];
    }
    
    __syncthreads();
    
    if (idx < n) {
        output[idx] = sharedData[threadIdx.x] * 2.0f;
    }
}
```

## 7. Optimization Techniques

### Memory Coalescing
- Ensure threads in a warp access contiguous memory
- Align data structures to memory boundaries
- Use appropriate data types

### Occupancy Optimization
- Balance registers per thread
- Optimize shared memory usage
- Choose appropriate block sizes

### Bank Conflicts
- Avoid shared memory bank conflicts
- Use padding when necessary
- Understand bank addressing

## 8. Practical Examples

### Matrix Multiplication
```cuda
__global__ void matrixMul(float* A, float* B, float* C, 
                         int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = 
                A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = 
                B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

### Common Pitfalls and Best Practices

1. **Error Checking**
   - Always check CUDA API return values
   - Use error checking macros
   - Implement proper cleanup

2. **Resource Management**
   - Free allocated memory
   - Properly synchronize when necessary
   - Handle device properties appropriately

3. **Performance Considerations**
   - Minimize host-device transfers
   - Use asynchronous operations when possible
   - Profile your code using NVIDIA tools

## Getting Started

1. Install CUDA Toolkit from NVIDIA website
2. Set up development environment
3. Verify installation with deviceQuery sample
4. Start with simple examples and gradually increase complexity

## Additional Resources

1. NVIDIA CUDA Documentation
2. CUDA C++ Best Practices Guide
3. CUDA Sample Projects
4. Online CUDA Communities and Forums

Remember to always profile your code and measure performance improvements. GPU programming requires a different mindset than traditional CPU programming, focusing on data parallelism and memory access patterns.