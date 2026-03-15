#include <cstdlib>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cuda/cmath>
#include <cublas_v2.h>

#define BLOCK_SIZE 32
#define STRIDE 4
#define BATCH_SIZE 32

// ==================== Initialization ====================

void initWeights(float* W, int matSize) {
    for (int i = 0; i < matSize; i++) {
        W[i] = 2.0f * rand() / (float)RAND_MAX - 1.0f;
    }
}

void initInput(float* A, int N) {
    for (int i = 0; i < N * BATCH_SIZE; i++) {
        A[i] = 2.0f * rand() / (float)RAND_MAX - 1.0f;
    }
}

// ==================== Kernels ====================

__global__ void colWiseMult(float* W, float* X, float* C, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float sharedA[BLOCK_SIZE * STRIDE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int M = cuda::ceil_div(N, BLOCK_SIZE);
    float sumS[STRIDE] = {0.0f};

    for (int m = 0; m < M; m++) {
        for (int i = 0; i < STRIDE; i++) {
            int rowA = ty + by * STRIDE * BLOCK_SIZE + i * BLOCK_SIZE;
            int colA = tx + m * BLOCK_SIZE;
            if (rowA < N && colA < N)
                sharedA[ty + i * BLOCK_SIZE][tx] = W[rowA * N + colA];
            else
                sharedA[ty + i * BLOCK_SIZE][tx] = 0.0f;
        }

        int rowB = ty + m * BLOCK_SIZE;
        int colB = tx + bx * BLOCK_SIZE;
        if (rowB < N && colB < BATCH_SIZE)
            sharedB[ty][tx] = X[rowB * BATCH_SIZE + colB];
        else
            sharedB[ty][tx] = 0.0f;

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) {
            for (int i = 0; i < STRIDE; i++) {
                sumS[i] += sharedA[ty + i * BLOCK_SIZE][k] * sharedB[k][tx];
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < STRIDE; i++) {
        int row = ty + by * BLOCK_SIZE * STRIDE + i * BLOCK_SIZE;
        int col = tx + bx * BLOCK_SIZE;
        if (row < N && col < BATCH_SIZE)
            C[row * BATCH_SIZE + col] = sumS[i];
    }
}

__global__ void ReLU(float* A, float* Y, int N) {
    int row = threadIdx.y + blockIdx.y * BLOCK_SIZE;
    int col = threadIdx.x + blockIdx.x * BLOCK_SIZE;

    if (row < N && col < BATCH_SIZE) {
        Y[row * BATCH_SIZE + col] = fmaxf(0.0f, A[row * BATCH_SIZE + col]);
    }
}

// ==================== Forward Pass ====================

void forwardPass(int N) {
    int resSize = N * BATCH_SIZE;
    int matSize = N * N;

    // Allocate inputs and weights
    float *W1 = nullptr, *W2 = nullptr, *X = nullptr;
    cudaMallocManaged(&W1, matSize * sizeof(float));
    cudaMallocManaged(&W2, matSize * sizeof(float));
    cudaMallocManaged(&X, resSize * sizeof(float));

    initWeights(W1, matSize);
    initWeights(W2, matSize);
    initInput(X, N);

    // Allocate intermediate and output buffers
    float *Y_pre = nullptr, *Y = nullptr, *Z = nullptr;
    cudaMallocManaged(&Y_pre, resSize * sizeof(float));  // W1 * X
    cudaMallocManaged(&Y, resSize * sizeof(float));       // ReLU(W1 * X)
    cudaMallocManaged(&Z, resSize * sizeof(float));       // W2 * Y

    // Kernel launch configurations
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 matmulBlocks(cuda::ceil_div(BATCH_SIZE, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE * STRIDE), 1);
    dim3 reluBlocks(cuda::ceil_div(BATCH_SIZE, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE), 1);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Step 1: Y_pre = W1 * X
    colWiseMult<<<matmulBlocks, threads>>>(W1, X, Y_pre, N);
    // cudaDeviceSynchronize();
    // Step 2: Y = ReLU(Y_pre)
    ReLU<<<reluBlocks, threads>>>(Y_pre, Y, N);
    // cudaDeviceSynchronize();
    // Step 3: Z = W2 * Y
    colWiseMult<<<matmulBlocks, threads>>>(W2, Y, Z, N);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;

    cudaEventElapsedTime(&ms, start, stop);
    printf("Time taken for forward pass (for N = %i) = %f ms \n", N, ms);
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(W1);
    cudaFree(W2);
    cudaFree(X);
    cudaFree(Y_pre);
    cudaFree(Y);
    cudaFree(Z);
}

// ==================== Main ====================

int main() {
    printf("Forward pass: Y = ReLU(W1*X), Z = W2*Y\n");
    printf("BLOCK_SIZE = %d, STRIDE = %d, BATCH_SIZE = %d\n\n", BLOCK_SIZE, STRIDE, BATCH_SIZE);

    for (int N = 32; N < 32000; N *= 2) {
        forwardPass(N);
    }

    return 0;
}
