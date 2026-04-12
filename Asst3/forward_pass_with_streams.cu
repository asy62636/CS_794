#include <cstdlib>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cuda/cmath>

#define BLOCK_SIZE 32
#define STRIDE 4
#define BATCH_SIZE 32
#define NUM_STREAMS 4
#define STREAM_BATCH (BATCH_SIZE / NUM_STREAMS)

// ==================== Initialization ====================

void initWeights(float* W, int matSize) {
    for (int i = 0; i < matSize; i++)
        W[i] = 2.0f * rand() / (float)RAND_MAX - 1.0f;
}

void initInput(float* A, int N) {
    for (int i = 0; i < N * BATCH_SIZE; i++)
        A[i] = 2.0f * rand() / (float)RAND_MAX - 1.0f;
}

// ==================== Kernels (reference) ====================

__global__ void colWiseMult_ref(float* W, float* X, float* C, int N) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;

    __shared__ float sharedA[BLOCK_SIZE * STRIDE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int M = cuda::ceil_div(N, BLOCK_SIZE);
    float sumS[STRIDE] = {0.0f};

    for (int m = 0; m < M; m++) {
        for (int i = 0; i < STRIDE; i++) {
            int rowA = ty + by * STRIDE * BLOCK_SIZE + i * BLOCK_SIZE;
            int colA = tx + m * BLOCK_SIZE;
            sharedA[ty + i * BLOCK_SIZE][tx] = (rowA < N && colA < N) ? W[rowA * N + colA] : 0.0f;
        }
        int rowB = ty + m * BLOCK_SIZE;
        int colB = tx + bx * BLOCK_SIZE;
        sharedB[ty][tx] = (rowB < N && colB < BATCH_SIZE) ? X[rowB * BATCH_SIZE + colB] : 0.0f;

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
            for (int i = 0; i < STRIDE; i++)
                sumS[i] += sharedA[ty + i * BLOCK_SIZE][k] * sharedB[k][tx];
        __syncthreads();
    }

    for (int i = 0; i < STRIDE; i++) {
        int row = ty + by * BLOCK_SIZE * STRIDE + i * BLOCK_SIZE;
        int col = tx + bx * BLOCK_SIZE;
        if (row < N && col < BATCH_SIZE)
            C[row * BATCH_SIZE + col] = sumS[i];
    }
}

__global__ void ReLU_ref(float* A, float* Y, int N) {
    int row = threadIdx.y + blockIdx.y * BLOCK_SIZE;
    int col = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    if (row < N && col < BATCH_SIZE)
        Y[row * BATCH_SIZE + col] = fmaxf(0.0f, A[row * BATCH_SIZE + col]);
}

// ==================== Kernels (streamed) ====================

__global__ void colWiseMult_streamed(float* W, float* X, float* C, int N, int B) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,  by = blockIdx.y;

    __shared__ float sharedA[BLOCK_SIZE * STRIDE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int M = cuda::ceil_div(N, BLOCK_SIZE);
    float sumS[STRIDE] = {0.0f};

    for (int m = 0; m < M; m++) {
        for (int i = 0; i < STRIDE; i++) {
            int rowA = ty + by * STRIDE * BLOCK_SIZE + i * BLOCK_SIZE;
            int colA = tx + m * BLOCK_SIZE;
            sharedA[ty + i * BLOCK_SIZE][tx] = (rowA < N && colA < N) ? W[rowA * N + colA] : 0.0f;
        }
        int rowB = ty + m * BLOCK_SIZE;
        int colB = tx + bx * BLOCK_SIZE;
        // colB is local to this stream's slice (0..STREAM_BATCH-1)
        // X is offset by s*STREAM_BATCH, so X[rowB * B + colB] correctly lands on
        // the right element: original_X[rowB * BATCH_SIZE + s*STREAM_BATCH + colB]
        sharedB[ty][tx] = (rowB < N && colB < STREAM_BATCH) ? X[rowB * B + colB] : 0.0f;
        //                                    ^^^^^^^^^^^^ was B, should be STREAM_BATCH

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
            for (int i = 0; i < STRIDE; i++)
                sumS[i] += sharedA[ty + i * BLOCK_SIZE][k] * sharedB[k][tx];
        __syncthreads();
    }

    for (int i = 0; i < STRIDE; i++) {
        int row = ty + by * BLOCK_SIZE * STRIDE + i * BLOCK_SIZE;
        int col = tx + bx * BLOCK_SIZE;
        if (row < N && col < STREAM_BATCH)  // ← was B, should be STREAM_BATCH
            C[row * B + col] = sumS[i];
    }
}

__global__ void ReLU_streamed(float* A, float* Y, int N, int B) {
    int row = threadIdx.y + blockIdx.y * BLOCK_SIZE;
    int col = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    if (row < N && col < STREAM_BATCH)      // ← was B, should be STREAM_BATCH
        Y[row * B + col] = fmaxf(0.0f, A[row * B + col]);
}

// ==================== Verify ====================

bool verifyResults(float* Z_ref, float* Z_streamed, int N) {
    float maxErr = 0.0f;
    int worstIdx = -1;
    for (int i = 0; i < N * BATCH_SIZE; i++) {
        float err = fabsf(Z_ref[i] - Z_streamed[i]);
        if (err > maxErr) { maxErr = err; worstIdx = i; }
    }
    bool pass = maxErr < 1e-3f;
    printf("  Verify: max_err = %e at index %d → %s\n",
           maxErr, worstIdx, pass ? "PASSED" : "FAILED");
    return pass;
}

// ==================== Forward Passes ====================

void forwardPassAndVerify(int N) {
    int resSize = N * BATCH_SIZE;
    int matSize = N * N;

    // Shared inputs — same data for both runs
    float *W1, *W2, *X;
    cudaMallocManaged(&W1, matSize * sizeof(float));
    cudaMallocManaged(&W2, matSize * sizeof(float));
    cudaMallocManaged(&X,  resSize * sizeof(float));

    srand(42);  // fixed seed so both runs use identical data
    initWeights(W1, matSize);
    initWeights(W2, matSize);
    initInput(X, N);

    // ---- Reference run ----
    float *Y_pre_ref, *Y_ref, *Z_ref;
    cudaMallocManaged(&Y_pre_ref, resSize * sizeof(float));
    cudaMallocManaged(&Y_ref,     resSize * sizeof(float));
    cudaMallocManaged(&Z_ref,     resSize * sizeof(float));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 matmulBlocks_ref(cuda::ceil_div(BATCH_SIZE,    BLOCK_SIZE),
                          cuda::ceil_div(N, BLOCK_SIZE * STRIDE), 1);
    dim3 reluBlocks_ref(  cuda::ceil_div(BATCH_SIZE,    BLOCK_SIZE),
                          cuda::ceil_div(N, BLOCK_SIZE),           1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    colWiseMult_ref<<<matmulBlocks_ref, threads>>>(W1, X, Y_pre_ref, N);
    ReLU_ref<<<reluBlocks_ref, threads>>>(Y_pre_ref, Y_ref, N);
    colWiseMult_ref<<<matmulBlocks_ref, threads>>>(W2, Y_ref, Z_ref, N);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_ref = 0;
    cudaEventElapsedTime(&ms_ref, start, stop);
    printf("[N=%5d] Reference  : %.4f ms\n", N, ms_ref);

    // ---- Streamed run ----
    float *Y_pre_s, *Y_s, *Z_s;
    cudaMallocManaged(&Y_pre_s, resSize * sizeof(float));
    cudaMallocManaged(&Y_s,     resSize * sizeof(float));
    cudaMallocManaged(&Z_s,     resSize * sizeof(float));

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamCreate(&streams[i]);

    dim3 matmulBlocks_str(cuda::ceil_div(STREAM_BATCH, BLOCK_SIZE),
                          cuda::ceil_div(N, BLOCK_SIZE * STRIDE), 1);
    dim3 reluBlocks_str(  cuda::ceil_div(STREAM_BATCH, BLOCK_SIZE),
                          cuda::ceil_div(N, BLOCK_SIZE),           1);

    cudaEventRecord(start);

    for (int s = 0; s < NUM_STREAMS; s++) {
        int colOffset = s * STREAM_BATCH;
        float* Xs   = X       + colOffset;
        float* Ypre = Y_pre_s + colOffset;
        float* Ys   = Y_s     + colOffset;
        float* Zs   = Z_s     + colOffset;

        colWiseMult_streamed<<<matmulBlocks_str, threads, 0, streams[s]>>>(W1, Xs,   Ypre, N, BATCH_SIZE);
        ReLU_streamed       <<<reluBlocks_str,   threads, 0, streams[s]>>>(    Ypre, Ys,   N, BATCH_SIZE);
        colWiseMult_streamed<<<matmulBlocks_str, threads, 0, streams[s]>>>(W2, Ys,   Zs,   N, BATCH_SIZE);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_str = 0;
    cudaEventElapsedTime(&ms_str, start, stop);
    printf("[N=%5d] Streamed   : %.4f ms\n", N, ms_str);

    // ---- Compare ----
    verifyResults(Z_ref, Z_s, N);
    printf("[N=%5d] Speedup    : %.2fx\n\n", N, ms_ref / ms_str);

    // ---- Cleanup ----
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < NUM_STREAMS; i++) cudaStreamDestroy(streams[i]);
    cudaFree(W1);    cudaFree(W2);    cudaFree(X);
    cudaFree(Y_pre_ref); cudaFree(Y_ref); cudaFree(Z_ref);
    cudaFree(Y_pre_s);   cudaFree(Y_s);   cudaFree(Z_s);
}

// ==================== Main ====================

int main() {
    printf("BLOCK_SIZE=%d, STRIDE=%d, BATCH_SIZE=%d, NUM_STREAMS=%d\n\n",
           BLOCK_SIZE, STRIDE, BATCH_SIZE, NUM_STREAMS);

    for (int N = 32; N < 32000; N *= 2)
        forwardPassAndVerify(N);

    return 0;
}