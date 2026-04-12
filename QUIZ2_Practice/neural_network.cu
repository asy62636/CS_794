#include <cstdlib>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cuda/cmath>
#include <cublas_v2.h>

#define BATCH_SIZE 32
#define STRIDE 4
#define NUM_STREAMS 4
#define BLOCK_SIZE 32
#define NEW_BATCH_SIZE (BATCH_SIZE/NUM_STREAMS)
#define NEW_BLOCK_SIZE 32

void initWeights(float* W, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            W[i * N + j] = (float)rand()/(float)RAND_MAX;
        }
    }
    return;
}

void initInput(float* A, int N){
    for(int i = 0; i < N; i++){
        for (int j = 0; j < BATCH_SIZE; j++){
            A[i * BATCH_SIZE + j] = (float)rand()/(float)RAND_MAX;
        }
    }
    return;
}

//--------------CPU Based Implementation for Comparison--------------

void matMulCPU(float* A, float* B, float* C, int N){
    for (int i = 0; i < N; i++){
        for (int k = 0; k < BATCH_SIZE; k++){
            float sum = 0;
            for (int j = 0; j < N; j++){
                sum += A[i * N + j] * B[j * BATCH_SIZE + k];
            }
            C[i * BATCH_SIZE + k] = sum;
        }
    }
    return;
}

void ReLUCPU(float* A, float* B, int N){
    for (int i = 0; i < N; i++){
        for (int k = 0; k < BATCH_SIZE; k++){
            B[i * BATCH_SIZE + k] = max(0.0, A[i * BATCH_SIZE + k]);
        }
    }
    return;
}

//--------------GPU Stream  Implementation - Alternate--------------
__global__ void matMulGPUStreamAlternate(float* A, float* B, float* C, int N, int stream){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float sharedA[STRIDE * BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int M = cuda::ceil_div(N, BLOCK_SIZE);
    float sumS[STRIDE] = {0.0f};

    for (int m = 0; m < M; m++){
        for (int s = 0; s < STRIDE; s++){
            int rowA = ty + by * BLOCK_SIZE * STRIDE + s * BLOCK_SIZE;
            int colA = tx + m * BLOCK_SIZE;
            sharedA[ty + s * BLOCK_SIZE][tx] = (rowA < N && colA < N) ? A[rowA * N + colA] : 0.0f;
        }
        int rowB = ty + m * BLOCK_SIZE;
        int colB = tx + bx * BLOCK_SIZE + stream * NEW_BATCH_SIZE;
        // sharedB[ty][tx] = (rowB < N && colB < NEW_BATCH_SIZE) ? B[rowB * NEW_BATCH_SIZE + colB] : 0.0f;
        sharedB[ty][tx] = (rowB < N && colB < BATCH_SIZE) ? B[rowB * BATCH_SIZE + colB] : 0.0f;
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++){
            for (int s = 0; s < STRIDE; s++){
                sumS[s] += sharedA[ty + s * BLOCK_SIZE][k] * sharedB[k][tx];
            }
        }
        __syncthreads();
    }
    for (int s = 0; s < STRIDE; s++){
        int row = ty + by * BLOCK_SIZE * STRIDE + s * BLOCK_SIZE;
        int col = tx + bx * BLOCK_SIZE + stream * NEW_BATCH_SIZE;
        if (row < N && col < BATCH_SIZE){
            C[row * BATCH_SIZE + col] = sumS[s];
        }
    }
}

__global__ void ReLUGPUStreamAlternate(float* A, float* B, int N, int stream){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = ty + by * BLOCK_SIZE;
    int col = tx + bx * BLOCK_SIZE + stream * NEW_BATCH_SIZE;
    if (row < N && col < BATCH_SIZE){
        B[row * BATCH_SIZE + col] = max(0.0, A[row * BATCH_SIZE + col]);
    }
    return;
}


//--------------GPU Based Implementation with Col Based Multiplication for B/4--------------

__global__ void matMulGPUStream(float* A, float* B, float* C, int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float sharedA[STRIDE * BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int M = cuda::ceil_div(N, BLOCK_SIZE);
    float sumS[STRIDE] = {0.0f};

    for (int m = 0; m < M; m++){
        for (int s = 0; s < STRIDE; s++){
            int rowA = ty + by * BLOCK_SIZE * STRIDE + s * BLOCK_SIZE;
            int colA = tx + m * BLOCK_SIZE;
            sharedA[ty + s * BLOCK_SIZE][tx] = (rowA < N && colA < N) ? A[rowA * N + colA] : 0.0f;
        }
        int rowB = ty + m * BLOCK_SIZE;
        int colB = tx + bx * BLOCK_SIZE;
        sharedB[ty][tx] = (rowB < N && colB < NEW_BATCH_SIZE) ? B[rowB * NEW_BATCH_SIZE + colB] : 0.0f;
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++){
            for (int s = 0; s < STRIDE; s++){
                sumS[s] += sharedA[ty + s * BLOCK_SIZE][k] * sharedB[k][tx];
            }
        }
        __syncthreads();
    }
    for (int s = 0; s < STRIDE; s++){
        int row = ty + by * BLOCK_SIZE * STRIDE + s * BLOCK_SIZE;
        int col = tx + bx * BLOCK_SIZE;
        if (row < N && col < NEW_BATCH_SIZE){
            C[row * NEW_BATCH_SIZE + col] = sumS[s];
        }
    }
}

__global__ void ReLUGPUStream(float* A, float* B, int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = ty + by * BLOCK_SIZE;
    int col = tx + bx * BLOCK_SIZE;
    if (row < N && col < NEW_BATCH_SIZE){
        B[row * NEW_BATCH_SIZE + col] = max(0.0, A[row * NEW_BATCH_SIZE + col]);
    }
    return;
}

void splitInput(float* X, float* batches[NUM_STREAMS], int N){
    for (int i = 0; i < N; i++){
        for (int k = 0; k < BATCH_SIZE; k++){
            //batches[k % NUM_STREAMS][i * NEW_BATCH_SIZE + k/NUM_STREAMS] = X[i * BATCH_SIZE + k];
            batches[k / NEW_BATCH_SIZE][i * NEW_BATCH_SIZE + k % NEW_BATCH_SIZE] = X[i * BATCH_SIZE + k];
        }
    }
    return;
}

void concatenate(float* batches[NUM_STREAMS], float* X, int N){
    for (int i = 0; i < N; i++){
        for (int n = 0; n < NUM_STREAMS; n++){
            for (int k = 0; k < NEW_BATCH_SIZE; k++){
                X[i * BATCH_SIZE + n * NEW_BATCH_SIZE + k] = batches[n][i * NEW_BATCH_SIZE + k];
            }
        }
    }
    return;
}

//--------------GPU Based Implementation with Col Based Multiplication--------------

__global__ void matMulGPU(float* A, float* B, float* C, int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float sharedA[STRIDE * BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int M = cuda::ceil_div(N, BLOCK_SIZE);
    float sumS[STRIDE] = {0.0f};

    for (int m = 0; m < M; m++){
        for (int s = 0; s < STRIDE; s++){
            int rowA = ty + by * BLOCK_SIZE * STRIDE + s * BLOCK_SIZE;
            int colA = tx + m * BLOCK_SIZE;
            sharedA[ty + s * BLOCK_SIZE][tx] = (rowA < N && colA < N) ? A[rowA * N + colA] : 0.0f;
        }
        int rowB = ty + m * BLOCK_SIZE;
        int colB = tx + bx * BLOCK_SIZE;
        sharedB[ty][tx] = (rowB < N && colB < BATCH_SIZE) ? B[rowB * BATCH_SIZE + colB] : 0.0f;
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++){
            for (int s = 0; s < STRIDE; s++){
                sumS[s] += sharedA[ty + s * BLOCK_SIZE][k] * sharedB[k][tx];
            }
        }
        __syncthreads();
    }
    for (int s = 0; s < STRIDE; s++){
        int row = ty + by * BLOCK_SIZE * STRIDE + s * BLOCK_SIZE;
        int col = tx + bx * BLOCK_SIZE;
        if (row < N && col < BATCH_SIZE){
            C[row * BATCH_SIZE + col] = sumS[s];
        }
    }
}

__global__ void ReLUGPU(float* A, float* B, int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = ty + by * BLOCK_SIZE;
    int col = tx + bx * BLOCK_SIZE;
    if (row < N && col < BATCH_SIZE){
        B[row * BATCH_SIZE + col] = max(0.0, A[row * BATCH_SIZE + col]);
    }
    return;
}

bool compare(float* A, float* B, int rows, int cols, float eps = 0.5){
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols;j++){
            if (fabs(A[i * cols + j] - B[i * cols + j]) > eps){
                printf("A[i][j] = %f\n", A[i * cols + j]);
                printf("B[i][j] = %f\n", B[i * cols + j]);
                printf("Error in %d, %d = %f \n", i, j, fabs(A[i * cols + j] - B[i * cols + j]));
                return false;
            } 
        }
    }
    return  true;
}

void neural(int N){
    printf("For N = %d\n", N);
    float* W1 = nullptr;
    float* W2 = nullptr;
    float* X = nullptr;
    float* Y1_CPU = (float*)malloc(N * BATCH_SIZE * sizeof(float));
    float* Y_CPU = (float*)malloc(N * BATCH_SIZE * sizeof(float));
    float* Z_CPU = (float*)malloc(N * BATCH_SIZE * sizeof(float));
    float* Y1 = nullptr;
    float* Y = nullptr;
    float* Z = nullptr;

    cudaMallocManaged(&W1, N * N * sizeof(float));
    cudaMallocManaged(&W2, N * N * sizeof(float));
    cudaMallocManaged(&X, N * BATCH_SIZE * sizeof(float));
    cudaMallocManaged(&Y1, N * BATCH_SIZE * sizeof(float));
    cudaMallocManaged(&Y, N * BATCH_SIZE * sizeof(float));
    cudaMallocManaged(&Z, N * BATCH_SIZE * sizeof(float));

    initWeights(W1, N);
    initWeights(W2, N);
    initInput(X, N);

    matMulCPU(W1, X, Y1_CPU, N);
    ReLUCPU(Y1_CPU, Y_CPU, N);
    matMulCPU(W2, Y_CPU, Z_CPU, N);

    dim3 threadsMult(NEW_BLOCK_SIZE, NEW_BLOCK_SIZE, 1);
    dim3 blocksMult(cuda::ceil_div(BATCH_SIZE, NEW_BLOCK_SIZE), cuda::ceil_div(N, NEW_BLOCK_SIZE * STRIDE), 1);
    dim3 threadsRELU(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocksRELU(cuda::ceil_div(BATCH_SIZE, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE), 1);

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    matMulGPU<<<blocksMult, threadsMult>>>(W1, X, Y1, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Cuda Encountered an error: %s\n", cudaGetErrorString(err));
    }
    // cudaDeviceSynchronize();
    // if (compare(Y1_CPU, Y1, N, BATCH_SIZE)){
    //     printf("SUCCESS in Y1\n");
    // }
    // else{
    //     printf("ERROR in Y1\n");
    // }

    
    ReLUGPU<<<blocksRELU, threadsRELU>>>(Y1, Y, N);
    // cudaDeviceSynchronize();
    // if (compare(Y, Y_CPU, N, BATCH_SIZE)){
    //     printf("SUCCESS in Y\n");
    // }
    // else{
    //     printf("ERROR IN Y\n");
    // }

    matMulGPU<<<blocksMult, threadsMult>>>(W2, Y, Z, N);
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess){
        printf("CUDA encountered an error: %s \n", cudaGetErrorString(err2));
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float ms1 = 0;
    cudaEventElapsedTime(&ms1, start1, stop1);
    printf("Time taken normally = %f ms \n", ms1);
    if (compare(Z_CPU, Z, N, BATCH_SIZE)){
        printf("SUCCESS in Z\n");
    }
    else{
        printf("ERROR in Z\n");
    }
    free(Y1_CPU);
    free(Y_CPU);
    free(Z_CPU);
    cudaFree(Y1);
    cudaFree(Y);

    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    float* inputSplits[NUM_STREAMS];
    float* outputSplits1[NUM_STREAMS];
    float* outputSplits2[NUM_STREAMS];
    float* outputSplits3[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++){
        cudaStreamCreate(&streams[i]);
        cudaMallocManaged(&inputSplits[i], N * NEW_BATCH_SIZE * sizeof(float));
        cudaMallocManaged(&outputSplits1[i], N * NEW_BATCH_SIZE * sizeof(float));
        cudaMallocManaged(&outputSplits2[i], N * NEW_BATCH_SIZE * sizeof(float));
        cudaMallocManaged(&outputSplits3[i], N * NEW_BATCH_SIZE * sizeof(float));
    }
    splitInput(X, inputSplits, N);

    dim3 threadsStream(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocksStream(cuda::ceil_div(NEW_BATCH_SIZE, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE * STRIDE), 1);
    dim3 reluBlocksStream(cuda::ceil_div(NEW_BATCH_SIZE, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE), 1);
    for (int i = 0; i < NUM_STREAMS; i++){
        matMulGPUStream<<<blocksStream, threadsStream, 0, streams[i]>>>(W1, inputSplits[i], outputSplits1[i], N);
        ReLUGPUStream<<<reluBlocksStream, threadsStream, 0, streams[i]>>>(outputSplits1[i], outputSplits2[i], N);
        matMulGPUStream<<<blocksStream, threadsStream, 0, streams[i]>>>(W2, outputSplits2[i], outputSplits3[i], N);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float ms2 = 0.0;
    cudaEventElapsedTime(&ms2, start2, stop2);
    printf("Time taken by streams = %f ms \n", ms2);
    for (int i = 0; i < NUM_STREAMS; i++){
        cudaStreamDestroy(streams[i]);
        cudaFree(outputSplits1[i]);
        cudaFree(outputSplits2[i]);
    }
    float* finalCheck = nullptr;
    cudaMallocManaged(&finalCheck, N * BATCH_SIZE * sizeof(float));
    concatenate(outputSplits3, finalCheck, N);
    for (int i = 0; i < NUM_STREAMS; i++){
        cudaFree(outputSplits3[i]);
    }
    if (compare(finalCheck, Z, N, BATCH_SIZE)){
        printf("Yayyy\n");
    }
    else{
        printf("Nayyy\n");
    }
    cudaFree(finalCheck);

    cudaStream_t multStreams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++){
        cudaStreamCreate(&multStreams[i]);
    }
    float* alt_Y1 = nullptr;
    float* alt_Y = nullptr;
    float* alt_Z = nullptr;
    cudaMallocManaged(&alt_Y1, N * BATCH_SIZE * sizeof(float));
    cudaMallocManaged(&alt_Y, N * BATCH_SIZE * sizeof(float));
    cudaMallocManaged(&alt_Z, N * BATCH_SIZE * sizeof(float));

    dim3 threadStreams(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blockStreams(cuda::ceil_div(NEW_BATCH_SIZE, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE * STRIDE));
    dim3 reluBlockStreams(cuda::ceil_div(NEW_BATCH_SIZE, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE));
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    cudaEventRecord(start3);
    for (int i = 0; i < NUM_STREAMS; i++){
        matMulGPUStreamAlternate<<<blockStreams, threadStreams, 0, multStreams[i]>>>(W1, X, alt_Y1, N, i);
        ReLUGPUStreamAlternate<<<reluBlocksStream, threadStreams, 0, multStreams[i]>>>(alt_Y1, alt_Y, N, i);
        matMulGPUStreamAlternate<<<blockStreams, threadStreams, 0, multStreams[i]>>>(W2, alt_Y, alt_Z, N, i);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    float ms4 = 0.0;
    cudaEventElapsedTime(&ms4, start3, stop3);
    printf("Alternate approach took = %f ms \n", ms4);
    if (compare(Z, alt_Z, N, BATCH_SIZE)){
        printf("Finally finished!\n");
    }
    else{
        printf("Nah brother, long way to go!\n");
    }
    for (int i = 0; i < NUM_STREAMS; i++){
        cudaStreamDestroy(multStreams[i]);
    }

    cudaFree(W1);
    cudaFree(W2);
    cudaFree(X);
    cudaFree(Z);
    cudaFree(alt_Y1);
    cudaFree(alt_Y);
    cudaFree(alt_Z);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);
    printf("\n");
    return;
}


int main(){
    // int N = 1024;
    // neural(N);
    for (int N = 1024; N < 20000; N *= 2){
        neural(N);
    }
    return 0;
}