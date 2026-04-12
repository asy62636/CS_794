#include <cstdlib>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cuda/cmath>
#include <cublas_v2.h>

#define BLOCK_SIZE 32
#define STRIDE 4

void initMat(float* A, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            A[i * N + j] = rand()/RAND_MAX;
        }
    }
    return;
}

void matMulCPU(float* A, float* B, float* C, int N){
    float sum;
    for (int i = 0; i < N; i++){
        for (int k = 0; k < N; k++){
            sum = 0;
            for (int j = 0; j < N; j++){
                sum += A[i * N + j] * B[j * N + k];
            }
            C[i * N + k] = sum;
        }
    }
    return;
}

bool compareMatrices(float* A, float* B, int N, float epsilon = 0.01){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if (fabs(A[i * N + j] - B[i * N + j]) > epsilon) return false;
        }
    }
    return true;
}

__global__ void matMulSimple(float* A, float* B, float* C, int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = ty + by * BLOCK_SIZE;
    int col = tx + bx * BLOCK_SIZE;

    if (!(row < N && col < N)){
        return;
    }

    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0;
    int M = cuda::ceil_div(N, BLOCK_SIZE);
    for (int m = 0; m < M; m++){
        int rowA = ty + by * BLOCK_SIZE;
        int colA = tx + m * BLOCK_SIZE;
        sharedA[ty][tx] = (rowA < N && colA < N) ? A[rowA * N + colA] : 0.0;
        int rowB = ty + m * BLOCK_SIZE;
        int colB = tx + bx * BLOCK_SIZE;
        sharedB[ty][tx] = (rowB < N && colB < N) ? B[rowB * N + colB] : 0.0;
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++){
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        __syncthreads();
    }
    if (row < N && col < N){
        C[row * N + col] = sum;
    }
}

__global__ void matMulRow(float* A, float* B, float* C, int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE * STRIDE];

    int M = cuda::ceil_div(N, BLOCK_SIZE);
    float sumS[STRIDE] = {0.0f};
    for (int m = 0; m < M; m++){
        int rowA = ty + by * BLOCK_SIZE;
        int colA = tx + m * BLOCK_SIZE;
        sharedA[ty][tx] = (rowA < N && colA < N) ? A[rowA * N + colA] : 0.0;

        for (int s = 0; s < STRIDE; s++){
            int rowB = ty + m * BLOCK_SIZE;
            int colB = tx + bx * BLOCK_SIZE + s * BLOCK_SIZE;
            sharedB[ty][tx + s * BLOCK_SIZE] = (rowB < N && colB < N) ? B[rowB * N + colB] : 0.0;
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++){
            for (int s = 0; s < STRIDE; s++){
                sumS[s] += sharedA[ty][k] * sharedB[k][tx + s * BLOCK_SIZE];
            }
        }
        __syncthreads();
    }
    for (int s = 0; s < STRIDE; s++){
        int row = ty + by * BLOCK_SIZE;
        int col = tx + bx * BLOCK_SIZE + s * BLOCK_SIZE;
        if (row < N && col < N) C[row * N + col] = sumS[s];
    }
    return;
}

__global__ void matMulCol(float* A, float* B, float* C, int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    float sumS[STRIDE] = {0.0f};
    __shared__ float sharedA[BLOCK_SIZE * STRIDE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int M = cuda::ceil_div(N, BLOCK_SIZE);
    for (int m = 0; m < M; m++){
        for (int s = 0; s < STRIDE; s++){
            int rowA = ty + by * BLOCK_SIZE + s * BLOCK_SIZE;
            int colA = tx + m * BLOCK_SIZE;
            sharedA[ty + s * BLOCK_SIZE][tx] = (rowA < N && colA < N) ? A[rowA * N + colA] : 0.0;
        }
        int rowB = ty + m * BLOCK_SIZE;
        int colB = tx + bx * BLOCK_SIZE;
        sharedB[ty][tx] = (rowB < N && colB< N) ? B[rowB * N + colB] : 0.0;
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++){
            for (int s = 0; s < STRIDE; s++){
                sumS[s] += sharedA[ty * s + BLOCK_SIZE][k] * sharedB[k][tx];
            }
        }
        __syncthreads();
    }

    for (int s = 0; s < STRIDE; s++){
        int row = ty + by * BLOCK_SIZE * STRIDE + s * BLOCK_SIZE;
        int col = tx + bx * BLOCK_SIZE;
        if (row < N && col < N) C[row * N + col] = sumS[s];
    }
    return;
}

void matMul(int N){
    int MatSize = N * N;
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonRes = (float*)malloc(MatSize * sizeof(float));

    cudaMallocManaged(&A, MatSize * sizeof(float));
    cudaMallocManaged(&B, MatSize * sizeof(float));
    cudaMallocManaged(&C, MatSize * sizeof(float));

    initMat(A, N);
    initMat(B, N);

    matMulCPU(A, B, comparisonRes, N);

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocks(cuda::ceil_div(N, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE), 1);
    matMulSimple<<<blocks, threads>>>(A, B, C, N);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float ms1 = 0;
    cudaEventElapsedTime(&ms1, start1, stop1);
    printf("Time taken by simple mat mul = %f\n", ms1);
    if (compareMatrices(C, comparisonRes, N)){
        printf("[SUCCESS] In simple tiling \n");
    }
    else{
        printf("[ERROR] In simple tiling \n");
    }
    cudaFree(C);
    
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    float* C_row = nullptr;
    cudaMallocManaged(&C_row, MatSize*sizeof(float));
    dim3 threadsRow(BLOCK_SIZE, BLOCK_SIZE,1);
    dim3 blocksRow(cuda::ceil_div(N, BLOCK_SIZE * STRIDE), cuda::ceil_div(N, BLOCK_SIZE), 1);
    cudaEventRecord(start1);
    matMulRow<<<blocksRow, threadsRow>>>(A, B, C_row, N);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float ms2 = 0;
    cudaEventElapsedTime(&ms2, start2, stop2);
    printf("TIme taken by row wise mat mul = %f \n", ms2);
    if (compareMatrices(C_row, comparisonRes, N)){
        printf("[SUCCESS] In row tiling \n");
    }
    else{
        printf("[ERROR] In row tiling \n");
    }
    cudaFree(C_row);

    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    float* C_col = nullptr;
    cudaMallocManaged(&C_col, MatSize*sizeof(float));
    dim3 threadscol(BLOCK_SIZE, BLOCK_SIZE,1);
    dim3 blocksCol(cuda::ceil_div(N, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE * STRIDE), 1);
    cudaEventRecord(start3);
    matMulCol<<<blocksCol, threadscol>>>(A, B, C_col, N);
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    float ms3 = 0;
    cudaEventElapsedTime(&ms3, start3, stop3);
    printf("Time taken by col wise mat mul = %f \n", ms3);
    if (compareMatrices(C_col, comparisonRes, N)){
        printf("[SUCCESS] In col tiling \n");
    }
    else{
        printf("[ERROR] In col tiling \n");
    }
    cudaFree(C_col);

    cudaFree(A);
    cudaFree(B);
    free(comparisonRes);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);
    return;
}

int main(){
    matMul(2000);
    return 0;
}