#include <cstdlib>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <stdio.h>
#include <cuda/cmath>
#include <chrono>
#include <ctime>
#include <cublas_v2.h>

#define BLOCK_SIZE 32
#define STRIDE 4

void initMat(float* A, float* B, float* C, int matrixSize){
    for (int i = 0; i < matrixSize; i++){
        A[i] = (float)rand()/ RAND_MAX;
        B[i] = (float)rand()/RAND_MAX;
        C[i] = 0.0f;
    }
}

void MatMulCPU(float* A, float* B, float* C, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            float sum = 0;
            for (int k = 0; k < N; k++){
                sum += A[i* N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void MatMulGPUNaive(float* A, float* B, float* C, int N){
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < N && col < N){
        float sum = 0.0f;
        for (int i = 0; i < N; i++){
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void MatMulGPUCoalescedAccess(float* A, float* B, float* C, int N){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < N && col < N){
        float sum = 0.0f;
        for (int i = 0; i < N; i++){
            sum += A[row*N + i] * B[i*N + col];
        }
        C[row*N + col] = sum;
    }
}

bool compareResults(float* C, float* refRes, int matSize, float epsilon = 0.001){
    for (int i = 0; i < matSize; i++){
        if (fabs(refRes[i] - C[i]) > epsilon) return false;
    }
    return true;
}

__global__ void simpleTiling(float* A, float* B, float* C, int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;
    for (int m = 0; m < cuda::ceil_div(N, BLOCK_SIZE); m++){
        if (row < N && (m * BLOCK_SIZE + tx) < N){
            sharedA[ty][tx] = A[row * N + m * BLOCK_SIZE + tx];
        }
        else{
            sharedA[ty][tx] = 0.0;
        }
        if (col < N && (ty + m * BLOCK_SIZE) < N){
            sharedB[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        }
        else{
            sharedB[ty][tx] = 0.0;
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++){
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;

    return;
}

__global__ void RowBasedTiling(float* A, float* B, float* C, int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = ty + by * BLOCK_SIZE;
    int col = tx + bx * BLOCK_SIZE * STRIDE;

    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE * STRIDE];

    float sumS[STRIDE] = {0.0f};

    for (int m = 0; m < cuda::ceil_div(N, BLOCK_SIZE); m++){
        int rowA = row;
        int colA = (m * BLOCK_SIZE + tx);
        if (rowA < N && colA < N){
            sharedA[ty][tx] = A[rowA * N + colA];
        }
        else{
            sharedA[ty][tx] = 0.0;
        }
        for (int s = 0; s < STRIDE; s++){
            int rowB = m * BLOCK_SIZE + ty;
            int colB = col + s * BLOCK_SIZE;
            if (rowB < N && colB < N){
                sharedB[ty][tx + s * BLOCK_SIZE] = B[rowB * N + colB];
            }
            else{
                sharedB[ty][tx + s * BLOCK_SIZE] = 0.0;
            }
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
        if (row < N && col + s * BLOCK_SIZE < N) C[row * N + col + s * BLOCK_SIZE] = sumS[s];
    }
}

__global__ void colBasedTiling(float* A, float* B, float* C, int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = ty + by * BLOCK_SIZE * STRIDE;
    int col = tx + bx * BLOCK_SIZE;

    __shared__ float sharedA[BLOCK_SIZE * STRIDE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    float sumS[STRIDE] = {0.0f};

    for (int m = 0; m < cuda::ceil_div(N, BLOCK_SIZE); m++){
        for (int s = 0; s < STRIDE; s++){
            int rowA = row + s * BLOCK_SIZE;
            int colA = tx + m * BLOCK_SIZE;
            if (rowA < N && colA < N){
                sharedA[ty + s * BLOCK_SIZE][tx] = A[rowA * N + colA];
            }
            else{
                sharedA[ty + s * BLOCK_SIZE][tx] = 0.0;
            }
        }
        int rowB = ty + m * BLOCK_SIZE;
        int colB = col;
        if (rowB < N && colB < N){
            sharedB[ty][tx] = B[rowB * N + colB];
        }
        else{
            sharedB[ty][tx] = 0.0;
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++){
            for (int s = 0; s < STRIDE; s++){
                sumS[s] += sharedA[ty + s * BLOCK_SIZE][k] * sharedB[k][tx];
            }
        }
        __syncthreads();
    }
    for (int s = 0; s < STRIDE; s++){
        int r = row + s * BLOCK_SIZE;
        int c = col;
        if (r < N && c < N) C[r * N + c] = sumS[s];
    }
}

void combinedFuncCall(int N){
    int matSize = N * N;

    float* A = nullptr;
    float* B = nullptr;
    float* refResult = (float*)malloc(matSize * sizeof(float));

    cudaMallocManaged(&A, matSize * sizeof(float));
    cudaMallocManaged(&B, matSize * sizeof(float));

    initMat(A, B, refResult, matSize);

    //basic CPU multiplication for reference result
    // MatMulCPU(A, B, refResult, N);


    //Naive multiplication on GPU
    cudaEvent_t start1, stop1;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocks(cuda::ceil_div(N, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE), 1);
    dim3 rowBasedBlocks(cuda::ceil_div(N, BLOCK_SIZE * STRIDE),cuda::ceil_div(N, BLOCK_SIZE), 1);
    dim3 colBasedBlocks(cuda::ceil_div(N, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE * STRIDE), 1);
    float* C = nullptr;
    cudaMallocManaged(&C, matSize * sizeof(float));
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    MatMulGPUNaive<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("1st Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaFree(C);
    float time1 = 0;
    cudaEventElapsedTime(&time1, start1, stop1);
    printf("Time taken for kernel 1 (for N = %i)= %f \n", N, time1);

    //Coalesced Memory access
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    float* C_coalesced = nullptr;
    cudaMallocManaged(&C_coalesced, matSize * sizeof(float));
    cudaEventRecord(start2);
    MatMulGPUCoalescedAccess<<<blocks, threads>>>(A, B, C_coalesced, N);
    cudaDeviceSynchronize();
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("2nd Kernel launch failed: %s\n", cudaGetErrorString(err2));
    }
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaFree(C_coalesced);
    float time2 = 0;
    cudaEventElapsedTime(&time2, start2, stop2);
    printf("Time taken for kernel 2 (for N = %i)= %f \n", N, time2);

    //cuBLAS kernel:
    cudaEvent_t start3, stop3;
    float* d_A = nullptr;
    float* d_B = nullptr;
    cudaMalloc(&d_A, matSize * sizeof(float));
    cudaMalloc(&d_B, matSize * sizeof(float));
    cudaMemcpy(d_A, A, matSize * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_B, B, matSize * sizeof(float), cudaMemcpyDefault);
    float* C_cublas = nullptr;
    cudaMallocManaged(&C_cublas, matSize * sizeof(float));
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaEventRecord(start3);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, C_cublas, N);
    cudaDeviceSynchronize();
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {
        printf("cuBLAS Kernel launch failed: %s\n", cudaGetErrorString(err3));
    }
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    cublasDestroy(handle);
    cudaFree(C_cublas);
    cudaFree(d_A);
    cudaFree(d_B);
    float time3 = 0;
    cudaEventElapsedTime(&time3, start3, stop3);
    printf("Time taken for kernel 3 (for N = %i)= %f \n", N, time3);
    

    //simpleTilingKernel
    cudaEvent_t start4, stop4;
    float* C_tiling = nullptr;
    cudaMallocManaged(&C_tiling, matSize * sizeof(float));
    cudaEventCreate(&start4);
    cudaEventCreate(&stop4);
    cudaEventRecord(start4);
    simpleTiling<<<blocks, threads>>>(A, B, C_tiling, N);
    cudaDeviceSynchronize();
    cudaError_t err4 = cudaGetLastError();
    if (err4 != cudaSuccess) {
        printf("4th Kernel launch failed: %s\n", cudaGetErrorString(err4));
    }
    cudaEventRecord(stop4);
    cudaEventSynchronize(stop4);
    // if (compareResults(C_tiling, refResult, matSize)){
    //     printf("[SUCCESS] correct tiling code written\n");
    // }
    // else{
    //     printf("[ERROR] Incorrect code written\n");
    // }
    float time4 = 0.0;
    cudaEventElapsedTime(&time4, start4, stop4);
    printf("Time taken for kernel 4 (for N = %i)= %f \n", N, time4);
    cudaFree(C_tiling);

    //RowBasedTilingKernel
    cudaEvent_t start5, stop5;
    float* C_Row_tiling = nullptr;
    cudaMallocManaged(&C_Row_tiling, matSize * sizeof(float));
    cudaEventCreate(&start5);
    cudaEventCreate(&stop5);
    cudaEventRecord(start5);
    RowBasedTiling<<<rowBasedBlocks, threads>>>(A, B, C_Row_tiling, N);
    cudaDeviceSynchronize();
    cudaError_t err5 = cudaGetLastError();
    if (err5 != cudaSuccess) {
        printf("5th Kernel launch failed: %s\n", cudaGetErrorString(err5));
    }
    cudaEventRecord(stop5);
    cudaEventSynchronize(stop5);
    // if (compareResults(C_Row_tiling, refResult, matSize)){
    //     printf("[SUCCESS] correct ROW tiling code written\n");
    // }
    // else{
    //     printf("[ERROR] Incorrect ROW code written\n");
    // }
    float time5 = 0.0;
    cudaEventElapsedTime(&time5, start5, stop5);
    printf("Time taken for kernel 5 (for N = %i)= %f \n", N, time5);
    cudaFree(C_Row_tiling);

    //ColBasedTilingKernel
    cudaEvent_t start6, stop6;
    float* C_Col_tiling = nullptr;
    cudaMallocManaged(&C_Col_tiling, matSize * sizeof(float));
    cudaEventCreate(&start6);
    cudaEventCreate(&stop6);
    cudaEventRecord(start6);
    colBasedTiling<<<colBasedBlocks, threads>>>(A, B, C_Col_tiling, N);
    cudaDeviceSynchronize();
    cudaError_t err6 = cudaGetLastError();
    if (err6 != cudaSuccess) {
        printf("6th Kernel launch failed: %s\n", cudaGetErrorString(err6));
    }
    cudaEventRecord(stop6);
    cudaEventSynchronize(stop6);
    // if (compareResults(C_Col_tiling, refResult, matSize)){
    //     printf("[SUCCESS] correct COL tiling code written\n");
    // }
    // else{
    //     printf("[ERROR] Incorrect COL code written\n");
    // }
    float time6 = 0.0;
    cudaEventElapsedTime(&time6, start6, stop6);
    printf("Time taken for kernel 6 (for N = %i)= %f \n", N, time6);
    cudaFree(C_Col_tiling);

    float total_time = time1 + time2 + time3 + time4 + time5 + time6;
    printf("Total time taken for the six kernels (for N = %i) = %f \n", N, total_time);

    double flops = 2.0 * N * N * N;
    float tflops1 = (flops / (time1 / 1000.0)) / 1e12;
    float tflops2 = (flops / (time2 / 1000.0)) / 1e12;
    float tflops3 = (flops / (time3 / 1000.0)) / 1e12;
    float tflops4 = (flops / (time4 / 1000.0)) / 1e12;
    float tflops5 = (flops / (time5 / 1000.0)) / 1e12;
    float tflops6 = (flops / (time6 / 1000.0)) / 1e12;

    printf("CSV,%d,%f,%f,%f,%f,%f,%f, %f, %f, %f\n", N, time1, time2, time3, tflops1, tflops2, tflops3, tflops4, tflops5, tflops6);
    printf("\n");
    // cudaFree(C_cublas);
    // cudaFree(C_coalesced);
    // cudaFree(C);
    cudaFree(A);
    cudaFree(B);
    free(refResult);

    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);
    cudaEventDestroy(start4);
    cudaEventDestroy(stop4);
    cudaEventDestroy(start5);
    cudaEventDestroy(stop5);
    cudaEventDestroy(start6);
    cudaEventDestroy(stop6);
    return;
}

int main(int argc, char** argv){
    // int N = 1024;
    // if (argc > 1){
    //     N = std::atoi(argv[1]);
    // }
    // combinedFuncCall(N);
    // return 0;
    printf("BLOCK_SIZE = %i\n", BLOCK_SIZE);
    for (int N = 1024; N < 32000; N *= 2){
        combinedFuncCall(N);
    }
    return 0;
}