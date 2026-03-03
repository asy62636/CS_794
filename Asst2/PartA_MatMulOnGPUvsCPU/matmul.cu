#include <cstdlib>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <stdio.h>
#include <cuda/cmath>
#include <chrono>
#include <ctime>
#include <cublas_v2.h>

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
    dim3 threads(32, 32, 1);
    dim3 blocks(cuda::ceil_div(N, 32), cuda::ceil_div(N, 32), 1);

    float* C = nullptr;
    cudaMallocManaged(&C, matSize * sizeof(float));
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    MatMulGPUNaive<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
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
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaFree(C_coalesced);
    float time2 = 0;
    cudaEventElapsedTime(&time2, start2, stop2);
    printf("Time taken for kernel 2 (for N = %i)= %f \n", N, time2);

    //cuBLAS kernel:
    cudaEvent_t start3, stop3;
    float* C_cublas = nullptr;
    cudaMallocManaged(&C_cublas, matSize * sizeof(float));
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    float alpha = 1.0f;
    float beta = 0.0f;
    cudaEventRecord(start3);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B, N, A, N, &beta, C_cublas, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    cublasDestroy(handle);
    cudaFree(C_cublas);
    float time3 = 0;
    cudaEventElapsedTime(&time3, start3, stop3);
    printf("Time taken for kernel 3 (for N = %i)= %f \n", N, time3);
    float total_time = time1 + time2 + time3;
    printf("Total time taken for the three kernels (for N = %i) = %f \n", N, total_time);

    double flops = 2.0 * N * N * N;
    float tflops1 = (flops / (time1 / 1000.0)) / 1e12;
    float tflops2 = (flops / (time2 / 1000.0)) / 1e12;
    float tflops3 = (flops / (time3 / 1000.0)) / 1e12;

    printf("CSV,%d,%f,%f,%f,%f,%f,%f\n", N, time1, time2, time3, tflops1, tflops2, tflops3);
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
    return;
}

int main(int argc, char** argv){
    // int N = 1024;
    // if (argc > 1){
    //     N = std::atoi(argv[1]);
    // }
    // combinedFuncCall(N);
    // return 0;
    for (int N = 1024; N < 32000; N *= 2){
        combinedFuncCall(N);
    }
    return 0;
}