#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <chrono>
#include <cuda/cmath>

// using namespace std::chrono;

__global__ void vecAdd(float* A, float* B, float* C, int veclength){
    int index = (threadIdx.x + blockIdx.x * blockDim.x);
    if (index < veclength){
        C[index] = A[index] + B[index];
    }
}

void initArray(float* A, int veclength){
    std::srand(std::time({}));
    for (int i = 0; i < veclength; i++){
        A[i] = rand()/float(RAND_MAX);
    }
}

bool compareArrays(float* A, float* B, int veclength, float epsilon = 1e-5){
    for (int i = 0; i < veclength; i++){
        if (fabs(A[i] - B[i]) > epsilon) return false;
    }
    return true;
}

void serialAdd(float* A, float* B, float* C, int veclength){
    for (int i = 0; i < veclength; i++){
        C[i] = A[i] + B[i];
    }
}

void explicitMemory(int vecLength){
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vecLength * sizeof(float));

    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    cudaMallocHost(&A, vecLength*sizeof(float));
    cudaMallocHost(&B, vecLength*sizeof(float));
    cudaMallocHost(&C, vecLength*sizeof(float));

    initArray(A, vecLength);
    initArray(B, vecLength);

    cudaMalloc(&devA, vecLength*sizeof(float));
    cudaMalloc(&devB, vecLength*sizeof(float));
    cudaMalloc(&devC, vecLength*sizeof(float));

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(devA, A, vecLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vecLength*sizeof(float), cudaMemcpyDefault);

    int threads = 256;
    int blocks = cuda::ceil_div(vecLength, threads);
    cudaEventRecord(start);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vecLength);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel runtime = %f \n", milliseconds);
    cudaError_t err = cudaPeekAtLastError();
    printf("Faced errors %s in running kernel function \n", cudaGetErrorString(err));

    auto start2 = std::chrono::high_resolution_clock::now();
    serialAdd(A, B, comparisonResult, vecLength);
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);
    printf("Time for serialised implementation = %lld milliseconds\n", duration.count());

    cudaMemcpy(C, devC, vecLength * sizeof(float), cudaMemcpyDefault);
    if (compareArrays(C, comparisonResult, vecLength)){
        printf("[SUCCESS] CPU and GPU results matched!");
    }
    else{
        printf("[ERROR] CPU and GPU results did not match!");
    }
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}

int main(int argc, char** argv){
    int vecLength = 100000000;
    if (argc > 1){
        vecLength = std::atoi(argv[1]);
    }
    printf("vecLength = %i \n", vecLength);
    explicitMemory(vecLength);
    return 0;
}