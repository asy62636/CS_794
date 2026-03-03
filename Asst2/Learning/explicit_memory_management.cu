#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>
#include <chrono>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

void explicitMemManagement(int vectorlength){
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*) malloc(vectorlength * sizeof(float));

    cudaMallocHost(&A, vectorlength * sizeof(float));
    cudaMallocHost(&B, vectorlength * sizeof(float));
    cudaMallocHost(&C, vectorlength * sizeof(float));
    cudaError_t err1 = cudaPeekAtLastError();
    printf("First error = %s\n", err1);

    initArray(A, vectorlength);
    initArray(B, vectorlength);

    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    cudaMalloc(&devA, vectorlength*sizeof(float));
    cudaMalloc(&devB, vectorlength*sizeof(float));
    cudaMalloc(&devC, vectorlength*sizeof(float));

    cudaError_t err2 = cudaPeekAtLastError();
    printf("First error = %s\n", err2);

    cudaMemcpy(devA, A, vectorlength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorlength*sizeof(float), cudaMemcpyDefault);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int threads = 256;
    int blocks = cuda::ceil_div(vectorlength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorlength);
    cudaError_t err3 = cudaPeekAtLastError();
    printf("First error = %s\n", err3);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by kernel = %f milliseconds \n", milliseconds);

    cudaMemcpy(C, devC, vectorlength*sizeof(float), cudaMemcpyDefault);
    cudaError_t err5 = cudaPeekAtLastError();
    printf("First error = %s\n", err5);

    auto start2 = std::chrono::high_resolution_clock::now();
    serialVecAdd(A, B, comparisonResult, vectorlength);
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);
    printf("Time for serial addition = %lld milliseconds \n", duration.count());

    if (vectorApproximatelyEqual(C, comparisonResult, vectorlength)){
        printf("CPU and GPU results match \n");
    }
    else{
        printf("CPU and GPU results do not match \n");
    }
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
    cudaError_t err4 = cudaPeekAtLastError();
    printf("First error = %s\n", err4);
}

int main(int argc, char** argv){
    int veclength = 1024;
    if (argc >= 2){
        veclength = std::atoi(argv[1]);
    }
    explicitMemManagement(veclength);
    return 0;
}