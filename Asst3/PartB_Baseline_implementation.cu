#include <cstdlib>
#include <cuda/cmath>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define N 1024
#define IN 512
#define H1 2048
#define H2 2048
#define H3 2048
#define OUT 512
#define BLOCK_SIZE 32
#define STRIDE 4

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

void initWeights(float* W, int matSize, int fan_in) {
    float scale = sqrtf(2.0f / fan_in);
    for (int i = 0; i < matSize; i++) {
        // Gaussian approximation using Box-Muller
        float u1 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 1.0f);
        W[i] = scale * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }
}

void initInput(float* A, int in = IN) {
    for (int i = 0; i < in * N; i++) {
        A[i] = 2.0f * rand() / (float)RAND_MAX - 1.0f;
    }
}

float loss(float* Z, const float* Y, int cols = OUT){
    float s = 0.0f;
    for (int i = 0; i < N * cols; i++){
        float diff = (Z[i] - Y[i]);
        s += diff * diff;
    }
    return s / (N * cols);
}


__global__ void colWiseMult(float* X, float* W, float* B, float* C, int in, int out){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float sharedA[BLOCK_SIZE * STRIDE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int M = cuda::ceil_div(in, BLOCK_SIZE);
    float sumS[STRIDE] = {0.0f};

    for (int m = 0; m < M; m++){
        for (int s = 0; s < STRIDE; s++){
            int rowA = ty + by * STRIDE * BLOCK_SIZE + s * BLOCK_SIZE;
            int colA = tx + m * BLOCK_SIZE;
            sharedA[ty + s * BLOCK_SIZE][tx] = (rowA < N && colA < in) ? X[rowA * in + colA]: 0.0f;
        }
        int rowB = ty + m * BLOCK_SIZE;
        int colB = tx + bx * BLOCK_SIZE;
        sharedB[ty][tx] = (rowB < in && colB < out) ? W[rowB * out + colB] : 0.0f;
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) {
            for (int i = 0; i < STRIDE; i++) {
                sumS[i] += sharedA[ty + i * BLOCK_SIZE][k] * sharedB[k][tx];
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < STRIDE; i++){
        int rowC = ty + by * STRIDE * BLOCK_SIZE + i * BLOCK_SIZE;
        int colC = tx + bx * BLOCK_SIZE;
        if (rowC < N && colC < out){
            C[rowC * out + colC] = sumS[i] + B[colC];
        }
    }
}

__global__ void ReLU(float* X, float* R, int rows, int cols){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int row = ty + by * BLOCK_SIZE;
    int col = tx + bx * BLOCK_SIZE;
    if (row < rows && col < cols){
        R[row * cols + col] = (X[row * cols + col] >= 0) ? X[row * cols + col] : 0.0f;
    }
}

__global__ void final_output_gradient(float* Z, float* Y, float* Z_grad){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < N && col < OUT){
        Z_grad[row * OUT + col] = 2 * (Z[row * OUT + col] - Y[row * OUT + col]) / (N * OUT);
    }
}

__global__ void W_grads(float* A, float* Z_grad, float* W_grad, int in, int out){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float sharedA[BLOCK_SIZE * STRIDE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int M = cuda::ceil_div(N, BLOCK_SIZE);
    float sumS[STRIDE] = {0.0f};

    for (int m = 0; m < M; m++){
        for (int s = 0; s < STRIDE; s++){
            int rowA = ty + by * BLOCK_SIZE * STRIDE + s * BLOCK_SIZE;
            int colA = tx + m * BLOCK_SIZE;
            sharedA[ty + s * BLOCK_SIZE][tx] = (rowA < in && colA < N) ? A[colA * in + rowA] : 0.0f;
        }
        int rowZ = ty + m * BLOCK_SIZE;
        int colZ = tx + bx * BLOCK_SIZE;
        sharedB[ty][tx] = (rowZ < N && colZ < out) ? Z_grad[rowZ * out + colZ] : 0.0f;
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++){
            for (int s = 0; s < STRIDE; s++){
                sumS[s] += sharedA[ty + s * BLOCK_SIZE][k] * sharedB[k][tx];
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < STRIDE; i++){
        int rowG = ty + by * BLOCK_SIZE * STRIDE + i * BLOCK_SIZE;
        int colG = tx + bx * BLOCK_SIZE;
        if (rowG < in && colG < out){
            W_grad[rowG * out + colG] = sumS[i];
        }
    }
}

__global__ void A_grads(float* W, float* Z_grad, float* A_grad, int in, int out){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float sharedA[BLOCK_SIZE * STRIDE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];

    int M = cuda::ceil_div(out, BLOCK_SIZE);
    float sumS[STRIDE] = {0.0f};

    for (int m = 0; m < M; m++){
        for (int s = 0; s < STRIDE; s++){
            int rowZ = ty + by * BLOCK_SIZE * STRIDE + s * BLOCK_SIZE;
            int colZ = tx + m * BLOCK_SIZE;
            sharedA[ty + s * BLOCK_SIZE][tx] = (colZ < out && rowZ < N) ? Z_grad[rowZ * out + colZ] : 0.0f;
        }
        int rowW = ty + m * BLOCK_SIZE;
        int colW = tx + bx * BLOCK_SIZE;
        sharedB[ty][tx] = (rowW < out && colW < in) ? W[colW * out + rowW] : 0.0f;
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++){
            for (int s = 0; s < STRIDE; s++){
                sumS[s] += sharedA[ty + s * BLOCK_SIZE][k] * sharedB[k][tx];
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < STRIDE; i++){
        int rowG = ty + by * BLOCK_SIZE * STRIDE + i * BLOCK_SIZE;
        int colG = tx + bx * BLOCK_SIZE;
        if (rowG < N && colG < in){
            A_grad[rowG * in + colG] = sumS[i];
        }
    }
}

__global__ void b_grads(float* z_grads, float* B_grads, int out){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int idx = tx + bx * blockDim.x;
    if (idx < out){
        float sum = 0.0f;
        for (int row = 0; row < N; row++){
            sum += z_grads[row * out + idx];
        }
        B_grads[idx] = sum;
    }
}

__global__ void Z_grads(float* A, float* Z, float* Zgrad, int out){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < N && col < out){
        float v = (Z[row * out + col] > 0) ? 1.0f : 0.0f;
        Zgrad[row * out + col] = A[row * out + col] * v;
    }
}

__global__ void W_update(float* W, float* Wgrad, float eta, int rows, int cols){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < rows && col < cols){
        W[row * cols + col] = W[row * cols + col] - eta * Wgrad[row * cols + col];
    }
}

__global__ void B_update(float* B, float* Bgrad, float eta, int cols){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < cols){
        B[idx] = B[idx] - eta * Bgrad[idx];
    }
}

void trainingLoop(const float* X, const float* Y, int epochs = 5, float eta = 0.1){
    size_t free_b, total_b, used_b;
    cudaMemGetInfo(&free_b, &total_b);
    used_b = total_b - free_b;

    printf("Total Memory before assignment = %.2f MB \n", total_b /(1024.0 * 1024.0));
    printf("Used Memory before assignment = %.2f MB \n", used_b /(1024.0 * 1024.0));
    printf("Free Memory before assignment  = %.2f MB \n", free_b /(1024.0 * 1024.0));
    float* Z1 = nullptr;
    float* A1 = nullptr;
    float* Z2 = nullptr;
    float* A2 = nullptr;
    float* Z3 = nullptr;
    float* A3 = nullptr;
    float* A4 = nullptr;

    // CUDA_CHECK(cudaMalloc(&Z1, N * H1 * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&A1, N * H1 * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&Z2, N * H2 * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&A2, N * H2 * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&Z3, N * H3 * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&A3, N * H3 * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&A4, N * OUT * sizeof(float)));
    
    float* dw1 = nullptr;
    float* db1 = nullptr;
    float* dw2 = nullptr;
    float* db2 = nullptr;
    float* dw3 = nullptr;
    float* db3 = nullptr;
    float* dw4 = nullptr;
    float* db4 = nullptr;
    float* dz4 = nullptr;
    float* dz3 = nullptr;
    float* dz2 = nullptr;
    float* dz1 = nullptr;
    float* da1 = nullptr;
    float* da2 = nullptr;
    float* da3 = nullptr;

    CUDA_CHECK(cudaMalloc(&dw1, IN * H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db1, H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw2, H1 * H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db2, H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw3, H2 * H3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db3, H3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw4, H3 * OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db4, OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dz4, N * OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dz3, N * H3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dz2, N * H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dz1, N * H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&da1, N * H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&da2, N * H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&da3, N * H3 * sizeof(float)));

    float* W1 = nullptr;
    float* b1 = nullptr;
    float* W2 = nullptr;
    float* b2 = nullptr;
    float* W3 = nullptr;
    float* b3 = nullptr;
    float* W4 = nullptr;
    float* b4 = nullptr;
    CUDA_CHECK(cudaMallocHost(&W1, IN * H1 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&b1, H1 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&W2, H1 * H2 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&b2, H2 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&W3, H2 * H3 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&b3, H3 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&W4, H3 * OUT * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&b4, OUT * sizeof(float)));
    
    initWeights(W1, IN * H1, IN);
    // initWeights(b1, H1);
    memset(b1, 0, H1 * sizeof(float));
    initWeights(W2, H1 * H2, H1);
    // initWeights(b2, H2);
    memset(b2, 0, H2 * sizeof(float));
    initWeights(W3, H2 * H3, H2);
    // initWeights(b3, H3);
    memset(b3, 0, H3 * sizeof(float));
    initWeights(W4, H3 * OUT, H3);
    // initWeights(b4, OUT);
    memset(b4, 0, OUT * sizeof(float));
    printf("Weights initialised \n");

    float* W1_GPU = nullptr;
    float* b1_GPU = nullptr;
    float* W2_GPU = nullptr;
    float* b2_GPU = nullptr;
    float* W3_GPU = nullptr;
    float* b3_GPU = nullptr;
    float* W4_GPU = nullptr;
    float* b4_GPU = nullptr;

    CUDA_CHECK(cudaMalloc(&W1_GPU, IN * H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b1_GPU, H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W2_GPU, H1 * H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b2_GPU, H2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W3_GPU, H2 * H3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b3_GPU, H3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W4_GPU, H3 * OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b4_GPU, OUT * sizeof(float)));


    CUDA_CHECK(cudaMemcpy(W1_GPU, W1, IN * H1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b1_GPU, b1, H1 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W2_GPU, W2, H1 * H2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b2_GPU, b2, H2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W3_GPU, W3, H2 * H3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b3_GPU, b3, H3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W4_GPU, W4, H3 * OUT * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b4_GPU, b4, OUT * sizeof(float), cudaMemcpyHostToDevice));

    //initialise input
    // float* X = nullptr;
    // CUDA_CHECK(cudaMallocHost(&X, N * IN * sizeof(float)));
    // initInput(X, IN);
    // printf("Input initialised \n");
    float* X_GPU = nullptr;
    CUDA_CHECK(cudaMalloc(&X_GPU, N * IN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(X_GPU, X, N * IN * sizeof(float), cudaMemcpyHostToDevice));

    //copy Y into device memory
    float* Y_GPU = nullptr;
    cudaMalloc(&Y_GPU, N * OUT * sizeof(float));
    cudaMemcpy(Y_GPU, Y, N * OUT * sizeof(float), cudaMemcpyHostToDevice);

    float* A4_host = nullptr;
    cudaMallocHost(&A4_host, N * OUT * sizeof(float));
    
    size_t free_t, total_t;

    cudaMemGetInfo(&free_t, &total_t);
    size_t used_t = total_t - free_t;

    printf("Total Memory after allotment = %.2f MB \n", total_t /(1024.0 * 1024.0));
    printf("Used Memory after allotment = %.2f MB \n", used_t /(1024.0 * 1024.0));
    printf("Free Memory after allotment = %.2f MB \n", free_t /(1024.0 * 1024.0));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

    cudaEvent_t start, stop, intermediate;
    float total_time = 0, forward_time = 0, backward_time = 0;
    float peak_forward_mem = 0, min_mem = 0, peak_backward_mem = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&intermediate);
    for (int i = 0; i < epochs; i++){
        cudaMalloc(&Z1, N * H1 * sizeof(float));
        cudaMalloc(&A1, N * H1 * sizeof(float));
        cudaMalloc(&Z2, N * H2 * sizeof(float));
        cudaMalloc(&A2, N * H2 * sizeof(float));
        cudaMalloc(&Z3, N * H3 * sizeof(float));
        cudaMalloc(&A3, N * H3 * sizeof(float));
        cudaMalloc(&A4, N * OUT * sizeof(float));

        cudaEventRecord(start);
        //forward pass
        //Z1 = X @ W1 + B1
        dim3 block1(cuda::ceil_div(H1, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE * STRIDE), 1);
        colWiseMult<<<block1, threads>>>(X_GPU, W1_GPU, b1_GPU, Z1, IN, H1);
        cudaDeviceSynchronize();
        //A1 = ReLU(Z1)
        dim3 block2(cuda::ceil_div(H1, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE), 1);
        ReLU<<<block2, threads>>>(Z1, A1, N, H1);
        cudaDeviceSynchronize();

        //Z2 = A1 @ W2 + B2
        dim3 block3(cuda::ceil_div(H2, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE * STRIDE), 1);
        colWiseMult<<<block3, threads>>>(A1, W2_GPU, b2_GPU, Z2, H1, H2);
        //A2 = ReLU(Z2)
        dim3 block4(cuda::ceil_div(H2, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE), 1);
        ReLU<<<block4, threads>>>(Z2, A2, N, H2);
        cudaDeviceSynchronize();

        //Z3 = A2 @ W3 + B3
        dim3 block5(cuda::ceil_div(H3, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE * STRIDE), 1);
        colWiseMult<<<block5, threads>>>(A2, W3_GPU, b3_GPU, Z3, H2, H3);
        //A3 = ReLU(Z3)
        dim3 block6(cuda::ceil_div(H3, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE), 1);
        ReLU<<<block6, threads>>>(Z3, A3, N, H3);
        cudaDeviceSynchronize();

        //Z4 = A3 @ W4 + B4
        dim3 block7(cuda::ceil_div(OUT, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE * STRIDE), 1);
        colWiseMult<<<block7, threads>>>(A3, W4_GPU, b4_GPU, A4, H3, OUT);
        cudaDeviceSynchronize();

        cudaEventRecord(intermediate);
        cudaEventSynchronize(intermediate);

        size_t free_peak, total_peak;
        cudaMemGetInfo(&free_peak, &total_peak);
        size_t peak_mem = total_peak - free_peak;
        printf("Peak memory during forward = %.2f MB\n", peak_mem / (1024.0 * 1024.0));
        peak_forward_mem += peak_mem/(1024.0 * 1024.0);

        cudaFree(Z1);
        cudaFree(Z2);
        cudaFree(A1);
        cudaFree(Z3);
        cudaFree(A3);

        cudaMemcpy(A4_host, A4, N * OUT * sizeof(float), cudaMemcpyDeviceToHost);
        float l = loss(A4_host, Y, OUT);

        //compute dZ4
        dim3 block8(cuda::ceil_div(OUT, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE), 1);
        final_output_gradient<<<block8, threads>>>(A4, Y_GPU, dz4);
        cudaDeviceSynchronize();

        size_t free_min, total_min;
        cudaMemGetInfo(&free_min, &total_min);
        size_t min_memory = total_min - free_min;
        printf("Min memory after forward = %.2f MB\n", min_memory / (1024.0 * 1024.0));
        min_mem += min_memory/ (1024.0 * 1024.0);

        //compute gradients of each layer
        cudaMalloc(&Z3, N * H3 * sizeof(float));
        cudaMalloc(&A3, N * H3 * sizeof(float));
        colWiseMult<<<block5, threads>>>(A2, W3_GPU, b3_GPU, Z3, H2, H3);
        cudaDeviceSynchronize();
        ReLU<<<block6, threads>>>(Z3, A3, N, H3);
        cudaDeviceSynchronize();
        dim3 block9(cuda::ceil_div(OUT, BLOCK_SIZE), cuda::ceil_div(H3, BLOCK_SIZE * STRIDE), 1);
        dim3 block10(cuda::ceil_div(H3, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE * STRIDE), 1);
        dim3 block11(cuda::ceil_div(H3, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE), 1);
        W_grads<<<block9, threads>>>(A3, dz4, dw4, H3, OUT);
        b_grads<<<cuda::ceil_div(OUT, 1024), 1024>>>(dz4, db4, OUT);
        A_grads<<<block10, threads>>>(W4_GPU, dz4, da3, H3, OUT);
        cudaDeviceSynchronize();
        Z_grads<<<block11, threads>>>(da3, Z3, dz3, H3);
        cudaDeviceSynchronize();
        cudaFree(Z3);
        cudaFree(A3);

        //need Z1, A1, Z2:
        cudaMalloc(&Z1, N * H1 * sizeof(float));
        cudaMalloc(&A1, N * H1 * sizeof(float));
        cudaMalloc(&Z2, N * H2 * sizeof(float));
        colWiseMult<<<block1, threads>>>(X_GPU, W1_GPU, b1_GPU, Z1, IN, H1);
        cudaDeviceSynchronize();
        ReLU<<<block2, threads>>>(Z1, A1, N, H1);
        cudaDeviceSynchronize();
        colWiseMult<<<block3, threads>>>(A1, W2_GPU, b2_GPU, Z2, H1, H2);
        cudaDeviceSynchronize();
        dim3 block12(cuda::ceil_div(H3, BLOCK_SIZE), cuda::ceil_div(H2, BLOCK_SIZE * STRIDE), 1);
        dim3 block13(cuda::ceil_div(H2, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE * STRIDE), 1);
        dim3 block14(cuda::ceil_div(H2, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE), 1);
        W_grads<<<block12, threads>>>(A2, dz3, dw3, H2, H3);
        b_grads<<<cuda::ceil_div(H3, 1024), 1024>>>(dz3, db3, H3);
        A_grads<<<block13, threads>>>(W3_GPU, dz3, da2, H2, H3);
        cudaDeviceSynchronize();
        Z_grads<<<block14, threads>>>(da2, Z2, dz2, H2);
        cudaDeviceSynchronize();

        dim3 block15(cuda::ceil_div(H2, BLOCK_SIZE), cuda::ceil_div(H1, BLOCK_SIZE * STRIDE), 1);
        dim3 block16(cuda::ceil_div(H1, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE * STRIDE), 1);
        dim3 block17(cuda::ceil_div(H1, BLOCK_SIZE), cuda::ceil_div(N, BLOCK_SIZE), 1);
        W_grads<<<block15, threads>>>(A1, dz2, dw2, H1, H2);
        b_grads<<<cuda::ceil_div(H2, 1024), 1024>>>(dz2, db2, H2);
        A_grads<<<block16, threads>>>(W2_GPU, dz2, da1, H1, H2);
        cudaDeviceSynchronize();
        Z_grads<<<block17, threads>>>(da1, Z1, dz1, H1);
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_peak, &total_peak);
        printf("Peak backward memory = %.2f MB\n", (total_peak - free_peak) / (1024.0 * 1024.0));
        peak_backward_mem += (total_peak - free_peak)/ (1024.0 * 1024.0);

        cudaFree(Z1);
        cudaFree(A1);
        cudaFree(Z2);
        cudaFree(A2);
        cudaFree(A4);

        dim3 block18(cuda::ceil_div(H1, BLOCK_SIZE), cuda::ceil_div(IN, BLOCK_SIZE * STRIDE), 1);
        W_grads<<<block18, threads>>>(X_GPU, dz1, dw1, IN, H1);
        b_grads<<<cuda::ceil_div(H1, 1024), 1024>>>(dz1, db1, H1);
        cudaDeviceSynchronize();

        //update weights
        dim3 block19(cuda::ceil_div(OUT, BLOCK_SIZE), cuda::ceil_div(H3, BLOCK_SIZE));
        W_update<<<block19, threads>>>(W4_GPU, dw4, eta, H3, OUT);
        B_update<<<cuda::ceil_div(OUT, 1024), 1024>>>(b4_GPU, db4, eta, OUT);
        dim3 block20(cuda::ceil_div(H3, BLOCK_SIZE), cuda::ceil_div(H2, BLOCK_SIZE));
        W_update<<<block20, threads>>>(W3_GPU, dw3, eta, H2, H3);
        B_update<<<cuda::ceil_div(H3, 1024), 1024>>>(b3_GPU, db3, eta, H3);
        dim3 block21(cuda::ceil_div(H2, BLOCK_SIZE), cuda::ceil_div(H1, BLOCK_SIZE));
        W_update<<<block21, threads>>>(W2_GPU, dw2, eta, H1, H2);
        B_update<<<cuda::ceil_div(H2, 1024), 1024>>>(b2_GPU, db2, eta, H2);
        dim3 block22(cuda::ceil_div(H1, BLOCK_SIZE), cuda::ceil_div(IN, BLOCK_SIZE));
        W_update<<<block22, threads>>>(W1_GPU, dw1, eta, IN, H1);
        B_update<<<cuda::ceil_div(H1, 1024), 1024>>>(b1_GPU, db1, eta, H1);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time_elapsed = 0;
        float f_time = 0;
        float b_time = 0;
        cudaEventElapsedTime(&time_elapsed, start, stop);
        cudaEventElapsedTime(&f_time, start, intermediate);
        cudaEventElapsedTime(&b_time, intermediate, stop);
        total_time += time_elapsed;
        forward_time += f_time;
        backward_time += b_time;
        printf("Loss in Epoch %d = %.2f | Time = %.2f ms |F_time = %.2f | B_time = %.2f \n", i, l, time_elapsed, f_time, b_time);
    }
    printf("Average time per iteration = %.2f ms\n", total_time/epochs);
    printf("Average forward time per iteration = %.2f ms\n", forward_time/epochs);
    printf("Average backward time per iteration = %.2f ms\n", backward_time/epochs);
    printf("Memory used = %.2f MB \n", (used_t - used_b)/(1024.0 * 1024.0));
    printf("Peak forward memory used = %.2f MB \n", peak_forward_mem/epochs);
    printf("Minimum memory used = %.2f MB \n", min_mem/epochs);
    printf("Peak backward memory used =%.2f MB \n", peak_backward_mem/epochs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(intermediate);
    // Add at end of trainingLoop:
    // cudaFree(Z1); cudaFree(A1); cudaFree(Z2); cudaFree(A2);
    // cudaFree(Z3); cudaFree(A3); cudaFree(A4);
    cudaFree(dw1); cudaFree(db1); cudaFree(dw2); cudaFree(db2);
    cudaFree(dw3); cudaFree(db3); cudaFree(dw4); cudaFree(db4);
    cudaFree(dz1); cudaFree(dz2); cudaFree(dz3); cudaFree(dz4);
    cudaFree(da1); cudaFree(da2); cudaFree(da3);
    cudaFree(W1_GPU); cudaFree(W2_GPU); cudaFree(W3_GPU); cudaFree(W4_GPU);
    cudaFree(b1_GPU); cudaFree(b2_GPU); cudaFree(b3_GPU); cudaFree(b4_GPU);
    cudaFree(X_GPU); cudaFree(Y_GPU);
    cudaFreeHost(W1); cudaFreeHost(W2); cudaFreeHost(W3); cudaFreeHost(W4);
    cudaFreeHost(b1); cudaFreeHost(b2); cudaFreeHost(b3); cudaFreeHost(b4); cudaFreeHost(A4_host);
}

int main(){
    float* X = nullptr;
    CUDA_CHECK(cudaMallocHost(&X, N * IN * sizeof(float)));
    initInput(X, IN);
    printf("Input initialised \n");
    float* Y = nullptr; 
    cudaMallocHost(&Y, N * OUT * sizeof(float));
    //need to write code to allocate values into Y
    initInput(Y, OUT);
    trainingLoop(X, Y, 100, 0.001);
    cudaFreeHost(X);
    cudaFreeHost(Y);
    return 0;
}