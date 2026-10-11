#include <cstdlib>
#include <stdio.h>
#include <cuda/cmath>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>

#define B 8
#define D 64
#define NH 16
#define Br 32
#define Bc 32
#define SMEM (Br * D + 2 * Bc * D + Br * Bc)

__global__ void flashAttention(float* Q, float* K, float* V, float* O, float* M, float* L, int N){
    int b = blockIdx.x;
    int h = blockIdx.y;

    int tx = threadIdx.x;

    int qkv_base = (b * NH + h) * N * D;
    int lm_base = (b * NH + h) * N;

    __shared__ float smem[SMEM];

    float* Qi = smem;
    float* Kj = Qi + Br * D;
    float* Vj = Kj + Bc * D;
    float* S = Vj + Bc * D;

    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;

    for (int j = 0; j < Tc; j++){
        //Load KV tile into shared memory
        for (int x = 0; x < D; x++){
            int kv_idx = qkv_base + (j * Bc + tx) * D + x;
            Kj[tx * D + x] = K[kv_idx];
            Vj[tx * D + x] = V[kv_idx];
        }
        __syncthreads();
        //inner loop
        for (int i = 0; i < Tr; i++){
            //Step 1 :load Q tile into shared memory
            for (int x = 0; x < D; x++){
                int Q_idx = qkv_base + (i * Br + tx) * D + x;
                Qi[tx * D + x] = Q[Q_idx];
            }
            __syncthreads();

            //Step 2 : load Scalars
            float mPrev = M[lm_base + i * Br + tx];
            float lPrev = L[lm_base + i * Br + tx];

            //Step 3 : compute dot products and find max of row

            for (int c = 0; c < Bc; c++){
                float sum = 0;
                for (int x = 0; x < D; x++){
                    sum += Qi[tx * D + x] * Kj[c * D + x];
                }
                S[tx * Bc + c] = sum / sqrtf(float(D));
            }
            float mNew = mPrev;
            for (int c = 0; c < Bc; c++){
                mNew = max(mNew, S[tx * Bc + c]);
            }

            //step 4 : exponentiate and accumulate tile sum along row
            float tile_norm = 0;
            for (int c = 0; c < Bc; c++){
                S[tx * Bc + c] = expf(S[tx * Bc + c] - mNew);
                tile_norm += S[tx * Bc + c];
            }
            float lNew = expf(mPrev - mNew) * lPrev + tile_norm;

            //Step5 : Online softmax update and output rescaling
            //Train of thought - first find the i, jth element of P . V and then I can simply do Oij = (Oij_old*expf(mPrev - mNew) * lprev + PVij)/LNew 
            //in computing PVij i need to do so for all j from 1 to D cause one thread computes all D elements of its row!
            //also i will have to figure out how to get i
            
            int i_row = i * Br + tx;
            for (int col_idx = 0; col_idx < D; col_idx++){
                float mul_res = 0;
                for (int r = 0; r < Bc; r++){
                    mul_res += S[tx * Bc + r] * Vj[r * D + col_idx];
                }
                int O_idx = qkv_base + i_row * D + col_idx;
                O[O_idx] = (O[O_idx] * expf(mPrev - mNew) * lPrev + mul_res) / lNew;
            }
            M[lm_base + i * Br + tx] = mNew;
            L[lm_base + i * Br + tx] = lNew;
            __syncthreads();
        }
    }
}

// helper: save float array to raw binary
void save_bin(const char* path, float* data, int n) {
    std::ofstream f(path, std::ios::binary);
    f.write((char*)data, n * sizeof(float));
}

// helper: load raw binary saved from Python (np.save with allow_pickle=False)
void load_bin(const char* path, float* data, int n) {
    std::ifstream f(path, std::ios::binary);
    f.read((char*)data, n * sizeof(float));
}

// int main() {
//     int N = 256;  // test one N first
//     size_t qkv_size = B * NH * N * D * sizeof(float);
//     size_t lm_size  = B * NH * N * sizeof(float);

//     // host buffers
//     float *hQ = new float[B*NH*N*D];
//     float *hK = new float[B*NH*N*D];
//     float *hV = new float[B*NH*N*D];
//     float *hO = new float[B*NH*N*D]();  // zero init

//     // load QKV saved from Python as raw binary
//     load_bin("N256_Q.bin", hQ, B*NH*N*D);
//     load_bin("N256_K.bin", hK, B*NH*N*D);
//     load_bin("N256_V.bin", hV, B*NH*N*D);

//     // device buffers
//     float *dQ, *dK, *dV, *dO, *dM, *dL;
//     cudaMalloc(&dQ, qkv_size);
//     cudaMalloc(&dK, qkv_size);
//     cudaMalloc(&dV, qkv_size);
//     cudaMalloc(&dO, qkv_size);
//     cudaMalloc(&dM, lm_size);
//     cudaMalloc(&dL, lm_size);

//     cudaMemcpy(dQ, hQ, qkv_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(dK, hK, qkv_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(dV, hV, qkv_size, cudaMemcpyHostToDevice);
//     cudaMemset(dO, 0,    qkv_size);
//     cudaMemset(dL, 0,    lm_size);

//     // init M to -inf
//     float* hM = new float[B*NH*N];
//     std::fill(hM, hM + B*NH*N, -1e38f);
//     cudaMemcpy(dM, hM, lm_size, cudaMemcpyHostToDevice);

//     // launch
//     dim3 grid(B, NH);
//     dim3 block(Br);
//     size_t smem = SMEM * sizeof(float);
//     flashAttention<<<grid, block>>>(dQ, dK, dV, dO, dM, dL, N);
//     cudaDeviceSynchronize();

//     // copy output back and save
//     cudaMemcpy(hO, dO, qkv_size, cudaMemcpyDeviceToHost);
//     save_bin("N256_cuda_out.bin", hO, B*NH*N*D);

//     printf("Done. Output saved to N256_cuda_out.bin\n");
//     return 0;
// }

int main() {
    int seq_lens[] = {32, 64, 128, 256, 512, 1024, 2048};
    int num_lens = 7;

    for (int idx = 0; idx < num_lens; idx++) {
        int N = seq_lens[idx];
        size_t qkv_size = (size_t)B * NH * N * D * sizeof(float);
        size_t lm_size  = (size_t)B * NH * N * sizeof(float);

        // host buffers
        float *hQ = new float[B*NH*N*D];
        float *hK = new float[B*NH*N*D];
        float *hV = new float[B*NH*N*D];
        float *hO = new float[B*NH*N*D]();

        // load QKV
        char path[64];
        sprintf(path, "N%d_Q.bin", N); load_bin(path, hQ, B*NH*N*D);
        sprintf(path, "N%d_K.bin", N); load_bin(path, hK, B*NH*N*D);
        sprintf(path, "N%d_V.bin", N); load_bin(path, hV, B*NH*N*D);

        // device buffers
        float *dQ, *dK, *dV, *dO, *dM, *dL;
        cudaMalloc(&dQ, qkv_size);
        cudaMalloc(&dK, qkv_size);
        cudaMalloc(&dV, qkv_size);
        cudaMalloc(&dO, qkv_size);
        cudaMalloc(&dM, lm_size);
        cudaMalloc(&dL, lm_size);

        cudaMemcpy(dQ, hQ, qkv_size, cudaMemcpyHostToDevice);
        cudaMemcpy(dK, hK, qkv_size, cudaMemcpyHostToDevice);
        cudaMemcpy(dV, hV, qkv_size, cudaMemcpyHostToDevice);
        cudaMemset(dO, 0, qkv_size);
        cudaMemset(dL, 0, lm_size);

        float* hM = new float[B*NH*N];
        std::fill(hM, hM + B*NH*N, -1e38f);
        cudaMemcpy(dM, hM, lm_size, cudaMemcpyHostToDevice);

        // launch
        dim3 grid(B, NH);
        dim3 block(Br);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        // warmup
        for (int w = 0; w < 3; w++)
            flashAttention<<<grid, block>>>(dQ, dK, dV, dO, dM, dL, N);
        cudaDeviceSynchronize();

        // timing
        int ITERS = 10;
        cudaEventRecord(start);
        for (int it = 0; it < ITERS; it++)
            flashAttention<<<grid, block>>>(dQ, dK, dV, dO, dM, dL, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        printf("N=%4d  avg runtime = %.3f ms\n", N, time / ITERS);
        
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // float time = 0;
        // cudaEventElapsedTime(&time, start, stop);
        // printf("Time taken for flash attention (with N = %d) = %.2f ms\n", N, time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("N=%d kernel error: %s\n", N, cudaGetErrorString(err));
        cudaDeviceSynchronize();

        // copy back and save
        cudaMemcpy(hO, dO, qkv_size, cudaMemcpyDeviceToHost);
        sprintf(path, "N%d_cuda_out.bin", N);
        save_bin(path, hO, B*NH*N*D);
        printf("N=%4d done\n", N);

        // cleanup
        cudaFree(dQ); cudaFree(dK); cudaFree(dV);
        cudaFree(dO); cudaFree(dM); cudaFree(dL);
        delete[] hQ; delete[] hK; delete[] hV;
        delete[] hO; delete[] hM;
    }
    return 0;
}