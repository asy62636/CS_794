// SGEMM optimisation ladder (CS 794, PA2 follow-up): C = A x B, square N x N, fp32, row-major.
//
//   k1_naive        : PA2 naive kernel (consecutive threads walk down a column -> uncoalesced)
//   k2_coalesced    : PA2 coalesced kernel (consecutive threads -> consecutive columns)
//   k3_smem_tiling  : PA2 32x32 shared-memory tiling, 1 output per thread
//   k4_row_coarsen  : PA2 tiling, 4 outputs per thread along a row (32x128 output tile)
//   k5_col_coarsen  : PA2 tiling, 4 outputs per thread along a column (128x32 output tile)
//   k6_reg_block_2d : 128x128 block tile, BK=8, each thread computes an 8x8 sub-tile held in
//                     registers (outer-product accumulation: 64 FMAs per 16 shared-memory loads)
//   k7_vectorized   : k6 + 128-bit (float4) global loads/stores and a transposed A tile in shared
//                     memory so the inner loop reads both operands with contiguous accesses
//   cublas_sgemm    : cuBLAS reference (row-major handled by computing C^T = B^T A^T)
//
// All inputs live in device memory (cudaMalloc) before timing, every kernel is warmed up once,
// and every result is checked against cuBLAS.
//
// Usage: ./gemm_bench [--n 1024,2048,...]   |   ./gemm_bench --profile 4096   (one launch each)

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                             \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        if ((call) != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__);      \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

constexpr int BLOCK_SIZE = 32;
constexpr int STRIDE = 4;

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ============================================================================================
// PA2 kernels (unchanged apart from cuda::ceil_div -> a local helper)
// ============================================================================================
__device__ __forceinline__ int dceil_div(int a, int b) { return (a + b - 1) / b; }

__global__ void MatMulGPUNaive(float* A, float* B, float* C, int N) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) sum += A[row * N + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void MatMulGPUCoalescedAccess(float* A, float* B, float* C, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) sum += A[row * N + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void simpleTiling(float* A, float* B, float* C, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];
    float sum = 0.0f;
    for (int m = 0; m < dceil_div(N, BLOCK_SIZE); m++) {
        if (row < N && (m * BLOCK_SIZE + tx) < N) sharedA[ty][tx] = A[row * N + m * BLOCK_SIZE + tx];
        else sharedA[ty][tx] = 0.0;
        if (col < N && (ty + m * BLOCK_SIZE) < N) sharedB[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        else sharedB[ty][tx] = 0.0;
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) sum += sharedA[ty][k] * sharedB[k][tx];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

__global__ void RowBasedTiling(float* A, float* B, float* C, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = ty + by * BLOCK_SIZE;
    int col = tx + bx * BLOCK_SIZE * STRIDE;
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE * STRIDE];
    float sumS[STRIDE] = {0.0f};
    for (int m = 0; m < dceil_div(N, BLOCK_SIZE); m++) {
        int rowA = row;
        int colA = (m * BLOCK_SIZE + tx);
        if (rowA < N && colA < N) sharedA[ty][tx] = A[rowA * N + colA];
        else sharedA[ty][tx] = 0.0;
        for (int s = 0; s < STRIDE; s++) {
            int rowB = m * BLOCK_SIZE + ty;
            int colB = col + s * BLOCK_SIZE;
            if (rowB < N && colB < N) sharedB[ty][tx + s * BLOCK_SIZE] = B[rowB * N + colB];
            else sharedB[ty][tx + s * BLOCK_SIZE] = 0.0;
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
            for (int s = 0; s < STRIDE; s++) sumS[s] += sharedA[ty][k] * sharedB[k][tx + s * BLOCK_SIZE];
        __syncthreads();
    }
    for (int s = 0; s < STRIDE; s++)
        if (row < N && col + s * BLOCK_SIZE < N) C[row * N + col + s * BLOCK_SIZE] = sumS[s];
}

__global__ void colBasedTiling(float* A, float* B, float* C, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = ty + by * BLOCK_SIZE * STRIDE;
    int col = tx + bx * BLOCK_SIZE;
    __shared__ float sharedA[BLOCK_SIZE * STRIDE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];
    float sumS[STRIDE] = {0.0f};
    for (int m = 0; m < dceil_div(N, BLOCK_SIZE); m++) {
        for (int s = 0; s < STRIDE; s++) {
            int rowA = row + s * BLOCK_SIZE;
            int colA = tx + m * BLOCK_SIZE;
            if (rowA < N && colA < N) sharedA[ty + s * BLOCK_SIZE][tx] = A[rowA * N + colA];
            else sharedA[ty + s * BLOCK_SIZE][tx] = 0.0;
        }
        int rowB = ty + m * BLOCK_SIZE;
        int colB = col;
        if (rowB < N && colB < N) sharedB[ty][tx] = B[rowB * N + colB];
        else sharedB[ty][tx] = 0.0;
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
            for (int s = 0; s < STRIDE; s++) sumS[s] += sharedA[ty + s * BLOCK_SIZE][k] * sharedB[k][tx];
        __syncthreads();
    }
    for (int s = 0; s < STRIDE; s++) {
        int r = row + s * BLOCK_SIZE;
        if (r < N && col < N) C[r * N + col] = sumS[s];
    }
}

// ============================================================================================
// New kernels: register blocking. Requires N % 128 == 0.
// Block computes a BM x BN tile of C with (BM*BN)/(TM*TN) = 256 threads; each thread owns a
// TM x TN = 8x8 sub-tile in registers. Per BK step a thread loads TM + TN = 16 values from
// shared memory and performs TM*TN = 64 FMAs (vs 2 loads per FMA in the 32x32 tiling kernel).
// ============================================================================================
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
constexpr int RB_THREADS = (BM * BN) / (TM * TN);  // 256

__global__ void __launch_bounds__(RB_THREADS)
    sgemm_reg_block_2d(int N, const float* __restrict__ A, const float* __restrict__ B,
                       float* __restrict__ C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int threadCol = threadIdx.x % (BN / TN);  // 0..15
    const int threadRow = threadIdx.x / (BN / TN);  // 0..15

    A += (size_t)blockIdx.y * BM * N;
    B += (size_t)blockIdx.x * BN;
    C += (size_t)blockIdx.y * BM * N + (size_t)blockIdx.x * BN;

    const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;  // 32 rows x 8 cols per pass
    constexpr int strideA = RB_THREADS / BK;                                // 32
    const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;  // 2 rows x 128 cols per pass
    constexpr int strideB = RB_THREADS / BN;                                // 2

    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int bk = 0; bk < N; bk += BK) {
#pragma unroll
        for (int off = 0; off < BM; off += strideA)
            As[(innerRowA + off) * BK + innerColA] = A[(size_t)(innerRowA + off) * N + innerColA];
#pragma unroll
        for (int off = 0; off < BK; off += strideB)
            Bs[(innerRowB + off) * BN + innerColB] = B[(size_t)(innerRowB + off) * N + innerColB];
        __syncthreads();
        A += BK;
        B += (size_t)BK * N;

#pragma unroll
        for (int dot = 0; dot < BK; dot++) {
#pragma unroll
            for (int i = 0; i < TM; i++) regM[i] = As[(threadRow * TM + i) * BK + dot];
#pragma unroll
            for (int i = 0; i < TN; i++) regN[i] = Bs[dot * BN + threadCol * TN + i];
#pragma unroll
            for (int m = 0; m < TM; m++)
#pragma unroll
                for (int n = 0; n < TN; n++) acc[m * TN + n] = fmaf(regM[m], regN[n], acc[m * TN + n]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < TM; m++)
#pragma unroll
        for (int n = 0; n < TN; n++)
            C[(size_t)(threadRow * TM + m) * N + threadCol * TN + n] = acc[m * TN + n];
}

__global__ void __launch_bounds__(RB_THREADS)
    sgemm_vectorized(int N, const float* __restrict__ A, const float* __restrict__ B,
                     float* __restrict__ C) {
    __shared__ float As[BK * BM];  // transposed: As[k][m]
    __shared__ float Bs[BK * BN];

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    A += (size_t)blockIdx.y * BM * N;
    B += (size_t)blockIdx.x * BN;
    C += (size_t)blockIdx.y * BM * N + (size_t)blockIdx.x * BN;

    // A tile 128x8 = 256 float4 (one per thread); B tile 8x128 = 256 float4 (one per thread)
    const int innerRowA = threadIdx.x / (BK / 4), innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4), innerColB = threadIdx.x % (BN / 4);

    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int bk = 0; bk < N; bk += BK) {
        const float4 a4 = reinterpret_cast<const float4*>(&A[(size_t)innerRowA * N + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = a4.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = a4.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = a4.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = a4.w;
        reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4*>(&B[(size_t)innerRowB * N + innerColB * 4])[0];
        __syncthreads();
        A += BK;
        B += (size_t)BK * N;

#pragma unroll
        for (int dot = 0; dot < BK; dot++) {
#pragma unroll
            for (int i = 0; i < TM; i++) regM[i] = As[dot * BM + threadRow * TM + i];
#pragma unroll
            for (int i = 0; i < TN; i++) regN[i] = Bs[dot * BN + threadCol * TN + i];
#pragma unroll
            for (int m = 0; m < TM; m++)
#pragma unroll
                for (int n = 0; n < TN; n++) acc[m * TN + n] = fmaf(regM[m], regN[n], acc[m * TN + n]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 t;
            t.x = acc[m * TN + n + 0];
            t.y = acc[m * TN + n + 1];
            t.z = acc[m * TN + n + 2];
            t.w = acc[m * TN + n + 3];
            reinterpret_cast<float4*>(&C[(size_t)(threadRow * TM + m) * N + threadCol * TN + n])[0] = t;
        }
    }
}

// ============================================================================================
// Helpers
// ============================================================================================
__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__global__ void init_uniform(float* p, size_t n, uint32_t seed) {  // uniform in [-1, 1)
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = (float)hash32((uint32_t)i ^ seed) * 4.6566129e-10f - 1.0f;
}

// Non-negative floats compare like their bit patterns, so atomicMax on the int view works.
__global__ void max_abs_diff(const float* x, const float* ref, size_t n, int* out_diff, int* out_ref) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float d = 0.f, r = 0.f;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        d = fmaxf(d, fabsf(x[i] - ref[i]));
        r = fmaxf(r, fabsf(ref[i]));
    }
    atomicMax(out_diff, __float_as_int(d));
    atomicMax(out_ref, __float_as_int(r));
}

static double rel_error(const float* dC, const float* dRef, size_t n) {
    int* d;
    CUDA_CHECK(cudaMalloc(&d, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d, 0, 2 * sizeof(int)));
    max_abs_diff<<<1024, 256>>>(dC, dRef, n, d, d + 1);
    CUDA_CHECK(cudaGetLastError());
    int h[2];
    CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));
    float diff, ref;
    memcpy(&diff, &h[0], sizeof(float));
    memcpy(&ref, &h[1], sizeof(float));
    return ref > 0 ? diff / ref : diff;
}

struct Ctx {
    int N;
    float *A, *B, *C, *Cref;
    cublasHandle_t handle;
};

static void launch(const std::string& k, Ctx& c) {
    const int N = c.N;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(ceil_div(N, BLOCK_SIZE), ceil_div(N, BLOCK_SIZE));
    if (k == "k1_naive") MatMulGPUNaive<<<blocks, threads>>>(c.A, c.B, c.C, N);
    else if (k == "k2_coalesced") MatMulGPUCoalescedAccess<<<blocks, threads>>>(c.A, c.B, c.C, N);
    else if (k == "k3_smem_tiling") simpleTiling<<<blocks, threads>>>(c.A, c.B, c.C, N);
    else if (k == "k4_row_coarsen")
        RowBasedTiling<<<dim3(ceil_div(N, BLOCK_SIZE * STRIDE), ceil_div(N, BLOCK_SIZE)), threads>>>(c.A, c.B, c.C, N);
    else if (k == "k5_col_coarsen")
        colBasedTiling<<<dim3(ceil_div(N, BLOCK_SIZE), ceil_div(N, BLOCK_SIZE * STRIDE)), threads>>>(c.A, c.B, c.C, N);
    else if (k == "k6_reg_block_2d")
        sgemm_reg_block_2d<<<dim3(N / BN, N / BM), RB_THREADS>>>(N, c.A, c.B, c.C);
    else if (k == "k7_vectorized")
        sgemm_vectorized<<<dim3(N / BN, N / BM), RB_THREADS>>>(N, c.A, c.B, c.C);
    else if (k == "cublas_sgemm") {
        const float alpha = 1.f, beta = 0.f;
        // Row-major C = A B  <=>  column-major C^T = B^T A^T, so pass B first.
        CUBLAS_CHECK(cublasSgemm(c.handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, c.B, N, c.A, N,
                                 &beta, c.C, N));
    }
    CUDA_CHECK(cudaGetLastError());
}

static std::vector<int> parse_list(const char* s) {
    std::vector<int> out;
    std::string str(s);
    size_t pos = 0;
    while (pos < str.size()) {
        size_t comma = str.find(',', pos);
        if (comma == std::string::npos) comma = str.size();
        out.push_back(std::atoi(str.substr(pos, comma - pos).c_str()));
        pos = comma + 1;
    }
    return out;
}

int main(int argc, char** argv) {
    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384};
    int profile_n = 0;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--n") && a + 1 < argc) sizes = parse_list(argv[++a]);
        else if (!strcmp(argv[a], "--profile") && a + 1 < argc) profile_n = std::atoi(argv[++a]);
    }
    if (profile_n > 0) sizes = {profile_n};

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | CC %d.%d | %.0f MB\n", prop.name, prop.multiProcessorCount, prop.major,
           prop.minor, prop.totalGlobalMem / 1048576.0);
    printf("CSV,device,%s,%d,%d.%d,%.0f\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor,
           prop.totalGlobalMem / 1048576.0);
    printf("CSV,header,kernel,N,iters,avg_ms,min_ms,tflops,pct_of_cublas,rel_err_vs_cublas,status\n");

    const std::vector<std::string> kernels = {"cublas_sgemm",   "k1_naive",       "k2_coalesced",
                                              "k3_smem_tiling", "k4_row_coarsen", "k5_col_coarsen",
                                              "k6_reg_block_2d", "k7_vectorized"};
    // With a display attached to this GPU, the driver kills kernels that run for more than a few
    // seconds. In that case skip runs predicted (from the previous N, cost ~ N^3) to exceed 4 s.
    const bool watchdog = prop.kernelExecTimeoutEnabled != 0;
    printf("kernel run-time watchdog: %s\n", watchdog ? "ON (long runs will be skipped)" : "off");
    std::vector<double> last_ms(kernels.size(), 0.0);
    int last_n = 0;

    Ctx c;
    CUBLAS_CHECK(cublasCreate(&c.handle));

    for (int N : sizes) {
        if (N % BM != 0) {
            printf("skipping N=%d (must be a multiple of %d)\n", N, BM);
            continue;
        }
        c.N = N;
        const size_t n = (size_t)N * N;
        CUDA_CHECK(cudaMalloc(&c.A, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.B, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.C, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c.Cref, n * sizeof(float)));
        init_uniform<<<ceil_div((int)std::min(n, (size_t)1 << 30), 256), 256>>>(c.A, n, 17u);
        init_uniform<<<ceil_div((int)std::min(n, (size_t)1 << 30), 256), 256>>>(c.B, n, 29u);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        const double flops = 2.0 * N * (double)N * N;
        double cublas_ms = 0.0;
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));

        for (size_t ki = 0; ki < kernels.size(); ki++) {
            const std::string& k = kernels[ki];
            if (profile_n > 0) {
                launch(k, c);
                CUDA_CHECK(cudaDeviceSynchronize());
                printf("profiled %s at N=%d\n", k.c_str(), N);
                continue;
            }
            // The two uncoalesced/untiled kernels take minutes at N=16384; skip them there.
            if (N > 8192 && (k == "k1_naive" || k == "k2_coalesced")) {
                printf("CSV,gemm,%s,%d,0,nan,nan,nan,nan,nan,SKIPPED\n", k.c_str(), N);
                continue;
            }
            const double r = last_n ? (double)N / last_n : 0.0;
            const double predicted = last_ms[ki] * r * r * r;
            if (watchdog && predicted > 4000.0) {
                printf("N=%5d %-16s skipped: predicted %.1f s per launch exceeds the display watchdog\n",
                       N, k.c_str(), predicted / 1000.0);
                printf("CSV,gemm,%s,%d,0,nan,nan,nan,nan,nan,SKIPPED_WATCHDOG\n", k.c_str(), N);
                last_ms[ki] = predicted;
                continue;
            }
            CUDA_CHECK(cudaMemset(c.C, 0, n * sizeof(float)));
            CUDA_CHECK(cudaEventRecord(e0));
            launch(k, c);  // warm-up
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float first_ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&first_ms, e0, e1));
            int iters = 10;
            if (first_ms * 10 > 60000.f) iters = std::max(3, (int)(60000.f / first_ms));

            double total = 0.0;
            float best = 1e30f;
            for (int it = 0; it < iters; it++) {
                CUDA_CHECK(cudaEventRecord(e0));
                launch(k, c);
                CUDA_CHECK(cudaEventRecord(e1));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                total += ms;
                best = std::min(best, ms);
            }
            const double avg = total / iters;
            last_ms[ki] = avg;
            double err = 0.0;
            if (k == "cublas_sgemm") {
                cublas_ms = avg;
                CUDA_CHECK(cudaMemcpy(c.Cref, c.C, n * sizeof(float), cudaMemcpyDeviceToDevice));
            } else {
                err = rel_error(c.C, c.Cref, n);
            }
            const double tflops = flops / (avg * 1e-3) / 1e12;
            const double pct = 100.0 * cublas_ms / avg;
            const bool pass = err < 1e-4;
            printf("N=%5d %-16s avg %10.3f ms  min %10.3f ms  %7.3f TFLOPS  %6.1f%% of cuBLAS  rel_err %.2e  %s\n",
                   N, k.c_str(), avg, best, tflops, pct, err, pass ? "PASS" : "FAIL");
            printf("CSV,gemm,%s,%d,%d,%.4f,%.4f,%.4f,%.2f,%.3e,%s\n", k.c_str(), N, iters, avg, best, tflops,
                   pct, err, pass ? "PASS" : "FAIL");
            fflush(stdout);
        }
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        cudaFree(c.A);
        cudaFree(c.B);
        cudaFree(c.C);
        cudaFree(c.Cref);
        last_n = N;
    }
    CUBLAS_CHECK(cublasDestroy(c.handle));
    return 0;
}
