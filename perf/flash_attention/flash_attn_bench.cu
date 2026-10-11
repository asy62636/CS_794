// FlashAttention forward-pass benchmark (CS 794, PA4 follow-up).
//
// Kernels compared (all compute O = softmax(Q K^T / sqrt(D)) V, shapes [B, NH, N, D]):
//   v1  : the original PA4 Part-D kernel, kept as the baseline. FA-1 loop order (outer loop over
//         K/V tiles, inner loop over Q tiles), one 32-thread block per (batch, head), running
//         max/sum (m, l) and the output O re-read and re-written in global memory every tile.
//   v2  : same algorithm and launch shape as v1, with every memory access fixed:
//           - cooperative tile loads/stores, so consecutive threads touch consecutive addresses
//             (coalesced global memory transactions instead of 32 scattered ones per warp)
//           - shared-memory rows padded by one float (stride D+1), so the 32 threads of a warp
//             reading "their own row" hit 32 different banks instead of the same bank
//   v3  : FlashAttention-2 style restructuring:
//           - loop order swapped: each block owns a tile of 64 query rows and streams over all
//             K/V tiles, so m, l and the output accumulator never leave registers and O is
//             written exactly once (no global read-modify-write per tile, no m/l buffers)
//           - parallel over query tiles too: grid = (N/64) x (B*NH) blocks instead of B*NH,
//             which fills every SM even for small batch * heads
//           - softmax in base 2 (exp2f) with log2(e)/sqrt(D) folded into Q once
//           - scores processed in register chunks of 16 keys, so the output is rescaled once per
//             chunk instead of once per key
//   v3h : v3 reading fp16 Q/K/V (half the global-memory traffic), fp32 math and fp32 output.
//
// Correctness: every kernel is checked against a float64 CPU reference (two heads, all or a
// strided subset of rows) and against v3 on the full output tensor.
//
// Usage:  ./flash_attn_bench                 -> full sweep, prints "CSV,..." rows
//         ./flash_attn_bench --n 1024,4096   -> custom sequence lengths
//         ./flash_attn_bench --profile 1024  -> one launch per kernel (for Nsight Compute)

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
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

constexpr int B = 8;    // batch size
constexpr int NH = 16;  // heads
constexpr int D = 64;   // head dimension
constexpr int BH = B * NH;

// ============================================================================================
// v1: original PA4 kernel (logic unchanged; macros renamed, max() -> fmaxf()).
// Requires N % 32 == 0. O must start at 0, L at 0, M at -1e38 before every launch.
// ============================================================================================
constexpr int V1_BR = 32;
constexpr int V1_BC = 32;
constexpr int V1_SMEM = V1_BR * D + 2 * V1_BC * D + V1_BR * V1_BC;

__global__ void flash_v1(float* Q, float* K, float* V, float* O, float* M, float* L, int N) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tx = threadIdx.x;

    int qkv_base = (b * NH + h) * N * D;
    int lm_base = (b * NH + h) * N;

    __shared__ float smem[V1_SMEM];
    float* Qi = smem;
    float* Kj = Qi + V1_BR * D;
    float* Vj = Kj + V1_BC * D;
    float* S = Vj + V1_BC * D;

    int Tr = (N + V1_BR - 1) / V1_BR;
    int Tc = (N + V1_BC - 1) / V1_BC;

    for (int j = 0; j < Tc; j++) {
        for (int x = 0; x < D; x++) {
            int kv_idx = qkv_base + (j * V1_BC + tx) * D + x;
            Kj[tx * D + x] = K[kv_idx];
            Vj[tx * D + x] = V[kv_idx];
        }
        __syncthreads();
        for (int i = 0; i < Tr; i++) {
            for (int x = 0; x < D; x++) {
                int Q_idx = qkv_base + (i * V1_BR + tx) * D + x;
                Qi[tx * D + x] = Q[Q_idx];
            }
            __syncthreads();

            float mPrev = M[lm_base + i * V1_BR + tx];
            float lPrev = L[lm_base + i * V1_BR + tx];

            for (int c = 0; c < V1_BC; c++) {
                float sum = 0;
                for (int x = 0; x < D; x++) sum += Qi[tx * D + x] * Kj[c * D + x];
                S[tx * V1_BC + c] = sum / sqrtf(float(D));
            }
            float mNew = mPrev;
            for (int c = 0; c < V1_BC; c++) mNew = fmaxf(mNew, S[tx * V1_BC + c]);

            float tile_norm = 0;
            for (int c = 0; c < V1_BC; c++) {
                S[tx * V1_BC + c] = expf(S[tx * V1_BC + c] - mNew);
                tile_norm += S[tx * V1_BC + c];
            }
            float lNew = expf(mPrev - mNew) * lPrev + tile_norm;

            int i_row = i * V1_BR + tx;
            for (int col_idx = 0; col_idx < D; col_idx++) {
                float mul_res = 0;
                for (int r = 0; r < V1_BC; r++) mul_res += S[tx * V1_BC + r] * Vj[r * D + col_idx];
                int O_idx = qkv_base + i_row * D + col_idx;
                O[O_idx] = (O[O_idx] * expf(mPrev - mNew) * lPrev + mul_res) / lNew;
            }
            M[lm_base + i * V1_BR + tx] = mNew;
            L[lm_base + i * V1_BR + tx] = lNew;
            __syncthreads();
        }
    }
}

// ============================================================================================
// v2: v1's algorithm with coalesced global access and bank-conflict-free shared memory.
// Requires N % 32 == 0. Same state initialisation as v1.
// ============================================================================================
constexpr int V2_BR = 32;
constexpr int V2_BC = 32;
constexpr int PAD_D = D + 1;      // padded row stride for [rows x D] tiles
constexpr int PAD_C = V2_BC + 1;  // padded row stride for the [Br x Bc] score tile

__global__ void __launch_bounds__(V2_BR)
    flash_v2(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
             float* __restrict__ O, float* __restrict__ M, float* __restrict__ L, int N) {
    // 4 * 32*65 + 32*33 floats = 37.5 KB of static shared memory (< 48 KB limit)
    __shared__ float Qi[V2_BR * PAD_D];
    __shared__ float Oi[V2_BR * PAD_D];
    __shared__ float Kj[V2_BC * PAD_D];
    __shared__ float Vj[V2_BC * PAD_D];
    __shared__ float S[V2_BR * PAD_C];

    const int tx = threadIdx.x;
    const size_t bh = (size_t)blockIdx.x * NH + blockIdx.y;
    const size_t qkv_base = bh * N * D;
    const size_t lm_base = bh * N;
    const float scale = rsqrtf((float)D);
    const int Tr = N / V2_BR;
    const int Tc = N / V2_BC;

    for (int j = 0; j < Tc; j++) {
        // Cooperative, coalesced K/V tile load: thread tx copies elements tx, tx+32, tx+64, ...
        const float* Kg = K + qkv_base + (size_t)j * V2_BC * D;
        const float* Vg = V + qkv_base + (size_t)j * V2_BC * D;
        for (int idx = tx; idx < V2_BC * D; idx += V2_BR) {
            const int r = idx / D, c = idx % D;
            Kj[r * PAD_D + c] = Kg[idx];
            Vj[r * PAD_D + c] = Vg[idx];
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++) {
            const size_t tile_off = qkv_base + (size_t)i * V2_BR * D;
            for (int idx = tx; idx < V2_BR * D; idx += V2_BR) {
                const int r = idx / D, c = idx % D;
                Qi[r * PAD_D + c] = Q[tile_off + idx];
                Oi[r * PAD_D + c] = O[tile_off + idx];
            }
            __syncthreads();

            const size_t row = lm_base + (size_t)i * V2_BR + tx;  // consecutive per thread
            const float mPrev = M[row];
            const float lPrev = L[row];

            // Scores: Qi row (stride 65 -> conflict-free across threads), Kj row (broadcast).
            float mNew = mPrev;
            for (int c = 0; c < V2_BC; c++) {
                float acc = 0.f;
#pragma unroll 16
                for (int x = 0; x < D; x++) acc = fmaf(Qi[tx * PAD_D + x], Kj[c * PAD_D + x], acc);
                acc *= scale;
                S[tx * PAD_C + c] = acc;
                mNew = fmaxf(mNew, acc);
            }
            float tile_sum = 0.f;
            for (int c = 0; c < V2_BC; c++) {
                const float p = expf(S[tx * PAD_C + c] - mNew);
                S[tx * PAD_C + c] = p;
                tile_sum += p;
            }
            const float alpha = expf(mPrev - mNew);
            const float lNew = alpha * lPrev + tile_sum;
            const float old_scale = alpha * lPrev / lNew;
            const float inv_l = 1.f / lNew;

            for (int x = 0; x < D; x++) {
                float pv = 0.f;
#pragma unroll 8
                for (int c = 0; c < V2_BC; c++) pv = fmaf(S[tx * PAD_C + c], Vj[c * PAD_D + x], pv);
                Oi[tx * PAD_D + x] = Oi[tx * PAD_D + x] * old_scale + pv * inv_l;
            }
            M[row] = mNew;
            L[row] = lNew;
            __syncthreads();

            // Coalesced write-back of the output tile.
            for (int idx = tx; idx < V2_BR * D; idx += V2_BR) {
                const int r = idx / D, c = idx % D;
                O[tile_off + idx] = Oi[r * PAD_D + c];
            }
            __syncthreads();
        }
    }
}

// ============================================================================================
// v3 / v3h: FlashAttention-2 style kernel. One thread per query row, 64 rows per block.
// Works for any N (partial tiles are masked). No state initialisation needed.
// ============================================================================================
constexpr int V3_BR = 64;   // query rows per block (= threads per block)
constexpr int V3_BC = 64;   // keys per shared-memory K/V tile
constexpr int V3_KCH = 16;  // keys per register chunk (one output rescale per chunk)

template <typename T>
__device__ __forceinline__ float to_float(T x);
template <>
__device__ __forceinline__ float to_float<float>(float x) {
    return x;
}
template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

template <typename T>
__global__ void __launch_bounds__(V3_BR)
    flash_v3(const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V,
             float* __restrict__ O, int N, float q_scale) {
    // Ks doubles as a padded [64 x 65] staging buffer for the Q tile load and the O tile store.
    __shared__ float Ks[V3_BC * PAD_D];
    __shared__ float Vs[V3_BC * D];

    const int tx = threadIdx.x;
    const int row0 = blockIdx.x * V3_BR;
    const size_t base = (size_t)blockIdx.y * N * D;

    // ---- Q tile: coalesced global load -> padded smem -> this thread's row into registers ----
    for (int idx = tx; idx < V3_BR * D; idx += V3_BR) {
        const int r = idx / D, c = idx % D;
        const int qrow = row0 + r;
        Ks[r * PAD_D + c] = (qrow < N) ? to_float(Q[base + (size_t)qrow * D + c]) : 0.f;
    }
    __syncthreads();
    float q[D], o[D];
#pragma unroll
    for (int x = 0; x < D; x++) {
        q[x] = Ks[tx * PAD_D + x] * q_scale;  // q_scale = log2(e) / sqrt(D)  -> base-2 softmax
        o[x] = 0.f;
    }
    __syncthreads();

    float m = -INFINITY;  // running max (base-2 units)
    float l = 0.f;        // running softmax denominator

    const int Tc = (N + V3_BC - 1) / V3_BC;
    for (int j = 0; j < Tc; j++) {
        const int key0 = j * V3_BC;
        for (int idx = tx; idx < V3_BC * D; idx += V3_BR) {
            const int r = idx / D, c = idx % D;
            const bool ok = (key0 + r) < N;
            const size_t g = base + (size_t)(key0 + r) * D + c;
            Ks[r * D + c] = ok ? to_float(K[g]) : 0.f;  // dense layout: reads are broadcasts
            Vs[r * D + c] = ok ? to_float(V[g]) : 0.f;
        }
        __syncthreads();

#pragma unroll 1
        for (int c0 = 0; c0 < V3_BC; c0 += V3_KCH) {
            float s[V3_KCH];
            float m_new = m;
#pragma unroll
            for (int c = 0; c < V3_KCH; c++) {
                float acc = 0.f;
#pragma unroll
                for (int x = 0; x < D; x++) acc = fmaf(q[x], Ks[(c0 + c) * D + x], acc);
                s[c] = (key0 + c0 + c < N) ? acc : -INFINITY;
                m_new = fmaxf(m_new, s[c]);
            }
            // Rescale previous accumulator once per chunk (alpha = 0 on the very first chunk).
            const float alpha = exp2f(m - m_new);
            l *= alpha;
#pragma unroll
            for (int x = 0; x < D; x++) o[x] *= alpha;
#pragma unroll
            for (int c = 0; c < V3_KCH; c++) {
                const float p = exp2f(s[c] - m_new);
                l += p;
#pragma unroll
                for (int x = 0; x < D; x++) o[x] = fmaf(p, Vs[(c0 + c) * D + x], o[x]);
            }
            m = m_new;
        }
        __syncthreads();
    }

    // ---- normalise once, stage through padded smem, coalesced store ----
    const float inv_l = 1.f / l;
#pragma unroll
    for (int x = 0; x < D; x++) Ks[tx * PAD_D + x] = o[x] * inv_l;
    __syncthreads();
    for (int idx = tx; idx < V3_BR * D; idx += V3_BR) {
        const int r = idx / D, c = idx % D;
        const int orow = row0 + r;
        if (orow < N) O[base + (size_t)orow * D + c] = Ks[r * PAD_D + c];
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

// Standard-normal values rounded to fp16, stored both as fp16 and as the identical fp32 value,
// so every kernel (and the CPU reference) sees exactly the same numbers.
__global__ void init_normal(float* f, __half* h, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t a = hash32((uint32_t)(2 * i) ^ seed);
    const uint32_t b = hash32((uint32_t)(2 * i + 1) ^ (seed * 0x9E3779B9U));
    const float u1 = ((float)a + 1.0f) * 2.3283064e-10f;  // (0, 1]
    const float u2 = (float)b * 2.3283064e-10f;           // [0, 1)
    const float z = sqrtf(-2.0f * logf(u1)) * cospif(2.0f * u2);
    h[i] = __float2half(z);
    f[i] = __half2float(h[i]);
}

__global__ void fill(float* p, size_t n, float v) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

static int ceil_div(size_t a, int b) { return (int)((a + b - 1) / b); }

struct Buffers {
    size_t n_qkv = 0, n_lm = 0;
    float *Qf = nullptr, *Kf = nullptr, *Vf = nullptr, *O = nullptr, *M = nullptr, *L = nullptr;
    __half *Qh = nullptr, *Kh = nullptr, *Vh = nullptr;
};

static void reset_state(Buffers& bf) {
    CUDA_CHECK(cudaMemset(bf.O, 0, bf.n_qkv * sizeof(float)));
    CUDA_CHECK(cudaMemset(bf.L, 0, bf.n_lm * sizeof(float)));
    fill<<<ceil_div(bf.n_lm, 256), 256>>>(bf.M, bf.n_lm, -1e38f);
    CUDA_CHECK(cudaGetLastError());
}

static void launch(const std::string& k, Buffers& bf, int N) {
    if (k == "v1") {
        flash_v1<<<dim3(B, NH), V1_BR>>>(bf.Qf, bf.Kf, bf.Vf, bf.O, bf.M, bf.L, N);
    } else if (k == "v2") {
        flash_v2<<<dim3(B, NH), V2_BR>>>(bf.Qf, bf.Kf, bf.Vf, bf.O, bf.M, bf.L, N);
    } else {
        const float q_scale = 1.4426950408889634f / sqrtf((float)D);
        dim3 grid(ceil_div(N, V3_BR), BH);
        if (k == "v3")
            flash_v3<float><<<grid, V3_BR>>>(bf.Qf, bf.Kf, bf.Vf, bf.O, N, q_scale);
        else
            flash_v3<__half><<<grid, V3_BR>>>(bf.Qh, bf.Kh, bf.Vh, bf.O, N, q_scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

// float64 reference for one (b,h) slice and a subset of rows; returns the max abs error.
static double cpu_check(const std::vector<float>& Q, const std::vector<float>& K,
                        const std::vector<float>& V, const std::vector<float>& O, int N, int bh,
                        int row_step) {
    const size_t base = (size_t)bh * N * D;
    std::vector<double> s(N);
    double worst = 0.0;
    for (int r = 0; r < N; r += row_step) {
        const float* q = &Q[base + (size_t)r * D];
        double mx = -1e300;
        for (int c = 0; c < N; c++) {
            const float* k = &K[base + (size_t)c * D];
            double acc = 0.0;
            for (int x = 0; x < D; x++) acc += (double)q[x] * (double)k[x];
            s[c] = acc / std::sqrt((double)D);
            mx = std::max(mx, s[c]);
        }
        double denom = 0.0;
        for (int c = 0; c < N; c++) {
            s[c] = std::exp(s[c] - mx);
            denom += s[c];
        }
        for (int x = 0; x < D; x++) {
            double acc = 0.0;
            for (int c = 0; c < N; c++) acc += s[c] * (double)V[base + (size_t)c * D + x];
            worst = std::max(worst, std::fabs(acc / denom - (double)O[base + (size_t)r * D + x]));
        }
    }
    return worst;
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

static void alloc_and_init(Buffers& bf, int N) {
    bf.n_qkv = (size_t)BH * N * D;
    bf.n_lm = (size_t)BH * N;
    CUDA_CHECK(cudaMalloc(&bf.Qf, bf.n_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bf.Kf, bf.n_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bf.Vf, bf.n_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bf.O, bf.n_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bf.Qh, bf.n_qkv * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&bf.Kh, bf.n_qkv * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&bf.Vh, bf.n_qkv * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&bf.M, bf.n_lm * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bf.L, bf.n_lm * sizeof(float)));
    const int blocks = ceil_div(bf.n_qkv, 256);
    init_normal<<<blocks, 256>>>(bf.Qf, bf.Qh, bf.n_qkv, 1234u + (uint32_t)N);
    init_normal<<<blocks, 256>>>(bf.Kf, bf.Kh, bf.n_qkv, 5678u + (uint32_t)N);
    init_normal<<<blocks, 256>>>(bf.Vf, bf.Vh, bf.n_qkv, 9012u + (uint32_t)N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void free_buffers(Buffers& bf) {
    cudaFree(bf.Qf); cudaFree(bf.Kf); cudaFree(bf.Vf); cudaFree(bf.O);
    cudaFree(bf.Qh); cudaFree(bf.Kh); cudaFree(bf.Vh); cudaFree(bf.M); cudaFree(bf.L);
    bf = Buffers();
}

int main(int argc, char** argv) {
    std::vector<int> seq_lens = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int profile_n = 0;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--n") && a + 1 < argc) seq_lens = parse_list(argv[++a]);
        else if (!strcmp(argv[a], "--profile") && a + 1 < argc) profile_n = std::atoi(argv[++a]);
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs: %d | CC %d.%d | %.0f MB | smem/block %zu B\n", prop.name,
           prop.multiProcessorCount, prop.major, prop.minor, prop.totalGlobalMem / 1048576.0,
           prop.sharedMemPerBlock);
    printf("CSV,device,%s,%d,%d.%d,%.0f\n", prop.name, prop.multiProcessorCount, prop.major,
           prop.minor, prop.totalGlobalMem / 1048576.0);

    const std::vector<std::string> kernels = {"v3", "v3h", "v2", "v1"};
    // With a display attached to this GPU, the driver kills kernels that run for more than a few
    // seconds. In that case skip runs predicted (from the previous N, cost ~ N^2) to exceed 4 s.
    const bool watchdog = prop.kernelExecTimeoutEnabled != 0;
    printf("kernel run-time watchdog: %s\n", watchdog ? "ON (long runs will be skipped)" : "off");
    std::vector<double> last_ms(kernels.size(), 0.0);
    int last_n = 0;

    if (profile_n > 0) {  // one launch of each kernel, for ncu
        Buffers bf;
        alloc_and_init(bf, profile_n);
        for (const auto& k : kernels) {
            reset_state(bf);
            launch(k, bf, profile_n);
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("profiled %s at N=%d\n", k.c_str(), profile_n);
        }
        free_buffers(bf);
        return 0;
    }

    printf("CSV,header,kernel,in_dtype,N,iters,avg_ms,min_ms,tflops,max_err_vs_fp64,"
           "max_diff_vs_v3,extra_state_MB,status\n");

    for (int N : seq_lens) {
        if (N % 32 != 0) {
            printf("skipping N=%d (must be a multiple of 32)\n", N);
            continue;
        }
        Buffers bf;
        alloc_and_init(bf, N);

        std::vector<float> hQ(bf.n_qkv), hK(bf.n_qkv), hV(bf.n_qkv), hO(bf.n_qkv), hRef;
        CUDA_CHECK(cudaMemcpy(hQ.data(), bf.Qf, bf.n_qkv * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hK.data(), bf.Kf, bf.n_qkv * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hV.data(), bf.Vf, bf.n_qkv * sizeof(float), cudaMemcpyDeviceToHost));

        const double flops = 4.0 * BH * (double)N * N * D;  // QK^T and PV
        const int row_step = N <= 1024 ? 1 : N / 512;      // CPU check: <= ~512 rows per head

        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));

        for (size_t ki = 0; ki < kernels.size(); ki++) {
            const std::string& k = kernels[ki];
            const double predicted = last_n ? last_ms[ki] * ((double)N / last_n) * ((double)N / last_n) : 0.0;
            if (watchdog && predicted > 4000.0) {
                printf("N=%5d %-4s skipped: predicted %.1f s per launch exceeds the display watchdog\n",
                       N, k.c_str(), predicted / 1000.0);
                printf("CSV,flash,%s,%s,%d,0,nan,nan,nan,nan,nan,nan,SKIPPED_WATCHDOG\n", k.c_str(),
                       k == "v3h" ? "fp16" : "fp32", N);
                last_ms[ki] = predicted;
                continue;
            }
            // warm-up (also decides the iteration count for very slow configs)
            reset_state(bf);
            CUDA_CHECK(cudaEventRecord(e0));
            launch(k, bf, N);
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float first_ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&first_ms, e0, e1));
            int iters = 10;
            if (first_ms * 10 > 60000.f) iters = std::max(3, (int)(60000.f / first_ms));

            double total = 0.0;
            float best = 1e30f;
            for (int it = 0; it < iters; it++) {
                reset_state(bf);  // v1/v2 accumulate into O/M/L, so every run starts clean
                CUDA_CHECK(cudaEventRecord(e0));
                launch(k, bf, N);
                CUDA_CHECK(cudaEventRecord(e1));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0.f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                total += ms;
                best = std::min(best, ms);
            }
            const double avg = total / iters;
            last_ms[ki] = avg;
            CUDA_CHECK(cudaMemcpy(hO.data(), bf.O, bf.n_qkv * sizeof(float), cudaMemcpyDeviceToHost));

            bool has_nan = false;
            for (size_t i = 0; i < bf.n_qkv && !has_nan; i++) has_nan = std::isnan(hO[i]);
            const double err = std::max(cpu_check(hQ, hK, hV, hO, N, 0, row_step),
                                        cpu_check(hQ, hK, hV, hO, N, BH - 1, row_step));
            double diff_v3 = 0.0;
            if (k == "v3") {
                hRef = hO;
            } else if (!hRef.empty()) {
                for (size_t i = 0; i < bf.n_qkv; i++)
                    diff_v3 = std::max(diff_v3, (double)std::fabs(hO[i] - hRef[i]));
            }
            const double extra_mb = (k == "v1" || k == "v2") ? 2.0 * bf.n_lm * 4 / 1048576.0 : 0.0;
            const bool pass = !has_nan && err < 1e-3 && diff_v3 < 1e-3;
            const char* in_dtype = (k == "v3h") ? "fp16" : "fp32";

            printf("N=%5d %-4s avg %10.3f ms  min %10.3f ms  %7.3f TFLOPS  err %.2e  diff_v3 %.2e  %s\n",
                   N, k.c_str(), avg, best, flops / (avg * 1e-3) / 1e12, err, diff_v3,
                   pass ? "PASS" : "FAIL");
            printf("CSV,flash,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.3e,%.3e,%.3f,%s\n", k.c_str(), in_dtype,
                   N, iters, avg, best, flops / (avg * 1e-3) / 1e12, err, diff_v3, extra_mb,
                   pass ? "PASS" : "FAIL");
            fflush(stdout);
        }
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        free_buffers(bf);
        last_n = N;
    }
    return 0;
}
