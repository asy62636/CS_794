# CS 794 follow-up: FlashAttention and SGEMM performance study

Follow-up work on PA2 (matrix multiplication) and PA4 (FlashAttention). The original kernels are
kept unchanged as baselines, and each new version changes one thing so its effect can be measured.

## How to run (Linux, NVIDIA GPU, CUDA toolkit + PyTorch installed)

```bash
git pull && git switch perf-benchmarks
cd perf
chmod +x run_all.sh profile_ncu.sh
./run_all.sh                       # ~10-15 min; or ./run_all.sh flash | ./run_all.sh gemm
sudo -E env "PATH=$PATH" ./profile_ncu.sh && sudo chown -R "$USER" results   # optional
git add results && git commit -m "Benchmark results" && git push
```

On a laptop: plug in the charger and use the performance power mode first.
Everything (logs, CSVs, ptxas register/spill reports, GPU clocks) lands in `perf/results/`.

## FlashAttention (`flash_attention/`)

Shapes follow PA4: B = 8, heads = 16, head dim D = 64, N = 32 ... 8192.
Every kernel is checked against a float64 CPU reference and against v3.

| Version | What changes | Why it matters |
|---|---|---|
| **v1** | Original PA4 Part D kernel. | Baseline. |
| **v2** | Same algorithm. Tiles are loaded and stored cooperatively, and shared-memory rows are padded to 65 floats. | In v1, thread `tx` copies its own 64-float row, so a warp's 32 loads are 256 bytes apart: 32 separate memory transactions instead of a few. Reading `Qi[tx*64 + x]` also puts all 32 threads on the same shared-memory bank (a 32-way conflict, which serialises the reads). A stride of 65 spreads them over 32 banks. |
| **v3** | FlashAttention-2 structure. Each block owns 64 query rows and streams over every K/V tile. The running max, running sum and the 64-float output accumulator stay in registers, and O is written once. | v1 re-reads and re-writes O, m and l in global memory for every (K tile, Q tile) pair: O(N²/Bc) extra traffic. v1 also only launches B·H = 128 blocks of a single warp each. v3 launches (N/64)·128 blocks, filling every SM. It also needs no m/l buffers. |
| | v3 also folds log2(e)/√D into Q once and uses `exp2f`. | This is the same trick FlashAttention-2 uses. A base-2 exponent maps to one hardware instruction. |
| | v3 rescales the accumulator once per 16 keys. | The online-softmax correction then costs 1/16 as many multiplies as rescaling once per key. |
| **v3h** | v3 reading fp16 Q/K/V, with fp32 maths. | Half the bytes per K/V tile load. |

`torch_attention_bench.py` measures PA4 Part A (naive attention, which builds the full N×N
matrix) and Part C (PyTorch's fused attention), in fp16 and fp32. It records peak extra memory
and out-of-memory failures. At N = 4096 the fp16 score matrix alone is 8·16·4096²·2 B = 4 GiB,
plus 4 GiB for the softmax output, which does not fit on an 8 GB GPU. The FlashAttention
kernels need no N×N storage.

## SGEMM (`gemm/`)

C = A·B in fp32 for N = 1024 ... 16384. Every kernel is checked against cuBLAS and reported as
TFLOPS and % of cuBLAS.

| Kernel | Idea |
|---|---|
| k1_naive / k2_coalesced | PA2. Consecutive threads read a column (k1) or consecutive columns (k2) of B. |
| k3_smem_tiling | PA2. 32×32 tiles of A and B in shared memory, so each element is read from global memory once per tile instead of 32 times. |
| k4_row_coarsen / k5_col_coarsen | PA2. Each thread computes 4 outputs, so one A (or B) value from shared memory is reused 4 times from a register. |
| **k6_reg_block_2d** | New. Each block computes a 128×128 tile of C, and each of its 256 threads an 8×8 sub-tile kept in registers. Per k-step a thread loads 8 + 8 values from shared memory and does 64 FMAs. That ratio is 2 loads per FMA in k3 and 0.25 here, so the kernel is limited by maths, not shared-memory bandwidth. |
| **k7_vectorized** | New. k6 plus 128-bit (`float4`) global loads and stores, with the A tile stored transposed in shared memory. Both operands of the inner loop are then contiguous. |
| cublas_sgemm | Reference. Row-major C = A·B is computed as column-major Cᵀ = Bᵀ·Aᵀ by swapping the operands, so no transpose is needed. |

## Notes on methodology

- All timings use CUDA events around the kernel only. Each kernel gets one warm-up run, then
  10 timed runs (at least 3 if a single run takes more than 6 s).
- PA2's original timing used `cudaMallocManaged`, so the first kernel also paid for page migration
  to the GPU. Here every input is in device memory before timing starts.
- v1/v2 accumulate into O, m and l, so their state is reset before every timed launch.
- `results/ptxas_*.txt` holds registers per thread and spill bytes per kernel.
- `profile_ncu.sh` records the kernel-level evidence: occupancy, bank conflicts, sectors per
  request (coalescing) and DRAM throughput.
