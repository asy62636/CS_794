import numpy as np
import torch

# ── Hyperparameters ────────────────────────────────────────────────────────────
B   = 8          # batch size
NH  = 16         # number of heads
D   = 64         # head dimension
SEQ_LENS = [32, 64, 128, 256, 512, 1024, 2048]
WARMUP_ITERS = 3
TIMING_ITERS = 10
DTYPE  = torch.float16
DEVICE = "cuda"

def check_cuda_vs_reference(N):
    # load cuda output
    cuda_out = np.fromfile(f"N{N}_cuda_out.bin", dtype=np.float32)
    cuda_out = cuda_out.reshape(B, NH, N, D)

    # load Part C reference
    ref = np.load(f"flash_attn_outputs/part_c/N{N}_output.npy")

    max_diff  = np.abs(cuda_out - ref).max()
    mean_diff = np.abs(cuda_out - ref).mean()
    passed = max_diff < 1e-3
    print(f"N={N:5d}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  {'✓ PASS' if passed else '✗ FAIL'}")

if __name__ == "__main__":
    for N in [32, 64, 128, 256, 512, 1024, 2048]:
        check_cuda_vs_reference(N)