"""
Flash Attention Assignment
Covers:
  - QKV tensor generation & saving  (setup)
  - Part A : Standard scaled dot-product attention (PyTorch matmul)
  - Part C : PyTorch built-in fused attention (correctness reference + timing)

Outputs written to  ./flash_attn_outputs/
  qkv/   – Q, K, V tensors for every N  (float16, saved as float32 for numpy)
  part_a/ – outputs + timings
  part_c/ – outputs + timings
"""

import os, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ── Hyperparameters ────────────────────────────────────────────────────────────
B   = 8          # batch size
NH  = 16         # number of heads
D   = 64         # head dimension
SEQ_LENS = [32, 64, 128, 256, 512, 1024, 2048]
WARMUP_ITERS = 3
TIMING_ITERS = 10
DTYPE  = torch.float16
DEVICE = "cuda"

# ── Output directories ─────────────────────────────────────────────────────────
BASE    = "flash_attn_outputs"
QKV_DIR = os.path.join(BASE, "qkv")
A_DIR   = os.path.join(BASE, "part_a")
C_DIR   = os.path.join(BASE, "part_c")
B_DIR   = os.path.join(BASE, "part_b")
for d in [QKV_DIR, A_DIR, B_DIR, C_DIR]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SETUP – generate & save QKV tensors
# ══════════════════════════════════════════════════════════════════════════════

def generate_and_save_qkv():
    """
    For every N, create Q K V of shape (B, NH, N, D) in float16 on CUDA,
    save them as float32 .npy files so numpy can read them back easily.
    Returns a dict  N -> (Q, K, V)  kept on GPU for immediate use.
    """
    print("=" * 60)
    print("SETUP: Generating & saving QKV tensors")
    print("=" * 60)

    tensors = {}
    torch.manual_seed(42)          # reproducibility

    for N in SEQ_LENS:
        Q = torch.randn(B, NH, N, D, dtype=DTYPE, device=DEVICE)
        K = torch.randn(B, NH, N, D, dtype=DTYPE, device=DEVICE)
        V = torch.randn(B, NH, N, D, dtype=DTYPE, device=DEVICE)

        # Save as float32 numpy arrays (float16 numpy support is limited)
        prefix = os.path.join(QKV_DIR, f"N{N}")
        np.save(f"{prefix}_Q.npy", Q.cpu().float().numpy())
        np.save(f"{prefix}_K.npy", K.cpu().float().numpy())
        np.save(f"{prefix}_V.npy", V.cpu().float().numpy())

        tensors[N] = (Q, K, V)
        print(f"  N={N:5d}  shape={tuple(Q.shape)}  saved to {prefix}_{{Q,K,V}}.npy")


    print()
    return tensors


def load_qkv(N: int):
    """Load previously saved QKV for sequence length N back onto GPU as float16."""
    prefix = os.path.join(QKV_DIR, f"N{N}")
    Q = torch.from_numpy(np.load(f"{prefix}_Q.npy")).to(dtype=DTYPE, device=DEVICE)
    K = torch.from_numpy(np.load(f"{prefix}_K.npy")).to(dtype=DTYPE, device=DEVICE)
    V = torch.from_numpy(np.load(f"{prefix}_V.npy")).to(dtype=DTYPE, device=DEVICE)
    return (Q, K, V)


# ══════════════════════════════════════════════════════════════════════════════
# PART A – Standard scaled dot-product attention
# ══════════════════════════════════════════════════════════════════════════════

def standard_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    O = softmax( Q K^T / sqrt(D) ) V
    - Uses torch.matmul for both multiplications
    - Uses torch.softmax(..., dim=-1) along the key dimension
    - Forms the full N×N attention matrix explicitly
    Shape: (B, NH, N, D)
    """
    K_T = torch.transpose(K, -2, -1)
    S = torch.matmul(Q, K_T) / torch.sqrt(torch.tensor([D]))[0]
    P = torch.softmax(S, dim = -1)
    res = torch.matmul(P, V)
    return res

def run_part_a(tensors: dict):
    print("=" * 60)
    print("PART A: Standard attention")
    print("=" * 60)

    runtimes = []
    
    for N in SEQ_LENS:
        Q, K, V = tensors[N]

        # Warmup
        torch.cuda.synchronize()
        for _ in range(WARMUP_ITERS):
            _ = standard_attention(Q, K, V)
        torch.cuda.synchronize()

        # Timing
        start = time.perf_counter()
        for _ in range(TIMING_ITERS):
            out = standard_attention(Q, K, V)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000 / TIMING_ITERS

        runtimes.append(elapsed_ms)

        # Save output
        out_path = os.path.join(A_DIR, f"N{N}_output.npy")
        np.save(out_path, out.cpu().float().numpy())

        print(f"  N={N:5d}  runtime={elapsed_ms:8.2f} ms  output saved → {out_path}")

    # Save timings
    np.save(os.path.join(A_DIR, "runtimes.npy"), np.array(runtimes))
    print()
    return runtimes

# ══════════════════════════════════════════════════════════════════════════════
# PART B – Flash Attention Algorithm in PyTorch
# ══════════════════════════════════════════════════════════════════════════════

def flash_attention(Q : torch.Tensor, K : torch.Tensor, V : torch.Tensor) -> torch.tensor:

    # B = 
    #define constants
    M_sram = 32 * 1024
    N = Q.shape[-2]
    Bc = M_sram // (4 * D)
    Br = min(M_sram//(4 * D), D)
    Tc = (N + Bc - 1)// Bc
    Tr = (N + Br - 1) // Br

    #define memory buffers
    O = torch.zeros(B, NH, N, D, device = Q.device, dtype = Q.dtype)
    L = torch.zeros(B, NH, N, device = Q.device, dtype = Q.dtype)
    M = torch.full((B, NH, N), float('-inf'), device = Q.device, dtype = Q.dtype)

    for j in range(Tc):
        #load K, V blocks
        Kj = K[:, :, j * Bc : min((j + 1) * Bc, N), :]
        Vj = V[:, :, j * Bc : min((j + 1) * Bc, N), :]

        for i in range(Tr):
            #load Q blocks and Li, Mi, Oi
            Qi = Q[:, :, i * Br : min((i+1) * Br, N), :]
            Oi_old = O[:, :, i * Br : min((i+1) * Br, N), :]
            Li_old = L[:, :, i * Br : min((i + 1) * Br, N)]
            Mi_old = M[:, :, i * Br : min((i + 1) * Br, N)]

            Sij = torch.matmul(Qi, torch.transpose(Kj, -2, -1)) / (D ** 0.5) #shape : [B x Nh x Br x Bc]
            mij = Sij.max(dim = -1).values #shape : [B x Nh x Br]
            Pij = torch.exp(Sij - mij.unsqueeze(-1)) #shape : [B x Nh x Br x Bc]
            lij = Pij.sum(dim = -1) # shape : [B x Nh x Br]

            m_new = torch.maximum(Mi_old, mij) #[B, Nh, Br]
            exp_old = torch.exp(Mi_old - m_new) #[B, Nh, Br]
            exp_new = torch.exp(mij -m_new) #[B, Nh, Br]
            l_new = exp_old * Li_old + exp_new * lij #[B, Nh, Br]

            O[:, :, i * Br : min((i+1)*Br, N), :] = (exp_old.unsqueeze(-1) * Li_old.unsqueeze(-1) * Oi_old + exp_new.unsqueeze(-1) * torch.matmul(Pij, Vj)) / l_new.unsqueeze(-1) #[B, Nh, Br, d]
            L[:, :, i * Br : min((i + 1) * Br, N)] = l_new
            M[:, :, i * Br : min((i + 1) * Br, N)] = m_new

    return O

def flaash_attention(Q : torch.Tensor, K : torch.tensor, V : torch.Tensor) -> torch.Tensor:
    #defining constants
    N = Q.shape[-2]
    M_sram = 32 * 1024
    Bc = M_sram // (4*D)
    Br = min(M_sram // (4 * D), D)
    Tc = (N + Bc - 1) // Bc
    Tr = (N + Br - 1) // Br

    #define memory buffers:
    O = torch.zeros((B, NH, N, D), device = Q.device, dtype = Q.dtype)
    L = torch.zeros((B, NH, N), device = Q.device, dtype = Q.dtype)
    M = torch.ones((B, NH, N), device = Q.device, dtype = Q.dtype) * float("-inf")

    for j in range(Tc):
        Kj = K[:, :, j * Bc : min((j + 1) * Bc, N), :] #B, NH, BC, D
        Vj = V[:, :, j * Bc : min((j + 1) * Bc, N), :] #B, NH, BC, D

        for i in range(Tr):
            Qi = Q[:, :, i * Br : min((i + 1) * Br, N), :] #B, NH, Br, D
            O_old = O[:, :, i * Br : min((i + 1) * Br, N), :] #B, NH, Br, D
            L_old = L[:, :, i * Br : min((i + 1) * Br, N)] #B, NH, Br
            M_old = M[:, :, i * Br : min((i + 1) * Br, N)] #B, NH, Br

            Sij = torch.matmul(Qi, torch.transpose(Kj, -2, -1)) / (D ** 0.5)# B, NH, Br, Bc
            mi = Sij.max(dim = -1).values #B, NH, Br
            pij = torch.exp(Sij - mi.unsqueeze(-1))
            li = pij.sum(dim = -1)

            mi_new = torch.maximum(M_old, mi)
            exp_old = torch.exp(M_old - mi_new)
            exp_new = torch.exp(mi - mi_new)
            li_new = exp_old * L_old + exp_new * li
            O[:, :, i * Br : min((i + 1) * Br, N), :] = (exp_old.unsqueeze(-1) * L_old.unsqueeze(-1) * O_old + exp_new.unsqueeze(-1) * torch.matmul(pij, Vj))/li_new.unsqueeze(-1)
            L[:, :, i * Br : min((i + 1) * Br, N)] = li_new
            M[:, :, i * Br : min((i + 1) * Br, N)] = mi_new

    return O



def run_part_b(tensors: dict):
    print("=" * 60)
    print("PART B: Flash attention")
    print("=" * 60)

    runtimes = []
    
    for N in SEQ_LENS:
        Q, K, V = tensors[N]

        # Warmup
        torch.cuda.synchronize()
        for _ in range(WARMUP_ITERS):
            _ = flaash_attention(Q, K, V)
        torch.cuda.synchronize()

        # Timing
        start = time.perf_counter()
        for _ in range(TIMING_ITERS):
            out = flaash_attention(Q, K, V)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000 / TIMING_ITERS

        runtimes.append(elapsed_ms)

        # Save output
        out_path = os.path.join(B_DIR, f"N{N}_output.npy")
        np.save(out_path, out.cpu().float().numpy())

        print(f"  N={N:5d}  runtime={elapsed_ms:8.2f} ms  output saved → {out_path}")

    # Save timings
    np.save(os.path.join(B_DIR, "runtimes.npy"), np.array(runtimes))
    print()
    return runtimes


# ══════════════════════════════════════════════════════════════════════════════
# PART C – PyTorch built-in fused attention
# ══════════════════════════════════════════════════════════════════════════════

def run_part_c(tensors: dict):
    """
    Uses torch.nn.functional.scaled_dot_product_attention which dispatches to
    FlashAttention (or an equivalent fused kernel) when available.
    This serves as both the performance upper-bound AND the correctness reference.
    """
    print("=" * 60)
    print("PART C: PyTorch built-in fused attention (reference)")
    print("=" * 60)

    runtimes = []

    for N in SEQ_LENS:
        Q, K, V = tensors[N]

        # Warmup
        torch.cuda.synchronize()
        for _ in range(WARMUP_ITERS):
            _ = F.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()

        # Timing
        start = time.perf_counter()
        for _ in range(TIMING_ITERS):
            out = F.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000 / TIMING_ITERS

        runtimes.append(elapsed_ms)

        # Save output (this is your correctness reference)
        out_path = os.path.join(C_DIR, f"N{N}_output.npy")
        np.save(out_path, out.cpu().float().numpy())

        print(f"  N={N:5d}  runtime={elapsed_ms:8.2f} ms  output saved → {out_path}")

    np.save(os.path.join(C_DIR, "runtimes.npy"), np.array(runtimes))
    print()
    return runtimes


# ══════════════════════════════════════════════════════════════════════════════
# CORRECTNESS CHECK – compare Part A against Part C reference
# ══════════════════════════════════════════════════════════════════════════════

def check_correctness_a_vs_c():
    print("=" * 60)
    print("CORRECTNESS: Part A vs Part C (reference)")
    print("=" * 60)

    all_pass = True
    for N in SEQ_LENS:
        Q, K, V = load_qkv(N)
        
        # Recompute in float32 for a fair numerical comparison
        Q32, K32, V32 = Q.float(), K.float(), V.float()
        
        out_a = standard_attention(Q32, K32, V32)
        torch.cuda.synchronize()
        out_c = F.scaled_dot_product_attention(Q32, K32, V32)
        torch.cuda.synchronize()

        max_diff  = (out_a - out_c).abs().max().item()
        mean_diff = (out_a - out_c).abs().mean().item()

        passed = max_diff < 1e-3
        all_pass = all_pass and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  N={N:5d}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  {status}")

    print(f"\n  Overall: {'ALL PASSED ✓' if all_pass else 'SOME FAILED ✗'}")
    print()

def check_correctness_b_vs_c():
    print("=" * 60)
    print("CORRECTNESS: Part B vs Part C (reference)")
    print("=" * 60)

    all_pass = True
    for N in SEQ_LENS:
        Q, K, V = load_qkv(N)
        
        # Recompute in float32 for a fair numerical comparison
        Q32, K32, V32 = Q.float(), K.float(), V.float()
        
        out_b = flaash_attention(Q32, K32, V32)
        torch.cuda.synchronize()
        out_c = F.scaled_dot_product_attention(Q32, K32, V32)
        torch.cuda.synchronize()

        max_diff  = (out_b - out_c).abs().max().item()
        mean_diff = (out_b - out_c).abs().mean().item()

        passed = max_diff < 1e-3
        all_pass = all_pass and passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  N={N:5d}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  {status}")

    print(f"\n  Overall: {'ALL PASSED ✓' if all_pass else 'SOME FAILED ✗'}")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING – runtime vs N (Parts A and C together)
# ══════════════════════════════════════════════════════════════════════════════

def plot_runtimes(rt_a, rt_c):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(SEQ_LENS, rt_a, marker="o", label="Part A – Standard attention")
    ax.plot(SEQ_LENS, rt_c, marker="s", label="Part C – PyTorch fused (FlashAttn)")

    ax.set_xlabel("Sequence length N", fontsize=12)
    ax.set_ylabel("Runtime (ms)", fontsize=12)
    ax.set_title("Attention Runtime vs Sequence Length\n(B=8, NH=16, D=64, float16, CUDA)", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(SEQ_LENS)
    ax.set_xticklabels(SEQ_LENS, rotation=45)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(BASE, "runtime_A_vs_C.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved → {plot_path}")
    plt.show()

def plot_runtimes2(rt_a, rt_b, rt_c):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(SEQ_LENS, rt_a, marker="o", label="Part A – Standard attention")
    ax.plot(SEQ_LENS, rt_b, marker="X", label="Part A – Flash attention")
    ax.plot(SEQ_LENS, rt_c, marker="s", label="Part C – PyTorch fused (FlashAttn)")

    ax.set_xlabel("Sequence length N", fontsize=12)
    ax.set_ylabel("Runtime (ms)", fontsize=12)
    ax.set_title("Attention Runtime vs Sequence Length\n(B=8, NH=16, D=64, float16, CUDA)", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(SEQ_LENS)
    ax.set_xticklabels(SEQ_LENS, rotation=45)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(BASE, "runtime_A_vs_C.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved → {plot_path}")
    plt.show()


def plot_runtime_1(rt_c):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(SEQ_LENS, rt_c, marker="s", label="Part C – PyTorch fused (FlashAttn)")

    ax.set_xlabel("Sequence length N", fontsize=12)
    ax.set_ylabel("Runtime (ms)", fontsize=12)
    ax.set_title("Part C Runtime vs Sequence Length\n(B=8, NH=16, D=64, float16, CUDA)", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(SEQ_LENS)
    ax.set_xticklabels(SEQ_LENS, rotation=45)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(BASE, "runtime_C.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved → {plot_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required!"
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # 1. Generate & save QKV (or reload if already saved)
    tensors = generate_and_save_qkv()

    # run this once to export QKV as raw binary for C++ to read
    for N in SEQ_LENS:
        Q, K, V = load_qkv(N)
        Q.cpu().float().numpy().tofile(f"N{N}_Q.bin")
        K.cpu().float().numpy().tofile(f"N{N}_K.bin")
        V.cpu().float().numpy().tofile(f"N{N}_V.bin")
    # print("="*60)
    # print("Loading Q, K, V from Disk")
    # tensors = {i : load_qkv(i) for i in SEQ_LENS}
    # print("="*60)


    # t = torch.tensor([D])
    # s = torch.sqrt(t)
    # t1 = torch.tensor([1/D])
    # s1 = torch.sqrt(t1)[0]

    # print(t, s[0], 1/s[0])
    # print(t1, s1)
    # 2. Run Part A
    rt_a = run_part_a(tensors)

    # 3. Run Part C (reference)
    rt_c = run_part_c(tensors)

    rt_b = run_part_b(tensors)

    # 4. Correctness check
    check_correctness_a_vs_c()
    check_correctness_b_vs_c()

    # 5. Plot
    plot_runtimes2(rt_a, rt_b, rt_c)