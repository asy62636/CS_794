"""
PyTorch attention baselines for the FlashAttention benchmark (CS 794, PA4 follow-up).

For every sequence length N it measures, on shapes [B=8, NH=16, N, D=64]:
  - naive : PA4 Part A, softmax(Q K^T / sqrt(D)) V with the full N x N matrix materialised
  - sdpa  : torch.nn.functional.scaled_dot_product_attention (fused kernel; FlashAttention-2
            backend for fp16, memory-efficient backend for fp32 when it can be forced)
in fp16 and fp32, recording average/min runtime (CUDA events, >= 10 timed iterations after
warm-up), peak extra GPU memory beyond the Q/K/V/O tensors, and out-of-memory failures.

Usage: python3 torch_attention_bench.py --out ../results/flash_attention_torch.csv
"""

import argparse
import contextlib
import csv
import math

import torch
import torch.nn.functional as F

B, NH, D = 8, 16, 64
SEQ_LENS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
WARMUP, ITERS = 2, 10


def naive_attention(q, k, v):
    s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


def sdpa_context(dtype):
    """Force a specific fused backend if this PyTorch version allows it; report its name."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if dtype == torch.float16:
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION), "sdpa_flash"
        return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION), "sdpa_mem_efficient"
    except Exception:  # older PyTorch: let it pick
        return contextlib.nullcontext(), "sdpa_auto"


def bench(fn, q, k, v):
    for _ in range(WARMUP):
        fn(q, k, v)
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn(q, k, v)
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))
    return sum(times) / len(times), min(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="flash_attention_torch.csv")
    ap.add_argument("--n", default=",".join(map(str, SEQ_LENS)))
    args = ap.parse_args()
    seq_lens = [int(x) for x in args.n.split(",")]

    dev = torch.cuda.get_device_properties(0)
    print(f"GPU: {dev.name} | {dev.total_memory / 2**20:.0f} MB | torch {torch.__version__} "
          f"| CUDA {torch.version.cuda}")

    rows = []
    for dtype in (torch.float16, torch.float32):
        dname = "fp16" if dtype == torch.float16 else "fp32"
        for N in seq_lens:
            torch.manual_seed(0)
            q, k, v = (torch.randn(B, NH, N, D, device="cuda", dtype=dtype) for _ in range(3))
            ref = None
            ctx, sdpa_name = sdpa_context(dtype)
            impls = [("naive", naive_attention, contextlib.nullcontext()),
                     (sdpa_name, F.scaled_dot_product_attention, ctx)]
            for name, fn, context in impls:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                status, avg, best, max_diff, extra_mb = "OK", float("nan"), float("nan"), float("nan"), float("nan")
                try:
                    with context:
                        avg, best = bench(fn, q, k, v)
                        out = fn(q, k, v)
                    torch.cuda.synchronize()
                    out_mb = out.numel() * out.element_size() / 2**20
                    extra_mb = (torch.cuda.max_memory_allocated() - base) / 2**20 - out_mb
                    if ref is None:
                        ref = out.float()
                    else:
                        max_diff = (out.float() - ref).abs().max().item()
                    del out
                except torch.cuda.OutOfMemoryError:
                    status = "OOM"
                except RuntimeError as exc:  # e.g. backend not available for this dtype
                    status = "ERROR:" + str(exc).splitlines()[0][:80].replace(",", ";")
                torch.cuda.empty_cache()
                flops = 4.0 * B * NH * N * N * D
                tflops = flops / (avg * 1e-3) / 1e12 if status == "OK" else float("nan")
                print(f"{dname} N={N:5d} {name:<18} avg {avg:10.3f} ms  min {best:10.3f} ms  "
                      f"{tflops:7.3f} TFLOPS  extra {extra_mb:9.1f} MB  diff {max_diff:.2e}  {status}",
                      flush=True)
                rows.append([name, dname, N, ITERS, f"{avg:.4f}", f"{best:.4f}", f"{tflops:.4f}",
                             f"{extra_mb:.1f}", f"{max_diff:.3e}", status])
            del q, k, v, ref
            torch.cuda.empty_cache()

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["impl", "dtype", "N", "iters", "avg_ms", "min_ms", "tflops",
                    "peak_extra_MB", "max_diff_vs_naive", "status"])
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
