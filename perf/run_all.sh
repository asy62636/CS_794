#!/usr/bin/env bash
# Builds and runs the CS 794 performance benchmarks; everything is written to perf/results/.
#
#   ./run_all.sh          # FlashAttention (CUDA + PyTorch) and GEMM
#   ./run_all.sh flash    # only FlashAttention
#   ./run_all.sh gemm     # only GEMM
#
# Before running on a laptop: plug in the charger and select the high-performance power mode,
# otherwise the GPU clocks (and every number below) drop.
set -u -o pipefail
cd "$(dirname "$0")"

WHAT="${1:-all}"
RES=results
BUILD=build
mkdir -p "$RES" "$BUILD"
LOG="$RES/run_log_${WHAT}.txt"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

FAILED=()

echo "=== $(date) | target: $WHAT ==="
echo "=== environment ==="
if ! nvidia-smi; then echo "nvidia-smi failed: is the NVIDIA driver loaded?"; exit 1; fi
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,clocks.max.sm,clocks.max.mem,power.limit,pstate,temperature.gpu \
    --format=csv | tee "$RES/gpu_info.csv"
if ! command -v nvcc >/dev/null; then
    echo "nvcc not found. Try: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi
nvcc --version
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())" \
    || echo "WARNING: PyTorch with CUDA not importable; the PyTorch baselines will fail"

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
ARCH="${ARCH:-sm_${CC:-89}}"
FLAGS="-O3 -std=c++17 -arch=$ARCH -lineinfo -Xptxas -v"
echo "=== compiling for $ARCH ==="

build() {  # build <name> <source> [extra flags...]; ptxas register/spill report saved per target
    local name=$1 src=$2
    shift 2
    if nvcc $FLAGS "$src" "$@" -o "$BUILD/$name" 2> "$RES/ptxas_$name.txt"; then
        grep -E "Compiling entry|registers|spill" "$RES/ptxas_$name.txt" | sed 's/^ptxas info *: //'
        echo "built $name"
    else
        cat "$RES/ptxas_$name.txt"
        echo "BUILD FAILED: $name"
        FAILED+=("build:$name")
        return 1
    fi
}

to_csv() {  # to_csv <raw log> <kind> <csv out>: keep the "CSV,<kind>,..." rows with a header
    grep '^CSV,header,' "$1" | head -1 | cut -d, -f3- > "$3"
    grep "^CSV,$2," "$1" | cut -d, -f3- >> "$3"
}

if [[ "$WHAT" == "all" || "$WHAT" == "flash" ]]; then
    if build flash_attn_bench flash_attention/flash_attn_bench.cu; then
        echo "=== FlashAttention: CUDA kernels v1/v2/v3/v3h (takes a few minutes) ==="
        if "./$BUILD/flash_attn_bench" | tee "$RES/flash_attention_cuda.txt"; then
            to_csv "$RES/flash_attention_cuda.txt" flash "$RES/flash_attention_cuda.csv"
        else
            FAILED+=("run:flash_attn_bench")
        fi
    fi
    echo "=== FlashAttention: PyTorch naive + SDPA baselines ==="
    python3 flash_attention/torch_attention_bench.py --out "$RES/flash_attention_torch.csv" \
        | tee "$RES/flash_attention_torch.txt" || FAILED+=("run:torch_attention_bench")
fi

if [[ "$WHAT" == "all" || "$WHAT" == "gemm" ]]; then
    if build gemm_bench gemm/gemm_bench.cu -lcublas; then
        echo "=== GEMM ladder (takes a few minutes) ==="
        if "./$BUILD/gemm_bench" | tee "$RES/gemm.txt"; then
            to_csv "$RES/gemm.txt" gemm "$RES/gemm.csv"
        else
            FAILED+=("run:gemm_bench")
        fi
    fi
fi

echo "=== clocks / throttling after the run ==="
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv | tee "$RES/gpu_state_after_${WHAT}.csv"

echo
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "ALL DONE. Results are in perf/results/. To send them back:"
else
    echo "FINISHED WITH FAILURES: ${FAILED[*]}  (the log above has the details). Send them back anyway:"
fi
echo "  git add perf/results && git commit -m \"Benchmark results ($WHAT)\" && git push"
