#!/usr/bin/env bash
# Optional: Nsight Compute metrics for every kernel (run ./run_all.sh first so build/ exists).
# GPU performance counters usually need root on Linux:
#   sudo -E env "PATH=$PATH" ./profile_ncu.sh && sudo chown -R "$USER" results
set -u
cd "$(dirname "$0")"
RES=results
BUILD=build
NCU="${NCU:-$(command -v ncu || echo /usr/local/cuda/bin/ncu)}"
mkdir -p "$RES"

# occupancy, throughput, register/shared-memory footprint, bank conflicts, and global-load
# coalescing (sectors per request: 4 is ideal for 32 consecutive fp32 loads, 32 is worst case)
METRICS="gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
launch__occupancy_limit_registers,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum"

run() {  # run <name> <kernel regex> <binary + args...>
    local name=$1 regex=$2
    shift 2
    echo "=== ncu: $name ==="
    "$NCU" --metrics "$METRICS" --csv --log-file "$RES/ncu_$name.csv" -k "regex:$regex" "$@"
    if grep -q "ERR_NVGPUCTRPERM" "$RES/ncu_$name.csv"; then
        echo "No permission for GPU counters. Re-run with:"
        echo "  sudo -E env \"PATH=\$PATH\" ./profile_ncu.sh && sudo chown -R \"\$USER\" results"
        exit 1
    fi
    echo "wrote $RES/ncu_$name.csv"
}

run flash_attention "flash_v" "$BUILD/flash_attn_bench" --profile 1024
run gemm "MatMul|Tiling|sgemm|gemm" "$BUILD/gemm_bench" --profile 4096
echo "Done. git add perf/results && git commit -m \"ncu metrics\" && git push"
