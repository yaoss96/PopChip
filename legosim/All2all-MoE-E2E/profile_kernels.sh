#!/usr/bin/env bash
# profile_kernels.sh -- Section-5 cost-model calibration harness.
#
# Runs the ISOLATED expert MLP kernel (moe_gpu_kernel_runner) standalone under
# each GPGPU-Sim SM config, with NO inter-chiplet traffic, and reads back
# gpu_tot_sim_cycle. This is exactly the doc's
#   c_{v,GPU} = gpu_tot_sim_cycle from a single-chiplet run of the isolated kernel
# extraction. Emits a CSV (type,tokens,hidden,inter,repeat,cycles) on stdout so
# a Python calibrator can turn it into measured params.json coefficients.
#
# Usage:
#   profile_kernels.sh <hidden> <inter> <repeat> "<token_list>" [out_csv]
#     token_list : space-separated compute_tokens values to sweep
#   env GPU_TYPES="GPU-L GPU-S" selects which library types to profile.
set -euo pipefail
shopt -s nullglob

HIDDEN="${1:-128}"
INTER="${2:-64}"
REPEAT="${3:-2}"
TOKEN_LIST="${4:-1 2 4 6 8 10 16 18 24 32}"
OUT_CSV="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIMULATOR_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SIMULATOR_ROOT"
export BENCHMARK_ROOT="$SIMULATOR_ROOT/benchmark/All2all-MoE-E2E"
export CUDA_INSTALL_PATH=/usr/local/cuda-11.5
export PATH="$CUDA_INSTALL_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$SIMULATOR_ROOT/.local_deps/lib:$SIMULATOR_ROOT/gpgpu-sim/lib/$GPGPUSIM_CONFIG:$CUDA_INSTALL_PATH/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CUOBJDUMP_SIM_FILE=1

RUNNER="$BENCHMARK_ROOT/build/bin/moe_gpu_kernel_runner"
[ -x "$RUNNER" ] || { echo "ERROR: runner not built at $RUNNER" >&2; exit 2; }

# Map library type -> GPGPU-Sim tested-cfg dir (matches codegen.py GPU_CFG).
declare -A CFG
CFG["GPU-L"]="$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM7_TITANV"
CFG["GPU-S"]="$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM6_TITANX"

TYPES="${GPU_TYPES:-GPU-L GPU-S}"

emit() { if [ -n "$OUT_CSV" ]; then echo "$1" >> "$OUT_CSV"; fi; echo "$1"; }

[ -n "$OUT_CSV" ] && : > "$OUT_CSV"
emit "type,tokens,hidden,inter,repeat,cycles"

WD="$(mktemp -d /tmp/legosim_profile.XXXXXX)"
trap 'rm -rf "$WD"' EXIT

for t in $TYPES; do
  cfgdir="${CFG[$t]:-}"
  [ -d "$cfgdir" ] || { echo "WARN: no cfg dir for $t ($cfgdir)" >&2; continue; }
  for tok in $TOKEN_LIST; do
    run="$WD/${t}_${tok}"; rm -rf "$run"; mkdir -p "$run"; cd "$run"
    cp -f "$cfgdir"/* . 2>/dev/null
    cyc=$("$RUNNER" "$tok" "$HIDDEN" "$INTER" "$REPEAT" 2>/dev/null \
          | grep -oE 'gpu_tot_sim_cycle *= *[0-9]+' | grep -oE '[0-9]+' | tail -1)
    [ -z "$cyc" ] && cyc="NA"
    emit "$t,$tok,$HIDDEN,$INTER,$REPEAT,$cyc"
    cd "$SIMULATOR_ROOT"
  done
done
