#!/usr/bin/env bash
# Run the FULL interchiplet end-to-end on an ILP-generated layout (Task 4).
#
# Unlike run_baremetal_e2e.sh (which uses a stock hand-written 4-GPU smoke yml),
# this consumes the artifacts emitted by src/codegen.py for an arbitrary ILP
# solution: bench.yml (baremetal router + N GPGPU-Sim workers + popnet),
# moe_coords.txt (the ILP (x,y) placement, read via $MOE_COORD_FILE), and the
# flattened mesh topology.  The ILP layout therefore actually drives the NoC.
#
# Usage:  run_ilp_e2e.sh <codegen_out_dir> [run_root]
#   codegen_out_dir : dir containing bench.yml, moe_coords.txt, topology/*.gv
#                     (e.g. <repo>/out/legosim_mixtral)
#   run_root        : new/empty per-run output directory. If omitted, RUN_ROOT
#                     is used; otherwise a unique /tmp directory is selected.
#
# Router runs BARE-METAL (no Sniper): the MoE gate is a lightweight control-flow
# process off the expert/NoC makespan critical path; this is documented honestly
# in the design doc. Experts use real GPGPU-Sim; NoC uses real popnet.
set -euo pipefail
shopt -s nullglob

die() {
  echo "ERROR: $*" >&2
  exit 2
}

coord_scalar() {
  local key="$1"
  local file="$2"
  awk -v wanted="$key" '
    $1 == wanted {
      gsub(/\r/, "", $2)
      print $2
      exit
    }
  ' "$file"
}

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum
  )
}

record_git_revision() {
  local label="$1"
  local repo="$2"
  local sha dirty
  if sha="$(git -C "$repo" rev-parse HEAD 2>/dev/null)"; then
    if git -C "$repo" diff --quiet --ignore-submodules -- 2>/dev/null &&
       git -C "$repo" diff --cached --quiet --ignore-submodules -- 2>/dev/null; then
      dirty=false
    else
      dirty=true
    fi
    printf '%s\t%s\tdirty=%s\t%s\n' "$label" "$sha" "$dirty" "$repo"
  else
    printf '%s\tNOT_A_GIT_REPOSITORY\t%s\n' "$label" "$repo"
  fi
}

[ "$#" -ge 1 ] && [ "$#" -le 2 ] ||
  die "usage: run_ilp_e2e.sh <codegen_out_dir> [run_root]"

CG_DIR="$(realpath -e -- "$1")"
[ -f "$CG_DIR/bench.yml" ]      || die "no bench.yml in $CG_DIR"
[ -f "$CG_DIR/moe_coords.txt" ] || die "no moe_coords.txt in $CG_DIR"

if [ "$#" -eq 2 ] && [ -n "${RUN_ROOT:-}" ]; then
  ARG_RUN="$(realpath -m -- "$2")"
  ENV_RUN="$(realpath -m -- "$RUN_ROOT")"
  [ "$ARG_RUN" = "$ENV_RUN" ] ||
    die "conflicting run roots: argument=$ARG_RUN RUN_ROOT=$ENV_RUN"
fi
RUN_REQUEST="${2:-${RUN_ROOT:-/tmp/legosim_ilp_e2e_$(date -u +%Y%m%dT%H%M%SZ)_$$}}"
RUN="$(realpath -m -- "$RUN_REQUEST")"

# Never erase a previous run. Requiring a new/empty directory makes accidental
# cross-run reuse and provenance loss impossible.
if [ -d "$RUN" ] && [ -n "$(find "$RUN" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  die "RUN_ROOT already exists and is not empty: $RUN"
fi
[ ! -e "$RUN" ] || [ -d "$RUN" ] || die "RUN_ROOT exists and is not a directory: $RUN"
mkdir -p "$RUN"

# Publish a complete, immutable evidence bundle even when a later validation
# step fails.  RUN itself stays on native Linux storage because InterChiplet
# uses FIFOs; RESULT_MIRROR_DIR may be on a persistent mounted filesystem.
mirror_run_evidence() {
  local runner_rc=$?
  local mirror_tmp manifest_tmp artifact mirror_failed=0
  trap - EXIT
  if [ -n "${RESULT_MIRROR_DIR:-}" ] && [ -d "${RUN:-}" ]; then
    set +e
    mirror_tmp="${RESULT_MIRROR_DIR}.tmp.$$"
    manifest_tmp="${mirror_tmp}.manifest.$$"
    if [ -e "$RESULT_MIRROR_DIR" ] || [ -e "$mirror_tmp" ]; then
      echo "ERROR: refusing to overwrite result mirror: $RESULT_MIRROR_DIR" >&2
      mirror_failed=1
    else
      mkdir -p "$(dirname "$RESULT_MIRROR_DIR")" &&
        mkdir "$mirror_tmp" || mirror_failed=1
      for artifact in \
        ilp_e2e_summary.txt \
        run_manifest.txt \
        command.txt \
        git_revisions.txt \
        source_input_manifest.sha256 \
        staged_input_manifest.sha256 \
        simulator_binaries.sha256 \
        simulator_binaries.end.sha256 \
        simulator_binaries.diff \
        harness_sources.sha256 \
        raw_output_manifest.sha256 \
        gpu_cycles.tsv \
        expected_dispatch_matrix.log \
        effective_dispatch_matrix.log \
        dispatch_matrix.diff \
        bench.txt \
        delayInfo.txt \
        interchiplet.stdout \
        interchiplet.stderr
      do
        if [ -f "$RUN/$artifact" ]; then
          cp -- "$RUN/$artifact" "$mirror_tmp/$artifact" ||
            mirror_failed=1
        fi
      done
      printf 'runner_exit_code=%s\n' "$runner_rc" \
        > "$mirror_tmp/runner_exit_code.txt" || mirror_failed=1
      tar -C "$RUN" -czf "$mirror_tmp/run_evidence.tar.gz" . ||
        mirror_failed=1
      hash_tree "$mirror_tmp" > "$manifest_tmp" || mirror_failed=1
      mv -- "$manifest_tmp" "$mirror_tmp/mirror_manifest.sha256" ||
        mirror_failed=1
      if [ "$mirror_failed" -eq 0 ]; then
        printf 'complete\n' > "$mirror_tmp/MIRROR_COMPLETE" ||
          mirror_failed=1
      fi
      if [ "$mirror_failed" -eq 0 ]; then
        mv -- "$mirror_tmp" "$RESULT_MIRROR_DIR" ||
          mirror_failed=1
      fi
    fi
    if [ "$mirror_failed" -ne 0 ] && [ "$runner_rc" -eq 0 ]; then
      runner_rc=2
    fi
  fi
  exit "$runner_rc"
}
trap mirror_run_evidence EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIMULATOR_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SIMULATOR_ROOT"
export BENCHMARK_ROOT="$SIMULATOR_ROOT/benchmark/All2all-MoE-E2E"
BENCHMARK_SOURCE_ROOT="$BENCHMARK_ROOT"
# Keep the scale-out ABI build isolated from the legacy build tree (whose
# CMake cache may refer to a different checkout).  Callers may override this
# with an absolute MOE_BUILD_ROOT, but a formal run refuses a missing binary.
BUILD_ROOT="${MOE_BUILD_ROOT:-$BENCHMARK_SOURCE_ROOT/build_scale128}"
[ -x "$BUILD_ROOT/bin/moe_router_cpu" ] || die "missing scale-out router binary: $BUILD_ROOT/bin/moe_router_cpu"
[ -x "$BUILD_ROOT/bin/moe_gpu_worker" ] || die "missing scale-out GPU worker binary: $BUILD_ROOT/bin/moe_gpu_worker"
[ -x "$BUILD_ROOT/bin/moe_gpu_kernel_runner" ] || die "missing GPGPU-Sim kernel runner: $BUILD_ROOT/bin/moe_gpu_kernel_runner"
export CUDA_INSTALL_PATH=/usr/local/cuda-11.5
export GPGPUSIM_CONFIG="${GPGPUSIM_CONFIG:-gcc-11.5.0/cuda-11050/release}"
export PATH="$CUDA_INSTALL_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$SIMULATOR_ROOT/gpgpu-sim/lib/$GPGPUSIM_CONFIG:$CUDA_INSTALL_PATH/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CUOBJDUMP_SIM_FILE=1
export MOE_GPGPUSIM_MAX_PARALLEL="${MOE_GPGPUSIM_MAX_PARALLEL:-16}"
case "$MOE_GPGPUSIM_MAX_PARALLEL" in
  ''|*[!0-9]*) die "MOE_GPGPUSIM_MAX_PARALLEL must be an integer" ;;
esac
[ "$MOE_GPGPUSIM_MAX_PARALLEL" -ge 1 ] &&
[ "$MOE_GPGPUSIM_MAX_PARALLEL" -le 128 ] ||
  die "MOE_GPGPUSIM_MAX_PARALLEL must be in [1,128]"

# Every run gets an immutable private copy of all simulation inputs.
# InterChiplet defines $BENCHMARK_ROOT as the directory containing the staged
# YAML (it does not expand arbitrary environment variables inside YAML), so
# expose the shared, immutable build through a symlink inside that private
# staging tree.
STAGE="$RUN/input"
mkdir -p "$STAGE/topology"
TOPOLOGY_FILES=("$CG_DIR"/topology/*.gv)
[ "${#TOPOLOGY_FILES[@]}" -gt 0 ] || die "no topology/*.gv in $CG_DIR"
cp -- "$CG_DIR/moe_coords.txt" "$STAGE/moe_coords.txt"
cp -- "${TOPOLOGY_FILES[@]}" "$STAGE/topology/"
for optional_input in \
  params.input.json solution.input.json coords.json linkDelay.txt \
  layout_report.txt dispatch_meta.json; do
  if [ -f "$CG_DIR/$optional_input" ]; then
    cp -- "$CG_DIR/$optional_input" "$STAGE/$optional_input"
  fi
done

# Normalize both the short-lived MOE_INPUT_ROOT experiment and older generated
# coordinate comments. Topology must resolve through InterChiplet's built-in
# staged $BENCHMARK_ROOT.
YML="$STAGE/bench.yml"
sed \
  -e 's|[$]MOE_INPUT_ROOT/topology/|$BENCHMARK_ROOT/topology/|g' \
  -e 's|[$]MOE_INPUT_ROOT/moe_coords[.]txt|$MOE_COORD_FILE|g' \
  -e 's|[$]BENCHMARK_ROOT/moe_coords[.]txt|$MOE_COORD_FILE|g' \
  "$CG_DIR/bench.yml" > "$YML"

ln -s "$BUILD_ROOT" "$STAGE/build"
export MOE_COORD_FILE="$STAGE/moe_coords.txt"

# A formal run must consume an explicit measured/recorded dispatch. Synthetic
# Zipf routing is available only when the caller opts in for a smoke test.
if [ -f "$CG_DIR/dispatch_matrix.txt" ]; then
  cp -- "$CG_DIR/dispatch_matrix.txt" "$STAGE/dispatch_matrix.txt"
  export MOE_DISPATCH_FILE="$STAGE/dispatch_matrix.txt"
  DISPATCH_MODE=real_dispatch_file
elif [ "${ALLOW_SYNTHETIC_DISPATCH:-0}" = "1" ]; then
  unset MOE_DISPATCH_FILE
  DISPATCH_MODE=zipf_synthetic
else
  die "no dispatch_matrix.txt in $CG_DIR (formal runs require it; set ALLOW_SYNTHETIC_DISPATCH=1 only for smoke tests)"
fi

# Derive mesh width (== -w) from the coord file (width = L*Y = mesh cols).
WIDTH="$(coord_scalar width "$MOE_COORD_FILE")"
NGPU="$(coord_scalar ngpu "$MOE_COORD_FILE")"
case "$WIDTH" in ''|*[!0-9]*) die "missing/non-integer width in $MOE_COORD_FILE: '$WIDTH'" ;; esac
case "$NGPU" in ''|*[!0-9]*) die "missing/non-integer ngpu in $MOE_COORD_FILE: '$NGPU'" ;; esac
[ "$WIDTH" -gt 0 ] || die "width must be positive: $WIDTH"
[ "$NGPU" -gt 0 ] || die "ngpu must be positive: $NGPU"
EXPECTED_GPU_INIT_BYTES="$(( (4 + 2 * NGPU) * 4 ))"
GPU_COORD_COUNT="$(awk '$1 == "gpu" { n++ } END { print n + 0 }' "$MOE_COORD_FILE")"
[ "$GPU_COORD_COUNT" -eq "$NGPU" ] ||
  die "moe_coords.txt declares ngpu=$NGPU but contains $GPU_COORD_COUNT gpu records"

if [ "$DISPATCH_MODE" = "real_dispatch_file" ]; then
  DISPATCH_NGPU="$(coord_scalar ngpu "$MOE_DISPATCH_FILE")"
  case "$DISPATCH_NGPU" in
    ''|*[!0-9]*) die "missing/non-integer ngpu in $MOE_DISPATCH_FILE: '$DISPATCH_NGPU'" ;;
  esac
  [ "$DISPATCH_NGPU" -eq "$NGPU" ] ||
    die "dispatch ngpu=$DISPATCH_NGPU does not match coordinate ngpu=$NGPU"
  awk '$1 == "round" { found=1; exit } END { exit(found ? 0 : 1) }' \
    "$MOE_DISPATCH_FILE" ||
    die "dispatch file has no round header: $MOE_DISPATCH_FILE"
  SIM_TOKENS_PER_SOURCE="$(coord_scalar tokens_per_source "$MOE_DISPATCH_FILE")"
  SIM_TOPK="$(coord_scalar topk "$MOE_DISPATCH_FILE")"
  SIM_HIDDEN="$(coord_scalar sim_hidden "$MOE_DISPATCH_FILE")"
  SIM_INTERMEDIATE="$(coord_scalar sim_intermediate "$MOE_DISPATCH_FILE")"
  SIM_FFN_REPEAT="$(coord_scalar sim_ffn_repeat "$MOE_DISPATCH_FILE")"
  for sim_value in \
    "$SIM_TOKENS_PER_SOURCE" "$SIM_TOPK" "$SIM_HIDDEN" \
    "$SIM_INTERMEDIATE" "$SIM_FFN_REPEAT"; do
    case "$sim_value" in
      ''|*[!0-9]*) die "dispatch simulation parameters must be positive integers" ;;
    esac
    [ "$sim_value" -gt 0 ] ||
      die "dispatch simulation parameters must be positive integers"
  done
  awk -v n="$NGPU" -v expected="$((SIM_TOKENS_PER_SOURCE * SIM_TOPK))" '
    $1 == "round" { reading=1; next }
    reading && $0 !~ /^[[:space:]]*#/ && $0 !~ /^[[:space:]]*$/ {
      gsub(/\r/, "", $0)
      if (NF != n) {
        printf "dispatch row %d has %d columns, expected %d\n", rows, NF, n > "/dev/stderr"
        exit 1
      }
      sum=0
      for (i=1; i<=NF; ++i) {
        if ($i !~ /^[0-9]+$/) {
          printf "dispatch row %d column %d is not a non-negative integer: %s\n", rows, i, $i > "/dev/stderr"
          exit 1
        }
        sum += $i
        columns[i] += $i
      }
      if (sum != expected) {
        printf "dispatch row %d sums to %d, expected %d\n", rows, sum, expected > "/dev/stderr"
        exit 1
      }
      rows++
      if (rows == n) exit
    }
    END {
      if (rows != n) {
        printf "dispatch has %d matrix rows, expected %d\n", rows, n > "/dev/stderr"
        exit 1
      }
      for (i=1; i<=n; ++i) {
        if (columns[i] <= 0) {
          printf "dispatch column %d has zero global token assignments\n", i-1 > "/dev/stderr"
          exit 1
        }
      }
    }
  ' "$MOE_DISPATCH_FILE" ||
    die "dispatch matrix shape/value validation failed"
else
  SIM_TOKENS_PER_SOURCE=
  SIM_TOPK=
  SIM_HIDDEN=
  SIM_INTERMEDIATE=
  SIM_FFN_REPEAT=
fi

# Provenance is written before launch so even a timed-out run remains auditable.
hash_tree "$CG_DIR" > "$RUN/source_input_manifest.sha256"
hash_tree "$STAGE" > "$RUN/staged_input_manifest.sha256"
SOURCE_MANIFEST_SHA="$(sha256sum "$RUN/source_input_manifest.sha256" | awk '{print $1}')"
STAGED_MANIFEST_SHA="$(sha256sum "$RUN/staged_input_manifest.sha256" | awk '{print $1}')"

{
  record_git_revision workspace "$(cd "$SIMULATOR_ROOT/../.." && pwd)"
  record_git_revision legosim "$SIMULATOR_ROOT"
  record_git_revision interchiplet "$SIMULATOR_ROOT/interchiplet"
  record_git_revision gpgpu-sim "$SIMULATOR_ROOT/gpgpu-sim"
  record_git_revision popnet "$SIMULATOR_ROOT/popnet_chiplet"
  record_git_revision snipersim "$SIMULATOR_ROOT/snipersim"
} > "$RUN/git_revisions.txt"

hash_simulator_binaries() {
  for binary in \
    "$SIMULATOR_ROOT/interchiplet/bin/interchiplet" \
    "$SIMULATOR_ROOT/popnet_chiplet/build/popnet" \
    "$BUILD_ROOT/bin/moe_router_cpu" \
    "$BUILD_ROOT/bin/moe_gpu_worker" \
    "$BUILD_ROOT/bin/moe_gpu_kernel_runner" \
    "$SIMULATOR_ROOT/gpgpu-sim/lib/$GPGPUSIM_CONFIG/libcudart.so"; do
    [ -f "$binary" ] && sha256sum "$binary"
  done
}
hash_simulator_binaries > "$RUN/simulator_binaries.sha256"
{
  for source in \
    "$SCRIPT_DIR/run_ilp_e2e.sh" \
    "$BENCHMARK_SOURCE_ROOT/moe_router_cpu.cpp" \
    "$BENCHMARK_SOURCE_ROOT/moe_gpu_worker.cu" \
    "$BENCHMARK_SOURCE_ROOT/moe_coords.h" \
    "$BENCHMARK_SOURCE_ROOT/CMakeLists.txt"; do
    [ -f "$source" ] && sha256sum "$source"
  done
} > "$RUN/harness_sources.sha256"

ROUNDS="${MOE_E2E_ROUNDS:-5}"
ERR_RATE="${MOE_E2E_ERR_RATE:-0.005}"
TIMEOUT="${MOE_E2E_TIMEOUT:-1800}s"
IC_CMD=(
  timeout --kill-after=10s "$TIMEOUT"
  "$SIMULATOR_ROOT/interchiplet/bin/interchiplet" "$YML"
  --cwd "$RUN" -w "$WIDTH" -f 4 -t "$ROUNDS" -e "$ERR_RATE"
)
{
  printf 'SIMULATOR_ROOT=%q\n' "$SIMULATOR_ROOT"
  printf 'BENCHMARK_SOURCE_ROOT=%q\n' "$BENCHMARK_SOURCE_ROOT"
  printf 'INTERCHIPLET_BENCHMARK_ROOT=%q\n' "$STAGE"
  printf 'MOE_COORD_FILE=%q\n' "$MOE_COORD_FILE"
  printf 'MOE_DISPATCH_FILE=%q\n' "${MOE_DISPATCH_FILE:-}"
  printf 'LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH"
  printf 'command='
  printf '%q ' "${IC_CMD[@]}"
  printf '\n'
} > "$RUN/command.txt"
{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "uname=$(uname -a)"
  echo "simulator_root=$SIMULATOR_ROOT"
  echo "benchmark_source_root=$BENCHMARK_SOURCE_ROOT"
  echo "build_root=$BUILD_ROOT"
  echo "interchiplet_benchmark_root=$STAGE"
  echo "codegen_source=$CG_DIR"
  echo "run_root=$RUN"
  echo "input_stage=$STAGE"
  echo "gpgpusim_config=$GPGPUSIM_CONFIG"
  echo "width=$WIDTH"
  echo "ngpu=$NGPU"
  echo "rounds_requested=$ROUNDS"
  echo "error_rate=$ERR_RATE"
  echo "dispatch_mode=$DISPATCH_MODE"
  echo "sim_tokens_per_source=${SIM_TOKENS_PER_SOURCE:-}"
  echo "sim_topk=${SIM_TOPK:-}"
  echo "sim_hidden=${SIM_HIDDEN:-}"
  echo "sim_intermediate=${SIM_INTERMEDIATE:-}"
  echo "sim_ffn_repeat=${SIM_FFN_REPEAT:-}"
  echo "gpgpusim_max_parallel=$MOE_GPGPUSIM_MAX_PARALLEL"
  echo "source_input_manifest_sha256=$SOURCE_MANIFEST_SHA"
  echo "staged_input_manifest_sha256=$STAGED_MANIFEST_SHA"
} > "$RUN/run_manifest.txt"

echo "SIMULATOR_ROOT=$SIMULATOR_ROOT"
echo "GPGPUSIM_CONFIG=$GPGPUSIM_CONFIG"
echo "CG_DIR=$CG_DIR"; echo "YML=$YML"; echo "RUN=$RUN"; echo "WIDTH=$WIDTH"
echo "MOE_COORD_FILE=$MOE_COORD_FILE"
echo "MOE_DISPATCH_FILE=${MOE_DISPATCH_FILE:-synthetic-opt-in}"
echo "SOURCE_INPUT_MANIFEST_SHA256=$SOURCE_MANIFEST_SHA"
echo "STAGED_INPUT_MANIFEST_SHA256=$STAGED_MANIFEST_SHA"

if ldd "$SIMULATOR_ROOT/popnet_chiplet/build/popnet" 2>&1 | grep -q "not found"; then
  die "popnet has unresolved shared libraries"
fi

IC_RC=0
"${IC_CMD[@]}" \
  > "$RUN/interchiplet.stdout" 2> "$RUN/interchiplet.stderr" || IC_RC=$?
echo "interchiplet_rc=$IC_RC"

echo "=== bench.txt ==="; wc -l "$RUN"/bench.txt 2>/dev/null || echo "NO bench.txt"
MAKESPAN=""
if [ -f "$RUN/bench.txt" ]; then
  MAKESPAN=$(awk '{for(i=1;i<=2;i++) if($i+0>m)m=$i} END{print m}' "$RUN/bench.txt")
  echo "makespan_cycles=$MAKESPAN"
fi
SYNC_MAKESPAN=$(awk '/Benchmark elapses/ {for(i=1;i<=NF;i++) if($i ~ /^cycle/) v=$(i-1)} END{print v}' "$RUN/interchiplet.stdout")
echo "synchronized_makespan_cycles=${SYNC_MAKESPAN:-missing}"

LAST_R=$(find "$RUN" -maxdepth 1 -type d -name 'proc_r*_p1_t0' -printf '%f\n' 2>/dev/null |
  sed -n 's/^proc_r\([0-9][0-9]*\)_p1_t0$/\1/p' | sort -n | tail -1)
LAST_P2=$(find "$RUN" -maxdepth 1 -type d -name 'proc_r*_p2_t0' -printf '%f\n' 2>/dev/null |
  sed -n 's/^proc_r\([0-9][0-9]*\)_p2_t0$/\1/p' | sort -n | tail -1)
GPU_LOGS=("$RUN"/proc_r"${LAST_R}"_p1_t*/gpgpusim.0.*.log)
POPNET_LOG="$RUN/proc_r${LAST_P2}_p2_t0/popnet_0.log"
ROUTER_LOG=("$RUN"/proc_r"${LAST_R}"_p1_t0/router.*.log)

echo "rounds_phase1=${LAST_R:-0} rounds_phase2=${LAST_P2:-0} n_gpu_workers=${#GPU_LOGS[@]}"
echo "=== popnet delay (latest phase2) ==="
grep -iE "average delay|total finished|total power" "$POPNET_LOG" 2>/dev/null | tail -3 || true
echo "=== gpu cycles (latest phase1) ==="
grep -h "gpu_tot_sim_cycle" "${GPU_LOGS[@]}" /dev/null 2>/dev/null || true
echo "=== router coord banner ==="
grep -h "using ILP layout coords" "${ROUTER_LOG[@]}" /dev/null 2>/dev/null | head -1 || true
if [ "$DISPATCH_MODE" = "real_dispatch_file" ]; then
  awk -v n="$NGPU" '
    BEGIN { row=0 }
    $1 == "round" { reading=1; next }
    reading && $0 !~ /^[[:space:]]*#/ && $0 !~ /^[[:space:]]*$/ {
      gsub(/\r/, "", $0)
      print "src " row " " $0
      row++
      if (row == n) exit
    }
  ' "$MOE_DISPATCH_FILE" > "$RUN/expected_dispatch_matrix.log"
  awk -v n="$NGPU" '
    $0 == "dispatch_matrix" { reading=1; next }
    reading && $1 == "src" {
      print
      rows++
      if (rows == n) exit
    }
  ' "${ROUTER_LOG[@]}" /dev/null > "$RUN/effective_dispatch_matrix.log"
  echo "=== effective dispatch matrix ==="
  cat "$RUN/effective_dispatch_matrix.log"
fi
echo "=== stdout tail ==="; tail -8 "$RUN/interchiplet.stdout"

# Validation
FAIL=0
[ "$IC_RC" -eq 0 ] || { echo "ERROR: interchiplet rc=$IC_RC" >&2; FAIL=1; }
grep -q '\[error\] Simulation process' "$RUN/interchiplet.stdout" && { echo "ERROR: a simulator subprocess failed" >&2; FAIL=1; }
[ -s "$RUN/bench.txt" ] || { echo "ERROR: empty/missing bench.txt" >&2; FAIL=1; }
[ -s "$RUN/delayInfo.txt" ] || { echo "ERROR: missing delayInfo.txt" >&2; FAIL=1; }
if [ "${#GPU_LOGS[@]}" -ne "$NGPU" ]; then
  echo "ERROR: expected $NGPU latest-round GPU logs, found ${#GPU_LOGS[@]}" >&2; FAIL=1
else
  for log in "${GPU_LOGS[@]}"; do
    grep -q 'gpu_tot_sim_cycle' "$log"          || { echo "ERROR: no GPU cycles in $log" >&2; FAIL=1; }
    grep -q 'external_gpgpusim_runner_rc=0' "$log" || { echo "ERROR: runner failed in $log" >&2; FAIL=1; }
    grep -q "max_parallel=$MOE_GPGPUSIM_MAX_PARALLEL" "$log" || { echo "ERROR: host concurrency protocol missing in $log" >&2; FAIL=1; }
    grep -q '\] done local_tokens=' "$log"      || { echo "ERROR: worker incomplete in $log" >&2; FAIL=1; }
    if [ -n "$SIM_FFN_REPEAT" ]; then
      grep -q "hidden=$SIM_HIDDEN inter=$SIM_INTERMEDIATE ffn_repeat=$SIM_FFN_REPEAT" "$log" ||
        { echo "ERROR: worker simulation parameters differ from dispatch metadata in $log" >&2; FAIL=1; }
    fi
  done
fi
grep -q 'using ILP layout coords' "${ROUTER_LOG[@]}" /dev/null 2>/dev/null || { echo "ERROR: router did NOT load ILP coord file" >&2; FAIL=1; }
grep -q '\[CPU router\] done' "${ROUTER_LOG[@]}" /dev/null 2>/dev/null || { echo "ERROR: router did not finish" >&2; FAIL=1; }
grep -q "gpu_init_protocol=row_col init_bytes_per_worker=$EXPECTED_GPU_INIT_BYTES" \
  "${ROUTER_LOG[@]}" /dev/null 2>/dev/null ||
  { echo "ERROR: router did not use compact row/column init protocol" >&2; FAIL=1; }
if [ "$DISPATCH_MODE" = "real_dispatch_file" ]; then
  grep -q 'trace_source=real_dispatch_file' "${ROUTER_LOG[@]}" /dev/null 2>/dev/null ||
    { echo "ERROR: router did NOT consume staged dispatch_matrix.txt" >&2; FAIL=1; }
  grep -q "tokens_per_gpu=$SIM_TOKENS_PER_SOURCE .*topk=$SIM_TOPK hidden=$SIM_HIDDEN inter=$SIM_INTERMEDIATE" \
    "${ROUTER_LOG[@]}" /dev/null 2>/dev/null ||
    { echo "ERROR: router simulation parameters differ from dispatch metadata" >&2; FAIL=1; }
  if ! diff -u "$RUN/expected_dispatch_matrix.log" \
                   "$RUN/effective_dispatch_matrix.log" \
                   > "$RUN/dispatch_matrix.diff"; then
    echo "ERROR: router's effective dispatch matrix differs from staged input" >&2
    FAIL=1
  fi
else
  grep -q 'trace_source=zipf_synthetic' "${ROUTER_LOG[@]}" /dev/null 2>/dev/null ||
    { echo "ERROR: router did not report explicit synthetic dispatch" >&2; FAIL=1; }
fi
grep -qi 'total finished:' "$POPNET_LOG" 2>/dev/null || { echo "ERROR: popnet did not finish traffic" >&2; FAIL=1; }
[ -z "${SYNC_MAKESPAN:-}" ] || [ "$SYNC_MAKESPAN" -gt 0 ] 2>/dev/null || { echo "ERROR: sync makespan zero" >&2; FAIL=1; }
[ -n "${LAST_R:-}" ] && [ "$LAST_R" -ge 2 ] 2>/dev/null || { echo "ERROR: NoC delay not replayed in a 2nd Phase 1" >&2; FAIL=1; }

POPNET_AVG_DELAY="$(awk 'tolower($0) ~ /average delay:/ { gsub(/\r/, ""); v=$NF } END { print v }' "$POPNET_LOG" 2>/dev/null || true)"
POPNET_TOTAL_FINISHED="$(awk 'tolower($0) ~ /total finished:/ { gsub(/\r/, ""); v=$NF } END { print v }' "$POPNET_LOG" 2>/dev/null || true)"
POPNET_TOTAL_POWER="$(awk 'tolower($0) ~ /total power:/ { gsub(/\r/, ""); v=$NF } END { print v }' "$POPNET_LOG" 2>/dev/null || true)"
GPU_CYCLE_MAX="$(
  grep -h 'gpu_tot_sim_cycle' "${GPU_LOGS[@]}" /dev/null 2>/dev/null |
    awk '{ v=$NF+0; if (v>m) m=v } END { if (NR) print m }' || true
)"
{
  for log in "${GPU_LOGS[@]}"; do
    awk -v file="${log#$RUN/}" '/gpu_tot_sim_cycle/ { value=$NF } END { if (value != "") print file "\t" value }' "$log"
  done
} > "$RUN/gpu_cycles.tsv"

# Cross-check the compact metrics before declaring a paper-grade PASS.  The
# bench trace records communication issue/completion timestamps, while the
# final InterChiplet banner includes the last synchronization completion.  The
# latter is the end-to-end makespan and may exceed the trace maximum, but can
# never be smaller.  For a file-backed dispatch, PopNet must also finish every
# non-local dispatch/combine transfer plus the router init/completion exchange
# for every GPU worker.
CYCLE_COMPLETION_DELTA=""
case "${MAKESPAN:-}" in ''|*[!0-9]*) FAIL=1 ;; esac
case "${SYNC_MAKESPAN:-}" in ''|*[!0-9]*) FAIL=1 ;; esac
if [ -n "${MAKESPAN:-}" ] && [ -n "${SYNC_MAKESPAN:-}" ] &&
   [ "$SYNC_MAKESPAN" -ge "$MAKESPAN" ] 2>/dev/null; then
  CYCLE_COMPLETION_DELTA="$((SYNC_MAKESPAN - MAKESPAN))"
else
  echo "ERROR: synchronized makespan '${SYNC_MAKESPAN:-missing}' is smaller than/invalid relative to bench max '${MAKESPAN:-missing}'" >&2
  FAIL=1
fi
[ -n "${LAST_P2:-}" ] && [ "$LAST_P2" -ge 2 ] 2>/dev/null ||
  { echo "ERROR: NoC phase did not complete two rounds" >&2; FAIL=1; }

GPU_CYCLE_ROWS="$(awk 'NF == 2 && $2 ~ /^[0-9]+$/ && $2 > 0 { n++ } END { print n + 0 }' "$RUN/gpu_cycles.tsv")"
if [ "$GPU_CYCLE_ROWS" -ne "$NGPU" ]; then
  echo "ERROR: gpu_cycles.tsv has $GPU_CYCLE_ROWS positive rows, expected $NGPU" >&2
  FAIL=1
fi

EXPECTED_POPNET_TOTAL_FINISHED=""
if [ "$DISPATCH_MODE" = "real_dispatch_file" ]; then
  EXPECTED_POPNET_TOTAL_FINISHED="$(
    awk -v n="$NGPU" '
      $1 == "round" { reading=1; next }
      reading && $0 !~ /^[[:space:]]*#/ && $0 !~ /^[[:space:]]*$/ {
        for (d=1; d<=n; ++d) if ((rows != d-1) && ($d+0 > 0)) nonlocal++
        rows++
        if (rows == n) {
          print 2 * nonlocal + 2 * n
          exit
        }
      }
    ' "$MOE_DISPATCH_FILE"
  )"
  case "$POPNET_TOTAL_FINISHED" in
    ''|*[!0-9]*)
      echo "ERROR: PopNet total-finished is not an integer: '${POPNET_TOTAL_FINISHED:-missing}'" >&2
      FAIL=1
      ;;
    *)
      if [ -z "$EXPECTED_POPNET_TOTAL_FINISHED" ] ||
         [ "$POPNET_TOTAL_FINISHED" -ne "$EXPECTED_POPNET_TOTAL_FINISHED" ]; then
        echo "ERROR: PopNet finished $POPNET_TOTAL_FINISHED messages, expected ${EXPECTED_POPNET_TOTAL_FINISHED:-missing}" >&2
        FAIL=1
      fi
      ;;
  esac
fi

# Prove that the shared simulator/build tree stayed immutable throughout this
# replay.  Without this end hash, a rebuild during a long scale run could mix
# worker executables while still leaving a superficially complete result.
hash_simulator_binaries > "$RUN/simulator_binaries.end.sha256"
BINARY_IMMUTABILITY=PASS
if ! diff -u "$RUN/simulator_binaries.sha256" \
             "$RUN/simulator_binaries.end.sha256" \
             > "$RUN/simulator_binaries.diff"; then
  echo "ERROR: simulator binaries changed during the run" >&2
  BINARY_IMMUTABILITY=FAIL
  FAIL=1
fi
BINARY_START_MANIFEST_SHA="$(sha256sum "$RUN/simulator_binaries.sha256" | awk '{print $1}')"
BINARY_END_MANIFEST_SHA="$(sha256sum "$RUN/simulator_binaries.end.sha256" | awk '{print $1}')"
BINARY_DIFF_SHA="$(sha256sum "$RUN/simulator_binaries.diff" | awk '{print $1}')"

if [ "$FAIL" -eq 0 ]; then
  VALIDATION=PASS
  echo "ILP_E2E_VALIDATION=PASS makespan=$MAKESPAN"
else
  VALIDATION=FAIL
  echo "ILP_E2E_VALIDATION=FAIL"
fi
# Persist a compact summary for closed_loop / the design doc.
{
  echo "validation=$VALIDATION"
  echo "interchiplet_rc=$IC_RC"
  echo "makespan_cycles=$MAKESPAN"
  echo "synchronized_makespan_cycles=${SYNC_MAKESPAN:-}"
  echo "completion_delta_cycles=${CYCLE_COMPLETION_DELTA:-}"
  echo "rounds_phase1=${LAST_R:-0}"
  echo "rounds_phase2=${LAST_P2:-0}"
  echo "rounds_requested=$ROUNDS"
  echo "n_gpu_workers=${#GPU_LOGS[@]}"
  echo "gpu_init_protocol=row_col"
  echo "gpu_init_bytes_per_worker=$EXPECTED_GPU_INIT_BYTES"
  echo "gpu_cycle_max=${GPU_CYCLE_MAX:-}"
  echo "popnet_total_finished=${POPNET_TOTAL_FINISHED:-}"
  echo "popnet_expected_finished=${EXPECTED_POPNET_TOTAL_FINISHED:-}"
  echo "popnet_average_delay=${POPNET_AVG_DELAY:-}"
  echo "popnet_total_power=${POPNET_TOTAL_POWER:-}"
  echo "dispatch_mode=$DISPATCH_MODE"
  echo "sim_tokens_per_source=${SIM_TOKENS_PER_SOURCE:-}"
  echo "sim_topk=${SIM_TOPK:-}"
  echo "sim_hidden=${SIM_HIDDEN:-}"
  echo "sim_intermediate=${SIM_INTERMEDIATE:-}"
  echo "sim_ffn_repeat=${SIM_FFN_REPEAT:-}"
  echo "gpgpusim_max_parallel=$MOE_GPGPUSIM_MAX_PARALLEL"
  echo "simulator_binaries_immutable=$BINARY_IMMUTABILITY"
  echo "simulator_binaries_start_manifest_sha256=$BINARY_START_MANIFEST_SHA"
  echo "simulator_binaries_end_manifest_sha256=$BINARY_END_MANIFEST_SHA"
  echo "simulator_binaries_diff_sha256=$BINARY_DIFF_SHA"
  echo "simulator_root=$SIMULATOR_ROOT"
  echo "benchmark_source_root=$BENCHMARK_SOURCE_ROOT"
  echo "interchiplet_benchmark_root=$STAGE"
  echo "codegen_source=$CG_DIR"
  echo "input_stage=$STAGE"
  echo "coord_file=$MOE_COORD_FILE"
  echo "dispatch_file=${MOE_DISPATCH_FILE:-}"
  echo "source_input_manifest_sha256=$SOURCE_MANIFEST_SHA"
  echo "staged_input_manifest_sha256=$STAGED_MANIFEST_SHA"
} > "$RUN/ilp_e2e_summary.txt"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN/run_manifest.txt"
RAW_MANIFEST_TMP="${RUN}.raw_output_manifest.sha256.$$"
hash_tree "$RUN" > "$RAW_MANIFEST_TMP"
mv -- "$RAW_MANIFEST_TMP" "$RUN/raw_output_manifest.sha256"
exit "$FAIL"
