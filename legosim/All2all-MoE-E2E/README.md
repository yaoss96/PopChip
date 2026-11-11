# All2all-MoE-E2E

End-to-end MoE layer smoke benchmark using the real LEGOSim simulators:

- CPU router/gating chiplet: Sniper (`snipersim/run-sniper`).
- GPU expert chiplets: GPGPU-Sim through CUDA `moe_gpu_worker` binaries.
- Inter-chiplet network: popnet_chiplet phase2 consuming generated `bench.txt`.

This is intentionally smaller than full DeepSeek-V3/R1, but it preserves the end-to-end MoE layer structure:

1. CPU router samples DeepSeek-like top-K expert assignments with Zipf skew.
2. CPU broadcasts the dispatch matrix to GPU workers.
3. GPU workers perform ordered dispatch all-to-all of token hidden-state payloads.
4. Each GPU runs a CUDA expert-MLP kernel for its local+remote expert tokens under GPGPU-Sim.
5. GPU workers perform combine all-to-all back to source GPUs.
6. CPU gathers completion summaries.
7. popnet simulates the generated inter-chiplet trace.

Default smoke dimensions in `moe_e2e.yml`:

```text
CPU router: node 35 = (5,5)
GPU workers: nodes 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3)
experts = 256
top-K = 8
tokens/GPU = 4
hidden = 128       # smoke-scaled for GPGPU-Sim
intermediate = 64  # smoke-scaled expert FFN
Zipf alpha = 1.2
```

The DeepSeek-like part is the routed-expert count/top-K/skewed expert selection. The compute size is smoke-scaled so GPGPU-Sim can finish locally.

Run from Windows Git Bash / Hermes terminal:

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Nvidia_SDKM_Ubuntu_22.04_JetPack_6.2.2 -- \
  bash "$SIMULATOR_ROOT/benchmark/All2all-MoE-E2E/run_moe_e2e_wsl.sh"
```

Run artifacts:

```text
/tmp/legosim_moe_e2e_run/bench.txt
/tmp/legosim_moe_e2e_run/delayInfo.txt
/tmp/legosim_moe_e2e_run/proc_*/*.log
```

Topology/parameter sweep:

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Nvidia_SDKM_Ubuntu_22.04_JetPack_6.2.2 -- bash -lc '
cd "$SIMULATOR_ROOT"
source setup_env.sh >/dev/null 2>&1 || true
python3 benchmark/All2all-MoE-E2E/run_topology_param_sweep.py \
  --out "$SIMULATOR_ROOT/benchmark/All2all-MoE-E2E/sweep_results" \
  --workloads hot0:4:128:64:1.2:0:202501,hot1:4:128:64:1.2:64:202502,hot2:4:128:64:1.2:128:202503 \
  --timeout 420
'
```

The sweep runs Sniper/GPGPU-Sim once per workload to generate `bench.txt`, then replays the same trace through popnet with:

- mesh
- torus
- dragonfly
- fat_tree
- star
- adaptive_hotcold
- adaptive_explicit

There are now two adaptive variants:

`adaptive_hotcold` is the legacy collapsed approximation:

- cold/DynLayer: fixed 6x6 mesh covering all chiplets
- hot/AspLayer: configurable direct shortcut links among predicted hot expert chiplet positions and active source GPU chiplets

`adaptive_explicit` is the explicit dual-layer/source-routed version:

- DynLayer nodes 0..35: fixed 6x6 mesh over all chiplet positions
- AspLayer nodes 36..71: one programmable high-layer node per chiplet position
- vertical turn-out edges connect chiplet i to AspLayer node 36+i
- hot-layer shortcut edges connect active GPU source positions to predicted hot expert-region positions on AspLayer
- `make_rich_trace_hotcold.py` emits a rich trace where each packet carries `phase=... layer=... class=... orig_src=... orig_dst=...`; hot MoE GPU-GPU packets are addressed to `36+src,36+dst`, while cold/control packets stay on DynLayer
- `asplayer.bitfile.json` records the configured AspLayer nodes/edges for this workload, analogous to a programmable active-interposer fabric bitfile/config
- adaptive-explicit GraphViz edges use different weights: DynLayer mesh `weight=100, energy_per_hop=1.0`; vertical turn-out `weight=8, energy_per_hop=0.10`; AspLayer configurable H-link-like shortcuts `weight=20, energy_per_hop=0.35`
- patched popnet parses the packet header metadata and prints layer/phase/class utilization counters

This follows the SHINSAI-style idea of a reusable active TSV interposer with programmable H-Link fabric and reconfigurable vertical links: the benchmark models DynLayer as the fixed NoC/mesh and AspLayer as a workload-programmed high-bandwidth fabric. Packet class selection is represented in the trace header and consumed by the patched popnet input path.

Rich-trace smoke result location:

```text
benchmark/All2all-MoE-E2E/sweep_results_shinsai_rich/
```

Important limitation:

This is an end-to-end simulator-backed smoke benchmark, not a full-size DeepSeek execution. Full DeepSeek hidden size 7168 and FFN dimensions are too expensive for a quick GPGPU-Sim run. Use this to validate methodology and then scale dimensions/traffic gradually.

## Driving the E2E benchmark with a real MoE routing trace

By default `moe_router_cpu` samples top-K expert assignments from a Zipf prior
inside the C++ binary. For HPCA-grade evaluation you can replace that with
**real top-K routing decisions captured from a HuggingFace MoE LLM running on
MMLU prompts**, while keeping Sniper / GPGPU-Sim / popnet exactly the same.

Pipeline:

```text
dump_real_moe_trace.py          # GPU side, runs in WSL
   (../All2all-MoE/)            #   HF MoE model + gate hook on MMLU prompts
        |
        v
real_moe_trace.jsonl            # per-(request, layer, token, top_k_experts)
        |
        v
real_trace_to_dispatch_matrix.py
        |
        v
dispatch_matrix.txt             # one or more 4x4 dispatch matrices
        |   MOE_DISPATCH_FILE=...
        v
moe_router_cpu (patched)        # loads matrix instead of Zipf; Sniper unchanged
        |
        v
moe_gpu_worker x 4 + popnet     # unchanged
```

One-shot run from Windows Git Bash:

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d <distro> -- \
  bash "$SIMULATOR_ROOT/benchmark/All2all-MoE-E2E/run_real_trace_wsl.sh"
```

Knobs (env vars):

- `REAL_MOE_MODEL` = `olmoe-1b-7b` (default, fits 8GB int8) | `granite-1b-a400m` |
  `qwen-moe-a2.7b` | `deepseek-v2-lite`
- `REAL_MOE_QUANT` = `int8` (default) | `int4` | `none`
- `REAL_MOE_N_PROMPTS`, `REAL_MOE_MAX_NEW_TOKENS`, `REAL_MOE_MAX_PROMPT_TOKENS`
- `REAL_MOE_SKIP_DUMP=1 REAL_MOE_TRACE=/path/trace.jsonl` to reuse an existing dump
- `REAL_TRACE_MODE` = `all` (Path A, single matrix per workload) | `per_layer` |
  `per_layer_phase` (Path B / 12-phase taxonomy)
- `REAL_TRACE_PHASES` = `both` | `prefill` | `decode`
- `REAL_TRACE_EXPERTS` = experts count used for the expert-to-GPU partition
  while converting the real trace. Default `0` uses the real model's expert
  count from trace `_meta`; override only for controlled remapping experiments.
- Real-trace adaptive topologies now choose hot expert regions automatically
  from `dispatch_matrix.txt`: destination-column sums identify the hottest
  resident-expert GPUs, then `GPU_NODES=[0,6,12,18]` maps those GPUs to chiplet
  nodes. Override with `--hot-nodes`, or change the count with
  `--hot-node-count` when invoking `run_topology_param_sweep.py` directly.
- Phase-aware adaptive configuration is enabled with `--phase-aware`. The sweep
  filters the real trace into `prefill` and `decode`, chooses hot nodes per
  phase, generates one AspLayer/H-Link-like fabric configuration per phase, and
  aggregates the phase runs. A one-time phase-switch reconfiguration cost is
  charged to adaptive topologies with `--phase-config-overhead-cycles` (default
  1000 cycles), modeling the CPU/software-programmed fabric setup used by
  SHINSAI-style TOB/COB/V-Link configuration. If the trace contains only one
  phase, no switch overhead is charged and the summary prints a warning.
- `SWEEP_OUT`, `SWEEP_WORKLOADS`, `SWEEP_TIMEOUT` (forwarded to
  `run_topology_param_sweep.py`)

Direct invocation without the wrapper:

```bash
python3 run_topology_param_sweep.py \
    --out /tmp/sweep_real \
    --workloads realtrace:4:128:64:1.2:0:202501 \
    --real-trace /tmp/real_moe_trace.jsonl \
    --real-trace-mode all
```

Phase-aware direct invocation:

```bash
python3 run_topology_param_sweep.py \
    --out /tmp/sweep_real_phaseaware \
    --workloads realtrace:4:128:64:1.2:0:202501 \
    --real-trace /tmp/real_moe_trace.jsonl \
    --phase-aware \
    --phase-config-overhead-cycles 1000
```

Verification: each workload's `sniper_log/sniper.router.5.5.log` (and the
sweep's `interchiplet.stdout`) contains a `trace_source=` line printed by the
patched router. `trace_source=real_dispatch_file` means the matrix was loaded
from the file; `trace_source=zipf_synthetic` means it fell back. This makes it
easy to A/B real vs Zipf in the same sweep output directory by running with
and without `--real-trace`.

### Files added for the real-trace path

- `real_trace_to_dispatch_matrix.py` -- JSONL -> dispatch_matrix.txt + dispatch_meta.json
- `moe_router_cpu.cpp` -- patched: reads `MOE_DISPATCH_FILE` env var, falls back to Zipf otherwise
- `run_topology_param_sweep.py` -- new flags: `--real-trace`, `--real-trace-mode`, `--real-trace-phases`, `--real-trace-experts`, `--hot-node-count`, `--hot-nodes`, `--phase-aware`, `--phase-config-overhead-cycles`
- `run_real_trace_wsl.sh` -- one-shot wrapper
