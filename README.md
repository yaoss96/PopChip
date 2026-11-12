# PopChip

Code and data for **PopChip: Popularity-Aware Chiplet Architecture Co-Design for
Mixture-of-Experts Inference** by Shangshang Yao (Information Engineering
University, Zhengzhou, China), Tong Zhao, and Zuoning Chen (Jiangnan Institute of
Computing Technology, Wuxi, China). Contact: yaoshangshang96@outlook.com.

PopChip is a mixed-integer linear program (MILP) whose only objective is the
makespan `T` of an MoE layer's dispatch–compute–combine protocol. One solve
returns:

- the typed bill of materials (how many GPU-L, GPU-S, ... dies to buy);
- the expert-to-die map;
- the die anchors on the interposer Mesh;
- the routing and schedule.

The solve runs under hard area and peak-power budgets. Every reported layout is
replayed independently in the [LEGOSim](https://github.com/FCAS-LAB/LEGOSIM_MICRO)
chiplet simulator (GPGPU-Sim workers + PopNet interposer).

## Repository layout

```
src/                      MILP (ilp_layout.py), workload generator, baselines,
                          protocol evaluator, LEGOSim codegen and replay drivers
experiments/              scripts behind every table and figure of the paper
  ispa_common.py          shared paths and solver settings
  realtrace/              routing traces of a trained MoE (Phi-tiny-MoE)
  legacy/                 earlier studies (certified 3-expert control, D4 break, ...)
legosim/All2all-MoE-E2E/  LEGOSim benchmark: CPU router + GPU expert workers
data/
  params/                 instances (experts_008 ... experts_128.json)
  layouts/<instance>/     replayed layouts: popchip, floorplet (FP-px), tap (TAP-px),
                          blind1-3, poprank_* (PopRank heuristics)
  realtrace/              recorded top-2 routing counts (32 layers x 3 corpora)
results/                  outputs reported in the paper
  legosim/measured.csv    every LEGOSim measurement (case, method, cycles,
                          layout, params)
```

Instances are named `e<experts>_m<mesh side>`, e.g. `e064_m16` is 64 experts
on a 16x16 Mesh.

## Requirements

- Python 3.7+ with `gurobipy` (Gurobi 10 or newer and a valid license). Scripts
  that call the solver in a subprocess use `GUROBI_PYTHON`, defaulting to the
  current interpreter.
- To replay layouts: Linux or WSL with a built LEGOSim checkout. Copy
  `legosim/All2all-MoE-E2E` into `$LEGOSIM_ROOT/benchmark/`.
- To record routing traces (optional): `torch`, `transformers>=4.46`,
  `huggingface_hub`, `pandas`, `pyarrow`.

## Quick start

Solve the 8-expert instance with the paper's settings:

```bash
python src/ilp_layout.py --params data/params/experts_008.json \
  --L 1 --X 6 --Y 6 --a-max 1440 --p-max 1120 --n-cpu 1 \
  --time-limit 45 --mipgap 0.01 --seed 1 --mip-focus 1 --mode ilp-perf \
  --noc-traffic deterministic-dispatch --dispatch-tokens-per-source 32 \
  --dispatch-hidden 128 --cpu-anchor free --threads 1 -o out/e008.json
```

Generate the placement proxies and the popularity-blind maps for a solution:

```bash
python src/generate_revalidation_baselines.py --params data/params/experts_008.json \
  --source out/e008.json --case-id e008_m06 --out-dir out/e008_prior \
  --floorplet-time-limit 45 --tap-wall-budget 15 --tap-seeds 11,29,47 --tap-alpha 0.7
python src/make_popularity_blind_variant.py --solution out/e008.json \
  --out-dir out/e008_blind --tag e008_m06 --seeds 1 2 3
```

Replay a layout in LEGOSim:

```bash
python src/replay_legosim_solutions.py --params data/params/experts_008.json \
  --layout e008_m06:popchip:data/layouts/e008_m06/popchip.json \
  --out-dir out/replay_e008 --legosim-root "$LEGOSIM_ROOT" --rounds 2
```

## Reproducing the paper

Run these from `experiments/` with a gurobipy interpreter. Analytical results
need only Gurobi; measured results need LEGOSim.

| Paper item | Command |
|---|---|
| Table II, Fig. 5 (from the measurements) | `python gen_tables.py` |
| Re-measure every replayed layout | `python make_replay_queue.py > queue.txt` then `sh replay_queue.sh queue.txt` |
| PopRank two-stage heuristics | `python exp_poprank.py e008_m06` (one per instance); `python gen_poprank_large.py` |
| Calibration of analytical T vs. measured cycles | `python calib.py` |
| Link slack, proxy coupling, token mass, budget sweep | `python thermal_and_budget.py` |
| Drift, Table III | `python exp_drift.py e008_m06` (per instance), then `python gen_drift_table.py` |
| BOM selection sweep (48 configurations) | `python exp_regime.py <worker> <n_workers>`; 600-s runs: `python exp_regime_long.py A1985 2080 1 600 [N_L]` |
| Table IV (certified 3-expert control) | `python legacy/counterexample.py` |
| Richer libraries (NPU, SRAM-CIM, mid GPU) | `python exp_hetero.py <experts> <extra types 0-3>` |
| Warm starts at 64 and 128 experts | `python exp_warm.py e064_m16` |
| Model sizes | `python exp_modelsize.py` |
| Budget sweep (Fig. 3) data | `results/budget_sweep/legosim_metrics.csv` |
| Encoding comparison (QAP ablation) | `results/encoding/qap_summary.csv` |

Trained-model routing (Phi-tiny-MoE, 16 experts, top-2):

```bash
python realtrace/download_assets.py          # model weights + WikiText-2 / GSM8K
python -P realtrace/collect_routing.py       # writes data/realtrace/routing_counts.json
python exp_realtrace.py stats
python exp_realtrace.py design 26            # median-skew layer
python exp_realtrace.py drift 26
python per_layer_benefit.py
python summarize_realtrace.py 26
```

`collect_routing.py` runs only transformers' built-in PhiMoE implementation
with safetensors weights. It never executes the model repository's remote code.

## What the numbers mean

- **Analytical T** is the MILP objective: the fixed-order protocol completion
  with profiled compute and contention-free hop latency, plus an aggregate
  link-capacity bound.
- A closed gap certifies T only. It does not certify measured cycles, which add
  PopNet arbitration and queueing.
- Analytical T ranks layouts as measured. It is optimistic by a nearly uniform
  factor per instance (about 6% at 8 experts, under 1% from 32 experts on);
  see `results/calibration.json`.
- Replays are deterministic to within a few cycles.

## License

The code is released under the MIT License. LEGOSim, Gurobi, the Phi-tiny-MoE
weights, and the WikiText-2/GSM8K corpora are third-party and keep their own
licenses.
