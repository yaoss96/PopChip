#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workload.py -- MoE task-graph + cost-model generator for the ILP.

Emits a single params.json consumed by ilp_layout.py (Section 5 "contract"
table of the design doc). Pure standard library so it runs under the Gurobi
py37 interpreter as well as system python3.

Design-doc mapping:
  * Section 2.1  task graph (DAG)   -> tasks[], edges[]
  * Section 2.2  traffic vol_e      -> edges[].vol
  * Section 2.3  expert grouping    -> --experts / --groups
  * Section 1.3  chiplet library    -> library{}
  * Section 5    calibration table  -> cycles / energy / area / power / tau / eps

IMPORTANT: the per-(task,type) cycles and energies produced here are an
ANALYTIC PLACEHOLDER cost model, clearly parameterised by FLOPs and bytes.
They are meant to be REPLACED by the profiling harness (GPGPU-Sim / Sniper)
per Section 5. Everything downstream reads params.json, so swapping in
measured numbers is a drop-in replacement -- nothing in the ILP is hand-guessed
at solve time.
"""

import argparse
import json
import math

# ----------------------------------------------------------------------------
# Chiplet library (Section 1.3). Numbers are placeholder calibration values;
# the profiling harness overwrites these. w,h are footprint in grid sites.
# area a_k in mm^2, power in W, memory M_k in MB.
# ----------------------------------------------------------------------------
LIBRARY = {
    # mem = per-chiplet weight-hosting capacity (HBM/local, MB). Per Section 9.6
    # expert weights live in HBM, not on-chip SRAM; C5 is a capacity stand-in.
    "GPU-L": {"simlet": "gpgpu-sim-80sm", "area": 400.0, "w": 2, "h": 2,
              "p_peak": 300.0, "p_static": 40.0, "mem": 8192.0},
    "GPU-S": {"simlet": "gpgpu-sim-20sm", "area": 110.0, "w": 1, "h": 1,
              "p_peak": 90.0,  "p_static": 12.0, "mem": 2048.0},
    "CPU":   {"simlet": "sniper-8core",   "area": 90.0,  "w": 1, "h": 1,
              "p_peak": 65.0,  "p_static": 15.0, "mem": 4096.0},
}

# Relative throughput knobs for the analytic cost model.
# GPU is fast on dense GEMM; CPU is fast on branchy/serial control (gate).
GEMM_CYC_PER_MFLOP = {"GPU-L": 900.0, "GPU-S": 3200.0, "CPU": 26000.0}
CTRL_CYC_PER_KTOK  = {"GPU-L": 5200.0, "GPU-S": 4800.0, "CPU": 1500.0}  # gate/router
# dynamic energy pJ per MFLOP (compute) -- rough placeholders
GEMM_PJ_PER_MFLOP  = {"GPU-L": 55.0, "GPU-S": 70.0, "CPU": 380.0}
CTRL_PJ_PER_KTOK   = {"GPU-L": 620.0, "GPU-S": 540.0, "CPU": 190.0}


def zipf_popularity(E, skew):
    """Expert popularity pi_e, normalised. skew=0 -> uniform; larger -> skewed.
    Section 2.2: do NOT assume pi_e = 1/E."""
    if skew <= 0:
        return [1.0 / E] * E
    w = [1.0 / ((r + 1) ** skew) for r in range(E)]
    s = sum(w)
    return [x / s for x in w]


def group_experts(pi, groups):
    """Popularity-balanced bin-packing of E experts into G groups (Section 2.3).
    Returns list of (group_pi, member_indices)."""
    E = len(pi)
    order = sorted(range(E), key=lambda e: -pi[e])
    buckets = [[0.0, []] for _ in range(groups)]
    for e in order:
        b = min(range(groups), key=lambda g: buckets[g][0])
        buckets[b][0] += pi[e]
        buckets[b][1].append(e)
    return [(b[0], b[1]) for b in buckets]


def build(config, experts, groups, topk, d_model, d_ff, batch, bytes_elem,
          skew):
    """Build the MoE single-layer task graph and cost model."""
    G = groups
    pi = zipf_popularity(experts, skew)
    grp = group_experts(pi, G)              # group popularities

    tasks = {}   # id -> {class, allowed, mem_mb, cyc{}, eta{}}
    edges = []   # {src, dst, vol}

    def add_task(tid, cls, allowed, mflop=0.0, ktok=0.0, kind="gemm",
                 mem_mb=0.0):
        cyc, eta = {}, {}
        for k in allowed:
            if kind == "gemm":
                cyc[k] = mflop * GEMM_CYC_PER_MFLOP[k]
                eta[k] = mflop * GEMM_PJ_PER_MFLOP[k] * 1e-12   # J
            else:  # control-heavy
                cyc[k] = ktok * CTRL_CYC_PER_KTOK[k]
                eta[k] = ktok * CTRL_PJ_PER_KTOK[k] * 1e-12
        tasks[tid] = {"class": cls, "allowed": allowed,
                      "mem_mb": mem_mb, "cyc": cyc, "eta": eta}

    B = batch
    d = d_model
    dff = d_ff
    b = bytes_elem
    # FLOP estimates (MFLOP). attn ~ 4*B*d^2 (qkv+proj, ignoring seq for a token unit)
    attn_mflop = 4.0 * B * d * d / 1e6
    # per-expert FFN: h1 = B_e*d*dff, h2 = B_e*dff*d ; B_e = pi_g*B*topk tokens
    ktok_total = B * topk / 1000.0

    # --- dense front ---
    add_task("t_attn", "attn", ["GPU-L", "GPU-S"], mflop=attn_mflop)
    add_task("t_gate", "gate", ["CPU", "GPU-S"], ktok=ktok_total, kind="ctrl")
    add_task("t_dispatch", "dispatch", ["GPU-L", "GPU-S"],
             mflop=0.15 * attn_mflop)
    edges.append({"src": "t_attn", "dst": "t_gate",
                  "vol": int(B * d * b)})
    edges.append({"src": "t_gate", "dst": "t_dispatch",
                  "vol": int(B * topk * 4)})   # top-k indices, 4B each

    # --- experts (grouped). One EXPERT BODY task per group (h1+h2 fused):
    #     placement puts h1 and h2 of the same expert on the SAME chiplet, so
    #     the internal h1->h2 edge is intra-chiplet (zero hops). This matches
    #     the doc's C6 "at most one expert body per GPU chiplet". ---
    expert_tasks = []
    for g, (gpi, members) in enumerate(grp):
        Bg = gpi * B * topk                       # tokens routed to this group
        ffn_mflop = 2.0 * Bg * d * dff / 1e6      # h1 + h2 fused
        mem_mb = len(members) * 2.0 * d * dff * b / (1024.0 * 1024.0)
        vol_in = int(gpi * B * topk * d * b)      # DISPATCH into / COMBINE out
        eg = "e(%d)" % g
        add_task(eg, "expert", ["GPU-L", "GPU-S"], mflop=ffn_mflop,
                 mem_mb=mem_mb)
        edges.append({"src": "t_dispatch", "dst": eg, "vol": vol_in})
        edges.append({"src": eg, "dst": "t_combine", "vol": vol_in})
        expert_tasks.append(eg)

    # --- combine + norm ---
    add_task("t_combine", "combine", ["GPU-L", "GPU-S"],
             mflop=0.15 * attn_mflop)
    add_task("t_norm", "norm", ["CPU", "GPU-S"], ktok=ktok_total, kind="ctrl")
    edges.append({"src": "t_combine", "dst": "t_norm",
                  "vol": int(B * d * b)})

    params = {
        "meta": {
            "config": config, "experts": experts, "groups": groups,
            "topk": topk, "d_model": d_model, "d_ff": d_ff, "batch": batch,
            "bytes_elem": bytes_elem, "skew": skew,
            "expert_pi": pi,
            "group_pi": [gp for gp, _ in grp],
            "note": "cycles/energy are ANALYTIC PLACEHOLDERS; replace via "
                    "profiling harness (GPGPU-Sim/Sniper), Section 5.",
        },
        "library": LIBRARY,
        "tasks": tasks,
        "edges": edges,
        "expert_tasks": expert_tasks,
        # Interconnect (Section 1.2 / 5). LEGOSim defaults + swept vertical.
        "interconnect": {
            "tau_h": 1.0,          # per-hop horizontal latency (cycles) - calibrated
            "tau_v": 0.2,          # per-hop vertical latency (lambda_tau=0.2)
            "eps_h": 1.17e-12,     # J/bit horizontal (LEGOSim default 1.17 pJ/bit)
            "eps_v": 0.1e-12,      # J/bit vertical (hybrid bond, swept 0.02-0.2)
            "t0": 500.0,           # fixed D2D serialization + packetization (cyc)
            "beta": 16.0,          # D2D bandwidth bytes/cycle
        },
    }
    return params


def build_dense(config, heads, ff_shards, d_model, d_ff, batch, bytes_elem,
                seqlen):
    """Build a NON-MoE (dense) LLM decoder-layer task graph + cost model.

    Two dense workloads share this builder (they differ only in shape):
      * "dense-llm"  : GPT/LLaMA-style dense layer -- multi-head attention
        feeding a tensor-parallel FFN split into `ff_shards` shards. No gate,
        no routing; the parallelism the placer must exploit is head-parallel
        attention + shard-parallel FFN with all-reduce style combine edges.
      * "attn-heavy" : long-context attention-dominated layer -- many parallel
        heads with large KV traffic and a small FFN, stressing the placer on
        attention communication rather than expert routing.

    The cost model reuses the SAME per-type FLOP/byte calibration as the MoE
    builder, so numbers are directly comparable. Parallel shards are marked as
    `expert_tasks` ONLY to reuse constraint C6 (at most one heavy shard per
    chiplet == tensor-parallel one-shard-per-chiplet); semantically these are
    dense tensor-parallel shards, not MoE experts."""
    tasks = {}
    edges = []

    def add_task(tid, cls, allowed, mflop=0.0, ktok=0.0, kind="gemm",
                 mem_mb=0.0):
        cyc, eta = {}, {}
        for k in allowed:
            if kind == "gemm":
                cyc[k] = mflop * GEMM_CYC_PER_MFLOP[k]
                eta[k] = mflop * GEMM_PJ_PER_MFLOP[k] * 1e-12
            else:
                cyc[k] = ktok * CTRL_CYC_PER_KTOK[k]
                eta[k] = ktok * CTRL_PJ_PER_KTOK[k] * 1e-12
        tasks[tid] = {"class": cls, "allowed": allowed,
                      "mem_mb": mem_mb, "cyc": cyc, "eta": eta}

    B = batch
    d = d_model
    dff = d_ff
    b = bytes_elem
    S = seqlen
    dh = d // heads                       # per-head dim
    # per-head attention FLOP: QKV proj + scores(S) + context(S) + out proj.
    # scaled by seqlen S to reflect long-context cost.
    head_mflop = (4.0 * B * dh * d + 2.0 * B * S * dh) / 1e6
    ktok_total = B / 1000.0

    # --- input layernorm / QKV fan-out (control-ish, cheap) ---
    add_task("t_ln", "norm", ["CPU", "GPU-S"], ktok=ktok_total, kind="ctrl")

    # --- parallel attention heads ---
    head_tasks = []
    for h in range(heads):
        th = "h(%d)" % h
        # KV cache footprint grows with context length S
        mem_mb = 2.0 * B * S * dh * b / (1024.0 * 1024.0)
        add_task(th, "attn_head", ["GPU-L", "GPU-S"], mflop=head_mflop,
                 mem_mb=mem_mb)
        edges.append({"src": "t_ln", "dst": th, "vol": int(B * d * b / heads)})
        head_tasks.append(th)

    # --- attention output combine (concat + out-proj) ---
    add_task("t_attn_out", "attn_out", ["GPU-L", "GPU-S"],
             mflop=0.25 * heads * head_mflop)
    for th in head_tasks:
        edges.append({"src": th, "dst": "t_attn_out",
                      "vol": int(B * dh * b)})

    # --- tensor-parallel FFN shards ---
    shard_tasks = []
    for sdx in range(ff_shards):
        ts = "ff(%d)" % sdx
        shard_mflop = 2.0 * B * d * (dff / ff_shards) / 1e6
        mem_mb = 2.0 * d * (dff / ff_shards) * b / (1024.0 * 1024.0)
        add_task(ts, "ffn_shard", ["GPU-L", "GPU-S"], mflop=shard_mflop,
                 mem_mb=mem_mb)
        edges.append({"src": "t_attn_out", "dst": ts,
                      "vol": int(B * d * b / ff_shards)})
        edges.append({"src": ts, "dst": "t_combine", "vol": int(B * d * b)})
        shard_tasks.append(ts)

    # --- all-reduce combine + output norm ---
    add_task("t_combine", "combine", ["GPU-L", "GPU-S"],
             mflop=0.15 * heads * head_mflop)
    add_task("t_norm", "norm", ["CPU", "GPU-S"], ktok=ktok_total, kind="ctrl")
    edges.append({"src": "t_combine", "dst": "t_norm",
                  "vol": int(B * d * b)})

    # heavy shards get exclusivity via C6 (tensor-parallel one per chiplet)
    parallel_tasks = shard_tasks + head_tasks

    params = {
        "meta": {
            "config": config, "dense": True, "heads": heads,
            "ff_shards": ff_shards, "d_model": d_model, "d_ff": d_ff,
            "batch": batch, "seqlen": S, "bytes_elem": bytes_elem,
            "note": "DENSE (non-MoE) LLM decoder layer; cycles/energy are "
                    "ANALYTIC PLACEHOLDERS, replace via GPGPU-Sim/Sniper.",
        },
        "library": LIBRARY,
        "tasks": tasks,
        "edges": edges,
        "expert_tasks": parallel_tasks,
        "interconnect": {
            "tau_h": 1.0, "tau_v": 0.2,
            "eps_h": 1.17e-12, "eps_v": 0.1e-12,
            "t0": 500.0, "beta": 16.0,
        },
    }
    return params


def main():
    ap = argparse.ArgumentParser(description="MoE workload -> params.json")
    ap.add_argument("--config", default="mixtral",
                    choices=["mixtral", "switch", "finegrained",
                             "dense-llm", "attn-heavy", "custom"])
    ap.add_argument("--experts", type=int, default=None)
    ap.add_argument("--groups", type=int, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=4096)
    ap.add_argument("--d-ff", type=int, default=14336)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--bytes-elem", type=int, default=2)   # bf16
    ap.add_argument("--skew", type=float, default=1.0)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--ff-shards", type=int, default=None)
    ap.add_argument("--seqlen", type=int, default=None)
    ap.add_argument("-o", "--out", default="params.json")
    a = ap.parse_args()

    # ----- dense (non-MoE) workloads -----
    dense_presets = {
        # LLaMA-2-13B-ish dense decoder layer, tensor-parallel FFN
        "dense-llm":  dict(heads=8, ff_shards=8, d_model=5120, d_ff=13824,
                           seqlen=2048),
        # long-context attention-dominated layer, small FFN
        "attn-heavy": dict(heads=8, ff_shards=4, d_model=4096, d_ff=8192,
                           seqlen=8192),
    }
    if a.config in dense_presets:
        p = dense_presets[a.config]
        heads    = a.heads     if a.heads     is not None else p["heads"]
        shards   = a.ff_shards if a.ff_shards is not None else p["ff_shards"]
        seqlen   = a.seqlen    if a.seqlen    is not None else p["seqlen"]
        d_model  = p["d_model"] if a.d_model == 4096 else a.d_model
        d_ff     = p["d_ff"]    if a.d_ff == 14336 else a.d_ff
        params = build_dense(a.config, heads, shards, d_model, d_ff,
                             a.batch, a.bytes_elem, seqlen)
        with open(a.out, "w") as fh:
            json.dump(params, fh, indent=2)
        m = params["meta"]
        print("[workload] config=%s (DENSE)  |V|=%d  |E|=%d  heads=%d "
              "ff_shards=%d seqlen=%d" % (a.config, len(params["tasks"]),
              len(params["edges"]), heads, shards, seqlen))
        print("[workload] wrote %s" % a.out)
        return

    # presets from Section 8 evaluation table
    presets = {
        "mixtral":     dict(experts=8,   groups=8,  topk=2, d_ff=14336, skew=0.7),
        "switch":      dict(experts=128, groups=16, topk=1, d_ff=8192,  skew=1.2),
        "finegrained": dict(experts=64,  groups=16, topk=6, d_ff=4096,  skew=1.0),
    }
    if a.config in presets:
        p = presets[a.config]
        experts = a.experts if a.experts is not None else p["experts"]
        groups  = a.groups  if a.groups  is not None else p["groups"]
        topk    = a.topk    if a.topk    is not None else p["topk"]
        d_ff    = p["d_ff"] if a.d_ff == 14336 else a.d_ff
        skew    = p["skew"] if a.skew == 1.0 else a.skew
    else:
        experts = a.experts or 8
        groups  = a.groups or 8
        topk    = a.topk or 2
        d_ff    = a.d_ff
        skew    = a.skew

    params = build(a.config, experts, groups, topk, a.d_model, d_ff,
                   a.batch, a.bytes_elem, skew)
    with open(a.out, "w") as fh:
        json.dump(params, fh, indent=2)
    m = params["meta"]
    print("[workload] config=%s  |V|=%d  |E|=%d  experts=%d groups=%d topk=%d"
          % (a.config, len(params["tasks"]), len(params["edges"]),
             experts, groups, topk))
    print("[workload] group_pi (skew=%.2f): %s"
          % (skew, ["%.3f" % g for g in m["group_pi"]]))
    print("[workload] wrote %s" % a.out)


if __name__ == "__main__":
    main()
