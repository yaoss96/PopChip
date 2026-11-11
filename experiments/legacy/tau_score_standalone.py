# -*- coding: utf-8 -*-
"""Standalone protocol-makespan scorer for a FIXED layout at a given tau_p.

Uses ilp_layout._evaluate_fixed_order_protocol directly (the same pure max-plus
recurrence that build_and_solve reports as protocol_makespan_cycles), so it can
score ANY layout -- including TAP's, which is not representable in the ILP's own
footprint constraints and therefore cannot be scored via fixed_solution.

Layout-independent constants are rebuilt from params exactly as build_and_solve
does (deterministic_dispatch, token/ctrl bytes, sync=1000, flit=32, token-linear
compute regressions).  Layout-dependent inputs (coords + expert->kind) come from
the layout JSON.  Validated to reproduce the ILP fixed_solution value on the
Floorplet layout at tau=104 (2079757.232).
"""
from __future__ import print_function
import os, sys, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT + "/src")
import ilp_layout as il


def _num_experts(params):
    meta = params.get("meta", {})
    if "groups" in meta:
        return int(meta["groups"])
    # fall back to counting e(g) capable tasks
    return int(meta.get("num_experts"))


def build_constants(params, tokens_per_source=32, dispatch_hidden=128,
                    sync=1000.0, flit=32.0):
    """Return the layout-independent protocol constants."""
    dm = il.deterministic_dispatch(params, tokens_per_source)
    G = len(dm)
    bytes_elem = int(params.get("meta", {}).get("bytes_elem", 2))
    token_bytes_per_phase = dispatch_hidden * bytes_elem
    init_bytes = 4.0 * (4 + 2 * G)
    done_bytes = 4.0
    column_tokens = [sum(dm[s][t] for s in range(G)) for t in range(G)]

    def compute_for(group, kind):
        col = column_tokens[group]
        if kind == "GPU-S":
            return 7431.145 + 243.523 * col
        if kind == "GPU-L":
            return 4575.206 + 0.6926 * col
        raise ValueError("no protocol compute profile for kind %r" % kind)

    return {
        "dm": dm, "G": G,
        "token_bytes_per_phase": float(token_bytes_per_phase),
        "init_bytes": float(init_bytes), "done_bytes": float(done_bytes),
        "sync": float(sync), "flit": float(flit),
        "column_tokens": column_tokens, "compute_for": compute_for,
    }


def _site_coord(site):
    return (float(site["i"]), float(site["j"]), float(site.get("l", 0)))


def layout_sites_by_group(layout, G):
    """Return (cpu_coord, [expert_coord_g], [expert_kind_g]) for g in 0..G-1."""
    cpu = None
    by_group = {}
    for s in layout["sites"]:
        if s["type"] == "CPU":
            cpu = _site_coord(s)
            continue
        for task in s.get("tasks", []):
            m = task.strip()
            # tasks look like 'e(12)'
            if m.startswith("e(") and m.endswith(")"):
                g = int(m[2:-1])
                by_group[g] = (_site_coord(s), s["type"])
    if cpu is None:
        raise ValueError("layout has no CPU site")
    coords, kinds = [], []
    for g in range(G):
        if g not in by_group:
            raise ValueError("layout missing expert e(%d)" % g)
        c, k = by_group[g]
        coords.append(c)
        kinds.append(k)
    return cpu, coords, kinds


def score_layout(params, layout, tau, const=None):
    if const is None:
        const = build_constants(params)
    G = const["G"]
    cpu, coords, kinds = layout_sites_by_group(layout, G)
    compute_cycles = [const["compute_for"](g, kinds[g]) for g in range(G)]
    res = il._evaluate_fixed_order_protocol(
        cpu, coords, const["dm"], compute_cycles,
        const["token_bytes_per_phase"], const["init_bytes"],
        const["done_bytes"], const["sync"], const["flit"], float(tau),
        expert_resource_ids=None, compute_order=None)
    return float(res["completion_cycles"])


if __name__ == "__main__":
    P = ROOT + "/data/params"
    B = ROOT + "/out/joint_bom_scaling_fresh_20260802/baselines/scaling/e064_m12/prior"
    params = json.load(open("%s/experts_064.json" % P))
    floor = json.load(open("%s/floorplet.json" % B))
    tap = json.load(open("%s/tap.json" % B))
    const = build_constants(params)
    print("G =", const["G"], "token_bytes/phase =",
          const["token_bytes_per_phase"], "init_bytes =", const["init_bytes"])
    f104 = score_layout(params, floor, 104, const)
    t104 = score_layout(params, tap, 104, const)
    print("floor@104 =", f104, " (ILP fixed_solution ref = 2079757.232)")
    print("tap@104   =", t104)
    print("rel-err floor vs ref = %.4e" % (abs(f104 - 2079757.232) / 2079757.232))
