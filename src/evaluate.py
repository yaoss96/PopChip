#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py -- independent placement scorer (the SINGLE source of truth).

Given params.json + a placement, compute the REAL objectives by longest-path
(critical-path) scheduling over the task DAG, using the exact interconnect and
compute cost model. BOTH the ILP solution and every baseline are scored through
this function, so the comparison is apples-to-apples (the ILP's own ObjVal is a
tightened relaxation; we never compare that against baseline heuristics).

A placement is:  {task_id: [l,i,j]}  plus  {(l,i,j): type}.

Objectives returned:
  makespan_cycles   (critical path over DAG, incl. link latency + serialization)
  energy_total_J    (compute + comm)
  byte_hop
  peak_power_W, total_area_mm2
  feasible + reasons[]  (capacity / overlap / type / expert-exclusivity checks)

Pure stdlib; runs under both interpreters.
"""

import json


def site_types_from_solution(sol):
    st = {}
    for s in sol["sites"]:
        st[(s["l"], s["i"], s["j"])] = s["type"]
    return st


def _footprint(k, anchor, lib):
    l, i, j = anchor
    w, h = lib[k]["w"], lib[k]["h"]
    return {(l, i + di, j + dj) for di in range(w) for dj in range(h)}


def check_feasible(params, task_site, site_types, grid, budgets=None):
    """Validate hard constraints. Returns (ok, reasons)."""
    lib = params["library"]
    tasks = params["tasks"]
    L, X, Y = grid["L"], grid["X"], grid["Y"]
    reasons = []

    # occupancy / overlap
    occupied = {}
    for anchor, k in site_types.items():
        for cell in _footprint(k, anchor, lib):
            if cell[1] >= X or cell[2] >= Y or cell[0] >= L:
                reasons.append("chiplet %s@%s out of grid" % (k, anchor))
            if cell in occupied:
                reasons.append("overlap at %s (%s vs %s)"
                               % (cell, occupied[cell], k))
            occupied[cell] = k

    # each task on an occupied anchor of a compatible type; capacity; expert excl
    load_mem = {a: 0.0 for a in site_types}
    expert_count = {a: 0 for a in site_types}
    for v, p in task_site.items():
        a = tuple(p)
        if a not in site_types:
            reasons.append("task %s on empty site %s" % (v, a))
            continue
        k = site_types[a]
        if k not in tasks[v]["allowed"]:
            reasons.append("task %s type %s not allowed" % (v, k))
        load_mem[a] += tasks[v]["mem_mb"]
        if v in set(params["expert_tasks"]):
            expert_count[a] += 1
    for a, k in site_types.items():
        if load_mem[a] > lib[k]["mem"] + 1e-6:
            reasons.append("mem overflow @%s: %.1f>%.1f"
                           % (a, load_mem[a], lib[k]["mem"]))
        if expert_count[a] > 1:
            reasons.append("expert-exclusivity violated @%s (%d)"
                           % (a, expert_count[a]))

    # ---- global budgets (area / package power / per-column density) ----
    if budgets:
        area = sum(lib[k]["area"] for k in site_types.values())
        power = sum(lib[k]["p_peak"] for k in site_types.values())
        if budgets.get("a_max") and area > budgets["a_max"] + 1e-6:
            reasons.append("area %.0f > a_max %.0f" % (area, budgets["a_max"]))
        if budgets.get("p_max") and power > budgets["p_max"] + 1e-6:
            reasons.append("power %.0f > p_max %.0f" % (power, budgets["p_max"]))
        if budgets.get("p_col"):
            col = {}
            for a, k in site_types.items():
                dens = lib[k]["p_peak"] / (lib[k]["w"] * lib[k]["h"])
                for cell in _footprint(k, a, lib):
                    key = (cell[1], cell[2])
                    col[key] = col.get(key, 0.0) + dens
            for key, val in col.items():
                if val > budgets["p_col"] + 1e-6:
                    reasons.append("col %s density %.0f > p_col %.0f"
                                   % (key, val, budgets["p_col"]))
    return (len(reasons) == 0), reasons


def evaluate(params, task_site, site_types, grid, strict=True, budgets=None):
    lib = params["library"]
    tasks = params["tasks"]
    edges = params["edges"]
    ic = params["interconnect"]

    ok, reasons = check_feasible(params, task_site, site_types, grid, budgets)
    if strict and not ok:
        return {"feasible": False, "reasons": reasons}

    # ---- compute durations (type-dependent) ----
    def dur(v):
        a = tuple(task_site[v])
        k = site_types.get(a)
        if k is None or k not in tasks[v]["cyc"]:
            # infeasible placement being scored leniently: pick worst legal type
            k = max(tasks[v]["cyc"], key=lambda kk: tasks[v]["cyc"][kk])
        return tasks[v]["cyc"][k]

    def eta(v):
        a = tuple(task_site[v])
        k = site_types.get(a)
        if k is None or k not in tasks[v]["eta"]:
            k = max(tasks[v]["eta"], key=lambda kk: tasks[v]["eta"][kk])
        return tasks[v]["eta"][k]

    # ---- link latency per edge ----
    tau_h, tau_v = ic["tau_h"], ic["tau_v"]
    t0, beta = ic["t0"], ic["beta"]
    eps_h, eps_v = ic["eps_h"], ic["eps_v"]

    def link_lat(e):
        ps, pd = tuple(task_site[e["src"]]), tuple(task_site[e["dst"]])
        dh = abs(ps[1] - pd[1]) + abs(ps[2] - pd[2])
        dv = abs(ps[0] - pd[0])
        if dh == 0 and dv == 0:
            return 0.0            # intra-chiplet, no D2D
        return t0 + tau_h * dh + tau_v * dv + e["vol"] / beta

    # ---- resource-constrained list schedule ----
    # Each CHIPLET is a machine that runs its assigned tasks serially (a chiplet
    # cannot execute two tasks at once). This matches ILP constraint C13
    # (chiplet busy-time <= T). We do list scheduling: a task becomes ready when
    # all predecessors finished AND their data has arrived (link latency); among
    # ready tasks contending for the same chiplet, earliest-ready-first, ties by
    # topo order. This is the honest makespan a real run would see.
    succ = {v: [] for v in tasks}
    pred = {v: [] for v in tasks}
    for e in edges:
        succ[e["src"]].append(e)
        pred[e["dst"]].append(e)
    indeg = {v: len(pred[v]) for v in tasks}
    queue = [v for v in tasks if indeg[v] == 0]
    topo = []
    while queue:
        v = queue.pop()
        topo.append(v)
        for e in succ[v]:
            indeg[e["dst"]] -= 1
            if indeg[e["dst"]] == 0:
                queue.append(e["dst"])
    if len(topo) != len(tasks):
        return {"feasible": False, "reasons": ["DAG has a cycle"]}
    topo_rank = {v: r for r, v in enumerate(topo)}

    # chiplet (anchor) each task lives on
    chip = {v: tuple(task_site[v]) for v in tasks}
    chip_free = {}                      # anchor -> time when machine is free
    finish = {}
    start = {}
    remaining = set(tasks)
    # event-driven greedy over topo order with machine availability
    # data-ready time = max over preds of (finish[pred] + link_lat)
    def data_ready(v):
        r = 0.0
        for e in pred[v]:
            if e["src"] not in finish:
                return None            # predecessor not scheduled yet
            r = max(r, finish[e["src"]] + link_lat(e))
        return r

    progressed = True
    while remaining and progressed:
        progressed = False
        # pick all tasks whose preds are done; schedule earliest-ready first
        ready = []
        for v in list(remaining):
            dr = data_ready(v)
            if dr is not None:
                ready.append((dr, topo_rank[v], v))
        ready.sort()
        for dr, _, v in ready:
            if v not in remaining:
                continue
            a = chip[v]
            s = max(dr, chip_free.get(a, 0.0))
            start[v] = s
            finish[v] = s + dur(v)
            chip_free[a] = finish[v]
            remaining.discard(v)
            progressed = True
    if remaining:
        return {"feasible": False, "reasons": ["scheduling deadlock"]}
    makespan = max(finish.values())

    # ---- energy / byte-hop / area / power ----
    e_comm = 0.0
    byte_hop = 0.0
    for e in edges:
        ps, pd = tuple(task_site[e["src"]]), tuple(task_site[e["dst"]])
        dh = abs(ps[1] - pd[1]) + abs(ps[2] - pd[2])
        dv = abs(ps[0] - pd[0])
        e_comm += 8 * e["vol"] * (eps_h * dh + eps_v * dv)
        byte_hop += e["vol"] * (dh + dv)
    e_comp = sum(eta(v) for v in tasks)
    area = sum(lib[k]["area"] for k in [site_types[a] for a in site_types])
    power = sum(lib[k]["p_peak"] for k in [site_types[a] for a in site_types])

    return {
        "feasible": ok, "reasons": reasons,
        "makespan_cycles": makespan,
        "energy_total_J": e_comm + e_comp,
        "energy_comm_J": e_comm, "energy_comp_J": e_comp,
        "byte_hop": byte_hop,
        "total_area_mm2": area, "peak_power_W": power,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Score a placement (sol JSON)")
    ap.add_argument("--params", required=True)
    ap.add_argument("--sol", required=True)
    a = ap.parse_args()
    params = json.load(open(a.params))
    sol = json.load(open(a.sol))
    st = site_types_from_solution(sol)
    ts = sol["task_site"]
    r = evaluate(params, ts, st, sol["grid"], strict=False)
    print(json.dumps(r, indent=2))
    if "makespan_cycles" in r:
        print("\n[eval] REAL makespan = %.1f  (ILP pred = %.1f)"
              % (r["makespan_cycles"], sol.get("predicted_makespan_cycles", -1)))
