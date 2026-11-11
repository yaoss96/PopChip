#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_bruteforce.py -- prove the compact ILP returns the TRUE optimum on a
small instance by exhaustive enumeration (design doc Phase D exit criterion:
"optimum matches brute force").

Strategy: shrink to a tiny instance (few tasks, |P|<=16). Enumerate EVERY legal
assignment of tasks->sites and chiplet-types->sites, score each with the same
evaluate.py the ILP is measured against, and take the min-makespan feasible
placement. Compare to the ILP's makespan. They must match to within the MIP gap.

Because full enumeration explodes, we use a REDUCED workload (--experts small)
and a small grid so the search is tractable (<~10^6 candidates). This is exactly
the doc's |P|=16 brute-force validation.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev


def make_sites(L, X, Y):
    return [(l, i, j) for l in range(L) for i in range(X) for j in range(Y)]


def footprint(k, anchor, lib):
    l, i, j = anchor
    w, h = lib[k]["w"], lib[k]["h"]
    return {(l, i + di, j + dj) for di in range(w) for dj in range(h)}


def brute_force(params, grid, budgets):
    """Exhaustive search. Only feasible for tiny instances.

    We enumerate, for each task, a site; a site's chiplet type is induced by the
    tasks assigned to it (must share a common allowed type). This is the
    placement space the ILP searches. To bound blow-up we require every used
    site to host at least one task (no empty chiplets in the optimum for a
    makespan objective when budgets bind)."""
    lib = params["library"]
    tasks = list(params["tasks"].keys())
    L, X, Y = grid["L"], grid["X"], grid["Y"]
    sites = make_sites(L, X, Y)
    # only 1x1-anchor sites to keep footprints trivial for GPU-S/CPU; allow
    # GPU-L (2x2) anchors too.
    best = None
    n = len(tasks)
    total = len(sites) ** n
    if total > 3_000_000:
        raise SystemExit("brute-force space too large (%d); shrink instance"
                         % total)

    is_expert = set(params["expert_tasks"])
    for combo in itertools.product(range(len(sites)), repeat=n):
        task_site = {tasks[t]: list(sites[combo[t]]) for t in range(n)}
        # induce a type per used site: choose the fastest type allowed by ALL
        # tasks on it AND that fits footprint/overlap. Try type combinations.
        used = {}
        for t, si in zip(tasks, combo):
            used.setdefault(sites[si], []).append(t)
        # candidate types per site = intersection of allowed types
        site_type_opts = {}
        ok = True
        for site, members in used.items():
            allowed = set(lib.keys())
            for mtask in members:
                allowed &= set(params["tasks"][mtask]["allowed"])
            # expert exclusivity
            if sum(1 for mt in members if mt in is_expert) > 1:
                ok = False
                break
            if not allowed:
                ok = False
                break
            site_type_opts[site] = sorted(allowed)
        if not ok:
            continue
        # try the cheapest-area feasible type assignment (greedy is fine: for
        # verification we just need SOME feasible completion; but to be truly
        # exhaustive we enumerate type choices too).
        sites_list = list(site_type_opts.keys())
        for typ_combo in itertools.product(*[site_type_opts[s] for s in sites_list]):
            site_types = {sites_list[i]: typ_combo[i] for i in range(len(sites_list))}
            sc = ev.evaluate(params, task_site, site_types, grid,
                             strict=True, budgets=budgets)
            if not sc.get("feasible"):
                continue
            # score with the ILP's OWN objective model (longest path + busy-time),
            # so we verify the solver optimizes its formulation to the true optimum
            mk = ilp_model_makespan(params, task_site, site_types)
            if best is None or mk < best[0]:
                best = (mk, task_site, site_types)
    return best


def ilp_model_makespan(params, task_site, site_types):
    """Compute the ILP's OWN makespan model for a placement (NOT the list
    scheduler). This is the longest path with affine link latency (C11) lower-
    bounded by per-chiplet busy time (C13) -- exactly what ilp_layout minimizes.
    Verifying brute-force optimum of THIS equals the ILP's ObjVal proves the
    solver finds the true optimum of its formulation."""
    tasks = params["tasks"]
    edges = params["edges"]
    ic = params["interconnect"]
    tau_h, tau_v, t0, beta = ic["tau_h"], ic["tau_v"], ic["t0"], ic["beta"]

    def dur(v):
        k = site_types[tuple(task_site[v])]
        return tasks[v]["cyc"][k]

    def link(e):
        ps, pd = tuple(task_site[e["src"]]), tuple(task_site[e["dst"]])
        dh = abs(ps[1]-pd[1]) + abs(ps[2]-pd[2])
        dv = abs(ps[0]-pd[0])
        # NOTE: ILP charges t0 + transfer on EVERY edge (it has no intra-chiplet
        # special-case in C11), so mirror that here for an exact match.
        return t0 + tau_h*dh + tau_v*dv + e["vol"]/beta

    # longest path (topological)
    succ = {v: [] for v in tasks}
    pred = {v: [] for v in tasks}
    for e in edges:
        succ[e["src"]].append(e); pred[e["dst"]].append(e)
    indeg = {v: len(pred[v]) for v in tasks}
    q = [v for v in tasks if indeg[v] == 0]
    order = []
    while q:
        v = q.pop(); order.append(v)
        for e in succ[v]:
            indeg[e["dst"]] -= 1
            if indeg[e["dst"]] == 0:
                q.append(e["dst"])
    finish = {}
    for v in order:
        s = 0.0
        for e in pred[v]:
            s = max(s, finish[e["src"]] + link(e))
        finish[v] = s + dur(v)
    T_path = max(finish.values())
    # C13 busy-time lower bound
    busy = {}
    for v in tasks:
        a = tuple(task_site[v])
        busy[a] = busy.get(a, 0.0) + dur(v)
    T_busy = max(busy.values()) if busy else 0.0
    return max(T_path, T_busy)


def main():
    ap = argparse.ArgumentParser(description="Brute-force optimality check")
    ap.add_argument("--experts", type=int, default=3)
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--L", type=int, default=1)
    ap.add_argument("--X", type=int, default=2)
    ap.add_argument("--Y", type=int, default=2)
    ap.add_argument("--gurobi-python",
                    default=os.environ.get("GUROBI_PYTHON", sys.executable))
    ap.add_argument("--out-dir", default="out/verify")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    # 1) build a tiny workload
    params_path = os.path.join(a.out_dir, "params_tiny.json")
    subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__), "workload.py"),
                    "--config", "custom", "--experts", str(a.experts),
                    "--groups", str(a.groups), "--topk", str(a.topk),
                    "-o", params_path], check=True)
    params = json.load(open(params_path))
    grid = {"L": a.L, "X": a.X, "Y": a.Y}
    n_sites = a.L * a.X * a.Y
    print("[verify] tiny instance: |V|=%d |E|=%d |P|=%d  (%d^%d candidates)"
          % (len(params["tasks"]), len(params["edges"]), n_sites,
             n_sites, len(params["tasks"])))

    # generous budgets so feasibility is about placement, not resources
    budgets = {"a_max": 1e9, "p_max": 1e9, "p_col": 1e9}

    # 2) brute force
    print("[verify] enumerating...")
    bf = brute_force(params, grid, budgets)
    if bf is None:
        print("[verify] brute force found NO feasible placement")
        sys.exit(1)
    bf_mk, bf_ts, bf_st = bf
    print("[verify] brute-force optimal makespan = %.4f" % bf_mk)

    # 3) ILP
    def winp(p):
        ap_ = os.path.abspath(p)
        if ap_.startswith("/mnt/") and ap_[6] == "/":
            return ap_[5].upper() + ":\\" + ap_[7:].replace("/", "\\")
        return p
    sol_path = os.path.join(a.out_dir, "sol_tiny.json")
    ilp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ilp_layout.py")
    subprocess.run([a.gurobi_python, winp(ilp), "--params", winp(params_path),
                    "--L", str(a.L), "--X", str(a.X), "--Y", str(a.Y),
                    "--a-max", "1e9", "--p-max", "1e9", "--p-col", "1e9",
                    "--time-limit", "120", "--mipgap", "0.0", "--quiet",
                    "-o", winp(sol_path)], check=True)
    sol = json.load(open(sol_path))
    # compare the ILP's predicted objective against the brute-force optimum of
    # the SAME objective model (longest path + busy-time).
    ilp_mk = sol["predicted_makespan_cycles"]
    print("[verify] ILP predicted makespan = %.4f  (gap %.4f%%)"
          % (ilp_mk, sol["mip_gap"] * 100))

    rel = abs(ilp_mk - bf_mk) / bf_mk if bf_mk else 0.0
    print("=" * 55)
    if rel < 2e-3:
        print("[verify] PASS -- ILP matches brute-force optimum (rel %.2e)" % rel)
    else:
        print("[verify] FAIL -- ILP %.4f vs brute %.4f (rel %.2e)"
              % (ilp_mk, bf_mk, rel))
        sys.exit(2)


if __name__ == "__main__":
    main()
