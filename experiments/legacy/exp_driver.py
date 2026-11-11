#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Driver for the reviewer-response Gurobi experiments.

Runs under the bundled Gurobi py37 interpreter.  Imports ilp_layout and calls
build_and_solve directly so we can (a) force a BOM (two-stage control) and
(b) toggle geometric symmetry breaking, all measured on the analytical T.
"""
from __future__ import print_function
import os, sys, json, argparse, copy

SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")) + "/src"
sys.path.insert(0, SRC)
import ilp_layout as il


def base_solver_opts(params, time_limit, mipgap=0.005, seed=1,
                     symmetry_break=True, geom=False, threads=1,
                     protocol_hop_cycles=None):
    opts = {
        "time_limit": time_limit, "mipgap": mipgap, "mipfocus": 1,
        "seed": seed, "symmetry_break": symmetry_break, "mode": "ilp-perf",
        "noc_clock_hz": None, "noc_traffic": "deterministic-dispatch",
        "dispatch_tokens_per_source": 32, "dispatch_hidden": 128,
        "dispatch_bytes_elem": params.get("meta", {}).get("bytes_elem", 2),
        "threads": threads,
        "stage0_protocol_evaluations": None, "stage0_protocol_wall_s": None,
        "stage0_protocol_rounds": None, "cpu_anchor": "free",
        "bom_counts": None, "placement_bundles": None,
        "warm_start_solution": None, "performance_reference_solution": None,
        "reference_source": None,
        # New flag consumed by the patched ilp_layout (ignored if absent):
        "geom_symmetry_break": geom,
    }
    if protocol_hop_cycles is not None:
        opts["protocol_hop_cycles"] = float(protocol_hop_cycles)
    return opts


def make_budgets(a_max, p_max, force_nl=None, has_cpu=True):
    b = {"a_max": a_max, "a_max_layer": None, "p_max": p_max,
         "nmax": {}, "nmin": {}}
    if has_cpu:
        b["nmin"]["CPU"] = 1
        b["nmax"]["CPU"] = 1
    if force_nl is not None:
        b["nmin"]["GPU-L"] = force_nl
        b["nmax"]["GPU-L"] = force_nl
    return b


def solve(params, L, X, Y, a_max, p_max, force_nl=None, geom=False,
          time_limit=45.0, mipgap=0.005, seed=1, symmetry_break=True,
          protocol_hop_cycles=None, warm_start_solution=None):
    budgets = make_budgets(a_max, p_max, force_nl,
                           has_cpu=("CPU" in params.get("library", {})))
    opts = base_solver_opts(params, time_limit, mipgap, seed,
                            symmetry_break, geom,
                            protocol_hop_cycles=protocol_hop_cycles)
    if warm_start_solution is not None:
        opts["warm_start_solution"] = warm_start_solution
    sol, stats = il.build_and_solve(params, L, X, Y, budgets, opts,
                                    verbose=False)
    if sol is None:
        return {"feasible": False, "status": stats.get("status")}
    counts = {}
    for st in sol["sites"]:
        counts[st["type"]] = counts.get(st["type"], 0) + 1
    return {
        "feasible": True,
        "nL": counts.get("GPU-L", 0), "nS": counts.get("GPU-S", 0),
        "T": sol["predicted_makespan_cycles"],
        "proto": sol.get("protocol_makespan_cycles"),
        "bound": sol.get("primary_objective_bound_cycles"),
        "gap": sol.get("mip_gap"),
        "wall": sol.get("solver_wall_s"),
        "ttff": sol.get("time_to_first_feasible_s"),
        "nodes": sol.get("primary_node_count"),
        "area": sol["total_area_mm2"], "power": sol["peak_power_W"],
        "status": sol.get("solver_status"),
    }


def max_feasible_nl(E, a_max, p_max, aL=400, aS=110, aCPU=90,
                    pL=300, pS=90, pCPU=65):
    """Greedy 'buy as many GPU-L as the budget allows', with E expert GPUs."""
    best = 0
    for nl in range(0, E + 1):
        area = aCPU + nl * aL + (E - nl) * aS
        power = pCPU + nl * pL + (E - nl) * pS
        if area <= a_max and power <= p_max:
            best = nl
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--L", type=int, default=1)
    ap.add_argument("--X", type=int, default=6)
    ap.add_argument("--Y", type=int, default=6)
    ap.add_argument("--a-max", type=float, required=True)
    ap.add_argument("--p-max", type=float, required=True)
    ap.add_argument("--experts", type=int, required=True)
    ap.add_argument("--time-limit", type=float, default=45.0)
    ap.add_argument("--geom", action="store_true")
    ap.add_argument("--twostage", action="store_true",
                    help="also solve greedy-max-L (forced BOM)")
    a = ap.parse_args()
    with open(a.params) as fh:
        params = json.load(fh)
    print("== JOINT (free BOM) ==")
    j = solve(params, a.L, a.X, a.Y, a.a_max, a.p_max, force_nl=None,
              geom=a.geom, time_limit=a.time_limit)
    print(json.dumps(j, indent=2))
    if a.twostage:
        maxl = max_feasible_nl(a.experts, a.a_max, a.p_max)
        print("== TWO-STAGE greedy-max-L (nL=%d forced) ==" % maxl)
        g = solve(params, a.L, a.X, a.Y, a.a_max, a.p_max, force_nl=maxl,
                  geom=a.geom, time_limit=a.time_limit)
        print(json.dumps(g, indent=2))
        if j.get("feasible") and g.get("feasible"):
            print(">> joint nL=%d T=%.1f | greedy nL=%d T=%.1f | greedy/joint=%.4f"
                  % (j["nL"], j["T"], g["nL"], g["T"], g["T"] / j["T"]))
