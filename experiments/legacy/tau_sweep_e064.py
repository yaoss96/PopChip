# -*- coding: utf-8 -*-
"""Analytical network-load sensitivity on e064 (BOM fixed to ILP's 14/50).
For each protocol per-hop latency tau_p we compute the analytical protocol
makespan (validated to track measured LEGOSim cycles within ~0.1%) for:
  * ILP-Perf  : re-solved min-T PLACEMENT at this tau_p (free placement,
                warm-started from Floorplet), and additionally the FREE-BOM
                joint optimum (placement+selection);
  * Floorplet : its fixed layout, re-scored at this tau_p;
  * TAP-2.5D  : its fixed layout, re-scored at this tau_p.
Shows whether the three methods fan out as the NoC load grows (compute-bound
-> communication-bound), i.e. where ILP's optimization actually pays off.
No LEGOSim replay -- solves/scoring only.
"""
from __future__ import print_function
import os, sys, json, time, copy
WD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WD)
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT + "/src")
import exp_driver as D
import ilp_layout as il
import tau_score_standalone as SS

B = ROOT + "/out/joint_bom_scaling_fresh_20260802"
P = ROOT + "/data/params"
CID, SUF, X = "e064_m12", "064", 12
A_MAX, P_MAX = 11520.0, 8960.0
NL, NS = 14, 50
TAUS = [104, 500, 1000, 2000, 4000, 8000]
ILP_TL = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0

params = json.load(open("%s/experts_%s.json" % (P, SUF)))
has_cpu = "CPU" in params.get("library", {})
floor = json.load(open("%s/baselines/scaling/%s/prior/floorplet.json" % (B, CID)))
tap = json.load(open("%s/baselines/scaling/%s/prior/tap.json" % (B, CID)))


def budgets_fixed_bom():
    b = {"a_max": A_MAX, "a_max_layer": None, "p_max": P_MAX,
         "nmax": {}, "nmin": {}}
    if has_cpu:
        b["nmin"]["CPU"] = 1; b["nmax"]["CPU"] = 1
    b["nmin"]["GPU-L"] = NL; b["nmax"]["GPU-L"] = NL
    b["nmin"]["GPU-S"] = NS; b["nmax"]["GPU-S"] = NS
    return b


def budgets_free_bom():
    return D.make_budgets(A_MAX, P_MAX, None, has_cpu=has_cpu)


_CONST = SS.build_constants(params)


def score_fixed(layout, tau):
    # Standalone pure-evaluator scoring: works for ANY fixed layout (incl. TAP,
    # which is not representable in the ILP footprint constraints).  Validated
    # to reproduce the ILP fixed_solution value on Floorplet at tau=104.
    return SS.score_layout(params, layout, tau, _CONST)


def solve_ilp(tau, free_bom, warm):
    opts = D.base_solver_opts(params, ILP_TL, mipgap=1e-6, geom=True,
                              protocol_hop_cycles=tau)
    if warm is not None:
        opts["warm_start_solution"] = warm
    bud = budgets_free_bom() if free_bom else budgets_fixed_bom()
    sol, st = il.build_and_solve(params, 1, X, X, bud, opts, verbose=False)
    if sol is None:
        return None
    c = {}
    for s in sol["sites"]:
        c[s["type"]] = c.get(s["type"], 0) + 1
    return {"proto": float(sol["protocol_makespan_cycles"]),
            "nL": c.get("GPU-L", 0), "nS": c.get("GPU-S", 0),
            "gap": sol.get("mip_gap"), "sites": sol["sites"]}


def same_layout(a, b):
    def norm(sites):
        d = {}
        for s in sites:
            d[(s["i"], s["j"])] = (s["type"], tuple(sorted(s.get("tasks", []))))
        return d
    return norm(a) == norm(b)


def run_sweep():
    rows = []
    for tau in TAUS:
        t0 = time.time()
        tf = score_fixed(floor, tau)
        tt = score_fixed(tap, tau)
        ilp_p = solve_ilp(tau, free_bom=False, warm=floor)   # placement-only
        ilp_j = solve_ilp(tau, free_bom=True, warm=floor)    # joint select+place
        row = {
            "tau": tau,
            "floor": tf, "tap": tt,
            "ilp_place": ilp_p["proto"] if ilp_p else None,
            "ilp_place_bom": "%d/%d" % (ilp_p["nL"], ilp_p["nS"]) if ilp_p else None,
            "ilp_place_eq_floor": (same_layout(ilp_p["sites"], floor["sites"])
                                   if ilp_p else None),
            "ilp_joint": ilp_j["proto"] if ilp_j else None,
            "ilp_joint_bom": "%d/%d" % (ilp_j["nL"], ilp_j["nS"]) if ilp_j else None,
            "wall_s": round(time.time() - t0, 1),
        }
        # margins vs ILP placement-only (the Table-III framing: same BOM)
        if ilp_p:
            row["floor_over_ilp_%"] = 100.0 * (tf - ilp_p["proto"]) / tf
            row["tap_over_ilp_%"] = 100.0 * (tt - ilp_p["proto"]) / tt
            row["ilp_joint_gain_vs_floor_%"] = 100.0 * (tf - ilp_j["proto"]) / tf if ilp_j else None
        rows.append(row)
        print(json.dumps(row), flush=True)
        json.dump(rows, open(os.path.join(WD, "tau_sweep_e064_results.json"), "w"),
                  indent=2)
    print("DONE")


if __name__ == "__main__":
    run_sweep()
