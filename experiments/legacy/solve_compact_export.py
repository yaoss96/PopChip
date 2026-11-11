#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Warm-start from the Floorplet layout, solve the joint min-T model, then run
the makespan-preserving byte-hop compaction stage, and SAVE THE FULL solution
JSON so it can be codegen'd + replayed in LEGOSim.  Reports the analytical
protocol makespan (tracks measured cycles to ~0.1%) before/after compaction.
"""
from __future__ import print_function
import os, sys, json, time
WD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WD)
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT + "/src")
import exp_driver as D
import ilp_layout as il

B = ROOT + "/out/joint_bom_scaling_fresh_20260802"
P = ROOT + "/data/params"
OUTDIR = os.path.join(WD, "compact_solutions")
if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)

# cid, suf, X, a_max, p_max, floor_measured_cyc, tap_measured_cyc
CASES = {
    "e008_m06": ("008", 6, 1440.0, 1120.0, 111744, 113013),
    "e016_m08": ("016", 8, 2880.0, 2240.0, 309398, 310237),
    "e032_m12": ("032", 12, 5760.0, 4480.0, 1077591, 1085736),
    "e064_m12": ("064", 12, 11520.0, 8960.0, 2082205, 2102613),
    "e128_m16": ("128", 16, 23040.0, 17920.0, 3112483, 3141563),
}


def run(cid, tl_primary, tl_tie, warm=True, tiebreak=True, out=None):
    suf, X, amax, pmax, floor_c, tap_c = CASES[cid]
    params = json.load(open("%s/experts_%s.json" % (P, suf)))
    floor = None
    if warm:
        floor = json.load(open(
            "%s/baselines/scaling/%s/prior/floorplet.json" % (B, cid)))
    budgets = D.make_budgets(amax, pmax, None,
                             has_cpu=("CPU" in params.get("library", {})))
    opts = D.base_solver_opts(params, tl_primary, mipgap=1e-6, geom=True)
    opts["warm_start_solution"] = floor
    opts["byte_hop_tiebreak"] = bool(tiebreak)
    opts["byte_hop_time_limit"] = tl_tie
    opts["byte_hop_mipgap"] = 1e-4
    t0 = time.time()
    sol, stats = il.build_and_solve(params, 1, X, X, budgets, opts,
                                    verbose=False)
    wall = time.time() - t0
    if sol is None:
        print("%s INFEASIBLE status=%s" % (cid, stats.get("status")))
        return None
    counts = {}
    for st in sol["sites"]:
        counts[st["type"]] = counts.get(st["type"], 0) + 1
    tb = sol.get("byte_hop_tiebreak", {})
    rec = {
        "case": cid,
        "predicted_makespan_cycles": sol["predicted_makespan_cycles"],
        "protocol_makespan_cycles": sol["protocol_makespan_cycles"],
        "primary_incumbent_cycles": sol.get("primary_incumbent_cycles"),
        "bound_cycles": sol.get("primary_objective_bound_cycles"),
        "byte_hop": sol.get("byte_hop"),
        "tiebreak": tb,
        "nL": counts.get("GPU-L", 0), "nS": counts.get("GPU-S", 0),
        "floor_measured": floor_c, "tap_measured": tap_c,
        "wall_s": round(wall, 1),
    }
    if out is None:
        out = os.path.join(OUTDIR, "%s_ilp_compact.json" % cid)
    json.dump(sol, open(out, "w"), indent=2)
    rec["solution_path"] = out
    proto = sol["protocol_makespan_cycles"]
    print("%s proto_makespan=%.0f (primary_T=%s) floor_meas=%d tap_meas=%d "
          "| proto<floor? %s  byte_hop=%.3e  tiebreak=%s wall=%.0fs"
          % (cid, proto, ("%.0f" % rec["primary_incumbent_cycles"])
             if rec["primary_incumbent_cycles"] else "NA",
             floor_c, tap_c, proto < floor_c, sol.get("byte_hop", 0.0),
             tb.get("enabled"), wall), flush=True)
    return rec


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "e008_m06"
    tlp = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    tlt = float(sys.argv[3]) if len(sys.argv) > 3 else 45.0
    r = run(cid, tlp, tlt)
    print(json.dumps(r, indent=2))
