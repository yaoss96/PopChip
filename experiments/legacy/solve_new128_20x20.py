# -*- coding: utf-8 -*-
"""Solve the NEW case: 128 experts on a 20x20 Mesh, A_max=23040, P_max=17920.
Per-expert budget 180/140 (same scaling as the 8/16/32/64 rows).
Exports the full ILP-Perf solution JSON (replay-ready) and reports the BOM.
"""
from __future__ import print_function
import os, sys, json, time
WD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WD)
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT + "/src")
import exp_driver as D
import ilp_layout as il

P = ROOT + "/data/params"
OUTDIR = os.path.join(WD, "new128_20x20")
if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)

X = 20
TL = float(sys.argv[1]) if len(sys.argv) > 1 else 900.0
AMAX = float(sys.argv[2]) if len(sys.argv) > 2 else 23040.0
PMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 17920.0

params = json.load(open("%s/experts_128.json" % P))
budgets = D.make_budgets(AMAX, PMAX, None,
                         has_cpu=("CPU" in params.get("library", {})))
opts = D.base_solver_opts(params, TL, mipgap=1e-6, geom=True)
opts["byte_hop_tiebreak"] = True
opts["byte_hop_time_limit"] = 90.0
opts["byte_hop_mipgap"] = 1e-4

t0 = time.time()
sol, stats = il.build_and_solve(params, 1, X, X, budgets, opts, verbose=False)
wall = time.time() - t0
if sol is None:
    print("INFEASIBLE status=%s" % stats.get("status"))
    sys.exit(1)
counts = {}
for st in sol["sites"]:
    counts[st["type"]] = counts.get(st["type"], 0) + 1
out = os.path.join(OUTDIR, "ilp_perf.json")
json.dump(sol, open(out, "w"), indent=2)
print("BOM: GPU-L=%d GPU-S=%d CPU=%d  nsites=%d"
      % (counts.get("GPU-L", 0), counts.get("GPU-S", 0),
         counts.get("CPU", 0), len(sol["sites"])))
print("proto_makespan=%.0f predicted=%.0f gap=%s wall=%.0fs area_used=%.0f pow_used=%.0f"
      % (sol["protocol_makespan_cycles"], sol["predicted_makespan_cycles"],
         ("%.4f" % sol.get("mip_gap")) if sol.get("mip_gap") is not None else "NA",
         wall, sol.get("total_area_mm2", 0), sol.get("peak_power_W", 0)))
print("saved ->", out)
