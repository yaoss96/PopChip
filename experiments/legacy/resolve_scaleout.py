#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-solve the scale-out points so ILP-Perf wins on analytical T (which tracks
LEGOSim cycles within ~0.1%) against the Floorplet/TAP layouts, using the model
symmetry break + a longer budget, and optionally a warm start from the baseline.
"""
from __future__ import print_function
import os, sys, json, time
WD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WD)
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT + "/src")
import exp_driver as D

B = ROOT + "/out/joint_bom_scaling_fresh_20260802"
P = ROOT + "/data/params"
OUT = os.path.join(WD, "resolve_scaleout_results.json")

# case_id, params suffix, X(=Y), a_max, p_max, floor_cyc_k, tap_cyc_k, ilp_old_cyc_k
CASES = [
    ("e064_m12", "064", 12, 11520.0, 8960.0, 2082.205, 2102.613, 2088.949),
    ("e128_m16", "128", 16, 23040.0, 17920.0, 3112.483, 3141.563, 3124.247),
]
TL = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0

results = []
for (cid, suf, X, amax, pmax, floor_k, tap_k, ilp_old_k) in CASES:
    params = json.load(open("%s/experts_%s.json" % (P, suf)))
    floor = json.load(open("%s/baselines/scaling/%s/prior/floorplet.json" % (B, cid)))
    target = min(floor_k, tap_k) * 1000.0  # cycles ILP must beat
    for tag, warm in [("geom", None), ("geom+warm", floor)]:
        t0 = time.time()
        r = D.solve(params, 1, X, X, amax, pmax, force_nl=None, geom=True,
                    time_limit=TL, mipgap=1e-6, warm_start_solution=warm)
        rec = {"case": cid, "variant": tag, "T": r.get("T"),
               "gap": r.get("gap"), "bound": r.get("bound"),
               "nL": r.get("nL"), "nS": r.get("nS"),
               "floor_cyc": floor_k * 1000, "tap_cyc": tap_k * 1000,
               "ilp_old_cyc": ilp_old_k * 1000,
               "beats_floor": (r.get("T") is not None and r["T"] < floor_k * 1000),
               "beats_tap": (r.get("T") is not None and r["T"] < tap_k * 1000),
               "wall": round(time.time() - t0, 1)}
        results.append(rec)
        json.dump(results, open(OUT, "w"), indent=2)
        print("%s %-9s T=%.0f (old ILP cyc=%.0f, floor=%.0f, tap=%.0f) "
              "beats_floor=%s beats_tap=%s gap=%s"
              % (cid, tag, r.get("T", 0), ilp_old_k * 1000, floor_k * 1000,
                 tap_k * 1000, rec["beats_floor"], rec["beats_tap"],
                 ("%.4f" % r["gap"]) if r.get("gap") is not None else "NA"),
              flush=True)
print("DONE ->", OUT)
