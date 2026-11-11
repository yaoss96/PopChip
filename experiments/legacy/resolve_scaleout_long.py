#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Longer warm-started re-solve of the two large scale-out points so ILP-Perf
does not merely edge past Floorplet/TAP but leads by a margin that clears the
analytical-surrogate error band (~0.35%).  Warm start = Floorplet layout, then
Gurobi improves.  Writes incrementally so we can watch the incumbent fall.
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
OUT = os.path.join(WD, "resolve_scaleout_long_results.json")

# case_id, params suffix, X(=Y), a_max, p_max, floor_cyc, tap_cyc, ilp_old_cyc
CASES = [
    ("e064_m12", "064", 12, 11520.0, 8960.0, 2082205.0, 2102613.0, 2088949.0),
    ("e128_m16", "128", 16, 23040.0, 17920.0, 3112483.0, 3141563.0, 3124247.0),
]
TL = float(sys.argv[1]) if len(sys.argv) > 1 else 1200.0

results = []
for (cid, suf, X, amax, pmax, floor_c, tap_c, ilp_old_c) in CASES:
    params = json.load(open("%s/experts_%s.json" % (P, suf)))
    floor = json.load(open("%s/baselines/scaling/%s/prior/floorplet.json" % (B, cid)))
    t0 = time.time()
    r = D.solve(params, 1, X, X, amax, pmax, force_nl=None, geom=True,
                time_limit=TL, mipgap=1e-6, warm_start_solution=floor)
    T = r.get("T")
    rec = {"case": cid, "variant": "geom+warm-long", "TL": TL, "T": T,
           "gap": r.get("gap"), "bound": r.get("bound"),
           "nL": r.get("nL"), "nS": r.get("nS"),
           "floor_cyc": floor_c, "tap_cyc": tap_c, "ilp_old_cyc": ilp_old_c,
           "margin_vs_floor_pct": (None if T is None else 100.0 * (floor_c - T) / floor_c),
           "margin_vs_tap_pct": (None if T is None else 100.0 * (tap_c - T) / tap_c),
           "beats_floor": (T is not None and T < floor_c),
           "beats_tap": (T is not None and T < tap_c),
           "wall": round(time.time() - t0, 1)}
    results.append(rec)
    json.dump(results, open(OUT, "w"), indent=2)
    print("%s T=%.0f floor=%.0f (margin %.3f%%) tap=%.0f (margin %.3f%%) gap=%s"
          % (cid, T or 0, floor_c, rec["margin_vs_floor_pct"] or 0,
             tap_c, rec["margin_vs_tap_pct"] or 0,
             ("%.4f" % r["gap"]) if r.get("gap") is not None else "NA"),
          flush=True)
print("DONE ->", OUT)
