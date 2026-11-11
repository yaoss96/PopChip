#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Problem-3: effect of model-level geometric symmetry breaking on the
residual gap, at the paper's scale-out caps (A=180E, P=140E)."""
from __future__ import print_function
import os, sys, json, time
WD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WD)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")) + "/src")
import exp_driver as D
import workload as W

OUT = os.path.join(WD, "symbreak_results.json")
TL = 45.0
# (experts, groups, mesh X=Y) ; caps = 180E / 140E
CASES = [(8, 8, 6), (16, 16, 8), (32, 16, 12)]

results = []
for (E, G, X) in CASES:
    p = W.build("mixtral", E, G, 2, 4096, 14336, 256, 2, 0.7)
    amax, pmax = 180.0 * E, 140.0 * E
    for geom in (False, True):
        t0 = time.time()
        r = D.solve(p, 1, X, X, amax, pmax, force_nl=None, geom=geom,
                    time_limit=TL, mipgap=1e-6)
        rec = {"E": E, "X": X, "amax": amax, "pmax": pmax, "geom": geom,
               "nL": r.get("nL"), "nS": r.get("nS"), "T": r.get("T"),
               "bound": r.get("bound"), "gap": r.get("gap"),
               "ttff": r.get("ttff"), "nodes": r.get("nodes"),
               "wall": r.get("wall"), "wall_measured": round(time.time()-t0, 1)}
        results.append(rec)
        json.dump(results, open(OUT, "w"), indent=2)
        print("E=%-3d geom=%-5s | nL/nS=%s/%s gap=%s bound=%s ttff=%s nodes=%s"
              % (E, geom, r.get("nL"), r.get("nS"),
                 ("%.4f" % r["gap"]) if r.get("gap") is not None else "NA",
                 ("%.0f" % r["bound"]) if r.get("bound") is not None else "None",
                 r.get("ttff"), r.get("nodes")), flush=True)
print("DONE ->", OUT)
