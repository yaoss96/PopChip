#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anytime evidence: residual gap vs solver time budget (16-expert instance,
symmetry break on).  Shows the 45 s snapshot is arbitrary and minutes-level
budgets close more of the gap."""
from __future__ import print_function
import os, sys, json, time
WD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WD)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")) + "/src")
import exp_driver as D
import workload as W

OUT = os.path.join(WD, "anytime_results.json")
p = W.build("mixtral", 16, 16, 2, 4096, 14336, 256, 2, 0.7)
amax, pmax = 180.0 * 16, 140.0 * 16
res = []
for tl in [15.0, 45.0, 120.0, 300.0]:
    t0 = time.time()
    r = D.solve(p, 1, 8, 8, amax, pmax, force_nl=None, geom=True,
                time_limit=tl, mipgap=1e-6)
    res.append({"tl": tl, "gap": r.get("gap"), "T": r.get("T"),
                "bound": r.get("bound"), "wall": round(time.time()-t0, 1)})
    json.dump(res, open(OUT, "w"), indent=2)
    print("tl=%-5g gap=%s T=%.0f bound=%.0f wall=%.1f"
          % (tl, ("%.4f" % r["gap"]) if r.get("gap") is not None else "NA",
             r.get("T", 0), r.get("bound") or 0, time.time()-t0), flush=True)
print("DONE ->", OUT)
