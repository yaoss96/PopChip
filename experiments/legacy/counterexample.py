#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Problem-1 counterexample: an instance where greedy max-L is beaten by the
joint MILP, on the analytical T.  Writes results incrementally."""
from __future__ import print_function
import os, sys, json, time
WD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WD)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")) + "/src")
import exp_driver as D
import workload as W

OUT = os.path.join(WD, "counterexample_results.json")

def gen(experts, groups, out, d_ff=14336):
    p = W.build("mixtral", experts, groups, 2, 4096, d_ff, 256, 2, 0.7)
    json.dump(p, open(out, "w"))
    return p

# (experts, groups, X, Y, a_max, p_max, hops, tl)
CASES = [
    (3, 3, 4, 4, 1350, 1000, [104, 8000, 20000], 30),
    (4, 4, 5, 5, 1500, 1100, [104, 12000, 25000], 30),
]

results = []
for (E, G, X, Y, amax, pmax, hops, tl) in CASES:
    p = gen(E, G, os.path.join(WD, "ce_E%d.json" % E))
    maxl = D.max_feasible_nl(E, amax, pmax)
    for hop in hops:
        rec = {"E": E, "X": X, "Y": Y, "amax": amax, "pmax": pmax,
               "hop": hop, "greedy_nL": maxl}
        t0 = time.time()
        j = D.solve(p, 1, X, Y, amax, pmax, force_nl=None,
                    time_limit=tl, protocol_hop_cycles=hop, mipgap=0.001)
        g = D.solve(p, 1, X, Y, amax, pmax, force_nl=maxl,
                    time_limit=tl, protocol_hop_cycles=hop, mipgap=0.001)
        rec["joint"] = j
        rec["greedy"] = g
        if j.get("feasible") and g.get("feasible") and j.get("T"):
            rec["greedy_over_joint"] = g["T"] / j["T"]
            # rigorous: greedy's own bound vs joint incumbent
            rec["joint_beats_greedy_bound"] = (
                g.get("bound") is not None and j["T"] < g["bound"] - 1e-6)
        rec["wall_s"] = round(time.time() - t0, 1)
        results.append(rec)
        json.dump(results, open(OUT, "w"), indent=2)
        print("E=%d hop=%-6g | joint nL=%s T=%.0f gap=%s | greedy nL=%d T=%.0f gap=%s | g/j=%s"
              % (E, hop, j.get("nL"), j.get("T", 0),
                 ("%.3f" % j["gap"]) if j.get("gap") is not None else "NA",
                 maxl, g.get("T", 0),
                 ("%.3f" % g["gap"]) if g.get("gap") is not None else "NA",
                 ("%.4f" % rec.get("greedy_over_joint", 0))), flush=True)
print("DONE ->", OUT)
