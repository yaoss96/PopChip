# -*- coding: utf-8 -*-
"""R2: seed the joint min-T MILP with the PopRank-G packing (a MIP start).
Same budgets/settings as the cold-start Table-III solve.  Usage: CASE_ID"""
from __future__ import print_function
import os, sys, time
import ispa_common as C
import drift_lib as D

INST = {"e008_m06": (8, 6, 1440.0, 1120.0, 45.0, False),
        "e016_m08": (16, 8, 2880.0, 2240.0, 45.0, False),
        "e032_m12": (32, 12, 5760.0, 4480.0, 45.0, False),
        "e064_m12": (64, 12, 11520.0, 8960.0, float(os.environ.get("WARM_TL", "45")), False),
        "e064_m16": (64, 16, 11520.0, 8960.0, 240.0, True),
        "e128_m16": (128, 16, 23040.0, 17920.0, float(os.environ.get("WARM_TL", "45")), False)}
cid = sys.argv[1]
E, X, A, P, TL, geom = INST[cid]
params = C.read_json(os.path.join(C.PARAMS_DIR, "experts_%03d.json" % E))
d = os.path.join(C.RESULTS, "poprank", cid)
seed_path = os.path.join(d, "poprank_greedy_raw.json")
if not os.path.exists(seed_path):
    seed_path = os.path.join(d, "poprank_greedy.json")
seed = C.read_json(seed_path)
T_seed = D.FastEval(params).T(seed)
sol, wall = C.solve(params, X, A, P, TL, geom=geom,
                    extra={"warm_start_solution": seed})
rec = {"case": cid, "seed": seed_path, "T_seed": T_seed, "time_limit": TL,
       "warm": C.summary(sol, wall=wall)}
if sol is not None:
    rec["warm"]["T_fast"] = D.FastEval(params).T(sol)
    C.write_json(os.path.join(d, "popchip_warm.json"), sol)
C.write_json(os.path.join(C.RESULTS, "warm_%s.json" % cid), rec)
print(cid, "seed T=%.1f" % T_seed, "warm T=%s bound=%s gap=%s" % (
    rec["warm"].get("T"), rec["warm"].get("bound"), rec["warm"].get("gap")), flush=True)
