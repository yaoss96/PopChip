# -*- coding: utf-8 -*-
"""Model size (variables / binaries / constraints) of every Table-III instance."""
from __future__ import print_function
import json, os, sys, time
import ispa_common as C
import ilp_layout as il

OUT = os.path.join(C.RESULTS, "modelsize.json")
INST = [("e008_m06", 8, 6, 1440.0, 1120.0), ("e016_m08", 16, 8, 2880.0, 2240.0),
        ("e032_m12", 32, 12, 5760.0, 4480.0), ("e064_m12", 64, 12, 11520.0, 8960.0),
        ("e064_m16", 64, 16, 11520.0, 8960.0), ("e128_m16", 128, 16, 23040.0, 17920.0)]
res = C.read_json(OUT) if os.path.exists(OUT) else {}
for cid, E, X, A, P in INST:
    if cid in res:
        continue
    p = C.read_json(os.path.join(C.PARAMS_DIR, "experts_%03d.json" % E))
    t0 = time.time()
    sol, st = il.build_and_solve(p, 1, X, X, C.budgets(A, P), C.solver_opts(1.0), verbose=False)
    res[cid] = {k: st.get(k) for k in ("num_vars", "num_bin_vars", "num_constrs", "num_genconstrs", "model_build_s")}
    res[cid]["wall"] = time.time() - t0
    C.write_json(OUT, res)
    print(cid, res[cid], flush=True)
