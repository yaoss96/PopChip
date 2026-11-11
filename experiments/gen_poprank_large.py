# -*- coding: utf-8 -*-
"""PopRank-G layouts (greedy max-L BOM, rank map, compact packing) for the
large instances, scored with the exact P1 evaluator (no MILP build)."""
from __future__ import print_function
import os, sys
import ispa_common as C
import drift_lib as D
import exp_poprank as PR

INST = {"e064_m12": (64, 12, 11520.0, 8960.0), "e064_m16": (64, 16, 11520.0, 8960.0),
        "e128_m16": (128, 16, 23040.0, 17920.0)}
for cid, (E, X, A, P) in sorted(INST.items()):
    params = C.read_json(os.path.join(C.PARAMS_DIR, "experts_%03d.json" % E))
    nL = PR.greedy_nl(E, A, P)
    pack = PR.compact_packing(params, X, PR.poprank_bundles(params, nL), A, P)
    T = D.FastEval(params).T(pack)
    out = os.path.join(C.RESULTS, "poprank", cid, "poprank_greedy_raw.json")
    C.write_json(out, pack)
    print(cid, "nL=%d" % nL, "T=%.1f" % T, out, flush=True)
