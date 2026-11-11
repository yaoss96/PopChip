# -*- coding: utf-8 -*-
"""Does a longer cold-start joint solve find the smaller-L BOM that BOM
enumeration finds in the latency-dominated regime?  Usage: CAP HOP BW TL"""
from __future__ import print_function
import copy, os, sys
import ispa_common as C
import exp_regime as R

lab, hop, bw, tl = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
nl = int(sys.argv[5]) if len(sys.argv) > 5 else None
cap = [c for c in R.CAPS if c[0] == lab][0]
params = copy.deepcopy(C.case_params("e008_m06"))
params["interconnect"]["beta"] = 16.0 * bw
kw = {} if nl is None else {"nmin": {"GPU-L": nl}, "nmax": {"GPU-L": nl}}
sol, wall = C.solve(params, 6, cap[1], cap[2], tl, hop=hop, flit=32.0 * bw, **kw)
rec = {"cap": lab, "hop": hop, "bw": bw, "time_limit": tl, "forced_nL": nl, "joint": C.summary(sol, wall=wall)}
C.write_json(os.path.join(C.RESULTS, "regime_long", "%s_h%d_bw%g_tl%d%s.json" % (lab, hop, bw, tl, "" if nl is None else "_nL%d" % nl)), rec)
print(rec["joint"]["nL"], rec["joint"]["T"], rec["joint"]["gap"], flush=True)
