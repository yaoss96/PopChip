from __future__ import print_function
import ispa_common as C, drift_lib as D
for cid in ["e008_m06", "e016_m08", "e032_m12"]:
    E = C.CASES[cid][0]
    p0 = C.case_params(cid)
    p1 = D.build_params(E, D.zipf(E, 0.7), p0["meta"]["config"])
    same = (p0["tasks"] == p1["tasks"] and p0["edges"] == p1["edges"] and p0["meta"]["group_pi"] == p1["meta"]["group_pi"])
    ev = D.FastEval(p0)
    for name in ["popchip", "floorplet", "tap", "blind1"]:
        sol = C.read_json(C.case_paths(cid)[name])
        print(cid, "params-identical=%s" % same, name, "fast=%.1f" % ev.T(sol), "milp=%.1f" % C.evaluate_fixed(p0, sol))
