# -*- coding: utf-8 -*-
"""R4: scalability with a richer heterogeneous library (solver-only).

Adds illustrative expert-only chiplet types with replay-scale token-linear
compute profiles (cycles = a + b * tokens; GPU-S/L keep their GPGPU-Sim fits):
  NPU   2x1, 180 mm^2, 110 W, 4 GB,  2500 + 45 n   (systolic expert engine)
  CIM   1x1,  70 mm^2,  25 W, 0.5 GB, 9000 + 150 n (SRAM compute-in-memory)
  GPU-M 1x2, 230 mm^2, 170 W, 4 GB,  5200 + 60 n   (mid-size GPU)
The coefficients are assumptions for a scalability study, not profiles.
Usage: python exp_hetero.py EXPERTS LIBRARY_SIZE
"""
from __future__ import print_function
import copy, json, os, sys, time
import ispa_common as C
import ilp_layout as il

OUT = os.path.join(C.RESULTS, "hetero")
EXTRA = [
    ("NPU", {"area": 180.0, "w": 2, "h": 1, "mem": 4096.0, "p_peak": 110.0,
             "p_static": 15.0, "simlet": "illustrative-npu",
             "proto_cycles": [2500.0, 45.0]}, 1.6),
    ("CIM", {"area": 70.0, "w": 1, "h": 1, "mem": 512.0, "p_peak": 25.0,
             "p_static": 5.0, "simlet": "illustrative-sram-cim",
             "proto_cycles": [9000.0, 150.0]}, 2.4),
    ("GPU-M", {"area": 230.0, "w": 1, "h": 2, "mem": 4096.0, "p_peak": 170.0,
               "p_static": 22.0, "simlet": "illustrative-gpu-m",
               "proto_cycles": [5200.0, 60.0]}, 1.9),
]
MESH = {8: 6, 16: 8, 32: 12}


def library_params(E, nextra):
    p = copy.deepcopy(C.case_params({8: "e008_m06", 16: "e016_m08", 32: "e032_m12"}[E]))
    for name, spec, ratio in EXTRA[:nextra]:
        p["library"][name] = dict(spec)
        for t, v in p["tasks"].items():
            if t.startswith("e("):
                v["allowed"] = list(v["allowed"]) + [name]
                v["cyc"][name] = v["cyc"]["GPU-L"] * ratio
                v["eta"][name] = v["eta"]["GPU-L"]
    return p


def main(E, nextra):
    X = MESH[E]
    p = library_params(E, nextra)
    a_max, p_max = 180.0 * E, 140.0 * E
    b = C.budgets(a_max, p_max)
    o = C.solver_opts(45.0, mipgap=0.01)
    t0 = time.time()
    sol, st = il.build_and_solve(p, 1, X, X, b, o, verbose=False)
    wall = time.time() - t0
    rec = {"experts": E, "types": 3 + nextra, "mesh": X,
           "library": sorted(p["library"]), "wall": wall,
           "num_vars": st.get("num_vars"), "num_bin_vars": st.get("num_bin_vars"),
           "num_constrs": st.get("num_constrs"),
           "num_genconstrs": st.get("num_genconstrs")}
    if sol is not None:
        rec.update(C.summary(sol, wall=wall))
        rec["bom"] = C.counts(sol)
        rec["expert_type"] = {}
        for s in sol["sites"]:
            for t in s["tasks"]:
                if t.startswith("e("):
                    rec["expert_type"][t] = s["type"]
        C.write_json(os.path.join(OUT, "E%d_T%d_solution.json" % (E, 3 + nextra)), sol)
    C.write_json(os.path.join(OUT, "E%d_T%d.json" % (E, 3 + nextra)), rec)
    print(json.dumps({k: rec.get(k) for k in ("experts", "types", "bom", "T", "gap", "ttff", "num_vars", "num_bin_vars", "num_constrs")}), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
