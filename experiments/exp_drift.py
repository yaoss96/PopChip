# -*- coding: utf-8 -*-
"""R3/R4: robustness of a PopChip package to popularity drift.

The BOM and die anchors are design-time decisions; the expert-to-die map is a
run-time decision.  For each Table-III package designed for Zipf(0.7), drift
the popularity and compare, on the analytical T:
  static     : keep the design-time map;
  remap-rank : rebind experts to the same dies by their new rank (no hardware
               change; die that served design rank r serves new rank r);
  remap-ls   : remap-rank + pairwise-swap local search on the exact protocol;
  redesign   : a fresh joint min-T solve for the drifted popularity (new
               hardware; reference point).
Usage: python exp_drift.py CASE_ID
"""
from __future__ import print_function
import json, os, random, sys, time
import ispa_common as C
import drift_lib as D

OUT = os.path.join(C.RESULTS, "drift")


def drifts(E):
    base = D.zipf(E, 0.7)
    out = {"a0.3": D.zipf(E, 0.3), "a1.0": D.zipf(E, 1.0),
           "a1.2": D.zipf(E, 1.2)}
    for s in (1, 2, 3):
        rng = random.Random(100 + s)
        v = list(base)
        for r in range(0, E - 1, 2):
            if rng.random() < 0.5:
                v[r], v[r + 1] = v[r + 1], v[r]
        out["mild_s%d" % s] = v
    for s in (1, 2, 3):
        rng = random.Random(200 + s)
        perm = list(range(E))
        rng.shuffle(perm)
        out["shuffle_s%d" % s] = [base[perm[g]] for g in range(E)]
    return out


def main(cid):
    E, X, A, P, TL = C.CASES[cid]
    params_A = C.case_params(cid)
    pi_A = list(params_A["meta"]["group_pi"])
    popchip = C.read_json(C.case_paths(cid)["popchip"])
    res = {"case": cid, "T_design": D.FastEval(params_A).T(popchip), "drift": {}}
    path = os.path.join(OUT, "%s.json" % cid)
    if os.path.exists(path):
        res = C.read_json(path)
    for name, pi_B in sorted(drifts(E).items()):
        if name in res["drift"]:
            continue
        pB = D.build_params(E, pi_B, params_A["meta"]["config"])
        ev = D.FastEval(pB)
        rec = {"pi": pi_B}
        rec["static"] = ev.T(popchip)
        remap = D.rank_remap(popchip, pi_A, pi_B)
        rec["remap_rank"] = ev.T(remap)
        ls, t_ls, evals, wall = D.swap_search(remap, ev, wall=60.0)
        rec["remap_ls"] = t_ls
        rec["remap_ls_evals"] = evals
        rec["remap_ls_wall"] = wall
        sol, w = C.solve(pB, X, A, P, TL, geom=(cid == "e064_m16"))
        rec["redesign"] = C.summary(sol, wall=w)
        C.write_json(os.path.join(OUT, cid, name + "_remap_rank.json"), remap)
        C.write_json(os.path.join(OUT, cid, name + "_remap_ls.json"), ls)
        C.write_json(os.path.join(OUT, cid, name + "_params.json"), pB)
        if sol is not None:
            C.write_json(os.path.join(OUT, cid, name + "_redesign.json"), sol)
        res["drift"][name] = rec
        C.write_json(path, res)
        print(cid, name, "static=%.0f remap=%.0f ls=%.0f redesign=%s" % (
            rec["static"], rec["remap_rank"], rec["remap_ls"],
            rec["redesign"].get("T")), flush=True)
    print("DONE", cid, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
