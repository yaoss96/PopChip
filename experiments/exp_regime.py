# -*- coding: utf-8 -*-
"""R1/R3: does joint BOM selection beat greedy max-L / BOM enumeration in
realistic regimes?  8-expert 6x6 instance, non-boundary caps, per-hop latency
and link-bandwidth sweeps.  Analytical T only (same model as the paper).

  joint : free BOM, one min-T solve
  enum  : one min-T solve per feasible N_L (N_S free); best of them
  greedy: the enumeration member with the largest feasible N_L
Usage: python exp_regime.py WORKER_ID N_WORKERS
"""
from __future__ import print_function
import copy, itertools, json, os, sys, time
import ispa_common as C

OUT = os.path.join(C.RESULTS, "regime")
E, X = 8, 6
TL = 45.0
# (label, A_max, P_max): area-limited non-boundary caps (power relaxed), the
# scale-out envelope, and a power-limited non-boundary cap (area relaxed)
CAPS = [("A1115", 1115.0, 2500.0), ("A1405", 1405.0, 2500.0),
        ("A1695", 1695.0, 2500.0), ("A1985", 1985.0, 2500.0),
        ("S1440", 1440.0, 1120.0), ("P1100", 2500.0, 1100.0)]
HOPS = [104.0, 520.0, 1040.0, 2080.0]
BW = [1.0, 0.25]


def max_nl(a_max, p_max):
    best = 0
    for nl in range(E + 1):
        if (90 + 400 * nl + 110 * (E - nl) <= a_max and
                65 + 300 * nl + 90 * (E - nl) <= p_max):
            best = nl
    return best


def run(cfg):
    (lab, a, p), hop, bw = cfg
    key = "%s_h%d_bw%g" % (lab, hop, bw)
    path = os.path.join(OUT, key + ".json")
    if os.path.exists(path):
        return
    params = copy.deepcopy(C.case_params("e008_m06"))
    params["interconnect"]["beta"] = 16.0 * bw
    kw = dict(hop=hop, flit=32.0 * bw)
    rec = {"cap": lab, "a_max": a, "p_max": p, "hop": hop, "bw": bw,
           "max_nL": max_nl(a, p)}
    sol, wall = C.solve(params, X, a, p, TL, **kw)
    rec["joint"] = C.summary(sol, wall=wall)
    rec["enum"] = {}
    for nl in range(rec["max_nL"] + 1):
        s2, w2 = C.solve(params, X, a, p, TL, nmin={"GPU-L": nl},
                         nmax={"GPU-L": nl}, **kw)
        rec["enum"][str(nl)] = C.summary(s2, wall=w2)
    feas = [(v["T"], int(k)) for k, v in rec["enum"].items() if v.get("feasible")]
    rec["enum_best_nL"] = min(feas)[1] if feas else None
    rec["enum_best_T"] = min(feas)[0] if feas else None
    g = rec["enum"].get(str(rec["max_nL"]), {})
    rec["greedy_T"] = g.get("T")
    rec["greedy_bound"] = g.get("bound")
    C.write_json(path, rec)
    print(key, "joint nL=%s T=%.0f | greedy T=%s | enum best nL=%s T=%s" % (
        rec["joint"].get("nL"), rec["joint"].get("T") or -1,
        rec["greedy_T"], rec["enum_best_nL"], rec["enum_best_T"]), flush=True)


def main(worker, nworkers):
    cfgs = list(itertools.product(CAPS, HOPS, BW))
    for i, cfg in enumerate(cfgs):
        if i % nworkers == worker:
            run(cfg)
    print("WORKER DONE", worker, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
