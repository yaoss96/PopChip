# -*- coding: utf-8 -*-
"""Decisive test: can ILP beat Floorplet (not just TAP) in a high-volume,
communication-bound regime where the directed-link capacity term (C10e) binds?

Floorplet minimizes traffic-weighted distance and is blind to link capacity;
ILP minimizes the full makespan T = max(protocol completion, capacity bound).
We compare the FULL makespan performance_proxy_cycles (=T.X), which includes
congestion, on the e032 shared BOM (7/25) at:
  * nominal      : tokens/src=32,  hidden=128,  tau_p=104
  * comm-bound   : tokens/src=256, hidden=1024, tau_p=4000
ILP fixed-BOM (placement only) and free-BOM (joint) are both solved; Floorplet
is scored via fixed_solution; TAP via the standalone protocol scorer (lower
bound -- it lacks the capacity term, so its true makespan is only worse).
"""
from __future__ import print_function
import os, sys, json, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT + "/paper_date/reviewer_response")
sys.path.insert(0, ROOT + "/src")
import exp_driver as D
import ilp_layout as il
import tau_score_standalone as SS

CID, SUF, X = "e032_m12", "032", 12
A_MAX, P_MAX = 5760.0, 4480.0
NL, NS = 7, 25
P = ROOT + "/data/params"
B = ROOT + "/out/joint_bom_scaling_fresh_20260802/baselines/scaling/" + CID + "/prior"
params = json.load(open("%s/experts_%s.json" % (P, SUF)))
has_cpu = "CPU" in params.get("library", {})
floor = json.load(open("%s/floorplet.json" % B))
tap = json.load(open("%s/tap.json" % B))
print("beta =", params.get("interconnect", {}).get("beta"), flush=True)


def budgets_fixed():
    b = {"a_max": A_MAX, "a_max_layer": None, "p_max": P_MAX,
         "nmax": {}, "nmin": {}}
    if has_cpu:
        b["nmin"]["CPU"] = 1; b["nmax"]["CPU"] = 1
    b["nmin"]["GPU-L"] = NL; b["nmax"]["GPU-L"] = NL
    b["nmin"]["GPU-S"] = NS; b["nmax"]["GPU-S"] = NS
    return b


def budgets_free():
    return D.make_budgets(A_MAX, P_MAX, None, has_cpu=has_cpu)


def opts_vol(tau, tokens, hidden, tl):
    o = D.base_solver_opts(params, tl, mipgap=1e-6, geom=True,
                           protocol_hop_cycles=tau)
    o["dispatch_tokens_per_source"] = tokens
    o["dispatch_hidden"] = hidden
    return o


def solve(tau, tokens, hidden, free, warm, tl):
    o = opts_vol(tau, tokens, hidden, tl)
    if warm is not None:
        o["warm_start_solution"] = warm
    bud = budgets_free() if free else budgets_fixed()
    sol, st = il.build_and_solve(params, 1, X, X, bud, o, verbose=False)
    if sol is None:
        return None
    c = {}
    for s in sol["sites"]:
        c[s["type"]] = c.get(s["type"], 0) + 1
    return {"T": float(sol["performance_proxy_cycles"]),
            "proto": float(sol["protocol_makespan_cycles"]),
            "bom": "%d/%d" % (c.get("GPU-L", 0), c.get("GPU-S", 0)),
            "gap": sol.get("mip_gap")}


def score_fixed(layout, tau, tokens, hidden, tl=45.0):
    o = opts_vol(tau, tokens, hidden, tl)
    o["fixed_solution"] = layout
    sol, st = il.build_and_solve(params, 1, X, X, budgets_fixed(), o,
                                 verbose=False)
    if sol is None:
        return None
    return {"T": float(sol["performance_proxy_cycles"]),
            "proto": float(sol["protocol_makespan_cycles"])}


for tag, tau, tk, hd in [("nominal", 104, 32, 128),
                         ("commbound", 4000, 256, 1024)]:
    print("\n=== %s: tau=%d tokens/src=%d hidden=%d ===" % (tag, tau, tk, hd),
          flush=True)
    const = SS.build_constants(params, tokens_per_source=tk, dispatch_hidden=hd)
    tap_proto = SS.score_layout(params, tap, tau, const)
    f = score_fixed(floor, tau, tk, hd)
    ilp = solve(tau, tk, hd, free=False, warm=floor, tl=60.0)
    joint = solve(tau, tk, hd, free=True, warm=floor, tl=90.0)
    print("floor      :", f, flush=True)
    print("ilp_fixBOM :", ilp, flush=True)
    print("ilp_joint  :", joint, flush=True)
    print("tap_proto  : %.1f (protocol only, no capacity term)" % tap_proto,
          flush=True)
    if f and ilp:
        print("  Floor vs ILP(fixBOM) on T: %+.3f%%"
              % (100.0 * (f["T"] - ilp["T"]) / f["T"]), flush=True)
    if f and joint:
        print("  Floor vs ILP(joint)  on T: %+.3f%%"
              % (100.0 * (f["T"] - joint["T"]) / f["T"]), flush=True)
    if ilp and tap_proto:
        print("  TAP(proto) vs ILP(fixBOM): %+.3f%%"
              % (100.0 * (tap_proto - ilp["T"]) / tap_proto), flush=True)
print("\nDONE", flush=True)
