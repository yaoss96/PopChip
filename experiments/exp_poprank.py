# -*- coding: utf-8 -*-
"""R1/R3: popularity-ranked two-stage heuristics vs the joint MILP.

Stage 1 buys the greedy max-L BOM, stage 2 maps experts by token load (hottest
to GPU-L), stage 3 places the fixed bundles with one of:
  greedy  : compact packing, hottest dies nearest the CPU router;
  fp-proto: exact MILP on traffic-weighted Manhattan distance of the *replayed*
            protocol traffic (all-to-all payloads + CPU control);
  min-T   : the paper's min-T MILP restricted to these fixed bundles.
Every layout is scored with the same analytical T (fixed-layout solve).
Usage: python exp_poprank.py CASE_ID
"""
from __future__ import print_function
import copy, json, os, sys, time
import ispa_common as C
import ilp_layout as il
import matched_layout_baseline as matched
import run_optimized_prior_proxies as proxies

OUT = os.path.join(C.RESULTS, "poprank")


def greedy_nl(E, a_max, p_max):
    best = 0
    for nl in range(E + 1):
        a = 90.0 + 400.0 * nl + 110.0 * (E - nl)
        p = 65.0 + 300.0 * nl + 90.0 * (E - nl)
        if a <= a_max and p <= p_max:
            best = nl
    return best


def ranked_groups(params):
    pi = params["meta"]["group_pi"]
    return sorted(range(len(pi)), key=lambda g: (-pi[g], g))


def poprank_bundles(params, nL):
    order = ranked_groups(params)
    hot = order[0]
    bundles = [("CPU", ["t_gate", "t_norm"])]
    for rank, g in enumerate(order):
        kind = "GPU-L" if rank < nL else "GPU-S"
        tasks = ["e(%d)" % g]
        if g == hot:
            tasks += ["t_attn", "t_dispatch", "t_combine"]
            if kind != "GPU-L" and "GPU-S" not in params["tasks"]["t_attn"]["allowed"]:
                raise ValueError("aux tasks incompatible")
        bundles.append((kind, tasks))
    return bundles


def compact_packing(params, X, bundles, a_max, p_max):
    lib = params["library"]
    anchors, footprint = il.anchors_and_footprints(1, X, X, lib)
    occupied = set()
    sites, task_site = [], {}

    def centre(kind, pa):
        return (pa[1] + (int(lib[kind]["w"]) - 1) / 2.0,
                pa[2] + (int(lib[kind]["h"]) - 1) / 2.0)

    mid = ((X - 1) / 2.0, (X - 1) / 2.0)
    ref = None
    for kind, tasks in bundles:
        target = mid if ref is None else ref
        best = None
        for pa in anchors[kind]:
            cells = footprint[kind][pa]
            if any(c in occupied for c in cells):
                continue
            cx, cy = centre(kind, pa)
            key = (abs(cx - target[0]) + abs(cy - target[1]), pa)
            if best is None or key < best[0]:
                best = (key, pa)
        if best is None:
            raise RuntimeError("packing failed for %s" % kind)
        pa = best[1]
        occupied.update(footprint[kind][pa])
        if ref is None:
            ref = centre(kind, pa)
        sites.append({"l": pa[0], "i": pa[1], "j": pa[2], "type": kind,
                      "tasks": list(tasks)})
        for t in tasks:
            task_site[t] = [pa[0], pa[1], pa[2]]
    return {"sites": sites, "task_site": task_site,
            "grid": {"L": 1, "X": X, "Y": X},
            "budgets": C.budgets(a_max, p_max),
            "optimization_mode": "ilp-perf"}


def protocol_edges(params):
    W = il.deterministic_dispatch(params, 32)
    E = len(W)
    tok = 128 * 2
    # One edge per unordered pair (lower id -> higher id) keeps the edge set
    # acyclic for the strict validator; the Floorplet weight is symmetric.
    edges = []
    for s in range(E):
        for d in range(s + 1, E):
            vol = 2 * (W[s][d] + W[d][s]) * tok
            if vol > 0:
                edges.append({"src": "e(%d)" % s, "dst": "e(%d)" % d,
                              "vol": int(vol)})
    ctrl = int(4 * (4 + 2 * E) + 4)
    for g in range(E):
        edges.append({"src": "t_gate", "dst": "e(%d)" % g, "vol": ctrl})
    return edges


def full_solution(params, layout):
    """Round-trip a raw layout through the fixed-layout solve -> replayable JSON."""
    X = int(layout["grid"]["X"])
    o = C.solver_opts(900.0, mipgap=1e-9, extra={"fixed_solution": layout})
    s, _ = il.build_and_solve(params, 1, X, X, C.budgets(1e9, 1e9), o,
                              verbose=False)
    return s


def main(cid):
    E, X, a_max, p_max, tl = C.CASES[cid]
    params = C.case_params(cid)
    res = {"case": cid, "experts": E, "mesh": X, "a_max": a_max,
           "p_max": p_max, "time_limit": tl}
    # analytic T of every Table-III layout (calibration against replay)
    res["table3_analytic_T"] = {}
    for name, path in sorted(C.case_paths(cid).items()):
        res["table3_analytic_T"][name] = C.evaluate_fixed(params, C.read_json(path))
    popchip = C.read_json(C.case_paths(cid)["popchip"])
    res["popchip"] = C.summary(popchip)
    nL = greedy_nl(E, a_max, p_max)
    res["greedy_nL"] = nL
    bundles = poprank_bundles(params, nL)
    pack = compact_packing(params, X, bundles, a_max, p_max)
    t0 = time.time()
    greedy_sol = full_solution(params, pack)
    res["poprank_greedy"] = C.summary(greedy_sol, wall=time.time() - t0)
    C.write_json(os.path.join(OUT, cid, "poprank_greedy.json"), greedy_sol)
    print(cid, "greedy", res["poprank_greedy"]["T"], flush=True)

    # fp-proto: exact traffic-weighted placement of the replayed traffic
    pmod = copy.deepcopy(params)
    pmod["edges"] = protocol_edges(params)
    state = matched._source_state(pmod, greedy_sol)
    t0 = time.time()
    fp_sol, fp_meta = proxies.exact_floorplet(pmod, greedy_sol, state, 45.0,
                                              seed=1, threads=1)
    fp_wall = time.time() - t0
    fp_full = full_solution(params, {"sites": fp_sol["sites"],
                                     "task_site": fp_sol["task_site"],
                                     "grid": fp_sol["grid"],
                                     "budgets": C.budgets(a_max, p_max)})
    res["poprank_fpproto"] = C.summary(fp_full, wall=fp_wall)
    res["poprank_fpproto"]["fp_meta"] = {k: fp_meta.get(k) for k in
                                         ("F_FP", "status", "gap", "wall_s")}
    C.write_json(os.path.join(OUT, cid, "poprank_fpproto.json"), fp_full)
    print(cid, "fp-proto", res["poprank_fpproto"]["T"], flush=True)

    # min-T placement of the fixed PopRank bundles (same budget as PopChip)
    rb = il.reference_bundles(greedy_sol, params["tasks"], params["library"])
    sol, wall = C.solve(params, X, a_max, p_max, tl,
                        geom=(cid == "e064_m16"),
                        extra={"placement_bundles": rb})
    res["poprank_mint"] = C.summary(sol, wall=wall)
    if sol is not None:
        C.write_json(os.path.join(OUT, cid, "poprank_mint.json"), sol)
    print(cid, "min-T", res["poprank_mint"].get("T"), flush=True)
    C.write_json(os.path.join(OUT, "%s.json" % cid), res)
    print(json.dumps(res, indent=1, sort_keys=True)[:4000])


if __name__ == "__main__":
    main(sys.argv[1])
