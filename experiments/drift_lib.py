# -*- coding: utf-8 -*-
"""Popularity-drift helpers: identity-grouped params, fast protocol evaluator,
and the runtime remapping policies (rank remap and swap local search)."""
from __future__ import print_function
import copy, math, random, re, time
import ispa_common as C
import ilp_layout as il
import workload as W

EXPERT = re.compile(r"^e\((\d+)\)$")


def zipf(E, a):
    w = [1.0 / ((r + 1) ** a) for r in range(E)]
    s = sum(w)
    return [x / s for x in w]


def build_params(E, pi, tag="drift"):
    """workload.build with expert g fixed to group g (no popularity re-sort)."""
    oz, og = W.zipf_popularity, W.group_experts
    try:
        W.zipf_popularity = lambda E_, skew: [float(x) for x in pi]
        W.group_experts = lambda p, G: [(p[g], [g]) for g in range(G)]
        return W.build(tag, E, E, 2, 4096, 14336, 256, 2, 0.7)
    finally:
        W.zipf_popularity, W.group_experts = oz, og


EXACT_S = [42450.0, 29390.0, 23726.0, 20619.0, 18338.0, 17002.0, 16488.0, 16120.0]
EXACT_L = [4663.0, 4670.0, 4615.0, 4598.0, 4573.0, 4570.0, 4559.0, 4557.0]


def compute_cycles(kind, group, column_tokens, exact_e8):
    n = column_tokens[group]
    if kind == "GPU-S":
        return EXACT_S[group] if exact_e8 else 7431.145 + 243.523 * n
    if kind == "GPU-L":
        return EXACT_L[group] if exact_e8 else 4575.206 + 0.6926 * n
    raise ValueError(kind)


class FastEval(object):
    """Pure-Python P1 evaluator (q=1, one expert per die), equal to the MILP's
    protocol clock; the link-capacity term of P2 is checked separately."""

    def __init__(self, params, hop=104.0, flit=32.0):
        self.params = params
        self.W = il.deterministic_dispatch(params, 32)
        self.E = len(self.W)
        self.col = [sum(self.W[s][d] for s in range(self.E)) for d in range(self.E)]
        self.exact = (self.E == 8 and self.col == [144, 89, 67, 54, 47, 41, 37, 33])
        self.hop, self.flit = float(hop), float(flit)

    def T(self, layout):
        ts = layout["task_site"]
        kind_at = {(s["l"], s["i"], s["j"]): s["type"] for s in layout["sites"]}
        cpu = [tuple(ts["t_gate"])]
        # CPU anchor = the CPU site
        cpu_site = [k for k, v in kind_at.items() if v == "CPU"][0]
        sites = [tuple(ts["e(%d)" % g]) for g in range(self.E)]
        cyc = [compute_cycles(kind_at[sites[g]], g, self.col, self.exact)
               for g in range(self.E)]
        r = il._evaluate_fixed_order_protocol(
            cpu_site, sites, self.W, cyc, 128 * 2, 4.0 * (4 + 2 * self.E), 4.0,
            1000.0, self.flit, self.hop)
        return r["completion_cycles"]


def expert_dies(layout):
    out = []
    for idx, s in enumerate(layout["sites"]):
        ex = [t for t in s.get("tasks", []) if EXPERT.match(t)]
        if len(ex) > 1:
            raise ValueError("q>1 site")
        if ex:
            out.append((idx, int(EXPERT.match(ex[0]).group(1))))
    return out


def relabel(layout, new_expert_at_site):
    """new_expert_at_site: {site_index: expert_id}; aux tasks stay put."""
    v = copy.deepcopy(layout)
    for idx, g in new_expert_at_site.items():
        tasks = [t for t in v["sites"][idx]["tasks"] if not EXPERT.match(t)]
        v["sites"][idx]["tasks"] = ["e(%d)" % g] + tasks
        s = v["sites"][idx]
        v["task_site"]["e(%d)" % g] = [s["l"], s["i"], s["j"]]
    return v


def rank_remap(layout, pi_design, pi_new):
    """Die that served design-rank r now serves the new rank-r expert."""
    rank_d = sorted(range(len(pi_design)), key=lambda g: (-pi_design[g], g))
    rank_n = sorted(range(len(pi_new)), key=lambda g: (-pi_new[g], g))
    pos = {g: r for r, g in enumerate(rank_d)}
    mapping = {}
    for idx, g in expert_dies(layout):
        mapping[idx] = rank_n[pos[g]]
    return relabel(layout, mapping)


def swap_search(layout, ev, wall=60.0, seed=1):
    """First-improvement pairwise expert swaps between expert dies."""
    t0 = time.time()
    cur = layout
    best = ev.T(cur)
    pairs = expert_dies(cur)
    improved = True
    evals = 0
    while improved and time.time() - t0 < wall:
        improved = False
        assign = dict(expert_dies(cur))
        idxs = sorted(assign)
        order = [(a, b) for ai, a in enumerate(idxs) for b in idxs[ai + 1:]]
        random.Random(seed + evals).shuffle(order)
        for a, b in order:
            if time.time() - t0 >= wall:
                break
            cand_map = {a: assign[b], b: assign[a]}
            cand = relabel(cur, cand_map)
            t = ev.T(cand)
            evals += 1
            if t < best - 1e-6:
                cur, best, improved = cand, t, True
                assign = dict(expert_dies(cur))
                break
    return cur, best, evals, time.time() - t0
