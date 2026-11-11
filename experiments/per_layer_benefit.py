# -*- coding: utf-8 -*-
"""Analytical popularity benefit in every Phi-tiny-MoE layer (WikiText routing):
PopRank-G packing vs. 25 popularity-blind relabelings of the same packing."""
from __future__ import print_function
import os, random, statistics
import ispa_common as C
import drift_lib as D
import exp_poprank as PR

E, X, A, P = 16, 8, 2880.0, 2240.0
st = C.read_json(os.path.join(C.RESULTS, "realtrace", "stats.json"))
wiki = st["domains"]["wiki"]["layers"]
res = {}
for layer in sorted(wiki, key=int):
    pi = sorted(wiki[layer]["pi"], reverse=True)
    params = D.build_params(E, pi, "phitiny-L%s" % layer)
    ev = D.FastEval(params)
    pack = PR.compact_packing(params, X, PR.poprank_bundles(params, PR.greedy_nl(E, A, P)), A, P)
    t_rank = ev.T(pack)
    dies = D.expert_dies(pack)
    ids = [g for _, g in dies]
    blind = []
    for s in range(25):
        perm = list(ids)
        random.Random(1000 + s).shuffle(perm)
        blind.append(ev.T(D.relabel(pack, {idx: g for (idx, _), g in zip(dies, perm)})))
    med = statistics.median(blind)
    res[layer] = {"top3": wiki[layer]["top3"], "T_rank": t_rank, "T_blind_med": med,
                  "reduction_pct": 100 * (1 - t_rank / med)}
C.write_json(os.path.join(C.RESULTS, "realtrace_per_layer.json"), res)
red = sorted(v["reduction_pct"] for v in res.values())
print("per-layer reduction vs blind: min %.2f median %.2f max %.2f" % (red[0], red[len(red) // 2], red[-1]))
for layer in sorted(res, key=lambda l: res[l]["top3"]):
    v = res[layer]
    print("L%-3s top3=%.3f reduction=%.2f%%" % (layer, v["top3"], v["reduction_pct"]))
