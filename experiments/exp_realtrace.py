# -*- coding: utf-8 -*-
"""R1-R4: trained-model routing (Phi-tiny-MoE, 16 experts, top-2).

Step "stats"  : per-layer popularity statistics from realtrace/routing_counts.json
Step "design" : 16-expert package for the chosen layer's WikiText popularity:
                PopChip (joint MILP), Blind x3, PopRank-G, FP-px, TAP-px layouts
Step "drift"  : the WikiText-designed package under the same layer's GSM8K and
                code popularity, and under each 512-token WikiText window
Usage: python exp_realtrace.py stats|design|drift
"""
from __future__ import print_function
import json, math, os, subprocess, sys
import ispa_common as C
import drift_lib as D
import exp_poprank as PR
import make_popularity_blind_variant as MB

RT = C.REALTRACE
OUT = os.path.join(C.RESULTS, "realtrace")
E, X, A, P, TL = 16, 8, 2880.0, 2240.0, 45.0


def load_counts():
    return C.read_json(os.path.join(RT, "routing_counts.json"))


def dist(counts_windows):
    tot = [0] * 16
    for w in counts_windows:
        for e in range(16):
            tot[e] += w[e]
    s = float(sum(tot))
    return [t / s for t in tot]


def zipf_fit(pi):
    v = sorted(pi, reverse=True)
    xs = [math.log(r + 1) for r in range(len(v)) if v[r] > 0]
    ys = [math.log(v[r]) for r in range(len(v)) if v[r] > 0]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return -b


def layer_stats(pi):
    v = sorted(pi, reverse=True)
    return {"top1": v[0], "top3": sum(v[:3]), "max_over_mean": v[0] * len(v),
            "zipf_alpha": zipf_fit(pi), "min": v[-1]}


def tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def stats():
    rc = load_counts()
    res = {"domains": {}}
    for dom, d in rc["domains"].items():
        layers = {}
        for l, wins in d["counts"].items():
            pi = dist(wins)
            st = layer_stats(pi)
            pis = [dist([w]) for w in wins]
            st["window_tv_mean"] = sum(tv(p, pi) for p in pis) / len(pis)
            st["window_top1_changes"] = len(set(max(range(16), key=lambda e: p[e]) for p in pis))
            st["pi"] = pi
            layers[l] = st
        res["domains"][dom] = {"windows": d["windows"], "layers": layers}
    wiki = res["domains"]["wiki"]["layers"]
    order = sorted(wiki, key=lambda l: wiki[l]["top3"])
    res["median_layer"] = order[len(order) // 2]
    res["max_layer"] = order[-1]
    res["min_layer"] = order[0]
    for dom in ("math", "code"):
        if dom in res["domains"]:
            res["cross_domain_tv_%s" % dom] = {
                l: tv(wiki[l]["pi"], res["domains"][dom]["layers"][l]["pi"]) for l in wiki}
    C.write_json(os.path.join(OUT, "stats.json"), res)
    t3 = sorted(wiki[l]["top3"] for l in wiki)
    al = sorted(wiki[l]["zipf_alpha"] for l in wiki)
    t1 = sorted(wiki[l]["top1"] for l in wiki)
    print("wiki top3 mass: min %.3f median %.3f max %.3f" % (t3[0], t3[len(t3) // 2], t3[-1]))
    print("wiki top1: min %.3f median %.3f max %.3f" % (t1[0], t1[len(t1) // 2], t1[-1]))
    print("wiki zipf alpha: min %.2f median %.2f max %.2f" % (al[0], al[len(al) // 2], al[-1]))
    for key in ("median_layer", "max_layer", "min_layer"):
        l = res[key]
        print(key, l, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in wiki[l].items() if k != "pi"})
    for dom in ("math", "code"):
        k = "cross_domain_tv_%s" % dom
        if k in res:
            v = sorted(res[k].values())
            print("TV(wiki,%s): median %.3f max %.3f" % (dom, v[len(v) // 2], v[-1]))
    return res


def design_params(layer, dom="wiki"):
    st = C.read_json(os.path.join(OUT, "stats.json"))
    pi = st["domains"][dom]["layers"][layer]["pi"]
    order = sorted(range(16), key=lambda e: (-pi[e], e))   # group g = rank g (wiki)
    return order, st


def params_for(layer, order, dom, st):
    pi = st["domains"][dom]["layers"][layer]["pi"]
    return D.build_params(E, [pi[e] for e in order], "phitiny-L%s-%s" % (layer, dom))


def design(layer):
    order, st = design_params(layer)
    d = os.path.join(OUT, "L%s" % layer)
    pw = params_for(layer, order, "wiki", st)
    C.write_json(os.path.join(d, "params_wiki.json"), pw)
    ev = D.FastEval(pw)
    rec = {"layer": layer, "order": order, "group_pi": pw["meta"]["group_pi"]}
    sol, wall = C.solve(pw, X, A, P, TL)
    rec["popchip"] = C.summary(sol, wall=wall)
    C.write_json(os.path.join(d, "popchip.json"), sol)
    for seed in (1, 2, 3):
        v = MB.make_variant(sol, seed)
        C.write_json(os.path.join(d, "blind_seed%d.json" % seed), v)
        rec["blind%d_T" % seed] = ev.T(v)
    nL = PR.greedy_nl(E, A, P)
    pack = PR.compact_packing(pw, X, PR.poprank_bundles(pw, nL), A, P)
    C.write_json(os.path.join(d, "poprank_greedy.json"), pack)
    rec["poprank_greedy_T"] = ev.T(pack)
    prior = os.path.join(d, "prior")
    if not os.path.isdir(prior):
        cmd = [sys.executable, os.path.join(C.SRC, "generate_revalidation_baselines.py"),
               "--params", os.path.join(d, "params_wiki.json"),
               "--source", os.path.join(d, "popchip.json"), "--case-id", "rt_L%s" % layer,
               "--out-dir", prior, "--floorplet-time-limit", "45", "--tap-wall-budget", "15",
               "--tap-seeds", "11,29,47", "--tap-alpha", "0.7", "--threads", "1"]
        subprocess.check_call(cmd)
    for m in ("floorplet", "tap"):
        rec["%s_T" % m] = ev.T(C.read_json(os.path.join(prior, m + ".json")))
    C.write_json(os.path.join(d, "design.json"), rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("group_pi",)}, indent=1))


def drift(layer):
    order, st = design_params(layer)
    d = os.path.join(OUT, "L%s" % layer)
    pw = C.read_json(os.path.join(d, "params_wiki.json"))
    pc = C.read_json(os.path.join(d, "popchip.json"))
    pi_w = pw["meta"]["group_pi"]
    res = {"layer": layer, "cases": {}}
    cases = []
    for dom in ("math", "code"):
        cases.append((dom, params_for(layer, order, dom, st)))
    rc = load_counts()
    for wi, w in enumerate(rc["domains"]["wiki"]["counts"][layer]):
        s = float(sum(w))
        pi = [w[e] / s for e in order]
        cases.append(("wiki_w%02d" % wi, D.build_params(E, pi, "phitiny-w%d" % wi)))
    for name, pB in cases:
        ev = D.FastEval(pB)
        pi_B = pB["meta"]["group_pi"]
        r = {"static": ev.T(pc)}
        remap = D.rank_remap(pc, pi_w, pi_B)
        r["remap_rank"] = ev.T(remap)
        ls, t_ls, evals, wall = D.swap_search(remap, ev, wall=20.0 if name.startswith("wiki_w") else 60.0)
        r["remap_ls"] = t_ls
        if not name.startswith("wiki_w"):
            sol, w2 = C.solve(pB, X, A, P, TL)
            r["redesign"] = C.summary(sol, wall=w2)
            C.write_json(os.path.join(d, "params_%s.json" % name), pB)
            C.write_json(os.path.join(d, "%s_remap_rank.json" % name), remap)
            if sol is not None:
                C.write_json(os.path.join(d, "%s_redesign.json" % name), sol)
        res["cases"][name] = r
        C.write_json(os.path.join(d, "drift.json"), res)
        print(name, {k: (round(v) if isinstance(v, float) else (v.get("T") if isinstance(v, dict) else v)) for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "stats":
        stats()
    else:
        layer = sys.argv[2] if len(sys.argv) > 2 else C.read_json(os.path.join(OUT, "stats.json"))["median_layer"]
        {"design": design, "drift": drift}[step](layer)
