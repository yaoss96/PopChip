#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prior_rerun.py -- re-run the two prior-work baselines (Floorplet, TAP-2.5D)
with multiple seeds, selecting the layout that is best under EACH method's OWN
published objective (min wirelength / min peak thermal density), then scoring
that layout's makespan/power/area/energy through the shared evaluate.py.

This is the honest fix for the "why are their makespans identical?" question:
each method now picks the placement IT prefers (its own objective), and we
report how that placement actually runs. Because the two objectives differ,
the selected layouts -- and thus their makespans -- differ."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev
import baselines as bl

CONFIGS = {
    "mixtral":     ("out/params_mixtral.json",     dict(a_max=2500, p_max=1300, p_col=340)),
    "switch":      ("out/params_switch.json",      dict(a_max=4000, p_max=2000, p_col=600)),
    "finegrained": ("out/params_finegrained.json", dict(a_max=4000, p_max=2200, p_col=700)),
    "dense-llm":   ("out/params_dense-llm.json",   dict(a_max=4000, p_max=2200, p_col=700)),
    "attn-heavy":  ("out/params_attn-heavy.json",  dict(a_max=6000, p_max=3600, p_col=900)),
}
GRID = {"L": 1, "X": 8, "Y": 8}
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def wl_cost(params):
    edges = params["edges"]
    def f(ts, st):
        tot = 0.0
        for e in edges:
            ps, pd = tuple(ts[e["src"]]), tuple(ts[e["dst"]])
            tot += e["vol"] * (abs(ps[1]-pd[1]) + abs(ps[2]-pd[2]) + abs(ps[0]-pd[0]))
        return tot
    return f


def thermal_cost(params):
    edges = params["edges"]; lib = params["library"]
    def f(ts, st):
        col = {}
        for a, k in st.items():
            dens = lib[k]["p_peak"] / (lib[k]["w"] * lib[k]["h"])
            for cell in bl.footprint(k, a, lib):
                key = (cell[1], cell[2]); col[key] = col.get(key, 0.0) + dens
        peak = max(col.values()) if col else 0.0
        wl = 0.0
        for e in edges:
            ps, pd = tuple(ts[e["src"]]), tuple(ts[e["dst"]])
            wl += e["vol"] * (abs(ps[1]-pd[1]) + abs(ps[2]-pd[2]) + abs(ps[0]-pd[0]))
        return peak * 1e3 + 1e-4 * wl
    return f


def best_by_objective(params, budgets, cost_fn, iters=12000):
    """Run _anneal across seeds, keep the layout with the lowest OWN-objective
    cost (what the published method would actually output)."""
    best = None  # (own_cost, ts, st)
    for s in SEEDS:
        res = bl._anneal(params, GRID, cost_fn, seed=s, budgets=budgets, iters=iters)
        if res is None:
            continue
        ts, st = res
        c = cost_fn(ts, st)
        if best is None or c < best[0]:
            best = (c, ts, st)
    return best


def main():
    print("%-12s %-10s %10s %8s %8s %8s %10s" %
          ("config", "method", "makespan", "power", "area", "energyuJ", "own_obj"))
    out = {}
    for cfg, (pp, bud) in CONFIGS.items():
        params = json.load(open(pp))
        budgets = dict(bud); budgets["e_bar"] = None
        for mth, cf in (("floorplet", wl_cost(params)),
                        ("thermal", thermal_cost(params))):
            b = best_by_objective(params, budgets, cf)
            if b is None:
                print("%-12s %-10s FAILED" % (cfg, mth)); continue
            own, ts, st = b
            sc = ev.evaluate(params, ts, st, GRID, strict=True, budgets=budgets)
            out[(cfg, mth)] = dict(mk=sc["makespan_cycles"], pw=sc["peak_power_W"],
                                   ar=sc["total_area_mm2"], en=sc["energy_total_J"],
                                   own=own)
            print("%-12s %-10s %10.3e %8.0f %8.0f %8.1f %10.3e" %
                  (cfg, mth, sc["makespan_cycles"], sc["peak_power_W"],
                   sc["total_area_mm2"], sc["energy_total_J"]*1e6, own))
    json.dump({f"{c}|{m}": v for (c, m), v in out.items()},
              open("out/prior_rerun.json", "w"), indent=2)
    print("wrote out/prior_rerun.json")


if __name__ == "__main__":
    main()
