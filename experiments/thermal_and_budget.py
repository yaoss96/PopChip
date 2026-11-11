# -*- coding: utf-8 -*-
"""Peak-power-coupling surrogate of PopChip / FP-px / TAP-px layouts, the GPU-L
token mass of each PopChip map, and the budget-sweep margins of Fig. 3."""
from __future__ import print_function
import csv
import io
import os
import ispa_common as C
import prior_layout_baseline as PB
import ilp_layout as il


def coupling_and_mass():
    for cid in sorted(C.CASES):
        params = C.case_params(cid)
        paths = C.case_paths(cid)
        th = {m: PB._thermal_coupling(params, C.read_json(paths[m]))
              for m in ("popchip", "floorplet", "tap")}
        pc = C.read_json(paths["popchip"])
        W = il.deterministic_dispatch(params, 32)
        col = [sum(W[s][d] for s in range(len(W))) for d in range(len(W))]
        lset = [int(t[2:-1]) for s in pc["sites"] if s["type"] == "GPU-L"
                for t in s["tasks"] if t.startswith("e(")]
        mass = 100.0 * sum(col[g] for g in lset) / sum(col)
        top = 100.0 * sum(sorted(col, reverse=True)[:len(lset)]) / sum(col)
        print("%s coupling PopChip=%.1f FP=%.1f TAP=%.1f (TAP %.1f%% lower) | "
              "GPU-L token mass %.1f%% (top-N_L %.1f%%, blind %.1f%%)" % (
                  cid, th["popchip"], th["floorplet"], th["tap"],
                  100 * (1 - th["tap"] / th["popchip"]), mass, top,
                  100.0 * len(lset) / len(col)))


def budget_sweep():
    path = os.path.join(C.RESULTS, "budget_sweep", "legosim_metrics.csv")
    rows = list(csv.DictReader(io.open(path, encoding="utf-8")))
    cyc = {}
    for r in rows:
        if r.get("suite") != "budget":
            continue
        key = (r["sweep_dimension"], float(r["budget_value"]))
        cyc.setdefault(key, {})[r["method"]] = float(r["synchronized_cycles"])
    for dim in ("area", "power"):
        keys = sorted(k for k in cyc if k[0] == dim)
        for other in ("floorplet", "tap", "matched_random"):
            red = [100 * (1 - cyc[k]["ilp_perf"] / cyc[k][other]) for k in keys
                   if "ilp_perf" in cyc[k] and other in cyc[k]]
            if red:
                print("%s sweep: PopChip reduction vs %s: %.2f..%.2f%%" % (dim, other, min(red), max(red)))


if __name__ == "__main__":
    coupling_and_mass()
    budget_sweep()
