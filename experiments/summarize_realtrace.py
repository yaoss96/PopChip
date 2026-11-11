# -*- coding: utf-8 -*-
"""Summarize the real-trace design + drift into numbers for the paper and the
two Phi-tiny-MoE rows of drift-table.tex (results/realtrace_drift.json)."""
from __future__ import print_function
import csv, glob, io, json, os, statistics, sys
import ispa_common as C

layer = sys.argv[1]
d = os.path.join(C.RESULTS, "realtrace", "L%s" % layer)
st = C.read_json(os.path.join(C.RESULTS, "realtrace", "stats.json"))
des = C.read_json(os.path.join(d, "design.json"))
out = {"layer": layer}

# measured replays of the design (case id rt_L<layer>)
meas = {r["method"]: r["measured_cycles"] for r in C.read_measured()
        if r["case"] == "rt_L%s" % layer}
out["measured"] = meas
if "popchip" in meas:
    pc = meas["popchip"]
    bl = [meas[k] for k in ("blind1", "blind2", "blind3") if k in meas]
    if bl:
        b = statistics.median(bl)
        out["red_vs_blind_pct"] = 100 * (1 - pc / float(b))
        out["blind_slower_pct"] = 100 * (b / float(pc) - 1)
    for k in ("poprank-greedy", "floorplet", "tap"):
        if k in meas:
            out["%s_slower_pct" % k] = 100 * (meas[k] / float(pc) - 1)
an = {"popchip": des["popchip"]["T"], "poprank": des["poprank_greedy_T"],
      "fp": des["floorplet_T"], "tap": des["tap_T"],
      "blind_med": statistics.median([des["blind%d_T" % s] for s in (1, 2, 3)])}
out["analytic"] = an
out["analytic_blind_slower_pct"] = 100 * (an["blind_med"] / an["popchip"] - 1)
out["bom"] = (des["popchip"]["nL"], des["popchip"]["nS"])
out["l_experts"] = des["popchip"]["l_experts"]
gp = des["group_pi"]
out["top3_mass"] = sum(sorted(gp, reverse=True)[:3])

rows = []
dr_path = os.path.join(d, "drift.json")
if os.path.exists(dr_path):
    dr = C.read_json(dr_path)["cases"]
    def ovh(r, with_red=True):
        vals = {"static": r["static"], "remap_rank": r["remap_rank"], "remap_ls": r["remap_ls"]}
        if with_red and r.get("redesign", {}).get("T"):
            vals["redesign"] = r["redesign"]["T"]
        best = min(vals.values())
        return {k: 100 * (v / best - 1) for k, v in vals.items()}
    dom = {k: ovh(v) for k, v in dr.items() if not k.startswith("wiki_w")}
    win = [ovh(v, with_red=False) for k, v in dr.items() if k.startswith("wiki_w")]
    out["domain_overheads"] = dom
    if dom:
        worst = {p: max(o.get(p, 0.0) for o in dom.values()) for p in ("static", "remap_rank", "remap_ls", "redesign")}
        rows.append(["Domain shift (Phi-tiny)", worst])
    if win:
        wworst = {p: max(o[p] for o in win) for p in ("static", "remap_rank", "remap_ls")}
        wmed = {p: statistics.median([o[p] for o in win]) for p in ("static", "remap_rank", "remap_ls")}
        out["window_worst"] = wworst
        out["window_median"] = wmed
        wworst["redesign"] = None
        rows.append(["512-token windows (Phi-tiny)", wworst])
wiki = st["domains"]["wiki"]["layers"]
out["wiki_top3_range"] = (min(v["top3"] for v in wiki.values()), max(v["top3"] for v in wiki.values()))
out["wiki_top3_median"] = statistics.median([v["top3"] for v in wiki.values()])
out["wiki_alpha_range"] = (min(v["zipf_alpha"] for v in wiki.values()), max(v["zipf_alpha"] for v in wiki.values()))
out["wiki_alpha_median"] = statistics.median([v["zipf_alpha"] for v in wiki.values()])
out["layer_stats"] = {k: v for k, v in wiki[layer].items() if k != "pi"}
for dom in ("math", "code"):
    k = "cross_domain_tv_%s" % dom
    if k in st:
        out["tv_%s_layer" % dom] = st[k][layer]
        out["tv_%s_median" % dom] = statistics.median(list(st[k].values()))
C.write_json(os.path.join(C.RESULTS, "realtrace_summary.json"), out)
C.write_json(os.path.join(C.RESULTS, "realtrace_drift.json"),
             {"rows": [[r[0], {k: (v if v is not None else 0.0) for k, v in r[1].items()}] for r in rows],
              "redesign_missing": [r[0] for r in rows if r[1].get("redesign") is None]})
print(json.dumps(out, indent=1, default=str))
