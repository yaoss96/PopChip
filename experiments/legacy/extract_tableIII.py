#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pull the compact Table-III numbers (BOM, cycles, area, power, gap) out of the
big solution JSONs for ILP-Perf and the Floorplet/TAP placements on ILP's BOM."""
from __future__ import print_function
import os, json

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
B = ROOT + "/out/joint_bom_scaling_fresh_20260802"
CASES = ["e008_m06", "e016_m08", "e032_m12", "e064_m12", "e128_m16"]

CYC_KEYS = ["synchronized_cycles", "measured_cycles", "legosim_cycles",
            "replay_cycles", "predicted_makespan_cycles",
            "protocol_makespan_cycles"]


def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        return {"__err__": str(e)}


def cyc(d):
    out = {}
    for k in CYC_KEYS:
        if k in d and isinstance(d[k], (int, float)):
            out[k] = d[k]
    return out


def summarize(tag, d):
    if "__err__" in d:
        print("   %-10s MISSING (%s)" % (tag, d["__err__"]))
        return
    counts = d.get("selected_type_counts") or {}
    if not counts and "sites" in d:
        for st in d["sites"]:
            counts[st["type"]] = counts.get(st["type"], 0) + 1
    print("   %-10s BOM L=%s S=%s | area=%s power=%s gap=%s"
          % (tag, counts.get("GPU-L"), counts.get("GPU-S"),
             d.get("total_area_mm2"), d.get("peak_power_W"),
             d.get("mip_gap")))
    print("             cycles: %s" % json.dumps(cyc(d)))


for cid in CASES:
    print("== %s ==" % cid)
    ilp = load("%s/solutions/scaling/%s/ilp_perf.json" % (B, cid))
    summarize("ILP-Perf", ilp)
    for name in ["floorplet", "tap"]:
        p = "%s/baselines/scaling/%s/prior/%s.json" % (B, cid, name)
        summarize(name, load(p))
    print()
