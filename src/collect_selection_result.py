# -*- coding: utf-8 -*-
"""Collect the popularity-aware vs popularity-blind SELECTION campaign and emit
the paper table data (measured LEGOSim cycles).

For each scale it reads the replay results.csv (ilp-perf + blind seeds), takes
the median blind makespan, and reports the selection penalty
(blind_median - ilp)/ilp.  Also reports the token concentration of the experts
ILP assigns to GPU-L (to quantify "hot"), from the codegen dispatch_meta.
"""
from __future__ import print_function
import csv, glob, json, os, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMP = ROOT + "/out/_selection_campaign"
SOLD = ROOT + "/out/joint_bom_scaling_fresh_20260802/solutions/scaling"
SCALES = [("e008_m06", 8, r"$6\times6$", "1/8"),
          ("e016_m08", 16, r"$8\times8$", "3/14"),
          ("e032_m12", 32, r"$12\times12$", "7/25"),
          ("e064_m12", 64, r"$12\times12$", "14/50"),
          ("e128_m16", 128, r"$16\times16$", "30/98")]


def read_scale(cid):
    f = "%s/%s/results.csv" % (CAMP, cid)
    if not os.path.exists(f):
        return None
    rows = {r["method"]: r for r in csv.DictReader(open(f))}
    if "ilp-perf" not in rows:
        return None
    ilp = float(rows["ilp-perf"]["synchronized_makespan_cycles"])
    blinds = [float(rows[m]["synchronized_makespan_cycles"])
              for m in ("blind1", "blind2", "blind3") if m in rows]
    if not blinds:
        return None
    med = statistics.median(blinds)
    return {"ilp": ilp, "blind_median": med, "blinds": blinds,
            "penalty_pct": 100.0 * (med - ilp) / ilp, "n_seeds": len(blinds)}


def gpu_l_token_share(cid):
    """From any codegen dispatch_meta in the campaign, report the fraction of
    tokens received by the experts ILP put on GPU-L."""
    pat = "%s/%s/layouts/%s/ilp-perf/codegen/dispatch_meta.json" % (CAMP, cid, cid)
    if not os.path.exists(pat):
        return None
    dm = json.load(open(pat))
    wl = dm.get("worker_token_assignments") or dm.get("global_group_actual_assignments")
    workers = dm.get("workers", [])
    if not wl or not workers:
        return None
    total = float(sum(wl))
    gpu_l_tokens = sum(wl[w["worker"]] for w in workers if w["type"] == "GPU-L")
    n_l = sum(1 for w in workers if w["type"] == "GPU-L")
    return {"gpu_l_share": 100.0 * gpu_l_tokens / total if total else 0.0,
            "n_gpu_l": n_l, "n_experts": len(wl)}


def main():
    print("%-10s %10s %14s %14s %8s  %s" %
          ("scale", "ILP", "blind_med", "penalty", "seeds", "GPU-L token share"))
    rows_tex = []
    for cid, E, mesh, bom in SCALES:
        r = read_scale(cid)
        share = gpu_l_token_share(cid)
        if r is None:
            print("%-10s  (pending)" % cid)
            rows_tex.append("%% %s pending" % cid)
            continue
        st = ("%.1f%% on %d GPU-L" % (share["gpu_l_share"], share["n_gpu_l"])
              if share else "n/a")
        print("%-10s %10d %14d %+13.2f%% %8d  %s"
              % (cid, r["ilp"], r["blind_median"], r["penalty_pct"],
                 r["n_seeds"], st))
        rows_tex.append("%d & %s & %s & %d & %d & %+.1f \\\\"
                        % (E, mesh, bom, int(round(r["ilp"] / 1000.0)) if False else r["ilp"],
                           int(r["blind_median"]), r["penalty_pct"]))
    print("\n--- LaTeX rows (Experts & Mesh & BOM & ILP & blind & penalty%) ---")
    for t in rows_tex:
        print(t)


if __name__ == "__main__":
    main()
