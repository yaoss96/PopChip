#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep.py -- the Section 8 evaluation sweeps + baseline comparison table.

Produces the plots/tables the design doc asks for:
  1. Baseline comparison  (ILP vs random/greedy/SA) on all three workloads,
     scored by the shared evaluate.py -> comparison.csv
  2. epsilon-constraint Pareto: sweep P_max, record (makespan, power, area,
     energy) -> pareto_<config>.csv
  3. lambda_tau sweep (vertical-hop cost): the doc's headline table showing the
     layout flips from "experts spread on one layer" to "experts stacked in
     columns above the router" as vertical links get cheaper -> lambda_sweep.csv
  4. architecture sweep (L,X,Y) at constant |P|=64 -> arch_sweep.csv

Everything shells out to the Gurobi py37 interpreter for the ILP and uses
system python3 for baselines/evaluation. Outputs plain CSV (no pandas needed).
"""

import argparse
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev
import baselines as bl


def winp(p):
    ap = os.path.abspath(p)
    if ap.startswith("/mnt/") and len(ap) > 6 and ap[6] == "/":
        return ap[5].upper() + ":\\" + ap[7:].replace("/", "\\")
    return p


def run_ilp(gpy, params, grid, budgets, out_sol, tl=60, gap=0.01, seed=1,
            t_hat=None):
    ilp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ilp_layout.py")
    args = [gpy, winp(ilp), "--params", winp(params),
            "--L", str(grid["L"]), "--X", str(grid["X"]), "--Y", str(grid["Y"]),
            "--a-max", str(budgets["a_max"]), "--p-max", str(budgets["p_max"]),
            "--p-col", str(budgets["p_col"]),
            "--time-limit", str(tl), "--mipgap", str(gap),
            "--seed", str(seed), "--quiet", "-o", winp(out_sol)]
    if budgets.get("e_bar") is not None:
        args += ["--e-bar", str(budgets["e_bar"])]
    if t_hat is not None:
        args += ["--t-hat", str(t_hat)]
    subprocess.run(args, capture_output=True, text=True)
    if not os.path.exists(out_sol):
        return None
    return json.load(open(out_sol))


def score(params, sol, budgets, grid):
    st = ev.site_types_from_solution(sol)
    return ev.evaluate(params, sol["task_site"], st, grid, strict=False,
                       budgets=budgets)


def patch_lambda(params_path, lam_tau, lam_eps, work_path):
    p = json.load(open(params_path))
    ic = p["interconnect"]
    ic["tau_v"] = ic["tau_h"] * lam_tau
    ic["eps_v"] = ic["eps_h"] * lam_eps
    with open(work_path, "w") as fh:
        json.dump(p, fh, indent=2)
    return work_path


def column_stack_score(sol):
    """How 'stacked' is the layout? Fraction of expert tasks that share a
    column (i,j) with the router or another expert across layers. Higher = more
    vertical stacking. This is the metric the lambda_tau flip is measured on."""
    cols = {}
    experts = []
    for s in sol["sites"]:
        col = (s["i"], s["j"])
        cols.setdefault(col, []).append((s["l"], s["type"], s["tasks"]))
    n_stacked_cols = sum(1 for c, occ in cols.items() if len(occ) > 1)
    n_used_cols = len(cols)
    max_layer = max((s["l"] for s in sol["sites"]), default=0)
    return {"n_used_columns": n_used_cols,
            "n_stacked_columns": n_stacked_cols,
            "max_layer_used": max_layer,
            "stack_ratio": n_stacked_cols / max(1, n_used_cols)}


# ---------------------------------------------------------------------------
def cmd_baselines(a):
    rows = []
    configs = {"mixtral": ("out/params_mixtral.json", dict(a_max=2500, p_max=1300, p_col=340)),
               "switch":  ("out/params_switch.json",  dict(a_max=4000, p_max=2000, p_col=600)),
               "finegrained": ("out/params_finegrained.json", dict(a_max=4000, p_max=2200, p_col=700)),
               "dense-llm": ("out/params_dense-llm.json", dict(a_max=4000, p_max=2200, p_col=700)),
               "attn-heavy": ("out/params_attn-heavy.json", dict(a_max=6000, p_max=3600, p_col=900))}
    grid = {"L": a.L, "X": a.X, "Y": a.Y}
    heur = ["random", "greedy", "sa", "floorplet", "thermal"]
    for cfg, (pp, bud) in configs.items():
        if not os.path.exists(pp):
            print("[sweep] missing %s, run workload.py first" % pp)
            continue
        budgets = dict(bud); budgets["e_bar"] = None
        params = json.load(open(pp))
        # ILP
        sol = run_ilp(a.gurobi_python, pp, grid, budgets,
                      "out/sweep_ilp_%s.json" % cfg, tl=a.time_limit,
                      gap=a.mipgap)
        if sol:
            sc = score(params, sol, budgets, grid)
            rows.append([cfg, "ILP-optimal", sc["makespan_cycles"],
                         sc["energy_total_J"], sc["byte_hop"],
                         sc["peak_power_W"], sc["total_area_mm2"],
                         sol.get("mip_gap"), sol.get("solver_wall_s")])
        # heuristics
        for mth in heur:
            res = bl.BASELINES[mth](params, grid, seed=0, budgets=budgets)
            if res is None:
                rows.append([cfg, mth, "INFEAS", "", "", "", "", "", ""])
                continue
            ts, stt = res
            sc = ev.evaluate(params, ts, stt, grid, strict=True, budgets=budgets)
            rows.append([cfg, mth, sc.get("makespan_cycles"),
                         sc.get("energy_total_J"), sc.get("byte_hop"),
                         sc.get("peak_power_W"), sc.get("total_area_mm2"),
                         "", ""])
    out = os.path.join(a.out, "comparison.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "method", "makespan_cycles", "energy_J",
                    "byte_hop", "peak_power_W", "area_mm2", "mip_gap",
                    "solver_wall_s"])
        w.writerows(rows)
    print("[sweep] wrote %s" % out)
    # pretty print
    print("\n%-12s %-12s %16s %12s %10s %8s" %
          ("config", "method", "makespan", "energy_J", "byte_hop", "power"))
    for r in rows:
        mk = ("%.3e" % r[2]) if isinstance(r[2], float) else r[2]
        en = ("%.3e" % r[3]) if isinstance(r[3], float) else r[3]
        bh = ("%.2e" % r[4]) if isinstance(r[4], float) else r[4]
        print("%-12s %-12s %16s %12s %10s %8s" %
              (r[0], r[1], mk, en, bh, r[5]))


def cmd_pareto(a):
    pp = {"mixtral": "out/params_mixtral.json",
          "switch": "out/params_switch.json",
          "finegrained": "out/params_finegrained.json"}[a.config]
    params = json.load(open(pp))
    grid = {"L": a.L, "X": a.X, "Y": a.Y}
    rows = []
    for pmax in a.pmax_list:
        budgets = {"a_max": a.a_max, "p_max": pmax, "p_col": pmax / 3.0 + 100,
                   "e_bar": None}
        sol = run_ilp(a.gurobi_python, pp, grid, budgets,
                      "out/pareto_%s_p%d.json" % (a.config, int(pmax)),
                      tl=a.time_limit, gap=a.mipgap)
        if not sol:
            rows.append([pmax, "INFEAS", "", "", "", ""])
            continue
        sc = score(params, sol, budgets, grid)
        rows.append([pmax, sc["makespan_cycles"], sc["peak_power_W"],
                     sc["total_area_mm2"], sc["energy_total_J"],
                     sol.get("mip_gap")])
        print("[pareto] p_max=%-6g makespan=%.3e power=%.0f area=%.0f gap=%.2f%%"
              % (pmax, sc["makespan_cycles"], sc["peak_power_W"],
                 sc["total_area_mm2"], (sol.get("mip_gap") or 0) * 100))
    out = os.path.join(a.out, "pareto_%s.csv" % a.config)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["p_max", "makespan_cycles", "peak_power_W", "area_mm2",
                    "energy_J", "mip_gap"])
        w.writerows(rows)
    print("[sweep] wrote %s" % out)


def cmd_lambda(a):
    pp = {"mixtral": "out/params_mixtral.json",
          "switch": "out/params_switch.json",
          "finegrained": "out/params_finegrained.json"}[a.config]
    grid = {"L": a.L, "X": a.X, "Y": a.Y}
    budgets = {"a_max": a.a_max, "p_max": a.p_max, "p_col": a.p_col,
               "e_bar": None}
    rows = []
    for lam in a.lambda_list:
        work = "out/params_%s_lam%.3f.json" % (a.config, lam)
        patch_lambda(pp, lam, lam, work)
        params = json.load(open(work))
        sol = run_ilp(a.gurobi_python, work, grid, budgets,
                      "out/lam_%s_%.3f.json" % (a.config, lam),
                      tl=a.time_limit, gap=a.mipgap)
        if not sol:
            rows.append([lam, "INFEAS", "", "", "", ""])
            continue
        sc = score(params, sol, budgets, grid)
        cs = column_stack_score(sol)
        rows.append([lam, sc["makespan_cycles"], cs["max_layer_used"],
                     cs["n_stacked_columns"], cs["stack_ratio"],
                     sol.get("mip_gap")])
        print("[lambda] lam_tau=%.3f makespan=%.3e max_layer=%d "
              "stacked_cols=%d stack_ratio=%.2f"
              % (lam, sc["makespan_cycles"], cs["max_layer_used"],
                 cs["n_stacked_columns"], cs["stack_ratio"]))
    out = os.path.join(a.out, "lambda_sweep_%s.csv" % a.config)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lambda_tau", "makespan_cycles", "max_layer_used",
                    "n_stacked_columns", "stack_ratio", "mip_gap"])
        w.writerows(rows)
    print("[sweep] wrote %s" % out)


def cmd_arch(a):
    pp = {"mixtral": "out/params_mixtral.json",
          "switch": "out/params_switch.json",
          "finegrained": "out/params_finegrained.json"}[a.config]
    params = json.load(open(pp))
    budgets = {"a_max": a.a_max, "p_max": a.p_max, "p_col": a.p_col,
               "e_bar": None}
    grids = [(1, 8, 8), (2, 4, 8), (4, 4, 4), (8, 2, 4)]
    rows = []
    for (L, X, Y) in grids:
        grid = {"L": L, "X": X, "Y": Y}
        sol = run_ilp(a.gurobi_python, pp, grid, budgets,
                      "out/arch_%s_%d%d%d.json" % (a.config, L, X, Y),
                      tl=a.time_limit, gap=a.mipgap)
        if not sol:
            rows.append(["%dx%dx%d" % (L, X, Y), "INFEAS", "", "", ""])
            continue
        sc = score(params, sol, budgets, grid)
        cs = column_stack_score(sol)
        rows.append(["%dx%dx%d" % (L, X, Y), sc["makespan_cycles"],
                     cs["max_layer_used"], cs["stack_ratio"],
                     sol.get("solver_wall_s")])
        print("[arch] grid=%dx%dx%d makespan=%.3e max_layer=%d wall=%.1fs"
              % (L, X, Y, sc["makespan_cycles"], cs["max_layer_used"],
                 sol.get("solver_wall_s") or 0))
    out = os.path.join(a.out, "arch_sweep_%s.csv" % a.config)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grid_LxXxY", "makespan_cycles", "max_layer_used",
                    "stack_ratio", "solver_wall_s"])
        w.writerows(rows)
    print("[sweep] wrote %s" % out)


def main():
    ap = argparse.ArgumentParser(description="ILP evaluation sweeps")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--L", type=int, default=4)
        sp.add_argument("--X", type=int, default=4)
        sp.add_argument("--Y", type=int, default=4)
        sp.add_argument("--a-max", type=float, default=4000.0)
        sp.add_argument("--p-max", type=float, default=2000.0)
        sp.add_argument("--p-col", type=float, default=600.0)
        sp.add_argument("--time-limit", type=float, default=60.0)
        sp.add_argument("--mipgap", type=float, default=0.01)
        sp.add_argument("--gurobi-python",
                        default=os.environ.get("GUROBI_PYTHON", sys.executable))
        sp.add_argument("--out", default="out/sweeps")

    p1 = sub.add_parser("baselines"); common(p1)
    p2 = sub.add_parser("pareto"); common(p2)
    p2.add_argument("--config", default="mixtral")
    p2.add_argument("--pmax-list", type=float, nargs="+",
                    default=[900, 1100, 1300, 1600, 2000, 3000])
    p3 = sub.add_parser("lambda"); common(p3)
    p3.add_argument("--config", default="mixtral")
    p3.add_argument("--lambda-list", type=float, nargs="+",
                    default=[1.0, 0.5, 0.2, 0.05])
    p4 = sub.add_parser("arch"); common(p4)
    p4.add_argument("--config", default="mixtral")

    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"baselines": cmd_baselines, "pareto": cmd_pareto,
     "lambda": cmd_lambda, "arch": cmd_arch}[a.cmd](a)


if __name__ == "__main__":
    main()
