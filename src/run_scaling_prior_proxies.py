#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare ILP, Floorplet, and TAP proxies on the scaling instances.

All methods are scored with the same independently recomputed C11--C13
analytical surrogate that the ILP certifies.  ``evaluate.py`` is also run as a
strict feasibility and list-schedule audit, but that intentionally different
execution metric is not used for the scaling comparison.  The two proxies
inherit the ILP solution's exact BOM and immutable task bundles, then change
physical placement only.  They are objective-matched reimplementations, not
the original Floorplet or TAP-2.5D toolchains.

Run with Gurobi's Python interpreter from the repository root::

  python \\
      src\\run_scaling_prior_proxies.py
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import sys

import gurobipy as gp

import ilp_layout
import matched_layout_baseline as matched
import run_optimized_prior_proxies as proxies
import verify_fixed_surrogate_solution as fixed_surrogate


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PAIRS = ((8, 6), (16, 8), (32, 12), (64, 12), (128, 16))


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        os.makedirs(parent)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path, rows):
    fields = [
        "experts", "mesh", "gpu_l", "gpu_s", "area_mm2", "power_W",
        "ilp_c11_c13_cycles", "ilp_solver_cycles", "ilp_solver_delta_cycles",
        "ilp_list_schedule_cycles", "ilp_status", "ilp_gap",
        "floorplet_c11_c13_cycles", "floorplet_list_schedule_cycles",
        "floorplet_F_FP", "floorplet_status", "floorplet_gap",
        "floorplet_wall_s", "tap_c11_c13_cycles",
        "tap_list_schedule_cycles", "tap_selection_cost", "tap_seed",
        "tap_wall_s",
        "floorplet_over_ilp", "tap_over_ilp",
    ]
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def budgets_for(experts):
    gpu_l = experts // 4
    gpu_s = experts - gpu_l
    return {
        "a_max": 90.0 + 110.0 * experts + 290.0 * gpu_l,
        "a_max_layer": None,
        "p_max": 65.0 + 90.0 * experts + 210.0 * gpu_l,
        "p_col": 300.0,
        "e_bar": None,
        "nmax": {"GPU-L": gpu_l, "GPU-S": gpu_s, "CPU": 1},
        "nmin": {"GPU-L": gpu_l, "GPU-S": gpu_s, "CPU": 1},
    }


def solve_ilp(params, experts, mesh, time_limit):
    opts = {
        "time_limit": float(time_limit),
        "mipgap": 0.01,
        "seed": 1,
        "symmetry_break": True,
        "mode": "certified-latency",
        "secondary_time": 0.0,
        "secondary_mipgap": 0.02,
        "certified_t_ref": 1.0,
        "threads": 1,
    }
    gp.setParam("Threads", 1)
    solution, stats = ilp_layout.build_and_solve(
        params, 1, mesh, mesh, budgets_for(experts), opts,
        T_hat=None, verbose=False)
    if solution is None:
        raise RuntimeError("ILP produced no solution for E=%d" % experts)
    return solution, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params-dir",
        default=os.path.join(ROOT_DIR, "out", "scaling_qap_experiment",
                             "params"))
    parser.add_argument(
        "--out-dir",
        default=os.path.join(ROOT_DIR, "out",
                             "scaling_prior_proxy_experiment"))
    parser.add_argument("--ilp-time-limit", type=float, default=30.0)
    parser.add_argument("--floorplet-time-limit", type=float, default=30.0)
    parser.add_argument("--tap-wall-budget", type=float, default=10.0)
    parser.add_argument("--tap-seeds", default="11,29,47")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args(argv)

    params_dir = os.path.abspath(args.params_dir)
    out_dir = os.path.abspath(args.out_dir)
    solutions_dir = os.path.join(out_dir, "solutions")
    if not os.path.isdir(solutions_dir):
        os.makedirs(solutions_dir)
    seeds = [int(item) for item in args.tap_seeds.split(",") if item]
    rows = []

    for experts, mesh in PAIRS:
        prefix = "e%03d_m%02d" % (experts, mesh)
        params_path = os.path.join(params_dir, "experts_%03d.json" % experts)
        params = read_json(params_path)
        ilp_path = os.path.join(solutions_dir, prefix + "_ilp.json")
        if args.reuse and os.path.isfile(ilp_path):
            ilp_solution = read_json(ilp_path)
        else:
            print("[ILP] E=%d mesh=%dx%d" % (experts, mesh, mesh), flush=True)
            ilp_solution, _ = solve_ilp(
                params, experts, mesh, args.ilp_time_limit)
            write_json(ilp_path, ilp_solution)

        state = matched._source_state(params, ilp_solution)
        ilp_score = state["source_score"]

        floor_path = os.path.join(solutions_dir, prefix + "_floorplet.json")
        floor_meta_path = os.path.join(
            solutions_dir, prefix + "_floorplet_meta.json")
        if (args.reuse and os.path.isfile(floor_path) and
                os.path.isfile(floor_meta_path)):
            floor_solution = read_json(floor_path)
            floor_meta = read_json(floor_meta_path)
        else:
            print("[Floorplet] E=%d" % experts, flush=True)
            gp.setParam("Threads", 1)
            floor_solution, floor_meta = proxies.exact_floorplet(
                params, ilp_solution, state, args.floorplet_time_limit,
                seed=1, verbose=False, threads=1)
            if floor_solution is None:
                raise RuntimeError("Floorplet produced no solution for E=%d" %
                                   experts)
            write_json(floor_path, floor_solution)
            write_json(floor_meta_path, floor_meta)

        tap_runs = []
        for seed in seeds:
            tap_path = os.path.join(
                solutions_dir, prefix + "_tap_seed%d.json" % seed)
            tap_meta_path = os.path.join(
                solutions_dir, prefix + "_tap_seed%d_meta.json" % seed)
            if (args.reuse and os.path.isfile(tap_path) and
                    os.path.isfile(tap_meta_path)):
                tap_solution = read_json(tap_path)
                tap_meta = read_json(tap_meta_path)
            else:
                print("[TAP] E=%d seed=%d" % (experts, seed), flush=True)
                ranges = proxies.deterministic_tap_ranges(params, state)
                tap_solution, tap_meta = proxies.tap_sa(
                    params, ilp_solution, state, args.tap_wall_budget, seed,
                    0.7, ranges)
                write_json(tap_path, tap_solution)
                write_json(tap_meta_path, tap_meta)
            tap_runs.append((tap_meta["selection_cost"], seed,
                             tap_solution, tap_meta))
        _, tap_seed, tap_solution, tap_meta = min(
            tap_runs, key=lambda item: (item[0], item[1]))

        # The scaling experiment reports the exact C11--C13 surrogate that the
        # ILP certifies.  evaluate.py intentionally models a different,
        # resource-constrained execution schedule and special-cases
        # intra-chiplet communication; mixing that number with the ILP bound
        # would attach the certificate to the wrong objective.
        ilp_cycles = fixed_surrogate.independent_c11_c13(
            params, ilp_solution)
        floor_cycles = fixed_surrogate.independent_c11_c13(
            params, floor_solution)
        tap_cycles = fixed_surrogate.independent_c11_c13(
            params, tap_solution)
        ilp_solver_cycles = float(ilp_solution["primary_incumbent_cycles"])
        solver_delta = ilp_cycles - ilp_solver_cycles
        if abs(solver_delta) > 1.0e-3:
            raise RuntimeError(
                "independent C11--C13 score disagrees with ILP by %.6f cycles "
                "for E=%d" % (solver_delta, experts))
        row = {
            "experts": experts,
            "mesh": mesh,
            "gpu_l": experts // 4,
            "gpu_s": experts - experts // 4,
            "area_mm2": ilp_score["total_area_mm2"],
            "power_W": ilp_score["peak_power_W"],
            "ilp_c11_c13_cycles": ilp_cycles,
            "ilp_solver_cycles": ilp_solver_cycles,
            "ilp_solver_delta_cycles": solver_delta,
            "ilp_list_schedule_cycles": ilp_score["makespan_cycles"],
            "ilp_status": ilp_solution.get("primary_status"),
            "ilp_gap": ilp_solution.get("primary_mip_gap"),
            "floorplet_c11_c13_cycles": floor_cycles,
            "floorplet_list_schedule_cycles": (
                floor_solution["strict_evaluate"]["makespan_cycles"]),
            "floorplet_F_FP": floor_meta.get("F_FP"),
            "floorplet_status": floor_meta.get("status"),
            "floorplet_gap": floor_meta.get("mip_gap"),
            "floorplet_wall_s": floor_meta.get("wall_s"),
            "tap_c11_c13_cycles": tap_cycles,
            "tap_list_schedule_cycles": (
                tap_solution["strict_evaluate"]["makespan_cycles"]),
            "tap_selection_cost": tap_meta.get("selection_cost"),
            "tap_seed": tap_seed,
            "tap_wall_s": tap_meta.get("wall_s"),
            "floorplet_over_ilp": floor_cycles / ilp_cycles,
            "tap_over_ilp": tap_cycles / ilp_cycles,
        }
        rows.append(row)
        write_csv(os.path.join(out_dir, "scaling_proxy_comparison.csv"), rows)
        print("[row] E=%d ILP=%.3f Floorplet=%.3f TAP=%.3f Mcycles" %
              (experts, ilp_cycles / 1.0e6, floor_cycles / 1.0e6,
               tap_cycles / 1.0e6), flush=True)

    metadata = {
        "claim_boundary": (
            "solver-only analytical comparison; proxies inherit exact ILP "
            "BOM and immutable task bundles; not original toolchains"),
        "common_metric": (
            "independent exact C11--C13 surrogate: maximum of the DAG "
            "longest path with unconditional per-edge t0/serialization and "
            "the maximum per-chiplet busy time"),
        "list_schedule_metric": (
            "evaluate.py makespan_cycles retained only as an audit field; "
            "it is not the certified scaling-table metric"),
        "ilp_time_limit_s": args.ilp_time_limit,
        "floorplet_time_limit_s": args.floorplet_time_limit,
        "tap_wall_budget_s_per_seed": args.tap_wall_budget,
        "tap_total_wall_budget_s": args.tap_wall_budget * len(seeds),
        "tap_seeds": seeds,
        "threads": 1,
        "pairs": list(PAIRS),
    }
    write_json(os.path.join(out_dir, "metadata.json"), metadata)
    print("[done] %s" % out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
