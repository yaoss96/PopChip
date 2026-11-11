#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run and audit the scaling ILP with its certified-latency Stage-2 tie-break.

This is deliberately a separate experiment driver.  It does not overwrite the
historical scaling/proxy solutions.  Each pair is solved with the same Stage-1
settings as ``run_scaling_prior_proxies.py`` (30 s by default, 1% MIP gap,
seed 1, one thread), followed by a 30 s communication-energy tie-break under
the Stage-1 incumbent plus a 1e-6 relative latency relaxation.

Examples (Gurobi's Python 3.7 interpreter)::

  python src/run_scaling_ilp_secondary.py --experts 8 --mesh 6
  python src/run_scaling_ilp_secondary.py --aggregate
"""

from __future__ import print_function

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import sys
import time

import gurobipy as gp

import evaluate
import ilp_layout
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


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def relpath(path):
    return os.path.relpath(os.path.abspath(path), ROOT_DIR).replace("\\", "/")


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


def site_types(solution):
    return evaluate.site_types_from_solution(solution)


def independent_placement_metrics(params, solution):
    types = site_types(solution)
    task_sites = dict((task, tuple(anchor))
                      for task, anchor in solution["task_site"].items())
    library = params["library"]
    interconnect = params["interconnect"]
    byte_hop = 0.0
    comm_energy = 0.0
    for edge in params["edges"]:
        source = task_sites[edge["src"]]
        target = task_sites[edge["dst"]]
        horizontal = (abs(source[1] - target[1]) +
                      abs(source[2] - target[2]))
        vertical = abs(source[0] - target[0])
        volume = float(edge["vol"])
        byte_hop += volume * (horizontal + vertical)
        comm_energy += (8.0 * volume *
                        (float(interconnect["eps_h"]) * horizontal +
                         float(interconnect["eps_v"]) * vertical))
    area = sum(float(library[kind]["area"]) for kind in types.values())
    power = sum(float(library[kind]["p_peak"]) for kind in types.values())
    return {
        "byte_hop": byte_hop,
        "energy_comm_J": comm_energy,
        "area_mm2": area,
        "power_W": power,
    }


def audit_solution(params, solution, experts, mesh):
    budgets = budgets_for(experts)
    errors = []
    types = site_types(solution)
    task_sites = solution.get("task_site", {})
    expected_tasks = set(params["tasks"])
    actual_tasks = set(task_sites)
    if actual_tasks != expected_tasks:
        errors.append("task set differs: missing=%s extra=%s" %
                      (sorted(expected_tasks - actual_tasks),
                       sorted(actual_tasks - expected_tasks)))
    if len(types) != len(solution.get("sites", [])):
        errors.append("duplicate site anchor in serialized solution")
    if solution.get("grid") != {"L": 1, "X": mesh, "Y": mesh}:
        errors.append("grid metadata mismatch")

    feasible, feasibility_reasons = evaluate.check_feasible(
        params, task_sites, types, solution["grid"], budgets)
    if not feasible:
        errors.extend(feasibility_reasons)

    counts = dict((kind, 0) for kind in params["library"])
    for kind in types.values():
        counts[kind] = counts.get(kind, 0) + 1
    for kind, expected in budgets["nmin"].items():
        if counts.get(kind, 0) != expected:
            errors.append("%s count %d != exact BOM %d" %
                          (kind, counts.get(kind, 0), expected))

    metrics = independent_placement_metrics(params, solution)
    if metrics["area_mm2"] > budgets["a_max"] + 1.0e-6:
        errors.append("area budget exceeded")
    if metrics["power_W"] > budgets["p_max"] + 1.0e-6:
        errors.append("package-power budget exceeded")

    library = params["library"]
    columns = {}
    for anchor, kind in types.items():
        density = (float(library[kind]["p_peak"]) /
                   (int(library[kind]["w"]) * int(library[kind]["h"])))
        for cell in evaluate._footprint(kind, anchor, library):
            column = (cell[1], cell[2])
            columns[column] = columns.get(column, 0.0) + density
    max_column_power = max(columns.values()) if columns else 0.0
    if max_column_power > budgets["p_col"] + 1.0e-6:
        errors.append("per-column power budget exceeded")

    c11_c13 = fixed_surrogate.independent_c11_c13(params, solution)
    primary = float(solution["primary_incumbent_cycles"])
    relative_relaxation = 1.0e-6
    absolute_relaxation = max(1.0e-6, abs(primary) * relative_relaxation)
    cap = primary + absolute_relaxation
    stage2_t = float(solution["predicted_makespan_cycles"])
    numeric_tolerance = max(1.0e-4, abs(cap) * 1.0e-9)
    if stage2_t > cap + numeric_tolerance:
        errors.append("Stage-2 T %.9f exceeds cap %.9f" % (stage2_t, cap))
    if c11_c13 > cap + numeric_tolerance:
        errors.append("independent C11-C13 %.9f exceeds cap %.9f" %
                      (c11_c13, cap))
    if solution.get("secondary_status") is None:
        errors.append("Stage 2 did not run")

    reported_checks = {
        "byte_hop": ("byte_hop", 1.0e-6),
        "energy_comm_J": ("energy_comm_J", 1.0e-15),
        "area_mm2": ("total_area_mm2", 1.0e-6),
        "power_W": ("peak_power_W", 1.0e-6),
    }
    for recomputed_key, (reported_key, absolute_tolerance) in reported_checks.items():
        reported = float(solution[reported_key])
        recomputed = float(metrics[recomputed_key])
        tolerance = max(absolute_tolerance, abs(recomputed) * 1.0e-12)
        if abs(reported - recomputed) > tolerance:
            errors.append("reported %s disagrees with independent recomputation" %
                          reported_key)

    return {
        "pass": not errors,
        "errors": errors,
        "all_C1_through_C9_checked": feasible and not feasibility_reasons,
        "exact_BOM_counts": counts,
        "budgets": budgets,
        "area_slack_mm2": budgets["a_max"] - metrics["area_mm2"],
        "package_power_slack_W": budgets["p_max"] - metrics["power_W"],
        "max_column_power_W": max_column_power,
        "column_power_slack_W": budgets["p_col"] - max_column_power,
        "independent_metrics": metrics,
        "independent_c11_c13_cycles": c11_c13,
        "stage1_incumbent_cycles": primary,
        "stage2_latency_relaxation_relative": relative_relaxation,
        "stage2_latency_relaxation_cycles": absolute_relaxation,
        "stage2_latency_cap_cycles": cap,
        "stage2_T_cycles": stage2_t,
        "stage2_T_cap_slack_cycles": cap - stage2_t,
        "independent_c11_c13_cap_slack_cycles": cap - c11_c13,
    }


def baseline_paths(baseline_dir, experts, mesh, tap_seed):
    stem = "e%03d_m%02d" % (experts, mesh)
    return {
        "stage1_ilp": os.path.join(baseline_dir, stem + "_ilp.json"),
        "floorplet": os.path.join(baseline_dir, stem + "_floorplet.json"),
        "tap": os.path.join(baseline_dir,
                            stem + "_tap_seed%d.json" % tap_seed),
    }


def selected_tap_seeds(comparison_csv):
    selected = {}
    with open(comparison_csv, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            selected[int(row["experts"])] = int(row["tap_seed"])
    return selected


def comparison(params, candidate, paths):
    candidate_metrics = independent_placement_metrics(params, candidate)
    output = {
        "stage2_ilp": {
            "path": None,
            "sha256": None,
            "byte_hop": candidate_metrics["byte_hop"],
            "energy_comm_J": candidate_metrics["energy_comm_J"],
        }
    }
    for method, path in paths.items():
        baseline = read_json(path)
        metrics = independent_placement_metrics(params, baseline)
        output[method] = {
            "path": relpath(path),
            "sha256": sha256(path),
            "byte_hop": metrics["byte_hop"],
            "energy_comm_J": metrics["energy_comm_J"],
            "stage2_over_baseline_byte_hop": (
                candidate_metrics["byte_hop"] / metrics["byte_hop"]
                if metrics["byte_hop"] else None),
            "stage2_byte_hop_reduction_percent": (
                100.0 * (1.0 - candidate_metrics["byte_hop"] /
                         metrics["byte_hop"])
                if metrics["byte_hop"] else None),
        }
    return output


def pair_name(experts, mesh):
    return "e%03d_m%02d" % (experts, mesh)


def solve_pair(args):
    expected = dict(PAIRS)
    if args.experts not in expected or expected[args.experts] != args.mesh:
        raise ValueError("expected one of %s" % (PAIRS,))

    out_dir = os.path.abspath(args.out_dir)
    for child in ("solutions", "audits", "commands", "logs"):
        path = os.path.join(out_dir, child)
        if not os.path.isdir(path):
            os.makedirs(path)
    name = pair_name(args.experts, args.mesh)
    solution_path = os.path.join(out_dir, "solutions",
                                 name + "_ilp_secondary.json")
    audit_path = os.path.join(out_dir, "audits", name + "_audit.json")
    command_path = os.path.join(out_dir, "commands", name + "_command.json")
    if os.path.exists(solution_path) and not args.overwrite:
        raise RuntimeError("refusing to overwrite existing %s" % solution_path)

    params_path = os.path.abspath(os.path.join(
        args.params_dir, "experts_%03d.json" % args.experts))
    params = read_json(params_path)
    comparison_csv = os.path.abspath(args.comparison_csv)
    tap_seed = selected_tap_seeds(comparison_csv)[args.experts]
    existing = baseline_paths(os.path.abspath(args.baseline_dir),
                              args.experts, args.mesh, tap_seed)
    for path in [params_path, comparison_csv] + list(existing.values()):
        if not os.path.isfile(path):
            raise IOError("required input missing: %s" % path)

    options = {
        "time_limit": float(args.primary_time),
        "mipgap": float(args.primary_mipgap),
        "seed": int(args.seed),
        "symmetry_break": True,
        "mode": "certified-latency",
        "secondary_time": float(args.secondary_time),
        "secondary_mipgap": float(args.secondary_mipgap),
        "secondary_latency_relax": 1.0e-6,
        "certified_t_ref": 1.0,
        "threads": int(args.threads),
    }
    command_record = {
        "started_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "gurobi_version": list(gp.gurobi.version()),
        "experts": args.experts,
        "mesh": [1, args.mesh, args.mesh],
        "budgets": budgets_for(args.experts),
        "solver_options": options,
        "inputs": {},
    }
    source_inputs = [
        os.path.abspath(__file__),
        os.path.join(SCRIPT_DIR, "ilp_layout.py"),
        os.path.join(SCRIPT_DIR, "evaluate.py"),
        os.path.join(SCRIPT_DIR, "verify_fixed_surrogate_solution.py"),
        params_path,
        comparison_csv,
    ] + list(existing.values())
    for path in source_inputs:
        command_record["inputs"][relpath(path)] = sha256(path)
    write_json(command_path, command_record)

    print("[run] %s Stage1=%.1fs/%.3g Stage2=%.1fs/%.3g seed=%d threads=%d" %
          (name, args.primary_time, args.primary_mipgap,
           args.secondary_time, args.secondary_mipgap,
           args.seed, args.threads), flush=True)
    gp.setParam("Threads", int(args.threads))
    started = time.time()
    solution, stats = ilp_layout.build_and_solve(
        params, 1, args.mesh, args.mesh, budgets_for(args.experts), options,
        T_hat=None, verbose=True)
    driver_wall = time.time() - started
    if solution is None:
        command_record["finished_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
        command_record["driver_wall_s"] = driver_wall
        command_record["failed_stats"] = stats
        write_json(command_path, command_record)
        raise RuntimeError("no feasible ILP solution for %s" % name)

    write_json(solution_path, solution)
    audited = audit_solution(params, solution, args.experts, args.mesh)
    compared = comparison(params, solution, existing)
    compared["stage2_ilp"]["path"] = relpath(solution_path)
    compared["stage2_ilp"]["sha256"] = sha256(solution_path)
    audit = {
        "pair": name,
        "experts": args.experts,
        "mesh": args.mesh,
        "pass": audited["pass"],
        "validation": audited,
        "comparison": compared,
        "solver": {
            "primary_wall_s": solution.get("primary_wall_s"),
            "primary_status": solution.get("primary_status"),
            "primary_mip_gap": solution.get("primary_mip_gap"),
            "primary_incumbent_cycles": solution.get("primary_incumbent_cycles"),
            "primary_objective_bound_cycles": solution.get(
                "primary_objective_bound_cycles"),
            "first_feasible_s": solution.get("time_to_first_feasible_s"),
            "secondary_wall_s": solution.get("secondary_wall_s"),
            "secondary_status": solution.get("secondary_status"),
            "secondary_mip_gap": solution.get("secondary_mip_gap"),
            "total_solver_wall_s": solution.get("solver_wall_s"),
            "driver_wall_s": driver_wall,
        },
        "artifacts": {
            "solution": relpath(solution_path),
            "solution_sha256": sha256(solution_path),
            "command": relpath(command_path),
            "params": relpath(params_path),
            "params_sha256": sha256(params_path),
        },
    }
    write_json(audit_path, audit)
    command_record["finished_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    command_record["driver_wall_s"] = driver_wall
    command_record["solution"] = relpath(solution_path)
    command_record["solution_sha256"] = sha256(solution_path)
    command_record["audit"] = relpath(audit_path)
    write_json(command_path, command_record)
    print("[audit] %s byte_hop=%.0f primary_gap=%.6g secondary_gap=%.6g PASS=%s" %
          (name, audited["independent_metrics"]["byte_hop"],
           solution.get("primary_mip_gap") or 0.0,
           solution.get("secondary_mip_gap") or 0.0,
           audited["pass"]), flush=True)
    if not audited["pass"]:
        print("[audit] errors: %s" % "; ".join(audited["errors"]),
              flush=True)
        return 1
    return 0


def aggregate(args):
    out_dir = os.path.abspath(args.out_dir)
    rows = []
    audits = []
    for experts, mesh in PAIRS:
        name = pair_name(experts, mesh)
        path = os.path.join(out_dir, "audits", name + "_audit.json")
        if not os.path.isfile(path):
            raise IOError("missing audit: %s" % path)
        audit = read_json(path)
        audits.append(audit)
        validation = audit["validation"]
        candidate = audit["comparison"]["stage2_ilp"]
        solver = audit["solver"]
        row = {
            "experts": experts,
            "mesh": mesh,
            "pass": audit["pass"],
            "primary_wall_s": solver["primary_wall_s"],
            "primary_gap": solver["primary_mip_gap"],
            "primary_incumbent_cycles": solver["primary_incumbent_cycles"],
            "primary_bound_cycles": solver["primary_objective_bound_cycles"],
            "secondary_wall_s": solver["secondary_wall_s"],
            "secondary_gap": solver["secondary_mip_gap"],
            "stage2_T_cycles": validation["stage2_T_cycles"],
            "stage2_latency_cap_cycles": validation["stage2_latency_cap_cycles"],
            "stage2_c11_c13_cycles": validation["independent_c11_c13_cycles"],
            "stage2_byte_hop": candidate["byte_hop"],
            "stage1_ilp_byte_hop": audit["comparison"]["stage1_ilp"]["byte_hop"],
            "floorplet_byte_hop": audit["comparison"]["floorplet"]["byte_hop"],
            "tap_byte_hop": audit["comparison"]["tap"]["byte_hop"],
            "reduction_vs_stage1_percent": audit["comparison"]["stage1_ilp"][
                "stage2_byte_hop_reduction_percent"],
            "reduction_vs_floorplet_percent": audit["comparison"]["floorplet"][
                "stage2_byte_hop_reduction_percent"],
            "reduction_vs_tap_percent": audit["comparison"]["tap"][
                "stage2_byte_hop_reduction_percent"],
            "area_mm2": validation["independent_metrics"]["area_mm2"],
            "power_W": validation["independent_metrics"]["power_W"],
            "max_column_power_W": validation["max_column_power_W"],
            "solution_sha256": audit["artifacts"]["solution_sha256"],
        }
        rows.append(row)

    fields = list(rows[0].keys())
    csv_path = os.path.join(out_dir, "secondary_summary.csv")
    temporary = csv_path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    manifest_path = os.path.join(out_dir, "secondary_audit.json")
    write_json(manifest_path, {
        "pass": all(item["pass"] for item in audits),
        "expected_pairs": [list(pair) for pair in PAIRS],
        "completed_pairs": len(audits),
        "claim_boundary": (
            "solver-only two-stage certified-latency layouts; Stage 2 minimizes "
            "communication energy under the Stage-1 incumbent plus 1e-6 "
            "relative latency relaxation; no LEGOSim was run by this driver"),
        "rows": rows,
    })
    print("[aggregate] %d/%d PASS=%s" %
          (sum(1 for item in audits if item["pass"]), len(PAIRS),
           all(item["pass"] for item in audits)), flush=True)
    print("[aggregate] %s" % csv_path, flush=True)
    return 0 if all(item["pass"] for item in audits) else 1


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experts", type=int)
    ap.add_argument("--mesh", type=int)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--params-dir", default=os.path.join(
        ROOT_DIR, "out", "scaling_qap_experiment", "params"))
    ap.add_argument("--baseline-dir", default=os.path.join(
        ROOT_DIR, "out", "scaling_prior_proxy_experiment_20260731",
        "solutions"))
    ap.add_argument("--comparison-csv", default=os.path.join(
        ROOT_DIR, "out", "scaling_prior_proxy_experiment_20260731",
        "scaling_proxy_comparison.csv"))
    ap.add_argument("--out-dir", default=os.path.join(
        ROOT_DIR, "out", "scaling_ilp_secondary_20260801"))
    ap.add_argument("--primary-time", type=float, default=30.0)
    ap.add_argument("--primary-mipgap", type=float, default=0.01)
    ap.add_argument("--secondary-time", type=float, default=30.0)
    ap.add_argument("--secondary-mipgap", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.aggregate:
        return aggregate(args)
    if args.experts is None or args.mesh is None:
        raise ValueError("--experts and --mesh are required unless --aggregate")
    return solve_pair(args)


if __name__ == "__main__":
    sys.exit(main())
