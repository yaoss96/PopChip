#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate placement proxies against the Stage-2 scaling ILP layouts.

This driver is intentionally separate from the historical proxy experiment.
For every requested expert/mesh pair it:

* reads the immutable Stage-2 ILP solution;
* preserves its exact BOM and every complete task bundle;
* runs the exact Floorplet-inspired MILP for 30 seconds;
* runs TAP-Spread SA for three 10-second seeds and selects the minimum
  ``selection_cost`` (seed is only a deterministic tie-break);
* independently rechecks strict feasibility, the exact BOM, the complete
  bundle multiset, C11--C13, and traffic-weighted byte-hop; and
* writes solutions, optimizer metadata/traces, logs, audits, command records,
  and a SHA-256 manifest without invoking LEGOSim.

Run with the Gurobi Python interpreter from the repository root, for example::

  python \\
      src\\run_scaling_secondary_proxies.py \\
      --pairs 8:6 16:8 32:12

Later pairs can be appended without rerunning completed pairs::

  python \\
      src\\run_scaling_secondary_proxies.py \\
      --pairs 64:12 128:16 --reuse
"""

from __future__ import print_function

import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
import platform
import sys
import time
import traceback

import gurobipy as gp

import evaluate
import matched_layout_baseline as matched
import run_optimized_prior_proxies as proxies
import verify_fixed_surrogate_solution as fixed_surrogate


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CANONICAL_PAIRS = ((8, 6), (16, 8), (32, 12), (64, 12), (128, 16))
INITIAL_PAIRS = ((8, 6), (16, 8), (32, 12))
DEFAULT_SEEDS = (11, 29, 47)
FLOAT_TOL = 1.0e-6


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


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


def write_csv(path, fields, rows):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_text(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
    os.replace(temporary, path)


def append_log(path, message):
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write("[%s] %s\n" % (utc_now(), message))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relpath(path):
    return os.path.relpath(os.path.abspath(path), ROOT_DIR).replace("\\", "/")


def pair_name(experts, mesh):
    return "e%03d_m%02d" % (experts, mesh)


def parse_pairs(values):
    if not values:
        return list(INITIAL_PAIRS)
    tokens = []
    for value in values:
        tokens.extend(item for item in value.replace(",", " ").split()
                      if item)
    pairs = []
    expected = dict(CANONICAL_PAIRS)
    for token in tokens:
        separator = ":" if ":" in token else "x" if "x" in token else None
        if separator is None:
            raise ValueError("pair %r must use experts:mesh (for example 32:12)" %
                             token)
        left, right = token.lower().split(separator, 1)
        pair = (int(left), int(right))
        if pair[0] not in expected or expected[pair[0]] != pair[1]:
            raise ValueError("unsupported pair %s; expected one of %s" %
                             (pair, CANONICAL_PAIRS))
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def type_counts(solution):
    counts = collections.Counter(
        site["type"] for site in solution.get("sites", []))
    return dict((kind, counts[kind]) for kind in sorted(counts))


def declared_bundle_signature(solution):
    signature = []
    for site in solution.get("sites", []):
        signature.append([str(site["type"]),
                          sorted(str(task) for task in site.get("tasks", []))])
    return sorted(signature, key=lambda item: (item[0], item[1]))


def mapped_bundle_signature(solution):
    types = evaluate.site_types_from_solution(solution)
    tasks_at = dict((anchor, []) for anchor in types)
    for task, raw_anchor in solution.get("task_site", {}).items():
        anchor = tuple(int(value) for value in raw_anchor)
        tasks_at.setdefault(anchor, []).append(str(task))
    signature = []
    for anchor, kind in types.items():
        signature.append([str(kind), sorted(tasks_at.get(anchor, []))])
    return sorted(signature, key=lambda item: (item[0], item[1]))


def independent_byte_hop(params, solution):
    task_site = dict((task, tuple(int(value) for value in anchor))
                     for task, anchor in solution["task_site"].items())
    total = 0.0
    for edge in params["edges"]:
        source = task_site[edge["src"]]
        target = task_site[edge["dst"]]
        hops = sum(abs(source[axis] - target[axis]) for axis in range(3))
        total += float(edge["vol"]) * hops
    return total


def independent_solution_audit(params, source, candidate):
    """Audit a proxy without trusting its serialized score or metadata."""
    errors = []
    expected_tasks = set(params["tasks"])
    actual_tasks = set(candidate.get("task_site", {}))
    if actual_tasks != expected_tasks:
        errors.append("task set mismatch: missing=%s extra=%s" %
                      (sorted(expected_tasks - actual_tasks),
                       sorted(actual_tasks - expected_tasks)))

    if candidate.get("grid") != source.get("grid"):
        errors.append("grid differs from Stage-2 ILP source")
    if candidate.get("budgets") != source.get("budgets"):
        errors.append("budgets differ from Stage-2 ILP source")

    source_types = evaluate.site_types_from_solution(source)
    candidate_types = evaluate.site_types_from_solution(candidate)
    if len(candidate_types) != len(candidate.get("sites", [])):
        errors.append("duplicate candidate site anchor")
    if len(source_types) != len(source.get("sites", [])):
        errors.append("duplicate source site anchor")

    source_bom = type_counts(source)
    candidate_bom = type_counts(candidate)
    bom_equal = candidate_bom == source_bom
    if not bom_equal:
        errors.append("exact BOM changed: source=%s candidate=%s" %
                      (source_bom, candidate_bom))

    source_declared = declared_bundle_signature(source)
    source_mapped = mapped_bundle_signature(source)
    candidate_declared = declared_bundle_signature(candidate)
    candidate_mapped = mapped_bundle_signature(candidate)
    source_serialization_consistent = source_declared == source_mapped
    candidate_serialization_consistent = (
        candidate_declared == candidate_mapped)
    bundles_equal = candidate_mapped == source_mapped
    if not source_serialization_consistent:
        errors.append("source sites[].tasks disagrees with source task_site")
    if not candidate_serialization_consistent:
        errors.append("candidate sites[].tasks disagrees with candidate task_site")
    if not bundles_equal:
        errors.append("immutable complete bundle multiset changed")

    strict_score = None
    strict_reasons = []
    try:
        strict_score, strict_reasons = matched._strict_score(
            params, candidate["task_site"], candidate_types,
            candidate["grid"], candidate["budgets"])
        if strict_reasons:
            errors.extend("strict evaluator: " + reason
                          for reason in strict_reasons)
    except Exception as exc:
        errors.append("strict evaluator raised: %s" % exc)

    c11_c13 = None
    byte_hop = None
    try:
        c11_c13 = float(fixed_surrogate.independent_c11_c13(
            params, candidate))
    except Exception as exc:
        errors.append("independent C11--C13 raised: %s" % exc)
    try:
        byte_hop = float(independent_byte_hop(params, candidate))
    except Exception as exc:
        errors.append("independent byte-hop raised: %s" % exc)

    reported = candidate.get("strict_evaluate") or {}
    if reported and not reported.get("feasible", False):
        errors.append("serialized strict_evaluate is not feasible")
    if byte_hop is not None and reported.get("byte_hop") is not None:
        tolerance = max(FLOAT_TOL, abs(byte_hop) * 1.0e-12)
        if abs(float(reported["byte_hop"]) - byte_hop) > tolerance:
            errors.append("serialized byte-hop disagrees with independent value")

    source_area = float(sum(params["library"][kind]["area"]
                            for kind in source_types.values()))
    candidate_area = float(sum(params["library"][kind]["area"]
                               for kind in candidate_types.values()))
    source_power = float(sum(params["library"][kind]["p_peak"]
                             for kind in source_types.values()))
    candidate_power = float(sum(params["library"][kind]["p_peak"]
                                for kind in candidate_types.values()))
    if abs(source_area - candidate_area) > FLOAT_TOL:
        errors.append("BOM area changed")
    if abs(source_power - candidate_power) > FLOAT_TOL:
        errors.append("BOM peak-power sum changed")

    return {
        "pass": not errors,
        "errors": errors,
        "strict_feasible": bool(strict_score and strict_score.get("feasible")
                                and not strict_reasons),
        "strict_reasons": strict_reasons,
        "exact_BOM_equal": bom_equal,
        "source_BOM": source_bom,
        "candidate_BOM": candidate_bom,
        "immutable_bundle_multiset_equal": bundles_equal,
        "source_serialization_consistent": source_serialization_consistent,
        "candidate_serialization_consistent": (
            candidate_serialization_consistent),
        "bundle_multiset_sha256": canonical_hash(candidate_mapped),
        "source_bundle_multiset_sha256": canonical_hash(source_mapped),
        "independent_c11_c13_cycles": c11_c13,
        "independent_byte_hop": byte_hop,
        "independent_area_mm2": candidate_area,
        "independent_peak_power_sum_W": candidate_power,
        "strict_evaluate": strict_score,
    }


def provenance(args, params_path, source_path):
    code_paths = [
        os.path.abspath(__file__),
        os.path.join(SCRIPT_DIR, "run_optimized_prior_proxies.py"),
        os.path.join(SCRIPT_DIR, "matched_layout_baseline.py"),
        os.path.join(SCRIPT_DIR, "evaluate.py"),
        os.path.join(SCRIPT_DIR, "verify_fixed_surrogate_solution.py"),
    ]
    return {
        "params_path": relpath(params_path),
        "params_sha256": sha256(params_path),
        "source_ilp_path": relpath(source_path),
        "source_ilp_sha256": sha256(source_path),
        "source_code_sha256": dict((relpath(path), sha256(path))
                                    for path in code_paths),
        "floorplet_time_limit_s": float(args.floorplet_time_limit),
        "floorplet_seed": 1,
        "floorplet_threads": 1,
        "tap_wall_budget_s_per_seed": float(args.tap_wall_budget),
        "tap_seeds": list(args.tap_seeds),
        "tap_alpha": float(args.tap_alpha),
        "selection_rule": "minimum selection_cost, then minimum seed",
        "legosim_invoked": False,
    }


def verify_reuse(meta, expected, label):
    actual = meta.get("provenance")
    if actual != expected:
        raise RuntimeError(
            "%s cannot be reused: provenance/settings/source hash changed" %
            label)


def ensure_absent(paths):
    existing = [path for path in paths if os.path.exists(path)]
    if existing:
        raise RuntimeError("refusing to overwrite without --reuse: %s" %
                           ", ".join(existing))


def run_pair(args, experts, mesh, paths):
    name = pair_name(experts, mesh)
    source_path = os.path.join(args.source_dir, "solutions",
                               name + "_ilp_secondary.json")
    params_path = os.path.join(args.params_dir,
                               "experts_%03d.json" % experts)
    if not os.path.isfile(source_path):
        raise IOError("Stage-2 ILP source is not available: %s" % source_path)
    if not os.path.isfile(params_path):
        raise IOError("parameter file is not available: %s" % params_path)
    params = read_json(params_path)
    source = read_json(source_path)
    state = matched._source_state(params, source)
    source_audit = independent_solution_audit(params, source, source)
    if not source_audit["pass"]:
        raise RuntimeError("Stage-2 source failed independent audit: %s" %
                           "; ".join(source_audit["errors"]))

    provenance_record = provenance(args, params_path, source_path)
    driver_log = os.path.join(paths["logs"], name + "_driver.log")
    command_path = os.path.join(paths["commands"], name + "_command.json")
    command = {
        "started_utc": utc_now(),
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "gurobi_version": list(gp.gurobi.version()),
        "pair": [experts, mesh],
        "protocol": provenance_record,
    }

    floor_solution_path = os.path.join(
        paths["solutions"], name + "_floorplet.json")
    floor_meta_path = os.path.join(
        paths["meta"], name + "_floorplet_meta.json")
    floor_log_path = os.path.join(
        paths["logs"], name + "_floorplet_gurobi.log")
    pair_audit_path = os.path.join(paths["audits"],
                                   name + "_proxy_audit.json")

    if args.reuse and os.path.isfile(floor_solution_path) and os.path.isfile(
            floor_meta_path):
        floor_solution = read_json(floor_solution_path)
        floor_sidecar = read_json(floor_meta_path)
        verify_reuse(floor_sidecar, provenance_record, name + " Floorplet")
        floor_meta = floor_sidecar["optimizer_metadata"]
        append_log(driver_log, "reused Floorplet artifacts")
    else:
        ensure_absent([floor_solution_path, floor_meta_path, floor_log_path])
        append_log(driver_log,
                   "start Floorplet exact MILP: limit=%.3fs seed=1 threads=1" %
                   args.floorplet_time_limit)
        print("[Floorplet] %s limit=%.1fs" %
              (name, args.floorplet_time_limit), flush=True)
        gp.setParam("Threads", 1)
        gp.setParam("LogFile", floor_log_path)
        try:
            floor_solution, floor_meta = proxies.exact_floorplet(
                params, source, state, args.floorplet_time_limit,
                seed=1, verbose=True, threads=1)
        finally:
            gp.setParam("LogFile", "")
        if floor_solution is None:
            raise RuntimeError("Floorplet returned no incumbent for %s" % name)
        floor_solution["reproducibility"] = provenance_record
        write_json(floor_solution_path, floor_solution)
        write_json(floor_meta_path, {
            "method": "floorplet-inspired-exact-MILP",
            "optimizer_metadata": floor_meta,
            "provenance": provenance_record,
            "solution_path": relpath(floor_solution_path),
            "solution_sha256": sha256(floor_solution_path),
        })
        append_log(driver_log,
                   "finish Floorplet F_FP=%.9g gap=%s wall=%.6fs" %
                   (float(floor_meta["F_FP"]), floor_meta.get("mip_gap"),
                    float(floor_meta["wall_s"])))

    tap_runs = []
    for seed in args.tap_seeds:
        tap_solution_path = os.path.join(
            paths["solutions"], name + "_tap_seed%d.json" % seed)
        tap_meta_path = os.path.join(
            paths["meta"], name + "_tap_seed%d_meta.json" % seed)
        tap_log_path = os.path.join(
            paths["logs"], name + "_tap_seed%d.log" % seed)
        if args.reuse and os.path.isfile(tap_solution_path) and os.path.isfile(
                tap_meta_path):
            tap_solution = read_json(tap_solution_path)
            tap_sidecar = read_json(tap_meta_path)
            verify_reuse(tap_sidecar, provenance_record,
                         "%s TAP seed %d" % (name, seed))
            tap_meta = tap_sidecar["optimizer_metadata"]
            append_log(driver_log, "reused TAP seed=%d artifacts" % seed)
        else:
            ensure_absent([tap_solution_path, tap_meta_path, tap_log_path])
            ranges = proxies.deterministic_tap_ranges(params, state)
            append_log(driver_log,
                       "start TAP seed=%d budget=%.3fs alpha=%.3f" %
                       (seed, args.tap_wall_budget, args.tap_alpha))
            print("[TAP] %s seed=%d budget=%.1fs" %
                  (name, seed, args.tap_wall_budget), flush=True)
            tap_solution, tap_meta = proxies.tap_sa(
                params, source, state, args.tap_wall_budget, seed,
                args.tap_alpha, ranges)
            tap_solution["reproducibility"] = provenance_record
            write_json(tap_solution_path, tap_solution)
            write_json(tap_meta_path, {
                "method": "tap-spread-SA",
                "optimizer_metadata": tap_meta,
                "provenance": provenance_record,
                "solution_path": relpath(tap_solution_path),
                "solution_sha256": sha256(tap_solution_path),
            })
            write_json(tap_log_path, {
                "pair": name,
                "seed": seed,
                "wall_budget_s": args.tap_wall_budget,
                "selection_cost": tap_meta["selection_cost"],
                "evaluations": tap_meta["evaluations"],
                "complete_anytime_trace": tap_meta["trace"],
            })
            append_log(driver_log,
                       "finish TAP seed=%d cost=%.12g eval=%d wall=%.6fs" %
                       (seed, float(tap_meta["selection_cost"]),
                        int(tap_meta["evaluations"]),
                        float(tap_meta["wall_s"])))
        tap_runs.append((float(tap_meta["selection_cost"]), int(seed),
                         tap_solution, tap_meta, tap_solution_path,
                         tap_meta_path))

    tap_runs.sort(key=lambda item: (item[0], item[1]))
    _, selected_seed, selected_tap, selected_tap_meta, selected_source_path, _ = (
        tap_runs[0])
    selected_path = os.path.join(paths["solutions"], name + "_tap.json")
    selection_path = os.path.join(paths["meta"],
                                  name + "_tap_selection.json")
    selection_record = {
        "pair": name,
        "selection_rule": "minimum selection_cost, then minimum seed",
        "selected_seed": selected_seed,
        "selected_selection_cost": selected_tap_meta["selection_cost"],
        "selected_seed_solution_path": relpath(selected_source_path),
        "runs": [{
            "seed": item[1],
            "selection_cost": item[0],
            "solution_path": relpath(item[4]),
            "solution_sha256": sha256(item[4]),
            "meta_path": relpath(item[5]),
            "meta_sha256": sha256(item[5]),
        } for item in tap_runs],
        "provenance": provenance_record,
    }
    if os.path.exists(selected_path):
        if not args.reuse:
            raise RuntimeError("refusing to overwrite %s" % selected_path)
        if sha256(selected_path) != sha256(selected_source_path):
            raise RuntimeError("reused TAP selection does not match selected seed")
    else:
        write_json(selected_path, selected_tap)
    selection_record["selected_solution_path"] = relpath(selected_path)
    selection_record["selected_solution_sha256"] = sha256(selected_path)
    if os.path.exists(selection_path) and args.reuse:
        old_selection = read_json(selection_path)
        if old_selection != selection_record:
            raise RuntimeError("reused TAP selection metadata changed")
    else:
        ensure_absent([selection_path])
        write_json(selection_path, selection_record)

    method_records = []
    candidates = [
        ("floorplet", None, floor_solution, floor_solution_path,
         floor_meta.get("F_FP")),
        ("tap", selected_seed, selected_tap, selected_path,
         selected_tap_meta.get("selection_cost")),
    ]
    for method, seed, solution, solution_path, native_objective in candidates:
        audit = independent_solution_audit(params, source, solution)
        if not audit["pass"]:
            raise RuntimeError("%s %s audit failed: %s" %
                               (name, method, "; ".join(audit["errors"])))
        method_records.append({
            "method": method,
            "selected_seed": seed,
            "native_proxy_objective": native_objective,
            "solution_path": relpath(solution_path),
            "solution_sha256": sha256(solution_path),
            "validation": audit,
        })

    source_c11 = float(fixed_surrogate.independent_c11_c13(params, source))
    source_byte_hop = float(independent_byte_hop(params, source))
    pair_audit = {
        "pair": name,
        "experts": experts,
        "mesh": mesh,
        "pass": all(record["validation"]["pass"]
                    for record in method_records),
        "claim_boundary": (
            "solver-only placement-proxy generation and independent audit; "
            "no LEGOSim execution or performance inference"),
        "source_ilp": {
            "path": relpath(source_path),
            "sha256": sha256(source_path),
            "independent_c11_c13_cycles": source_c11,
            "independent_byte_hop": source_byte_hop,
            "BOM": type_counts(source),
            "bundle_multiset_sha256": source_audit[
                "source_bundle_multiset_sha256"],
            "validation": source_audit,
        },
        "proxies": method_records,
        "provenance": provenance_record,
    }
    write_json(pair_audit_path, pair_audit)
    append_log(driver_log,
               "PASS selected TAP seed=%d; independent proxy audits=2/2" %
               selected_seed)

    command["finished_utc"] = utc_now()
    command["status"] = "PASS"
    command["artifacts"] = {
        "audit": relpath(pair_audit_path),
        "audit_sha256": sha256(pair_audit_path),
        "driver_log": relpath(driver_log),
        "floorplet_solution": relpath(floor_solution_path),
        "floorplet_solution_sha256": sha256(floor_solution_path),
        "selected_tap_solution": relpath(selected_path),
        "selected_tap_solution_sha256": sha256(selected_path),
    }
    if os.path.exists(command_path) and args.reuse:
        # Pair command records are immutable evidence.  Reuse validates the
        # artifacts but deliberately does not rewrite the original invocation.
        old_command = read_json(command_path)
        if old_command.get("protocol") != provenance_record:
            raise RuntimeError("reused command provenance changed for %s" % name)
    else:
        ensure_absent([command_path])
        write_json(command_path, command)
    print("[audit] %s proxies=2/2 PASS TAP-seed=%d" %
          (name, selected_seed), flush=True)
    return pair_audit


def aggregate(args, paths, requested_pairs):
    audits = []
    for experts, mesh in CANONICAL_PAIRS:
        path = os.path.join(paths["audits"],
                            pair_name(experts, mesh) + "_proxy_audit.json")
        if os.path.isfile(path):
            audit = read_json(path)
            if not audit.get("pass"):
                raise RuntimeError("stored audit is not PASS: %s" % path)
            audits.append(audit)

    rows = []
    for audit in audits:
        source = audit["source_ilp"]
        by_method = dict((item["method"], item)
                         for item in audit["proxies"])
        floor = by_method["floorplet"]
        tap = by_method["tap"]
        floor_validation = floor["validation"]
        tap_validation = tap["validation"]
        rows.append({
            "experts": audit["experts"],
            "mesh": audit["mesh"],
            "gpu_l": audit["experts"] // 4,
            "gpu_s": audit["experts"] - audit["experts"] // 4,
            "ilp_c11_c13_cycles": source["independent_c11_c13_cycles"],
            "ilp_byte_hop": source["independent_byte_hop"],
            "floorplet_c11_c13_cycles": floor_validation[
                "independent_c11_c13_cycles"],
            "floorplet_byte_hop": floor_validation[
                "independent_byte_hop"],
            "floorplet_F_FP": floor["native_proxy_objective"],
            "tap_c11_c13_cycles": tap_validation[
                "independent_c11_c13_cycles"],
            "tap_byte_hop": tap_validation["independent_byte_hop"],
            "tap_selection_cost": tap["native_proxy_objective"],
            "tap_seed": tap["selected_seed"],
            "floorplet_over_ilp_c11_c13": (
                floor_validation["independent_c11_c13_cycles"] /
                source["independent_c11_c13_cycles"]),
            "tap_over_ilp_c11_c13": (
                tap_validation["independent_c11_c13_cycles"] /
                source["independent_c11_c13_cycles"]),
            "floorplet_over_ilp_byte_hop": (
                floor_validation["independent_byte_hop"] /
                source["independent_byte_hop"]),
            "tap_over_ilp_byte_hop": (
                tap_validation["independent_byte_hop"] /
                source["independent_byte_hop"]),
            "area_mm2": floor_validation["independent_area_mm2"],
            "peak_power_sum_W": floor_validation[
                "independent_peak_power_sum_W"],
            "floorplet_pass": floor_validation["pass"],
            "tap_pass": tap_validation["pass"],
            "source_ilp_sha256": source["sha256"],
            "floorplet_solution_sha256": floor["solution_sha256"],
            "tap_solution_sha256": tap["solution_sha256"],
        })
    rows.sort(key=lambda row: row["experts"])
    fields = [
        "experts", "mesh", "gpu_l", "gpu_s",
        "ilp_c11_c13_cycles", "ilp_byte_hop",
        "floorplet_c11_c13_cycles", "floorplet_byte_hop",
        "floorplet_F_FP", "tap_c11_c13_cycles", "tap_byte_hop",
        "tap_selection_cost", "tap_seed",
        "floorplet_over_ilp_c11_c13", "tap_over_ilp_c11_c13",
        "floorplet_over_ilp_byte_hop", "tap_over_ilp_byte_hop",
        "area_mm2", "peak_power_sum_W", "floorplet_pass", "tap_pass",
        "source_ilp_sha256", "floorplet_solution_sha256",
        "tap_solution_sha256",
    ]
    comparison_path = os.path.join(args.out_dir, "proxy_comparison.csv")
    write_csv(comparison_path, fields, rows)

    completed_pairs = [[audit["experts"], audit["mesh"]] for audit in audits]
    metadata = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "status": ("complete" if len(audits) == len(CANONICAL_PAIRS)
                   else "partial_scale_sweep"),
        "pass": all(audit["pass"] for audit in audits),
        "requested_pairs_this_invocation": [list(pair)
                                             for pair in requested_pairs],
        "canonical_pairs": [list(pair) for pair in CANONICAL_PAIRS],
        "completed_pairs": completed_pairs,
        "completed_count": len(audits),
        "protocol": {
            "source": "Stage-2 certified-latency ILP layouts",
            "source_dir": relpath(args.source_dir),
            "fixed_search_space": (
                "exact source BOM and immutable complete task bundles; "
                "physical placement only"),
            "floorplet": (
                "exact MILP encoding of traffic-weighted Manhattan F_FP"),
            "floorplet_time_limit_s": args.floorplet_time_limit,
            "floorplet_seed": 1,
            "floorplet_threads": 1,
            "tap": (
                "wall-clock SA on 0.7 normalized thermal coupling plus 0.3 "
                "normalized unweighted wirelength"),
            "tap_wall_budget_s_per_seed": args.tap_wall_budget,
            "tap_seeds": list(args.tap_seeds),
            "tap_total_budget_s_per_pair": (
                args.tap_wall_budget * len(args.tap_seeds)),
            "tap_selection": "minimum selection_cost, then minimum seed",
            "independent_audit": (
                "strict evaluate plus independent BOM/bundle multiset, "
                "C11--C13, and traffic-weighted byte-hop recomputation"),
            "claim_boundary": (
                "proxy generation is solver-only; this driver never invokes "
                "LEGOSim, and analytical scores are not LEGOSim cycles"),
        },
        "comparison_csv": relpath(comparison_path),
        "comparison_csv_sha256": sha256(comparison_path),
        "audit_paths": [relpath(os.path.join(
            paths["audits"], pair_name(audit["experts"], audit["mesh"]) +
            "_proxy_audit.json")) for audit in audits],
    }
    metadata_path = os.path.join(args.out_dir, "metadata.json")
    write_json(metadata_path, metadata)

    manifest_path = os.path.join(args.out_dir, "SHA256SUMS")
    manifest_rows = []
    for root, directories, filenames in os.walk(args.out_dir):
        directories.sort()
        for filename in sorted(filenames):
            path = os.path.join(root, filename)
            if os.path.abspath(path) == os.path.abspath(manifest_path):
                continue
            manifest_rows.append("%s  %s" % (sha256(path), relpath(path)))
    write_text(manifest_path, "\n".join(manifest_rows) + "\n")
    print("[aggregate] %d/%d pairs PASS; %s" %
          (len(audits), len(CANONICAL_PAIRS), comparison_path), flush=True)
    return audits


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs", nargs="*", default=None,
        help=("expert:mesh pairs separated by spaces or commas; default is "
              "8:6 16:8 32:12"))
    parser.add_argument("--source-dir", default=os.path.join(
        ROOT_DIR, "out", "scaling_ilp_secondary_20260801"))
    parser.add_argument("--params-dir", default=os.path.join(
        ROOT_DIR, "out", "scaling_qap_experiment", "params"))
    parser.add_argument("--out-dir", default=os.path.join(
        ROOT_DIR, "out", "scaling_secondary_proxy_experiment_20260801"))
    parser.add_argument("--floorplet-time-limit", type=float, default=30.0)
    parser.add_argument("--tap-wall-budget", type=float, default=10.0)
    parser.add_argument("--tap-seeds", default="11,29,47")
    parser.add_argument("--tap-alpha", type=float, default=0.7)
    parser.add_argument("--reuse", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.source_dir = os.path.abspath(args.source_dir)
    args.params_dir = os.path.abspath(args.params_dir)
    args.out_dir = os.path.abspath(args.out_dir)
    args.tap_seeds = tuple(int(item) for item in
                           args.tap_seeds.replace(" ", ",").split(",")
                           if item)
    if args.floorplet_time_limit != 30.0:
        raise ValueError("fair protocol requires Floorplet time limit = 30 s")
    if args.tap_wall_budget != 10.0:
        raise ValueError("fair protocol requires TAP budget = 10 s per seed")
    if args.tap_seeds != DEFAULT_SEEDS:
        raise ValueError("fair protocol requires TAP seeds 11,29,47")
    if abs(args.tap_alpha - 0.7) > 1.0e-12:
        raise ValueError("fair protocol requires TAP alpha = 0.7")
    requested_pairs = parse_pairs(args.pairs)

    paths = {}
    for child in ("solutions", "meta", "logs", "audits", "commands"):
        path = os.path.join(args.out_dir, child)
        if not os.path.isdir(path):
            os.makedirs(path)
        paths[child] = path

    completed = []
    try:
        for experts, mesh in requested_pairs:
            completed.append(run_pair(args, experts, mesh, paths))
        aggregate(args, paths, requested_pairs)
    except Exception:
        sys.stderr.write(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
