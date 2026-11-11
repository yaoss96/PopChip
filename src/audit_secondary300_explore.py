#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the E64/E128 300-second Stage-2 exploration."""

from __future__ import print_function

import argparse
import csv
import json
import os

import audit_secondary_layout_compatibility as compat
import run_scaling_ilp_secondary as runner
import verify_fixed_surrogate_solution as fixed_surrogate


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PAIRS = ((64, 12), (128, 16))


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def type_changes(reference, candidate):
    old = compat.task_types(reference)
    new = compat.task_types(candidate)
    return [
        {"task": task, "reference_type": old[task],
         "candidate_type": new[task]}
        for task in sorted(old) if old[task] != new[task]
    ]


def bundle_comparison(reference, candidate):
    old, old_hash = compat.bundle_signature(reference)
    new, new_hash = compat.bundle_signature(candidate)
    return {
        "identical": old == new,
        "reference_signature_sha256": old_hash,
        "candidate_signature_sha256": new_hash,
        "removed": compat.difference(old, new),
        "added": compat.difference(new, old),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explore-dir", default=os.path.join(
        ROOT_DIR, "out", "scaling_ilp_secondary300_explore_20260801"))
    parser.add_argument("--formal30-dir", default=os.path.join(
        ROOT_DIR, "out", "scaling_ilp_secondary_20260801"))
    parser.add_argument("--params-dir", default=os.path.join(
        ROOT_DIR, "out", "scaling_qap_experiment", "params"))
    parser.add_argument("--baseline-dir", default=os.path.join(
        ROOT_DIR, "out", "scaling_prior_proxy_experiment_20260731",
        "solutions"))
    parser.add_argument("--comparison-csv", default=os.path.join(
        ROOT_DIR, "out", "scaling_prior_proxy_experiment_20260731",
        "scaling_proxy_comparison.csv"))
    args = parser.parse_args(argv)

    explore_dir = os.path.abspath(args.explore_dir)
    formal_dir = os.path.abspath(args.formal30_dir)
    baseline_dir = os.path.abspath(args.baseline_dir)
    seeds = compat.selected_tap_seeds(os.path.abspath(args.comparison_csv))
    rows = []
    details = []
    for experts, mesh in PAIRS:
        stem = "e%03d_m%02d" % (experts, mesh)
        params = read_json(os.path.join(
            os.path.abspath(args.params_dir), "experts_%03d.json" % experts))
        explore_path = os.path.join(
            explore_dir, "solutions", stem + "_ilp_secondary.json")
        formal_path = os.path.join(
            formal_dir, "solutions", stem + "_ilp_secondary.json")
        old_path = os.path.join(baseline_dir, stem + "_ilp.json")
        floor_path = os.path.join(baseline_dir, stem + "_floorplet.json")
        tap_path = os.path.join(
            baseline_dir, stem + "_tap_seed%d.json" % seeds[experts])
        audit_path = os.path.join(explore_dir, "audits", stem + "_audit.json")
        solutions = dict((name, read_json(path)) for name, path in (
            ("explore300", explore_path), ("formal30", formal_path),
            ("old_stage1", old_path), ("floorplet", floor_path),
            ("tap", tap_path)))
        metrics = dict((name, runner.independent_placement_metrics(params, sol))
                       for name, sol in solutions.items())
        cycles = dict((name, fixed_surrogate.independent_c11_c13(params, sol))
                      for name, sol in solutions.items())
        pair_audit = read_json(audit_path)
        explore = solutions["explore300"]
        primary = float(explore["primary_incumbent_cycles"])
        cap = primary + max(1.0e-6, abs(primary) * 1.0e-6)
        formal_byte_hop = metrics["formal30"]["byte_hop"]
        explore_byte_hop = metrics["explore300"]["byte_hop"]
        reduction = formal_byte_hop - explore_byte_hop
        formal_bundle = bundle_comparison(solutions["formal30"], explore)
        old_bundle = bundle_comparison(solutions["old_stage1"], explore)
        formal_type_changes = type_changes(solutions["formal30"], explore)
        old_type_changes = type_changes(solutions["old_stage1"], explore)
        old_experts = compat.expert_types(params, solutions["old_stage1"])
        new_experts = compat.expert_types(params, explore)
        detail = {
            "experts": experts,
            "mesh": mesh,
            "pass": bool(pair_audit["pass"]),
            "solver": pair_audit["solver"],
            "latency_cap_cycles": cap,
            "independent_c11_c13_cycles": cycles,
            "independent_metrics": metrics,
            "byte_hop_reduction_vs_formal30": reduction,
            "byte_hop_reduction_vs_formal30_percent": (
                100.0 * reduction / formal_byte_hop),
            "found_lower_byte_hop_incumbent": reduction > 0.0,
            "primary_incumbent_matches_formal30": (
                abs(primary - float(solutions["formal30"][
                    "primary_incumbent_cycles"])) <= 1.0e-6),
            "formal30_bundle_comparison": formal_bundle,
            "old_stage1_bundle_comparison": old_bundle,
            "formal30_task_type_changes": formal_type_changes,
            "old_stage1_task_type_changes": old_type_changes,
            "old_stage1_expert_assignment": old_experts,
            "explore300_expert_assignment": new_experts,
            "GPU_L_S_expert_assignment_identical_to_old": (
                old_experts == new_experts),
            "artifacts": {
                "solution": os.path.relpath(explore_path, ROOT_DIR).replace(
                    "\\", "/"),
                "solution_sha256": compat.sha256_file(explore_path),
                "audit": os.path.relpath(audit_path, ROOT_DIR).replace(
                    "\\", "/"),
                "audit_sha256": compat.sha256_file(audit_path),
                "log": os.path.relpath(os.path.join(
                    explore_dir, "logs", stem + ".log"), ROOT_DIR).replace(
                        "\\", "/"),
            },
        }
        details.append(detail)
        rows.append({
            "experts": experts,
            "mesh": mesh,
            "pass": pair_audit["pass"],
            "primary_wall_s": pair_audit["solver"]["primary_wall_s"],
            "primary_gap": pair_audit["solver"]["primary_mip_gap"],
            "secondary_wall_s": pair_audit["solver"]["secondary_wall_s"],
            "secondary_gap": pair_audit["solver"]["secondary_mip_gap"],
            "formal30_byte_hop": formal_byte_hop,
            "explore300_byte_hop": explore_byte_hop,
            "reduction_byte_hop": reduction,
            "reduction_percent": 100.0 * reduction / formal_byte_hop,
            "found_lower_incumbent": reduction > 0.0,
            "bundle_identical_to_formal30": formal_bundle["identical"],
            "task_types_identical_to_formal30": not formal_type_changes,
            "GPU_L_S_experts_identical_to_old": old_experts == new_experts,
            "floorplet_byte_hop": metrics["floorplet"]["byte_hop"],
            "tap_byte_hop": metrics["tap"]["byte_hop"],
            "solution_sha256": detail["artifacts"]["solution_sha256"],
        })

    summary_path = os.path.join(explore_dir, "exploration_summary.csv")
    with open(summary_path + ".tmp", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(summary_path + ".tmp", summary_path)
    audit_path = os.path.join(explore_dir, "exploration_audit.json")
    write_json(audit_path, {
        "pass": all(row["pass"] for row in details),
        "claim_boundary": (
            "exploratory solver-only run; Stage 1 is 30 s and Stage 2 is "
            "300 s; no LEGOSim result is implied"),
        "rows": details,
    })

    manifest_path = os.path.join(explore_dir, "artifact_hashes.json")
    artifacts = []
    for directory, _, filenames in os.walk(explore_dir):
        for filename in filenames:
            path = os.path.join(directory, filename)
            if os.path.abspath(path) == os.path.abspath(manifest_path):
                continue
            artifacts.append({
                "path": os.path.relpath(path, ROOT_DIR).replace("\\", "/"),
                "bytes": os.path.getsize(path),
                "sha256": compat.sha256_file(path),
            })
    for path in (os.path.abspath(__file__),
                 os.path.join(SCRIPT_DIR, "run_scaling_ilp_secondary.py")):
        artifacts.append({
            "path": os.path.relpath(path, ROOT_DIR).replace("\\", "/"),
            "bytes": os.path.getsize(path),
            "sha256": compat.sha256_file(path),
        })
    artifacts.sort(key=lambda item: item["path"])
    write_json(manifest_path, {"artifacts": artifacts})
    print("[audit300] wrote %s" % audit_path)
    for row in rows:
        print("[audit300] E=%d lower=%s byte_hop %.0f -> %.0f (%.3f%%)" %
              (row["experts"], row["found_lower_incumbent"],
               row["formal30_byte_hop"], row["explore300_byte_hop"],
               row["reduction_percent"]))
    return 0 if all(row["pass"] for row in details) else 1


if __name__ == "__main__":
    raise SystemExit(main())
