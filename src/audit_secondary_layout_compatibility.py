#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare two-stage ILP bundles/types with the historical proxy source ILP."""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os

import evaluate
import verify_fixed_surrogate_solution as fixed_surrogate


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PAIRS = ((8, 6), (16, 8), (32, 12), (64, 12), (128, 16))


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def task_types(solution):
    types = evaluate.site_types_from_solution(solution)
    return dict((task, types[tuple(anchor)])
                for task, anchor in solution["task_site"].items())


def bundle_signature(solution):
    types = evaluate.site_types_from_solution(solution)
    bundles = dict((anchor, []) for anchor in types)
    for task, raw_anchor in solution["task_site"].items():
        bundles[tuple(raw_anchor)].append(task)
    signature = [
        {"type": types[anchor], "tasks": sorted(tasks)}
        for anchor, tasks in bundles.items()
    ]
    signature.sort(key=lambda item: (item["type"], item["tasks"]))
    raw = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode(
        "utf-8")
    return signature, sha256_bytes(raw)


def expert_types(params, solution):
    mapping = task_types(solution)
    output = {"GPU-L": [], "GPU-S": [], "other": []}
    for task in params["expert_tasks"]:
        kind = mapping[task]
        output[kind if kind in output else "other"].append(task)
    for kind in output:
        output[kind].sort()
    return output


def difference(left, right):
    return [item for item in left if item not in right]


def selected_tap_seeds(path):
    result = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[int(row["experts"])] = int(row["tap_seed"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secondary-dir", default=os.path.join(
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

    secondary_dir = os.path.abspath(args.secondary_dir)
    tap_seeds = selected_tap_seeds(os.path.abspath(args.comparison_csv))
    rows = []
    for experts, mesh in PAIRS:
        stem = "e%03d_m%02d" % (experts, mesh)
        params = read_json(os.path.join(
            os.path.abspath(args.params_dir), "experts_%03d.json" % experts))
        old_path = os.path.join(os.path.abspath(args.baseline_dir),
                                stem + "_ilp.json")
        floor_path = os.path.join(os.path.abspath(args.baseline_dir),
                                  stem + "_floorplet.json")
        tap_path = os.path.join(
            os.path.abspath(args.baseline_dir),
            stem + "_tap_seed%d.json" % tap_seeds[experts])
        new_path = os.path.join(secondary_dir, "solutions",
                                stem + "_ilp_secondary.json")
        old_solution = read_json(old_path)
        new_solution = read_json(new_path)
        floor_solution = read_json(floor_path)
        tap_solution = read_json(tap_path)

        old_types = task_types(old_solution)
        new_types = task_types(new_solution)
        type_changes = [
            {"task": task, "old_type": old_types[task],
             "new_type": new_types[task]}
            for task in sorted(old_types)
            if old_types[task] != new_types[task]
        ]
        old_bundles, old_bundle_hash = bundle_signature(old_solution)
        new_bundles, new_bundle_hash = bundle_signature(new_solution)
        bundles_identical = old_bundles == new_bundles
        old_experts = expert_types(params, old_solution)
        new_experts = expert_types(params, new_solution)
        expert_changes = {
            "GPU-L_removed": difference(old_experts["GPU-L"],
                                        new_experts["GPU-L"]),
            "GPU-L_added": difference(new_experts["GPU-L"],
                                      old_experts["GPU-L"]),
            "GPU-S_removed": difference(old_experts["GPU-S"],
                                        new_experts["GPU-S"]),
            "GPU-S_added": difference(new_experts["GPU-S"],
                                      old_experts["GPU-S"]),
        }
        primary = float(new_solution["primary_incumbent_cycles"])
        cap = primary + max(1.0e-6, abs(primary) * 1.0e-6)
        tolerance = max(1.0e-4, abs(cap) * 1.0e-9)
        cycles = {
            "new_stage2": fixed_surrogate.independent_c11_c13(
                params, new_solution),
            "old_stage1": fixed_surrogate.independent_c11_c13(
                params, old_solution),
            "floorplet": fixed_surrogate.independent_c11_c13(
                params, floor_solution),
            "tap": fixed_surrogate.independent_c11_c13(params, tap_solution),
        }
        under_cap = dict((method, value <= cap + tolerance)
                         for method, value in cycles.items())
        proxy_regeneration_required = bool(type_changes) or not bundles_identical
        row = {
            "experts": experts,
            "mesh": mesh,
            "pass": True,
            "new_stage1_incumbent_cycles": primary,
            "new_stage1_cap_cycles": cap,
            "task_to_chiplet_type_identical": not type_changes,
            "task_type_changes": type_changes,
            "bundle_signature_identical": bundles_identical,
            "old_bundle_signature_sha256": old_bundle_hash,
            "new_bundle_signature_sha256": new_bundle_hash,
            "bundles_removed_from_old": difference(old_bundles, new_bundles),
            "bundles_added_in_new": difference(new_bundles, old_bundles),
            "old_expert_assignment": old_experts,
            "new_expert_assignment": new_experts,
            "expert_assignment_changes": expert_changes,
            "GPU_L_S_expert_assignment_identical": old_experts == new_experts,
            "independent_c11_c13_cycles": cycles,
            "within_new_stage1_cap": under_cap,
            "proxy_regeneration_required": proxy_regeneration_required,
            "reason": (
                "new two-stage ILP changed task type and/or immutable bundles"
                if proxy_regeneration_required else
                "task types and immutable bundles match the proxy source ILP"),
        }
        rows.append(row)

    output = {
        "pass": all(row["pass"] for row in rows),
        "interpretation": (
            "Floorplet and TAP files were generated from the historical "
            "Stage1-only ILP. proxy_regeneration_required is true whenever "
            "the new two-stage solution changes task-to-type mapping or the "
            "coordinate-independent chiplet bundle multiset."),
        "rows": rows,
    }
    json_path = os.path.join(secondary_dir, "compatibility_audit.json")
    write_json(json_path, output)
    csv_path = os.path.join(secondary_dir, "compatibility_summary.csv")
    fields = [
        "experts", "mesh", "task_to_chiplet_type_identical",
        "bundle_signature_identical", "GPU_L_S_expert_assignment_identical",
        "floorplet_within_new_stage1_cap", "tap_within_new_stage1_cap",
        "old_stage1_within_new_stage1_cap", "proxy_regeneration_required",
        "old_bundle_signature_sha256", "new_bundle_signature_sha256",
    ]
    with open(csv_path + ".tmp", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "experts": row["experts"],
                "mesh": row["mesh"],
                "task_to_chiplet_type_identical": row[
                    "task_to_chiplet_type_identical"],
                "bundle_signature_identical": row[
                    "bundle_signature_identical"],
                "GPU_L_S_expert_assignment_identical": row[
                    "GPU_L_S_expert_assignment_identical"],
                "floorplet_within_new_stage1_cap": row[
                    "within_new_stage1_cap"]["floorplet"],
                "tap_within_new_stage1_cap": row[
                    "within_new_stage1_cap"]["tap"],
                "old_stage1_within_new_stage1_cap": row[
                    "within_new_stage1_cap"]["old_stage1"],
                "proxy_regeneration_required": row[
                    "proxy_regeneration_required"],
                "old_bundle_signature_sha256": row[
                    "old_bundle_signature_sha256"],
                "new_bundle_signature_sha256": row[
                    "new_bundle_signature_sha256"],
            })
    os.replace(csv_path + ".tmp", csv_path)
    artifact_rows = []
    manifest_path = os.path.join(secondary_dir, "artifact_hashes.json")
    for directory, _, filenames in os.walk(secondary_dir):
        for filename in sorted(filenames):
            path = os.path.join(directory, filename)
            if os.path.abspath(path) == os.path.abspath(manifest_path):
                continue
            artifact_rows.append({
                "path": os.path.relpath(path, ROOT_DIR).replace("\\", "/"),
                "bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            })
    for path in (os.path.abspath(__file__),
                 os.path.join(SCRIPT_DIR, "run_scaling_ilp_secondary.py")):
        artifact_rows.append({
            "path": os.path.relpath(path, ROOT_DIR).replace("\\", "/"),
            "bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
        })
    artifact_rows.sort(key=lambda item: item["path"])
    write_json(manifest_path, {"artifacts": artifact_rows})
    print("[compatibility] wrote %s" % json_path)
    print("[compatibility] wrote %s" % manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
