#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect the fresh five-scale joint-BOM replay into a strict 15-row CSV.

This collector is intentionally scaling-only.  It binds every audited
LEGOSim row to one schema-v5 revalidation manifest, verifies the exact current
solution hashes, and requires explicit evidence that GPU counts were variables
of the sole joint min-T MILP.  It never mixes the superseded E/4 scaling run.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import collect_legosim_metrics as common


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    "out/joint_bom_scaling_fresh_20260802/manifest.json")
DEFAULT_OUT_PREFIX = (
    "out/joint_bom_scaling_fresh_20260802/paper/scaling_metrics")
METHODS = common.OPTIMIZATION_METHODS
EXPECTED_CASES = common.EXPECTED_SCALING_CASES


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect manifest-bound fresh joint-BOM scaling metrics")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--audit", action="append", required=True,
        help="PASS audit_scaling_legosim_results.py JSON; repeat as needed")
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object: %s" % path)
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _same_protocol(actual, expected, label):
    for field, wanted in expected.items():
        got = actual.get(field)
        if isinstance(wanted, float):
            try:
                matched = math.isclose(float(got), wanted,
                                       rel_tol=1.0e-12, abs_tol=1.0e-12)
            except (TypeError, ValueError):
                matched = False
        else:
            matched = type(got) is type(wanted) and got == wanted
        if not matched:
            raise ValueError("%s.%s=%r, expected %r" %
                             (label, field, got, wanted))


def _site_counts(solution):
    counts = {"cpu_count": 0, "gpu_l_count": 0, "gpu_s_count": 0}
    for site in solution.get("sites") or []:
        kind = str(site.get("type") or "").upper()
        if kind == "CPU":
            counts["cpu_count"] += 1
        elif kind == "GPU-L":
            counts["gpu_l_count"] += 1
        elif kind == "GPU-S":
            counts["gpu_s_count"] += 1
    return counts


def _bundle_multiset(solution, label):
    """Return a coordinate-independent typed task-bundle multiset.

    Floorplet/TAP are placement-only controls: they may move a complete
    chiplet bundle, but they must not change its type or resident task set.
    Comparing only aggregate L/S counts would not be strong enough to prove
    that contract.
    """
    sites = solution.get("sites")
    task_site = solution.get("task_site")
    if not isinstance(sites, list) or not isinstance(task_site, dict):
        raise ValueError("%s lacks typed sites/task_site mappings" % label)
    bundles = []
    seen_tasks = set()
    seen_anchors = set()
    task_anchors = {}
    for index, site in enumerate(sites):
        if not isinstance(site, dict):
            raise ValueError("%s site %d is not an object" % (label, index))
        kind = str(site.get("type") or "").upper()
        if kind not in ("CPU", "GPU-L", "GPU-S"):
            raise ValueError("%s site %d has unknown chiplet type" %
                             (label, index))
        raw_tasks = site.get("tasks")
        if (not isinstance(raw_tasks, list) or
                any(not isinstance(task, str) for task in raw_tasks)):
            raise ValueError("%s site %d has a non-list/string task set" %
                             (label, index))
        try:
            anchor = (int(site["l"]), int(site["i"]), int(site["j"]))
        except (KeyError, TypeError, ValueError):
            raise ValueError("%s site %d has an invalid anchor" %
                             (label, index))
        if anchor in seen_anchors:
            raise ValueError("%s places multiple bundles at anchor %r" %
                             (label, anchor))
        seen_anchors.add(anchor)
        tasks = tuple(sorted(raw_tasks))
        if not tasks:
            raise ValueError("%s site %d is an empty chiplet bundle" %
                             (label, index))
        for task in tasks:
            if task in seen_tasks:
                raise ValueError("%s assigns task %s to multiple bundles" %
                                 (label, task))
            seen_tasks.add(task)
            task_anchors[task] = anchor
        bundles.append((kind, tasks))
    if not bundles:
        raise ValueError("%s has no chiplet bundles" % label)
    if (any(not isinstance(task, str) for task in task_site) or
            set(task_site) != seen_tasks):
        raise ValueError("%s task_site differs from resident task bundles" %
                         label)
    for task, raw_anchor in task_site.items():
        if (not isinstance(raw_anchor, (list, tuple)) or
                len(raw_anchor) != 3):
            raise ValueError("%s task_site[%s] has an invalid anchor" %
                             (label, task))
        try:
            anchor = tuple(int(value) for value in raw_anchor)
        except (TypeError, ValueError):
            raise ValueError("%s task_site[%s] has an invalid anchor" %
                             (label, task))
        if anchor != task_anchors[task]:
            raise ValueError(
                "%s task_site[%s] disagrees with its resident bundle" %
                (label, task))
    return tuple(sorted(bundles))


def _verify_solution_record(case_row, out_root, case_id, method):
    raw_path, raw_digest = common._manifest_solution_entry(case_row, method)
    path = common._manifest_path(raw_path, out_root / "manifest.json")
    expected = common._fresh_solution_path(
        out_root, "scaling", case_id, method)
    if path != expected or not common._is_within(path, out_root):
        raise ValueError("non-canonical %s/%s solution path" %
                         (case_id, method))
    digest = str(raw_digest or "").lower()
    if (not path.is_file() or
            not common.scaling_audit.SHA256_RE.match(digest) or
            _sha256(path) != digest):
        raise ValueError("solution hash mismatch for %s/%s" %
                         (case_id, method))
    if method == "ilp_perf":
        checked = common._verify_public_min_T_solution(
            path, digest, "%s/%s" % (case_id, method))
        return path, digest, checked["solution"], checked["bom"]
    solution = _load(path)
    return path, digest, solution, None


def _verify_manifest(path):
    manifest = _load(path)
    if manifest.get("schema_version") != 5 or manifest.get("status") != "complete":
        raise ValueError("scaling manifest is not complete schema-v5")
    _same_protocol(manifest.get("protocol") or {}, common.FRESH_PROTOCOL,
                   "manifest.protocol")
    out_root = common._manifest_path(manifest.get("out_root"), path)
    if out_root != path.parent:
        raise ValueError("manifest out_root differs from its parent")
    if (int(manifest.get("selected_case_count", -1)) != 5 or
            int(manifest.get("expected_replay_rows", -1)) != 15):
        raise ValueError("manifest is not the five-case/15-row scaling run")

    for name, record in (manifest.get("drivers") or {}).items():
        driver = Path(str(record.get("path") or "")).resolve()
        if not driver.is_file() or _sha256(driver) != record.get("sha256"):
            raise ValueError("recorded driver changed: %s" % name)

    case_rows = {str(row.get("case_id")): row
                 for row in manifest.get("cases") or []}
    if set(case_rows) != set(EXPECTED_CASES):
        raise ValueError("manifest scaling case matrix is incomplete")

    invocation_rows = {}
    for invocation in manifest.get("replay_invocations") or []:
        cases = invocation.get("cases") or []
        if len(cases) != 1 or not str(cases[0]).startswith("scaling:"):
            raise ValueError("invalid scaling replay invocation")
        case_id = str(cases[0]).split(":", 1)[1]
        if case_id in invocation_rows:
            raise ValueError("duplicate replay invocation for %s" % case_id)
        if invocation.get("status") != "complete":
            raise ValueError("replay invocation is incomplete for %s" % case_id)
        results = common._manifest_path(invocation.get("results_json"), path)
        expected_results = (out_root / "replays" / "scaling" / case_id /
                            "results.json").resolve()
        if results != expected_results or not results.is_file():
            raise ValueError("non-canonical replay results for %s" % case_id)
        digest = str(invocation.get("results_sha256") or "").lower()
        if (not common.scaling_audit.SHA256_RE.match(digest) or
                _sha256(results) != digest):
            raise ValueError("replay results hash mismatch for %s" % case_id)
        invocation_rows[case_id] = {"path": results, "sha256": digest}
    if set(invocation_rows) != set(EXPECTED_CASES):
        raise ValueError("manifest replay invocation matrix is incomplete")

    contracts = {}
    solutions = {}
    bundle_contracts = {}
    for case_id, (experts, _) in EXPECTED_CASES.items():
        row = case_rows[case_id]
        if row.get("suite") != "scaling":
            raise ValueError("%s is not marked scaling" % case_id)
        expected_area = common.SCALING_AREA_MM2_PER_EXPERT * experts
        expected_power = common.SCALING_POWER_W_PER_EXPERT * experts
        budgets = row.get("budgets") or {}
        if (float(budgets.get("a_max", -1)) != expected_area or
                float(budgets.get("p_max", -1)) != expected_power or
                row.get("scaling_budget_contract_version") !=
                common.SCALING_BUDGET_CONTRACT_VERSION or
                row.get("budget_policy") != common.SCALING_BUDGET_POLICY or
                row.get("budget_target_bom") is not None):
            raise ValueError("%s violates the exogenous budget policy" % case_id)
        for required in ("ilp_perf", "prior"):
            if (row.get(required) or {}).get("status") != "complete":
                raise ValueError("%s/%s is incomplete" % (case_id, required))
        for method in METHODS:
            sol_path, digest, solution, contract = _verify_solution_record(
                row, out_root, case_id, method)
            solutions[(case_id, method)] = {
                "path": sol_path, "sha256": digest, "solution": solution}
            if method == "ilp_perf":
                if (contract["area_budget_mm2"] != expected_area or
                        contract["power_budget_W"] != expected_power):
                    raise ValueError("%s solution caps differ from manifest" %
                                     case_id)
                contracts[case_id] = contract
                bundle_contracts[case_id] = _bundle_multiset(
                    solution, "%s/%s" % (case_id, method))
            else:
                if _site_counts(solution) != {
                        key: contracts[case_id][key]
                        for key in ("cpu_count", "gpu_l_count", "gpu_s_count")}:
                    raise ValueError("%s/%s changed the ILP-selected BOM" %
                                     (case_id, method))
                if _bundle_multiset(
                        solution, "%s/%s" % (case_id, method)) != \
                        bundle_contracts[case_id]:
                    raise ValueError(
                        "%s/%s changed the ILP-selected typed task bundles" %
                        (case_id, method))
    return {
        "manifest": manifest, "manifest_path": path, "out_root": out_root,
        "contracts": contracts, "solutions": solutions,
        "invocations": invocation_rows,
    }


def _audit_rows(paths, run):
    rows = {}
    audit_hashes = {}
    for path in paths:
        report = _load(path)
        audit_hashes[str(path)] = _sha256(path)
        if (report.get("audit_status") != "PASS" or
                report.get("global_errors") not in ([], None)):
            raise ValueError("scaling audit is not PASS: %s" % path)
        for source in report.get("rows") or []:
            case_id = str(source.get("case_id") or "")
            method = common.normalize_method(source.get("method"))
            key = (case_id, method)
            if case_id not in EXPECTED_CASES or method not in METHODS:
                raise ValueError("unexpected audit row %r" % (key,))
            if key in rows:
                raise ValueError("duplicate audit row %r" % (key,))
            if (source.get("audit_status") != "PASS" or
                    source.get("legosim_validation") != "PASS" or
                    source.get("errors") not in ([], None)):
                raise ValueError("failed audit row %r" % (key,))
            expected_results = run["invocations"][case_id]["path"]
            if Path(str(source.get("results_json") or "")).resolve() != \
                    expected_results:
                raise ValueError("audit source mismatch for %r" % (key,))
            evidence = Path(str(source.get("evidence_dir") or "")).resolve()
            if not common._is_within(evidence, run["out_root"]):
                raise ValueError("audit evidence is outside run for %r" %
                                 (key,))
            contract = run["contracts"][case_id]
            for field in ("cpu_count", "gpu_l_count", "gpu_s_count"):
                if int(source.get(field, -1)) != int(contract[field]):
                    raise ValueError("audit BOM mismatch for %r/%s" %
                                     (key, field))
            rows[key] = source
    expected = {(case_id, method) for case_id in EXPECTED_CASES
                for method in METHODS}
    if set(rows) != expected:
        raise ValueError("audit matrix is not the exact five-by-three set")
    return rows, audit_hashes


def _positive(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s is not numeric" % label)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("%s is not positive" % label)
    return number


def _build_rows(run, audited):
    output = []
    for case_id, (experts, mesh) in EXPECTED_CASES.items():
        contract = run["contracts"][case_id]
        for method in METHODS:
            source = audited[(case_id, method)]
            cycles = int(_positive(source.get("legosim_cycles"), "cycles"))
            power = _positive(source.get("orion_noc_power_W"), "NoC power")
            active = int(_positive(source.get("noc_active_cycles"),
                                   "NoC active cycles"))
            energy = _positive(source.get(
                "nominal_orion_dynamic_noc_energy_uJ"), "NoC energy")
            expected_energy = power * active / 1000.0
            if not math.isclose(energy, expected_energy,
                                rel_tol=1.0e-9, abs_tol=1.0e-9):
                raise ValueError("NoC energy mismatch for %s/%s" %
                                 (case_id, method))
            solution = run["solutions"][(case_id, method)]
            output.append({
                "schema_version": 1,
                "suite": "scaling",
                "experiment_id": run["out_root"].name,
                "case_id": case_id,
                "sweep_dimension": "experts",
                "budget_value": experts,
                "area_budget_mm2": contract["area_budget_mm2"],
                "power_budget_W": contract["power_budget_W"],
                "area_used_mm2": contract["area_used_mm2"],
                "experts": experts,
                "mesh": mesh,
                "method": method,
                "method_raw": source.get("method_raw") or method,
                "cpu_count": contract["cpu_count"],
                "gpu_l_count": contract["gpu_l_count"],
                "gpu_s_count": contract["gpu_s_count"],
                "shared_bom_peak_power_W":
                    contract["shared_bom_peak_power_W"],
                "bom_selection_provenance":
                    contract["bom_selection_provenance"],
                "gpu_counts_fixed": contract["gpu_counts_fixed"],
                "fixed_bom": contract["fixed_bom"],
                "solver_mip_gap": contract["solver_mip_gap"],
                "synchronized_cycles": cycles,
                "orion_noc_power_W": power,
                "noc_active_cycles": active,
                "nominal_orion_dynamic_noc_energy_uJ": energy,
                "legosim_validation": "PASS",
                "audit_status": "PASS",
                "evidence_tier": "strict_scaling_joint_bom_v4",
                "solution_path": str(solution["path"]),
                "solution_sha256": solution["sha256"],
                "evidence_dir": source.get("evidence_dir"),
                "source_results_json": source.get("results_json"),
                "errors": "",
            })
    return sorted(output, key=lambda row: (
        int(row["experts"]), METHODS.index(row["method"])))


def _write_csv(path, rows):
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(temporary), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=common.CSV_FIELDS,
                                extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(temporary), "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def main(argv=None):
    args = _parse_args(argv)
    manifest_path = _resolve(args.manifest)
    audit_paths = [_resolve(value) for value in args.audit]
    out_prefix = _resolve(args.out_prefix)
    run = _verify_manifest(manifest_path)
    audited, audit_hashes = _audit_rows(audit_paths, run)
    rows = _build_rows(run, audited)
    csv_path = Path(str(out_prefix) + ".csv")
    json_path = Path(str(out_prefix) + ".json")
    report = {
        "schema_version": 1,
        "audit_status": "PASS",
        "row_count": len(rows),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "audit_sha256": audit_hashes,
        "budget_policy": common.SCALING_BUDGET_POLICY,
        "gpu_counts_fixed": False,
        "active_objective": "min T",
        "rows": rows,
    }
    _write_csv(csv_path, rows)
    _write_json(json_path, report)
    print(json.dumps({
        "status": "PASS", "rows": len(rows),
        "csv": str(csv_path), "json": str(json_path)},
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        sys.exit(2)
