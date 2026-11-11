#!/usr/bin/env python3
"""Independent semantic audit for the fixed 13-case ILP-Power revalidation.

The auditor never imports the optimizer.  It reconstructs the active C1--C8
placement constraints, reference
bundle signatures, and codegen's deterministic balanced dispatch directly
from params/reference/solution coordinates, then writes a separate report.
Large secondary gaps are retained as caveats rather than confused with layout
infeasibility.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re


OBJECTIVE_KIND = (
    "codegen deterministic balanced-dispatch expert-to-expert "
    "Manhattan dynamic-energy proxy")
FLOAT_REL = 1.0e-9
FLOAT_ABS = 1.0e-12


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(a, b, rel=FLOAT_REL, absolute=FLOAT_ABS):
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(
        a, b, rel_tol=rel, abs_tol=absolute)


def largest_remainder(weights, total):
    if not weights:
        return []
    norm = float(sum(weights))
    if norm <= 0.0:
        weights = [1.0] * len(weights)
        norm = float(len(weights))
    exact = [total * float(weight) / norm for weight in weights]
    base = [int(math.floor(value)) for value in exact]
    order = sorted(range(len(weights)),
                   key=lambda i: (-(exact[i] - base[i]), i))
    for index in order[:total - sum(base)]:
        base[index] += 1
    return base


def balanced_transport(column_totals, nrows, row_total):
    if sum(column_totals) != nrows * row_total:
        raise ValueError("dispatch margins disagree")
    matrix = [[0] * len(column_totals) for _ in range(nrows)]
    cursor = 0
    for column, total in enumerate(column_totals):
        quotient, remainder = divmod(int(total), nrows)
        for source in range(nrows):
            matrix[source][column] = quotient
        for _ in range(remainder):
            matrix[cursor][column] += 1
            cursor = (cursor + 1) % nrows
    if ([sum(row) for row in matrix] != [row_total] * nrows or
            [sum(matrix[row][column] for row in range(nrows))
             for column in range(len(column_totals))] != column_totals):
        raise AssertionError("balanced dispatch construction failed")
    return matrix


def deterministic_dispatch(params, tokens):
    meta = params.get("meta", {})
    groups = int(meta.get("groups", len(params.get("expert_tasks", []))))
    topk = int(meta.get("topk", 1))
    weights = list(meta.get("group_pi", []))
    if len(weights) != groups:
        weights = [1.0] * groups
    row_total = int(tokens) * topk
    totals = largest_remainder(weights, groups * row_total)
    return balanced_transport(totals, groups, row_total)


def anchor(site):
    return tuple(int(site[key]) for key in ("l", "i", "j"))


def normalized_solution(solution, params, errors, label):
    library = params["library"]
    tasks = params["tasks"]
    sites = {}
    serialized_tasks = {}
    for number, site in enumerate(solution.get("sites", [])):
        try:
            at = anchor(site)
            kind = str(site["type"])
        except Exception:
            errors.append("%s site %d is malformed" % (label, number))
            continue
        if at in sites:
            errors.append("%s has duplicate anchor %s" % (label, at))
        if kind not in library:
            errors.append("%s uses unknown chiplet type %s" % (label, kind))
        sites[at] = kind
        serialized_tasks[at] = sorted(str(v) for v in site.get("tasks", []))

    task_site = {}
    for task, raw in solution.get("task_site", {}).items():
        try:
            at = tuple(int(value) for value in raw)
        except Exception:
            errors.append("%s task %s has malformed anchor" % (label, task))
            continue
        task_site[str(task)] = at
    if set(task_site) != set(tasks):
        errors.append("%s task set differs from params" % label)

    mapped = {at: [] for at in sites}
    for task, at in task_site.items():
        if at not in sites:
            errors.append("%s task %s maps to absent anchor %s" %
                          (label, task, at))
            continue
        mapped[at].append(task)
        kind = sites[at]
        if task in tasks and kind not in tasks[task]["allowed"]:
            errors.append("%s task %s is incompatible with %s" %
                          (label, task, kind))
    for at in sites:
        if sorted(mapped[at]) != serialized_tasks.get(at, []):
            errors.append("%s serialized site tasks disagree at %s" %
                          (label, at))
        if not mapped[at]:
            errors.append("%s has empty occupied bundle at %s" % (label, at))

    signature = sorted([[sites[at], sorted(mapped[at])] for at in sites],
                       key=lambda item: (item[0], item[1]))
    bom = {kind: sum(1 for value in sites.values() if value == kind)
           for kind in library}
    return sites, task_site, mapped, signature, bom


def audit_constraints(solution, params, grid, budgets, sites, task_site,
                      mapped, reference_bom, errors):
    library = params["library"]
    tasks = params["tasks"]
    L, X, Y = (int(grid[key]) for key in ("L", "X", "Y"))
    occupied = {}
    area = 0.0
    power = 0.0
    layer_area = {layer: 0.0 for layer in range(L)}

    for at, kind in sites.items():
        layer, row, col = at
        spec = library[kind]
        cells = [(layer, row + di, col + dj)
                 for di in range(int(spec["w"]))
                 for dj in range(int(spec["h"]))]
        for cell in cells:
            if not (0 <= cell[0] < L and 0 <= cell[1] < X and
                    0 <= cell[2] < Y):
                errors.append("C1 %s@%s footprint leaves grid" % (kind, at))
            if cell in occupied:
                errors.append("C1 footprint overlap at %s" % (cell,))
            occupied[cell] = (kind, at)
        area += float(spec["area"])
        power += float(spec["p_peak"])
        if 0 <= layer < L:
            layer_area[layer] += float(spec["area"])

        used_mem = sum(float(tasks[task]["mem_mb"])
                       for task in mapped.get(at, []) if task in tasks)
        if used_mem > float(spec["mem"]) + FLOAT_ABS:
            errors.append("C5 memory exceeded at %s" % (at,))
        expert_count = sum(1 for task in mapped.get(at, [])
                           if task in set(params.get("expert_tasks", [])))
        if expert_count > 1:
            errors.append("C6 multiple expert bodies at %s" % (at,))

    actual_bom = {kind: sum(1 for value in sites.values() if value == kind)
                  for kind in library}
    if actual_bom != reference_bom:
        errors.append("C2 BOM differs from reference")
    serialized_budgets = solution.get("budgets", {})
    for kind, limit in serialized_budgets.get("nmin", {}).items():
        if actual_bom.get(kind, 0) < int(limit):
            errors.append("C2 nmin violated for %s" % kind)
    for kind, limit in serialized_budgets.get("nmax", {}).items():
        if actual_bom.get(kind, 0) > int(limit):
            errors.append("C2 nmax violated for %s" % kind)

    if area > float(budgets["a_max"]) + FLOAT_ABS:
        errors.append("C7 area budget exceeded")
    a_layer = serialized_budgets.get("a_max_layer")
    if a_layer is not None:
        for layer, value in layer_area.items():
            if value > float(a_layer) + FLOAT_ABS:
                errors.append("C7 layer %d area exceeded" % layer)
    if power > float(budgets["p_max"]) + FLOAT_ABS:
        errors.append("C8 peak-power-model budget exceeded")
    if not close(solution.get("total_area_mm2"), area):
        errors.append("serialized total area disagrees with recomputation")
    if not close(solution.get("peak_power_W"), power):
        errors.append("serialized peak power disagrees with recomputation")
    return area, power, actual_bom


def command_value(command, flag):
    if flag not in command:
        return None
    index = command.index(flag)
    return command[index + 1] if index + 1 < len(command) else None


def audit_case(case, artifact_index):
    errors = []
    key = case["key"]
    params_path = Path(case["params_path"]).resolve()
    reference_path = Path(case["reference_path"]).resolve()
    solution_path = Path(case["solution_path"]).resolve()
    for kind, path, expected in (
            ("params", params_path, case["params_sha256"]),
            ("reference", reference_path, case["reference_sha256"]),
            ("solution", solution_path, case["solution_sha256"])):
        if not path.is_file():
            errors.append("%s file is missing" % kind)
            continue
        actual_hash = sha256(path)
        if actual_hash.lower() != str(expected).lower():
            errors.append("%s SHA differs from manifest" % kind)
        recorded = artifact_index.get((key, kind))
        if recorded is None or recorded[0] != path or \
                recorded[1].lower() != actual_hash.lower():
            errors.append("%s artifact-hash record mismatch" % kind)

    if errors and not all(path.is_file() for path in
                          (params_path, reference_path, solution_path)):
        return {"key": key, "status": "FAIL", "errors": errors}
    params = load_json(params_path)
    reference = load_json(reference_path)
    solution = load_json(solution_path)
    grid = case["grid"]
    budgets = case["budgets"]

    ref_errors = []
    _, _, _, ref_signature, ref_bom = normalized_solution(
        reference, params, ref_errors, "reference")
    sites, task_site, mapped, signature, bom = normalized_solution(
        solution, params, errors, "solution")
    errors.extend(ref_errors)
    if signature != ref_signature:
        errors.append("complete bundle signature differs from reference")
    if bom != ref_bom:
        errors.append("BOM differs from reference")

    cpu_sites = [at for at, kind in sites.items() if kind == "CPU"]
    if len(cpu_sites) != 1:
        errors.append("expected exactly one freely placed CPU: %s" % cpu_sites)
    if solution.get("cpu_anchor_policy") != "free":
        errors.append("serialized CPU placement policy is not free")
    comparison = solution.get("comparison_reference", {})
    if comparison.get("kind") != "placement-only-bundles" or \
            comparison.get("fixed_complete_bundles") is not True:
        errors.append("solution does not declare complete-bundle placement-only mode")
    source = comparison.get("source")
    if source is None or Path(source).resolve() != reference_path:
        errors.append("serialized comparison reference path mismatch")
    fixed_bom = comparison.get("fixed_bom")
    if fixed_bom != ref_bom:
        errors.append("serialized fixed BOM differs from reference")

    area, power, recomputed_bom = audit_constraints(
        solution, params, grid, budgets, sites, task_site, mapped, ref_bom,
        errors)
    if solution.get("grid") != grid:
        errors.append("solution grid differs from manifest")
    serialized_budgets = solution.get("budgets", {})
    for field in ("a_max", "p_max"):
        if not close(serialized_budgets.get(field), budgets[field]):
            errors.append("solution budget %s differs from manifest" % field)

    reference_cycles = solution.get("performance_reference_cycles")
    cap = solution.get("performance_cap_cycles")
    predicted = solution.get("predicted_makespan_cycles")
    expected_cap = float(reference_cycles) * 1.20
    if not close(solution.get("performance_relaxation"), 0.20):
        errors.append("performance relaxation is not 20 percent")
    if not close(cap, expected_cap):
        errors.append("performance cap is not 1.20*Tref")
    if float(predicted) > float(cap) + max(FLOAT_ABS, abs(float(cap))*FLOAT_REL):
        errors.append("serialized T exceeds the 20-percent cap")

    command_path = (Path(case["solution_path"]).parents[2] / "commands" /
                    (case["key"].replace(":", "_") + ".json"))
    command_record = load_json(command_path)
    command = command_record.get("command", [])
    expected_flags = {
        "--mode": "ilp-power", "--power-perf-relax": "0.20",
        "--noc-traffic": "deterministic-dispatch",
        "--dispatch-tokens-per-source": "32",
        "--dispatch-hidden": "128", "--cpu-anchor": "free",
        "--threads": "1",
    }
    for flag, expected in expected_flags.items():
        if command_value(command, flag) != expected:
            errors.append("command protocol mismatch for %s" % flag)
    placement_from = command_value(command, "--placement-from")
    if placement_from is None or Path(placement_from).resolve() != reference_path:
        errors.append("command placement reference mismatch")

    proxy = solution.get("noc_power_proxy", {})
    if solution.get("optimization_mode") != "ilp-power" or \
            solution.get("mode_label") != "ILP-Power":
        errors.append("solution mode metadata is not ILP-Power")
    if proxy.get("objective_kind") != OBJECTIVE_KIND:
        errors.append("NoC objective kind mismatch")
    if proxy.get("traffic_kind") != "deterministic-dispatch":
        errors.append("NoC traffic kind mismatch")
    if proxy.get("measured_orion_power") is not False:
        errors.append("proxy incorrectly claims measured ORION power")
    tokens = int(proxy.get("dispatch_tokens_per_source", -1))
    hidden = 128
    bytes_elem = int(params.get("meta", {}).get("bytes_elem", 2))
    token_bytes = 2 * hidden * bytes_elem
    if tokens != 32 or proxy.get("dispatch_token_bytes_two_phases") != token_bytes:
        errors.append("dispatch proxy payload configuration mismatch")

    matrix = deterministic_dispatch(params, tokens)
    groups = len(matrix)
    group_task = {}
    for task in params["tasks"]:
        match = re.match(r"^e\((\d+)\)$", task)
        if match:
            group_task[int(match.group(1))] = task
    missing = [group for group in range(groups) if group not in group_task]
    if missing:
        errors.append("missing expert groups for dispatch: %s" % missing)
    assignment_hops = 0
    byte_hop = 0
    energy = 0.0
    eps_h = float(params["interconnect"]["eps_h"])
    eps_v = float(params["interconnect"]["eps_v"])
    if not missing:
        for source in range(groups):
            for target in range(groups):
                count = matrix[source][target]
                if source == target or count <= 0:
                    continue
                a = task_site[group_task[source]]
                b = task_site[group_task[target]]
                dh = abs(a[1]-b[1]) + abs(a[2]-b[2])
                dv = abs(a[0]-b[0])
                assignment_hops += count * (dh + dv)
                byte_hop += count * token_bytes * (dh + dv)
                energy += 8 * count * token_bytes * (eps_h*dh + eps_v*dv)
    if proxy.get("dispatch_assignment_hops") != assignment_hops:
        errors.append("dispatch assignment-hop recomputation mismatch")
    if proxy.get("dispatch_byte_hop") != byte_hop:
        errors.append("dispatch byte-hop recomputation mismatch")
    if not close(proxy.get("energy_J"), energy):
        errors.append("dispatch energy recomputation mismatch")
    if not close(proxy.get("deadline_cycles"), cap):
        errors.append("proxy deadline differs from performance cap")

    task_dag_energy = 0.0
    for edge in params["edges"]:
        a = task_site[edge["src"]]
        b = task_site[edge["dst"]]
        dh = abs(a[1]-b[1]) + abs(a[2]-b[2])
        dv = abs(a[0]-b[0])
        task_dag_energy += 8 * float(edge["vol"]) * (eps_h*dh + eps_v*dv)
    if not close(proxy.get("task_dag_energy_J"), task_dag_energy):
        errors.append("task-DAG energy recomputation mismatch")

    secondary_gap = solution.get("secondary_mip_gap")
    finite_secondary = False
    try:
        finite_secondary = math.isfinite(float(secondary_gap))
    except (TypeError, ValueError):
        pass
    gap_caveat = (not finite_secondary or float(secondary_gap) > 0.02)
    secondary_gap_report = (float(secondary_gap) if finite_secondary
                            else str(secondary_gap))
    return {
        "key": key,
        "suite": case["suite"],
        "case_id": case["case_id"],
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "reference_bundle_signature_match": signature == ref_signature,
        "bom": recomputed_bom,
        "cpu_anchor": cpu_sites[0] if len(cpu_sites) == 1 else cpu_sites,
        "area_mm2": area,
        "area_limit_mm2": float(budgets["a_max"]),
        "peak_power_model_W": power,
        "power_limit_W": float(budgets["p_max"]),
        "performance_reference_cycles": reference_cycles,
        "performance_cap_cycles": cap,
        "predicted_T_cycles": predicted,
        "primary_mip_gap": solution.get("primary_mip_gap"),
        "secondary_mip_gap": secondary_gap_report,
        "secondary_status": solution.get("secondary_status"),
        "secondary_gap_caveat": gap_caveat,
        "dispatch_assignment_hops": assignment_hops,
        "dispatch_byte_hop": byte_hop,
        "dispatch_energy_J": energy,
        "solution_sha256": sha256(solution_path),
        "reference_sha256": sha256(reference_path),
        "params_sha256": sha256(params_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?",
                        default="out/ilp_power_reval_20260801")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = load_json(root / "manifest.json")
    artifacts = load_json(root / "artifact_hashes.json")
    artifact_index = {}
    for item in artifacts.get("artifacts", []):
        artifact_index[(item["case"], item["kind"])] = (
            Path(item["path"]).resolve(), str(item["sha256"]))
    rows = [audit_case(case, artifact_index)
            for case in manifest.get("cases", [])]
    failures = [row for row in rows if row["status"] != "PASS"]
    caveats = [row for row in rows if row.get("secondary_gap_caveat")]
    report = {
        "schema_version": 1,
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "audit_scope": (
            "independent hashes, reference bundles/BOM, active C1--C8, free "
            "CPU placement, 20-percent T cap, deterministic dispatch and proxy "
            "recomputation"),
        "root": str(root),
        "expected_case_count": 13,
        "case_count": len(rows),
        "pass_count": len(rows) - len(failures),
        "fail_count": len(failures),
        "status": ("PASS" if len(rows) == 13 and not failures else "FAIL"),
        "secondary_gap_caveat_count": len(caveats),
        "claim_boundary": (
            "PASS proves serialized placement feasibility and metric "
            "recomputation, not optimality of the secondary NoC proxy; "
            "secondary gaps are reported per case."),
        "rows": rows,
    }
    prefix = (Path(args.output_prefix).resolve() if args.output_prefix else
              root / "independent_audit")
    with Path(str(prefix) + ".json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    fields = [
        "key", "status", "reference_bundle_signature_match", "area_mm2",
        "area_limit_mm2", "peak_power_model_W", "power_limit_W",
        "max_column_power_density_W", "column_limit_W",
        "performance_reference_cycles", "performance_cap_cycles",
        "predicted_T_cycles", "primary_mip_gap", "secondary_mip_gap",
        "secondary_status", "secondary_gap_caveat",
        "dispatch_assignment_hops", "dispatch_byte_hop",
        "dispatch_energy_J",
    ]
    with Path(str(prefix) + ".csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("[audit-ilp-power] %s %d/%d PASS; secondary-gap caveats=%d" %
          (report["status"], report["pass_count"], len(rows), len(caveats)))
    for row in failures:
        print("[audit-ilp-power] FAIL %s: %s" %
              (row["key"], "; ".join(row["errors"])))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
