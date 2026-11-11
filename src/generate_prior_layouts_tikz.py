#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the strict A1550 ILP/Floorplet/TAP layout comparison.

The drawing is deliberately downstream of the canonical schema-v5 manifest
and the audited unified LEGOSim CSV.  Coordinates are read only from the three
manifest-bound solution JSON files; synchronized cycles are read only from
the corresponding PASS rows in the CSV and cross-checked against the
manifest-bound replay.  Every input is validated before the output is
atomically replaced.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_CSV = "out/unified_legosim_data/legosim_metrics.csv"
DEFAULT_OUTPUT = "paper_date/prior-layouts-tikz.tex"
CASE_ID = "area_1550"
CASE_KEY = "budget:area_1550"
GRID = {"L": 1, "X": 6, "Y": 6}
AREA_BUDGET = 1550.0
POWER_BUDGET = 1625.0
EXPECTED_BOM = {"CPU": 1, "GPU-L": 2, "GPU-S": 6}
EXPECTED_PEAK_POWER_W = 1205.0
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_BUDGET_CASES = (
    "area_970", "area_1260", "area_1550", "area_2130",
    "power_785", "power_995", "power_1205", "power_1625",
)
EXPECTED_SCALING_CASES = (
    "e008_m06", "e016_m08", "e032_m12", "e064_m12", "e128_m16",
)
BUDGET_METHODS = ("ilp_perf", "matched_random", "floorplet", "tap")
SCALING_METHODS = ("ilp_perf", "floorplet", "tap")
EXPECTED_CSV_KEYS = (
    {("budget", case_id, method)
     for case_id in EXPECTED_BUDGET_CASES for method in BUDGET_METHODS} |
    {("scaling", case_id, method)
     for case_id in EXPECTED_SCALING_CASES for method in SCALING_METHODS}
)
EXPECTED_CASE_KEYS = (
    {("budget", case_id) for case_id in EXPECTED_BUDGET_CASES} |
    {("scaling", case_id) for case_id in EXPECTED_SCALING_CASES}
)

METHODS = (
    {"key": "ilp_perf", "manifest": "ilp-perf", "panel": "a",
     "title": "ILP-Perf"},
    {"key": "floorplet", "manifest": "floorplet", "panel": "b",
     "title": r"Floorplet proxy~\cite{floorplet}"},
    {"key": "tap", "manifest": "tap", "panel": "c",
     "title": r"TAP-2.5D proxy~\cite{tap25d}"},
)

TYPE_SPECS = {
    "CPU": {"width": 1, "height": 1, "area": 90.0, "peak_W": 65.0,
            "macro": "priorcpuchip", "short": "C"},
    "GPU-L": {"width": 2, "height": 2, "area": 400.0,
              "peak_W": 300.0, "macro": "priorlgpu", "short": "L"},
    "GPU-S": {"width": 1, "height": 1, "area": 110.0,
              "peak_W": 90.0, "macro": "priorsgpu", "short": "S"},
}
TASK_ORDER = (
    "e(0)", "e(1)", "e(2)", "e(3)", "e(4)", "e(5)", "e(6)", "e(7)",
    "t_attn", "t_gate", "t_dispatch", "t_combine", "t_norm",
)
TASK_SHORT = {
    "e(0)": "e0", "e(1)": "e1", "e(2)": "e2", "e(3)": "e3",
    "e(4)": "e4", "e(5)": "e5", "e(6)": "e6", "e(7)": "e7",
    "t_attn": "At", "t_gate": "Gt", "t_dispatch": "Ds",
    "t_combine": "Cb", "t_norm": "Nm",
}

REQUIRED_CSV_COLUMNS = {
    "schema_version", "suite", "experiment_id", "case_id",
    "sweep_dimension", "budget_value", "area_budget_mm2",
    "power_budget_W", "experts", "mesh", "method", "method_raw",
    "cpu_count", "gpu_l_count", "gpu_s_count",
    "shared_bom_peak_power_W", "synchronized_cycles",
    "orion_noc_power_W", "noc_active_cycles",
    "nominal_orion_dynamic_noc_energy_uJ", "legosim_validation",
    "audit_status", "evidence_tier", "solution_path", "solution_sha256",
    "evidence_dir", "source_results_json", "errors",
}

MANIFEST_PROTOCOL = {
    "schema_version": 5,
    "fresh_ilp_perf": True,
    "production_methods": {
        "budget": ["ilp-perf", "matched-random", "floorplet", "tap"],
        "scaling": ["ilp-perf", "floorplet", "tap"],
    },
    "historical_layout_reference": False,
    "local_power_density_constraint": False,
    "cpu_anchor": "free",
    "cpu_count": 1,
    "package_area_and_peak_power_budgets": True,
    "expert_slots_main_legosim": 1,
    "explicit_task_schedule": True,
    "conditional_same_site_ordering": True,
    "mesh_flow_capacity": True,
    "noc_traffic": "deterministic-dispatch",
    "ilp_perf_objective": "sole-joint-min-T",
    "primary_objective": "minimize joint MILP makespan T",
    "active_objective_count": 1,
    "active_objective": "min T",
    "joint_selection_placement_scheduling": True,
    "protocol_completion_role": "sole-primary-objective",
    "protocol_completion_in_T": True,
    "protocol_completion_used_as_objective": True,
    "legacy_analytical_task_makespan_role": "hard-feasibility-only",
    "public_performance_fields": "replay-aligned-protocol-T",
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate strict A1550 ILP/Floorplet/TAP TikZ")
    parser.add_argument(
        "--canonical-root", required=True,
        help="complete schema-v5 canonical experiment root")
    parser.add_argument(
        "--metrics-csv", "--input", dest="metrics_csv",
        default=DEFAULT_METRICS_CSV,
        help="audited unified LEGOSim metrics CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only", action="store_true",
        help="validate and render in memory without replacing the output")
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _recorded_path(value, anchor):
    if value in (None, ""):
        raise ValueError("missing path recorded relative to %s" % anchor)
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(anchor).resolve().parent / path
    return path.resolve()


def _is_within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _read_bytes(path, label):
    path = Path(path)
    if not path.is_file():
        raise ValueError("%s not found: %s" % (label, path))
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError("cannot read %s %s: %s" % (label, path, exc))


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _reject_json_constant(value):
    raise ValueError("non-finite JSON constant %s" % value)


def _json_from_bytes(raw, label):
    try:
        value = json.loads(raw.decode("utf-8"),
                           parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("invalid JSON in %s: %s" % (label, exc))
    if not isinstance(value, dict):
        raise ValueError("%s is not a JSON object" % label)
    return value


def _load_json(path, label):
    raw = _read_bytes(path, label)
    return _json_from_bytes(raw, str(path)), raw


def _expect_exact(container, field, expected, label):
    actual = container.get(field) if isinstance(container, dict) else None
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError("%s.%s=%r, expected %r" %
                         (label, field, actual, expected))
    return actual


def _finite(value, label, positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s is not numeric: %r" % (label, value))
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError("%s is not finite%s: %r" %
                         (label, " and positive" if positive else "", value))
    return number


def _require_close(value, expected, label):
    number = _finite(value, label)
    tolerance = max(1.0e-9, 1.0e-9 * max(abs(number), abs(expected)))
    if abs(number - float(expected)) > tolerance:
        raise ValueError("%s=%.12g, expected %.12g" %
                         (label, number, expected))
    return number


def _strict_int(value, label, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s is not an integer: %r" % (label, value))
    if minimum is not None and value < minimum:
        raise ValueError("%s is below %d: %r" % (label, minimum, value))
    return value


def _csv_int(value, label, minimum=None):
    text = str(value or "").strip()
    try:
        number = int(text)
    except ValueError:
        raise ValueError("%s is not an integer: %r" % (label, value))
    if str(number) != text:
        raise ValueError("%s is not a canonical integer: %r" % (label, value))
    if minimum is not None and number < minimum:
        raise ValueError("%s is below %d: %r" % (label, minimum, value))
    return number


def _canonical_solution_path(root, method):
    if method == "ilp_perf":
        return (root / "solutions" / "budget" / CASE_ID /
                "ilp_perf.json").resolve()
    return (root / "baselines" / "budget" / CASE_ID / "prior" /
            (method + ".json")).resolve()


def _manifest_solution_record(case, method, manifest_path, root):
    if method == "ilp_perf":
        stage = case.get("ilp_perf") or {}
        raw_path = stage.get("solution_path")
        raw_digest = stage.get("solution_sha256")
    else:
        stage = case.get("prior") or {}
        raw_path = stage.get(method + "_solution_path")
        digests = stage.get("solution_sha256") or {}
        raw_digest = digests.get(method) if isinstance(digests, dict) else None
    if stage.get("status") != "complete":
        raise ValueError("manifest %s stage is not complete" % method)
    path = _recorded_path(raw_path, manifest_path)
    canonical = _canonical_solution_path(root, method)
    if path != canonical or not _is_within(path, root):
        raise ValueError("manifest %s solution path is not canonical: %s" %
                         (method, path))
    digest = str(raw_digest or "").strip().lower()
    if not SHA256_RE.match(digest):
        raise ValueError("manifest %s solution SHA-256 is invalid" % method)
    raw = _read_bytes(path, "%s solution" % method)
    if _digest(raw) != digest:
        raise ValueError("manifest %s solution SHA-256 mismatch" % method)
    return {"path": path, "sha256": digest, "raw": raw}


def _load_manifest(root):
    markers = sorted(path.name for path in root.iterdir()
                     if path.name.upper().startswith("SUPERSEDED"))
    if markers:
        raise ValueError("canonical root is marked SUPERSEDED: %s" %
                         ", ".join(markers))
    manifest_path = (root / "manifest.json").resolve()
    manifest, raw = _load_json(manifest_path, "canonical manifest")
    _expect_exact(manifest, "schema_version", 5, "manifest")
    _expect_exact(manifest, "status", "complete", "manifest")
    _expect_exact(manifest, "selected_case_count", 13, "manifest")
    _expect_exact(manifest, "expected_ilp_solutions", 13, "manifest")
    _expect_exact(manifest, "expected_replay_rows", 47, "manifest")
    if _recorded_path(manifest.get("out_root"), manifest_path) != root:
        raise ValueError("manifest.out_root does not match canonical root")

    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("manifest lacks protocol object")
    for field, expected in MANIFEST_PROTOCOL.items():
        _expect_exact(protocol, field, expected, "manifest.protocol")

    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest.cases is not a list")
    case_keys = [(str(case.get("suite")), str(case.get("case_id")))
                 for case in cases if isinstance(case, dict)]
    if (len(cases) != 13 or len(case_keys) != 13 or
            len(set(case_keys)) != 13 or set(case_keys) != EXPECTED_CASE_KEYS):
        raise ValueError("manifest case matrix is not the canonical 13 cases")
    matches = [case for case in cases if isinstance(case, dict) and
               case.get("suite") == "budget" and
               case.get("case_id") == CASE_ID]
    if len(matches) != 1:
        raise ValueError("manifest must contain exactly one budget/%s case" %
                         CASE_ID)
    case = matches[0]
    _expect_exact(case, "key", CASE_KEY, "manifest A1550 case")
    _expect_exact(case, "grid", GRID, "manifest A1550 case")
    budgets = case.get("budgets")
    if not isinstance(budgets, dict):
        raise ValueError("manifest A1550 case lacks budgets")
    _require_close(budgets.get("a_max"), AREA_BUDGET,
                   "manifest A1550 a_max")
    _require_close(budgets.get("p_max"), POWER_BUDGET,
                   "manifest A1550 p_max")

    records = {
        method["key"]: _manifest_solution_record(
            case, method["key"], manifest_path, root)
        for method in METHODS
    }

    invocations = manifest.get("replay_invocations")
    if not isinstance(invocations, list):
        raise ValueError("manifest lacks replay_invocations")
    budget_runs = [row for row in invocations
                   if isinstance(row, dict) and row.get("name") == "budget"]
    if len(budget_runs) != 1:
        raise ValueError("manifest must contain exactly one budget replay")
    replay_record = budget_runs[0]
    _expect_exact(replay_record, "status", "complete",
                  "manifest budget replay")
    if CASE_KEY not in (replay_record.get("cases") or []):
        raise ValueError("manifest budget replay omits %s" % CASE_KEY)
    replay_path = _recorded_path(replay_record.get("results_json"),
                                 manifest_path)
    canonical_replay = (root / "replays" / "budget" / "results.json").resolve()
    if replay_path != canonical_replay or not _is_within(replay_path, root):
        raise ValueError("manifest budget replay path is not canonical")
    replay_sha = str(replay_record.get("results_sha256") or "").lower()
    if not SHA256_RE.match(replay_sha):
        raise ValueError("manifest budget replay SHA-256 is invalid")
    replay, replay_raw = _load_json(replay_path, "manifest-bound budget replay")
    if _digest(replay_raw) != replay_sha:
        raise ValueError("manifest budget replay SHA-256 mismatch")
    _expect_exact(replay, "status", "complete", "budget replay")
    _expect_exact(replay, "codegen_only", False, "budget replay")
    _expect_exact(replay, "rounds", 2, "budget replay")

    expected_paths = {}
    for value in replay_record.get("expected_solutions") or []:
        if not isinstance(value, dict) or value.get("case_id") != CASE_ID:
            continue
        raw_method = str(value.get("method") or "").replace("-", "_")
        if raw_method in records:
            if raw_method in expected_paths:
                raise ValueError("duplicate A1550 expected solution %s" %
                                 raw_method)
            expected_paths[raw_method] = _recorded_path(
                value.get("solution_path"), manifest_path)
    if set(expected_paths) != set(records):
        raise ValueError("budget replay expected_solutions omits an A1550 method")
    for method, path in expected_paths.items():
        if path != records[method]["path"]:
            raise ValueError("budget replay expected %s path is not manifest-bound" %
                             method)

    return {
        "path": manifest_path,
        "sha256": _digest(raw),
        "value": manifest,
        "case": case,
        "solutions": records,
        "replay_path": replay_path,
        "replay_sha256": replay_sha,
        "replay": replay,
    }


def _verify_no_post_T_objective(solution):
    for stage in ("secondary", "tertiary", "quaternary"):
        for suffix in ("objective", "objective_value", "status", "mip_gap"):
            field = stage + "_" + suffix
            if solution.get(field) is not None:
                raise ValueError("ILP-Perf contains post-T field %s" % field)
        if solution.get(stage + "_has_incumbent") is not False:
            raise ValueError("ILP-Perf does not mark %s stage as not run" % stage)
        if _finite(solution.get(stage + "_wall_s"),
                   "ILP-Perf %s_wall_s" % stage) != 0.0:
            raise ValueError("ILP-Perf contains a post-T %s stage" % stage)
        certificate = solution.get(stage + "_gap_certificate")
        if (not isinstance(certificate, dict) or
                certificate.get("certified") is not False or
                certificate.get("reason") != "stage-not-run"):
            raise ValueError("ILP-Perf %s stage certificate is not stage-not-run" %
                             stage)
    for field in ("secondary_time_limit_s", "tertiary_time_limit_s"):
        if solution.get(field) is not None:
            raise ValueError("ILP-Perf contains post-T limit %s" % field)


def _same_public_T(value, incumbent, label):
    number = _finite(value, label, positive=True)
    tolerance = max(1.0e-6, 1.0e-9 * max(abs(number), abs(incumbent)))
    if abs(number - incumbent) > tolerance:
        raise ValueError("%s %.12g differs from public T %.12g" %
                         (label, number, incumbent))


def _expected_directed_links():
    links = set()
    for row in range(6):
        for column in range(6):
            here = (0, row, column)
            if row + 1 < 6:
                there = (0, row + 1, column)
                links.add((here, there))
                links.add((there, here))
            if column + 1 < 6:
                there = (0, row, column + 1)
                links.add((here, there))
                links.add((there, here))
    return links


def _validate_ilp_contract(solution):
    _expect_exact(solution, "optimization_mode", "ilp-perf", "ILP-Perf")
    _expect_exact(solution, "cpu_anchor_policy", "free", "ILP-Perf")
    _expect_exact(solution, "primary_objective",
                  "minimize joint MILP makespan T", "ILP-Perf")
    _expect_exact(solution, "primary_has_incumbent", True, "ILP-Perf")
    _expect_exact(solution, "performance_cap_enforced", False, "ILP-Perf")
    if solution.get("performance_cap_cycles") is not None:
        raise ValueError("ILP-Perf contains a performance cap")
    if solution.get("compute_selection_cap_cycles") is not None:
        raise ValueError("ILP-Perf contains a compute-selection cap")

    slots = solution.get("expert_slots")
    if slots != {"CPU": 1, "GPU-L": 1, "GPU-S": 1}:
        raise ValueError("ILP-Perf expert_slots is not q=1: %r" % slots)
    objective = solution.get("objective_contract")
    if not isinstance(objective, dict):
        raise ValueError("ILP-Perf lacks objective_contract")
    expected_objective = {
        "active_objective_count": 1,
        "active_objective": "min T",
        "joint_selection_placement_scheduling": True,
        "protocol_completion_in_T": True,
        "protocol_used_as_separate_objective": False,
        "legacy_task_schedule_role": "hard-feasibility-only",
        "energy_or_byte_hop_tie_break": False,
        "T_definition": (
            "fixed-order replay max-plus completion plus Mesh capacity epigraph"),
    }
    for field, expected in expected_objective.items():
        _expect_exact(objective, field, expected,
                      "ILP-Perf objective_contract")

    protocol = solution.get("protocol_model")
    if not isinstance(protocol, dict):
        raise ValueError("ILP-Perf lacks protocol_model")
    for field, expected in {
            "included_in_T": True,
            "diagnostic_only": False,
            "used_as_objective": True,
            "used_as_separate_objective": False,
            "co_resident_compute_nonoverlap": True,
    }.items():
        _expect_exact(protocol, field, expected,
                      "ILP-Perf protocol_model")
    incumbent = _finite(solution.get("primary_incumbent_cycles"),
                        "ILP-Perf primary_incumbent_cycles", positive=True)
    for field in ("predicted_makespan_cycles", "performance_proxy_cycles",
                  "tightened_performance_proxy_cycles",
                  "protocol_makespan_cycles"):
        _same_public_T(solution.get(field), incumbent, "ILP-Perf " + field)
    _same_public_T(protocol.get("completion_cycles"), incumbent,
                   "ILP-Perf protocol_model.completion_cycles")
    _verify_no_post_T_objective(solution)

    power_proxy = solution.get("noc_power_proxy")
    if not isinstance(power_proxy, dict):
        raise ValueError("ILP-Perf lacks diagnostic noc_power_proxy")
    _expect_exact(power_proxy, "diagnostic_only", True,
                  "ILP-Perf noc_power_proxy")
    _expect_exact(power_proxy, "used_as_objective", False,
                  "ILP-Perf noc_power_proxy")

    routing = solution.get("routing")
    if not isinstance(routing, dict):
        raise ValueError("ILP-Perf lacks routed Mesh capacity data")
    for field, expected in {
            "kind": "phase-separated-source-aggregated-splittable-packet-flow",
            "capacity_horizon": "T",
            "directed_full_duplex": True,
            "directed_link_count": 120,
    }.items():
        _expect_exact(routing, field, expected, "ILP-Perf routing")
    _finite(routing.get("flow_scale_bytes"),
            "ILP-Perf routing.flow_scale_bytes", positive=True)
    actual_links = set()
    rows = routing.get("link_loads")
    if not isinstance(rows, list) or len(rows) != 120:
        raise ValueError("ILP-Perf routing.link_loads is not the 120-link Mesh")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("ILP-Perf routing link %d is not an object" % index)
        try:
            src = tuple(row["src"])
            dst = tuple(row["dst"])
        except (KeyError, TypeError):
            raise ValueError("ILP-Perf routing link %d lacks endpoints" % index)
        if (len(src) != 3 or len(dst) != 3 or
                any(isinstance(v, bool) or not isinstance(v, int)
                    for v in src + dst)):
            raise ValueError("ILP-Perf routing link %d has invalid endpoints" %
                             index)
        if (src, dst) in actual_links:
            raise ValueError("ILP-Perf routing repeats directed link %r" %
                             ((src, dst),))
        actual_links.add((src, dst))
    if actual_links != _expected_directed_links():
        raise ValueError("ILP-Perf routing does not enumerate the directed 6x6 Mesh")


def _bundle_signature(bundle_rows):
    canonical = [
        {"type": chip_type, "tasks": list(tasks)}
        for chip_type, tasks in sorted(bundle_rows)
    ]
    payload = json.dumps(canonical, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "sha256:" + _digest(payload)


def _validate_layout(record, method):
    solution = _json_from_bytes(record["raw"], str(record["path"]))
    if method == "ilp_perf":
        _validate_ilp_contract(solution)
    elif method == "floorplet":
        _expect_exact(solution, "method", "floorplet-inspired-exact-MILP",
                      "Floorplet")
        _expect_exact(solution, "proxy_not_original_tool", True, "Floorplet")
        _expect_exact(solution, "search_space",
                      "exact BOM and immutable complete task bundles",
                      "Floorplet")
    elif method == "tap":
        _expect_exact(solution, "method", "tap-spread-SA", "TAP")
        _expect_exact(solution, "proxy_not_original_tool", True, "TAP")
        _expect_exact(solution, "search_space",
                      "exact BOM and immutable complete task bundles", "TAP")
    else:
        raise AssertionError("unknown method %s" % method)

    _expect_exact(solution, "grid", GRID, "%s solution" % method)
    budgets = solution.get("budgets")
    if not isinstance(budgets, dict):
        raise ValueError("%s solution lacks budgets" % method)
    if "p_col" in budgets:
        raise ValueError("%s unexpectedly serializes removed C9/p_col" %
                         method)
    _require_close(budgets.get("a_max"), AREA_BUDGET,
                   "%s a_max" % method)
    _require_close(budgets.get("p_max"), POWER_BUDGET,
                   "%s p_max" % method)
    if budgets.get("a_max_layer") is not None:
        raise ValueError("%s unexpectedly enables a per-layer area budget" %
                         method)

    sites = solution.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("%s solution lacks sites" % method)
    occupancy = {}
    task_anchors = {}
    counts = {key: 0 for key in TYPE_SPECS}
    normalized = []
    rank = {task: index for index, task in enumerate(TASK_ORDER)}
    for index, site in enumerate(sites):
        label = "%s site %d" % (method, index)
        if not isinstance(site, dict):
            raise ValueError("%s is not an object" % label)
        chip_type = site.get("type")
        if chip_type not in TYPE_SPECS:
            raise ValueError("%s has invalid type %r" % (label, chip_type))
        layer = _strict_int(site.get("l"), label + " l", 0)
        row = _strict_int(site.get("i"), label + " i", 0)
        column = _strict_int(site.get("j"), label + " j", 0)
        spec = TYPE_SPECS[chip_type]
        if layer != 0 or row + spec["height"] > 6 or \
                column + spec["width"] > 6:
            raise ValueError("%s footprint leaves the 6x6 Mesh" % label)
        tasks = site.get("tasks")
        if (not isinstance(tasks, list) or not tasks or
                len(set(tasks)) != len(tasks)):
            raise ValueError("%s has invalid tasks" % label)
        for task in tasks:
            if task not in TASK_SHORT:
                raise ValueError("%s contains unexpected task %r" %
                                 (label, task))
            if task in task_anchors:
                raise ValueError("task %s appears at multiple sites" % task)
            task_anchors[task] = (layer, row, column)
        experts = [task for task in tasks if task.startswith("e(")]
        if len(experts) > 1:
            raise ValueError("%s violates q=1 expert occupancy" % label)
        for occupied_row in range(row, row + spec["height"]):
            for occupied_column in range(column, column + spec["width"]):
                cell = (layer, occupied_row, occupied_column)
                if cell in occupancy:
                    raise ValueError("%s overlaps %s at %r" %
                                     (label, occupancy[cell], cell))
                occupancy[cell] = label
        counts[chip_type] += 1
        normalized.append({
            "type": chip_type, "l": layer, "i": row, "j": column,
            "x": column, "y": 6 - row - spec["height"],
            "tasks": tuple(sorted(tasks, key=lambda task: rank[task])),
        })

    if set(task_anchors) != set(TASK_ORDER):
        raise ValueError("%s task set is not the exact A1550 workload" % method)
    if counts != EXPECTED_BOM:
        raise ValueError("%s BOM %r, expected %r" %
                         (method, counts, EXPECTED_BOM))
    task_site = solution.get("task_site")
    if not isinstance(task_site, dict) or set(task_site) != set(TASK_ORDER):
        raise ValueError("%s task_site does not match tasks" % method)
    for task, anchor in task_anchors.items():
        recorded = task_site.get(task)
        if (not isinstance(recorded, list) or tuple(recorded) != anchor or
                any(isinstance(v, bool) or not isinstance(v, int)
                    for v in recorded)):
            raise ValueError("%s task_site[%s] disagrees with sites" %
                             (method, task))

    area = sum(TYPE_SPECS[kind]["area"] * count
               for kind, count in counts.items())
    peak = sum(TYPE_SPECS[kind]["peak_W"] * count
               for kind, count in counts.items())
    _require_close(area, AREA_BUDGET, "%s derived area" % method)
    _require_close(peak, EXPECTED_PEAK_POWER_W,
                   "%s derived peak power" % method)
    if method == "ilp_perf":
        _require_close(solution.get("total_area_mm2"), area,
                       "ILP-Perf total_area_mm2")
        _require_close(solution.get("peak_power_W"), peak,
                       "ILP-Perf peak_power_W")
    else:
        strict = solution.get("strict_evaluate")
        if not isinstance(strict, dict):
            raise ValueError("%s lacks strict_evaluate" % method)
        _expect_exact(strict, "feasible", True, "%s strict_evaluate" % method)
        _expect_exact(strict, "reasons", [], "%s strict_evaluate" % method)
        _require_close(strict.get("total_area_mm2"), area,
                       "%s strict area" % method)
        _require_close(strict.get("peak_power_W"), peak,
                       "%s strict peak power" % method)

    bundles = tuple(sorted(
        (site["type"], tuple(sorted(site["tasks"]))) for site in normalized))
    signature = _bundle_signature(bundles)
    if method == "ilp_perf" and solution.get("bundle_signature") != signature:
        raise ValueError("ILP-Perf bundle_signature disagrees with its sites")
    normalized.sort(key=lambda item: (item["i"], item["j"], item["type"]))
    return {
        "method": method, "path": record["path"],
        "sha256": record["sha256"], "solution": solution,
        "sites": normalized, "bom": counts, "bundles": bundles,
        "bundle_signature": signature, "area": area, "peak_power": peak,
    }


def _replay_rows(replay, replay_path, manifest_records):
    layouts = replay.get("layouts")
    if not isinstance(layouts, list):
        raise ValueError("budget replay lacks layouts")
    selected = {}
    for row in layouts:
        if not isinstance(row, dict) or row.get("case_id") != CASE_ID:
            continue
        method = str(row.get("method") or "").strip().lower().replace("-", "_")
        if method not in manifest_records:
            continue
        if method in selected:
            raise ValueError("budget replay duplicates A1550/%s" % method)
        if row.get("status") != "ok":
            raise ValueError("budget replay A1550/%s is not ok" % method)
        record = manifest_records[method]
        path = _recorded_path(row.get("solution_path"), replay_path)
        digest = str(row.get("solution_sha256") or "").lower()
        if path != record["path"] or digest != record["sha256"]:
            raise ValueError("budget replay A1550/%s is not manifest-bound" %
                             method)
        legosim = row.get("legosim")
        if not isinstance(legosim, dict):
            raise ValueError("budget replay A1550/%s lacks LEGOSim" % method)
        _expect_exact(legosim, "status", "ok", "A1550/%s LEGOSim" % method)
        _expect_exact(legosim, "validation", "PASS",
                      "A1550/%s LEGOSim" % method)
        cycles = _strict_int(legosim.get("synchronized_makespan_cycles"),
                             "A1550/%s replay cycles" % method, 1)
        selected[method] = {"cycles": cycles}
    if set(selected) != set(manifest_records):
        raise ValueError("budget replay lacks an A1550 comparison row")
    return selected


def _load_metrics(path, root, manifest_data, replay_rows):
    raw = _read_bytes(path, "unified metrics CSV")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid UTF-8 in metrics CSV %s: %s" % (path, exc))
    reader = csv.DictReader(io.StringIO(text, newline=""))
    missing = sorted(REQUIRED_CSV_COLUMNS - set(reader.fieldnames or []))
    if missing:
        raise ValueError("unified CSV missing columns: %s" % ", ".join(missing))
    rows = list(reader)
    if len(rows) != len(EXPECTED_CSV_KEYS):
        raise ValueError("unified CSV has %d rows, expected exactly %d" %
                         (len(rows), len(EXPECTED_CSV_KEYS)))
    by_key = {}
    for line_number, row in enumerate(rows, 2):
        key = (str(row.get("suite") or "").strip(),
               str(row.get("case_id") or "").strip(),
               str(row.get("method") or "").strip())
        if key not in EXPECTED_CSV_KEYS:
            raise ValueError("CSV line %d has unexpected key %r" %
                             (line_number, key))
        if key in by_key:
            raise ValueError("unified CSV duplicates %r" % (key,))
        by_key[key] = (line_number, row)
        if str(row.get("audit_status") or "").strip() != "PASS":
            raise ValueError("CSV line %d audit_status is not PASS" %
                             line_number)
        if str(row.get("legosim_validation") or "").strip() != "PASS":
            raise ValueError("CSV line %d legosim_validation is not PASS" %
                             line_number)
        if str(row.get("errors") or "").strip():
            raise ValueError("CSV line %d contains audit errors" % line_number)
        expected_tier = ("fresh_constraint_replay_v1"
                         if key[0] == "budget" else "strict_scaling_v3")
        if str(row.get("evidence_tier") or "").strip() != expected_tier:
            raise ValueError("CSV line %d evidence_tier is not %s" %
                             (line_number, expected_tier))
        _csv_int(row.get("synchronized_cycles"),
                 "CSV line %d synchronized_cycles" % line_number, 1)
    if set(by_key) != EXPECTED_CSV_KEYS:
        raise ValueError("unified CSV is not the complete 47-row matrix")

    selected = {}
    for line_number, row in enumerate(rows, 2):
        if (str(row.get("suite") or "").strip() != "budget" or
                str(row.get("case_id") or "").strip() != CASE_ID):
            continue
        method = str(row.get("method") or "").strip()
        if method not in manifest_data["solutions"]:
            continue
        if method in selected:
            raise ValueError("unified CSV duplicates A1550/%s" % method)
        selected[method] = (line_number, row)
    if set(selected) != set(manifest_data["solutions"]):
        raise ValueError("unified CSV lacks an A1550 comparison row")

    metrics = {}
    for method, (line_number, row) in selected.items():
        label = "CSV line %d (A1550/%s)" % (line_number, method)
        if _csv_int(row.get("schema_version"), label + " schema_version", 1) != 1:
            raise ValueError("%s schema_version is not 1" % label)
        if str(row.get("experiment_id") or "").strip() != root.name:
            raise ValueError("%s experiment_id does not match canonical root" % label)
        for field, expected in {
                "sweep_dimension": "area", "mesh": "6x6",
                "legosim_validation": "PASS", "audit_status": "PASS",
                "evidence_tier": "fresh_constraint_replay_v1",
        }.items():
            if str(row.get(field) or "").strip() != expected:
                raise ValueError("%s %s is not %s" % (label, field, expected))
        if str(row.get("errors") or "").strip():
            raise ValueError("%s contains audit errors" % label)
        if _csv_int(row.get("experts"), label + " experts", 1) != 8:
            raise ValueError("%s experts is not 8" % label)
        _require_close(row.get("budget_value"), AREA_BUDGET,
                       label + " budget_value")
        _require_close(row.get("area_budget_mm2"), AREA_BUDGET,
                       label + " area_budget_mm2")
        _require_close(row.get("power_budget_W"), POWER_BUDGET,
                       label + " power_budget_W")
        csv_bom = {
            "CPU": _csv_int(row.get("cpu_count"), label + " cpu_count", 0),
            "GPU-L": _csv_int(row.get("gpu_l_count"),
                              label + " gpu_l_count", 0),
            "GPU-S": _csv_int(row.get("gpu_s_count"),
                              label + " gpu_s_count", 0),
        }
        if csv_bom != EXPECTED_BOM:
            raise ValueError("%s BOM %r, expected %r" %
                             (label, csv_bom, EXPECTED_BOM))
        _require_close(row.get("shared_bom_peak_power_W"),
                       EXPECTED_PEAK_POWER_W,
                       label + " shared_bom_peak_power_W")
        solution = manifest_data["solutions"][method]
        csv_solution = _recorded_path(row.get("solution_path"), path)
        csv_sha = str(row.get("solution_sha256") or "").strip().lower()
        if csv_solution != solution["path"] or csv_sha != solution["sha256"]:
            raise ValueError("%s is not bound to its manifest solution" % label)
        source_results = _recorded_path(row.get("source_results_json"), path)
        if source_results != manifest_data["replay_path"]:
            raise ValueError("%s source_results_json is not manifest-bound" % label)
        cycles = _csv_int(row.get("synchronized_cycles"),
                          label + " synchronized_cycles", 1)
        if cycles != replay_rows[method]["cycles"]:
            raise ValueError("%s synchronized_cycles disagrees with replay" % label)
        metrics[method] = {"cycles": cycles, "line_number": line_number}
    return metrics, _digest(raw)


def _task_label(site):
    tasks = [TASK_SHORT[task] for task in site["tasks"]]
    chip = TYPE_SPECS[site["type"]]["short"]
    if site["type"] == "CPU":
        lines = [chip]
    else:
        lines = [chip + ":" + tasks.pop(0)]
    while tasks:
        lines.append("+".join(tasks[:2]))
        del tasks[:2]
    return r"\\".join(lines)


def _tex_integer(value):
    number = _finite(value, "TeX integer")
    if not number.is_integer():
        raise ValueError("TeX integer is not integral: %r" % value)
    return ("{:,}".format(int(number))).replace(",", "{,}")


def _safe_comment(value):
    return str(value).replace("\r", " ").replace("\n", " ")


def _tex_text(root, manifest_data, metrics_path, metrics_sha, layouts, metrics):
    lines = [
        "%% Auto-generated by generate_prior_layouts_tikz.py; do not edit.",
        "%% Canonical root: %s" % _safe_comment(root.as_posix()),
        "%% Manifest: %s" % _safe_comment(manifest_data["path"].as_posix()),
        "%% Manifest SHA-256: %s" % manifest_data["sha256"],
        "%% Budget replay: %s" %
        _safe_comment(manifest_data["replay_path"].as_posix()),
        "%% Budget replay SHA-256: %s" % manifest_data["replay_sha256"],
        "%% Unified metrics CSV: %s" % _safe_comment(metrics_path.as_posix()),
        "%% Unified metrics CSV SHA-256: %s" % metrics_sha,
        "%% Contract: sole public min T; q=1; free CPU; directed Mesh capacity;",
        "%%           C9 disabled; no post-T objective; immutable A1550 bundles.",
    ]
    for method in METHODS:
        layout = layouts[method["key"]]
        lines.extend([
            "%% %s solution: %s" %
            (method["key"], _safe_comment(layout["path"].as_posix())),
            "%% %s solution SHA-256: %s" %
            (method["key"], layout["sha256"]),
            "%% %s unified CSV line: %d" %
            (method["key"], metrics[method["key"]]["line_number"]),
        ])
    lines.extend([
        "%% Shared bundle signature: %s" %
        layouts["ilp_perf"]["bundle_signature"],
        "%% Coordinates: source (l,i,j) maps to TikZ (x,y)=(j,6-i-height).",
        "%% At/Gt/Ds/Cb/Nm = attention/gate/dispatch/combine/norm.",
        r"\centering",
        r"\begingroup",
        r"\tikzset{",
        r"  priorgrid/.style={draw=black!28,line width=0.25pt},",
        r"  priorlarge/.style={draw=blue!65!black,fill=morandiA!62,line width=0.55pt},",
        r"  priorsmall/.style={draw=green!45!black,fill=morandiC!62,line width=0.55pt},",
        r"  priorcpu/.style={draw=orange!65!black,fill=morandiD!78,line width=0.55pt},",
        r"  priorlabel/.style={font=\sffamily\tiny,inner sep=0.3pt,align=center}",
        r"}",
        r"\newcommand{\priormesh}{%",
        r"  \draw[priorgrid,step=1] (0,0) grid (6,6);",
        r"  \draw[black!65,line width=0.45pt] (0,0) rectangle (6,6);",
        r"}",
        r"\newcommand{\priorsgpu}[3]{%",
        r"  \filldraw[priorsmall] (#1,#2) rectangle ++(1,1);",
        r"  \node[priorlabel] at (#1+0.5,#2+0.5) {#3};",
        r"}",
        r"\newcommand{\priorlgpu}[3]{%",
        r"  \filldraw[priorlarge] (#1,#2) rectangle ++(2,2);",
        r"  \node[priorlabel] at (#1+1,#2+1) {#3};",
        r"}",
        r"\newcommand{\priorcpuchip}[3]{%",
        r"  \filldraw[priorcpu] (#1,#2) rectangle ++(1,1);",
        r"  \node[priorlabel] at (#1+0.5,#2+0.5) {#3};",
        r"}",
    ])
    for index, method in enumerate(METHODS):
        key = method["key"]
        layout = layouts[key]
        if index:
            lines.append(r"\hfill")
        lines.extend([
            r"\begin{minipage}{0.31\textwidth}",
            r"\centering",
            r"\begin{tikzpicture}[x=0.48cm,y=0.48cm]",
            r"  \priormesh",
        ])
        for site in layout["sites"]:
            lines.append(
                "  %% source (l,i,j)=(%d,%d,%d), type=%s, tasks=[%s]; "
                "TikZ=(%d,%d)" %
                (site["l"], site["i"], site["j"], site["type"],
                 ",".join(site["tasks"]), site["x"], site["y"]))
            macro = TYPE_SPECS[site["type"]]["macro"]
            lines.append(r"  \%s{%d}{%d}{%s}" %
                         (macro, site["x"], site["y"], _task_label(site)))
        lines.extend([
            r"\end{tikzpicture}\\[-0.5mm]",
            (r"\footnotesize (%s) %s: %s cycles" %
             (method["panel"], method["title"],
              _tex_integer(metrics[key]["cycles"]))),
            r"\end{minipage}",
        ])
    lines.append(r"\endgroup")
    return "\n".join(lines) + "\n"


def _render(canonical_root, metrics_path):
    manifest = _load_manifest(canonical_root)
    layouts = {
        method["key"]: _validate_layout(
            manifest["solutions"][method["key"]], method["key"])
        for method in METHODS
    }
    reference_bundles = layouts["ilp_perf"]["bundles"]
    reference_bom = layouts["ilp_perf"]["bom"]
    for method in ("floorplet", "tap"):
        if layouts[method]["bom"] != reference_bom:
            raise ValueError("%s BOM differs from ILP-Perf" % method)
        if layouts[method]["bundles"] != reference_bundles:
            raise ValueError("%s task bundles differ from ILP-Perf" % method)
    replay = _replay_rows(manifest["replay"], manifest["replay_path"],
                          manifest["solutions"])
    metrics, metrics_sha = _load_metrics(
        metrics_path, canonical_root, manifest, replay)
    text = _tex_text(canonical_root, manifest, metrics_path, metrics_sha,
                     layouts, metrics)
    return text, {
        "manifest_sha256": manifest["sha256"],
        "replay_sha256": manifest["replay_sha256"],
        "metrics_csv_sha256": metrics_sha,
        "solution_sha256": {
            method: layout["sha256"] for method, layout in layouts.items()
        },
        "synchronized_cycles": {
            method: metric["cycles"] for method, metric in metrics.items()
        },
        "shared_bundle_signature": layouts["ilp_perf"]["bundle_signature"],
    }


def _atomic_write(destination, text):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".tmp.", dir=str(destination.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    args = _parse_args(argv)
    root = _resolve(args.canonical_root)
    if not root.is_dir():
        raise ValueError("canonical root not found: %s" % root)
    metrics_path = _resolve(args.metrics_csv)
    output_path = _resolve(args.output)
    text, metadata = _render(root, metrics_path)
    if not args.check_only:
        _atomic_write(output_path, text)
    report = {
        "status": "PASS", "check_only": bool(args.check_only),
        "canonical_root": str(root), "metrics_csv": str(metrics_path),
        "output": str(output_path),
        "output_sha256": _digest(text.encode("utf-8")),
    }
    report.update(metadata)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        sys.exit(2)
