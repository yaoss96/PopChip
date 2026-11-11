#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export one provenance-preserving CSV for real LEGOSim replay metrics.

The output is deliberately long-form: one row per suite/case/method.  This
keeps measured methods and provenance explicit.  Scaling
rows must come from :mod:`audit_scaling_legosim_results` PASS reports.  The
older budget experiment predates the immutable-binary end manifest, so its
available mirrored evidence is rechecked here and labelled
``legacy_budget_mirror_v1`` rather than silently claiming the newer tier.

Only synchronized LEGOSim cycles and measured PopNet/ORION NoC metrics are
exported; analytical latency or power is never used as a fallback.  Nominal
dynamic NoC energy uses PopNet's active interval from ``delayInfo.txt`` at a
nominal 1-GHz clock.

The legacy CLI remains available for reproducing the earlier split budget
collection.  New constrained-model experiments must instead use
``--fresh-budget-results``.  In that mode ``--budget-core`` supplies only the
eight immutable case/budget records: every measured budget row must come from
one newly generated replay manifest whose SHA-256 is bound by the fresh run's
``manifest.json``.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import audit_scaling_legosim_results as scaling_audit
from run_scaling_qap_experiment import (
    SCALING_AREA_MM2_PER_EXPERT,
    SCALING_BUDGET_CONTRACT_VERSION,
    SCALING_BUDGET_POLICY,
    SCALING_POWER_W_PER_EXPERT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZATION_METHODS = ("ilp_perf", "floorplet", "tap")
BUDGET_METHODS = ("ilp_perf", "matched_random", "floorplet", "tap")
EXPECTED_BUDGET_CASES = (
    "area_970", "area_1260", "area_1550", "area_2130",
    "power_785", "power_995", "power_1205", "power_1625",
)
EXPECTED_SCALING_CASES = {
    "e008_m06": (8, "6x6"),
    "e016_m08": (16, "8x8"),
    "e032_m12": (32, "12x12"),
    "e064_m12": (64, "12x12"),
    "e128_m16": (128, "16x16"),
}
EXPECTED_BUDGET_ROWS = len(EXPECTED_BUDGET_CASES) * len(BUDGET_METHODS)
EXPECTED_SCALING_ROWS = len(EXPECTED_SCALING_CASES) * len(
    OPTIMIZATION_METHODS)
EXPECTED_FULL_ROWS = EXPECTED_BUDGET_ROWS + EXPECTED_SCALING_ROWS
FRESH_PROTOCOL = {
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
    "gpu_type_counts_fixed": False,
    "gpu_type_count_definition": "N_c=sum_p x[c,p]",
    "scaling_budget_contract_version": SCALING_BUDGET_CONTRACT_VERSION,
    "scaling_budget_policy": SCALING_BUDGET_POLICY,
    "scaling_area_mm2_per_expert": SCALING_AREA_MM2_PER_EXPERT,
    "scaling_power_W_per_expert": SCALING_POWER_W_PER_EXPERT,
    "scaling_budget_target_bom": None,
    "expert_slots_main_legosim": 1,
    "explicit_task_schedule": True,
    "conditional_same_site_ordering": True,
    "mesh_flow_capacity": True,
    "noc_traffic": "deterministic-dispatch",
    "expected_full_replay_rows": EXPECTED_FULL_ROWS,
    "ilp_perf_objective": "sole-joint-min-T",
    "primary_objective": "minimize joint MILP makespan T",
    "active_objective_count": 1,
    "active_objective": "min T",
    "joint_selection_placement_scheduling": True,
    "primary_incumbent_gap_bound_metric": (
        "replay-aligned end-to-end protocol completion T"),
    "protocol_completion_role": "sole-primary-objective",
    "protocol_completion_in_T": True,
    "protocol_completion_used_as_objective": True,
    "legacy_analytical_task_makespan_role": "hard-feasibility-only",
    "public_performance_fields": "replay-aligned-protocol-T",
}
METHOD_ALIASES = dict(scaling_audit.METHOD_ALIASES)
METHOD_ALIASES.update({
    "matched": "matched_random",
    "matched-random": "matched_random",
    "matched_random": "matched_random",
    "resource-matched-random-layout": "matched_random",
})
CASE_RE = re.compile(r"^(?P<kind>area|power)_(?P<value>[0-9]+(?:\.[0-9]+)?)$")
SCALE_RE = re.compile(r"(?:^|_)e(?P<experts>[0-9]+)(?:_|$)", re.I)

CSV_FIELDS = [
    "schema_version",
    "suite",
    "experiment_id",
    "case_id",
    "sweep_dimension",
    "budget_value",
    "area_budget_mm2",
    "power_budget_W",
    "area_used_mm2",
    "experts",
    "mesh",
    "method",
    "method_raw",
    "cpu_count",
    "gpu_l_count",
    "gpu_s_count",
    "shared_bom_peak_power_W",
    "bom_selection_provenance",
    "gpu_counts_fixed",
    "fixed_bom",
    "solver_mip_gap",
    "synchronized_cycles",
    "orion_noc_power_W",
    "noc_active_cycles",
    "nominal_orion_dynamic_noc_energy_uJ",
    "legosim_validation",
    "audit_status",
    "evidence_tier",
    "solution_path",
    "solution_sha256",
    "evidence_dir",
    "source_results_json",
    "errors",
]


def normalize_method(value):
    raw = str(value or "").strip().lower()
    return METHOD_ALIASES.get(raw, raw.replace("-", "_"))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Combine audited scaling and real budget LEGOSim metrics")
    parser.add_argument(
        "--scaling-audit", action="append", default=[],
        help=("PASS JSON emitted by audit_scaling_legosim_results.py; repeat "
              "when scaling cases are audited in separate replay groups"))
    parser.add_argument(
        "--budget-core",
        default=("out/corrected_e2e_runs/"
                 "corrected_e2e_20260731_v2/raw_results.json"),
        help="budget raw_results.json containing ILP and matched replay rows")
    parser.add_argument(
        "--budget-proxies",
        default="out/legosim_proxy_replay_corrected_20260731/results.json",
        help="budget results.json containing Floorplet and TAP replay rows")
    parser.add_argument(
        "--additional-results", action="append", default=[],
        help="additional legacy budget replay results.json; repeat as needed")
    parser.add_argument(
        "--fresh-budget-results",
        help=("single 32-row budget results.json emitted by "
              "run_full_constraint_revalidation.py; when present, the old "
              "budget core is metadata-only and legacy proxy/additional "
              "result files are never read"))
    parser.add_argument(
        "--fresh-run-manifest",
        help=("fresh run manifest.json; defaults to the run root inferred "
              "from <root>/replays/budget/results.json"))
    parser.add_argument(
        "--out-prefix", default="out/unified_legosim_data/legosim_metrics",
        help="output prefix; .json and .csv are appended")
    parser.add_argument(
        "--require-complete-47", action="store_true",
        help=("require the exact fresh paper matrix: eight budget cases by "
              "four methods plus five scaling cases by three methods"))
    return parser.parse_args(argv)


def _load_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object: %s" % path)
    return value


def _manifest_path(value, manifest_path):
    """Resolve a path recorded by the fresh-run manifest."""
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _is_within(path, root):
    """Python-3.7-compatible containment check for provenance paths."""
    try:
        return os.path.commonpath((str(Path(path).resolve()),
                                   str(Path(root).resolve()))) == str(
                                       Path(root).resolve())
    except ValueError:
        # Different Windows drives, for example, can have no common path.
        return False


def _required_integer(value, label):
    if isinstance(value, bool):
        raise ValueError("%s is not an integer: %r" % (label, value))
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s is not an integer: %r" % (label, value))
    if not math.isfinite(numeric) or numeric != float(parsed):
        raise ValueError("%s is not an integer: %r" % (label, value))
    return parsed


def _fresh_solution_path(out_root, suite, case_id, method):
    """Return the one canonical solution path for a paper replay row."""
    if method == "ilp_perf":
        return (out_root / "solutions" / suite / case_id /
                "ilp_perf.json").resolve()
    baseline = out_root / "baselines" / suite / case_id
    if method == "matched_random":
        return (baseline / "matched" / "representative.json").resolve()
    if method in ("floorplet", "tap"):
        return (baseline / "prior" / (method + ".json")).resolve()
    raise ValueError("unknown fresh solution method %r" % method)


def _manifest_solution_entry(case_row, method):
    """Extract one solution path/hash pair from a schema-v5 case row."""
    if method == "ilp_perf":
        record = case_row.get("ilp_perf") or {}
        return record.get("solution_path"), record.get("solution_sha256")
    if method == "matched_random":
        record = case_row.get("matched_random") or {}
        digests = record.get("solution_sha256")
        if isinstance(digests, dict):
            # Schema-v5 runner records method-labelled baseline digests as a
            # map, matching the Floorplet/TAP provenance shape.  Retain the
            # legacy scalar form for older focused fixtures, but never infer a
            # digest from another method's key.
            digest = digests.get("matched-random")
            if digest is None:
                digest = digests.get("matched_random")
        else:
            digest = digests
        return record.get("solution_path"), digest
    if method in ("floorplet", "tap"):
        record = case_row.get("prior") or {}
        digests = record.get("solution_sha256") or {}
        return (record.get(method + "_solution_path"),
                digests.get(method) if isinstance(digests, dict) else None)
    raise ValueError("unknown fresh solution method %r" % method)


def _finite_positive(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s is not numeric: %r" % (label, value))
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("%s is not finite and positive: %r" %
                         (label, value))
    return number


def _same_public_T(actual, expected, label):
    number = _finite_positive(actual, label)
    tolerance = max(1.0e-6, 1.0e-9 * max(abs(number), abs(expected)))
    if abs(number - expected) > tolerance:
        raise ValueError("%s %.12g differs from primary incumbent T %.12g" %
                         (label, number, expected))


def _verify_no_post_T_objectives(solution, label):
    """Fail closed unless all post-T optimization stages are explicitly off."""
    for stage in ("secondary", "tertiary", "quaternary"):
        for suffix in ("objective", "objective_value", "status", "mip_gap"):
            field = stage + "_" + suffix
            if solution.get(field) is not None:
                raise ValueError("%s contains a post-T objective field %s" %
                                 (label, field))
        if solution.get(stage + "_has_incumbent") is not False:
            raise ValueError("%s does not mark %s as stage-not-run" %
                             (label, stage))
        try:
            wall = float(solution.get(stage + "_wall_s"))
        except (TypeError, ValueError):
            raise ValueError("%s lacks %s stage-not-run wall time" %
                             (label, stage))
        if wall != 0.0:
            raise ValueError("%s contains a post-T objective (%s wall time)" %
                             (label, stage))
        certificate = solution.get(stage + "_gap_certificate")
        if (not isinstance(certificate, dict) or
                certificate.get("certified") is not False or
                certificate.get("reason") != "stage-not-run"):
            raise ValueError("%s does not mark %s as stage-not-run" %
                             (label, stage))
    for field in ("secondary_time_limit_s", "tertiary_time_limit_s"):
        if solution.get(field) is not None:
            raise ValueError("%s contains a post-T time limit %s" %
                             (label, field))


def _verify_public_min_T_solution(path, digest, label):
    """Validate the schema-v5 public protocol-T contract of one ILP result."""
    path = Path(path).resolve()
    digest = str(digest or "").lower()
    if not path.is_file():
        raise ValueError("fresh solution is missing for %s: %s" %
                         (label, path))
    if (not scaling_audit.SHA256_RE.match(digest) or
            scaling_audit._sha256(path) != digest):
        raise ValueError("fresh solution SHA-256 mismatch for %s: %s" %
                         (label, path))
    solution = _load_json(path)
    if solution.get("optimization_mode") != "ilp-perf":
        raise ValueError("%s is not an ilp-perf solution" % label)
    if solution.get("primary_objective") != \
            "minimize joint MILP makespan T":
        raise ValueError("%s does not minimize the sole joint T" % label)

    objective = solution.get("objective_contract")
    if not isinstance(objective, dict):
        raise ValueError("%s lacks objective_contract" % label)
    expected_objective = {
        "active_objective_count": 1,
        "active_objective": "min T",
        "joint_selection_placement_scheduling": True,
        "chiplet_count_definition": "N_c=sum_p x[c,p]",
        "gpu_type_counts_fixed": False,
        "protocol_completion_in_T": True,
        "protocol_used_as_separate_objective": False,
        "legacy_task_schedule_role": "hard-feasibility-only",
        "energy_or_byte_hop_tie_break": False,
    }
    for field, expected in expected_objective.items():
        actual = objective.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                "%s objective_contract.%s=%r, expected %r" %
                (label, field, actual, expected))

    protocol_model = solution.get("protocol_model")
    if not isinstance(protocol_model, dict):
        raise ValueError("%s lacks protocol_model" % label)
    expected_protocol = {
        "included_in_T": True,
        "diagnostic_only": False,
        "used_as_objective": True,
        "used_as_separate_objective": False,
    }
    for field, expected in expected_protocol.items():
        actual = protocol_model.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError("%s protocol_model.%s=%r, expected %r" %
                             (label, field, actual, expected))

    incumbent = _finite_positive(
        solution.get("primary_incumbent_cycles"),
        "%s primary_incumbent_cycles" % label)
    for field in ("predicted_makespan_cycles", "performance_proxy_cycles",
                  "tightened_performance_proxy_cycles",
                  "protocol_makespan_cycles"):
        _same_public_T(solution.get(field), incumbent,
                       "%s %s" % (label, field))
    _same_public_T(protocol_model.get("completion_cycles"), incumbent,
                   "%s protocol_model.completion_cycles" % label)
    _verify_no_post_T_objectives(solution, label)

    sites = solution.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("%s lacks non-empty sites" % label)
    bom = {"cpu_count": 0, "gpu_l_count": 0, "gpu_s_count": 0}
    for site in sites:
        kind = str((site or {}).get("type") or "").upper()
        if kind == "CPU":
            bom["cpu_count"] += 1
        elif kind == "GPU-L":
            bom["gpu_l_count"] += 1
        elif kind == "GPU-S":
            bom["gpu_s_count"] += 1
    if bom["cpu_count"] != 1:
        raise ValueError("%s does not contain exactly one CPU" % label)
    selected_counts = solution.get("selected_type_counts")
    selection = solution.get("resource_selection")
    if not isinstance(selected_counts, dict) or not isinstance(selection, dict):
        raise ValueError("%s lacks auditable resource selection" % label)
    normalized_counts = {
        str(kind): int(count) for kind, count in selected_counts.items()
        if int(count) != 0}
    expected_counts = {
        "CPU": bom["cpu_count"], "GPU-L": bom["gpu_l_count"],
        "GPU-S": bom["gpu_s_count"]}
    expected_counts = {kind: count for kind, count in expected_counts.items()
                       if count != 0}
    if normalized_counts != expected_counts:
        raise ValueError("%s selected_type_counts differs from sites" % label)
    if (selection.get("count_definition") != "N_c=sum_p x[c,p]" or
            selection.get("joint_with_mapping_and_placement") is not True or
            selection.get("fixed_bom") is not False or
            selection.get("selected_counts") != selected_counts):
        raise ValueError("%s resource_selection is not fresh joint selection" %
                         label)
    budgets = solution.get("budgets")
    if not isinstance(budgets, dict):
        raise ValueError("%s lacks package budgets" % label)
    gpu_bounds = []
    for bound_name in ("nmin", "nmax"):
        for kind in (budgets.get(bound_name) or {}):
            if str(kind).upper().startswith("GPU"):
                gpu_bounds.append("%s.%s" % (bound_name, kind))
    if gpu_bounds:
        raise ValueError("%s fixes/bounds GPU counts: %s" %
                         (label, ", ".join(sorted(gpu_bounds))))
    area_budget = _finite_positive(
        budgets.get("a_max"), "%s budgets.a_max" % label)
    power_budget = _finite_positive(
        budgets.get("p_max"), "%s budgets.p_max" % label)
    area_used = _finite_positive(
        solution.get("total_area_mm2"), "%s total_area_mm2" % label)
    power_used = _finite_positive(
        solution.get("peak_power_W"), "%s peak_power_W" % label)
    tolerance = 1.0e-6
    if area_used > area_budget + tolerance:
        raise ValueError("%s exceeds its area budget" % label)
    if power_used > power_budget + tolerance:
        raise ValueError("%s exceeds its peak-power budget" % label)
    for field, expected in (
            ("area_budget_mm2", area_budget),
            ("area_used_mm2", area_used),
            ("peak_power_budget_W", power_budget),
            ("peak_power_used_W", power_used)):
        try:
            actual = float(selection.get(field))
        except (TypeError, ValueError):
            raise ValueError("%s resource_selection.%s is invalid" %
                             (label, field))
        if not math.isclose(actual, expected, rel_tol=1.0e-12,
                            abs_tol=1.0e-6):
            raise ValueError("%s resource_selection.%s differs" %
                             (label, field))
    gap = solution.get("primary_mip_gap")
    if gap is not None:
        try:
            gap = float(gap)
        except (TypeError, ValueError):
            raise ValueError("%s primary_mip_gap is invalid" % label)
        if not math.isfinite(gap) or gap < 0.0:
            raise ValueError("%s primary_mip_gap is invalid" % label)
    bom.update({
        "area_budget_mm2": area_budget,
        "power_budget_W": power_budget,
        "area_used_mm2": area_used,
        "shared_bom_peak_power_W": power_used,
        "bom_selection_provenance": "fresh-joint-min-T-MILP",
        "gpu_counts_fixed": False,
        "fixed_bom": False,
        "solver_mip_gap": gap,
    })
    return {"path": path, "sha256": digest, "solution": solution,
            "bom": bom}


def _verify_fresh_run_manifest(manifest_path, budget_results_path):
    """Bind a fresh replay file to the exact new-constraint run manifest.

    This check is intentionally structural and fatal.  A legacy replay can
    have the same row schema, so accepting it and merely marking rows FAIL
    would still risk publishing a mixed table.
    """
    manifest_path = Path(manifest_path).resolve()
    budget_results_path = Path(budget_results_path).resolve()
    superseded = [path for path in manifest_path.parent.iterdir()
                  if path.name.upper().startswith("SUPERSEDED")]
    if superseded:
        raise ValueError(
            "fresh run root is marked SUPERSEDED and cannot be collected: %s" %
            manifest_path.parent)
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 5:
        raise ValueError(
            "fresh run manifest schema_version=%r, expected 5" %
            manifest.get("schema_version"))
    if manifest.get("status") != "complete":
        raise ValueError("fresh run manifest is not complete: %s" %
                         manifest_path)
    protocol = manifest.get("protocol") or {}
    for name, expected in FRESH_PROTOCOL.items():
        actual = protocol.get(name)
        if isinstance(expected, float):
            try:
                matched = math.isclose(float(actual), expected,
                                       rel_tol=1.0e-12, abs_tol=1.0e-12)
            except (TypeError, ValueError):
                matched = False
        else:
            # bool is intentionally type-sensitive: 1 is not accepted as a
            # substitute for an explicit protocol declaration.
            matched = (type(actual) is type(expected) and actual == expected)
        if not matched:
            raise ValueError(
                "fresh protocol %s=%r, expected %r" %
                (name, actual, expected))

    out_root = _manifest_path(manifest.get("out_root"), manifest_path)
    if out_root != manifest_path.parent:
        raise ValueError("fresh manifest out_root does not match its parent")
    expected_budget_path = (
        out_root / "replays" / "budget" / "results.json").resolve()
    if budget_results_path != expected_budget_path:
        raise ValueError(
            "fresh budget results must be %s, got %s" %
            (expected_budget_path, budget_results_path))
    if _required_integer(manifest.get("selected_case_count"),
                         "fresh selected_case_count") != 13:
        raise ValueError("fresh manifest must select exactly 13 cases")
    if _required_integer(manifest.get("expected_replay_rows"),
                         "fresh expected_replay_rows") != EXPECTED_FULL_ROWS:
        raise ValueError("fresh manifest must plan exactly %d replay rows" %
                         EXPECTED_FULL_ROWS)

    manifest_cases = manifest.get("cases")
    if not isinstance(manifest_cases, list):
        raise ValueError("fresh manifest has no cases list")
    actual_cases = [(str(row.get("suite")), str(row.get("case_id")))
                    for row in manifest_cases]
    expected_cases = (
        [("budget", case_id) for case_id in EXPECTED_BUDGET_CASES] +
        [("scaling", case_id)
         for case_id in sorted(EXPECTED_SCALING_CASES)])
    if sorted(actual_cases) != sorted(expected_cases):
        raise ValueError("fresh manifest case matrix is not the 13-case "
                         "paper protocol")
    solution_records = {}
    case_contracts = {}
    for row in manifest_cases:
        suite = str(row.get("suite"))
        case_id = str(row.get("case_id"))
        required = ["ilp_perf", "prior"]
        methods = list(OPTIMIZATION_METHODS)
        if suite == "budget":
            required.append("matched_random")
            methods = list(BUDGET_METHODS)
        for name in required:
            if (row.get(name) or {}).get("status") != "complete":
                raise ValueError("fresh manifest %s:%s/%s is not complete" %
                                 (suite, case_id, name))
        for method in methods:
            raw_path, raw_digest = _manifest_solution_entry(row, method)
            solution_path = _manifest_path(raw_path, manifest_path)
            canonical_path = _fresh_solution_path(
                out_root, suite, case_id, method)
            if solution_path != canonical_path:
                raise ValueError(
                    "fresh manifest %s:%s/%s solution path is not canonical" %
                    (suite, case_id, method))
            if not _is_within(solution_path, out_root):
                raise ValueError("fresh solution is outside run root: %s" %
                                 solution_path)
            digest = str(raw_digest or "").lower()
            label = "%s:%s/%s" % (suite, case_id, method)
            if method == "ilp_perf":
                checked = _verify_public_min_T_solution(
                    solution_path, digest, label)
                manifest_budgets = row.get("budgets") or {}
                for field, manifest_name in (("area_budget_mm2", "a_max"),
                                             ("power_budget_W", "p_max")):
                    try:
                        recorded = float(manifest_budgets[manifest_name])
                    except (KeyError, TypeError, ValueError):
                        raise ValueError(
                            "fresh manifest %s has invalid %s" %
                            (label, manifest_name))
                    if not math.isclose(
                            checked["bom"][field], recorded,
                            rel_tol=1.0e-12, abs_tol=1.0e-6):
                        raise ValueError(
                            "fresh manifest %s %s differs from solution" %
                            (label, manifest_name))
                if suite == "scaling":
                    experts, _ = EXPECTED_SCALING_CASES[case_id]
                    expected_area = SCALING_AREA_MM2_PER_EXPERT * experts
                    expected_power = SCALING_POWER_W_PER_EXPERT * experts
                    if (row.get("scaling_budget_contract_version") !=
                            SCALING_BUDGET_CONTRACT_VERSION or
                            row.get("budget_policy") !=
                            SCALING_BUDGET_POLICY or
                            row.get("budget_target_bom") is not None or
                            not math.isclose(recorded, expected_power,
                                             rel_tol=0.0, abs_tol=1.0e-9) or
                            not math.isclose(
                                float(manifest_budgets["a_max"]),
                                expected_area, rel_tol=0.0, abs_tol=1.0e-9)):
                        raise ValueError(
                            "fresh manifest %s violates the exogenous "
                            "scaling-budget contract" % label)
                case_contracts[(suite, case_id)] = checked["bom"]
            else:
                if not solution_path.is_file():
                    raise ValueError("fresh solution is missing for %s: %s" %
                                     (label, solution_path))
                if (not scaling_audit.SHA256_RE.match(digest) or
                        scaling_audit._sha256(solution_path) != digest):
                    raise ValueError(
                        "fresh solution SHA-256 mismatch for %s: %s" %
                        (label, solution_path))
                # Also reject malformed/non-object solution payloads before
                # accepting a replay hash which happens to bind those bytes.
                _load_json(solution_path)
            solution_records[(suite, case_id, method)] = {
                "path": solution_path,
                "sha256": digest,
            }

    invocations = manifest.get("replay_invocations")
    if not isinstance(invocations, list):
        raise ValueError("fresh manifest has no replay_invocations list")
    by_case = {}
    for invocation in invocations:
        cases = invocation.get("cases") or []
        if len(cases) != 1 and invocation.get("name") != "budget":
            raise ValueError("fresh scaling replay must contain one case")
        if invocation.get("name") == "budget":
            key = "budget"
            suite_name = "budget"
            expected_rows = EXPECTED_BUDGET_ROWS
            expected_case_keys = {
                "budget:%s" % case_id for case_id in EXPECTED_BUDGET_CASES}
            if set(str(value) for value in cases) != expected_case_keys:
                raise ValueError("fresh budget invocation case set mismatch")
        else:
            if len(cases) != 1 or not str(cases[0]).startswith("scaling:"):
                raise ValueError("invalid fresh scaling replay invocation")
            case_id = str(cases[0]).split(":", 1)[1]
            if case_id not in EXPECTED_SCALING_CASES:
                raise ValueError("unexpected fresh scaling case %s" % case_id)
            if invocation.get("name") != "scaling_" + case_id:
                raise ValueError("fresh scaling invocation name mismatch")
            key = "scaling:%s" % case_id
            suite_name = "scaling"
            expected_rows = len(OPTIMIZATION_METHODS)
        if key in by_case:
            raise ValueError("duplicate fresh replay invocation for %s" % key)
        if invocation.get("status") != "complete":
            raise ValueError("fresh replay invocation %s is not complete" %
                             key)
        if _required_integer(invocation.get("expected_rows"),
                             "%s expected_rows" % key) != expected_rows:
            raise ValueError("fresh replay invocation %s row count mismatch" %
                             key)
        results_path = _manifest_path(invocation.get("results_json"),
                                      manifest_path)
        if key == "budget":
            canonical_results_path = expected_budget_path
            expected_methods = BUDGET_METHODS
            expected_case_ids = EXPECTED_BUDGET_CASES
        else:
            case_id = key.split(":", 1)[1]
            canonical_results_path = (
                out_root / "replays" / "scaling" / case_id /
                "results.json").resolve()
            expected_methods = OPTIMIZATION_METHODS
            expected_case_ids = (case_id,)
        if results_path != canonical_results_path:
            raise ValueError("fresh replay result path is not canonical: %s" %
                             results_path)
        if not _is_within(results_path, out_root) or not results_path.is_file():
            raise ValueError("fresh replay result is missing/outside run: %s" %
                             results_path)
        raw_layout_keys = invocation.get("expected_layout_keys")
        if not isinstance(raw_layout_keys, list):
            raise ValueError("fresh replay %s has no expected layout keys" %
                             key)
        try:
            layout_keys = {
                (str(value[0]), normalize_method(value[1]))
                for value in raw_layout_keys if len(value) == 2}
        except (TypeError, IndexError):
            raise ValueError("fresh replay %s has invalid layout keys" % key)
        expected_layout_keys = {
            (case_id, method) for case_id in expected_case_ids
            for method in expected_methods}
        if (len(raw_layout_keys) != expected_rows or
                len(layout_keys) != expected_rows or
                layout_keys != expected_layout_keys):
            raise ValueError("fresh replay %s layout-key matrix mismatch" %
                             key)
        raw_expected_solutions = invocation.get("expected_solutions")
        if not isinstance(raw_expected_solutions, list):
            raise ValueError("fresh replay %s has no expected_solutions" % key)
        planned_solutions = {}
        for value in raw_expected_solutions:
            if not isinstance(value, dict):
                raise ValueError(
                    "fresh replay %s has invalid expected_solutions" % key)
            solution_case = str(value.get("case_id") or "")
            solution_method = normalize_method(value.get("method"))
            solution_key = (solution_case, solution_method)
            if solution_key in planned_solutions:
                raise ValueError(
                    "fresh replay %s has duplicate expected solution %r" %
                    (key, solution_key))
            canonical = solution_records.get(
                (suite_name, solution_case, solution_method))
            planned_path = _manifest_path(
                value.get("solution_path"), manifest_path)
            if canonical is None or planned_path != canonical["path"]:
                raise ValueError(
                    "fresh replay %s expected solution is not manifest-bound: "
                    "%r" % (key, solution_key))
            planned_solutions[solution_key] = canonical
        if (len(planned_solutions) != expected_rows or
                set(planned_solutions) != expected_layout_keys):
            raise ValueError(
                "fresh replay %s expected-solution matrix mismatch" % key)
        digest = str(invocation.get("results_sha256") or "").lower()
        if (not scaling_audit.SHA256_RE.match(digest) or
                scaling_audit._sha256(results_path) != digest):
            raise ValueError("fresh replay result SHA-256 mismatch: %s" %
                             results_path)
        replay = _load_json(results_path)
        if replay.get("status") != "complete":
            raise ValueError("fresh replay result is not complete: %s" %
                             results_path)
        layouts = replay.get("layouts")
        if not isinstance(layouts, list):
            raise ValueError("fresh replay result has no layouts: %s" %
                             results_path)
        replay_keys = []
        for record in layouts:
            if not isinstance(record, dict):
                raise ValueError("fresh replay %s contains a non-object row" %
                                 key)
            case_key = str(record.get("case_id") or "")
            method_key = normalize_method(record.get("method"))
            replay_key = (case_key, method_key)
            replay_keys.append(replay_key)
            if record.get("status") != "ok":
                raise ValueError("fresh replay %s has non-ok row %r" %
                                 (key, replay_key))
            planned = planned_solutions.get(replay_key)
            record_path = _resolve_path(
                record.get("solution_path"), results_path)
            record_digest = str(record.get("solution_sha256") or "").lower()
            if (planned is None or record_path != planned["path"] or
                    record_digest != planned["sha256"]):
                raise ValueError(
                    "fresh replay %s row %r is not bound to its manifest "
                    "solution" % (key, replay_key))
        if (len(replay_keys) != expected_rows or
                len(set(replay_keys)) != expected_rows or
                set(replay_keys) != expected_layout_keys):
            raise ValueError("fresh replay %s result matrix mismatch" % key)
        by_case[key] = {
            "results_path": results_path,
            "results_sha256": digest,
            "expected_rows": expected_rows,
        }

    expected_invocations = {"budget"} | {
        "scaling:%s" % case_id for case_id in EXPECTED_SCALING_CASES}
    if set(by_case) != expected_invocations:
        raise ValueError("fresh manifest replay invocation matrix mismatch")
    if by_case["budget"]["results_path"] != budget_results_path:
        raise ValueError("budget results path is not the manifest-bound replay")
    return {
        "manifest_path": manifest_path,
        "out_root": out_root,
        "run_id": out_root.name,
        "invocations": by_case,
        "solutions": solution_records,
        "case_contracts": case_contracts,
    }


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with open(str(temporary), "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with open(str(temporary), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS,
                                extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            row["errors"] = " | ".join(row.get("errors") or [])
            writer.writerow(row)
    os.replace(str(temporary), str(path))


def _number(value, label, errors, integer=False, positive=False):
    try:
        result = int(value) if integer else float(value)
    except (TypeError, ValueError):
        errors.append("%s is not numeric: %r" % (label, value))
        return None
    if not math.isfinite(float(result)):
        errors.append("%s is not finite" % label)
        return None
    if positive and result <= 0:
        errors.append("%s is not positive: %r" % (label, value))
    return result


def _resolve_path(value, source_json):
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = source_json.parent / path
    return path.resolve()


def _same_number(label, actual, expected, errors, tolerance=1e-12):
    if actual is None or expected is None:
        return
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance,
                        abs_tol=tolerance):
        errors.append("%s=%r, expected %r" % (label, actual, expected))


def _mesh_from_codegen(record, source_json, errors):
    codegen = record.get("codegen") or {}
    directory = _resolve_path(codegen.get("codegen_dir"), source_json)
    coords = directory / "coords.json" if directory else None
    if not coords or not coords.is_file():
        errors.append("missing codegen coords.json")
        return ""
    try:
        value = _load_json(coords).get("mesh") or {}
        rows, cols = int(value["rows"]), int(value["cols"])
        if rows <= 0 or cols <= 0:
            raise ValueError("non-positive mesh")
        return "%dx%d" % (rows, cols)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append("invalid codegen mesh: %s" % exc)
        return ""


def _verify_budget_evidence(record, source_json):
    """Recheck the evidence that exists in the pre-v3 budget replay."""
    errors = []
    legosim = record.get("legosim") or {}
    summary_path = _resolve_path(legosim.get("summary_path"), source_json)
    summary = {}
    if not summary_path or not summary_path.is_file():
        errors.append("missing ilp_e2e_summary.txt")
        evidence_dir = summary_path.parent if summary_path else None
    else:
        evidence_dir = summary_path.parent
        try:
            summary = scaling_audit._parse_key_values(summary_path)
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append("invalid ilp_e2e_summary.txt: %s" % exc)

    embedded = legosim.get("summary") or {}
    for label, actual, expected in (
            ("record status", record.get("status"), "ok"),
            ("legosim status", legosim.get("status"), "ok"),
            ("JSON validation", legosim.get("validation"), "PASS"),
            ("summary validation", summary.get("validation"), "PASS"),
            ("dispatch mode", summary.get("dispatch_mode"),
             "real_dispatch_file"),
            ("interchiplet rc", summary.get("interchiplet_rc"), "0")):
        # Nested budget-core records do not carry a top-level status.
        if label == "record status" and actual is None:
            continue
        if actual != expected:
            errors.append("%s=%r, expected %r" % (label, actual, expected))

    cycles = _number(summary.get("synchronized_makespan_cycles"),
                     "synchronized cycles", errors, integer=True,
                     positive=True)
    bench_cycles = _number(summary.get("makespan_cycles"),
                           "bench cycles", errors, integer=True,
                           positive=True)
    if bench_cycles is not None and cycles is not None and bench_cycles > cycles:
        errors.append("bench cycles exceed synchronized cycles")
    power = _number(summary.get("popnet_total_power"),
                    "PopNet/ORION NoC power", errors, positive=True)
    if evidence_dir:
        noc_active_cycles = scaling_audit._read_delay_info_active_cycles(
            evidence_dir / "delayInfo.txt", errors)
    else:
        errors.append("missing delayInfo.txt")
        noc_active_cycles = None
    noc_energy_uJ = (power * noc_active_cycles / 1000.0
                     if power is not None and
                     noc_active_cycles is not None else None)
    embedded_cycles = _number(
        embedded.get("synchronized_makespan_cycles"),
        "embedded synchronized cycles", errors, integer=True, positive=True)
    embedded_power = _number(embedded.get("popnet_total_power"),
                             "embedded NoC power", errors, positive=True)
    _same_number("embedded synchronized cycles", embedded_cycles, cycles,
                 errors)
    _same_number("embedded NoC power", embedded_power, power, errors)
    measured = _number(legosim.get("measured_makespan_cycles"),
                       "measured cycles", errors, integer=True, positive=True)
    # The legacy runner named the pre-router completion value "measured".
    # Modern replay records expose both quantities explicitly and use
    # ``measured`` for the synchronized end-to-end completion value.  Detect
    # the schema from the explicit fields instead of silently imposing the
    # legacy meaning on new, stricter records.
    modern_cycles = ("bench_makespan_cycles" in legosim or
                     "synchronized_makespan_cycles" in legosim)
    if modern_cycles:
        recorded_bench = _number(
            legosim.get("bench_makespan_cycles"),
            "recorded bench cycles", errors, integer=True, positive=True)
        recorded_sync = _number(
            legosim.get("synchronized_makespan_cycles"),
            "recorded synchronized cycles", errors, integer=True,
            positive=True)
        _same_number("recorded/summary bench cycles", recorded_bench,
                     bench_cycles, errors)
        _same_number("recorded/summary synchronized cycles", recorded_sync,
                     cycles, errors)
        _same_number("measured/synchronized cycles", measured, cycles,
                     errors)
    else:
        _same_number("measured/bench cycles", measured, bench_cycles, errors)

    for name in ("rounds_requested", "rounds_phase1", "rounds_phase2"):
        value = _number(summary.get(name), name, errors, integer=True,
                        positive=True)
        if value is not None and value != 2:
            errors.append("%s=%r, expected 2" % (name, value))

    if evidence_dir and evidence_dir.is_dir():
        diff_path = evidence_dir / "dispatch_matrix.diff"
        if not diff_path.is_file() or diff_path.stat().st_size:
            errors.append("dispatch_matrix.diff is missing or non-empty")
        expected = evidence_dir / "expected_dispatch_matrix.log"
        effective = evidence_dir / "effective_dispatch_matrix.log"
        if not expected.is_file() or not effective.is_file():
            errors.append("missing expected/effective dispatch evidence")
        elif scaling_audit._sha256(expected) != scaling_audit._sha256(effective):
            errors.append("expected/effective dispatch hashes differ")
        for key, name in (
                ("source_input_manifest_sha256",
                 "source_input_manifest.sha256"),
                ("staged_input_manifest_sha256",
                 "staged_input_manifest.sha256")):
            path = evidence_dir / name
            digest = str(summary.get(key, "")).lower()
            if not scaling_audit.SHA256_RE.match(digest):
                errors.append("%s is not a SHA-256" % key)
            elif not path.is_file() or scaling_audit._sha256(path) != digest:
                errors.append("%s file/hash mismatch" % key)

    solution_path = _resolve_path(record.get("solution_path"), source_json)
    solution_sha = str(record.get("solution_sha256") or "").lower()
    if not solution_path or not solution_path.is_file():
        errors.append("missing solution JSON")
    elif not scaling_audit.SHA256_RE.match(solution_sha):
        errors.append("invalid solution SHA-256")
    elif scaling_audit._sha256(solution_path) != solution_sha:
        errors.append("solution SHA-256 mismatch")

    return {
        "cycles": cycles,
        "power": power,
        "noc_active_cycles": noc_active_cycles,
        "noc_energy_uJ": noc_energy_uJ,
        "validation": summary.get("validation"),
        "evidence_dir": str(evidence_dir) if evidence_dir else "",
        "solution_path": str(solution_path) if solution_path else "",
        "solution_sha256": solution_sha,
        "errors": errors,
    }


def _budget_row(record, source_json, experiment_id, case_meta,
                evidence_tier="legacy_budget_mirror_v1"):
    checked = _verify_budget_evidence(record, source_json)
    errors = checked["errors"]
    metrics = record.get("metrics") or record.get("solution_metrics") or {}
    case_id = str(record.get("case_id") or case_meta.get("case_id") or "")
    match = CASE_RE.match(case_id)
    if not match:
        errors.append("invalid budget case_id: %r" % case_id)
    method_raw = str(record.get("method") or "")
    method = normalize_method(method_raw)
    allowed = set(OPTIMIZATION_METHODS) | {"matched_random"}
    if method not in allowed:
        errors.append("unsupported method: %r" % method_raw)
    mesh = _mesh_from_codegen(record, source_json, errors)
    experts = metrics.get("gpu_workers")
    if experts is None:
        experts = (record.get("codegen") or {}).get(
            "dispatch_meta", {}).get("ngpu")
    budgets = case_meta.get("budgets") or {}
    dimension = (case_meta.get("sweep_dimension") or
                 (match.group("kind") if match else ""))
    budget_value = (case_meta.get("budget_value") if case_meta else None)
    if budget_value is None and match:
        budget_value = float(match.group("value"))
    return {
        "schema_version": 1,
        "suite": "budget",
        "experiment_id": experiment_id,
        "case_id": case_id,
        "sweep_dimension": dimension,
        "budget_value": budget_value,
        "area_budget_mm2": budgets.get("a_max"),
        "power_budget_W": budgets.get("p_max"),
        "experts": experts,
        "mesh": mesh,
        "method": method,
        "method_raw": method_raw,
        "cpu_count": metrics.get("cpu_count"),
        "gpu_l_count": metrics.get("gpu_l_count"),
        "gpu_s_count": metrics.get("gpu_s_count"),
        "shared_bom_peak_power_W": "",
        "synchronized_cycles": checked["cycles"],
        "orion_noc_power_W": checked["power"],
        "noc_active_cycles": checked["noc_active_cycles"],
        "nominal_orion_dynamic_noc_energy_uJ": checked["noc_energy_uJ"],
        "legosim_validation": checked["validation"],
        "audit_status": "PASS" if not errors else "FAIL",
        "evidence_tier": evidence_tier,
        "solution_path": checked["solution_path"],
        "solution_sha256": checked["solution_sha256"],
        "evidence_dir": checked["evidence_dir"],
        "source_results_json": str(source_json),
        "errors": errors,
    }


def _scaling_rows(report_path):
    report = _load_json(report_path)
    if report.get("audit_status") != "PASS":
        raise ValueError("scaling audit is not PASS: %s" % report_path)
    rows = report.get("rows") or report.get("audit_rows")
    if not isinstance(rows, list):
        raise ValueError("scaling audit has no row list: %s" % report_path)
    output = []
    for source in rows:
        if source.get("audit_status") != "PASS":
            raise ValueError("scaling audit contains a failed row: %s" %
                             report_path)
        raw_method = source.get("method_raw") or source.get("method")
        method = normalize_method(source.get("method"))
        if method not in OPTIMIZATION_METHODS:
            raise ValueError("unsupported scaling method %r" % method)
        source_results = source.get("results_json")
        if source_results:
            source_results = str(_resolve_path(source_results, report_path))
        output.append({
            "schema_version": 1,
            "suite": "scaling",
            "experiment_id": Path(report.get("input_root") or
                                   report.get("source_root") or
                                   report_path.stem).name,
            "case_id": source.get("case_id"),
            "sweep_dimension": "experts",
            "budget_value": source.get("experts"),
            "area_budget_mm2": "",
            "power_budget_W": "",
            "experts": source.get("experts"),
            "mesh": source.get("mesh"),
            "method": method,
            "method_raw": raw_method,
            "cpu_count": source.get("cpu_count"),
            "gpu_l_count": source.get("gpu_l_count"),
            "gpu_s_count": source.get("gpu_s_count"),
            "shared_bom_peak_power_W": "",
            "synchronized_cycles": source.get("legosim_cycles"),
            "orion_noc_power_W": source.get("orion_noc_power_W"),
            "noc_active_cycles": source.get("noc_active_cycles"),
            "nominal_orion_dynamic_noc_energy_uJ": source.get(
                "nominal_orion_dynamic_noc_energy_uJ"),
            "legosim_validation": source.get("legosim_validation"),
            "audit_status": "PASS",
            "evidence_tier": "strict_scaling_joint_bom_v4",
            "solution_path": "",
            "solution_sha256": "",
            "evidence_dir": source.get("evidence_dir"),
            "source_results_json": source_results or "",
            "errors": [],
        })
    return output


def _budget_case_metadata(core, strict_fresh=False):
    cases = core.get("cases")
    if not isinstance(cases, list):
        raise ValueError("budget core has no cases list")
    case_meta = {}
    for case in cases:
        case_id = str(case.get("case_id") or "")
        if not CASE_RE.match(case_id):
            raise ValueError("invalid budget metadata case_id: %r" % case_id)
        if case_id in case_meta:
            raise ValueError("duplicate budget metadata case_id: %s" %
                             case_id)
        budgets = case.get("budgets")
        if not isinstance(budgets, dict):
            raise ValueError("budget case %s has no budgets object" % case_id)
        for name in ("a_max", "p_max"):
            try:
                value = float(budgets[name])
            except (KeyError, TypeError, ValueError):
                raise ValueError("budget case %s has invalid %s" %
                                 (case_id, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("budget case %s has non-positive %s" %
                                 (case_id, name))
        case_meta[case_id] = case
    if strict_fresh and set(case_meta) != set(EXPECTED_BUDGET_CASES):
        raise ValueError("fresh budget metadata must contain exactly the "
                         "eight paper cases")
    return case_meta


def _fresh_budget_rows(results_path, case_meta, fresh_run):
    """Collect the exact 8x4 fresh budget matrix from one replay file."""
    results_path = Path(results_path).resolve()
    result = _load_json(results_path)
    if result.get("status") != "complete":
        raise ValueError("fresh budget replay manifest is not complete: %s" %
                         results_path)
    layouts = result.get("layouts")
    if not isinstance(layouts, list):
        raise ValueError("fresh budget replay has no layouts list: %s" %
                         results_path)
    expected_keys = {
        (case_id, method)
        for case_id in EXPECTED_BUDGET_CASES for method in BUDGET_METHODS}
    actual_keys = []
    for record in layouts:
        case_id = str(record.get("case_id") or "")
        method = normalize_method(record.get("method"))
        actual_keys.append((case_id, method))
        if record.get("status") != "ok":
            raise ValueError("fresh budget replay contains non-ok row %s/%s" %
                             (case_id, method))
        if case_id not in case_meta:
            raise ValueError("fresh budget replay has unknown case %s" %
                             case_id)
        for label, raw_path in (
                ("solution", record.get("solution_path")),
                ("codegen", (record.get("codegen") or {}).get(
                    "codegen_dir")),
                ("LEGOSim summary", (record.get("legosim") or {}).get(
                    "summary_path"))):
            resolved = _resolve_path(raw_path, results_path)
            if (not resolved or
                    not _is_within(resolved, fresh_run["out_root"])):
                raise ValueError("fresh %s path is missing/outside run for "
                                 "%s/%s" % (label, case_id, method))
    if (len(actual_keys) != EXPECTED_BUDGET_ROWS or
            len(set(actual_keys)) != EXPECTED_BUDGET_ROWS):
        raise ValueError(
            "fresh budget replay must contain %d unique rows" %
            EXPECTED_BUDGET_ROWS)
    if set(actual_keys) != expected_keys:
        missing = sorted(expected_keys - set(actual_keys))
        extra = sorted(set(actual_keys) - expected_keys)
        raise ValueError("fresh budget replay matrix mismatch; missing=%r, "
                         "extra=%r" % (missing, extra))

    rows = []
    for record in layouts:
        rows.append(_budget_row(
            record, results_path, fresh_run["run_id"],
            case_meta[str(record.get("case_id"))],
            evidence_tier="fresh_constraint_replay_v1"))
    return rows


def _attach_fresh_case_contracts(rows, fresh_run):
    """Attach the manifest-bound joint-selection contract to every fresh row."""
    contracts = fresh_run.get("case_contracts") or {}
    for row in rows:
        key = (str(row.get("suite")), str(row.get("case_id")))
        contract = contracts.get(key)
        if not isinstance(contract, dict):
            raise ValueError("fresh row %s:%s has no ILP BOM contract" % key)
        for field in ("cpu_count", "gpu_l_count", "gpu_s_count"):
            actual = _required_integer(
                row.get(field), "fresh row %s:%s/%s %s" %
                (key[0], key[1], row.get("method"), field))
            if actual != contract[field]:
                raise ValueError(
                    "fresh row %s:%s/%s changed the shared BOM (%s=%r, "
                    "expected %r)" %
                    (key[0], key[1], row.get("method"), field, actual,
                     contract[field]))
        for field in (
                "area_budget_mm2", "power_budget_W", "area_used_mm2",
                "shared_bom_peak_power_W", "bom_selection_provenance",
                "gpu_counts_fixed", "fixed_bom", "solver_mip_gap"):
            row[field] = contract[field]


def _require_exact_complete_matrix(rows, fresh_run):
    """Reject anything other than the paper's fresh 47-row matrix."""
    expected = {
        ("budget", case_id, method)
        for case_id in EXPECTED_BUDGET_CASES for method in BUDGET_METHODS}
    expected.update({
        ("scaling", case_id, method)
        for case_id in EXPECTED_SCALING_CASES
        for method in OPTIMIZATION_METHODS})
    actual = [(str(row.get("suite")), str(row.get("case_id")),
               str(row.get("method"))) for row in rows]
    if (len(actual) != EXPECTED_FULL_ROWS or
            len(set(actual)) != EXPECTED_FULL_ROWS):
        raise ValueError(
            "complete paper collection requires %d unique rows" %
            EXPECTED_FULL_ROWS)
    if set(actual) != expected:
        missing = sorted(expected - set(actual))
        extra = sorted(set(actual) - expected)
        raise ValueError("complete paper matrix mismatch; missing=%r, "
                         "extra=%r" % (missing, extra))

    budget_results = fresh_run["invocations"]["budget"]["results_path"]
    for row in rows:
        if _required_integer(row.get("cpu_count"), "fresh row cpu_count") != 1:
            raise ValueError("fresh row %s/%s/%s does not have one CPU" %
                             (row["suite"], row["case_id"], row["method"]))
        source = Path(str(row.get("source_results_json") or "")).resolve()
        if row["suite"] == "budget":
            if source != budget_results:
                raise ValueError("budget row came from a non-fresh source")
            if (_required_integer(row.get("experts"),
                                  "budget experts") != 8 or
                    row.get("mesh") != "6x6"):
                raise ValueError("budget row workload/mesh mismatch")
        else:
            experts, mesh = EXPECTED_SCALING_CASES[row["case_id"]]
            if (_required_integer(row.get("experts"),
                                  "scaling experts") != experts or
                    row.get("mesh") != mesh):
                raise ValueError("scaling row %s workload/mesh mismatch" %
                                 row["case_id"])
            expected_source = fresh_run["invocations"][
                "scaling:%s" % row["case_id"]]["results_path"]
            if source != expected_source:
                raise ValueError("scaling row %s came from a non-fresh source" %
                                 row["case_id"])
            evidence = Path(str(row.get("evidence_dir") or "")).resolve()
            if not _is_within(evidence, fresh_run["out_root"]):
                raise ValueError("scaling evidence is outside the fresh run")


def collect(args):
    rows = []
    for raw_path in args.scaling_audit:
        rows.extend(_scaling_rows(Path(raw_path).resolve()))

    core_path = Path(args.budget_core).resolve()
    core = _load_json(core_path)
    if core.get("status") != "complete":
        raise ValueError("budget core experiment is not complete")
    fresh_results_value = getattr(args, "fresh_budget_results", None)
    strict_complete = bool(getattr(args, "require_complete_47", False))
    fresh_run = None
    if fresh_results_value:
        if getattr(args, "additional_results", []):
            raise ValueError("--additional-results cannot be combined with "
                             "--fresh-budget-results")
        fresh_results_path = Path(fresh_results_value).resolve()
        manifest_value = getattr(args, "fresh_run_manifest", None)
        if manifest_value:
            manifest_path = Path(manifest_value).resolve()
        else:
            try:
                manifest_path = fresh_results_path.parents[2] / "manifest.json"
            except IndexError:
                raise ValueError("cannot infer the fresh run manifest path")
        fresh_run = _verify_fresh_run_manifest(
            manifest_path, fresh_results_path)
        case_meta = _budget_case_metadata(core, strict_fresh=True)
        # Deliberately do not inspect core case layouts, --budget-proxies, or
        # any historical replay here.  The single manifest-bound file is the
        # sole source of all 32 budget measurements.
        rows.extend(_fresh_budget_rows(
            fresh_results_path, case_meta, fresh_run))
    else:
        if strict_complete:
            raise ValueError("--require-complete-47 requires "
                             "--fresh-budget-results")
        case_meta = _budget_case_metadata(core)
        cases = core.get("cases")
        core_id = str((core.get("experiment") or {}).get("run_id") or
                      (core.get("experiment") or {}).get("experiment_id") or
                      core_path.parent.name)
        for case in cases:
            layouts = case.get("layouts")
            if not isinstance(layouts, list):
                raise ValueError("budget case %s has no layouts" %
                                 case.get("case_id"))
            for record in layouts:
                record = dict(record)
                record.setdefault("case_id", case.get("case_id"))
                rows.append(_budget_row(record, core_path, core_id, case))

        proxy_path = Path(args.budget_proxies).resolve()
        result_paths = [proxy_path] + [Path(value).resolve()
                                       for value in args.additional_results]
        for result_path in result_paths:
            result = _load_json(result_path)
            if result.get("status") != "complete":
                raise ValueError(
                    "budget replay manifest is not complete: %s" %
                    result_path)
            layouts = result.get("layouts")
            if not isinstance(layouts, list):
                raise ValueError("replay manifest has no layouts: %s" %
                                 result_path)
            for record in layouts:
                meta = case_meta.get(str(record.get("case_id")), {})
                rows.append(_budget_row(record, result_path,
                                        result_path.parent.name, meta))

    if strict_complete:
        _attach_fresh_case_contracts(rows, fresh_run)
        _require_exact_complete_matrix(rows, fresh_run)

    seen = {}
    global_errors = []
    for row in rows:
        key = (row["suite"], row["case_id"], row["method"])
        if key in seen:
            message = "duplicate suite/case/method: %s/%s/%s" % key
            global_errors.append(message)
            row["errors"].append(message)
            row["audit_status"] = "FAIL"
        else:
            seen[key] = row

    missing = []
    cases_by_suite = {}
    for row in rows:
        cases_by_suite.setdefault(row["suite"], set()).add(row["case_id"])
    methods_by_case = {}
    for row in rows:
        methods_by_case.setdefault((row["suite"], row["case_id"]), set()).add(
            row["method"])
    for suite, suite_cases in sorted(cases_by_suite.items()):
        for case_id in sorted(suite_cases):
            for method in OPTIMIZATION_METHODS:
                if method not in methods_by_case[(suite, case_id)]:
                    missing.append({"suite": suite, "case_id": case_id,
                                    "method": method})
    rows.sort(key=lambda row: (row["suite"], str(row["case_id"]),
                               row["method"]))
    failed = [row for row in rows if row["audit_status"] != "PASS"]
    report = {
        "schema_version": 1,
        "audit_status": "PASS" if not failed and not global_errors else "FAIL",
        "metric_contract": {
            "cycles": "LEGOSim synchronized_makespan_cycles",
            "power": "PopNet/ORION NoC-only total power (W)",
            "shared_bom_peak_power": (
                "manifest-bound CPU+GPU package peak power (W), shared by "
                "all methods in a case"),
            "noc_active_cycles": (
                "max(srcTime + final packetDelay) from delayInfo.txt"),
            "nominal_orion_dynamic_noc_energy_uJ": (
                "orion_noc_power_W * noc_active_cycles / 1000 at a "
                "nominal 1-GHz clock"),
            "analytical_fallback": False,
        },
        "optimization_methods": list(OPTIMIZATION_METHODS),
        "production_methods": {
            "budget": list(BUDGET_METHODS),
            "scaling": list(OPTIMIZATION_METHODS),
        },
        "baseline_methods": ["matched_random"],
        "budget_collection_mode": (
            "fresh_manifest_bound" if fresh_run else "legacy_split"),
        "budget_metadata_json": str(core_path),
        "fresh_run_manifest": (str(fresh_run["manifest_path"])
                               if fresh_run else ""),
        "fresh_run_root": (str(fresh_run["out_root"])
                           if fresh_run else ""),
        "strict_complete_47": strict_complete,
        "row_count": len(rows),
        "pass_count": len(rows) - len(failed),
        "fail_count": len(failed),
        "missing_optional_method_rows": missing,
        "global_errors": global_errors,
        "rows": rows,
    }
    return report


def main(argv=None):
    args = _parse_args(argv)
    if not args.scaling_audit:
        raise ValueError("at least one --scaling-audit JSON is required")
    report = collect(args)
    prefix = Path(args.out_prefix).resolve()
    json_path = Path(str(prefix) + ".json")
    csv_path = Path(str(prefix) + ".csv")
    _atomic_json(json_path, report)
    _atomic_csv(csv_path, report["rows"])
    print(json.dumps({
        "audit_status": report["audit_status"],
        "rows": report["row_count"],
        "passed": report["pass_count"],
        "failed": report["fail_count"],
        "missing_optional": len(report["missing_optional_method_rows"]),
        "json": str(json_path),
        "csv": str(csv_path),
    }, indent=2, sort_keys=True))
    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        sys.exit(2)
