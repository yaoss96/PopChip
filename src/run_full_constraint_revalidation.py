#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fresh constrained-model solve, baseline generation, and LEGOSim replay.

The old revalidation driver starts from historical placements.  That is
unsuitable after changing the formulation (free CPU placement, configurable
expert capacity, conditional co-resident ordering, and Mesh-flow capacity).
This driver instead owns the complete dependency chain for every paper case:

    fresh ILP-Perf (sole joint min-T protocol objective)
        -> Perf-BOM/bundle-matched Matched-Random, Floorplet, TAP
        -> fresh codegen and LEGOSim/PopNet replay

Historical experiment files are read only for the eight published 6x6 budget
points and workload parameters.  Scale-out package limits instead follow one
target-BOM-independent rule, Amax=180E mm2 and Pmax=140E W.  No historical
layout is used as a solve or baseline reference.  Budget cases replay four
methods (8*4=32 layouts), while scaling cases replay three methods (5*3=15
layouts), for 47 fresh rows total.
"""

from __future__ import print_function

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from run_scaling_qap_experiment import (
    SCALING_AREA_MM2_PER_EXPERT,
    SCALING_BUDGET_CONTRACT_VERSION,
    SCALING_BUDGET_POLICY,
    SCALING_POWER_W_PER_EXPERT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = "out/full_constraint_revalidation_perf_only_20260801"
DEFAULT_BUDGET_SOURCE = (
    "out/corrected_e2e_runs/corrected_e2e_20260731_v2/raw_results.json")
DEFAULT_SCALE_SOURCE = "src/run_scaling_qap_experiment.py"
DEFAULT_SCALE_PARAMS = "out/scaling_qap_experiment/params"
DEFAULT_GUROBI_PYTHON = os.environ.get("GUROBI_PYTHON", sys.executable)
SCALES = ((8, 6), (16, 8), (32, 12), (64, 12), (128, 16))
BUDGET_METHODS = ("ilp-perf", "matched-random", "floorplet", "tap")
SCALING_METHODS = ("ilp-perf", "floorplet", "tap")
REPLAY_COST_RANK = {
    "scaling_e128_m16": 6,
    "scaling_e064_m12": 5,
    "scaling_e032_m12": 4,
    "scaling_e016_m08": 3,
    "scaling_e008_m06": 2,
    "budget": 1,
}
PRIMARY_T_REL_TOL = 1.0e-9
PRIMARY_T_ABS_TOL_CYCLES = 1.0e-6
ILP_TOTAL_BUDGET_S = 45.0
FLOORPLET_BUDGET_S = 45.0
TAP_SEEDS = (11, 29, 47)
TAP_BUDGET_S_PER_SEED = 15.0
TAP_TOTAL_BUDGET_S = TAP_BUDGET_S_PER_SEED * len(TAP_SEEDS)
FORMAL_WALL_BUDGETS_S_PER_CASE = {
    "ilp_perf_total": ILP_TOTAL_BUDGET_S,
    "floorplet": FLOORPLET_BUDGET_S,
    "tap_total": TAP_TOTAL_BUDGET_S,
    "tap_per_seed": TAP_BUDGET_S_PER_SEED,
}
PROTOCOL = {
    "schema_version": 5,
    "fresh_ilp_perf": True,
    "production_methods": {
        "budget": list(BUDGET_METHODS),
        "scaling": list(SCALING_METHODS),
    },
    "expected_full_replay_rows": 47,
    "historical_layout_reference": False,
    "cpu_count": 1,
    "cpu_anchor": "free",
    "package_area_and_peak_power_budgets": True,
    "scaling_budget_contract_version": SCALING_BUDGET_CONTRACT_VERSION,
    "scaling_budget_policy": SCALING_BUDGET_POLICY,
    "scaling_area_mm2_per_expert": SCALING_AREA_MM2_PER_EXPERT,
    "scaling_power_W_per_expert": SCALING_POWER_W_PER_EXPERT,
    "scaling_budget_target_bom": None,
    "local_power_density_constraint": False,
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
    "gpu_type_counts_fixed": False,
    "gpu_type_count_definition": "N_c=sum_p x[c,p]",
    "primary_incumbent_gap_bound_metric": (
        "replay-aligned end-to-end protocol completion T"),
    "protocol_completion_role": "sole-primary-objective",
    "protocol_completion_in_T": True,
    "protocol_completion_used_as_objective": True,
    "legacy_analytical_task_makespan_role": "hard-feasibility-only",
    "public_performance_fields": "replay-aligned-protocol-T",
    "stage0_public_T_evidence": "protocol_T_start_cycles",
    "first_feasible_clock": (
        "model-build-and-stage0-refinement-plus-primary-runtime"),
    "dispatch_tokens_per_source": 32,
    "dispatch_hidden": 128,
    "threads": 1,
    "formal_wall_budgets_s_per_case": dict(
        FORMAL_WALL_BUDGETS_S_PER_CASE),
    "ilp_budget_scope": (
        "total wall time including formulation, Stage-0 refinement, and "
        "Gurobi optimization"),
    "ilp_total_time_limit_s": ILP_TOTAL_BUDGET_S,
    "gurobi_mip_focus": 1,
    "matched_random_seeds": list(range(100)),
    "matched_random_selection_metric": "public_T_cycles",
    "matched_random_selection_population": (
        "all 100 strictly legal candidates"),
    "matched_random_legacy_task_makespan_scored": False,
    "floorplet_time_limit_s": FLOORPLET_BUDGET_S,
    "tap_wall_budget_s_per_seed": TAP_BUDGET_S_PER_SEED,
    "tap_total_wall_budget_s": TAP_TOTAL_BUDGET_S,
    "tap_seeds": list(TAP_SEEDS),
    "tap_alpha": 0.7,
}

PROTOCOL_CYCLE_FIELDS = (
    "protocol_makespan_cycles",
    "protocol_completion_cycles",
    "protocol_surrogate_cycles",
    "event_dag_makespan_cycles",
)
PROTOCOL_CONTAINERS = (
    "protocol_model", "protocol_surrogate", "performance_proxy",
    "protocol_event_dag")
PROTOCOL_PROFILE_FIELDS = (
    "protocol_profile", "performance_proxy_profile",
    "protocol_event_profile", "event_dag_profile")
PROTOCOL_EVENT_FIELDS = (
    "protocol_event_metadata", "performance_proxy_event_metadata",
    "protocol_events", "event_dag_events", "protocol_event_count")
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the complete fresh-constraint 47-layout protocol")
    parser.add_argument(
        "--phase", choices=("solve-only", "baselines-only", "replay-only",
                            "all"), default="solve-only")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--budget-source", default=DEFAULT_BUDGET_SOURCE)
    parser.add_argument(
        "--scale-source", default=DEFAULT_SCALE_SOURCE,
        help=("source file defining the target-BOM-independent scaling "
              "budget policy; it is content-hashed into the manifest"))
    parser.add_argument("--scale-params", default=DEFAULT_SCALE_PARAMS)
    parser.add_argument("--gurobi-python", default=DEFAULT_GUROBI_PYTHON)
    parser.add_argument("--system-python", default=sys.executable)
    parser.add_argument(
        "--case", action="append", default=[], metavar="SUITE:CASE",
        help="select budget:area_970 or scaling:e008_m06; repeat as needed")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--budget-e2e-timeout", type=int, default=1800)
    parser.add_argument("--scale-e2e-timeout", type=int, default=21600)
    parser.add_argument("--budget-legosim-timeout", type=int, default=2400)
    parser.add_argument("--scale-legosim-timeout", type=int, default=25200)
    parser.add_argument("--moe-gpgpusim-max-parallel", type=int, default=16)
    parser.add_argument("--replay-jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse verified artifacts in an existing protocol root")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate inputs and print commands without creating output")
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_json(path, label):
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read %s %s: %s" % (label, path, exc))
    if not isinstance(value, dict):
        raise ValueError("%s must be a JSON object: %s" % (label, path))
    return value


def _require_file(path, label):
    if not Path(path).is_file():
        raise ValueError("missing %s: %s" % (label, path))


def _sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with open(str(temporary), "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with open(str(temporary), "w", encoding="utf-8", newline="") as handle:
        handle.write(value)
    os.replace(str(temporary), str(path))


def _budget_cases(source_path):
    source = _load_json(source_path, "budget metadata")
    if source.get("status") != "complete":
        raise ValueError("budget metadata is not complete")
    params_record = source.get("params") or {}
    params = _resolve(params_record.get("path"))
    _require_file(params, "budget params")
    expected_sha = str(params_record.get("sha256") or "")
    if expected_sha and _sha256(params) != expected_sha:
        raise ValueError("budget params SHA-256 mismatch")
    rows = source.get("cases")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("budget metadata must contain exactly 8 cases")
    experiment = source.get("experiment") or {}
    cases = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        budgets = row.get("budgets") or {}
        source_metrics = ((row.get("ilp") or {}).get("metrics") or {})
        try:
            performance_start_bom = {
                "CPU": 1,
                "GPU-L": int(source_metrics["gpu_l_count"]),
                "GPU-S": int(source_metrics["gpu_s_count"]),
            }
        except (KeyError, TypeError, ValueError):
            raise ValueError("%s budget metadata lacks its tier BOM" % case_id)
        cases.append({
            "key": "budget:%s" % case_id,
            "suite": "budget",
            "case_id": case_id,
            "params": params,
            "grid": (1, 6, 6),
            # Deliberately retain only package budgets.  Historical p_col is
            # the removed C9 and must never enter a fresh command.
            "budgets": {"a_max": float(budgets["a_max"]),
                        "p_max": float(budgets["p_max"])},
            # The published tier seeds a strong performance-first incumbent.
            # It is not a retained BOM: the sole min-T MILP remains free to
            # choose a different legal chiplet mix under the package budgets.
            "performance_start_bom": performance_start_bom,
            "mipgap": float(experiment.get("mipgap_target", 1.0e-6)),
            "source_metadata": source_path,
        })
    return cases


def _scale_cases(policy_source, params_root):
    """Build scale cases without reading a historical target BOM.

    ``policy_source`` is hashed as provenance only.  The numerical package
    limits are computed from one declared continuous per-expert rule; no
    integer GPU-L/GPU-S count participates in either right-hand side.
    """
    _require_file(policy_source, "scaling budget policy source")
    cases = []
    for experts, side in SCALES:
        case_id = "e%03d_m%02d" % (experts, side)
        params = params_root / ("experts_%03d.json" % experts)
        _require_file(params, "%s params" % case_id)
        params_value = _load_json(params, "%s params" % case_id)
        if int((params_value.get("meta") or {}).get("experts", -1)) != experts:
            raise ValueError("%s params expert count mismatch" % case_id)
        mesh = (1, side, side)
        budgets = {
            "a_max": float(SCALING_AREA_MM2_PER_EXPERT * experts),
            "p_max": float(SCALING_POWER_W_PER_EXPERT * experts),
        }
        cases.append({
            "key": "scaling:%s" % case_id,
            "suite": "scaling",
            "case_id": case_id,
            "experts": experts,
            "params": params.resolve(),
            "grid": mesh,
            "budgets": budgets,
            # Stage-0 is a legal MIP start, never a retained count contract.
            # Its exact typed BOM can include an auxiliary GPU, so the fresh
            # output rather than a predicted start supplies Sel. L/S.
            "performance_start_bom": None,
            "mipgap": 0.01,
            "source_metadata": policy_source.resolve(),
            "scaling_budget_contract_version": (
                SCALING_BUDGET_CONTRACT_VERSION),
            "budget_policy": SCALING_BUDGET_POLICY,
            "budget_target_bom": None,
        })
    return cases


def _select_cases(cases, requested):
    by_key = {case["key"]: case for case in cases}
    if len(by_key) != len(cases):
        raise ValueError("duplicate canonical case key")
    if not requested:
        return cases
    duplicates = sorted(key for key in set(requested)
                        if requested.count(key) > 1)
    if duplicates:
        raise ValueError("duplicate --case: %s" % ", ".join(duplicates))
    unknown = sorted(set(requested) - set(by_key))
    if unknown:
        raise ValueError("unknown --case: %s" % ", ".join(unknown))
    return [by_key[key] for key in requested]


def _case_root(out_root, case):
    return out_root / "solutions" / case["suite"] / case["case_id"]


def _solution_path(out_root, case, method):
    root = _case_root(out_root, case)
    if method == "ilp-perf":
        return root / "ilp_perf.json"
    baseline = out_root / "baselines" / case["suite"] / case["case_id"]
    if method == "matched-random":
        return baseline / "matched" / "representative.json"
    if method == "floorplet":
        return baseline / "prior" / "floorplet.json"
    if method == "tap":
        return baseline / "prior" / "tap.json"
    raise ValueError("unknown method %s" % method)


def _methods(case):
    return BUDGET_METHODS if case["suite"] == "budget" else SCALING_METHODS


def _ilp_command(case, method, out_path, gurobi_python, out_root):
    if method != "ilp-perf":
        raise ValueError("invalid ILP method %s" % method)
    l_dim, x_dim, y_dim = case["grid"]
    command = [
        str(gurobi_python), str(PROJECT_ROOT / "src" / "ilp_layout.py"),
        "--params", str(case["params"]),
        "--L", str(l_dim), "--X", str(x_dim), "--Y", str(y_dim),
        "--a-max", str(case["budgets"]["a_max"]),
        "--p-max", str(case["budgets"]["p_max"]),
        "--n-cpu", "1",
        "--time-limit", str(ILP_TOTAL_BUDGET_S),
        "--mipgap", str(case["mipgap"]),
        "--seed", "1",
        "--mip-focus", "1",
        "--mode", "ilp-perf",
        "--noc-traffic", "deterministic-dispatch",
        "--dispatch-tokens-per-source", "32",
        "--dispatch-hidden", "128",
        "--cpu-anchor", "free",
        "--threads", "1",
    ]
    command.extend(["--quiet", "-o", str(out_path)])
    return command


def _matched_command(case, out_root, system_python):
    if case["suite"] != "budget":
        raise ValueError("Matched-Random is only part of the budget suite")
    command = [
        str(system_python),
        str(PROJECT_ROOT / "src" / "matched_layout_baseline.py"),
        "--params", str(case["params"]),
        "--solution", str(_solution_path(out_root, case, "ilp-perf")),
        "--seeds",
    ]
    command.extend(str(seed) for seed in PROTOCOL["matched_random_seeds"])
    command.extend([
        "--out-dir", str(_solution_path(
            out_root, case, "matched-random").parent)])
    return command


def _prior_command(case, out_root, gurobi_python):
    return [
        str(gurobi_python),
        str(PROJECT_ROOT / "src" / "generate_revalidation_baselines.py"),
        "--params", str(case["params"]),
        "--source", str(_solution_path(out_root, case, "ilp-perf")),
        "--case-id", case["case_id"],
        "--out-dir", str(_solution_path(
            out_root, case, "floorplet").parent),
        "--floorplet-time-limit", str(FLOORPLET_BUDGET_S),
        "--tap-wall-budget", str(TAP_BUDGET_S_PER_SEED),
        "--tap-seeds", ",".join(str(seed) for seed in TAP_SEEDS),
        "--tap-alpha", "0.7",
        "--threads", "1",
    ]


def _replay_groups(cases):
    groups = []
    budget = [case for case in cases if case["suite"] == "budget"]
    if budget:
        groups.append(("budget", budget))
    for case in cases:
        if case["suite"] == "scaling":
            groups.append(("scaling_" + case["case_id"], [case]))
    return groups


def _replay_command(cases, out_root, replay_dir, system_python, args):
    params = {str(case["params"]) for case in cases}
    if len(params) != 1:
        raise ValueError("one replay invocation must share one params file")
    is_budget = cases[0]["suite"] == "budget"
    command = [
        str(system_python),
        str(PROJECT_ROOT / "src" / "replay_legosim_solutions.py"),
        "--params", next(iter(params)),
    ]
    for case in cases:
        for method in _methods(case):
            command.extend([
                "--layout", "%s:%s:%s" % (
                    case["case_id"], method,
                    _solution_path(out_root, case, method)),
            ])
    command.extend([
        "--out-dir", str(replay_dir),
        "--rounds", str(args.rounds),
        "--moe-e2e-timeout", str(
            args.budget_e2e_timeout if is_budget else args.scale_e2e_timeout),
        "--legosim-timeout", str(
            args.budget_legosim_timeout if is_budget
            else args.scale_legosim_timeout),
        "--moe-gpgpusim-max-parallel",
        str(args.moe_gpgpusim_max_parallel),
        "--fail-fast",
    ])
    return command


def _run_command(command, stdout_path, stderr_path):
    process = subprocess.run(
        command, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, universal_newlines=True)
    _atomic_text(stdout_path, process.stdout)
    _atomic_text(stderr_path, process.stderr)
    return process.returncode


def _nonempty_metadata(value):
    if isinstance(value, (dict, list, tuple, str)):
        return len(value) > 0
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    return value is not None


def _first_protocol_value(solution, fields, nested_fields=()):
    for field in fields:
        value = solution.get(field)
        if value is not None:
            return value, field
    for container_name in PROTOCOL_CONTAINERS:
        container = solution.get(container_name)
        if not isinstance(container, dict):
            continue
        for field in tuple(fields) + tuple(nested_fields):
            value = container.get(field)
            if value is not None:
                return value, "%s.%s" % (container_name, field)
    return None, None


def _protocol_surrogate_contract(solution, method):
    """Validate the replay-aligned protocol completion used as MILP T.

    The function name is retained for callers which imported the previous
    validator.  The protocol completion is no longer a diagnostic surrogate:
    it is the sole public optimization objective and certificate metric.
    """
    raw_cycles = solution.get("protocol_makespan_cycles")
    cycle_field = "protocol_makespan_cycles"
    try:
        cycles = float(raw_cycles)
    except (TypeError, ValueError):
        raise RuntimeError(
            "%s lacks numeric replay-aligned protocol_makespan_cycles" %
            method)
    if not math.isfinite(cycles) or not (cycles > 0.0):
        raise RuntimeError("%s has a non-positive protocol makespan" % method)

    profile, profile_field = _first_protocol_value(
        solution, PROTOCOL_PROFILE_FIELDS,
        ("profile", "protocol_profile", "event_profile",
         "phase_profile"))
    if not _nonempty_metadata(profile):
        # A structured protocol container may itself be the profile.
        for container_name in PROTOCOL_CONTAINERS:
            container = solution.get(container_name)
            if (isinstance(container, dict) and container and any(
                    key in container for key in
                    ("phases", "phase_completion", "worker_compute",
                     "event_count", "events"))):
                profile, profile_field = container, container_name
                break
    if not _nonempty_metadata(profile):
        raise RuntimeError("%s lacks protocol profile metadata" % method)

    events, event_field = _first_protocol_value(
        solution, PROTOCOL_EVENT_FIELDS,
        ("events", "event_metadata", "event_count", "protocol_events"))
    if not _nonempty_metadata(events) and isinstance(profile, dict):
        for key in ("events", "event_metadata", "event_count",
                    "protocol_events"):
            if _nonempty_metadata(profile.get(key)):
                events, event_field = profile.get(key), "%s.%s" % (
                    profile_field, key)
                break
    if not _nonempty_metadata(events):
        raise RuntimeError("%s lacks protocol event metadata" % method)

    protocol_model = solution.get("protocol_model")
    if (not isinstance(protocol_model, dict) or
            protocol_model.get("diagnostic_only") is not False or
            protocol_model.get("used_as_objective") is not True or
            protocol_model.get("included_in_T") is not True or
            protocol_model.get("used_as_separate_objective") is not False):
        raise RuntimeError(
            "%s does not mark protocol completion as the sole T objective" %
            method)
    try:
        model_cycles = float(protocol_model.get("completion_cycles"))
    except (TypeError, ValueError):
        raise RuntimeError(
            "%s protocol_model lacks numeric completion_cycles" % method)
    tolerance = max(
        PRIMARY_T_ABS_TOL_CYCLES,
        PRIMARY_T_REL_TOL * max(abs(cycles), abs(model_cycles)))
    if (not math.isfinite(model_cycles) or model_cycles <= 0.0 or
            abs(model_cycles - cycles) > tolerance):
        raise RuntimeError(
            "%s protocol_model completion differs from protocol makespan" %
            method)

    return {
        "cycles": cycles,
        "cycles_field": cycle_field,
        "profile": profile,
        "profile_field": profile_field,
        "event_metadata": events,
        "event_metadata_field": event_field,
        "role": "sole-primary-objective",
        "is_optimization_objective": True,
        # Optimality (when available) comes from the primary bound/gap, not
        # from the mere existence of this incumbent metric.
        "is_optimality_certificate": False,
    }


def _complete_bundle_contract(solution, method):
    """Validate the final typed task bundles used by every baseline."""
    sites = solution.get("sites")
    task_site = solution.get("task_site")
    if not isinstance(sites, list) or not sites:
        raise RuntimeError("%s lacks non-empty serialized sites" % method)
    if not isinstance(task_site, dict) or not task_site:
        raise RuntimeError("%s lacks a serialized task-to-site map" % method)

    anchors = set()
    bundled_tasks = {}
    for ordinal, site in enumerate(sites):
        try:
            anchor = (int(site["l"]), int(site["i"]), int(site["j"]))
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("%s site %d has an invalid anchor" %
                               (method, ordinal))
        tasks = site.get("tasks")
        if (anchor in anchors or not site.get("type") or
                not isinstance(tasks, list) or not tasks):
            raise RuntimeError("%s site %d is not a complete typed bundle" %
                               (method, ordinal))
        anchors.add(anchor)
        for task in tasks:
            name = str(task)
            if name in bundled_tasks:
                raise RuntimeError("%s task %s occurs in multiple bundles" %
                                   (method, name))
            bundled_tasks[name] = anchor

    if set(str(name) for name in task_site) != set(bundled_tasks):
        raise RuntimeError(
            "%s sites and task_site do not cover the same tasks" % method)
    for task, expected_anchor in bundled_tasks.items():
        try:
            recorded_anchor = tuple(int(value) for value in task_site[task])
        except (TypeError, ValueError):
            raise RuntimeError("%s task %s has an invalid task_site entry" %
                               (method, task))
        if recorded_anchor != expected_anchor:
            raise RuntimeError("%s task %s disagrees with its bundle anchor" %
                               (method, task))

    signature = str(solution.get("bundle_signature") or "")
    if not signature:
        raise RuntimeError("%s lacks its complete-bundle signature" % method)
    return {
        "bundle_count": len(sites),
        "task_count": len(bundled_tasks),
        "bundle_signature": signature,
    }


def _primary_min_t_contract(value, method):
    """Validate the sole joint min-T incumbent, bound, and gap evidence."""
    if "primary_mip_gap" not in value:
        raise RuntimeError("%s lacks the primary T MIP gap field" % method)
    if value.get("primary_has_incumbent") is not True:
        raise RuntimeError("%s primary min-T solve lacks an incumbent" % method)
    try:
        incumbent = float(value.get("primary_incumbent_cycles"))
    except (TypeError, ValueError):
        raise RuntimeError(
            "%s primary min-T solve lacks a finite positive incumbent" %
            method)
    if not math.isfinite(incumbent) or incumbent <= 0.0:
        raise RuntimeError(
            "%s primary min-T solve lacks a finite positive incumbent" %
            method)

    raw_bound = value.get("primary_objective_bound_cycles")
    if raw_bound is None:
        bound = None
    else:
        try:
            bound = float(raw_bound)
        except (TypeError, ValueError):
            raise RuntimeError("%s has a non-numeric primary T bound" % method)
        # A time-limited minimization can legitimately retain the trivial
        # lower bound zero even when its incumbent T is positive.
        if not math.isfinite(bound) or bound < 0.0:
            raise RuntimeError("%s has an invalid primary T bound" % method)
        tolerance = max(
            PRIMARY_T_ABS_TOL_CYCLES,
            PRIMARY_T_REL_TOL * max(abs(incumbent), abs(bound)))
        if bound > incumbent + tolerance:
            raise RuntimeError(
                "%s primary T bound exceeds its incumbent" % method)

    raw_gap = value.get("primary_mip_gap")
    certificate = value.get("primary_gap_certificate")
    if raw_gap is None:
        if not isinstance(certificate, dict):
            raise RuntimeError(
                "%s uncertified primary T gap lacks certificate evidence" %
                method)
        reason = certificate.get("reason")
        if (certificate.get("certified") is not False or
                not isinstance(reason, str) or not reason.strip()):
            raise RuntimeError(
                "%s uncertified primary T gap lacks a structured reason" %
                method)
        return {
            "objective": "T",
            "gap_field": "primary_mip_gap",
            "gap": None,
            "gap_certified": False,
            "gap_reason": reason,
            "has_incumbent": True,
            "incumbent_cycles": incumbent,
            "bound_cycles": bound,
        }

    try:
        gap = float(raw_gap)
    except (TypeError, ValueError):
        raise RuntimeError("%s has a non-numeric primary T MIP gap" % method)
    if not math.isfinite(gap):
        raise RuntimeError("%s has a non-finite primary T MIP gap" % method)
    if gap < 0.0:
        raise RuntimeError("%s has a negative primary T MIP gap" % method)
    if (not isinstance(certificate, dict) or
            certificate.get("certified") is not True):
        raise RuntimeError(
            "%s finite primary T MIP gap lacks a certified marker" % method)
    if bound is None:
        raise RuntimeError("%s certified primary T gap lacks a bound" % method)
    return {
        "objective": "T",
        "gap_field": "primary_mip_gap",
        "gap": gap,
        "gap_certified": True,
        "gap_reason": None,
        "has_incumbent": True,
        "incumbent_cycles": incumbent,
        "bound_cycles": bound,
    }


def _joint_min_t_output_contract(value, method, incumbent):
    """Validate that every public performance alias is protocol MILP T.

    ``analytical_task_makespan_cycles`` is deliberately not a public
    performance alias.  It serializes the legacy scheduling horizon T_task,
    which is retained only to witness hard task-feasibility constraints.
    """
    expected_objective = "minimize joint MILP makespan T"
    if value.get("primary_objective") != expected_objective:
        raise RuntimeError(
            "%s primary_objective is not the sole joint min-T objective" %
            method)

    contract = value.get("objective_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("%s lacks the joint min-T objective contract" %
                           method)
    expected_contract = {
        "active_objective_count": 1,
        "active_objective": "min T",
        "joint_selection_placement_scheduling": True,
        "protocol_completion_in_T": True,
        "protocol_used_as_separate_objective": False,
        "legacy_task_schedule_role": "hard-feasibility-only",
    }
    for field, expected in expected_contract.items():
        actual = contract.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError(
                "%s objective_contract.%s=%r, expected %r" %
                (method, field, actual, expected))
    if contract.get("energy_or_byte_hop_tie_break") is not False:
        raise RuntimeError(
            "%s enables an energy/byte-hop objective or tie-break" % method)

    public_fields = (
        "predicted_makespan_cycles",
        "performance_proxy_cycles",
        "tightened_performance_proxy_cycles",
        "protocol_makespan_cycles",
    )
    normalized = {}
    for field in public_fields:
        try:
            cycles = float(value.get(field))
        except (TypeError, ValueError):
            raise RuntimeError("%s lacks numeric %s" % (method, field))
        if not math.isfinite(cycles) or cycles <= 0.0:
            raise RuntimeError("%s has invalid %s" % (method, field))
        tolerance = max(
            PRIMARY_T_ABS_TOL_CYCLES,
            PRIMARY_T_REL_TOL * max(abs(cycles), abs(incumbent)))
        if abs(cycles - incumbent) > tolerance:
            raise RuntimeError(
                "%s %s differs from the primary incumbent T" %
                (method, field))
        normalized[field] = cycles

    try:
        task_makespan = float(value.get("analytical_task_makespan_cycles"))
    except (TypeError, ValueError):
        raise RuntimeError(
            "%s lacks numeric feasibility-only analytical_task_makespan_cycles"
            % method)
    if not math.isfinite(task_makespan) or task_makespan <= 0.0:
        raise RuntimeError(
            "%s has invalid feasibility-only analytical task makespan" %
            method)
    try:
        task_alias = float(value.get("legacy_task_horizon_cycles"))
    except (TypeError, ValueError):
        raise RuntimeError(
            "%s lacks numeric legacy_task_horizon_cycles" % method)
    task_tolerance = max(
        PRIMARY_T_ABS_TOL_CYCLES,
        PRIMARY_T_REL_TOL * max(abs(task_makespan), abs(task_alias)))
    if (not math.isfinite(task_alias) or task_alias <= 0.0 or
            abs(task_alias - task_makespan) > task_tolerance):
        raise RuntimeError(
            "%s legacy task horizon differs from analytical T_task" % method)
    return {
        "primary_objective": expected_objective,
        "objective_contract": expected_contract,
        "incumbent_cycles": incumbent,
        "public_performance_cycles": normalized,
        "feasibility_task_makespan_cycles": task_makespan,
    }


def _no_post_t_objective_contract(value, method):
    """Require secondary, tertiary, and quaternary stages to be unrun."""
    for stage in ("secondary", "tertiary", "quaternary"):
        if (value.get(stage + "_objective") is not None or
                value.get(stage + "_objective_value") is not None or
                value.get(stage + "_has_incumbent") is not False or
                value.get(stage + "_status") is not None or
                value.get(stage + "_mip_gap") is not None):
            raise RuntimeError(
                "%s contains an active post-T objective (%s)" %
                (method, stage))
        try:
            wall = float(value.get(stage + "_wall_s"))
        except (TypeError, ValueError):
            raise RuntimeError(
                "%s lacks %s stage-not-run wall time" % (method, stage))
        if wall != 0.0:
            raise RuntimeError(
                "%s contains an active post-T objective (%s wall time)" %
                (method, stage))
        certificate = value.get(stage + "_gap_certificate")
        if (not isinstance(certificate, dict) or
                certificate.get("certified") is not False or
                certificate.get("reason") != "stage-not-run"):
            raise RuntimeError(
                "%s does not mark %s as stage-not-run" % (method, stage))
    for field in ("secondary_time_limit_s", "tertiary_time_limit_s"):
        if value.get(field) is not None:
            raise RuntimeError(
                "%s contains an active post-T time limit (%s)" %
                (method, field))


def _package_budget_contract(value, method):
    """Require serialized package area/power feasibility evidence."""
    budgets = value.get("budgets")
    if not isinstance(budgets, dict):
        raise RuntimeError("%s lacks package budgets" % method)
    if "p_col" in budgets:
        raise RuntimeError("%s unexpectedly serializes removed C9/p_col" %
                           method)
    normalized = {}
    for field in ("a_max", "p_max"):
        try:
            number = float(budgets.get(field))
        except (TypeError, ValueError):
            raise RuntimeError("%s lacks numeric budget %s" %
                               (method, field))
        if not math.isfinite(number) or number <= 0.0:
            raise RuntimeError("%s has invalid budget %s" % (method, field))
        normalized[field] = number
    for field, budget_field in (("total_area_mm2", "a_max"),
                                ("peak_power_W", "p_max")):
        try:
            used = float(value.get(field))
        except (TypeError, ValueError):
            raise RuntimeError("%s lacks numeric %s" % (method, field))
        if not math.isfinite(used) or used <= 0.0:
            raise RuntimeError("%s has invalid %s" % (method, field))
        if used > normalized[budget_field] + 1.0e-6:
            raise RuntimeError(
                "%s exceeds %s: %.9g > %.9g" %
                (method, budget_field, used, normalized[budget_field]))
        normalized[field] = used
    return normalized


def _first_feasible_contract(value, method):
    """Verify that reported first-feasible time uses the end-to-end clock.

    Stage-0 protocol placement refinement runs while the model is being built.
    Therefore first-feasible time must include the full model-build interval
    (and hence that refinement) before adding the first MIP incumbent time.
    """
    try:
        model_build = float(value.get("model_build_s"))
        formulation_build = float(value.get("formulation_build_s"))
        stage0_total = float(value.get("stage0_total_s"))
        first_feasible = float(value.get("time_to_first_feasible_s"))
        primary_wall = float(value.get("primary_wall_s"))
        requested_total = float(value.get("requested_total_time_limit_s"))
        primary_limit = float(value.get("primary_solver_time_limit_s"))
    except (TypeError, ValueError):
        raise RuntimeError(
            "%s lacks numeric first-feasible timing evidence" % method)
    timing_values = (model_build, formulation_build, stage0_total,
                     first_feasible, primary_wall, requested_total,
                     primary_limit)
    if (not all(math.isfinite(number) for number in timing_values) or
            min(timing_values) < 0.0):
        raise RuntimeError("%s has invalid first-feasible timing evidence" %
                           method)
    if abs(requested_total - ILP_TOTAL_BUDGET_S) > 1.0e-9:
        raise RuntimeError(
            "%s does not use the formal 45 s total ILP budget" % method)
    if model_build >= requested_total:
        raise RuntimeError(
            "%s exhausts the formal total ILP budget before Gurobi" % method)
    expected_primary_limit = requested_total - model_build
    allocation_tolerance = max(
        1.0e-3, 1.0e-9 * abs(expected_primary_limit))
    if abs(primary_limit - expected_primary_limit) > allocation_tolerance:
        raise RuntimeError(
            "%s primary solver limit does not subtract model build and "
            "Stage-0 refinement from the total budget" % method)
    build_tolerance = max(
        1.0e-3, 1.0e-9 * max(abs(model_build),
                              abs(formulation_build + stage0_total)))
    if abs(model_build - formulation_build - stage0_total) > build_tolerance:
        raise RuntimeError(
            "%s model_build_s does not include formulation and Stage-0 "
            "refinement" % method)

    stage0 = value.get("stage0_feasible_start")
    if not isinstance(stage0, dict):
        raise RuntimeError("%s lacks Stage-0 timing evidence" % method)
    refinement = stage0.get("protocol_placement_refinement")
    if not isinstance(refinement, dict):
        raise RuntimeError(
            "%s lacks Stage-0 protocol-refinement timing" % method)
    try:
        refinement_wall = float(refinement.get("wall_s"))
    except (TypeError, ValueError):
        raise RuntimeError(
            "%s lacks numeric Stage-0 protocol-refinement time" % method)
    if (not math.isfinite(refinement_wall) or refinement_wall < 0.0 or
            refinement_wall > stage0_total + 1.0e-6):
        raise RuntimeError(
            "%s Stage-0 time does not include protocol refinement" %
            method)

    trace = value.get("anytime_trace")
    if trace is None:
        trace = []
    if not isinstance(trace, list):
        raise RuntimeError("%s has invalid anytime_trace" % method)
    if trace:
        try:
            solver_first = float(trace[0]["wall_s"])
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(
                "%s has invalid first anytime incumbent" % method)
    else:
        solver_first = primary_wall
    if (not math.isfinite(solver_first) or solver_first < 0.0 or
            solver_first > primary_wall + 1.0e-6):
        raise RuntimeError("%s has invalid first incumbent runtime" % method)
    expected = model_build + solver_first
    tolerance = max(1.0e-6, 1.0e-9 * max(abs(expected),
                                        abs(first_feasible)))
    if abs(first_feasible - expected) > tolerance:
        raise RuntimeError(
            "%s time_to_first_feasible_s does not include model build and "
            "Stage-0 refinement" % method)
    return {
        "model_build_s": model_build,
        "formulation_build_s": formulation_build,
        "stage0_total_s": stage0_total,
        "stage0_protocol_refinement_s": refinement_wall,
        "primary_first_incumbent_s": solver_first,
        "time_to_first_feasible_s": first_feasible,
        "requested_total_time_limit_s": requested_total,
        "primary_solver_time_limit_s": primary_limit,
    }


def _validate_solution(path, method, case=None):
    _require_file(path, "%s solution" % method)
    value = _load_json(path, "%s solution" % method)
    if method == "ilp-perf":
        if value.get("optimization_mode") != "ilp-perf":
            raise RuntimeError("%s has optimization_mode=%r" %
                               (path, value.get("optimization_mode")))
        package = _package_budget_contract(value, str(path))
        if value.get("cpu_anchor_policy") != "free":
            raise RuntimeError("%s does not use free CPU placement" % path)
        cpu_count = sum(
            "CPU" in str(site.get("type", "")).upper()
            for site in value.get("sites") or [])
        if cpu_count != 1:
            raise RuntimeError("%s has %d CPU chiplets, expected exactly one" %
                               (path, cpu_count))
        if case is not None and case.get("performance_start_bom") is not None:
            expected_start_bom = {
                str(kind): int(count)
                for kind, count in case["performance_start_bom"].items()
                if int(count) != 0
            }
            start = value.get("stage0_feasible_start") or {}
            if (start.get("accepted_for_submission") is not True or
                    "performance-first" not in str(
                        start.get("selection_policy") or "")):
                raise RuntimeError(
                    "%s lacks an accepted performance-first MIP start" % path)
            start_counts = {
                str(kind): int(count)
                for kind, count in (start.get("type_counts") or {}).items()
                if int(count) != 0
            }
            if start_counts != expected_start_bom:
                raise RuntimeError(
                    "%s performance-first start BOM %r differs from %r" %
                    (path, start_counts, expected_start_bom))
            try:
                start_t = float(start["protocol_T_start_cycles"])
                incumbent_t = float(value["primary_incumbent_cycles"])
            except (KeyError, TypeError, ValueError):
                raise RuntimeError(
                    "%s lacks Stage-0 protocol-T/incumbent evidence" % path)
            if (not math.isfinite(start_t) or start_t <= 0.0 or
                    not math.isfinite(incumbent_t) or incumbent_t <= 0.0):
                raise RuntimeError(
                    "%s has invalid Stage-0 protocol-T/incumbent evidence" %
                    path)
            if incumbent_t > start_t + max(1.0e-6, 1.0e-9 * abs(start_t)):
                raise RuntimeError(
                    "%s min-T incumbent %.9g is worse than its submitted "
                    "performance-first protocol-T start %.9g" %
                    (path, incumbent_t, start_t))
            solver_configuration = value.get("solver_configuration") or {}
            if int(solver_configuration.get("gurobi_mip_focus", -1)) != 1:
                raise RuntimeError("%s does not use Gurobi MIPFocus=1" % path)
        slots = value.get("expert_slots") or {}
        if not slots or any(int(capacity) != 1
                            for capacity in slots.values()):
            raise RuntimeError("%s is not the q=1 LEGOSim main protocol" % path)
        routing = value.get("routing") or {}
        if routing.get("kind") not in (
                "source-aggregated-splittable-packet-flow",
                "phase-separated-source-aggregated-splittable-packet-flow"):
            raise RuntimeError("%s lacks the explicit Mesh flow model" % path)
        residual = float(routing.get("maximum_flow_residual_KiB", 1.0))
        utilization = float(routing.get("maximum_link_utilization", 2.0))
        if residual > 1.0e-4 or utilization > 1.0001:
            raise RuntimeError("%s has invalid flow residual/utilization" % path)
        schedule = value.get("task_schedule")
        if not isinstance(schedule, dict) or not schedule:
            raise RuntimeError("%s lacks the explicit task schedule" % path)
        task_site = value.get("task_site") or {}
        if set(str(task) for task in schedule) != set(
                str(task) for task in task_site):
            raise RuntimeError(
                "%s task schedule does not cover the placed tasks" % path)
        protocol = _protocol_surrogate_contract(value, method)
        if (protocol.get("role") != "sole-primary-objective" or
                protocol.get("is_optimization_objective") is not True):
            raise RuntimeError(
                "%s does not use protocol completion as min T" % path)
        noc_power = value.get("noc_power_proxy")
        if (not isinstance(noc_power, dict) or
                noc_power.get("diagnostic_only") is not True or
                noc_power.get("used_as_objective") is not False):
            raise RuntimeError(
                "%s does not keep the NoC-power proxy diagnostic-only" % path)
        _complete_bundle_contract(value, method)
        primary = _primary_min_t_contract(value, str(path))
        _joint_min_t_output_contract(
            value, str(path), primary["incumbent_cycles"])
        _no_post_t_objective_contract(value, str(path))
        _first_feasible_contract(value, str(path))
        comparison = value.get("comparison_reference") or {}
        refinement = value.get("internal_bundle_refinement") or {}
        if (comparison.get("fixed_bom") is not None or
                comparison.get("kind") is not None or
                comparison.get("fixed_complete_bundles") or
                refinement.get("enabled") is not False or
                refinement.get("coordinates_fixed") is not False or
                value.get("reference_coordinates_warm_started") or
                value.get("performance_cap_enforced") or
                value.get("performance_reference_cycles") is not None or
                value.get("performance_relaxation") is not None or
                value.get("performance_cap_cycles") is not None):
            raise RuntimeError("%s is not a fresh uncapped min-T solve" % path)
        gpu_count_bounds = []
        recorded_budgets = value.get("budgets") or {}
        for bound_name in ("nmin", "nmax"):
            bounds = recorded_budgets.get(bound_name) or {}
            gpu_count_bounds.extend(
                "%s.%s" % (bound_name, kind) for kind in bounds
                if str(kind).upper().startswith("GPU"))
        if gpu_count_bounds:
            raise RuntimeError(
                "%s fixes/bounds GPU counts outside Amax/Pmax: %s" %
                (path, ", ".join(sorted(gpu_count_bounds))))
        selected_counts = value.get("selected_type_counts")
        selection = value.get("resource_selection") or {}
        actual_counts = {}
        for site in value.get("sites") or []:
            kind = str(site.get("type"))
            actual_counts[kind] = actual_counts.get(kind, 0) + 1
        normalized_selected_counts = (
            {str(kind): int(count) for kind, count in selected_counts.items()
             if int(count) != 0}
            if isinstance(selected_counts, dict) else None)
        if (not isinstance(selected_counts, dict) or
                normalized_selected_counts != actual_counts or
                selection.get("count_definition") != "N_c=sum_p x[c,p]" or
                selection.get("joint_with_mapping_and_placement") is not True or
                selection.get("fixed_bom") is not False or
                selection.get("selected_counts") != selected_counts):
            raise RuntimeError(
                "%s lacks auditable joint GPU-count selection evidence" % path)
    elif method == "matched-random":
        baseline = value.get("baseline") or {}
        if (not value.get("feasible") or
                not baseline.get("strict_evaluate")):
            raise RuntimeError("%s matched baseline is not strictly feasible" %
                               path)
        scoring = baseline.get("public_T_scoring") or {}
        if (baseline.get("representative_selection_metric") !=
                "public_T_cycles" or
                scoring.get("metric") != "public_T_cycles" or
                scoring.get("active_objective") != "min T" or
                scoring.get("protocol_completion_in_T") is not True or
                scoring.get("directed_mesh_capacity_in_T") is not True or
                scoring.get("legacy_task_makespan_role") !=
                "hard-feasibility-only" or
                scoring.get("legacy_task_makespan_scored") is not False or
                scoring.get("energy_or_byte_hop_tie_break") is not False):
            raise RuntimeError(
                "%s matched baseline was not selected by the public min-T "
                "metric" % path)
        try:
            public_t = float(value["public_T_cycles"])
            protocol_t = float(value["protocol_completion_cycles"])
            capacity_t = float(value[
                "directed_mesh_capacity_bound_cycles"])
            predicted_t = float(value["predicted_makespan_cycles"])
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("%s lacks the matched public-T decomposition" %
                               path)
        expected_t = max(protocol_t, capacity_t)
        tolerance = max(1.0e-6, 1.0e-9 * max(
            abs(public_t), abs(expected_t), abs(predicted_t)))
        if (not all(math.isfinite(number) and number > 0.0
                    for number in (public_t, protocol_t, capacity_t,
                                   predicted_t)) or
                abs(public_t - expected_t) > tolerance or
                abs(predicted_t - public_t) > tolerance):
            raise RuntimeError("%s has an inconsistent matched public T" % path)

        summary_path = Path(path).parent / "summary.json"
        _require_file(summary_path, "matched-random selection summary")
        summary = _load_json(summary_path, "matched-random selection summary")
        runs = summary.get("runs") or []
        seeds = summary.get("seeds") or []
        if (summary.get("schema_version") != 2 or
                summary.get("representative_selection_metric") !=
                "public_T_cycles" or
                summary.get("all_candidates_scored_with_public_T") is not
                True or
                int(summary.get("candidate_count", -1)) != 100 or
                len(runs) != 100 or len(seeds) != 100 or
                sorted(int(seed) for seed in seeds) != list(range(100)) or
                len({int(run.get("seed", -1)) for run in runs}) != 100 or
                any(not math.isfinite(float(run.get(
                    "public_T_cycles", float("nan")))) for run in runs)):
            raise RuntimeError(
                "%s matched summary is not the 100-candidate public-T "
                "protocol" % summary_path)
        try:
            representative_seed = int(summary["representative_seed"])
            representative_t = float(
                summary["representative_public_T_cycles"])
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(
                "%s lacks the matched representative binding" % summary_path)
        if (representative_seed != int(value.get("seed", -1)) or
                abs(representative_t - public_t) > tolerance):
            raise RuntimeError(
                "%s does not bind summary.json to representative.json" % path)
    else:
        strict = value.get("strict_evaluate") or {}
        if not strict.get("feasible"):
            raise RuntimeError("%s baseline is not strictly feasible" % path)
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("%s lacks optimizer budget metadata" % path)
        if method == "floorplet":
            try:
                recorded_budget = float(metadata.get("time_limit_s"))
            except (TypeError, ValueError):
                raise RuntimeError(
                    "%s lacks the Floorplet time budget" % path)
            if abs(recorded_budget - FLOORPLET_BUDGET_S) > 1.0e-9:
                raise RuntimeError(
                    "%s is not a formal 45 s Floorplet run" % path)
        elif method == "tap":
            try:
                recorded_budget = float(metadata.get("wall_budget_s"))
                recorded_seed = int(metadata.get("seed"))
            except (TypeError, ValueError):
                raise RuntimeError("%s lacks TAP budget/seed metadata" % path)
            if (abs(recorded_budget - TAP_BUDGET_S_PER_SEED) > 1.0e-9 or
                    recorded_seed not in TAP_SEEDS):
                raise RuntimeError(
                    "%s is not a formal 15 s TAP seed run" % path)
    if case is not None:
        expected_grid = {"L": case["grid"][0], "X": case["grid"][1],
                         "Y": case["grid"][2]}
        if value.get("grid") != expected_grid:
            raise RuntimeError("%s grid does not match %s" %
                               (path, case["key"]))
        budgets = value.get("budgets") or {}
        for name in ("a_max", "p_max"):
            if abs(float(budgets.get(name, -1.0)) -
                   float(case["budgets"][name])) > 1.0e-6:
                raise RuntimeError("%s %s does not match %s" %
                                   (path, name, case["key"]))
    return value


def _bom_signature(solution):
    return sorted(str(site.get("type"))
                  for site in solution.get("sites") or [])


def _bundle_signature(solution):
    return sorted((str(site.get("type")),
                   tuple(sorted(str(task)
                                for task in site.get("tasks") or [])))
                  for site in solution.get("sites") or [])


def _validate_reference_matching(case, out_root, methods):
    perf = _validate_solution(
        _solution_path(out_root, case, "ilp-perf"), "ilp-perf", case)
    perf_bom = _bom_signature(perf)
    perf_bundles = _bundle_signature(perf)
    for method in methods:
        if method == "ilp-perf":
            continue
        value = _validate_solution(
            _solution_path(out_root, case, method), method, case)
        if _bom_signature(value) != perf_bom:
            raise RuntimeError("%s changed the ILP-Perf BOM for %s" %
                               (method, case["key"]))
        if method in ("floorplet", "tap"):
            if _bundle_signature(value) != perf_bundles:
                raise RuntimeError("%s changed an ILP-Perf task bundle for %s" %
                                   (method, case["key"]))


def _run_ilp(case, method, out_root, gurobi_python, manifest, resume):
    final_path = _solution_path(out_root, case, method)
    row = manifest["cases_by_key"][case["key"]][method.replace("-", "_")]
    if final_path.exists():
        if not resume:
            raise RuntimeError("refusing to overwrite solution: %s" % final_path)
        _validate_solution(final_path, method, case)
        row.update({"status": "complete", "solution_sha256":
                    _sha256(final_path), "reused": True})
        return
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(final_path.name + ".tmp.%d" % os.getpid())
    command = _ilp_command(case, method, temporary, gurobi_python, out_root)
    recorded = _ilp_command(case, method, final_path, gurobi_python, out_root)
    row["command"] = recorded
    log_base = out_root / "logs" / case["suite"] / (
        case["case_id"] + "_" + method.replace("-", "_"))
    returncode = _run_command(
        command, Path(str(log_base) + ".stdout.log"),
        Path(str(log_base) + ".stderr.log"))
    if returncode != 0:
        if temporary.exists():
            temporary.unlink()
        row.update({"status": "failed", "returncode": returncode})
        raise RuntimeError("%s failed for %s; see %s" %
                           (method, case["key"], log_base))
    _validate_solution(temporary, method, case)
    os.replace(str(temporary), str(final_path))
    row.update({"status": "complete", "returncode": returncode,
                "solution_sha256": _sha256(final_path), "reused": False})


def _run_baseline_command(case, label, command, output_paths, out_root,
                          manifest, resume):
    row = manifest["cases_by_key"][case["key"]][label]
    existing = [path.exists() for path in output_paths]
    if any(existing):
        if not all(existing) or not resume:
            raise RuntimeError("partial/existing %s output for %s" %
                               (label, case["key"]))
        for path, method in output_paths.items():
            _validate_solution(path, method, case)
        row.update({"status": "complete", "reused": True,
                    "solution_sha256": {
                        method: _sha256(path)
                        for path, method in output_paths.items()}})
        return
    final_dirs = {path.parent for path in output_paths}
    if len(final_dirs) != 1:
        raise RuntimeError("%s outputs do not share one directory" % label)
    final_dir = next(iter(final_dirs))
    if final_dir.exists():
        raise RuntimeError("partial %s output directory: %s" %
                           (label, final_dir))
    temporary_dir = final_dir.with_name(
        final_dir.name + ".tmp.%d" % os.getpid())
    if temporary_dir.exists():
        raise RuntimeError("temporary baseline directory exists: %s" %
                           temporary_dir)
    execution_command = list(command)
    try:
        out_index = execution_command.index("--out-dir") + 1
    except ValueError:
        raise RuntimeError("%s command lacks --out-dir" % label)
    execution_command[out_index] = str(temporary_dir)
    row["command"] = command
    log_base = out_root / "logs" / case["suite"] / (
        case["case_id"] + "_" + label)
    returncode = _run_command(
        execution_command, Path(str(log_base) + ".stdout.log"),
        Path(str(log_base) + ".stderr.log"))
    if returncode != 0:
        row.update({"status": "failed", "returncode": returncode})
        raise RuntimeError("%s failed for %s; see %s" %
                           (label, case["key"], log_base))
    temporary_outputs = {
        temporary_dir / path.name: method
        for path, method in output_paths.items()}
    for path, method in temporary_outputs.items():
        _validate_solution(path, method, case)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(temporary_dir), str(final_dir))
    for path, method in output_paths.items():
        _validate_solution(path, method, case)
    row.update({"status": "complete", "returncode": returncode,
                "reused": False, "solution_sha256": {
                    method: _sha256(path)
                    for path, method in output_paths.items()}})


def _manifest_case(case, out_root, gurobi_python, system_python):
    result = {
        "key": case["key"], "suite": case["suite"],
        "case_id": case["case_id"],
        "grid": {"L": case["grid"][0], "X": case["grid"][1],
                 "Y": case["grid"][2]},
        "budgets": case["budgets"],
        "params_path": str(case["params"]),
        "params_sha256": _sha256(case["params"]),
        "source_metadata_path": str(case["source_metadata"]),
        "source_metadata_sha256": _sha256(case["source_metadata"]),
        "formal_wall_budgets_s": dict(FORMAL_WALL_BUDGETS_S_PER_CASE),
    }
    if case["suite"] == "scaling":
        result.update({
            "scaling_budget_contract_version": case[
                "scaling_budget_contract_version"],
            "budget_policy": case["budget_policy"],
            "budget_target_bom": case["budget_target_bom"],
        })
    result["ilp_perf"] = {
        "status": "pending",
        "solution_path": str(_solution_path(out_root, case, "ilp-perf")),
        "command": _ilp_command(
            case, "ilp-perf", _solution_path(out_root, case, "ilp-perf"),
            gurobi_python, out_root),
    }
    if case["suite"] == "budget":
        result["matched_random_seed_count"] = len(
            PROTOCOL["matched_random_seeds"])
        result["matched_random"] = {
            "status": "pending",
            "solution_path": str(_solution_path(
                out_root, case, "matched-random")),
            "command": _matched_command(case, out_root, system_python),
        }
    result["prior"] = {
        "status": "pending",
        "floorplet_solution_path": str(_solution_path(
            out_root, case, "floorplet")),
        "tap_solution_path": str(_solution_path(out_root, case, "tap")),
        "command": _prior_command(case, out_root, gurobi_python),
    }
    return result


def _new_manifest(cases, out_root, gurobi_python, system_python):
    rows = [_manifest_case(case, out_root, gurobi_python, system_python)
            for case in cases]
    return {
        "schema_version": 5,
        "status": "planned",
        "project_root": str(PROJECT_ROOT),
        "out_root": str(out_root),
        "protocol": PROTOCOL,
        "interpreters": {
            "gurobi_python": str(gurobi_python),
            "system_python": str(system_python),
        },
        "drivers": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (
                ("ilp_layout.py", PROJECT_ROOT / "src" / "ilp_layout.py"),
                ("matched_layout_baseline.py", PROJECT_ROOT / "src" /
                 "matched_layout_baseline.py"),
                ("run_metaheuristic_surrogate_audit.py", PROJECT_ROOT / "src" /
                 "run_metaheuristic_surrogate_audit.py"),
                ("run_optimized_prior_proxies.py", PROJECT_ROOT / "src" /
                 "run_optimized_prior_proxies.py"),
                ("generate_revalidation_baselines.py", PROJECT_ROOT / "src" /
                 "generate_revalidation_baselines.py"),
                ("replay_legosim_solutions.py", PROJECT_ROOT / "src" /
                 "replay_legosim_solutions.py"),
                ("orchestrator", Path(__file__).resolve()),
            )},
        "selected_case_count": len(cases),
        "expected_ilp_solutions": len(cases),
        "expected_replay_rows": sum(len(_methods(case)) for case in cases),
        "formal_wall_budgets_s_per_case": dict(
            FORMAL_WALL_BUDGETS_S_PER_CASE),
        "matched_random_seed_count": len(
            PROTOCOL["matched_random_seeds"]),
        "cases": rows,
        # Runtime-only convenience map; removed before serialization.
        "cases_by_key": {row["key"]: row for row in rows},
        "replay_invocations": [],
    }


def _serializable_manifest(manifest):
    value = dict(manifest)
    value.pop("cases_by_key", None)
    return value


def _save_manifest(out_root, manifest):
    _atomic_json(out_root / "manifest.json", _serializable_manifest(manifest))


def _load_manifest(out_root, cases):
    manifest = _load_json(out_root / "manifest.json", "run manifest")
    if manifest.get("schema_version") != 5:
        raise RuntimeError("existing manifest schema does not match")
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("existing manifest protocol does not match")
    if manifest.get("formal_wall_budgets_s_per_case") != \
            FORMAL_WALL_BUDGETS_S_PER_CASE:
        raise RuntimeError("existing manifest formal wall budgets do not match")
    if manifest.get("matched_random_seed_count") != len(
            PROTOCOL["matched_random_seeds"]):
        raise RuntimeError(
            "existing manifest matched-random seed count does not match")
    expected = [case["key"] for case in cases]
    actual = [row.get("key") for row in manifest.get("cases") or []]
    if expected != actual:
        raise RuntimeError("existing manifest selected-case list does not match")
    expected_rows = sum(len(_methods(case)) for case in cases)
    if (manifest.get("expected_ilp_solutions") != len(cases) or
            manifest.get("expected_replay_rows") != expected_rows):
        raise RuntimeError("existing manifest expected counts do not match")
    rows = {row["key"]: row for row in manifest["cases"]}
    for case in cases:
        row = rows[case["key"]]
        if row.get("budgets") != case["budgets"]:
            raise RuntimeError("package budgets changed for %s" % case["key"])
        if case["suite"] == "scaling":
            for field in ("scaling_budget_contract_version",
                          "budget_policy", "budget_target_bom"):
                if row.get(field) != case.get(field):
                    raise RuntimeError(
                        "%s changed for %s" % (field, case["key"]))
        if row.get("formal_wall_budgets_s") != \
                FORMAL_WALL_BUDGETS_S_PER_CASE:
            raise RuntimeError(
                "formal wall budgets changed for %s" % case["key"])
        if (case["suite"] == "budget" and
                row.get("matched_random_seed_count") != len(
                    PROTOCOL["matched_random_seeds"])):
            raise RuntimeError(
                "matched-random seed count changed for %s" % case["key"])
        for label, path in (("params", case["params"]),
                            ("source_metadata", case["source_metadata"])):
            if row.get(label + "_path") != str(path):
                raise RuntimeError("%s path changed for %s" %
                                   (label, case["key"]))
            if row.get(label + "_sha256") != _sha256(path):
                raise RuntimeError("%s content changed for %s" %
                                   (label, case["key"]))
    for record in (manifest.get("drivers") or {}).values():
        path = Path(str(record.get("path")))
        _require_file(path, "recorded protocol driver")
        if record.get("sha256") != _sha256(path):
            raise RuntimeError("protocol driver changed since run start: %s" %
                               path)
    manifest["cases_by_key"] = {
        row["key"]: row for row in manifest["cases"]}
    return manifest


def _solve(cases, out_root, gurobi_python, manifest, resume):
    manifest["status"] = "solving"
    _save_manifest(out_root, manifest)
    for case in cases:
        _run_ilp(case, "ilp-perf", out_root, gurobi_python,
                 manifest, resume)
        _save_manifest(out_root, manifest)
    manifest["status"] = "solve_complete"
    _save_manifest(out_root, manifest)


def _baselines(cases, out_root, gurobi_python, system_python, manifest,
               resume):
    manifest["status"] = "generating_baselines"
    _save_manifest(out_root, manifest)
    for case in cases:
        _validate_solution(_solution_path(out_root, case, "ilp-perf"),
                           "ilp-perf", case)
        if case["suite"] == "budget":
            matched_path = _solution_path(out_root, case, "matched-random")
            _run_baseline_command(
                case, "matched_random",
                _matched_command(case, out_root, system_python),
                {matched_path: "matched-random"}, out_root, manifest, resume)
            _save_manifest(out_root, manifest)
        floor = _solution_path(out_root, case, "floorplet")
        tap = _solution_path(out_root, case, "tap")
        _run_baseline_command(
            case, "prior", _prior_command(case, out_root, gurobi_python),
            {floor: "floorplet", tap: "tap"}, out_root, manifest, resume)
        _validate_reference_matching(
            case, out_root,
            ("ilp-perf", "matched-random", "floorplet", "tap")
            if case["suite"] == "budget" else
            ("ilp-perf", "floorplet", "tap"))
        _save_manifest(out_root, manifest)
    manifest["status"] = "baselines_complete"
    _save_manifest(out_root, manifest)


def _replay_plan(cases, out_root, system_python, args):
    invocations = []
    for name, group in _replay_groups(cases):
        if name == "budget":
            replay_dir = out_root / "replays" / "budget"
        else:
            replay_dir = out_root / "replays" / "scaling" / group[0]["case_id"]
        log_base = out_root / "logs" / ("replay_" + name)
        invocations.append({
            "name": name,
            "cases": [case["key"] for case in group],
            "expected_rows": sum(len(_methods(case)) for case in group),
            "expected_layout_keys": [
                [case["case_id"], method]
                for case in group for method in _methods(case)],
            # A case/method key alone is not enough to resume a replay: the
            # solution file may have been regenerated in place.  Keep the
            # canonical inputs in the plan so validation can bind every
            # measured row to the exact current solution bytes.
            "expected_solutions": [{
                "case_id": case["case_id"],
                "method": method,
                "solution_path": str(_solution_path(
                    out_root, case, method).resolve()),
            } for case in group for method in _methods(case)],
            "expected_params_path": str(Path(
                group[0]["params"]).resolve()),
            "command": _replay_command(
                group, out_root, replay_dir, system_python, args),
            "out_dir": str(replay_dir),
            "results_json": str(replay_dir / "results.json"),
            "estimated_cost_rank": REPLAY_COST_RANK[name],
            "stdout_log": str(Path(str(log_base) + ".stdout.log")),
            "stderr_log": str(Path(str(log_base) + ".stderr.log")),
            "status": "planned",
        })
    return invocations


def _execute_replay(item):
    result = dict(item)
    returncode = _run_command(
        result["command"], Path(result["stdout_log"]),
        Path(result["stderr_log"]))
    result["returncode"] = returncode
    result["status"] = "complete" if returncode == 0 else "failed"
    return result


def _run_replays(invocations, jobs):
    ordered = sorted(invocations,
                     key=lambda item: (-item["estimated_cost_rank"],
                                       item["name"]))
    if jobs == 1:
        return [_execute_replay(item) for item in ordered]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(_execute_replay, ordered))


def _validate_replay_result(item):
    results = Path(item["results_json"])
    value = _load_json(results, "%s replay results" % item["name"])
    if value.get("status") != "complete":
        raise RuntimeError("%s replay status is not complete" % item["name"])
    layouts = value.get("layouts")
    if not isinstance(layouts, list):
        raise RuntimeError("%s replay has no layouts list" % item["name"])
    if len(layouts) != int(item["expected_rows"]):
        raise RuntimeError("%s replay has %d rows, expected %d" %
                           (item["name"], len(layouts),
                            int(item["expected_rows"])))
    actual = sorted([str(row.get("case_id")), str(row.get("method"))]
                    for row in layouts)
    expected = sorted(item["expected_layout_keys"])
    if actual != expected:
        raise RuntimeError("%s replay case/method keys do not match plan" %
                           item["name"])
    failed = [row for row in layouts if row.get("status") != "ok"]
    if failed:
        raise RuntimeError("%s replay contains %d non-ok rows" %
                           (item["name"], len(failed)))

    expected_params = Path(item["expected_params_path"]).resolve()
    actual_params_raw = value.get("params_path")
    if not actual_params_raw:
        raise RuntimeError("%s replay lacks params_path" % item["name"])
    actual_params = Path(str(actual_params_raw)).resolve()
    if actual_params != expected_params:
        raise RuntimeError(
            "%s replay params path does not match the current plan" %
            item["name"])
    if not expected_params.is_file():
        raise RuntimeError("%s current params file is missing: %s" %
                           (item["name"], expected_params))
    params_digest = str(value.get("params_sha256") or "").lower()
    if params_digest != _sha256(expected_params):
        raise RuntimeError(
            "%s replay params SHA-256 does not match the current file" %
            item["name"])

    rows = {(str(row.get("case_id")), str(row.get("method"))): row
            for row in layouts}
    for expected_solution in item["expected_solutions"]:
        key = (str(expected_solution["case_id"]),
               str(expected_solution["method"]))
        row = rows[key]
        current_path = Path(
            expected_solution["solution_path"]).resolve()
        replay_path_raw = row.get("solution_path")
        if not replay_path_raw:
            raise RuntimeError(
                "%s replay row %s:%s lacks solution_path" %
                (item["name"], key[0], key[1]))
        replay_path = Path(str(replay_path_raw)).resolve()
        if replay_path != current_path:
            raise RuntimeError(
                "%s replay row %s:%s solution path does not match the "
                "current plan" % (item["name"], key[0], key[1]))
        if not current_path.is_file():
            raise RuntimeError(
                "%s current solution is missing for %s:%s: %s" %
                (item["name"], key[0], key[1], current_path))
        recorded_digest = str(row.get("solution_sha256") or "").lower()
        if recorded_digest != _sha256(current_path):
            raise RuntimeError(
                "%s replay row %s:%s SHA-256 does not match the current "
                "solution" % (item["name"], key[0], key[1]))
    return value


def _verify_all_layouts(cases, out_root):
    for case in cases:
        _validate_reference_matching(case, out_root, _methods(case))


def _archive_invalid_replay(path):
    """Move a partial replay aside so --resume is recoverable and auditable."""
    for index in range(1, 10000):
        archive = path.with_name(path.name + ".invalid.%04d" % index)
        if not archive.exists():
            os.replace(str(path), str(archive))
            return archive
    raise RuntimeError("too many archived replay attempts beside %s" % path)


def _replay(cases, out_root, system_python, args, manifest, resume):
    # Re-evaluate current solution and baseline bytes before replay; a stale
    # manifest status is never sufficient.
    _verify_all_layouts(cases, out_root)
    plan = _replay_plan(cases, out_root, system_python, args)
    pending = []
    completed = []
    for item in plan:
        replay_dir = Path(item["out_dir"])
        results = Path(item["results_json"])
        if replay_dir.exists():
            if not resume:
                raise RuntimeError("existing replay output: %s" % replay_dir)
            try:
                if not results.is_file():
                    raise RuntimeError("results.json is missing")
                _validate_replay_result(item)
            except (RuntimeError, ValueError) as exc:
                archive = _archive_invalid_replay(replay_dir)
                item["archived_invalid_output"] = str(archive)
                item["archived_invalid_reason"] = str(exc)
                pending.append(item)
            else:
                item["status"] = "complete"
                item["reused"] = True
                item["results_sha256"] = _sha256(results)
                completed.append(item)
        else:
            pending.append(item)
    manifest["status"] = "replaying"
    manifest["replay_invocations"] = completed + pending
    _save_manifest(out_root, manifest)
    executed = _run_replays(pending, args.replay_jobs)
    for item in executed:
        results = Path(item["results_json"])
        if item["status"] == "complete" and results.is_file():
            try:
                _validate_replay_result(item)
                item["results_sha256"] = _sha256(results)
                item["reused"] = False
            except (RuntimeError, ValueError) as exc:
                item["status"] = "failed"
                item["error"] = str(exc)
        elif item["status"] == "complete":
            item["status"] = "failed"
            item["error"] = "replay returned zero but results.json is missing"
    all_results = completed + executed
    manifest["replay_invocations"] = all_results
    failed = [item for item in all_results if item["status"] != "complete"]
    manifest["status"] = "replay_failed" if failed else "complete"
    _save_manifest(out_root, manifest)
    if failed:
        raise RuntimeError("%d replay invocation(s) failed: %s" % (
            len(failed), ", ".join(item["name"] for item in failed)))


def _validate_args(args):
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.budget_e2e_timeout <= 0 or args.scale_e2e_timeout <= 0:
        raise ValueError("replay timeouts must be positive")
    if args.budget_legosim_timeout < args.budget_e2e_timeout + 600:
        raise ValueError("--budget-legosim-timeout must be at least "
                         "--budget-e2e-timeout + 600 seconds")
    if args.scale_legosim_timeout < args.scale_e2e_timeout + 600:
        raise ValueError("--scale-legosim-timeout must be at least "
                         "--scale-e2e-timeout + 600 seconds")
    if args.moe_gpgpusim_max_parallel <= 0:
        raise ValueError("--moe-gpgpusim-max-parallel must be positive")


def _dry_run(cases, out_root, gurobi_python, system_python, args):
    solves = []
    baselines = []
    for case in cases:
        solves.append(_ilp_command(
            case, "ilp-perf", _solution_path(
                out_root, case, "ilp-perf"), gurobi_python, out_root))
        if case["suite"] == "budget":
            baselines.append(_matched_command(case, out_root, system_python))
        baselines.append(_prior_command(case, out_root, gurobi_python))
    replays = _replay_plan(cases, out_root, system_python, args)
    print(json.dumps({
        "dry_run": True,
        "phase": args.phase,
        "out_root": str(out_root),
        "selected_case_count": len(cases),
        "expected_ilp_solutions": len(cases),
        "expected_replay_rows": sum(len(_methods(case)) for case in cases),
        "protocol": PROTOCOL,
        "solve_commands": solves,
        "baseline_commands": baselines,
        "replay_invocations": replays,
    }, indent=2, sort_keys=True))


def main(argv=None):
    args = _parse_args(argv)
    _validate_args(args)
    out_root = _resolve(args.out_root)
    gurobi_python = _resolve(args.gurobi_python)
    system_python = _resolve(args.system_python)
    for path, label in (
            (gurobi_python, "Gurobi Python"),
            (system_python, "system Python"),
            (PROJECT_ROOT / "src" / "ilp_layout.py", "ilp_layout.py"),
            (PROJECT_ROOT / "src" / "matched_layout_baseline.py",
             "matched baseline"),
            (PROJECT_ROOT / "src" / "generate_revalidation_baselines.py",
             "prior baseline wrapper"),
            (PROJECT_ROOT / "src" / "replay_legosim_solutions.py",
             "replay driver")):
        _require_file(path, label)

    all_cases = (_budget_cases(_resolve(args.budget_source)) +
                 _scale_cases(_resolve(args.scale_source),
                              _resolve(args.scale_params)))
    if len(all_cases) != 13:
        raise ValueError("canonical plan must contain 13 cases")
    cases = _select_cases(all_cases, args.case)

    if args.dry_run:
        _dry_run(cases, out_root, gurobi_python, system_python, args)
        return 0

    needs_new_root = args.phase in ("solve-only", "all") and not args.resume
    if needs_new_root:
        if out_root.exists():
            raise RuntimeError("refusing to overwrite --out-root: %s" % out_root)
        out_root.mkdir(parents=True)
        manifest = _new_manifest(cases, out_root, gurobi_python, system_python)
        _save_manifest(out_root, manifest)
    else:
        if not out_root.is_dir():
            raise RuntimeError("phase/resume requires existing out root: %s" %
                               out_root)
        manifest = _load_manifest(out_root, cases)

    if args.phase in ("solve-only", "all"):
        _solve(cases, out_root, gurobi_python, manifest, args.resume)
    if args.phase in ("baselines-only", "all"):
        _baselines(cases, out_root, gurobi_python, system_python, manifest,
                   args.resume)
    if args.phase in ("replay-only", "all"):
        _replay(cases, out_root, system_python, args, manifest, args.resume)

    print(json.dumps({
        "status": manifest["status"],
        "phase": args.phase,
        "selected_cases": len(cases),
        "expected_replay_rows": sum(len(_methods(case)) for case in cases),
        "out_root": str(out_root),
        "manifest": str(out_root / "manifest.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        sys.exit(2)
