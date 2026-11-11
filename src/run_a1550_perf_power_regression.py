#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict two-mode A1550 solve/replay regression gate.

This is intentionally a narrow smoke test, not another paper experiment
driver.  It reuses ``run_full_constraint_revalidation.py`` for the canonical
fresh A1550 ILP-Perf -> ILP-Power solve dependency, then invokes
``replay_legosim_solutions.py`` with exactly those two solutions.  No random,
Floorplet, or TAP baseline is generated or replayed.

The final audit records both solver-side and measured metrics.  A publishable
two-mode pair must show the intended measured ordering: ILP-Perf has fewer
LEGOSim cycles, while ILP-Power has lower NoC energy or average NoC power.  It
also rejects the stronger failure where ILP-Power dominates ILP-Perf.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import run_full_constraint_revalidation as full_validation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_DRIVER = PROJECT_ROOT / "src" / "run_full_constraint_revalidation.py"
REPLAY_DRIVER = PROJECT_ROOT / "src" / "replay_legosim_solutions.py"
DEFAULT_OUT_ROOT = "out/a1550_perf_power_regression"
DEFAULT_BUDGET_SOURCE = (
    "out/corrected_e2e_runs/corrected_e2e_20260731_v2/raw_results.json")
DEFAULT_SCALE_SOURCE = "out/scaling_ilp_secondary_20260801"
DEFAULT_SCALE_PARAMS = "out/scaling_qap_experiment/params"
DEFAULT_GUROBI_PYTHON = os.environ.get("GUROBI_PYTHON", sys.executable)
CASE_ID = "area_1550"
CASE_KEY = "budget:" + CASE_ID
METHODS = ("ilp-perf", "ilp-power")


class RegressionFailure(RuntimeError):
    """A completed run violated the strict regression contract."""


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Solve and replay only A1550 ILP-Perf/ILP-Power, then reject a "
            "measured dominated performance point"))
    parser.add_argument(
        "--phase", choices=("all", "solve-only", "replay-only",
                            "audit-only"), default="all")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--budget-source", default=DEFAULT_BUDGET_SOURCE)
    parser.add_argument("--scale-source", default=DEFAULT_SCALE_SOURCE)
    parser.add_argument("--scale-params", default=DEFAULT_SCALE_PARAMS)
    parser.add_argument("--gurobi-python", default=DEFAULT_GUROBI_PYTHON)
    parser.add_argument("--system-python", default=sys.executable)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--moe-e2e-timeout", type=int, default=1800)
    parser.add_argument("--legosim-timeout", type=int, default=2400)
    parser.add_argument("--moe-gpgpusim-max-parallel", type=int, default=16)
    parser.add_argument(
        "--replay-results", default=None,
        help=("results.json to audit; default is "
              "<out-root>/replays/a1550_perf_power/results.json"))
    parser.add_argument(
        "--report", default=None,
        help=("audit JSON path; default is "
              "<out-root>/a1550_perf_power_regression.json"))
    parser.add_argument(
        "--max-primary-gap", type=float, default=None,
        help=("optionally fail when either Stage-0 analytical-T MIP gap "
              "exceeds this value"))
    parser.add_argument(
        "--max-protocol-gap", type=float, default=None,
        help=("optionally fail when either protocol event-DAG stage MIP gap "
              "exceeds this value"))
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse the verified solve root and/or completed two-layout replay")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the exact two-solve/two-replay plan without writing files")
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
        raise RegressionFailure("cannot read %s %s: %s" %
                                (label, path, exc))
    if not isinstance(value, dict):
        raise RegressionFailure("%s must be a JSON object: %s" %
                                (label, path))
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with open(str(temporary), "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise RegressionFailure("missing/non-numeric %s: %r" %
                                (label, value))
    if not math.isfinite(number):
        raise RegressionFailure("non-finite %s: %r" % (label, value))
    return number


def _same_number(actual, expected, label, tolerance=1.0e-9):
    actual_number = _finite_float(actual, label)
    expected_number = _finite_float(expected, "expected " + label)
    scale = max(1.0, abs(actual_number), abs(expected_number))
    if abs(actual_number - expected_number) > tolerance * scale:
        raise RegressionFailure(
            "%s mismatch: recorded %.12g, recomputed %.12g" %
            (label, actual_number, expected_number))
    return actual_number


def _recorded_path(raw, label):
    if not raw:
        raise RegressionFailure("missing %s" % label)
    return Path(str(raw)).resolve()


def _verify_replay_identity(results, results_path, rows, paths):
    """Bind replay evidence to the current parameter and solution bytes."""
    params_path = _recorded_path(
        results.get("params_path"), "replay params_path")
    if not params_path.is_file():
        raise RegressionFailure("replay params file is missing: %s" %
                                params_path)
    params_digest = str(results.get("params_sha256") or "").lower()
    current_params_digest = _sha256(params_path)
    if params_digest != current_params_digest:
        raise RegressionFailure(
            "replay params SHA-256 does not match the current file")

    identities = {}
    for method in METHODS:
        row = rows[method]
        replay_path = _recorded_path(
            row.get("solution_path"),
            "%s replay solution_path" % method)
        if replay_path != paths[method]:
            raise RegressionFailure(
                "%s replay solution path does not match the audited "
                "solution" % method)
        current_digest = _sha256(paths[method])
        replay_digest = str(row.get("solution_sha256") or "").lower()
        if replay_digest != current_digest:
            raise RegressionFailure(
                "%s replay SHA-256 does not match the current solution" %
                method)
        identities[method] = {
            "solution_path": str(paths[method]),
            "solution_sha256": current_digest,
            "replay_solution_path": str(replay_path),
            "replay_solution_sha256": replay_digest,
        }
    return {
        "params_path": str(params_path),
        "params_sha256": current_params_digest,
        "results_path": str(Path(results_path).resolve()),
        "solutions": identities,
    }, _load_json(params_path, "replay params")


def _hardware_metrics(solution, params, row, method):
    """Recompute the shared chiplet BOM; do not call it measured power."""
    library = params.get("library")
    if not isinstance(library, dict) or not library:
        raise RegressionFailure("replay params lack a chiplet library")
    sites = solution.get("sites")
    if not isinstance(sites, list) or not sites:
        raise RegressionFailure("%s solution lacks occupied sites" % method)

    counts = {}
    area = 0.0
    cpu_peak = 0.0
    gpu_peak = 0.0
    for site in sites:
        chiplet_type = str(site.get("type") or "")
        entry = library.get(chiplet_type)
        if not isinstance(entry, dict):
            raise RegressionFailure(
                "%s site uses unknown chiplet type %r" %
                (method, chiplet_type))
        counts[chiplet_type] = counts.get(chiplet_type, 0) + 1
        area += _finite_float(
            entry.get("area"), "%s %s area" % (method, chiplet_type))
        peak = _finite_float(
            entry.get("p_peak"),
            "%s %s p_peak" % (method, chiplet_type))
        if chiplet_type == "CPU":
            cpu_peak += peak
        elif chiplet_type.startswith("GPU"):
            gpu_peak += peak
        else:
            raise RegressionFailure(
                "%s A1550 BOM contains unsupported type %s" %
                (method, chiplet_type))

    cpu_count = counts.get("CPU", 0)
    gpu_l_count = counts.get("GPU-L", 0)
    gpu_s_count = counts.get("GPU-S", 0)
    gpu_count = sum(count for chiplet_type, count in counts.items()
                    if chiplet_type.startswith("GPU"))
    total_peak = cpu_peak + gpu_peak
    _same_number(solution.get("total_area_mm2"), area,
                 method + " solution total_area_mm2")
    _same_number(solution.get("peak_power_W"), total_peak,
                 method + " solution peak_power_W")

    replay_metrics = row.get("solution_metrics") or {}
    _same_number(replay_metrics.get("total_area_mm2"), area,
                 method + " replay total_area_mm2")
    _same_number(replay_metrics.get("peak_power_W"), total_peak,
                 method + " replay peak_power_W")
    for label, actual, expected in (
            ("cpu_count", replay_metrics.get("cpu_count"), cpu_count),
            ("gpu_l_count", replay_metrics.get("gpu_l_count"),
             gpu_l_count),
            ("gpu_s_count", replay_metrics.get("gpu_s_count"),
             gpu_s_count)):
        if int(_finite_float(actual, "%s replay %s" %
                             (method, label))) != expected:
            raise RegressionFailure(
                "%s replay %s does not match the solution BOM" %
                (method, label))
    return {
        "type_counts": dict(sorted(counts.items())),
        "occupied_chiplets": len(sites),
        "cpu_count": cpu_count,
        "gpu_count": gpu_count,
        "gpu_l_count": gpu_l_count,
        "gpu_s_count": gpu_s_count,
        "area_mm2": area,
        "cpu_peak_power_W": cpu_peak,
        "gpu_peak_power_W": gpu_peak,
        "cpu_plus_gpu_peak_power_W": total_peak,
        "power_semantics": (
            "library p_peak BOM property; not LEGOSim dynamic power"),
        "source": "params.library multiplied by occupied solution sites",
    }


def _solution_paths(out_root):
    root = Path(out_root) / "solutions" / "budget" / CASE_ID
    return {
        "ilp-perf": root / "ilp_perf.json",
        "ilp-power": root / "ilp_power.json",
    }


def _default_replay_dir(out_root):
    return Path(out_root) / "replays" / "a1550_perf_power"


def _budget_params_path(budget_source):
    source = _load_json(budget_source, "budget metadata")
    record = source.get("params") or {}
    raw = record.get("path")
    if not raw:
        raise RegressionFailure("budget metadata lacks params.path")
    params = _resolve(raw)
    if not params.is_file():
        raise RegressionFailure("budget params do not exist: %s" % params)
    expected = str(record.get("sha256") or "")
    if expected and _sha256(params) != expected:
        raise RegressionFailure("budget params SHA-256 mismatch: %s" % params)
    return params


def _solve_command(args, out_root):
    command = [
        str(_resolve(args.system_python)), str(FULL_DRIVER),
        "--phase", "solve-only",
        "--case", CASE_KEY,
        "--out-root", str(out_root),
        "--budget-source", str(_resolve(args.budget_source)),
        "--scale-source", str(_resolve(args.scale_source)),
        "--scale-params", str(_resolve(args.scale_params)),
        "--gurobi-python", str(_resolve(args.gurobi_python)),
        "--system-python", str(_resolve(args.system_python)),
    ]
    if args.resume:
        command.append("--resume")
    return command


def _replay_command(args, out_root, params_path, replay_dir):
    paths = _solution_paths(out_root)
    command = [
        str(_resolve(args.system_python)), str(REPLAY_DRIVER),
        "--params", str(params_path),
    ]
    for method in METHODS:
        command.extend([
            "--layout", "%s:%s:%s" %
            (CASE_ID, method, paths[method]),
        ])
    command.extend([
        "--out-dir", str(replay_dir),
        "--rounds", str(args.rounds),
        "--python", str(_resolve(args.system_python)),
        "--moe-e2e-timeout", str(args.moe_e2e_timeout),
        "--legosim-timeout", str(args.legosim_timeout),
        "--moe-gpgpusim-max-parallel",
        str(args.moe_gpgpusim_max_parallel),
        "--fail-fast",
    ])
    return command


def _run_command(command, label):
    process = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if process.returncode != 0:
        raise RegressionFailure("%s failed with exit code %d" %
                                (label, process.returncode))


def _bundle_signature(solution):
    return sorted((
        str(site.get("type")),
        tuple(sorted(str(task) for task in site.get("tasks") or [])),
    ) for site in solution.get("sites") or [])


def _surrogate_metric(solution, method):
    try:
        contract = full_validation._protocol_surrogate_contract(
            solution, method)
    except RuntimeError as exc:
        raise RegressionFailure(str(exc))
    return {
        "field": contract["cycles_field"],
        "cycles": contract["cycles"],
        "profile_field": contract["profile_field"],
        "profile": contract["profile"],
        "event_metadata_field": contract["event_metadata_field"],
        "event_metadata": contract["event_metadata"],
    }


def _stage0_gap(solution, method):
    try:
        return full_validation._stage0_gap_contract(solution, method)
    except RuntimeError as exc:
        raise RegressionFailure(str(exc))


def _stage0_metric(solution):
    for field in ("analytical_task_makespan_cycles",
                  "stage0_T_cycles", "predicted_makespan_cycles"):
        if solution.get(field) is not None:
            return {"field": field,
                    "cycles": _finite_float(solution.get(field), field)}
    return {"field": None, "cycles": None}


def _solver_metrics(solution, method):
    if solution.get("optimization_mode") != method:
        raise RegressionFailure(
            "%s solution has optimization_mode=%r" %
            (method, solution.get("optimization_mode")))
    stage0 = _stage0_gap(solution, method)
    protocol = _surrogate_metric(solution, method)
    try:
        protocol_contract = full_validation._protocol_surrogate_contract(
            solution, method)
        bundle_freeze = full_validation._bundle_freeze_contract(
            solution, method)
    except RuntimeError as exc:
        raise RegressionFailure(str(exc))
    secondary_gap = solution.get("secondary_mip_gap")
    if secondary_gap is not None:
        secondary_gap = _finite_float(
            secondary_gap, method + " secondary_mip_gap")
    noc = solution.get("noc_power_proxy") or {}
    routing = solution.get("routing") or {}
    cap = _finite_float(
        solution.get("performance_cap_cycles"),
        method + " protocol performance cap")
    if protocol["cycles"] > cap * (1.0 + 1.0e-7):
        raise RegressionFailure(
            "%s protocol surrogate %.9g exceeds cap %.9g" %
            (method, protocol["cycles"], cap))
    operating_status_field = "secondary_status"
    operating_wall_field = "secondary_wall_s"
    tie_break = protocol_contract.get("tie_break")
    if method == "ilp-perf":
        protocol_stage = {
            "status": solution.get(operating_status_field),
            "status_field": operating_status_field,
            "gap": protocol_contract["operating_gap"],
            "gap_field": protocol_contract["operating_gap_field"],
            "gap_certified": protocol_contract[
                "operating_gap_certificate"]["certified"],
            "gap_reason": protocol_contract[
                "operating_gap_certificate"]["reason"],
            "surrogate_cycles": protocol["cycles"],
            "wall_s": solution.get(operating_wall_field),
            "wall_field": operating_wall_field,
            "role": "operating protocol-completion objective",
            "is_protocol_certificate": protocol_contract[
                "operating_gap_certificate"]["is_protocol_certificate"],
        }
    else:
        # Power's protocol minimization is an optional tertiary tie-break.
        # Its possibly missing gap is diagnostic only, never a certificate.
        protocol_stage = {
            "status": tie_break["status"],
            "status_field": "tertiary_status",
            "gap": tie_break["gap"],
            "gap_field": tie_break["gap_field"],
            "gap_certified": tie_break["gap_certified"],
            "gap_reason": tie_break["gap_reason"],
            "surrogate_cycles": protocol["cycles"],
            "wall_s": solution.get("tertiary_wall_s"),
            "wall_field": "tertiary_wall_s",
            "role": tie_break["role"],
            "is_protocol_certificate": False,
        }
    operating_stage = {
        "kind": protocol_contract["operating_objective_kind"],
        "status": solution.get(operating_status_field),
        "status_field": operating_status_field,
        "gap": protocol_contract["operating_gap"],
        "gap_field": protocol_contract["operating_gap_field"],
        "gap_certificate": protocol_contract[
            "operating_gap_certificate"],
        "wall_s": solution.get(operating_wall_field),
        "wall_field": operating_wall_field,
    }
    return {
        "solver": {
            "stage0_T": {
                "status": solution.get("primary_status"),
                "gap": stage0["gap"],
                "gap_field": stage0["field"],
                "gap_certified": stage0["gap_certified"],
                "gap_reason": stage0["gap_reason"],
                "gap_certificate": {
                    "certified": stage0["gap_certified"],
                    "reason": stage0["gap_reason"],
                },
                "has_incumbent": stage0["has_incumbent"],
                "incumbent_cycles": solution.get(
                    "primary_incumbent_cycles"),
                "bound_cycles": solution.get(
                    "primary_objective_bound_cycles"),
                "wall_s": solution.get("primary_wall_s"),
            },
            "operating_objective_stage": operating_stage,
            "protocol_stage": protocol_stage,
            "other_stage_diagnostics": {
                "secondary_status": solution.get("secondary_status"),
                "secondary_gap": secondary_gap,
                "secondary_wall_s": solution.get("secondary_wall_s"),
                "tertiary_status": solution.get("tertiary_status"),
                "tertiary_gap": solution.get("tertiary_mip_gap"),
                "tertiary_wall_s": solution.get("tertiary_wall_s"),
                "quaternary_status": solution.get("quaternary_status"),
                "quaternary_gap": solution.get("quaternary_mip_gap"),
                "quaternary_wall_s": solution.get("quaternary_wall_s"),
            },
            "time_to_first_feasible_s": solution.get(
                "time_to_first_feasible_s"),
        },
        "surrogate": {
            "protocol_latency": protocol,
            "stage0_T_diagnostic": _stage0_metric(solution),
            "performance_reference_cycles": solution.get(
                "performance_reference_cycles"),
            "performance_cap_cycles": cap,
            "performance_cap_domain": "protocol-event-dag",
            "performance_relaxation": solution.get(
                "performance_relaxation"),
            "noc_dynamic_hop_energy_J": noc.get("energy_J"),
            "routed_byte_hop": routing.get("routed_byte_hop"),
            "maximum_link_utilization": routing.get(
                "maximum_link_utilization"),
        },
        "bundle_freeze": bundle_freeze,
    }


def _delay_info_path(results_path, row):
    legosim = row.get("legosim") or {}
    summary_path = legosim.get("summary_path")
    if summary_path:
        candidate = Path(str(summary_path)).resolve().parent / "delayInfo.txt"
        if candidate.is_file():
            return candidate
    candidate = (Path(results_path).resolve().parent / "layouts" /
                 str(row.get("case_id")) / str(row.get("method")) /
                 "legosim_result" / "delayInfo.txt")
    if candidate.is_file():
        return candidate
    raise RegressionFailure(
        "missing delayInfo.txt for %s:%s" %
        (row.get("case_id"), row.get("method")))


def _noc_active_cycles(delay_path):
    maximum = None
    line_count = 0
    try:
        with open(str(delay_path), "r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                fields = raw.split()
                if not fields:
                    continue
                if len(fields) < 6:
                    raise RegressionFailure(
                        "malformed %s line %d" % (delay_path, line_number))
                start = _finite_float(
                    fields[0], "%s line %d start" %
                    (delay_path, line_number))
                final_delay = _finite_float(
                    fields[-1], "%s line %d final delay" %
                    (delay_path, line_number))
                completion = start + final_delay
                maximum = completion if maximum is None else max(
                    maximum, completion)
                line_count += 1
    except OSError as exc:
        raise RegressionFailure("cannot read %s: %s" % (delay_path, exc))
    if maximum is None or maximum <= 0.0:
        raise RegressionFailure("empty/non-positive delay evidence: %s" %
                                delay_path)
    return maximum, line_count


def _measured_metrics(results_path, row, expected_gpu_workers):
    if row.get("status") != "ok":
        raise RegressionFailure(
            "%s replay status is %r" %
            (row.get("method"), row.get("status")))
    codegen = row.get("codegen") or {}
    if not codegen.get("dispatch_formal"):
        raise RegressionFailure("%s replay lacks formal dispatch" %
                                row.get("method"))
    legosim = row.get("legosim") or {}
    if legosim.get("status") != "ok" or legosim.get("validation") != "PASS":
        raise RegressionFailure("%s LEGOSim validation is not PASS" %
                                row.get("method"))
    summary = legosim.get("summary") or {}
    cycles = int(_finite_float(
        legosim.get("synchronized_makespan_cycles"),
        str(row.get("method")) + " synchronized cycles"))
    bench_cycles = int(_finite_float(
        legosim.get("bench_makespan_cycles"),
        str(row.get("method")) + " bench cycles"))
    measured_cycles = int(_finite_float(
        legosim.get("measured_makespan_cycles"),
        str(row.get("method")) + " measured cycles"))
    completion_delta = int(_finite_float(
        legosim.get("completion_delta_cycles"),
        str(row.get("method")) + " completion delta"))
    if measured_cycles != cycles:
        raise RegressionFailure(
            "%s measured cycles differ from synchronized cycles" %
            row.get("method"))
    if cycles - bench_cycles != completion_delta:
        raise RegressionFailure(
            "%s synchronized/bench completion delta is inconsistent" %
            row.get("method"))
    summary_cycles = int(_finite_float(
        summary.get("synchronized_makespan_cycles"),
        str(row.get("method")) + " summary synchronized cycles"))
    summary_bench = int(_finite_float(
        summary.get("makespan_cycles"),
        str(row.get("method")) + " summary bench cycles"))
    summary_delta = int(_finite_float(
        summary.get("completion_delta_cycles"),
        str(row.get("method")) + " summary completion delta"))
    if ((summary_cycles, summary_bench, summary_delta) !=
            (cycles, bench_cycles, completion_delta)):
        raise RegressionFailure(
            "%s LEGOSim JSON and summary cycle fields disagree" %
            row.get("method"))
    gpu_cycle_max = int(_finite_float(
        summary.get("gpu_cycle_max"),
        str(row.get("method")) + " GPU cycle maximum"))
    n_gpu_workers = int(_finite_float(
        summary.get("n_gpu_workers"),
        str(row.get("method")) + " GPU worker count"))
    if n_gpu_workers != int(expected_gpu_workers):
        raise RegressionFailure(
            "%s LEGOSim GPU worker count %d does not match BOM count %d" %
            (row.get("method"), n_gpu_workers,
             int(expected_gpu_workers)))
    power = _finite_float(
        summary.get("popnet_total_power"),
        str(row.get("method")) + " PopNet power")
    average_delay = _finite_float(
        summary.get("popnet_average_delay"),
        str(row.get("method")) + " PopNet average delay")
    delay_path = _delay_info_path(results_path, row)
    active_cycles, lines = _noc_active_cycles(delay_path)
    expected_finished = summary.get("popnet_total_finished")
    if expected_finished is not None and int(expected_finished) != lines:
        raise RegressionFailure(
            "%s delayInfo has %d records but PopNet reports %s" %
            (row.get("method"), lines, expected_finished))
    return {
        "cycles": cycles,
        "cycle_metric": "LEGOSim synchronized_makespan_cycles",
        "bench_cycles": bench_cycles,
        "completion_delta_cycles": completion_delta,
        "gpu_cycle_max": gpu_cycle_max,
        "n_gpu_workers": n_gpu_workers,
        "noc_average_power_W": power,
        "noc_power_source": "LEGOSim PopNet/ORION popnet_total_power",
        "noc_average_delay_cycles": average_delay,
        "noc_active_cycles": active_cycles,
        "noc_dynamic_energy_J_at_1GHz": power * active_cycles / 1.0e9,
        "noc_dynamic_energy_uJ_at_1GHz": power * active_cycles / 1.0e3,
        "popnet_finished": expected_finished,
        "delay_records": lines,
        "delay_info_path": str(delay_path),
    }


def audit_pair(perf_solution_path, power_solution_path, results_path,
               max_primary_gap=None, max_protocol_gap=None,
               require_only_pair=True):
    """Return a complete report, marking the regression gate PASS or FAIL."""
    paths = {
        "ilp-perf": Path(perf_solution_path).resolve(),
        "ilp-power": Path(power_solution_path).resolve(),
    }
    solutions = {
        method: _load_json(paths[method], method + " solution")
        for method in METHODS
    }
    results_path = Path(results_path).resolve()
    results = _load_json(results_path, "two-layout replay results")
    if results.get("status") != "complete":
        raise RegressionFailure("replay results status is not complete")
    layouts = results.get("layouts")
    if not isinstance(layouts, list):
        raise RegressionFailure("replay results lack layouts list")
    if require_only_pair and len(layouts) != 2:
        raise RegressionFailure(
            "strict replay must contain exactly two rows, found %d" %
            len(layouts))
    rows = {}
    for row in layouts:
        key = (str(row.get("case_id")), str(row.get("method")))
        if key[0] == CASE_ID and key[1] in METHODS:
            if key[1] in rows:
                raise RegressionFailure("duplicate replay row %s:%s" % key)
            rows[key[1]] = row
    if sorted(rows) != sorted(METHODS):
        raise RegressionFailure("replay does not contain both A1550 methods")

    replay_identity, params = _verify_replay_identity(
        results, results_path, rows, paths)

    failures = []
    if _bundle_signature(solutions["ilp-perf"]) != _bundle_signature(
            solutions["ilp-power"]):
        failures.append("ILP-Power changed the ILP-Perf complete bundles")

    methods = {}
    for method in METHODS:
        metrics = _solver_metrics(solutions[method], method)
        metrics["hardware_bom"] = _hardware_metrics(
            solutions[method], params, rows[method], method)
        metrics["measured"] = _measured_metrics(
            results_path, rows[method],
            metrics["hardware_bom"]["gpu_count"])
        metrics["solution_path"] = str(paths[method])
        metrics["solution_sha256"] = _sha256(paths[method])
        methods[method] = metrics
        if max_primary_gap is not None:
            primary_gap = metrics["solver"]["stage0_T"]["gap"]
            if primary_gap is None:
                failures.append(
                    "%s Stage-0 T has no finite gap certificate" % method)
            elif primary_gap > max_primary_gap:
                failures.append(
                    "%s Stage-0 T gap %.6g exceeds %.6g" %
                    (method, primary_gap, max_primary_gap))
        if method == "ilp-perf" and max_protocol_gap is not None:
            protocol_gap = metrics["solver"]["protocol_stage"]["gap"]
            if protocol_gap is None:
                failures.append(
                    "%s protocol stage has no finite gap certificate" %
                    method)
            elif protocol_gap > max_protocol_gap:
                failures.append(
                    "%s protocol-stage gap %.6g exceeds %.6g" %
                    (method, protocol_gap, max_protocol_gap))

    perf_bom = methods["ilp-perf"]["hardware_bom"]
    power_bom = methods["ilp-power"]["hardware_bom"]
    shared_fields = (
        "type_counts", "occupied_chiplets", "cpu_count", "gpu_count",
        "gpu_l_count", "gpu_s_count", "area_mm2",
        "cpu_peak_power_W", "gpu_peak_power_W",
        "cpu_plus_gpu_peak_power_W")
    if any(perf_bom[field] != power_bom[field]
           for field in shared_fields):
        failures.append(
            "ILP-Perf and ILP-Power do not have an identical shared BOM")
    shared_bom = {field: perf_bom[field] for field in shared_fields}
    shared_bom.update({
        "power_semantics": perf_bom["power_semantics"],
        "source": perf_bom["source"],
        "reporting_rule": (
            "CPU+GPU peak is listed once as the shared BOM; PopNet NoC "
            "power is measured and remains method-specific"),
    })

    perf = methods["ilp-perf"]["measured"]
    power = methods["ilp-power"]["measured"]
    power_faster = power["cycles"] < perf["cycles"]
    power_lower_energy = (
        power["noc_dynamic_energy_J_at_1GHz"] <
        perf["noc_dynamic_energy_J_at_1GHz"])
    power_lower_average_power = (
        power["noc_average_power_W"] < perf["noc_average_power_W"])
    perf_faster = perf["cycles"] < power["cycles"]
    power_saves_noc = power_lower_energy or power_lower_average_power
    dominated_cycles_energy = power_faster and power_lower_energy
    dominated_cycles_power = power_faster and power_lower_average_power
    if dominated_cycles_energy:
        failures.append(
            "ILP-Perf is dominated on measured cycles and NoC energy")
    if dominated_cycles_power:
        failures.append(
            "ILP-Perf is dominated on measured cycles and average NoC power")
    if not perf_faster:
        failures.append(
            "ILP-Perf does not have fewer measured cycles than ILP-Power")
    if not power_saves_noc:
        failures.append(
            "ILP-Power does not reduce measured NoC energy or average power")

    comparison = {
        "power_minus_perf": {
            "cycles": power["cycles"] - perf["cycles"],
            "cycles_percent": 100.0 * (
                float(power["cycles"]) / float(perf["cycles"]) - 1.0),
            "noc_average_power_W": (
                power["noc_average_power_W"] -
                perf["noc_average_power_W"]),
            "noc_average_power_percent": 100.0 * (
                power["noc_average_power_W"] /
                perf["noc_average_power_W"] - 1.0),
            "noc_dynamic_energy_J_at_1GHz": (
                power["noc_dynamic_energy_J_at_1GHz"] -
                perf["noc_dynamic_energy_J_at_1GHz"]),
            "noc_dynamic_energy_percent": 100.0 * (
                power["noc_dynamic_energy_J_at_1GHz"] /
                perf["noc_dynamic_energy_J_at_1GHz"] - 1.0),
            "surrogate_cycles": (
                methods["ilp-power"]["surrogate"][
                    "protocol_latency"]["cycles"] -
                methods["ilp-perf"]["surrogate"][
                    "protocol_latency"]["cycles"]),
        },
        "power_faster": power_faster,
        "perf_faster": perf_faster,
        "power_lower_noc_energy": power_lower_energy,
        "power_lower_noc_average_power": power_lower_average_power,
        "power_saves_noc": power_saves_noc,
        "perf_dominated_on_cycles_energy": dominated_cycles_energy,
        "perf_dominated_on_cycles_power": dominated_cycles_power,
        "perf_dominated_on_all_three": (
            power_faster and power_lower_energy and
            power_lower_average_power),
    }
    return {
        "schema_version": 2,
        "case_id": CASE_ID,
        "status": "PASS" if not failures else "FAIL",
        "gate": {
            "contract": (
                "require ILP-Perf to have fewer measured cycles and "
                "ILP-Power to reduce measured NoC energy or average "
                "power; also reject a dominated ILP-Perf point"),
            "failures": failures,
            "max_primary_gap": max_primary_gap,
            "max_protocol_gap": max_protocol_gap,
        },
        "shared_bom": shared_bom,
        "methods": methods,
        "comparison": comparison,
        "replay_identity": replay_identity,
        "replay_results_path": str(results_path),
        "replay_results_sha256": _sha256(results_path),
    }


def _solver_only_report(out_root, max_primary_gap=None,
                        max_protocol_gap=None):
    paths = _solution_paths(out_root)
    solutions = {
        method: _load_json(paths[method], method + " solution")
        for method in METHODS
    }
    failures = []
    if _bundle_signature(solutions["ilp-perf"]) != _bundle_signature(
            solutions["ilp-power"]):
        failures.append("ILP-Power changed the ILP-Perf complete bundles")
    methods = {}
    for method in METHODS:
        methods[method] = _solver_metrics(solutions[method], method)
        methods[method]["solution_path"] = str(paths[method])
        if max_primary_gap is not None:
            primary_gap = methods[method]["solver"]["stage0_T"]["gap"]
            if primary_gap is None:
                failures.append(
                    "%s Stage-0 T has no finite gap certificate" % method)
            elif primary_gap > max_primary_gap:
                failures.append(
                    "%s Stage-0 T gap %.6g exceeds %.6g" %
                    (method, primary_gap, max_primary_gap))
        if method == "ilp-perf" and max_protocol_gap is not None:
            protocol_gap = methods[method]["solver"][
                "protocol_stage"]["gap"]
            if protocol_gap is None:
                failures.append(
                    "%s protocol stage has no finite gap certificate" %
                    method)
            elif protocol_gap > max_protocol_gap:
                failures.append(
                    "%s protocol-stage gap %.6g exceeds %.6g" %
                    (method, protocol_gap, max_protocol_gap))
    return {
        "schema_version": 1,
        "case_id": CASE_ID,
        "status": "SOLVE_COMPLETE" if not failures else "FAIL",
        "gate": {"failures": failures,
                 "max_primary_gap": max_primary_gap,
                 "max_protocol_gap": max_protocol_gap},
        "methods": methods,
    }


def _validate_args(args):
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.moe_e2e_timeout <= 0 or args.legosim_timeout <= 0:
        raise ValueError("timeouts must be positive")
    if args.moe_gpgpusim_max_parallel <= 0:
        raise ValueError("--moe-gpgpusim-max-parallel must be positive")
    if args.max_primary_gap is not None and args.max_primary_gap < 0.0:
        raise ValueError("--max-primary-gap must be nonnegative")
    if args.max_protocol_gap is not None and args.max_protocol_gap < 0.0:
        raise ValueError("--max-protocol-gap must be nonnegative")


def main(argv=None):
    args = _parse_args(argv)
    _validate_args(args)
    out_root = _resolve(args.out_root)
    replay_dir = _default_replay_dir(out_root)
    results_path = (_resolve(args.replay_results)
                    if args.replay_results else replay_dir / "results.json")
    report_path = (_resolve(args.report) if args.report else
                   out_root / "a1550_perf_power_regression.json")
    params_path = _budget_params_path(_resolve(args.budget_source))
    solve_command = _solve_command(args, out_root)
    replay_command = _replay_command(
        args, out_root, params_path, replay_dir)

    plan = {
        "case": CASE_KEY,
        "phase": args.phase,
        "out_root": str(out_root),
        "solutions": {method: str(path) for method, path in
                      _solution_paths(out_root).items()},
        "solve_command": solve_command,
        "replay_command": replay_command,
        "replay_layout_count": 2,
        "replay_methods": list(METHODS),
        "replay_results": str(results_path),
        "report": str(report_path),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if args.phase in ("all", "solve-only"):
        _run_command(solve_command, "A1550 two-mode solve")

    if args.phase == "solve-only":
        report = _solver_only_report(
            out_root, args.max_primary_gap, args.max_protocol_gap)
        _atomic_json(report_path, report)
        print(json.dumps({
            "status": report["status"],
            "report": str(report_path),
            "methods": report["methods"],
        }, indent=2, sort_keys=True))
        if report["status"] == "FAIL":
            raise RegressionFailure("solver-only A1550 gate failed; see %s" %
                                    report_path)
        return 0

    if args.phase in ("all", "replay-only"):
        for method, path in _solution_paths(out_root).items():
            if not path.is_file():
                raise RegressionFailure("missing %s solution: %s" %
                                        (method, path))
        if replay_dir.exists():
            if not args.resume:
                raise RegressionFailure(
                    "refusing to overwrite replay directory: %s" % replay_dir)
            if not (replay_dir / "results.json").is_file():
                raise RegressionFailure(
                    "resume replay directory lacks results.json: %s" %
                    replay_dir)
        else:
            _run_command(replay_command, "A1550 two-layout LEGOSim replay")

    report = audit_pair(
        _solution_paths(out_root)["ilp-perf"],
        _solution_paths(out_root)["ilp-power"],
        results_path,
        max_primary_gap=args.max_primary_gap,
        max_protocol_gap=args.max_protocol_gap,
        require_only_pair=(args.replay_results is None))
    _atomic_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "solver": {method: report["methods"][method]["solver"]
                   for method in METHODS},
        "surrogate": {method: report["methods"][method]["surrogate"]
                      for method in METHODS},
        "measured": {method: report["methods"][method]["measured"]
                     for method in METHODS},
        "comparison": report["comparison"],
        "failures": report["gate"]["failures"],
    }, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise RegressionFailure(
            "A1550 Perf/Power regression failed; see %s" % report_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RegressionFailure as exc:
        print("regression failure: %s" % exc, file=sys.stderr)
        sys.exit(1)
    except (OSError, RuntimeError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        sys.exit(2)
