#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Solve and replay the 13 fixed-reference ILP-Power revalidation cases.

This is an orchestration-only driver: it does not implement or modify an ILP.
It derives the eight budget cases and five scaling cases from the canonical
experiments, invokes ``ilp_layout.py`` with one fixed protocol, and invokes
``replay_legosim_solutions.py`` in a separate, resumable replay phase.

The driver never overwrites an output root.  Solver output is written to a
temporary path and atomically renamed only after valid JSON is produced.
Manifests and hashes are likewise replaced atomically.
"""

from __future__ import print_function

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = "out/ilp_power_reval_rho20_20260801"
DEFAULT_BUDGET_SOURCE = (
    "out/corrected_e2e_runs/corrected_e2e_20260731_v2/raw_results.json")
DEFAULT_SCALE_SOURCE = "out/scaling_ilp_secondary_20260801"
DEFAULT_SCALE_PARAMS = "out/scaling_qap_experiment/params"
DEFAULT_GUROBI_PYTHON = os.environ.get("GUROBI_PYTHON", sys.executable)
SCALES = ((8, 6), (16, 8), (32, 12), (64, 12), (128, 16))
# Longest-processing-time-first order for two replay slots.  Values are ranks,
# not claimed wall-clock predictions; they encode the ordering observed from
# the canonical LEGOSim scaling runs.  The eight small budget cases share one
# invocation and are intentionally submitted last.
REPLAY_COST_RANK = {
    "scaling_e128_m16": 6,
    "scaling_e064_m12": 5,
    "scaling_e032_m12": 4,
    "scaling_e016_m08": 3,
    "scaling_e008_m06": 2,
    "budget": 1,
}

FIXED_PROTOCOL = {
    "mode": "ilp-power",
    "reference_policy": "placement-from",
    "cpu_anchor": "free",
    "power_perf_relax": 0.20,
    "noc_traffic": "deterministic-dispatch",
    "dispatch_tokens_per_source": 32,
    "dispatch_hidden": 128,
    "threads": 1,
    "primary_time_limit_s": 30.0,
    "secondary_time_limit_s": 30.0,
    "seed": 1,
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the fixed 13-case ILP-Power solve/replay protocol")
    parser.add_argument(
        "--phase", choices=("solve-only", "replay-only", "all"),
        default="solve-only",
        help="execution phase (default: solve-only)")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--budget-source", default=DEFAULT_BUDGET_SOURCE)
    parser.add_argument("--scale-source", default=DEFAULT_SCALE_SOURCE)
    parser.add_argument("--scale-params", default=DEFAULT_SCALE_PARAMS)
    parser.add_argument("--gurobi-python", default=DEFAULT_GUROBI_PYTHON)
    parser.add_argument("--system-python", default=sys.executable)
    parser.add_argument(
        "--case", action="append", default=[], metavar="SUITE:CASE",
        help=("select a case such as budget:area_970 or "
              "scaling:e008_m06; repeat as needed"))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--budget-e2e-timeout", type=int, default=1800)
    parser.add_argument("--scale-e2e-timeout", type=int, default=21600)
    parser.add_argument(
        "--budget-legosim-timeout", type=int, default=2400,
        help="outer replay timeout for budget invocation (default: 2400s)")
    parser.add_argument(
        "--scale-legosim-timeout", type=int, default=25200,
        help="outer replay timeout for scaling invocations (default: 25200s)")
    parser.add_argument("--moe-gpgpusim-max-parallel", type=int, default=16)
    parser.add_argument(
        "--replay-jobs", type=int, choices=(1, 2), default=1,
        help=("independent replay subprocesses to run concurrently; each "
              "still receives --moe-gpgpusim-max-parallel (default: 1)"))
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate inputs and print all commands without creating output")
    return parser.parse_args(argv)


def _load_json(path, label):
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read %s %s: %s" % (label, path, exc))
    if not isinstance(value, dict):
        raise ValueError("%s must be a JSON object: %s" % (label, path))
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


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with open(str(temporary), "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with open(str(temporary), "w", encoding="utf-8", newline="") as handle:
        handle.write(value)
    os.replace(str(temporary), str(path))


def _resolve(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _require_file(path, label):
    if not path.is_file():
        raise ValueError("missing %s: %s" % (label, path))


def _grid(solution, label):
    value = solution.get("grid") or {}
    try:
        grid = (int(value["L"]), int(value["X"]), int(value["Y"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid %s grid" % label)
    if any(item <= 0 for item in grid):
        raise ValueError("non-positive %s grid" % label)
    return grid


def _budget_cases(source_path):
    source = _load_json(source_path, "budget experiment")
    if source.get("status") != "complete":
        raise ValueError("budget experiment is not complete")
    params = _resolve((source.get("params") or {}).get("path"))
    _require_file(params, "budget params")
    expected_params_sha = str((source.get("params") or {}).get("sha256", ""))
    if _sha256(params) != expected_params_sha:
        raise ValueError("budget params SHA-256 mismatch")
    rows = source.get("cases")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("budget source must contain exactly 8 cases")
    cases = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        budgets = row.get("budgets") or {}
        reference = _resolve((row.get("ilp") or {}).get("solution_path"))
        _require_file(reference, "%s reference" % case_id)
        solution = _load_json(reference, "%s reference" % case_id)
        grid = _grid(solution, "%s reference" % case_id)
        if grid != (1, 6, 6):
            raise ValueError("%s reference grid is %r, expected 1x6x6" %
                             (case_id, grid))
        cases.append({
            "key": "budget:%s" % case_id,
            "suite": "budget",
            "case_id": case_id,
            "params": params,
            "reference": reference,
            "grid": grid,
            "budgets": {
                "a_max": float(budgets["a_max"]),
                "p_max": float(budgets["p_max"]),
            },
            "mipgap": float((source.get("experiment") or {}).get(
                "mipgap_target", 1e-6)),
            "secondary_mipgap": float((source.get("experiment") or {}).get(
                "secondary_mipgap_target", 0.02)),
            "source_metadata": source_path,
        })
    return cases


def _scale_cases(source_root, params_root):
    cases = []
    for experts, side in SCALES:
        pair = "e%03d_m%02d" % (experts, side)
        command_path = source_root / "commands" / (pair + "_command.json")
        reference = source_root / "solutions" / (pair + "_ilp_secondary.json")
        params = params_root / ("experts_%03d.json" % experts)
        for path, label in ((command_path, "scale command"),
                            (reference, "scale reference"),
                            (params, "scale params")):
            _require_file(path, "%s %s" % (pair, label))
        command = _load_json(command_path, "%s command" % pair)
        solution = _load_json(reference, "%s reference" % pair)
        grid = _grid(solution, "%s reference" % pair)
        if grid != (1, side, side):
            raise ValueError("%s reference grid is %r" % (pair, grid))
        params_value = _load_json(params, "%s params" % pair)
        if (params_value.get("meta") or {}).get("experts") != experts:
            raise ValueError("%s params expert count mismatch" % pair)
        budgets = command.get("budgets") or {}
        options = command.get("solver_options") or {}
        cases.append({
            "key": "scaling:%s" % pair,
            "suite": "scaling",
            "case_id": pair,
            "params": params.resolve(),
            "reference": reference.resolve(),
            "grid": grid,
            "budgets": {
                "a_max": float(budgets["a_max"]),
                "p_max": float(budgets["p_max"]),
            },
            "mipgap": float(options.get("mipgap", 0.01)),
            "secondary_mipgap": float(options.get(
                "secondary_mipgap", 0.02)),
            "source_metadata": command_path.resolve(),
        })
    return cases


def _solution_path(out_root, case):
    return (out_root / "solutions" / case["suite"] /
            (case["case_id"] + "_ilp_power.json"))


def _solver_command(case, out_path, gurobi_python):
    l_dim, x_dim, y_dim = case["grid"]
    budgets = case["budgets"]
    return [
        str(gurobi_python), str(PROJECT_ROOT / "src" / "ilp_layout.py"),
        "--params", str(case["params"]),
        "--L", str(l_dim), "--X", str(x_dim), "--Y", str(y_dim),
        "--a-max", str(budgets["a_max"]),
        "--p-max", str(budgets["p_max"]),
        "--time-limit", "30.0",
        "--mipgap", str(case["mipgap"]),
        "--seed", "1",
        "--mode", "ilp-power",
        "--secondary-time", "30.0",
        "--secondary-mipgap", str(case["secondary_mipgap"]),
        "--power-perf-relax", "0.20",
        "--noc-traffic", "deterministic-dispatch",
        "--dispatch-tokens-per-source", "32",
        "--dispatch-hidden", "128",
        "--placement-from", str(case["reference"]),
        "--cpu-anchor", "free",
        "--threads", "1",
        "--quiet", "-o", str(out_path),
    ]


def _replay_command(cases, out_dir, system_python, args):
    if not cases:
        raise ValueError("cannot build an empty replay command")
    params = {str(case["params"]) for case in cases}
    if len(params) != 1:
        raise ValueError("one replay invocation must share one params file")
    is_budget = cases[0]["suite"] == "budget"
    timeout = (args.budget_e2e_timeout if is_budget
               else args.scale_e2e_timeout)
    outer_timeout = (args.budget_legosim_timeout if is_budget
                     else args.scale_legosim_timeout)
    command = [
        str(system_python),
        str(PROJECT_ROOT / "src" / "replay_legosim_solutions.py"),
        "--params", next(iter(params)),
    ]
    for case in cases:
        command.extend([
            "--layout", "%s:ilp-power:%s" %
            (case["case_id"], _solution_path(_resolve(args.out_root), case))])
    command.extend([
        "--out-dir", str(out_dir),
        "--rounds", str(args.rounds),
        "--moe-e2e-timeout", str(timeout),
        "--legosim-timeout", str(outer_timeout),
        "--moe-gpgpusim-max-parallel",
        str(args.moe_gpgpusim_max_parallel),
        "--fail-fast",
    ])
    return command


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


def _case_manifest(case, out_root, gurobi_python):
    output = _solution_path(out_root, case)
    return {
        "key": case["key"],
        "suite": case["suite"],
        "case_id": case["case_id"],
        "grid": {"L": case["grid"][0], "X": case["grid"][1],
                 "Y": case["grid"][2]},
        "budgets": case["budgets"],
        "mipgap": case["mipgap"],
        "secondary_mipgap": case["secondary_mipgap"],
        "params_path": str(case["params"]),
        "params_sha256": _sha256(case["params"]),
        "reference_path": str(case["reference"]),
        "reference_sha256": _sha256(case["reference"]),
        "source_metadata_path": str(case["source_metadata"]),
        "source_metadata_sha256": _sha256(case["source_metadata"]),
        "solution_path": str(output),
        "solve_command": _solver_command(case, output, gurobi_python),
        "solve_status": "pending",
        "solution_sha256": None,
        "replay_status": "pending",
    }


def _protocol_manifest(cases, out_root, gurobi_python, system_python):
    return {
        "schema_version": 1,
        "status": "planned",
        "project_root": str(PROJECT_ROOT),
        "out_root": str(out_root),
        "fixed_protocol": FIXED_PROTOCOL,
        "gurobi_python": str(gurobi_python),
        "system_python": str(system_python),
        "drivers": {
            "ilp_layout.py": {
                "path": str(PROJECT_ROOT / "src" / "ilp_layout.py"),
                "sha256": _sha256(PROJECT_ROOT / "src" / "ilp_layout.py"),
            },
            "replay_legosim_solutions.py": {
                "path": str(PROJECT_ROOT / "src" /
                            "replay_legosim_solutions.py"),
                "sha256": _sha256(PROJECT_ROOT / "src" /
                                  "replay_legosim_solutions.py"),
            },
            "orchestrator": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "selected_case_count": len(cases),
        "cases": [_case_manifest(case, out_root, gurobi_python)
                  for case in cases],
        "replay_invocations": [],
    }


def _artifact_hashes(manifest):
    artifacts = []
    for case in manifest["cases"]:
        for kind in ("params", "reference", "source_metadata", "solution"):
            path = case.get(kind + "_path")
            digest = case.get(kind + "_sha256")
            if path and digest:
                artifacts.append({"case": case["key"], "kind": kind,
                                  "path": path, "sha256": digest})
    for replay in manifest.get("replay_invocations") or []:
        results = replay.get("results_json")
        if results and Path(results).is_file():
            artifacts.append({"case": replay["name"], "kind": "replay",
                              "path": results, "sha256": _sha256(Path(results))})
    return {"schema_version": 1, "artifacts": artifacts}


def _run_command(command, stdout_path, stderr_path):
    process = subprocess.run(command, cwd=str(PROJECT_ROOT),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    _atomic_text(stdout_path, process.stdout)
    _atomic_text(stderr_path, process.stderr)
    return process.returncode


def _solve(cases, out_root, manifest, gurobi_python):
    manifest["status"] = "solving"
    _atomic_json(out_root / "manifest.json", manifest)
    rows = {row["key"]: row for row in manifest["cases"]}
    for case in cases:
        row = rows[case["key"]]
        final_path = _solution_path(out_root, case)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            raise RuntimeError("refusing to overwrite solution: %s" % final_path)
        temporary = final_path.with_name(
            final_path.name + ".tmp.%d" % os.getpid())
        command = _solver_command(case, temporary, gurobi_python)
        row["solve_command"] = command[:-2] + ["-o", str(final_path)]
        command_record = {
            "schema_version": 1,
            "case": case["key"],
            "command": row["solve_command"],
            "temporary_output": str(temporary),
        }
        command_path = (out_root / "commands" /
                        (case["suite"] + "_" + case["case_id"] + ".json"))
        _atomic_json(command_path, command_record)
        log_base = out_root / "logs" / (case["suite"] + "_" + case["case_id"])
        returncode = _run_command(command, Path(str(log_base) + ".stdout.log"),
                                  Path(str(log_base) + ".stderr.log"))
        if returncode != 0:
            row["solve_status"] = "failed"
            row["returncode"] = returncode
            if temporary.exists():
                temporary.unlink()
            manifest["status"] = "solve_failed"
            _atomic_json(out_root / "manifest.json", manifest)
            raise RuntimeError("solver failed for %s; see %s" %
                               (case["key"], log_base))
        solution = _load_json(temporary, "%s solution" % case["key"])
        if str(solution.get("optimization_mode")) != "ilp-power":
            if temporary.exists():
                temporary.unlink()
            raise RuntimeError("%s solution is not mode=ilp-power" % case["key"])
        os.replace(str(temporary), str(final_path))
        row["solve_status"] = "complete"
        row["solution_sha256"] = _sha256(final_path)
        row["returncode"] = returncode
        _atomic_json(out_root / "manifest.json", manifest)
        _atomic_json(out_root / "artifact_hashes.json",
                     _artifact_hashes(manifest))
    manifest["status"] = "solve_complete"
    _atomic_json(out_root / "manifest.json", manifest)


def _verify_solutions(cases, out_root, manifest):
    rows = {row["key"]: row for row in manifest.get("cases") or []}
    for case in cases:
        row = rows.get(case["key"])
        if not row:
            raise RuntimeError("manifest lacks case %s" % case["key"])
        path = _solution_path(out_root, case)
        _require_file(path, "%s ILP-Power solution" % case["key"])
        if row.get("solution_sha256") != _sha256(path):
            raise RuntimeError("solution hash mismatch for %s" % case["key"])
        solution = _load_json(path, "%s solution" % case["key"])
        if solution.get("optimization_mode") != "ilp-power":
            raise RuntimeError("solution mode mismatch for %s" % case["key"])
        for kind in ("params", "reference", "source_metadata"):
            expected_path = row.get(kind + "_path")
            expected_sha = row.get(kind + "_sha256")
            current_path = case[kind if kind != "source_metadata"
                                else "source_metadata"]
            if expected_path != str(current_path):
                raise RuntimeError("%s path mismatch for %s" %
                                   (kind, case["key"]))
            if expected_sha != _sha256(current_path):
                raise RuntimeError("%s hash mismatch for %s" %
                                   (kind, case["key"]))


def _replay_groups(cases):
    groups = []
    budget = [case for case in cases if case["suite"] == "budget"]
    if budget:
        groups.append(("budget", budget))
    for case in cases:
        if case["suite"] == "scaling":
            groups.append(("scaling_" + case["case_id"], [case]))
    return groups


def _execute_replay_invocation(invocation):
    """Run one isolated replay without reading or writing the parent manifest."""
    result = dict(invocation)
    returncode = _run_command(
        result["command"], Path(result["stdout_log"]),
        Path(result["stderr_log"]))
    result["returncode"] = returncode
    result["results_json"] = str(Path(result["out_dir"]) / "results.json")
    result["status"] = "complete" if returncode == 0 else "failed"
    return result


def _execute_replay_invocations(invocations, replay_jobs):
    """Execute independent commands; result order follows the input plan."""
    if replay_jobs not in (1, 2):
        raise ValueError("replay_jobs must be 1 or 2")
    if replay_jobs == 1:
        return [_execute_replay_invocation(item) for item in invocations]
    scheduled = _replay_submission_order(invocations, replay_jobs)
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=replay_jobs) as executor:
        # executor.map submits in iterable order.  LPT reduces the two-slot
        # tail; workers never mutate the shared manifest.
        return list(executor.map(_execute_replay_invocation, scheduled))


def _replay_submission_order(invocations, replay_jobs):
    """Return the actual executor submission order."""
    if replay_jobs not in (1, 2):
        raise ValueError("replay_jobs must be 1 or 2")
    if replay_jobs == 1:
        return list(invocations)
    return sorted(invocations,
                  key=lambda item: (-int(item["estimated_cost_rank"]),
                                    item["name"]))


def _replay(cases, out_root, manifest, system_python, args, dry_run=False):
    if not dry_run:
        _verify_solutions(cases, out_root, manifest)
        manifest["status"] = "replaying"
        _atomic_json(out_root / "manifest.json", manifest)
    invocations = []
    for name, group in _replay_groups(cases):
        if name == "budget":
            replay_dir = out_root / "replays" / "budget"
        else:
            replay_dir = out_root / "replays" / "scaling" / group[0]["case_id"]
        command = _replay_command(group, replay_dir, system_python, args)
        log_base = out_root / "logs" / ("replay_" + name)
        invocation = {"name": name, "cases": [case["key"] for case in group],
                      "command": command, "out_dir": str(replay_dir),
                      "status": "planned", "results_json": None,
                      "estimated_cost_rank": REPLAY_COST_RANK[name],
                      "stdout_log": str(Path(str(log_base) + ".stdout.log")),
                      "stderr_log": str(Path(str(log_base) + ".stderr.log"))}
        invocations.append(invocation)
        if dry_run:
            continue
        if replay_dir.exists():
            raise RuntimeError("refusing to overwrite replay: %s" % replay_dir)
    if not dry_run:
        # No worker writes manifest.json.  Commit the complete batch once all
        # child processes have terminated, so concurrent completion cannot
        # lose or interleave state updates.
        completed = _execute_replay_invocations(invocations,
                                                args.replay_jobs)
        invocations = completed
        manifest["replay_invocations"] = completed
        manifest["replay_configuration"] = {
            "replay_jobs": args.replay_jobs,
            "moe_gpgpusim_max_parallel_per_job":
                args.moe_gpgpusim_max_parallel,
            "maximum_configured_parallel_gpu_simulators":
                args.replay_jobs * args.moe_gpgpusim_max_parallel,
            "budget_timeouts_s": {
                "runner_internal_moe_e2e": args.budget_e2e_timeout,
                "outer_replay_legosim": args.budget_legosim_timeout,
            },
            "scaling_timeouts_s": {
                "runner_internal_moe_e2e": args.scale_e2e_timeout,
                "outer_replay_legosim": args.scale_legosim_timeout,
            },
        }
        failed = [item for item in completed if item["status"] != "complete"]
        completed_keys = {key for item in completed
                          if item["status"] == "complete"
                          for key in item["cases"]}
        for row in manifest["cases"]:
            if row["key"] in completed_keys:
                row["replay_status"] = "complete"
        manifest["status"] = "replay_failed" if failed else "complete"
        _atomic_json(out_root / "manifest.json", manifest)
        _atomic_json(out_root / "artifact_hashes.json",
                     _artifact_hashes(manifest))
        if failed:
            raise RuntimeError("%d replay invocation(s) failed: %s" % (
                len(failed), ", ".join(item["name"] for item in failed)))
    return _replay_submission_order(invocations, args.replay_jobs)


def main(argv=None):
    args = _parse_args(argv)
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

    out_root = _resolve(args.out_root)
    gurobi_python = _resolve(args.gurobi_python)
    system_python = _resolve(args.system_python)
    _require_file(gurobi_python, "Gurobi Python")
    _require_file(system_python, "system Python")
    _require_file(PROJECT_ROOT / "src" / "ilp_layout.py", "ilp_layout.py")
    _require_file(PROJECT_ROOT / "src" / "replay_legosim_solutions.py",
                  "replay driver")

    all_cases = (_budget_cases(_resolve(args.budget_source)) +
                 _scale_cases(_resolve(args.scale_source),
                              _resolve(args.scale_params)))
    if len(all_cases) != 13:
        raise ValueError("canonical plan must contain 13 cases")
    cases = _select_cases(all_cases, args.case)

    if args.phase == "replay-only":
        if not out_root.is_dir():
            raise RuntimeError("replay-only requires existing out root: %s" %
                               out_root)
        manifest = _load_json(out_root / "manifest.json", "run manifest")
        if manifest.get("fixed_protocol") != FIXED_PROTOCOL:
            raise RuntimeError("run manifest fixed protocol does not match")
        if not args.case:
            manifest_keys = [row.get("key")
                             for row in manifest.get("cases") or []]
            cases = _select_cases(all_cases, manifest_keys)
    else:
        if out_root.exists():
            raise RuntimeError("refusing to overwrite --out-root: %s" % out_root)
        manifest = _protocol_manifest(cases, out_root, gurobi_python,
                                      system_python)

    if args.dry_run:
        if args.phase == "replay-only":
            _verify_solutions(cases, out_root, manifest)
        solve_commands = [
            _solver_command(case, _solution_path(out_root, case),
                            gurobi_python) for case in cases]
        replay_commands = _replay(cases, out_root, manifest, system_python,
                                  args, dry_run=True)
        print(json.dumps({
            "dry_run": True,
            "phase": args.phase,
            "out_root": str(out_root),
            "available_case_count": len(all_cases),
            "selected_case_count": len(cases),
            "fixed_protocol": FIXED_PROTOCOL,
            "replay_configuration": {
                "replay_jobs": args.replay_jobs,
                "moe_gpgpusim_max_parallel_per_job":
                    args.moe_gpgpusim_max_parallel,
                "maximum_configured_parallel_gpu_simulators":
                    args.replay_jobs * args.moe_gpgpusim_max_parallel,
                "budget_timeouts_s": {
                    "runner_internal_moe_e2e": args.budget_e2e_timeout,
                    "outer_replay_legosim": args.budget_legosim_timeout,
                },
                "scaling_timeouts_s": {
                    "runner_internal_moe_e2e": args.scale_e2e_timeout,
                    "outer_replay_legosim": args.scale_legosim_timeout,
                },
            },
            "solve_commands": solve_commands,
            "replay_invocations": replay_commands,
        }, indent=2, sort_keys=True))
        return 0

    if args.phase in ("solve-only", "all"):
        out_root.mkdir(parents=True, exist_ok=False)
        _atomic_json(out_root / "manifest.json", manifest)
        _atomic_json(out_root / "artifact_hashes.json",
                     _artifact_hashes(manifest))
        _solve(cases, out_root, manifest, gurobi_python)
    if args.phase in ("replay-only", "all"):
        _replay(cases, out_root, manifest, system_python, args)

    print(json.dumps({
        "status": manifest["status"],
        "phase": args.phase,
        "selected_cases": len(cases),
        "out_root": str(out_root),
        "manifest": str(out_root / "manifest.json"),
        "artifact_hashes": str(out_root / "artifact_hashes.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        sys.exit(2)
