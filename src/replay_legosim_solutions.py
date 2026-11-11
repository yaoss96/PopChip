#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay arbitrary existing layout solutions through codegen and LEGOSim.

This driver deliberately contains no optimizer invocation.  Each repeated
``--layout`` argument names an already-produced solution as
``CASE:METHOD:PATH``.  The formal codegen and LEGOSim validation routines are
shared with ``run_budget_mesh_experiment.py`` so replay settings and PASS
criteria remain identical to the budget experiment.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import re
import sys
import uuid
from pathlib import Path

import run_budget_mesh_experiment as budget


CSV_FIELDS = [
    "case_id",
    "method",
    "status",
    "solution_path",
    "solution_sha256",
    "predicted_makespan_cycles",
    "total_area_mm2",
    "peak_power_W",
    "gpu_l_count",
    "gpu_s_count",
    "cpu_count",
    "codegen_status",
    "dispatch_formal",
    "codegen_wall_s",
    "legosim_status",
    "legosim_validation",
    "measured_makespan_cycles",
    "bench_makespan_cycles",
    "synchronized_makespan_cycles",
    "completion_delta_cycles",
    "popnet_average_delay",
    "popnet_total_power",
    "rounds_requested",
    "moe_e2e_timeout_s",
    "moe_gpgpusim_max_parallel",
    "legosim_wall_s",
    "codegen_dir",
    "legosim_run_root",
    "legosim_summary_path",
    "error",
]

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "Codegen and replay existing layout JSON files in LEGOSim; "
            "never solve or optimize a layout."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python src/replay_legosim_solutions.py ^\n"
            "    --params out/experiment/params_mixtral.json ^\n"
            "    --layout area_970:ilp:out/experiment/cases/area_970/ilp_solution.json ^\n"
            "    --layout area_970:matched-random:out/matched/area_970/representative.json ^\n"
            "    --out-dir out/replay_area --rounds 4\n\n"
            "On Windows, quote each --layout value when its path contains "
            "spaces.  The parser splits only the first two colons, so an "
            "absolute C:\\... path is accepted."))
    ap.add_argument("--params", required=True, help="existing params JSON")
    ap.add_argument(
        "--layout", action="append", required=True, metavar="CASE:METHOD:PATH",
        help="existing solution JSON; repeat for every layout")
    ap.add_argument(
        "--out-dir", required=True,
        help="new output directory; execution refuses if it already exists")
    ap.add_argument("--rounds", type=int, default=4,
                    help="LEGOSim rounds (default: 4)")
    ap.add_argument("--project-root", default=None,
                    help="project root (default: parent of this script)")
    ap.add_argument("--python", dest="system_python", default=sys.executable,
                    help="Python executable used for codegen")
    ap.add_argument(
        "--legosim-root", default=None,
        help="LEGOSIM root (default: <project>/LEGOSIM_MICRO-main/LEGOSIM_MICRO-main)")
    ap.add_argument("--wsl-exe", default="wsl.exe")
    ap.add_argument("--wsl-distro", default=None)
    ap.add_argument("--codegen-timeout", type=float, default=300.0)
    ap.add_argument("--legosim-timeout", type=float, default=7200.0)
    ap.add_argument(
        "--moe-e2e-timeout", "--e2e-timeout",
        dest="moe_e2e_timeout", type=int,
        default=budget.DEFAULT_MOE_E2E_TIMEOUT_S,
        help=("runner-internal timeout in seconds, passed explicitly as "
              "MOE_E2E_TIMEOUT (default: %(default)s; max: {})".format(
                  budget.MAX_MOE_E2E_TIMEOUT_S)))
    ap.add_argument(
        "--moe-gpgpusim-max-parallel", "--gpgpusim-max-parallel",
        dest="moe_gpgpusim_max_parallel", type=int,
        default=budget.DEFAULT_MOE_GPGPUSIM_MAX_PARALLEL,
        help=("host GPU-simulator concurrency, passed explicitly as "
              "MOE_GPGPUSIM_MAX_PARALLEL "
              "(default: %(default)s; max: {})".format(
                  budget.MAX_MOE_GPGPUSIM_MAX_PARALLEL)))
    ap.add_argument(
        "--codegen-only", action="store_true",
        help="generate and validate formal inputs without launching LEGOSim")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="validate inputs and print the plan without creating output")
    ap.add_argument("--fail-fast", action="store_true")
    return ap.parse_args(argv)


def _resolve(root, value):
    return budget._resolve_path(root, value)


def _parse_layout(raw, project_root):
    parts = str(raw).split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "--layout must be CASE:METHOD:PATH, got %r" % raw)
    case_id, method, path_text = parts
    for label, value in (("case", case_id), ("method", method)):
        if not _SAFE_COMPONENT.match(value) or value in (".", ".."):
            raise ValueError(
                "unsafe %s component in --layout: %r" % (label, value))
    path = _resolve(project_root, path_text)
    return {"case_id": case_id, "method": method, "solution_path": path}


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(str(tmp), "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(str(tmp), str(path))


def _atomic_csv(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(str(tmp), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(tmp), str(path))


def _load_json(path, label):
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            value = json.load(fh)
    except (OSError, ValueError) as ex:
        raise ValueError("cannot read %s %s: %s" % (label, path, ex))
    if not isinstance(value, dict):
        raise ValueError("%s must contain a JSON object: %s" % (label, path))
    return value


def _csv_row(record):
    metrics = record.get("solution_metrics", {})
    codegen = record.get("codegen") or {}
    legosim = record.get("legosim") or {}
    summary = legosim.get("summary") or {}
    runner_env = record.get("legosim_runner_environment") or {}
    return {
        "case_id": record["case_id"],
        "method": record["method"],
        "status": record["status"],
        "solution_path": record["solution_path"],
        "solution_sha256": record["solution_sha256"],
        "predicted_makespan_cycles": metrics.get(
            "predicted_makespan_cycles"),
        "total_area_mm2": metrics.get("total_area_mm2"),
        "peak_power_W": metrics.get("peak_power_W"),
        "gpu_l_count": metrics.get("gpu_l_count"),
        "gpu_s_count": metrics.get("gpu_s_count"),
        "cpu_count": metrics.get("cpu_count"),
        "codegen_status": codegen.get("status"),
        "dispatch_formal": codegen.get("dispatch_formal"),
        "codegen_wall_s": (codegen.get("process") or {}).get("wall_s"),
        "legosim_status": (legosim.get("status")
                            if legosim else "not_requested"),
        "legosim_validation": legosim.get("validation"),
        "measured_makespan_cycles": legosim.get(
            "measured_makespan_cycles"),
        "bench_makespan_cycles": legosim.get("bench_makespan_cycles"),
        "synchronized_makespan_cycles": legosim.get(
            "synchronized_makespan_cycles"),
        "completion_delta_cycles": legosim.get(
            "completion_delta_cycles"),
        "popnet_average_delay": summary.get("popnet_average_delay"),
        "popnet_total_power": summary.get("popnet_total_power"),
        "rounds_requested": summary.get("rounds_requested"),
        "moe_e2e_timeout_s": runner_env.get("MOE_E2E_TIMEOUT"),
        "moe_gpgpusim_max_parallel": runner_env.get(
            "MOE_GPGPUSIM_MAX_PARALLEL"),
        "legosim_wall_s": (legosim.get("process") or {}).get("wall_s"),
        "codegen_dir": codegen.get("codegen_dir"),
        "legosim_run_root": legosim.get("run_root"),
        "legosim_summary_path": legosim.get("summary_path"),
        "error": record.get("error", ""),
    }


def _write_outputs(out_dir, manifest):
    _atomic_json(Path(out_dir) / "results.json", manifest)
    _atomic_csv(Path(out_dir) / "results.csv",
                [_csv_row(item) for item in manifest["layouts"]])


def main(argv=None):
    args = _parse_args(argv)
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    # Shared LEGOSim helpers use the budget driver's internal attribute name.
    args.legosim_rounds = args.rounds
    if args.codegen_timeout <= 0 or args.legosim_timeout <= 0:
        raise ValueError("timeouts must be positive")
    if not (1 <= args.moe_e2e_timeout <= budget.MAX_MOE_E2E_TIMEOUT_S):
        raise ValueError(
            "--moe-e2e-timeout must be in [1,%d] seconds" %
            budget.MAX_MOE_E2E_TIMEOUT_S)
    if not (1 <= args.moe_gpgpusim_max_parallel <=
            budget.MAX_MOE_GPGPUSIM_MAX_PARALLEL):
        raise ValueError(
            "--moe-gpgpusim-max-parallel must be in [1,%d]" %
            budget.MAX_MOE_GPGPUSIM_MAX_PARALLEL)

    project_root = (_resolve(Path.cwd(), args.project_root)
                    if args.project_root else budget.DEFAULT_PROJECT_ROOT)
    params_path = _resolve(project_root, args.params)
    out_dir = _resolve(project_root, args.out_dir)
    legosim_root = (
        _resolve(project_root, args.legosim_root)
        if args.legosim_root else
        (project_root / "LEGOSIM_MICRO-main" / "LEGOSIM_MICRO-main").resolve())
    runner = (legosim_root / "benchmark" / "All2all-MoE-E2E" /
              "run_ilp_e2e.sh")
    legosim_runner_environment = budget._legosim_runner_environment(args)

    if out_dir.exists():
        raise RuntimeError("refusing to overwrite existing --out-dir: %s" %
                           out_dir)
    if not params_path.is_file():
        raise RuntimeError("params JSON not found: %s" % params_path)
    _load_json(params_path, "params")
    if not (project_root / "src" / "codegen.py").is_file():
        raise RuntimeError("codegen.py not found below project root: %s" %
                           project_root)
    if not args.codegen_only and not runner.is_file():
        raise RuntimeError("LEGOSim runner not found: %s" % runner)

    layouts = [_parse_layout(raw, project_root) for raw in args.layout]
    seen = set()
    for item in layouts:
        key = (item["case_id"], item["method"])
        if key in seen:
            raise ValueError("duplicate case/method: %s:%s" % key)
        seen.add(key)
        if not item["solution_path"].is_file():
            raise RuntimeError("solution JSON not found: %s" %
                               item["solution_path"])
        item["solution"] = _load_json(item["solution_path"], "solution")

    plan = {
        "project_root": str(project_root),
        "params_path": str(params_path),
        "out_dir": str(out_dir),
        "runner": str(runner),
        "rounds": args.rounds,
        "legosim_runner_environment": legosim_runner_environment,
        "codegen_only": args.codegen_only,
        "layouts": [{
            "case_id": item["case_id"],
            "method": item["method"],
            "solution_path": str(item["solution_path"]),
        } for item in layouts],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_utc": budget._utc_now(),
        "finished_utc": None,
        "project_root": str(project_root),
        "params_path": str(params_path),
        "params_sha256": budget._sha256(params_path),
        "legosim_runner": str(runner),
        "rounds": args.rounds,
        "legosim_runner_environment": legosim_runner_environment,
        "codegen_only": args.codegen_only,
        "layouts": [],
    }
    _write_outputs(out_dir, manifest)

    any_failed = False
    for item in layouts:
        method_dir = out_dir / "layouts" / item["case_id"] / item["method"]
        method_dir.mkdir(parents=True, exist_ok=False)
        record = {
            "case_id": item["case_id"],
            "method": item["method"],
            "status": "running",
            "solution_path": str(item["solution_path"]),
            "solution_sha256": budget._sha256(item["solution_path"]),
            "solution_metrics": budget._solution_metrics(item["solution"]),
            "legosim_runner_environment": legosim_runner_environment,
            "codegen": None,
            "legosim": None,
            "error": "",
        }
        try:
            codegen = budget._codegen_layout(
                args, project_root, params_path, item["solution_path"],
                method_dir / "codegen", method_dir)
            record["codegen"] = codegen
            if codegen.get("status") != "ok":
                record["status"] = "codegen_failed"
                record["error"] = codegen.get("error", "")
            elif args.codegen_only:
                record["status"] = "codegen_only_ok"
            else:
                run_root = "/tmp/ilp4moe_replay_%s" % uuid.uuid4().hex
                legosim = budget._run_legosim(
                    args, project_root, runner, method_dir / "codegen",
                    run_root, method_dir)
                record["legosim"] = legosim
                if legosim.get("status") == "ok":
                    record["status"] = "ok"
                else:
                    record["status"] = "legosim_failed"
                    record["error"] = legosim.get("error", "")
        except Exception as ex:  # preserve partial evidence and continue
            record["status"] = "failed"
            record["error"] = "%s: %s" % (type(ex).__name__, ex)

        failed = record["status"] not in ("ok", "codegen_only_ok")
        any_failed = any_failed or failed
        manifest["layouts"].append(record)
        _write_outputs(out_dir, manifest)
        if failed and args.fail_fast:
            break

    manifest["finished_utc"] = budget._utc_now()
    manifest["status"] = "failed" if any_failed else "complete"
    _write_outputs(out_dir, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "out_dir": str(out_dir),
        "results_json": str(out_dir / "results.json"),
        "results_csv": str(out_dir / "results.csv"),
        "layouts_completed": len(manifest["layouts"]),
    }, indent=2, sort_keys=True))
    return 1 if any_failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as ex:
        print("error: %s" % ex, file=sys.stderr)
        sys.exit(2)
