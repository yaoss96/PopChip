#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run exact-BOM prior-art-objective proxies on an existing Mesh sweep.

The source experiment contains eight budget cases.  Even where an area case
and a power case share the same chiplet BOM, their source ILP solutions may
attach a different complete task bundle to a chiplet.  For a strict
layout-only comparison, this driver therefore generates and evaluates both
proxies independently for every case.  No expert or other task is reassigned.

The proxy generator's claim boundary is retained throughout the outputs:
these are Floorplet-inspired and TAP-2.5D-inspired proxy layouts,
not executions of the original authors' tools.
"""

from __future__ import print_function

import argparse
import collections
import copy
import csv
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import prior_layout_baseline as prior
import run_budget_mesh_experiment as budget


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE = (
    "out/budget_mesh_experiment/newrepo_mesh_budget_full_v2_20260731")
METHODS = ("floorplet-inspired", "tap-2.5d-inspired")
CLAIM_BOUNDARY = (
    "Prior-art-inspired exact-BOM proxies on the repository's discrete Mesh; "
    "not executions of the original Floorplet or TAP-2.5D tools.")

CSV_FIELDS = [
    "experiment_id",
    "case_id",
    "sweep_dimension",
    "budget_value",
    "a_max_mm2",
    "p_max_W",
    "p_col_W",
    "layout_id",
    "bom_id",
    "representative_case_id",
    "method",
    "status",
    "predicted_makespan_cycles",
    "measured_makespan_cycles",
    "legosim_validation",
    "popnet_average_delay",
    "popnet_total_power",
    "ilp_speedup_vs_method_predicted",
    "ilp_speedup_vs_method_measured",
    "total_area_mm2",
    "peak_power_W_model",
    "candidate_pool_size",
    "seed_offset",
    "selected_seed",
    "proxy_search_wall_s",
    "traffic_weighted_wirelength",
    "unweighted_net_wirelength",
    "thermal_coupling_surrogate",
    "selection_cost",
    "codegen_status",
    "dispatch_formal",
    "legosim_status",
    "legosim_wall_s",
    "solution_path",
    "codegen_dir",
    "legosim_run_root",
    "legosim_summary_path",
    "layout_coords_json",
    "source_case_dir",
    "error",
]


def _utc_now():
    return datetime.datetime.now(
        datetime.timezone.utc).replace(
            microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path):
    with open(str(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(str(temporary), "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_csv(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(str(temporary), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in CSV_FIELDS})
    os.replace(str(temporary), str(path))


def _atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(str(temporary), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(value)
    os.replace(str(temporary), str(path))


def _resolve(base, value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def _new_output_dir(out_root, run_id):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if run_id:
        if not re.match(r"^[A-Za-z0-9_.-]+$", run_id):
            raise ValueError(
                "--run-id may contain only letters, digits, '.', '_' and '-'")
        name = run_id
    else:
        stamp = datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = "prior_mesh_%s_p%d_%s" % (
            stamp, os.getpid(), uuid.uuid4().hex[:8])
    output = out_root / name
    output.mkdir(parents=False, exist_ok=False)
    return output.resolve()


def _bom_counts(solution):
    counts = collections.Counter(
        str(site["type"]) for site in solution.get("sites", []))
    return {name: counts[name] for name in sorted(counts)}


def _bom_key(counts):
    return json.dumps(counts, sort_keys=True, separators=(",", ":"))


def _source_ilp_layout(case):
    for layout in case.get("layouts", []):
        if layout.get("method") == "ilp":
            return layout
    return None


def _legosim_values(record):
    if not isinstance(record, dict):
        return {
            "status": "not_available",
            "validation": None,
            "measured_makespan_cycles": None,
            "popnet_average_delay": None,
            "popnet_total_power": None,
            "wall_s": None,
            "run_root": None,
            "summary_path": None,
            "error": "",
        }
    process = record.get("process") or {}
    summary = record.get("summary") or {}
    return {
        "status": record.get("status"),
        "validation": record.get("validation"),
        "measured_makespan_cycles": record.get(
            "measured_makespan_cycles"),
        "popnet_average_delay": summary.get("popnet_average_delay"),
        "popnet_total_power": summary.get("popnet_total_power"),
        "wall_s": process.get("wall_s"),
        "run_root": record.get("run_root"),
        "summary_path": record.get("summary_path"),
        "error": record.get("error", ""),
    }


def _layout_coordinates(solution):
    return [{
        "type": site.get("type"),
        "l": site.get("l"),
        "i": site.get("i"),
        "j": site.get("j"),
        "tasks": list(site.get("tasks") or []),
    } for site in solution.get("sites", [])]


def _call_with_heartbeat(label, function):
    """Run a blocking stage while emitting a progress line every 45 seconds."""
    stopped = threading.Event()
    started = time.monotonic()

    def heartbeat():
        while not stopped.wait(45.0):
            print("[wait] %s elapsed=%.1fs" % (
                label, time.monotonic() - started), flush=True)

    worker = threading.Thread(target=heartbeat)
    worker.daemon = True
    worker.start()
    print("[stage] %s" % label, flush=True)
    try:
        return function()
    finally:
        stopped.set()
        worker.join(timeout=1.0)
        print("[stage] %s done elapsed=%.1fs" % (
            label, time.monotonic() - started), flush=True)


def _load_source(source_dir):
    raw_path = Path(source_dir) / "raw_results.json"
    params_path = Path(source_dir) / "params_mixtral.json"
    if not raw_path.exists():
        raise ValueError("source raw_results.json not found: %s" % raw_path)
    if not params_path.exists():
        raise ValueError("source params_mixtral.json not found: %s" %
                         params_path)
    raw = _read_json(raw_path)
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("source experiment has no cases")

    case_records = []
    bom_classes = collections.OrderedDict()
    for case in cases:
        case_id = case.get("case_id")
        source_solution = (
            Path(source_dir) / "cases" / str(case_id) / "ilp_solution.json")
        if not source_solution.exists():
            candidate = case.get("ilp", {}).get("solution_path")
            if candidate:
                source_solution = Path(candidate)
        if not source_solution.exists():
            raise ValueError("ILP solution not found for %s: %s" %
                             (case_id, source_solution))
        solution = _read_json(source_solution)
        counts = _bom_counts(solution)
        key = _bom_key(counts)
        entry = {
            "case": case,
            "solution": solution,
            "solution_path": source_solution.resolve(),
            "bom_key": key,
            "bom_counts": counts,
        }
        case_records.append(entry)
        if key not in bom_classes:
            bom_classes[key] = "bom_%02d" % (len(bom_classes) + 1)

    groups = []
    for index, entry in enumerate(case_records, 1):
        group = {
            "counts": entry["bom_counts"],
            "representative": entry,
            "members": [entry],
            "bom_id": bom_classes[entry["bom_key"]],
            "layout_id": "layout_%02d_%s" % (
                index, entry["case"]["case_id"]),
        }
        compact = "_".join(
            "%s%d" % (name.lower().replace("-", ""), count)
            for name, count in sorted(group["counts"].items()))
        group["bom_id"] = "%s_%s" % (group["bom_id"], compact)
        groups.append(group)
    return raw, params_path.resolve(), case_records, groups


def _proxy_record(args, project_root, params_path, output_dir, group,
                  runner):
    bom_dir = Path(output_dir) / "layouts" / group["layout_id"]
    proxy_dir = bom_dir / "proxy_search"
    bom_dir.mkdir(parents=True, exist_ok=False)
    representative = group["representative"]
    started = time.monotonic()
    summary = _call_with_heartbeat(
        "%s proxy-search" % group["layout_id"],
        lambda: prior.run(
            str(params_path),
            str(representative["solution_path"]),
            str(proxy_dir),
            candidates=args.candidates,
            seed_offset=args.seed_offset,
            tap_alpha=args.tap_alpha))
    search_wall = time.monotonic() - started

    record = {
        "layout_id": group["layout_id"],
        "bom_id": group["bom_id"],
        "chiplet_type_counts": copy.deepcopy(group["counts"]),
        "representative_case_id":
            representative["case"]["case_id"],
        "mapped_case_ids": [
            member["case"]["case_id"] for member in group["members"]],
        "complete_task_bundles_preserved": True,
        "source_ilp_solution": str(
            representative["solution_path"]),
        "source_ilp_solution_sha256": _sha256(
            representative["solution_path"]),
        "proxy_search_summary_path": str(
            (proxy_dir / "summary.json").resolve()),
        "proxy_search": summary,
        "search_wall_s_driver": search_wall,
        "layouts": {},
    }

    for method in METHODS:
        method_dir = bom_dir / method
        method_dir.mkdir(parents=True, exist_ok=False)
        output = summary["outputs"][method]
        solution_path = Path(output["path"]).resolve()
        solution = _read_json(solution_path)
        codegen = None
        legosim = None
        if not args.skip_codegen:
            codegen_dir = method_dir / "codegen"
            codegen = _call_with_heartbeat(
                "%s %s codegen" % (group["layout_id"], method),
                lambda: budget._codegen_layout(
                    args, project_root, params_path, solution_path,
                    codegen_dir, method_dir))
            if args.run_legosim and codegen.get("status") == "ok":
                run_root = "/tmp/ilp4moe_prior_mesh_%s" % uuid.uuid4().hex
                legosim = _call_with_heartbeat(
                    "%s %s LEGOSim" % (group["layout_id"], method),
                    lambda: budget._run_legosim(
                        args, project_root, runner, codegen_dir,
                        run_root, method_dir))
        record["layouts"][method] = {
            "method": method,
            "solution_path": str(solution_path),
            "solution_sha256": _sha256(solution_path),
            "metrics": budget._solution_metrics(solution),
            "objective": copy.deepcopy(
                solution.get("baseline", {}).get("objective", {})),
            "layout_coordinates": _layout_coordinates(solution),
            "selected_seed": output["seed"],
            "strict_feasible": output["strict_feasible"],
            "codegen": codegen,
            "legosim": legosim,
        }
    return record


def _rows(raw, output_dir):
    groups = {
        group["layout_id"]: group
        for group in raw.get("layout_groups", [])
    }
    case_map = raw.get("case_to_layout", {})
    source_cases = {
        case.get("case_id"): case
        for case in raw.get("source_cases", [])
    }
    rows = []
    for case_id, layout_id in case_map.items():
        source_case = source_cases[case_id]
        budgets = source_case.get("budgets") or {}
        group = groups.get(layout_id)
        ilp = source_case.get("ilp") or {}
        ilp_metrics = ilp.get("metrics") or {}
        ilp_layout = _source_ilp_layout(source_case) or {}
        ilp_sim = _legosim_values(ilp_layout.get("legosim"))
        ilp_status = (
            "ok" if ilp.get("status") == "ok" and
            ilp_sim["status"] in ("ok", "not_available") else
            (ilp.get("status") or "unknown"))
        common = {
            "experiment_id": Path(output_dir).name,
            "case_id": case_id,
            "sweep_dimension": source_case.get("sweep_dimension"),
            "budget_value": source_case.get("budget_value"),
            "a_max_mm2": budgets.get("a_max"),
            "p_max_W": budgets.get("p_max"),
            "p_col_W": budgets.get("p_col"),
            "layout_id": layout_id,
            "bom_id": group.get("bom_id") if group else None,
            "representative_case_id": (
                group.get("representative_case_id") if group else None),
            "source_case_dir": source_case.get("case_dir"),
        }
        ilp_row = dict(common)
        ilp_row.update({
            "method": "ilp",
            "status": ilp_status,
            "predicted_makespan_cycles": ilp_metrics.get(
                "predicted_makespan_cycles"),
            "measured_makespan_cycles":
                ilp_sim["measured_makespan_cycles"],
            "legosim_validation": ilp_sim["validation"],
            "popnet_average_delay": ilp_sim["popnet_average_delay"],
            "popnet_total_power": ilp_sim["popnet_total_power"],
            "total_area_mm2": ilp_metrics.get("total_area_mm2"),
            "peak_power_W_model": ilp_metrics.get("peak_power_W"),
            "solution_path": ilp.get("solution_path"),
            "codegen_status": (
                ilp_layout.get("codegen", {}).get("status")),
            "dispatch_formal": (
                ilp_layout.get("codegen", {}).get("dispatch_formal")),
            "legosim_status": ilp_sim["status"],
            "legosim_wall_s": ilp_sim["wall_s"],
            "legosim_run_root": ilp_sim["run_root"],
            "legosim_summary_path": ilp_sim["summary_path"],
            "layout_coords_json": json.dumps(
                _layout_coordinates(_read_json(ilp["solution_path"])),
                sort_keys=True, separators=(",", ":"))
            if ilp.get("solution_path") and
            Path(ilp["solution_path"]).exists() else None,
            "error": ilp.get("error", "") or ilp_sim["error"],
        })
        rows.append(ilp_row)

        if not group:
            continue
        for method in METHODS:
            layout = group.get("layouts", {}).get(method)
            if not layout:
                continue
            metrics = layout.get("metrics") or {}
            objective = layout.get("objective") or {}
            codegen = layout.get("codegen") or {}
            sim = _legosim_values(layout.get("legosim"))
            if raw["experiment"]["skip_codegen"]:
                status = "generated"
            elif codegen.get("status") != "ok":
                status = "codegen_failed"
            elif raw["experiment"]["run_legosim"] and sim["status"] != "ok":
                status = "legosim_failed"
            else:
                status = "ok"
            row = dict(common)
            row.update({
                "method": method,
                "status": status,
                "predicted_makespan_cycles": metrics.get(
                    "predicted_makespan_cycles"),
                "measured_makespan_cycles":
                    sim["measured_makespan_cycles"],
                "legosim_validation": sim["validation"],
                "popnet_average_delay": sim["popnet_average_delay"],
                "popnet_total_power": sim["popnet_total_power"],
                "total_area_mm2": metrics.get("total_area_mm2"),
                "peak_power_W_model": metrics.get("peak_power_W"),
                "candidate_pool_size":
                    raw["experiment"]["candidate_pool_size"],
                "seed_offset": raw["experiment"]["seed_offset"],
                "selected_seed": layout.get("selected_seed"),
                "proxy_search_wall_s":
                    group.get("proxy_search", {}).get("search_wall_s"),
                "traffic_weighted_wirelength": objective.get(
                    "traffic_weighted_wirelength"),
                "unweighted_net_wirelength": objective.get(
                    "unweighted_net_wirelength"),
                "thermal_coupling_surrogate": objective.get(
                    "thermal_coupling_surrogate"),
                "selection_cost": objective.get("selection_cost"),
                "codegen_status": (
                    codegen.get("status") if codegen else "not_requested"),
                "dispatch_formal": codegen.get("dispatch_formal"),
                "legosim_status": sim["status"],
                "legosim_wall_s": sim["wall_s"],
                "solution_path": layout.get("solution_path"),
                "codegen_dir": codegen.get("codegen_dir"),
                "legosim_run_root": sim["run_root"],
                "legosim_summary_path": sim["summary_path"],
                "layout_coords_json": json.dumps(
                    layout.get("layout_coordinates") or [],
                    sort_keys=True, separators=(",", ":")),
                "error": (codegen.get("error", "") or sim["error"]),
            })
            ilp_predicted = ilp_row["predicted_makespan_cycles"]
            proxy_predicted = row["predicted_makespan_cycles"]
            if (isinstance(ilp_predicted, (int, float)) and
                    ilp_predicted > 0 and
                    isinstance(proxy_predicted, (int, float))):
                row["ilp_speedup_vs_method_predicted"] = (
                    proxy_predicted / ilp_predicted)
            ilp_measured = ilp_row["measured_makespan_cycles"]
            proxy_measured = row["measured_makespan_cycles"]
            if (isinstance(ilp_measured, (int, float)) and
                    ilp_measured > 0 and
                    isinstance(proxy_measured, (int, float))):
                row["ilp_speedup_vs_method_measured"] = (
                    proxy_measured / ilp_measured)
            rows.append(row)
    return rows


def _fmt_number(value):
    if value is None or value == "":
        return "--"
    if isinstance(value, float) and not value.is_integer():
        return "%.3f" % value
    if isinstance(value, (int, float)):
        return str(int(value))
    return str(value)


def _write_summaries(output_dir, raw, rows):
    summary = {
        "schema_version": 1,
        "status": raw["status"],
        "claim_boundary": CLAIM_BOUNDARY,
        "source_experiment": raw["experiment"]["source_experiment"],
        "candidate_pool_size":
            raw["experiment"]["candidate_pool_size"],
        "seed_offset": raw["experiment"]["seed_offset"],
        "tap_alpha": raw["experiment"]["tap_alpha"],
        "unique_bom_count": len(set(
            group["bom_id"] for group in raw.get("layout_groups", []))),
        "layout_instance_count": len(raw.get("layout_groups", [])),
        "budget_case_count": len(raw.get("case_to_layout", {})),
        "simulator_runs_requested": (
            len(raw.get("layout_groups", [])) * len(METHODS)
            if raw["experiment"]["run_legosim"] else 0),
        "rows": rows,
    }
    _atomic_json(Path(output_dir) / "summary.json", summary)

    lines = [
        "# Prior-art-inspired Mesh baseline summary",
        "",
        CLAIM_BOUNDARY,
        "",
        "- Source experiment: `%s`" %
        raw["experiment"]["source_experiment"],
        "- Unique BOMs / independent layouts / budget cases: %d / %d / %d" % (
            len(set(group["bom_id"]
                    for group in raw.get("layout_groups", []))),
            len(raw.get("layout_groups", [])),
            len(raw.get("case_to_layout", {}))),
        "- Candidate pool: %d per case; seed offset: %d; TAP alpha: %.3g" % (
            raw["experiment"]["candidate_pool_size"],
            raw["experiment"]["seed_offset"],
            raw["experiment"]["tap_alpha"]),
        "- Real simulator requested: %s" %
        ("yes" if raw["experiment"]["run_legosim"] else "no"),
        "",
        "| Case | Method | Area (mm^2) | Power model (W) | "
        "Predicted cycles | Measured cycles | ILP speedup (measured) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |" % (
                row["case_id"],
                row["method"],
                _fmt_number(row.get("total_area_mm2")),
                _fmt_number(row.get("peak_power_W_model")),
                _fmt_number(row.get("predicted_makespan_cycles")),
                _fmt_number(row.get("measured_makespan_cycles")),
                _fmt_number(row.get(
                    "ilp_speedup_vs_method_measured"))))
    lines.extend([
        "",
        "Every budget case is searched and simulated independently. Complete "
        "per-chiplet task bundles are fixed; only whole bundles move.",
        "",
    ])
    _atomic_text(Path(output_dir) / "summary.md", "\n".join(lines))


def _persist(output_dir, raw):
    rows = _rows(raw, output_dir)
    _atomic_json(Path(output_dir) / "raw_results.json", raw)
    _atomic_csv(Path(output_dir) / "raw_results.csv", rows)
    _write_summaries(output_dir, raw, rows)
    return rows


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run exact-BOM prior-art-objective Mesh baselines")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--source-experiment", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--out-root", default="out/prior_mesh_baseline_experiment")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--candidates", type=int, default=20000,
        help="deterministic candidate layouts per budget case")
    parser.add_argument("--seed-offset", type=int, default=100000)
    parser.add_argument("--tap-alpha", type=float, default=0.7)
    parser.add_argument(
        "--python", dest="system_python", default=sys.executable)
    parser.add_argument("--codegen-timeout", type=float, default=300.0)
    parser.add_argument(
        "--skip-codegen", action="store_true",
        help="generate proxies only (intended for lightweight tests)")
    parser.add_argument("--run-legosim", action="store_true")
    parser.add_argument("--legosim-root", default=None)
    parser.add_argument("--wsl-exe", default="wsl.exe")
    parser.add_argument("--wsl-distro", default=None)
    parser.add_argument("--legosim-timeout", type=float, default=7200.0)
    parser.add_argument("--legosim-rounds", type=int, default=2)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args):
    if args.candidates < 2:
        raise ValueError("--candidates must be at least 2")
    if not 0.0 <= args.tap_alpha <= 1.0:
        raise ValueError("--tap-alpha must be in [0,1]")
    if args.codegen_timeout <= 0:
        raise ValueError("--codegen-timeout must be positive")
    if args.legosim_timeout <= 0:
        raise ValueError("--legosim-timeout must be positive")
    if args.legosim_rounds <= 0:
        raise ValueError("--legosim-rounds must be positive")
    if args.run_legosim and args.skip_codegen:
        raise ValueError("--run-legosim cannot be combined with --skip-codegen")


def main(argv=None):
    args = _parse_args(argv)
    _validate_args(args)
    project_root = (
        _resolve(Path.cwd(), args.project_root)
        if args.project_root else DEFAULT_PROJECT_ROOT)
    source_dir = _resolve(project_root, args.source_experiment)
    out_root = _resolve(project_root, args.out_root)
    source_raw, source_params, case_records, groups = _load_source(
        source_dir)

    legosim_root = (
        _resolve(project_root, args.legosim_root)
        if args.legosim_root else
        (project_root / "LEGOSIM_MICRO-main" /
         "LEGOSIM_MICRO-main").resolve())
    runner = (
        legosim_root / "benchmark" / "All2all-MoE-E2E" /
        "run_ilp_e2e.sh")
    plan = {
        "project_root": str(project_root),
        "source_experiment": str(source_dir),
        "output_root": str(out_root),
        "candidate_pool_size": args.candidates,
        "seed_offset": args.seed_offset,
        "tap_alpha": args.tap_alpha,
        "unique_bom_count": len(set(
            group["bom_id"] for group in groups)),
        "layout_instance_count": len(groups),
        "budget_case_count": len(case_records),
        "methods": list(METHODS),
        "run_legosim": args.run_legosim,
        "planned_simulator_runs": (
            len(groups) * len(METHODS) if args.run_legosim else 0),
        "groups": [{
            "layout_id": group["layout_id"],
            "bom_id": group["bom_id"],
            "counts": group["counts"],
            "representative_case_id":
                group["representative"]["case"]["case_id"],
            "mapped_case_ids": [
                member["case"]["case_id"] for member in group["members"]],
        } for group in groups],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    required = [
        project_root / "src" / "prior_layout_baseline.py",
        project_root / "src" / "matched_layout_baseline.py",
        project_root / "src" / "run_budget_mesh_experiment.py",
        project_root / "src" / "codegen.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing required source files: %s" %
                           ", ".join(missing))
    system_python = budget._normalize_executable(args.system_python)
    if not budget._command_available(system_python):
        raise RuntimeError("system Python not found: %s" % system_python)
    args.system_python = system_python
    if args.run_legosim:
        if not runner.exists():
            raise RuntimeError("LEGOSim runner not found: %s" % runner)
        if os.name == "nt" and not budget._command_available(
                budget._normalize_executable(args.wsl_exe)):
            raise RuntimeError("wsl.exe not found: %s" % args.wsl_exe)

    output_dir = _new_output_dir(out_root, args.run_id)
    params_path = output_dir / "params_mixtral.json"
    shutil.copy2(str(source_params), str(params_path))
    case_to_layout = {}
    for group in groups:
        for member in group["members"]:
            case_to_layout[
                member["case"]["case_id"]] = group["layout_id"]
    raw = {
        "schema_version": 1,
        "status": "running",
        "started_utc": _utc_now(),
        "finished_utc": None,
        "claim_boundary": CLAIM_BOUNDARY,
        "experiment": {
            "experiment_id": output_dir.name,
            "experiment_dir": str(output_dir),
            "project_root": str(project_root),
            "source_experiment": str(source_dir),
            "source_raw_results": str(
                (source_dir / "raw_results.json").resolve()),
            "source_raw_results_sha256": _sha256(
                source_dir / "raw_results.json"),
            "topology": "mesh",
            "grid": copy.deepcopy(
                source_raw.get("experiment", {}).get(
                    "grid", {"L": 1, "X": 6, "Y": 6})),
            "methods": list(METHODS),
            "original_tools_executed": False,
            "complete_task_bundles_preserved": True,
            "candidate_pool_size": args.candidates,
            "seed_offset": args.seed_offset,
            "tap_alpha": args.tap_alpha,
            "skip_codegen": args.skip_codegen,
            "run_legosim": args.run_legosim,
            "legosim_root": str(legosim_root),
            "legosim_rounds": args.legosim_rounds,
            "deduplicated_by_exact_bom": False,
            "unique_bom_count": len(set(
                group["bom_id"] for group in groups)),
            "layout_instance_count": len(groups),
            "budget_case_count": len(case_records),
            "system_python": system_python,
        },
        "host": {
            "platform": platform.platform(),
            "os_name": os.name,
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "sources": {
            path.name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            } for path in required
        },
        "params": {
            "path": str(params_path.resolve()),
            "sha256": _sha256(params_path),
            "source_path": str(source_params),
        },
        "source_cases": copy.deepcopy(source_raw["cases"]),
        "case_to_layout": case_to_layout,
        "layout_groups": [],
        "errors": [],
    }
    _persist(output_dir, raw)

    failed = False
    for group in groups:
        print("[run] %-24s case=%s bom=%s" % (
            group["layout_id"],
            group["representative"]["case"]["case_id"],
            group["bom_id"]),
              flush=True)
        try:
            result = _proxy_record(
                args, project_root, params_path, output_dir, group, runner)
            raw["layout_groups"].append(result)
            for method in METHODS:
                layout = result["layouts"][method]
                codegen = layout.get("codegen")
                sim = layout.get("legosim")
                if (codegen and codegen.get("status") != "ok") or (
                        args.run_legosim and
                        (not sim or sim.get("status") != "ok")):
                    failed = True
        except (OSError, RuntimeError, ValueError) as exc:
            failed = True
            raw["errors"].append({
                "layout_id": group["layout_id"],
                "bom_id": group["bom_id"],
                "error": "%s: %s" % (type(exc).__name__, exc),
            })
            if args.fail_fast:
                raw["status"] = "running_with_errors"
                _persist(output_dir, raw)
                break
        raw["status"] = "running_with_errors" if failed else "running"
        _persist(output_dir, raw)

    raw["status"] = "complete_with_errors" if failed else "complete"
    raw["finished_utc"] = _utc_now()
    rows = _persist(output_dir, raw)
    print("[done] status=%s, rows=%d" % (raw["status"], len(rows)))
    print("[done] JSON   : %s" % (output_dir / "raw_results.json"))
    print("[done] CSV    : %s" % (output_dir / "raw_results.csv"))
    print("[done] summary: %s" % (output_dir / "summary.md"))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        sys.exit(2)
