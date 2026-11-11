#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible Mixtral budget sweep on the default LEGOSim 6x6 mesh.

The experiment is deliberately split into two one-dimensional sweeps:

* area budgets 970/1260/1550/2130 mm^2 with package power relaxed to 1625 W;
* power budgets 785/995/1205/1625 W with area relaxed to 2130 mm^2.

Every point solves the same L=1, X=6, Y=6 placement with
``ilp_layout.py --mode certified-latency``.  The resulting layout is always
passed through codegen with a formal deterministic dispatch trace
(32 tokens/source, hidden=128, intermediate=64, repeat=1).  Real LEGOSim is
opt-in via ``--run-legosim`` because it is substantially more expensive than
the ILP solve.

The driver runs under native Windows or WSL/Linux.  On Windows, LEGOSim is
entered through ``wsl.exe``; under WSL/Linux, its bash runner is called
directly.  Windows Gurobi Python 3.7 is configurable with
``--gurobi-python`` (or ``GUROBI_PYTHON``).

Outputs are written to a new, unique experiment directory below
``out/budget_mesh_experiment`` by default.  ``raw_results.json`` is an
incremental provenance-rich record, and ``raw_results.csv`` is the flat table
intended for plotting.  No previous experiment directory or LEGOSim run root
is reused.
"""

from __future__ import print_function

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_AREA_BUDGETS = [970.0, 1260.0, 1550.0, 2130.0]
DEFAULT_POWER_BUDGETS = [785.0, 995.0, 1205.0, 1625.0]
GRID = {"L": 1, "X": 6, "Y": 6}
CERTIFIED_MODE = "certified-latency"
# Simulated NoC/compute scale.  Defaults are the downscaled smoke values GPGPU-Sim
# runs fastest on; env overrides dial toward production (real d_model=4096) so
# inter-chiplet all-to-all traffic becomes a meaningful fraction of makespan and
# the layout (hop distance / link load) actually matters.  The SAME scale applies
# to every method, so speedup ratios stay a fair apples-to-apples comparison.
DISPATCH_CONFIG = {
    "tokens_per_source": int(os.environ.get("MOE_SIM_TOKENS", "32")),
    "sim_hidden": int(os.environ.get("MOE_SIM_HIDDEN", "128")),
    "sim_intermediate": int(os.environ.get("MOE_SIM_INTER", "64")),
    "sim_ffn_repeat": int(os.environ.get("MOE_SIM_FFN_REPEAT", "1")),
}
DEFAULT_MOE_E2E_TIMEOUT_S = 1800
MAX_MOE_E2E_TIMEOUT_S = 7 * 24 * 60 * 60
DEFAULT_MOE_GPGPUSIM_MAX_PARALLEL = 16
MAX_MOE_GPGPUSIM_MAX_PARALLEL = 128

CSV_FIELDS = [
    "experiment_id",
    "case_id",
    "workload",
    "sweep_dimension",
    "budget_value",
    "method",
    "mode",
    "L",
    "X",
    "Y",
    "a_max_mm2",
    "p_max_W",
    "p_col_W",
    "status",
    "returncode",
    "timed_out",
    "predicted_makespan_cycles",
    "measured_makespan_cycles",
    "legosim_validation",
    "mip_gap",
    "latency_lower_bound_cycles",
    "latency_upper_bound_cycles",
    "certificate_status",
    "solver_status",
    "primary_status",
    "secondary_status",
    "solver_wall_s",
    "model_build_s",
    "primary_wall_s",
    "secondary_wall_s",
    "secondary_mip_gap",
    "time_to_first_feasible_s",
    "primary_node_count",
    "orchestrator_solve_wall_s",
    "total_area_mm2",
    "area_slack_mm2",
    "peak_power_W",
    "power_slack_W",
    "energy_total_J",
    "energy_comm_J",
    "byte_hop",
    "occupied_sites",
    "gpu_workers",
    "gpu_l_count",
    "gpu_s_count",
    "cpu_count",
    "codegen_status",
    "dispatch_formal",
    "legosim_status",
    "legosim_wall_s",
    "ilp_speedup_vs_method_predicted",
    "ilp_speedup_vs_method_measured",
    "method_search_time_over_ilp",
    "solution_path",
    "codegen_dir",
    "legosim_run_root",
    "legosim_summary_path",
    "error",
]


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


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
    with open(str(tmp), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})
    os.replace(str(tmp), str(path))


def _tail(path, limit=4000):
    try:
        with open(str(path), "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - limit), os.SEEK_SET)
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _run_logged(command, cwd, stdout_path, stderr_path, timeout=None,
                env=None):
    """Run without a shell, write complete logs, and return process metadata."""
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started_utc = _utc_now()
    started = time.monotonic()
    timed_out = False
    returncode = None
    error = ""
    try:
        with open(str(stdout_path), "w", encoding="utf-8", errors="replace") as out:
            with open(str(stderr_path), "w", encoding="utf-8",
                      errors="replace") as err:
                proc = subprocess.run(
                    [str(part) for part in command],
                    cwd=str(cwd),
                    stdout=out,
                    stderr=err,
                    env=env,
                    timeout=timeout,
                    universal_newlines=True,
                    check=False,
                )
                returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        error = "process timeout after %.1f seconds" % float(timeout)
    except OSError as ex:
        error = "%s: %s" % (type(ex).__name__, ex)
    wall = time.monotonic() - started
    return {
        "command": [str(part) for part in command],
        "cwd": str(Path(cwd).resolve()),
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "wall_s": round(wall, 6),
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
        "error": error,
    }


def _git_provenance(root):
    result = {"commit": None, "dirty": None}
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True, check=False)
        if sha.returncode != 0:
            return result
        result["commit"] = sha.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True, check=False)
        result["dirty"] = bool(dirty.stdout.strip())
    except OSError:
        pass
    return result


def _resolve_path(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    return path.resolve()


_WINDOWS_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def _windows_to_wsl(value):
    raw = str(value)
    match = _WINDOWS_PATH.match(raw)
    if not match:
        if raw.startswith("/"):
            return raw
        raise ValueError("cannot translate non-drive Windows path to WSL: %s"
                         % raw)
    drive, rest = match.groups()
    return "/mnt/%s/%s" % (drive.lower(), rest.replace("\\", "/"))


def _wsl_to_windows(value):
    raw = str(value).replace("\\", "/")
    # On native Windows, Path("/mnt/c/...").resolve() would incorrectly
    # prepend the current drive before we get a chance to recognize it.
    if not raw.startswith("/mnt/"):
        raw = str(Path(raw).resolve()).replace("\\", "/")
    match = re.match(r"^/mnt/([A-Za-z])(?:/(.*))?$", raw)
    if match:
        drive, rest = match.groups()
        rest = (rest or "").replace("/", "\\")
        return "%s:\\%s" % (drive.upper(), rest)
    try:
        proc = subprocess.run(
            ["wslpath", "-w", raw],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except OSError:
        pass
    raise ValueError(
        "Windows python.exe cannot access POSIX path %s; keep the experiment "
        "under /mnt/<drive> or supply a Linux Gurobi interpreter" % raw)


def _normalize_executable(value):
    raw = str(value)
    if os.name == "nt" and raw.startswith("/mnt/"):
        return _wsl_to_windows(raw)
    if os.name != "nt" and _WINDOWS_PATH.match(raw):
        return _windows_to_wsl(raw)
    return raw


def _interpreter_path(executable, path):
    """Format a file argument for the selected Python interpreter."""
    path = Path(path).resolve()
    if os.name != "nt" and str(executable).lower().endswith(".exe"):
        return _wsl_to_windows(path)
    return str(path)


def _external_path(value, base):
    """Turn a path emitted by a child interpreter into a host-visible Path."""
    raw = str(value)
    if os.name != "nt" and _WINDOWS_PATH.match(raw):
        path = Path(_windows_to_wsl(raw))
    elif os.name == "nt" and raw.startswith("/mnt/"):
        path = Path(_wsl_to_windows(raw))
    else:
        path = Path(raw)
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def _command_available(command):
    raw = str(command)
    if os.path.isabs(raw) or "/" in raw or "\\" in raw:
        return Path(raw).exists()
    return shutil.which(raw) is not None


def _number(value):
    if value is None or isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if re.match(r"^[+-]?\d+$", text):
        try:
            return int(text)
        except ValueError:
            return value
    try:
        return float(text)
    except ValueError:
        return value


def _parse_summary(path):
    parsed = {}
    with open(str(path), "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            parsed[key.strip()] = _number(value.strip())
    return parsed


def _first(mapping, names, default=None):
    for name in names:
        if isinstance(mapping, dict) and mapping.get(name) is not None:
            return mapping[name]
    return default


def _solution_metrics(solution):
    certificate = solution.get("latency_certificate", {})
    if not isinstance(certificate, dict):
        certificate = {}
    sites = solution.get("sites", [])
    counts = {}
    for site in sites:
        kind = str(site.get("type", ""))
        counts[kind] = counts.get(kind, 0) + 1
    predicted = _first(
        solution,
        ["predicted_makespan_cycles", "makespan_cycles",
         "latency_upper_bound_cycles"])
    upper = _first(
        certificate,
        ["upper_bound_cycles", "incumbent_cycles",
         "latency_upper_bound_cycles"],
        _first(solution, ["primary_incumbent_cycles",
                          "latency_upper_bound_cycles"], predicted))
    lower = _first(
        certificate,
        ["lower_bound_cycles", "best_bound_cycles",
         "latency_lower_bound_cycles"],
        _first(solution, ["primary_objective_bound_cycles",
                          "latency_lower_bound_cycles",
                          "best_bound_cycles"]))
    gap = _first(
        certificate,
        ["relative_gap", "mip_gap", "latency_mip_gap"],
        _first(solution, ["primary_mip_gap", "latency_mip_gap", "mip_gap"]))
    cert_status = _first(
        certificate,
        ["status", "certificate_status", "certified"],
        _first(solution, ["certificate_status", "certified_latency"]))
    return {
        "predicted_makespan_cycles": predicted,
        "mip_gap": gap,
        "latency_lower_bound_cycles": lower,
        "latency_upper_bound_cycles": upper,
        "certificate_status": cert_status,
        "solver_status": solution.get("solver_status"),
        "primary_status": solution.get("primary_status"),
        "secondary_status": solution.get("secondary_status"),
        "solver_wall_s": solution.get("solver_wall_s"),
        "model_build_s": solution.get("model_build_s"),
        "primary_wall_s": solution.get("primary_wall_s"),
        "secondary_wall_s": solution.get("secondary_wall_s"),
        "secondary_mip_gap": solution.get("secondary_mip_gap"),
        "time_to_first_feasible_s": solution.get(
            "time_to_first_feasible_s"),
        "primary_node_count": solution.get("primary_node_count"),
        "total_area_mm2": solution.get("total_area_mm2"),
        "peak_power_W": solution.get("peak_power_W"),
        "energy_total_J": solution.get("energy_total_J"),
        "energy_comm_J": solution.get("energy_comm_J"),
        "byte_hop": solution.get("byte_hop"),
        "occupied_sites": len(sites),
        "gpu_workers": sum(value for key, value in counts.items()
                           if key.upper().startswith("GPU")),
        "gpu_l_count": counts.get("GPU-L", 0),
        "gpu_s_count": counts.get("GPU-S", 0),
        "cpu_count": counts.get("CPU", 0),
    }


def _budget_slug(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return ("%g" % value).replace("-", "m").replace(".", "p")


def _new_experiment_dir(out_root, run_id=None):
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
        name = "mixtral_L1_mesh6x6_%s_p%d_%s" % (
            stamp, os.getpid(), uuid.uuid4().hex[:8])
    path = out_root / name
    path.mkdir(parents=False, exist_ok=False)
    return path.resolve()


def _base_env():
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    if os.name != "nt":
        env["LC_ALL"] = "C"
    return env


def _build_ilp_command(args, project_root, params_path, solution_path,
                       budgets):
    gpy = _normalize_executable(args.gurobi_python)
    ilp = project_root / "src" / "ilp_layout.py"
    command = [
        gpy,
        _interpreter_path(gpy, ilp),
        "--params", _interpreter_path(gpy, params_path),
        "--L", str(GRID["L"]),
        "--X", str(GRID["X"]),
        "--Y", str(GRID["Y"]),
        "--a-max", str(budgets["a_max"]),
        "--p-max", str(budgets["p_max"]),
        "--p-col", str(budgets["p_col"]),
        "--time-limit", str(args.time_limit),
        "--mipgap", str(args.mipgap),
        "--seed", str(args.seed),
        "--mode", CERTIFIED_MODE,
        "--secondary-time", str(args.secondary_time),
        "--secondary-mipgap", str(args.secondary_mipgap),
        "--quiet",
        "-o", _interpreter_path(gpy, solution_path),
    ]
    return command


def _build_python_command(python_exe, script, arguments):
    python_exe = _normalize_executable(python_exe)
    command = [python_exe, _interpreter_path(python_exe, script)]
    for value, is_path in arguments:
        if is_path:
            command.append(_interpreter_path(python_exe, value))
        else:
            command.append(str(value))
    return command


def _codegen_layout(args, project_root, params_path, solution_path,
                    codegen_dir, log_dir):
    codegen = project_root / "src" / "codegen.py"
    command = _build_python_command(
        args.system_python,
        codegen,
        [
            ("--params", False), (params_path, True),
            ("--sol", False), (solution_path, True),
            ("--out-dir", False), (codegen_dir, True),
            ("--router-mode", False), ("baremetal", False),
            ("--sim-tokens-per-source", False),
            (DISPATCH_CONFIG["tokens_per_source"], False),
            ("--sim-hidden", False), (DISPATCH_CONFIG["sim_hidden"], False),
            ("--sim-intermediate", False),
            (DISPATCH_CONFIG["sim_intermediate"], False),
            ("--sim-ffn-repeat", False), (DISPATCH_CONFIG["sim_ffn_repeat"], False),
        ],
    )
    process = _run_logged(
        command, project_root,
        log_dir / "codegen.stdout.log",
        log_dir / "codegen.stderr.log",
        timeout=args.codegen_timeout,
        env=_base_env(),
    )
    record = {
        "status": "failed",
        "process": process,
        "codegen_dir": str(Path(codegen_dir).resolve()),
        "dispatch_meta": None,
        "dispatch_formal": False,
        "error": "",
    }
    if process["returncode"] != 0:
        record["error"] = process["error"] or _tail(
            process["stderr_path"])
        return record

    try:
        with open(str(solution_path), "r", encoding="utf-8") as fh:
            solution_for_mesh = json.load(fh)
        grid = solution_for_mesh["grid"]
        mesh_rows = int(grid["X"])
        mesh_cols = int(grid["L"]) * int(grid["Y"])
        if mesh_rows <= 0 or mesh_cols <= 0:
            raise ValueError("mesh dimensions must be positive")
    except (OSError, ValueError, KeyError, TypeError) as ex:
        record["error"] = "cannot derive generated mesh from solution: %s" % ex
        return record

    required = [
        Path(codegen_dir) / "bench.yml",
        Path(codegen_dir) / "moe_coords.txt",
        Path(codegen_dir) / "dispatch_matrix.txt",
        Path(codegen_dir) / "dispatch_meta.json",
        Path(codegen_dir) / "topology" /
        ("mesh_%dx%d.gv" % (mesh_rows, mesh_cols)),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        record["error"] = "codegen missing required artifacts: %s" % (
            ", ".join(missing))
        return record

    try:
        with open(str(Path(codegen_dir) / "dispatch_meta.json"),
                  "r", encoding="utf-8") as fh:
            dispatch_meta = json.load(fh)
    except (OSError, ValueError) as ex:
        record["error"] = "cannot parse dispatch_meta.json: %s" % ex
        return record

    mismatches = []
    for key, expected in DISPATCH_CONFIG.items():
        if dispatch_meta.get(key) != expected:
            mismatches.append("%s=%r (expected %r)" % (
                key, dispatch_meta.get(key), expected))
    if dispatch_meta.get("trace_kind") != "model_derived_deterministic":
        mismatches.append(
            "trace_kind is not model_derived_deterministic")
    if dispatch_meta.get("runtime_sampling") is not False:
        mismatches.append("runtime_sampling is not false")
    if dispatch_meta.get("dispatch_algorithm") != (
            "global_largest_remainder_cyclic_transport_v1"):
        mismatches.append("unexpected/missing global dispatch algorithm")
    if dispatch_meta.get("per_source_rows_replicated") is not False:
        mismatches.append("per-source dispatch rows are marked replicated")
    if dispatch_meta.get("global_group_max_abs_error") != 0:
        mismatches.append("global group apportionment error is nonzero")
    if dispatch_meta.get("zero_worker_count") != 0:
        mismatches.append("formal dispatch contains zero-token workers")
    worker_assignments = dispatch_meta.get("worker_token_assignments")
    if (not isinstance(worker_assignments, list) or
            len(worker_assignments) != dispatch_meta.get("ngpu") or
            any(not isinstance(value, int) or value <= 0
                for value in worker_assignments)):
        mismatches.append(
            "worker_token_assignments must contain one positive integer per GPU")
    per_source_assignments = dispatch_meta.get(
        "per_source_route_assignments")
    meta_tokens = dispatch_meta.get("tokens_per_source")
    meta_topk = dispatch_meta.get("topk")
    expected_per_source = (
        meta_tokens * meta_topk
        if isinstance(meta_tokens, int) and isinstance(meta_topk, int)
        else None)
    if (not isinstance(per_source_assignments, list) or
            len(per_source_assignments) != dispatch_meta.get("ngpu") or
            any(value != expected_per_source
                for value in per_source_assignments)):
        mismatches.append(
            "per-source route totals do not equal tokens_per_source*topk")
    if mismatches:
        record["dispatch_meta"] = dispatch_meta
        record["error"] = "formal-dispatch validation failed: %s" % (
            "; ".join(mismatches))
        return record

    # The LEGOSim runner stages these exact optional provenance filenames.
    shutil.copy2(str(params_path),
                 str(Path(codegen_dir) / "params.input.json"))
    shutil.copy2(str(solution_path),
                 str(Path(codegen_dir) / "solution.input.json"))
    record.update({
        "status": "ok",
        "dispatch_meta": dispatch_meta,
        "dispatch_formal": True,
    })
    return record


def _legosim_runner_environment(args):
    """Return the runner controls that must cross the WSL boundary exactly."""
    return {
        "MOE_E2E_TIMEOUT": int(getattr(
            args, "moe_e2e_timeout", DEFAULT_MOE_E2E_TIMEOUT_S)),
        "MOE_GPGPUSIM_MAX_PARALLEL": int(getattr(
            args, "moe_gpgpusim_max_parallel",
            DEFAULT_MOE_GPGPUSIM_MAX_PARALLEL)),
    }


def _legosim_command(args, runner, codegen_dir, run_root, mirror_dir):
    runner_env = _legosim_runner_environment(args)
    if os.name == "nt":
        command = [_normalize_executable(args.wsl_exe)]
        if args.wsl_distro:
            command.extend(["-d", args.wsl_distro])
        command.extend([
            "--",
            "env", "MOE_E2E_ROUNDS=%d" % args.legosim_rounds,
            "MOE_E2E_TIMEOUT=%d" % runner_env["MOE_E2E_TIMEOUT"],
            "MOE_GPGPUSIM_MAX_PARALLEL=%d" %
            runner_env["MOE_GPGPUSIM_MAX_PARALLEL"],
            "RESULT_MIRROR_DIR=%s" % _windows_to_wsl(mirror_dir),
            "bash",
            _windows_to_wsl(runner),
            _windows_to_wsl(codegen_dir),
            _windows_to_wsl(run_root),
        ])
        return command, _base_env()

    env = _base_env()
    env["MOE_E2E_ROUNDS"] = str(args.legosim_rounds)
    env["MOE_E2E_TIMEOUT"] = str(runner_env["MOE_E2E_TIMEOUT"])
    env["MOE_GPGPUSIM_MAX_PARALLEL"] = str(
        runner_env["MOE_GPGPUSIM_MAX_PARALLEL"])
    env["RESULT_MIRROR_DIR"] = str(Path(mirror_dir).resolve())
    return [
        "bash", str(Path(runner).resolve()),
        str(Path(codegen_dir).resolve()),
        str(Path(run_root).resolve()),
    ], env


def _run_legosim(args, project_root, runner, codegen_dir, run_root, log_dir):
    run_root_text = str(run_root)
    external_wsl_root = os.name == "nt" and run_root_text.startswith("/")
    if not external_wsl_root:
        run_root_path = Path(run_root_text)
        if run_root_path.exists():
            raise RuntimeError(
                "refusing to reuse LEGOSim run root: %s" % run_root_path)
        run_root_path.parent.mkdir(parents=True, exist_ok=True)
    mirror_dir = Path(log_dir) / "legosim_result"
    command, env = _legosim_command(
        args, runner, codegen_dir, run_root_text, mirror_dir)
    process = _run_logged(
        command, project_root,
        log_dir / "legosim.stdout.log",
        log_dir / "legosim.stderr.log",
        timeout=args.legosim_timeout,
        env=env,
    )
    summary_path = mirror_dir / "ilp_e2e_summary.txt"
    summary = None
    summary_error = ""
    if summary_path.exists():
        try:
            summary = _parse_summary(summary_path)
        except OSError as ex:
            summary_error = "cannot parse summary: %s" % ex
    else:
        summary_error = "LEGOSim summary not found: %s" % summary_path

    validation = summary.get("validation") if summary else None
    bench_makespan = summary.get("makespan_cycles") if summary else None
    synchronized_makespan = (
        summary.get("synchronized_makespan_cycles") if summary else None)
    ok = (process["returncode"] == 0 and
          str(validation).upper() == "PASS" and
          isinstance(bench_makespan, (int, float)) and
          isinstance(synchronized_makespan, (int, float)) and
          bench_makespan > 0 and
          synchronized_makespan >= bench_makespan)
    error = ""
    if not ok:
        error = process["error"]
        if not error and process["returncode"] not in (None, 0):
            error = (_tail(process["stderr_path"]) or
                     _tail(process["stdout_path"]))
        if not error:
            error = summary_error
        if not error:
            error = (_tail(process["stderr_path"]) or
                     _tail(process["stdout_path"]))
    return {
        "status": "ok" if ok else "failed",
        "runner_environment": _legosim_runner_environment(args),
        "process": process,
        "run_root": (run_root_text if external_wsl_root else
                     str(Path(run_root_text).resolve())),
        "summary_path": (str(summary_path.resolve())
                         if summary_path.exists() else str(summary_path)),
        "summary": summary,
        "validation": validation,
        # The final synchronized completion time is the end-to-end metric.
        # Keep the bench-trace maximum separately for audit/debugging.
        "measured_makespan_cycles": synchronized_makespan,
        "bench_makespan_cycles": bench_makespan,
        "synchronized_makespan_cycles": synchronized_makespan,
        "completion_delta_cycles": (
            synchronized_makespan - bench_makespan
            if (isinstance(synchronized_makespan, (int, float)) and
                isinstance(bench_makespan, (int, float))) else None),
        "error": error,
    }


def _layout_row(experiment_id, case, method, solution, solve_process,
                solution_path, codegen_record=None, legosim_record=None,
                error=""):
    metrics = _solution_metrics(solution) if solution else {}
    a_max = case["budgets"]["a_max"]
    p_max = case["budgets"]["p_max"]
    area = metrics.get("total_area_mm2")
    power = metrics.get("peak_power_W")
    status = "solve_failed"
    if solution:
        status = "ok"
        if codegen_record and codegen_record.get("status") != "ok":
            status = "codegen_failed"
        if legosim_record and legosim_record.get("status") != "ok":
            status = "legosim_failed"
    row = {
        "experiment_id": experiment_id,
        "case_id": case["case_id"],
        "workload": "mixtral",
        "sweep_dimension": case["sweep_dimension"],
        "budget_value": case["budget_value"],
        "method": method,
        "mode": CERTIFIED_MODE if method == "ilp" else "matched-layout",
        "L": GRID["L"],
        "X": GRID["X"],
        "Y": GRID["Y"],
        "a_max_mm2": a_max,
        "p_max_W": p_max,
        "p_col_W": case["budgets"]["p_col"],
        "status": status,
        "returncode": solve_process.get("returncode")
        if solve_process else None,
        "timed_out": solve_process.get("timed_out")
        if solve_process else None,
        "orchestrator_solve_wall_s": solve_process.get("wall_s")
        if solve_process else None,
        "area_slack_mm2": (a_max - area if isinstance(
            area, (int, float)) else None),
        "power_slack_W": (p_max - power if isinstance(
            power, (int, float)) else None),
        "codegen_status": codegen_record.get("status")
        if codegen_record else None,
        "dispatch_formal": codegen_record.get("dispatch_formal")
        if codegen_record else None,
        "legosim_status": legosim_record.get("status")
        if legosim_record else "not_requested",
        "legosim_wall_s": (
            legosim_record.get("process", {}).get("wall_s")
            if legosim_record else None),
        "measured_makespan_cycles": (
            legosim_record.get("measured_makespan_cycles")
            if legosim_record else None),
        "legosim_validation": (
            legosim_record.get("validation")
            if legosim_record else None),
        "solution_path": str(Path(solution_path).resolve())
        if solution_path else None,
        "codegen_dir": codegen_record.get("codegen_dir")
        if codegen_record else None,
        "legosim_run_root": legosim_record.get("run_root")
        if legosim_record else None,
        "legosim_summary_path": legosim_record.get("summary_path")
        if legosim_record else None,
        "error": error or (
            legosim_record.get("error") if legosim_record else "") or (
            codegen_record.get("error") if codegen_record else ""),
    }
    row.update(metrics)
    return row


def _process_layout(args, project_root, params_path, experiment_dir, case,
                    method, solution_path, solution, solve_process, runner):
    method_dir = Path(case["case_dir"]) / method
    method_dir.mkdir(parents=True, exist_ok=True)
    codegen_dir = method_dir / "codegen"
    codegen_record = _codegen_layout(
        args, project_root, params_path, solution_path,
        codegen_dir, method_dir)
    legosim_record = None
    if args.run_legosim and codegen_record["status"] == "ok":
        # InterChiplet uses filesystem FIFOs for process communication.  Keep
        # them on WSL/Linux's native /tmp rather than DrvFS (/mnt/c), where
        # open() can fail.  Compact evidence is mirrored into method_dir.
        run_root = "/tmp/ilp4moe_budget_mesh_%s" % uuid.uuid4().hex
        legosim_record = _run_legosim(
            args, project_root, runner, codegen_dir, run_root, method_dir)
    row = _layout_row(
        Path(experiment_dir).name, case, method, solution,
        solve_process, solution_path, codegen_record, legosim_record)
    return {
        "method": method,
        "solution_path": str(Path(solution_path).resolve()),
        "solution_sha256": _sha256(solution_path),
        "metrics": _solution_metrics(solution),
        "codegen": codegen_record,
        "legosim": legosim_record,
    }, row


def _resolve_representative(summary, baseline_dir):
    candidates = []
    if isinstance(summary, dict):
        for key in ("representative_path", "representative_solution",
                    "representative"):
            value = summary.get(key)
            if isinstance(value, str):
                candidates.append(_external_path(value, baseline_dir))
    candidates.append(Path(baseline_dir) / "representative.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _run_matched_baseline(args, project_root, params_path, case,
                          ilp_solution_path):
    script = project_root / "src" / "matched_layout_baseline.py"
    baseline_dir = Path(case["case_dir"]) / "matched_search"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    arguments = [
        ("--params", False), (params_path, True),
        ("--solution", False), (ilp_solution_path, True),
        ("--seeds", False),
    ]
    arguments.extend((seed, False) for seed in args.baseline_seeds)
    arguments.extend([("--out-dir", False), (baseline_dir, True)])
    command = _build_python_command(
        args.system_python, script, arguments)
    process = _run_logged(
        command, project_root,
        baseline_dir / "baseline.stdout.log",
        baseline_dir / "baseline.stderr.log",
        timeout=args.baseline_timeout,
        env=_base_env(),
    )
    summary_path = baseline_dir / "summary.json"
    summary = None
    if summary_path.exists():
        try:
            with open(str(summary_path), "r", encoding="utf-8") as fh:
                summary = json.load(fh)
        except (OSError, ValueError):
            summary = None
    representative = _resolve_representative(summary, baseline_dir)
    error = ""
    if process["returncode"] != 0:
        error = process["error"] or _tail(process["stderr_path"])
    elif representative is None:
        error = "matched baseline did not emit representative.json"
    return {
        "status": "ok" if not error else "failed",
        "process": process,
        "out_dir": str(baseline_dir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "summary": summary,
        "representative_path": (str(representative)
                                if representative else None),
        "error": error,
    }, representative


def _enrich_comparison(rows):
    ilp = next((row for row in rows if row.get("method") == "ilp"), None)
    if not ilp:
        return
    for row in rows:
        if row is ilp:
            continue
        ilp_pred = ilp.get("predicted_makespan_cycles")
        other_pred = row.get("predicted_makespan_cycles")
        if (isinstance(ilp_pred, (int, float)) and ilp_pred > 0 and
                isinstance(other_pred, (int, float))):
            row["ilp_speedup_vs_method_predicted"] = other_pred / ilp_pred
        ilp_meas = ilp.get("measured_makespan_cycles")
        other_meas = row.get("measured_makespan_cycles")
        if (isinstance(ilp_meas, (int, float)) and ilp_meas > 0 and
                isinstance(other_meas, (int, float))):
            row["ilp_speedup_vs_method_measured"] = other_meas / ilp_meas
        ilp_time = ilp.get("orchestrator_solve_wall_s")
        other_time = row.get("orchestrator_solve_wall_s")
        if (isinstance(ilp_time, (int, float)) and ilp_time > 0 and
                isinstance(other_time, (int, float))):
            row["method_search_time_over_ilp"] = other_time / ilp_time


def _parse_seed_values(values):
    seeds = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                seeds.append(int(token))
    if not seeds:
        raise ValueError("at least one matched-baseline seed is required")
    return seeds


def _validate_args(args):
    for label, values in (
            ("area", args.area_budgets),
            ("power", args.power_budgets)):
        if not values or any(value <= 0 for value in values):
            raise ValueError("%s budgets must all be positive" % label)
        if len(set(values)) != len(values):
            raise ValueError("%s budgets must be unique" % label)
    for label in ("time_limit", "mipgap", "secondary_mipgap", "p_col",
                  "codegen_timeout", "legosim_timeout", "baseline_timeout"):
        if getattr(args, label) <= 0:
            raise ValueError("--%s must be positive" % label.replace("_", "-"))
    if args.secondary_time < 0:
        raise ValueError("--secondary-time must be non-negative")
    if args.legosim_rounds <= 0:
        raise ValueError("--legosim-rounds must be positive")
    if not (1 <= args.moe_e2e_timeout <= MAX_MOE_E2E_TIMEOUT_S):
        raise ValueError(
            "--moe-e2e-timeout must be in [1,%d] seconds" %
            MAX_MOE_E2E_TIMEOUT_S)
    if not (1 <= args.moe_gpgpusim_max_parallel <=
            MAX_MOE_GPGPUSIM_MAX_PARALLEL):
        raise ValueError(
            "--moe-gpgpusim-max-parallel must be in [1,%d]" %
            MAX_MOE_GPGPUSIM_MAX_PARALLEL)


def _parse_args(argv=None):
    default_gurobi = os.environ.get(
        "GUROBI_PYTHON",
        sys.executable)
    ap = argparse.ArgumentParser(
        description="Mixtral certified-ILP budget sweep on a 6x6 mesh")
    ap.add_argument(
        "--project-root", default=None,
        help="ILP4MOE root (default: inferred from this script)")
    ap.add_argument(
        "--out-root", default="out/budget_mesh_experiment",
        help="parent directory for a new unique experiment directory")
    ap.add_argument(
        "--run-id", default=None,
        help="explicit unique experiment directory name (must not exist)")
    ap.add_argument(
        "--params", default=None,
        help="optional Mixtral params JSON; default regenerates it")
    ap.add_argument(
        "--python", dest="system_python", default=sys.executable,
        help="system Python for workload/codegen/baseline")
    ap.add_argument(
        "--gurobi-python", default=default_gurobi,
        help="Gurobi Python 3.7 executable (or GUROBI_PYTHON)")
    ap.add_argument(
        "--legosim-root", default=None,
        help=("LEGOSIM root; default: "
              "<project>/LEGOSIM_MICRO-main/LEGOSIM_MICRO-main"))
    ap.add_argument("--wsl-exe", default="wsl.exe")
    ap.add_argument(
        "--wsl-distro", default=None,
        help="optional distro passed to wsl.exe -d on native Windows")
    ap.add_argument(
        "--area-budgets", nargs="+", type=float,
        default=list(DEFAULT_AREA_BUDGETS))
    ap.add_argument(
        "--power-budgets", nargs="+", type=float,
        default=list(DEFAULT_POWER_BUDGETS))
    ap.add_argument(
        "--relaxed-area", type=float, default=None,
        help="area cap during the power sweep (default: max area budget)")
    ap.add_argument(
        "--relaxed-power", type=float, default=None,
        help="power cap during the area sweep (default: max power budget)")
    ap.add_argument(
        "--p-col", type=float, default=340.0,
        help="fixed per-column thermal cap (non-binding for L=1 Mixtral)")
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--mipgap", type=float, default=0.005)
    ap.add_argument(
        "--secondary-time", type=float, default=3.0,
        help="short certified-latency tie-break stage in seconds")
    ap.add_argument(
        "--secondary-mipgap", type=float, default=0.02,
        help="MIP gap target for the secondary placement/resource stage")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--solver-timeout", type=float, default=None,
        help="outer process timeout (default: ILP time limit + 120 seconds)")
    ap.add_argument("--codegen-timeout", type=float, default=300.0)
    ap.add_argument(
        "--run-legosim", action="store_true",
        help="run real LEGOSim after codegen (off by default)")
    ap.add_argument("--legosim-timeout", type=float, default=7200.0)
    ap.add_argument("--legosim-rounds", type=int, default=4)
    ap.add_argument(
        "--moe-e2e-timeout", "--e2e-timeout",
        dest="moe_e2e_timeout", type=int,
        default=DEFAULT_MOE_E2E_TIMEOUT_S,
        help=("runner-internal timeout in seconds, passed explicitly as "
              "MOE_E2E_TIMEOUT (default: %(default)s; max: {})".format(
                  MAX_MOE_E2E_TIMEOUT_S)))
    ap.add_argument(
        "--moe-gpgpusim-max-parallel", "--gpgpusim-max-parallel",
        dest="moe_gpgpusim_max_parallel", type=int,
        default=DEFAULT_MOE_GPGPUSIM_MAX_PARALLEL,
        help=("host GPU-simulator concurrency, passed explicitly as "
              "MOE_GPGPUSIM_MAX_PARALLEL "
              "(default: %(default)s; max: {})".format(
                  MAX_MOE_GPGPUSIM_MAX_PARALLEL)))
    ap.add_argument(
        "--with-matched-baseline", action="store_true",
        help=("also run matched_layout_baseline.py when present; if absent, "
              "warn and continue ILP-only"))
    ap.add_argument(
        "--baseline-seeds", nargs="+", default=["0", "1", "2", "3", "4"])
    ap.add_argument("--baseline-timeout", type=float, default=1800.0)
    ap.add_argument(
        "--fail-fast", action="store_true",
        help="stop after the first failed budget point")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print the resolved experiment plan without creating or running it")
    return ap.parse_args(argv)


def _make_cases(args, experiment_dir):
    relaxed_area = (args.relaxed_area if args.relaxed_area is not None
                    else max(args.area_budgets))
    relaxed_power = (args.relaxed_power if args.relaxed_power is not None
                     else max(args.power_budgets))
    if relaxed_area < max(args.area_budgets):
        raise ValueError("--relaxed-area must be >= every area budget")
    if relaxed_power < max(args.power_budgets):
        raise ValueError("--relaxed-power must be >= every power budget")
    cases = []
    for value in args.area_budgets:
        case_id = "area_%s" % _budget_slug(value)
        cases.append({
            "case_id": case_id,
            "sweep_dimension": "area",
            "budget_value": value,
            "budgets": {
                "a_max": value,
                "p_max": relaxed_power,
                "p_col": args.p_col,
            },
            "case_dir": str(Path(experiment_dir) / "cases" / case_id),
        })
    for value in args.power_budgets:
        case_id = "power_%s" % _budget_slug(value)
        cases.append({
            "case_id": case_id,
            "sweep_dimension": "power",
            "budget_value": value,
            "budgets": {
                "a_max": relaxed_area,
                "p_max": value,
                "p_col": args.p_col,
            },
            "case_dir": str(Path(experiment_dir) / "cases" / case_id),
        })
    return cases, relaxed_area, relaxed_power


def main(argv=None):
    args = _parse_args(argv)
    args.baseline_seeds = _parse_seed_values(args.baseline_seeds)
    _validate_args(args)

    project_root = (_resolve_path(Path.cwd(), args.project_root)
                    if args.project_root else DEFAULT_PROJECT_ROOT)
    out_root = _resolve_path(project_root, args.out_root)
    legosim_root = (
        _resolve_path(project_root, args.legosim_root)
        if args.legosim_root
        else (project_root / "LEGOSIM_MICRO-main" /
              "LEGOSIM_MICRO-main").resolve())
    runner = (
        legosim_root / "benchmark" / "All2all-MoE-E2E" /
        "run_ilp_e2e.sh")

    planned_dir = out_root / (
        args.run_id or "<unique_mixtral_L1_mesh6x6_timestamp_pid_uuid>")
    plan_cases, relaxed_area, relaxed_power = _make_cases(
        args, planned_dir)
    if args.dry_run:
        print(json.dumps({
            "project_root": str(project_root),
            "planned_experiment_dir": str(planned_dir),
            "gurobi_python": _normalize_executable(args.gurobi_python),
            "legosim_root": str(legosim_root),
            "run_legosim": args.run_legosim,
            "legosim_runner_environment": _legosim_runner_environment(args),
            "certified_mode": CERTIFIED_MODE,
            "grid": GRID,
            "dispatch": DISPATCH_CONFIG,
            "relaxed_area": relaxed_area,
            "relaxed_power": relaxed_power,
            "cases": [{
                "case_id": case["case_id"],
                "sweep_dimension": case["sweep_dimension"],
                "budget_value": case["budget_value"],
                "budgets": case["budgets"],
            } for case in plan_cases],
        }, indent=2, sort_keys=True))
        return 0

    required = [
        project_root / "src" / "workload.py",
        project_root / "src" / "ilp_layout.py",
        project_root / "src" / "codegen.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing required source files: %s" %
                           ", ".join(missing))

    gurobi_exe = _normalize_executable(args.gurobi_python)
    if not _command_available(gurobi_exe):
        raise RuntimeError(
            "Gurobi Python not found: %s (set --gurobi-python or "
            "GUROBI_PYTHON)" % gurobi_exe)
    system_python = _normalize_executable(args.system_python)
    if not _command_available(system_python):
        raise RuntimeError("system Python not found: %s" % system_python)
    if args.run_legosim:
        if not runner.exists():
            raise RuntimeError("LEGOSim runner not found: %s" % runner)
        if os.name == "nt" and not _command_available(
                _normalize_executable(args.wsl_exe)):
            raise RuntimeError("wsl.exe not found: %s" % args.wsl_exe)

    matched_script = project_root / "src" / "matched_layout_baseline.py"
    matched_available = matched_script.exists()
    if args.with_matched_baseline and not matched_available:
        print("[warn] matched baseline requested but %s is absent; "
              "continuing ILP-only" % matched_script, file=sys.stderr)

    experiment_dir = _new_experiment_dir(out_root, args.run_id)
    cases, relaxed_area, relaxed_power = _make_cases(
        args, experiment_dir)
    raw_path = experiment_dir / "raw_results.json"
    csv_path = experiment_dir / "raw_results.csv"
    rows = []
    sources = {
        path.name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in required
    }
    if matched_available:
        sources[matched_script.name] = {
            "path": str(matched_script.resolve()),
            "sha256": _sha256(matched_script),
        }
    raw = {
        "schema_version": 1,
        "status": "running",
        "started_utc": _utc_now(),
        "finished_utc": None,
        "experiment": {
            "experiment_id": experiment_dir.name,
            "experiment_dir": str(experiment_dir),
            "project_root": str(project_root),
            "workload": "mixtral",
            "grid": GRID,
            "topology": "mesh",
            "mode": CERTIFIED_MODE,
            "area_budgets_mm2": args.area_budgets,
            "power_budgets_W": args.power_budgets,
            "relaxed_area_mm2": relaxed_area,
            "relaxed_power_W": relaxed_power,
            "p_col_W": args.p_col,
            "time_limit_s": args.time_limit,
            "mipgap_target": args.mipgap,
            "secondary_time_s": args.secondary_time,
            "secondary_mipgap_target": args.secondary_mipgap,
            "seed": args.seed,
            "dispatch": DISPATCH_CONFIG,
            "run_legosim": args.run_legosim,
            "legosim_root": str(legosim_root),
            "legosim_rounds": args.legosim_rounds,
            "legosim_runner_environment": _legosim_runner_environment(args),
            "gurobi_python": gurobi_exe,
            "system_python": system_python,
            "matched_baseline": {
                "requested": args.with_matched_baseline,
                "available": matched_available,
                "seeds": args.baseline_seeds,
                "skip_reason": (
                    "not requested" if not args.with_matched_baseline
                    else ("script missing" if not matched_available else None)),
            },
        },
        "host": {
            "platform": platform.platform(),
            "os_name": os.name,
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "git": _git_provenance(project_root),
        "sources": sources,
        "workload_generation": None,
        "params": None,
        "cases": [],
    }
    _atomic_json(raw_path, raw)
    _atomic_csv(csv_path, rows)

    params_path = experiment_dir / "params_mixtral.json"
    if args.params:
        params_source = _resolve_path(project_root, args.params)
        if not params_source.exists():
            raw["status"] = "failed"
            raw["finished_utc"] = _utc_now()
            raw["fatal_error"] = "params file not found: %s" % params_source
            _atomic_json(raw_path, raw)
            raise RuntimeError(raw["fatal_error"])
        shutil.copy2(str(params_source), str(params_path))
        raw["workload_generation"] = {
            "status": "copied",
            "source": str(params_source),
            "source_sha256": _sha256(params_source),
        }
    else:
        workload = project_root / "src" / "workload.py"
        workload_command = _build_python_command(
            args.system_python,
            workload,
            [
                ("--config", False), ("mixtral", False),
                ("-o", False), (params_path, True),
            ],
        )
        workload_process = _run_logged(
            workload_command, project_root,
            experiment_dir / "workload.stdout.log",
            experiment_dir / "workload.stderr.log",
            timeout=300,
            env=_base_env(),
        )
        raw["workload_generation"] = {
            "status": ("ok" if workload_process["returncode"] == 0
                       and params_path.exists() else "failed"),
            "process": workload_process,
        }
        if (workload_process["returncode"] != 0 or
                not params_path.exists()):
            raw["status"] = "failed"
            raw["finished_utc"] = _utc_now()
            raw["fatal_error"] = (
                workload_process["error"] or
                _tail(workload_process["stderr_path"]) or
                "workload generator did not emit params JSON")
            _atomic_json(raw_path, raw)
            raise RuntimeError(raw["fatal_error"])

    try:
        with open(str(params_path), "r", encoding="utf-8") as fh:
            params = json.load(fh)
        if params.get("meta", {}).get("config") not in (None, "mixtral"):
            raise ValueError("params meta.config is not mixtral")
        for key in ("library", "tasks", "edges", "interconnect"):
            if key not in params:
                raise ValueError("params JSON lacks %s" % key)
    except (OSError, ValueError) as ex:
        raw["status"] = "failed"
        raw["finished_utc"] = _utc_now()
        raw["fatal_error"] = "invalid params JSON: %s" % ex
        _atomic_json(raw_path, raw)
        raise RuntimeError(raw["fatal_error"])
    raw["params"] = {
        "path": str(params_path.resolve()),
        "sha256": _sha256(params_path),
        "meta": params.get("meta"),
    }
    _atomic_json(raw_path, raw)

    failed = False
    solver_timeout = (args.solver_timeout
                      if args.solver_timeout is not None
                      else args.time_limit + args.secondary_time + 120.0)
    for case in cases:
        case_dir = Path(case["case_dir"])
        case_dir.mkdir(parents=True, exist_ok=False)
        solution_path = case_dir / "ilp_solution.json"
        command = _build_ilp_command(
            args, project_root, params_path, solution_path,
            case["budgets"])
        print("[run] %-12s a_max=%g p_max=%g" % (
            case["case_id"], case["budgets"]["a_max"],
            case["budgets"]["p_max"]), flush=True)
        ilp_process = _run_logged(
            command, project_root,
            case_dir / "ilp.stdout.log",
            case_dir / "ilp.stderr.log",
            timeout=solver_timeout,
            env=_base_env(),
        )
        solution = None
        solution_error = ""
        if solution_path.exists():
            try:
                with open(str(solution_path), "r", encoding="utf-8") as fh:
                    solution = json.load(fh)
            except (OSError, ValueError) as ex:
                solution_error = "invalid ILP solution JSON: %s" % ex
        if ilp_process["returncode"] != 0:
            solution_error = (
                ilp_process["error"] or _tail(ilp_process["stderr_path"]) or
                _tail(ilp_process["stdout_path"]) or
                "ILP process returned %r" % ilp_process["returncode"])
            solution = None
        elif solution is None and not solution_error:
            solution_error = "ILP process emitted no solution JSON"

        case_record = {
            "case_id": case["case_id"],
            "sweep_dimension": case["sweep_dimension"],
            "budget_value": case["budget_value"],
            "budgets": case["budgets"],
            "case_dir": str(case_dir.resolve()),
            "ilp": {
                "status": "ok" if solution else "failed",
                "process": ilp_process,
                "solution_path": str(solution_path.resolve()),
                "solution_sha256": (
                    _sha256(solution_path) if solution else None),
                "metrics": (
                    _solution_metrics(solution) if solution else None),
                "error": solution_error,
            },
            "layouts": [],
            "matched_baseline": None,
        }
        case_rows = []
        if solution is None:
            failed = True
            case_rows.append(_layout_row(
                experiment_dir.name, case, "ilp", None, ilp_process,
                solution_path, error=solution_error))
        else:
            layout_record, row = _process_layout(
                args, project_root, params_path, experiment_dir, case,
                "ilp", solution_path, solution, ilp_process, runner)
            case_record["layouts"].append(layout_record)
            case_rows.append(row)
            if row["status"] != "ok":
                failed = True

            if args.with_matched_baseline and matched_available:
                baseline_record, representative = _run_matched_baseline(
                    args, project_root, params_path, case, solution_path)
                case_record["matched_baseline"] = baseline_record
                if representative:
                    try:
                        with open(str(representative), "r",
                                  encoding="utf-8") as fh:
                            baseline_solution = json.load(fh)
                        baseline_layout, baseline_row = _process_layout(
                            args, project_root, params_path, experiment_dir,
                            case, "matched", representative,
                            baseline_solution, baseline_record["process"],
                            runner)
                        case_record["layouts"].append(baseline_layout)
                        case_rows.append(baseline_row)
                        if baseline_row["status"] != "ok":
                            failed = True
                    except (OSError, ValueError) as ex:
                        failed = True
                        baseline_record["status"] = "failed"
                        baseline_record["error"] = (
                            "invalid representative solution: %s" % ex)
                        case_rows.append(_layout_row(
                            experiment_dir.name, case, "matched", None,
                            baseline_record["process"], representative,
                            error=baseline_record["error"]))
                else:
                    failed = True
                    case_rows.append(_layout_row(
                        experiment_dir.name, case, "matched", None,
                        baseline_record["process"], None,
                        error=baseline_record["error"]))

        _enrich_comparison(case_rows)
        rows.extend(case_rows)
        raw["cases"].append(case_record)
        raw["status"] = "running_with_errors" if failed else "running"
        _atomic_json(raw_path, raw)
        _atomic_csv(csv_path, rows)
        if args.fail_fast and failed:
            break

    raw["status"] = "complete_with_errors" if failed else "complete"
    raw["finished_utc"] = _utc_now()
    raw["row_count"] = len(rows)
    _atomic_json(raw_path, raw)
    _atomic_csv(csv_path, rows)
    print("[done] status=%s" % raw["status"])
    print("[done] JSON: %s" % raw_path)
    print("[done] CSV : %s" % csv_path)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        sys.exit(2)
