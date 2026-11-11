#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible min-T ILP scaling and fixed-block QAP representation audit.

This file deliberately lives outside the paper sources.  It has two roles:

1. Reuse ``workload.py`` and ``ilp_layout.py`` unchanged to measure the full
   ILP-Perf model at 8--128 *unaggregated* expert groups.  Its only objective
   is ``min T``.  For deterministic dispatch, public ``T`` is the epigraph of
   fixed-order max-plus protocol completion and directed Mesh capacity; the
   legacy C11--C13 horizon ``T_task`` is feasibility-only.  Area and package
   CPU+GPU peak power are constraints, and the CPU anchor is free.
2. Separately isolate a communication-only, fixed-block placement subproblem
   and encode the same restricted instance in three ways:

   * separable: coordinate/absolute-value Manhattan linearization;
   * kb: the actual Kaufman--Broeckx big-M linearization (one continuous
     auxiliary per facility/location assignment);
   * product: explicit edge/location-pair product variables (the formulation
     that has O(|E||P|^2) auxiliary variables).

The controlled QAP representation ablation uses homogeneous sites, assigns
every fixed task block to one distinct site, and minimizes traffic-weighted
Manhattan byte-hops.  It has no chiplet selection, task scheduling, directed
Mesh flow/capacity model, or heterogeneous timing/resource constraints.  It
is therefore not a competing full-system method; the restriction only makes
the three representations mathematically comparable.

Run from the repository root with system Python (the driver launches Gurobi's
Python 3.7 interpreter for each isolated worker):

  python src/run_scaling_qap_experiment.py \
      --out out/scaling_qap_experiment --time-limit 30

The script is standard-library-only in worker mode and Python-3.7 compatible.
"""

from __future__ import print_function

import argparse
import csv
try:
    import ctypes
except ImportError:  # Gurobi's minimal Windows Python may omit _ctypes.pyd.
    ctypes = None
import datetime
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time


SCRIPT_PATH = os.path.abspath(__file__)
SRC_DIR = os.path.dirname(SCRIPT_PATH)
ROOT_DIR = os.path.dirname(SRC_DIR)
DEFAULT_GUROBI_PYTHON = os.environ.get("GUROBI_PYTHON", sys.executable)
PERF_MODE = "ilp-perf"
PERF_MODE_LABEL = "ILP-Perf"
PERF_OBJECTIVE = "min T"
PERF_RESULT_KIND = "full_joint_protocol_min_t"
MIP_FOCUS = 1
EXPERIMENT_CONTRACT_VERSION = "ilp-perf-protocol-min-t-qap-scope-v2"
# Scaling budgets are declared as a system-level package-growth policy, not
# reconstructed from an integer GPU-L/GPU-S target.  Both limits include the
# one CPU and every selected GPU.  Keeping the coefficients independent of the
# chiplet-library increments is important: the integer BOM is an output of the
# MILP, not hidden in the right-hand side of C7/C8.
SCALING_AREA_MM2_PER_EXPERT = 180.0
SCALING_POWER_W_PER_EXPERT = 140.0
SCALING_BUDGET_POLICY = "linear-package-envelope-180mm2-140W-per-expert-v1"
SCALING_BUDGET_CONTRACT_VERSION = "target-bom-independent-scaling-budget-v1"
HOMOGENEOUS_CONTROL_POLICY = "all-gpu-s-control-envelope-v1"
QAP_RESULT_KIND = "communication_only_fixed_block_qap_representation_ablation"
QAP_OBJECTIVE = "min traffic-weighted Manhattan byte-hops"
QAP_SCOPE = {
    "role": "representation ablation; not a full-system competitor",
    "blocks": "fixed generated task blocks",
    "sites": "homogeneous mesh coordinates with injective assignment",
    "included": ["task-to-site assignment", "Manhattan byte-hop objective"],
    "excluded": [
        "chiplet type/BOM selection",
        "heterogeneous compute timing",
        "non-preemptive task scheduling",
        "fixed-order replay protocol and max-plus completion",
        "directed Mesh routing and link-capacity constraints",
        "area and package peak-power constraints",
    ],
}


def _mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def _write_json(path, value):
    parent = os.path.dirname(os.path.abspath(path))
    _mkdir(parent)
    with open(path, "w") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)


def _read_json(path):
    with open(path, "r") as fh:
        return json.load(fh)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, AttributeError):
        return False


def _peak_rss_mb():
    """Best-effort peak resident set size for the current worker process."""
    if os.name == "nt" and ctypes is not None:
        try:
            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb)
            if ok:
                return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)
        except Exception:
            return None
    if os.name == "nt":
        try:
            raw = subprocess.check_output([
                "powershell.exe", "-NoProfile", "-Command",
                "(Get-Process -Id %d).PeakWorkingSet64" % os.getpid(),
            ], universal_newlines=True).strip()
            return float(raw) / (1024.0 * 1024.0)
        except Exception:
            return None
    try:
        import resource
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return None


def _status_name(code, GRB):
    names = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return names.get(code, "STATUS_%s" % code)


def _make_anytime_callback(gp, GRB, trace, state, mipgap_target,
                           sample_period_s=0.5):
    """Record both incumbent events and periodic bound progress."""
    def append_point(runtime, incumbent, bound, nodes, event):
        if (incumbent is not None and _finite(incumbent) and
                abs(float(incumbent)) < 1e90 and _finite(bound) and
                abs(float(bound)) < 1e90):
            gap = abs(float(incumbent) - float(bound)) / max(
                abs(float(incumbent)), 1.0e-12)
        else:
            gap = None
            if not (_finite(incumbent) and abs(float(incumbent)) < 1e90):
                incumbent = None
        row = {
            "wall_s": float(runtime),
            "incumbent": incumbent,
            "bound": float(bound) if _finite(bound) and abs(float(bound)) < 1e90 else None,
            "gap": gap,
            "nodes": float(nodes) if nodes is not None else None,
            "event": event,
        }
        trace.append(row)
        if incumbent is not None and state.get("first_feasible_s") is None:
            state["first_feasible_s"] = float(runtime)
        if (gap is not None and gap <= mipgap_target and
                state.get("target_gap_s") is None):
            state["target_gap_s"] = float(runtime)

    def callback(model, where):
        try:
            if where == GRB.Callback.MIPSOL:
                runtime = model.cbGet(GRB.Callback.RUNTIME)
                incumbent = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
                nodes = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
                append_point(runtime, incumbent, bound, nodes, "incumbent")
                state["last_sample_s"] = float(runtime)
            elif where == GRB.Callback.MIP:
                runtime = float(model.cbGet(GRB.Callback.RUNTIME))
                if runtime - state.get("last_sample_s", -1.0e30) < sample_period_s:
                    return
                incumbent = model.cbGet(GRB.Callback.MIP_OBJBST)
                bound = model.cbGet(GRB.Callback.MIP_OBJBND)
                nodes = model.cbGet(GRB.Callback.MIP_NODCNT)
                append_point(runtime, incumbent, bound, nodes, "periodic")
                state["last_sample_s"] = runtime
        except Exception:
            # Measurement must not affect optimization.
            return
    return callback


def _full_run_contract(experts, scenario, args):
    """Return the active full-model budgets/options and budget-tier metadata.

    ``heterogeneous-main`` uses one exogenous package-growth rule for every
    scale: 180 mm2 and 140 W per expert, including the shared CPU overhead.
    These continuous limits are not computed from any candidate integer BOM.
    GPU type counts are also absent from ``nmin``/``nmax``.  Consequently,
    ILP-Perf jointly selects the GPU-L/GPU-S counts, assigns tasks, places
    chiplets, routes traffic, and schedules work while minimizing T.  The
    exact CPU count is a workload requirement, but its coordinate is free.

    ``homogeneous-control`` deliberately retains the tight all-GPU-S envelope
    as a control, rather than being reported as a joint-BOM experiment.
    """
    if scenario not in ("heterogeneous-main", "homogeneous-control"):
        raise ValueError("unknown scaling scenario: %s" % scenario)
    experts = int(experts)
    if scenario == "heterogeneous-main":
        a_max = SCALING_AREA_MM2_PER_EXPERT * experts
        p_max = SCALING_POWER_W_PER_EXPERT * experts
        budget_policy = SCALING_BUDGET_POLICY
        area_rate = SCALING_AREA_MM2_PER_EXPERT
        power_rate = SCALING_POWER_W_PER_EXPERT
        description = (
            "exogenous linear package envelope; GPU-L/GPU-S counts are "
            "MILP outputs and are not encoded in either budget")
    else:
        # Deliberately tight control: the all-GPU-S bill of materials exactly
        # consumes both resource limits.  This is not the joint-BOM result.
        a_max = 90.0 + 110.0 * experts
        p_max = 65.0 + 90.0 * experts
        budget_policy = HOMOGENEOUS_CONTROL_POLICY
        area_rate = None
        power_rate = None
        description = "tight all-GPU-S control envelope"
    budgets = {
        "a_max": a_max,
        "p_max": p_max,
        "nmax": {"CPU": 1},
        "nmin": {"CPU": 1},
    }
    opts = {
        "time_limit": float(args.time_limit),
        "mipgap": float(args.mipgap),
        "mipfocus": MIP_FOCUS,
        "seed": int(args.seed),
        "symmetry_break": True,
        "mode": PERF_MODE,
        "threads": int(args.threads),
        "cpu_anchor": "free",
        "noc_traffic": "deterministic-dispatch",
        "dispatch_tokens_per_source": 32,
        "dispatch_hidden": 128,
    }
    tier = {
        "policy": budget_policy,
        "area_mm2_per_expert": area_rate,
        "power_W_per_expert": power_rate,
        "target_bom": None,
        "description": description,
    }
    return budgets, opts, tier


def _validate_perf_solution_contract(sol):
    """Fail closed if a full worker did not solve the sole active contract."""
    objective = sol.get("objective_contract") or {}
    solver = sol.get("solver_configuration") or {}
    protocol = sol.get("protocol_model") or {}
    failures = []
    checks = (
        (sol.get("optimization_mode") == PERF_MODE,
         "optimization_mode must be ilp-perf"),
        (objective.get("active_objective_count") == 1,
         "active_objective_count must be 1"),
        (objective.get("active_objective") == PERF_OBJECTIVE,
         "active objective must be min T"),
        (objective.get("joint_selection_placement_scheduling") is True,
         "joint selection/placement/scheduling must be active"),
        (objective.get("protocol_completion_in_T") is True,
         "fixed-order protocol completion must be included in T"),
        (objective.get("protocol_used_as_separate_objective") is False,
         "protocol completion must not be a separate objective"),
        (objective.get("legacy_task_schedule_role") ==
         "hard-feasibility-only",
         "legacy T_task schedule must be hard-feasibility-only"),
        (objective.get("T_definition") ==
         "fixed-order replay max-plus completion plus Mesh capacity epigraph",
         "T must be the protocol/Mesh-capacity epigraph"),
        (protocol.get("included_in_T") is True,
         "protocol_model must be included in T"),
        (protocol.get("used_as_separate_objective") is False,
         "protocol_model must not be a separate objective"),
        (protocol.get("kind") == "fixed-order-max-plus-event-DAG",
         "full worker must use the fixed-order max-plus protocol"),
        (objective.get("energy_or_byte_hop_tie_break") is False,
         "energy/byte-hop tie-break must be absent"),
        (sol.get("performance_cap_enforced") is False,
         "performance cap must be absent"),
        (sol.get("secondary_objective") is None,
         "secondary objective must be absent"),
        (sol.get("tertiary_objective") is None,
         "tertiary objective must be absent"),
        (sol.get("quaternary_objective") is None,
         "quaternary objective must be absent"),
        (sol.get("cpu_anchor_policy") == "free",
         "CPU anchor policy must be free"),
        (solver.get("gurobi_mip_focus") == MIP_FOCUS,
         "Gurobi MIPFocus must be 1"),
        (solver.get("changes_objective") is False,
         "MIPFocus must not change the objective"),
    )
    for passed, message in checks:
        if not passed:
            failures.append(message)
    if failures:
        raise ValueError("invalid ILP-Perf solution contract: %s" %
                         "; ".join(failures))
    return True


def _selected_type_counts(sol):
    counts = {}
    for site in sol.get("sites") or []:
        kind = site.get("type")
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def _validate_result_record_contract(row, family):
    """Reject stale cached rows from any retired experiment contract."""
    if row.get("contract_version") != EXPERIMENT_CONTRACT_VERSION:
        raise ValueError(
            "cached %s result has a stale or missing contract_version" % family)
    if row.get("mip_focus") != MIP_FOCUS:
        raise ValueError("cached %s result was not run with MIPFocus=1" % family)
    if family == "scaling":
        if row.get("kind") != PERF_RESULT_KIND:
            raise ValueError("cached scaling result is not a full min-T run")
        if row.get("optimization_mode") != PERF_MODE:
            raise ValueError("cached scaling result is not ILP-Perf")
        if row.get("active_objective") != PERF_OBJECTIVE:
            raise ValueError("cached scaling result does not minimize only T")
        if row.get("T_definition") != (
                "fixed-order replay max-plus completion plus Mesh capacity "
                "epigraph"):
            raise ValueError("cached scaling result has the wrong T definition")
        if row.get("legacy_task_schedule_role") != "hard-feasibility-only":
            raise ValueError(
                "cached scaling result does not keep T_task feasibility-only")
        if row.get("protocol_completion_in_T") is not True:
            raise ValueError(
                "cached scaling result excludes protocol completion from T")
        if row.get("protocol_used_as_separate_objective") is not False:
            raise ValueError(
                "cached scaling result uses a separate protocol objective")
        if row.get("cpu_anchor_policy") != "free":
            raise ValueError("cached scaling result does not use a free CPU anchor")
        if row.get("gpu_counts_fixed") is not False:
            raise ValueError("cached scaling result fixes GPU counts")
        if row.get("scaling_budget_contract_version") != (
                SCALING_BUDGET_CONTRACT_VERSION):
            raise ValueError(
                "cached scaling result uses a stale or missing budget contract")
        scenario = row.get("scenario")
        if scenario not in ("heterogeneous-main", "homogeneous-control"):
            raise ValueError("cached scaling result has an unknown scenario")
        expected_policy = (SCALING_BUDGET_POLICY
                           if scenario == "heterogeneous-main" else
                           HOMOGENEOUS_CONTROL_POLICY)
        if row.get("budget_policy") != expected_policy:
            raise ValueError("cached scaling result has the wrong budget policy")
        if row.get("budget_target_bom") is not None:
            raise ValueError("scaling budget must not serialize a target BOM")
        budgets = row.get("budgets") or {}
        gpu_bound_keys = {"GPU-L", "GPU-S"}
        if (gpu_bound_keys.intersection((budgets.get("nmin") or {}).keys()) or
                gpu_bound_keys.intersection(
                    (budgets.get("nmax") or {}).keys())):
            raise ValueError("scaling budget directly fixes a GPU type count")
        try:
            experts = int(row.get("experts"))
        except (TypeError, ValueError):
            raise ValueError("cached scaling result has no valid expert count")
        if scenario == "heterogeneous-main":
            expected_area = SCALING_AREA_MM2_PER_EXPERT * experts
            expected_power = SCALING_POWER_W_PER_EXPERT * experts
            if (not _finite(budgets.get("a_max")) or
                    not _finite(budgets.get("p_max")) or
                    float(budgets.get("a_max")) != expected_area or
                    float(budgets.get("p_max")) != expected_power):
                raise ValueError(
                    "heterogeneous scaling caps violate the linear envelope")
            if (row.get("budget_area_mm2_per_expert") !=
                    SCALING_AREA_MM2_PER_EXPERT or
                    row.get("budget_power_W_per_expert") !=
                    SCALING_POWER_W_PER_EXPERT):
                raise ValueError(
                    "heterogeneous scaling row has the wrong budget rates")
    elif family == "qap":
        if row.get("kind") != QAP_RESULT_KIND:
            raise ValueError("cached QAP result is not the scoped representation ablation")
        if row.get("comparison_role") != QAP_SCOPE["role"]:
            raise ValueError("cached QAP result has the wrong comparison scope")
        if row.get("objective") != QAP_OBJECTIVE:
            raise ValueError("cached QAP result has the wrong restricted objective")
    else:
        raise ValueError("unknown result family: %s" % family)
    return True


def _worker_full(args):
    import gurobipy as gp
    from gurobipy import GRB

    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    import ilp_layout

    params = _read_json(args.params)
    experts = int(args.experts)
    scenario = args.scaling_scenario
    budgets, opts, budget_tier = _full_run_contract(experts, scenario, args)
    # Change the default environment before ilp_layout constructs its model.
    gp.setParam("Threads", int(args.threads))
    gp.setParam("MIPFocus", MIP_FOCUS)
    wall0 = time.time()
    sol, stats = ilp_layout.build_and_solve(
        params, 1, int(args.mesh), int(args.mesh), budgets, opts,
        T_hat=None, verbose=True)
    total_wall = time.time() - wall0
    if sol is None:
        result = {
            "contract_version": EXPERIMENT_CONTRACT_VERSION,
            "kind": PERF_RESULT_KIND,
            "scenario": scenario,
            "optimization_mode": PERF_MODE,
            "mode_label": PERF_MODE_LABEL,
            "active_objective": PERF_OBJECTIVE,
            "T_definition": (
                "fixed-order replay max-plus completion plus Mesh capacity "
                "epigraph"),
            "legacy_task_schedule_role": "hard-feasibility-only",
            "protocol_completion_in_T": True,
            "protocol_used_as_separate_objective": False,
            "mip_focus": MIP_FOCUS,
            "cpu_anchor_policy": "free",
            "scaling_budget_contract_version": (
                SCALING_BUDGET_CONTRACT_VERSION),
            "budget_policy": budget_tier["policy"],
            "budget_area_mm2_per_expert": budget_tier[
                "area_mm2_per_expert"],
            "budget_power_W_per_expert": budget_tier[
                "power_W_per_expert"],
            "budget_target_bom": budget_tier["target_bom"],
            "gpu_counts_fixed": False,
            "budgets": budgets,
            "experts": experts,
            "mesh": int(args.mesh),
            "sites": int(args.mesh) ** 2,
            "tasks": len(params["tasks"]),
            "edges": len(params["edges"]),
            "feasible": False,
            "status_code": stats.get("status"),
            "status": _status_name(stats.get("status"), GRB),
            "model_build_s": stats.get("model_build_s"),
            "solve_wall_s": stats.get("wall"),
            "total_worker_wall_s": total_wall,
            "peak_rss_mb": _peak_rss_mb(),
            "trace": stats.get("anytime_trace", []),
        }
    else:
        _validate_perf_solution_contract(sol)
        type_counts = _selected_type_counts(sol)
        gpu_l = int(type_counts.get("GPU-L", 0))
        gpu_s = int(type_counts.get("GPU-S", 0))
        gap = sol.get("primary_mip_gap")
        target_gap_time = sol.get("primary_wall_s") if (
            gap is not None and gap <= float(args.mipgap) + 1.0e-12) else None
        expert_assignment = []
        group_pi = params.get("meta", {}).get("group_pi", [])
        for site in sol.get("sites", []):
            for task in site.get("tasks", []):
                match = re.match(r"^e\((\d+)\)$", task)
                if match:
                    group = int(match.group(1))
                    expert_assignment.append({
                        "group": group,
                        "popularity": (group_pi[group]
                                       if group < len(group_pi) else None),
                        "chiplet_type": site.get("type"),
                        "site": [site.get("l"), site.get("i"), site.get("j")],
                    })
        expert_assignment.sort(key=lambda r: r["group"])
        gpu_l_groups = [r["group"] for r in expert_assignment
                        if r["chiplet_type"] == "GPU-L"]
        ranked_hot = sorted(expert_assignment,
                            key=lambda r: -(r["popularity"] or 0.0))
        hot_count = gpu_l
        hot_groups = set(r["group"] for r in ranked_hot[:hot_count])
        top_hot_on_l = (float(sum(1 for g in gpu_l_groups if g in hot_groups)) /
                        hot_count) if hot_count else None
        gpu_l_popularity = sum((r["popularity"] or 0.0)
                               for r in expert_assignment
                               if r["chiplet_type"] == "GPU-L")
        result = {
            "contract_version": EXPERIMENT_CONTRACT_VERSION,
            "kind": PERF_RESULT_KIND,
            "scenario": scenario,
            "optimization_mode": PERF_MODE,
            "mode_label": PERF_MODE_LABEL,
            "active_objective": PERF_OBJECTIVE,
            "T_definition": (sol.get("objective_contract") or {}).get(
                "T_definition"),
            "legacy_task_schedule_role": (
                (sol.get("objective_contract") or {}).get(
                    "legacy_task_schedule_role")),
            "protocol_completion_in_T": (
                (sol.get("objective_contract") or {}).get(
                    "protocol_completion_in_T")),
            "protocol_used_as_separate_objective": (
                (sol.get("objective_contract") or {}).get(
                    "protocol_used_as_separate_objective")),
            "objective_contract": sol.get("objective_contract"),
            "solver_configuration": sol.get("solver_configuration"),
            "mip_focus": MIP_FOCUS,
            "cpu_anchor_policy": sol.get("cpu_anchor_policy"),
            "scaling_budget_contract_version": (
                SCALING_BUDGET_CONTRACT_VERSION),
            "budget_policy": budget_tier["policy"],
            "budget_area_mm2_per_expert": budget_tier[
                "area_mm2_per_expert"],
            "budget_power_W_per_expert": budget_tier[
                "power_W_per_expert"],
            "budget_target_bom": budget_tier["target_bom"],
            "gpu_counts_fixed": False,
            "budgets": budgets,
            "gpu_l": gpu_l,
            "gpu_s": gpu_s,
            "experts": experts,
            "mesh": int(args.mesh),
            "sites": int(args.mesh) ** 2,
            "tasks": len(params["tasks"]),
            "edges": len(params["edges"]),
            "feasible": True,
            "status_code": sol.get("primary_status"),
            "status": _status_name(sol.get("primary_status"), GRB),
            "model_build_s": sol.get("model_build_s"),
            "solve_wall_s": sol.get("primary_wall_s"),
            "total_worker_wall_s": total_wall,
            "first_feasible_s": sol.get("time_to_first_feasible_s"),
            "target_gap_time_s": target_gap_time,
            "final_gap": gap,
            "incumbent_cycles": sol.get("primary_incumbent_cycles"),
            "bound_cycles": sol.get("primary_objective_bound_cycles"),
            "public_T_cycles": sol.get("predicted_makespan_cycles"),
            "protocol_completion_cycles": sol.get(
                "protocol_makespan_cycles"),
            "legacy_T_task_cycles": sol.get(
                "legacy_task_horizon_cycles"),
            "nodes": sol.get("primary_node_count"),
            "area_mm2": sol.get("total_area_mm2"),
            "package_peak_power_W": sol.get("peak_power_W"),
            "expert_assignment": expert_assignment,
            "gpu_l_expert_groups": gpu_l_groups,
            "gpu_l_popularity_share": gpu_l_popularity,
            "top_hot_groups_on_gpu_l_fraction": top_hot_on_l,
            "peak_rss_mb": _peak_rss_mb(),
            "trace": sol.get("anytime_trace", []),
        }
    _write_json(args.result, result)
    return 0


def _qap_data(params, mesh):
    tasks = sorted(params["tasks"].keys())
    task_index = dict((v, i) for i, v in enumerate(tasks))
    sites = [(i, j) for i in range(mesh) for j in range(mesh)]
    edges = []
    for edge in params["edges"]:
        edges.append((task_index[edge["src"]], task_index[edge["dst"]],
                      float(edge["vol"])))
    return tasks, sites, edges


def _build_qap_model(gp, GRB, formulation, params, mesh):
    """Build one representation of the restricted fixed-block QAP ablation.

    This intentionally does not call or approximate the complete ILP-Perf
    scheduling/routing model.  Only injective assignment and traffic-weighted
    Manhattan byte-hops are shared across these three encodings.
    """
    tasks, sites, edges = _qap_data(params, mesh)
    V = range(len(tasks))
    P = range(len(sites))
    m = gp.Model("qap_%s" % formulation)
    x = m.addVars([(v, p) for v in V for p in P],
                  vtype=GRB.BINARY, name="x")
    m.addConstrs((gp.quicksum(x[v, p] for p in P) == 1 for v in V),
                 name="assign")
    m.addConstrs((gp.quicksum(x[v, p] for v in V) <= 1 for p in P),
                 name="capacity")

    def dist(p, q):
        return (abs(sites[p][0] - sites[q][0]) +
                abs(sites[p][1] - sites[q][1]))

    if formulation == "separable":
        di = m.addVars(range(len(edges)), lb=0.0, name="di")
        dj = m.addVars(range(len(edges)), lb=0.0, name="dj")
        for eidx, (u, v, vol) in enumerate(edges):
            ui = gp.quicksum(sites[p][0] * x[u, p] for p in P)
            vi = gp.quicksum(sites[p][0] * x[v, p] for p in P)
            uj = gp.quicksum(sites[p][1] * x[u, p] for p in P)
            vj = gp.quicksum(sites[p][1] * x[v, p] for p in P)
            m.addConstr(di[eidx] >= ui - vi, name="dip_%d" % eidx)
            m.addConstr(di[eidx] >= vi - ui, name="dim_%d" % eidx)
            m.addConstr(dj[eidx] >= uj - vj, name="djp_%d" % eidx)
            m.addConstr(dj[eidx] >= vj - uj, name="djm_%d" % eidx)
        m.setObjective(gp.quicksum(
            vol * (di[eidx] + dj[eidx])
            for eidx, (_, _, vol) in enumerate(edges)), GRB.MINIMIZE)
        aux_count = 2 * len(edges)

    elif formulation == "kb":
        # Exact Kaufman--Broeckx/Glover big-M linearization:
        #   w[v,p] = x[v,p] * sum_{(v,u)} flow[v,u] d[p,q] x[u,q].
        # Nonnegative costs let one lower big-M inequality plus min(sum w)
        # enforce equality.  M is a valid, tighter rectangular-assignment upper
        # bound: for each outgoing neighbour, take its farthest legal q.
        w = m.addVars([(v, p) for v in V for p in P], lb=0.0, name="w")
        outgoing = dict((v, []) for v in V)
        for u, v, vol in edges:
            outgoing[u].append((v, vol))
        for v in V:
            for p in P:
                terms = []
                big_m = 0.0
                for u, vol in outgoing[v]:
                    terms.extend(vol * dist(p, q) * x[u, q] for q in P)
                    big_m += vol * max(dist(p, q) for q in P)
                expr = gp.quicksum(terms) if terms else 0.0
                m.addConstr(w[v, p] >= expr - big_m * (1 - x[v, p]),
                            name="kb_%d_%d" % (v, p))
        m.setObjective(gp.quicksum(w[v, p] for v in V for p in P),
                       GRB.MINIMIZE)
        aux_count = len(tasks) * len(sites)

    elif formulation == "product":
        # Explicit product expansion. z is binary to match the usual direct
        # assignment encoding and to expose its |E||P|^2 model-size cost.
        zkeys = [(eidx, p, q) for eidx in range(len(edges))
                 for p in P for q in P]
        z = m.addVars(zkeys, vtype=GRB.BINARY, name="z")
        for eidx, (u, v, vol) in enumerate(edges):
            for p in P:
                for q in P:
                    zv = z[eidx, p, q]
                    m.addConstr(zv <= x[u, p])
                    m.addConstr(zv <= x[v, q])
                    m.addConstr(zv >= x[u, p] + x[v, q] - 1)
        m.setObjective(gp.quicksum(
            vol * dist(p, q) * z[eidx, p, q]
            for eidx, (_, _, vol) in enumerate(edges)
            for p in P for q in P), GRB.MINIMIZE)
        aux_count = len(edges) * len(sites) * len(sites)
    else:
        raise ValueError("unknown formulation: %s" % formulation)

    return m, {
        "tasks": len(tasks),
        "edges": len(edges),
        "sites": len(sites),
        "assignment_binaries": len(tasks) * len(sites),
        "auxiliary_variables_constructed": aux_count,
    }


def _worker_qap(args):
    import gurobipy as gp
    from gurobipy import GRB

    params = _read_json(args.params)
    build0 = time.time()
    m, dimensions = _build_qap_model(
        gp, GRB, args.formulation, params, int(args.mesh))
    m.Params.OutputFlag = 1
    m.Params.Threads = int(args.threads)
    m.Params.Seed = int(args.seed)
    m.Params.TimeLimit = float(args.time_limit)
    m.Params.MIPGap = float(args.mipgap)
    m.Params.MIPFocus = MIP_FOCUS
    m.update()
    build_s = time.time() - build0
    counts = {
        "model_variables": int(m.NumVars),
        "model_binary_variables": int(m.NumBinVars),
        "model_integer_variables": int(m.NumIntVars),
        "model_continuous_variables": int(m.NumVars - m.NumIntVars),
        "model_constraints": int(m.NumConstrs),
        "model_nonzeros": int(m.NumNZs),
    }
    trace = []
    state = {"first_feasible_s": None, "target_gap_s": None,
             "last_sample_s": -1.0e30}
    cb = _make_anytime_callback(
        gp, GRB, trace, state, float(args.mipgap), 0.5)
    solve0 = time.time()
    m.optimize(cb)
    solve_s = time.time() - solve0
    status_code = int(m.Status)
    feasible = m.SolCount > 0
    incumbent = float(m.ObjVal) if feasible else None
    try:
        bound = float(m.ObjBound)
    except Exception:
        bound = None
    try:
        gap = float(m.MIPGap) if feasible else None
    except Exception:
        gap = None
    try:
        nodes = float(m.NodeCount)
    except Exception:
        nodes = None
    if feasible:
        final = {
            "wall_s": solve_s, "incumbent": incumbent, "bound": bound,
            "gap": gap, "nodes": nodes, "event": "final",
        }
        trace.append(final)
    target_gap_time = state.get("target_gap_s")
    if (target_gap_time is None and gap is not None and
            gap <= float(args.mipgap) + 1.0e-12):
        target_gap_time = solve_s
    result = {
        "contract_version": EXPERIMENT_CONTRACT_VERSION,
        "kind": QAP_RESULT_KIND,
        "comparison_role": QAP_SCOPE["role"],
        "objective": QAP_OBJECTIVE,
        "scope": QAP_SCOPE,
        "mip_focus": MIP_FOCUS,
        "formulation": args.formulation,
        "experts": int(args.experts),
        "mesh": int(args.mesh),
        "feasible": feasible,
        "status_code": status_code,
        "status": _status_name(status_code, GRB),
        "model_build_s": build_s,
        "solve_wall_s": solve_s,
        "total_worker_wall_s": build_s + solve_s,
        "first_feasible_s": state.get("first_feasible_s"),
        "target_gap_time_s": target_gap_time,
        "final_gap": gap,
        "objective_byte_hops": incumbent,
        "bound_byte_hops": bound,
        "nodes": nodes,
        "peak_rss_mb": _peak_rss_mb(),
        "trace": trace,
    }
    result.update(dimensions)
    result.update(counts)
    _write_json(args.result, result)
    return 0


def _parse_gurobi_log(text_value):
    parsed = {}
    matches = re.findall(
        r"Optimize a model with\s+([0-9,]+) rows,\s+([0-9,]+) columns and\s+([0-9,]+) nonzeros",
        text_value)
    if matches:
        rows, cols, nz = matches[-1]
        parsed.update({
            "model_constraints": int(rows.replace(",", "")),
            "model_variables": int(cols.replace(",", "")),
            "model_nonzeros": int(nz.replace(",", "")),
        })
    matches = re.findall(
        r"Variable types:\s+([0-9,]+) continuous,\s+([0-9,]+) integer \(([0-9,]+) binary\)",
        text_value)
    if matches:
        # The first line is the constructed model; later lines describe the
        # presolved model and are not comparable across formulations.
        continuous, integer, binary = matches[0]
        parsed.update({
            "model_continuous_variables": int(continuous.replace(",", "")),
            "model_integer_variables": int(integer.replace(",", "")),
            "model_binary_variables": int(binary.replace(",", "")),
        })
    return parsed


def _run_worker(gurobi_python, common, log_path):
    started = time.time()
    proc = subprocess.Popen(
        [gurobi_python, SCRIPT_PATH] + common,
        cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True)
    output, _ = proc.communicate()
    with open(log_path, "w") as fh:
        fh.write(output)
    return {
        "returncode": proc.returncode,
        "driver_observed_wall_s": time.time() - started,
        "log": output,
    }


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict((f, _csv_value(row.get(f))) for f in fields))


def _generate_params(out_dir, experts):
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    import workload
    params = workload.build(
        "scaling-e%d" % experts, experts, experts, 2,
        4096, 14336, 256, 2, 0.7)
    path = os.path.join(out_dir, "params", "experts_%03d.json" % experts)
    _write_json(path, params)
    return path, params


def _flatten_trace(result, family):
    rows = []
    for seq, point in enumerate(result.get("trace") or []):
        incumbent = point.get("incumbent")
        bound = point.get("bound")
        # ilp_layout uses names specialized for latency.
        if incumbent is None:
            incumbent = point.get("incumbent_cycles")
        if bound is None:
            bound = point.get("bound_cycles")
        rows.append({
            "family": family,
            "formulation": result.get("formulation", PERF_RESULT_KIND),
            "scenario": result.get("scenario", ""),
            "experts": result.get("experts"),
            "mesh": result.get("mesh"),
            "sequence": seq,
            "wall_s": point.get("wall_s"),
            "incumbent": incumbent,
            "bound": bound,
            "gap": point.get("gap"),
            "nodes": point.get("nodes"),
            "event": point.get("event", "incumbent"),
        })
    # ilp_layout records MIPSOL events.  If the bound/incumbent does not change
    # again before timeout, explicitly add the terminal observation so a
    # gap-vs-time plot shows the plateau rather than ending at first feasible.
    final_wall = result.get("solve_wall_s")
    final_gap = result.get("final_gap")
    if family == "scaling" and final_wall is not None and final_gap is not None:
        last_wall = rows[-1].get("wall_s") if rows else None
        if last_wall is None or float(final_wall) - float(last_wall) > 1.0e-6:
            rows.append({
                "family": family,
                "formulation": result.get("formulation", PERF_RESULT_KIND),
                "scenario": result.get("scenario", ""),
                "experts": result.get("experts"),
                "mesh": result.get("mesh"),
                "sequence": len(rows),
                "wall_s": final_wall,
                "incumbent": result.get("incumbent_cycles"),
                "bound": result.get("bound_cycles"),
                "gap": final_gap,
                "nodes": result.get("nodes"),
                "event": "final",
            })
    return rows


def _projection_rows():
    """Model-size preflight for explicit products; no solver allocation."""
    rows = []
    for experts, mesh in [(8, 6), (16, 8), (32, 12), (64, 12), (128, 16)]:
        tasks = experts + 5
        edges = 2 * experts + 3
        sites = mesh * mesh
        product_aux = edges * sites * sites
        product_link_constraints = 3 * product_aux
        rows.append({
            "experts": experts, "mesh": mesh, "tasks": tasks,
            "edges": edges, "sites": sites,
            "assignment_binaries": tasks * sites,
            "product_auxiliary_binaries": product_aux,
            "product_link_constraints": product_link_constraints,
            "constructed": experts <= 16,
            "skip_reason": ("" if experts <= 16 else
                            "preflight cap: explicit product model exceeds safe experiment size"),
        })
    return rows


def _make_plots(out_dir, scaling_rows, qap_rows, trace_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:
        return {"status": "skipped", "reason": str(ex)}

    plot_dir = os.path.join(out_dir, "plots")
    _mkdir(plot_dir)

    xs = sorted(set(r["experts"] for r in scaling_rows))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    style = {
        "heterogeneous-main": ("-", "heterogeneous"),
        "homogeneous-control": ("--", "homogeneous control"),
    }
    for scenario in ("heterogeneous-main", "homogeneous-control"):
        rr = [r for r in scaling_rows if r.get("scenario") == scenario]
        rr.sort(key=lambda r: r["experts"])
        if not rr:
            continue
        line, label = style[scenario]
        ax.plot([r["experts"] for r in rr],
                [r.get("first_feasible_s") or float("nan") for r in rr],
                "o" + line, label=label + ": first feasible")
        ax.plot([r["experts"] for r in rr],
                [r.get("target_gap_time_s") or float("nan") for r in rr],
                "s" + line, label=label + ": target gap")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Experts (= placement groups)")
    ax.set_ylabel("Time (s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(plot_dir, "scaling_runtime.png")
    fig.savefig(p1, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for experts in xs:
        pts = [r for r in trace_rows if r["family"] == "scaling" and
               r.get("scenario") == "heterogeneous-main" and
               r["experts"] == experts and r.get("gap") is not None]
        if pts:
            ax.step([max(float(r["wall_s"]), 1e-4) for r in pts],
                    [max(float(r["gap"]), 1e-7) for r in pts], where="post",
                    label="E=%d" % experts)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Solve time (s)")
    ax.set_ylabel("Solver relative gap")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p2 = os.path.join(plot_dir, "scaling_gap_vs_time.png")
    fig.savefig(p2, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    forms = ["separable", "kb", "product"]
    marks = {"separable": "o", "kb": "s", "product": "^"}
    for form in forms:
        rr = [r for r in qap_rows if r.get("formulation") == form]
        rr.sort(key=lambda x: x["experts"])
        if rr:
            ax.plot([r["experts"] for r in rr],
                    [r.get("total_worker_wall_s") or float("nan") for r in rr],
                    marker=marks[form], label=form)
    ax.set_yscale("log")
    ax.set_xlabel("Experts in fixed-block representation ablation")
    ax.set_ylabel("Build + solve time (s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p3 = os.path.join(plot_dir, "qap_formulation_runtime.png")
    fig.savefig(p3, dpi=220)
    plt.close(fig)
    return {"status": "ok", "files": [p1, p2, p3]}


def _write_readme(out_dir, metadata, scaling_rows, qap_rows):
    scaling_done = sum(1 for r in scaling_rows if r.get("feasible"))
    scaling_at_target = sum(1 for r in scaling_rows
                            if r.get("target_gap_time_s") is not None)
    qap_done = sum(1 for r in qap_rows if r.get("feasible"))
    text_value = "# ILP-Perf scaling and fixed-block QAP representation audit\n\n"
    text_value += "Generated: %s\n\n" % metadata["generated_at"]
    text_value += "## What is measured\n\n"
    text_value += (
        "`scaling_summary.csv` runs the repository's complete ILP-Perf model. "
        "Its sole objective is `min T`, where T is the epigraph of the "
        "fixed-order max-plus replay-protocol completion and the directed "
        "Mesh capacity bound. The legacy C11--C13 task horizon `T_task` is a "
        "hard feasibility schedule only; there is no "
        "power mode, performance cap, or energy/byte-hop tie-break. Area and "
        "package CPU+GPU peak power remain hard budgets. The CPU count is one, "
        "but its coordinate is free. Experts are not aggregated: groups=experts. "
        "The expert/site pairs are 8/6x6, 16/8x8, 32/12x12, 64/12x12, "
        "and 128/16x16. Heterogeneous-main applies the same exogenous package "
        "growth rule at every scale: Amax=180E mm2 and Pmax=140E W, including "
        "the CPU. These limits are not reconstructed from an integer GPU mix, "
        "and no GPU-L/GPU-S count appears in nmin/nmax; both counts are therefore "
        "outputs of each fresh MILP solve. The optional homogeneous-control "
        "envelope deliberately admits only all-GPU-S. Runs use one solver "
        "thread, MIPFocus=1, and the configured wall-clock limit. First-feasible "
        "time is an incumbent callback timestamp. Target-gap time is present "
        "only when the solver reaches the configured relative MIP gap; blank "
        "means it did not do so within the limit.\n\n")
    text_value += (
        "The scaling workloads use `workload.py`'s analytic placeholder costs; "
        "they are formulation-scaling experiments, not large end-to-end "
        "LEGOSim measurements. `metadata.json` records SHA-256 hashes of the "
        "exact model, workload generator, and runner. Any files under "
        "`superseded_nonformal/` are quarantined historical diagnostics and "
        "are not read into the CSVs or plots.\n\n")
    text_value += (
        "Bound semantics: ILP-Perf minimizes native cycle-valued T with unit "
        "objective scale. `target_gap_time_s` means the solver established the "
        "configured relative MIP gap, not necessarily zero-gap exactness. Any "
        "time-limited row with an open gap is an incumbent and must not be "
        "described as optimal.\n\n")
    text_value += (
        "`qap_summary.csv` is only a controlled, communication-only fixed-block "
        "representation ablation. All three encodings see the same generated "
        "DAG, homogeneous sites, injective task-to-site assignments, traffic "
        "weights, and Manhattan byte-hop objective. They have no chiplet "
        "selection, heterogeneous compute timing, non-preemptive scheduling, "
        "fixed-order replay protocol, or directed Mesh routing/link-capacity "
        "model. The QAP rows therefore "
        "compare representation cost only and are not full-system competitors "
        "to ILP-Perf.\n\n")
    text_value += "## Formulation nomenclature (important)\n\n"
    text_value += (
        "Kaufman--Broeckx is the compact big-M/Glover QAP linearization: it "
        "adds one continuous auxiliary for each assignment, i.e. O(|V||P|), "
        "but normally has a weak LP relaxation. The O(|E||P|^2) construction "
        "is the explicit edge/location-pair product (Lawler-style) model. "
        "These must not be described as the same formulation.\n\n")
    text_value += "## Completeness\n\n"
    text_value += "- Full scaling feasible runs: %d/%d\n" % (scaling_done, len(scaling_rows))
    text_value += "- Full scaling runs reaching target gap: %d/%d\n" % (
        scaling_at_target, len(scaling_rows))
    text_value += "- QAP feasible runs: %d/%d\n" % (qap_done, len(qap_rows))
    text_value += "- Solver limit per run: %s s; target gap: %s\n" % (
        metadata["time_limit_s"], metadata["mipgap_target"])
    text_value += (
        "- `product_preflight.csv` projects explicit-product size for all "
        "scaling points. Models above E=16 are intentionally not allocated; "
        "this is a documented safety cap, not a solver result.\n\n")
    text_value += "## Headline data\n\n"
    text_value += "Heterogeneous full model (seconds; `--` = target gap not reached):\n\n"
    text_value += "| Experts | Mesh | First feasible | Target gap | Final gap | Status |\n"
    text_value += "|---:|:---:|---:|---:|---:|:---|\n"
    main_rows = [r for r in scaling_rows
                 if r.get("scenario") == "heterogeneous-main"]
    main_rows.sort(key=lambda r: r["experts"])
    for row in main_rows:
        target = row.get("target_gap_time_s")
        target_text = "--" if target is None else "%.3f" % float(target)
        gap = row.get("final_gap")
        gap_text = "--" if gap is None else "%.4g" % float(gap)
        text_value += "| %d | %dx%d | %.3f | %s | %s | %s |\n" % (
            row["experts"], row["mesh"], row["mesh"],
            float(row.get("first_feasible_s") or 0.0), target_text,
            gap_text, row.get("status"))
    text_value += "\nControlled QAP formulation runs:\n\n"
    text_value += "| Formulation | Experts | Vars | Constraints | Build+solve (s) | Gap | Status |\n"
    text_value += "|:---|---:|---:|---:|---:|---:|:---|\n"
    for row in qap_rows:
        gap = row.get("final_gap")
        gap_text = "--" if gap is None else "%.4g" % float(gap)
        text_value += "| %s | %d | %d | %d | %.3f | %s | %s |\n" % (
            row["formulation"], row["experts"],
            int(row.get("model_variables") or 0),
            int(row.get("model_constraints") or 0),
            float(row.get("total_worker_wall_s") or 0.0), gap_text,
            row.get("status"))
    text_value += "\n"
    text_value += "## Files\n\n"
    text_value += (
        "- `scaling_summary.csv`, `scaling_gap_trace.csv`\n"
        "- `qap_summary.csv`, `qap_gap_trace.csv`\n"
        "- `product_preflight.csv`\n"
        "- `params/`, `results/`, `logs/`, `plots/`\n"
        "- `metadata.json`\n")
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write(text_value)


def _driver(args):
    out_dir = os.path.abspath(args.out)
    _mkdir(out_dir)
    _mkdir(os.path.join(out_dir, "params"))
    _mkdir(os.path.join(out_dir, "results"))
    _mkdir(os.path.join(out_dir, "logs"))
    if not os.path.isfile(args.gurobi_python):
        raise SystemExit("Gurobi Python not found: %s" % args.gurobi_python)

    scaling_pairs = [(8, 6), (16, 8), (32, 12), (64, 12), (128, 16)]
    params_by_expert = {}
    params_objects = {}
    for experts, _ in scaling_pairs:
        path, obj = _generate_params(out_dir, experts)
        params_by_expert[experts] = path
        params_objects[experts] = obj

    metadata = {
        "contract_version": EXPERIMENT_CONTRACT_VERSION,
        "generated_at": datetime.datetime.now().astimezone().isoformat(),
        "root": ROOT_DIR,
        "script": SCRIPT_PATH,
        "script_sha256": _sha256_file(SCRIPT_PATH),
        "ilp_layout_sha256": _sha256_file(os.path.join(SRC_DIR, "ilp_layout.py")),
        "workload_sha256": _sha256_file(os.path.join(SRC_DIR, "workload.py")),
        "driver_python": sys.executable,
        "gurobi_python": os.path.abspath(args.gurobi_python),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "time_limit_s": float(args.time_limit),
        "mipgap_target": float(args.mipgap),
        "threads": int(args.threads),
        "seed": int(args.seed),
        "mip_focus": MIP_FOCUS,
        "full_model_contract": {
            "mode": PERF_MODE,
            "mode_label": PERF_MODE_LABEL,
            "active_objective_count": 1,
            "active_objective": PERF_OBJECTIVE,
            "T_definition": (
                "max(fixed-order replay max-plus protocol completion, "
                "directed Mesh capacity bound)"),
            "legacy_task_schedule_role": "hard-feasibility-only",
            "task_dag_fallback_T_definition": (
                "T_task plus phase NoC completion bound"),
            "protocol_used_as_separate_objective": False,
            "cpu_anchor": "free",
            "resource_constraints": [
                "area", "package CPU+GPU peak power"],
            "scaling_budget_contract_version": (
                SCALING_BUDGET_CONTRACT_VERSION),
            "heterogeneous_scaling_budget": {
                "policy": SCALING_BUDGET_POLICY,
                "a_max_formula": "180 * experts mm2",
                "p_max_formula": "140 * experts W",
                "area_mm2_per_expert": SCALING_AREA_MM2_PER_EXPERT,
                "power_W_per_expert": SCALING_POWER_W_PER_EXPERT,
                "target_bom": None,
                "gpu_count_bounds": None,
            },
            "retired_or_absent": [
                "power mode", "local power-density C9",
                "performance cap", "energy/byte-hop tie-break"],
        },
        "qap_representation_ablation": {
            "kind": QAP_RESULT_KIND,
            "objective": QAP_OBJECTIVE,
            "scope": QAP_SCOPE,
        },
        "scaling_pairs": scaling_pairs,
        "workload": {
            "topk": 2, "d_model": 4096, "d_ff": 14336,
            "batch": 256, "bytes_elem": 2, "zipf_skew": 0.7,
            "groups_equal_experts": True,
            "cost_model": "workload.py analytic placeholder",
        },
    }
    try:
        gv = subprocess.check_output(
            [args.gurobi_python, "-c",
             "import gurobipy as gp; print('.'.join(map(str,gp.gurobi.version())))"],
            universal_newlines=True).strip()
        metadata["gurobi_version"] = gv
    except Exception as ex:
        metadata["gurobi_version_error"] = str(ex)

    scaling_rows = []
    scaling_trace = []
    if args.scaling_scenarios == "both":
        scaling_scenarios = ("heterogeneous-main", "homogeneous-control")
    else:
        scaling_scenarios = (args.scaling_scenarios,)
    for scenario in scaling_scenarios:
        scenario_slug = scenario.replace("-", "_")
        for experts, mesh in scaling_pairs:
            result_path = os.path.join(
                out_dir, "results", "scaling_%s_e%03d_m%02d.json" %
                (scenario_slug, experts, mesh))
            log_path = os.path.join(
                out_dir, "logs", "scaling_%s_e%03d_m%02d.log" %
                (scenario_slug, experts, mesh))
            print("[scaling] %s experts=%d mesh=%dx%d" %
                  (scenario, experts, mesh, mesh), flush=True)
            cmd = [
                "--worker", "full", "--scaling-scenario", scenario,
                "--params", params_by_expert[experts],
                "--result", result_path, "--experts", str(experts),
                "--mesh", str(mesh), "--time-limit", str(args.time_limit),
                "--mipgap", str(args.mipgap), "--threads", str(args.threads),
                "--seed", str(args.seed), "--mip-focus", str(MIP_FOCUS),
            ]
            if args.reuse_scaling and os.path.isfile(result_path):
                with open(log_path, "r") as fh:
                    old_log = fh.read()
                proc = {"returncode": 0, "driver_observed_wall_s": 0.0,
                        "log": old_log}
            else:
                # A failed fresh worker must never leave a pre-contract JSON
                # looking like its result.
                if os.path.isfile(result_path):
                    os.remove(result_path)
                proc = _run_worker(args.gurobi_python, cmd, log_path)
            if os.path.isfile(result_path):
                row = _read_json(result_path)
                _validate_result_record_contract(row, "scaling")
                row.update(_parse_gurobi_log(proc["log"]))
            else:
                row = {
                    "contract_version": EXPERIMENT_CONTRACT_VERSION,
                    "kind": PERF_RESULT_KIND, "scenario": scenario,
                    "optimization_mode": PERF_MODE,
                    "active_objective": PERF_OBJECTIVE,
                    "mip_focus": MIP_FOCUS,
                    "experts": experts, "mesh": mesh, "sites": mesh * mesh,
                    "tasks": len(params_objects[experts]["tasks"]),
                    "edges": len(params_objects[experts]["edges"]),
                    "feasible": False, "status": "WORKER_FAILED",
                }
            row["worker_returncode"] = proc["returncode"]
            row["driver_observed_wall_s"] = proc["driver_observed_wall_s"]
            scaling_rows.append(row)
            scaling_trace.extend(_flatten_trace(row, "scaling"))

    qap_rows = []
    qap_trace = []
    # The 4-expert point checks all formulations cheaply; 8 and 16 expose
    # scaling.  Explicit products above 16 are projected, not allocated.
    qap_pairs = [(4, 3), (8, 6), (16, 8)]
    for experts, mesh in qap_pairs:
        if experts not in params_by_expert:
            path, obj = _generate_params(out_dir, experts)
            params_by_expert[experts] = path
            params_objects[experts] = obj
        for formulation in ("separable", "kb", "product"):
            result_path = os.path.join(
                out_dir, "results", "qap_%s_e%03d_m%02d.json" %
                (formulation, experts, mesh))
            log_path = os.path.join(
                out_dir, "logs", "qap_%s_e%03d_m%02d.log" %
                (formulation, experts, mesh))
            print("[qap] %s experts=%d mesh=%dx%d" %
                  (formulation, experts, mesh, mesh), flush=True)
            cmd = [
                "--worker", "qap", "--formulation", formulation,
                "--params", params_by_expert[experts], "--result", result_path,
                "--experts", str(experts), "--mesh", str(mesh),
                "--time-limit", str(args.time_limit),
                "--mipgap", str(args.mipgap), "--threads", str(args.threads),
                "--seed", str(args.seed), "--mip-focus", str(MIP_FOCUS),
            ]
            if args.reuse_qap and os.path.isfile(result_path):
                proc = {"returncode": 0, "driver_observed_wall_s": 0.0,
                        "log": ""}
            else:
                if os.path.isfile(result_path):
                    os.remove(result_path)
                proc = _run_worker(args.gurobi_python, cmd, log_path)
            if os.path.isfile(result_path):
                row = _read_json(result_path)
                _validate_result_record_contract(row, "qap")
            else:
                row = {
                    "contract_version": EXPERIMENT_CONTRACT_VERSION,
                    "kind": QAP_RESULT_KIND,
                    "comparison_role": QAP_SCOPE["role"],
                    "objective": QAP_OBJECTIVE,
                    "mip_focus": MIP_FOCUS,
                    "formulation": formulation,
                    "experts": experts, "mesh": mesh, "feasible": False,
                    "status": "WORKER_FAILED",
                }
            row["worker_returncode"] = proc["returncode"]
            row["driver_observed_wall_s"] = proc["driver_observed_wall_s"]
            qap_rows.append(row)
            qap_trace.extend(_flatten_trace(row, "qap"))

    scaling_fields = [
        "contract_version", "scaling_budget_contract_version", "scenario",
        "optimization_mode",
        "active_objective", "T_definition", "legacy_task_schedule_role",
        "protocol_completion_in_T", "protocol_used_as_separate_objective",
        "mip_focus",
        "cpu_anchor_policy", "experts", "mesh", "sites", "tasks", "edges",
        "budget_policy", "budget_area_mm2_per_expert",
        "budget_power_W_per_expert", "budget_target_bom",
        "gpu_counts_fixed", "gpu_l", "gpu_s", "status", "feasible",
        "model_variables", "model_binary_variables", "model_continuous_variables",
        "model_constraints", "model_nonzeros", "model_build_s",
        "first_feasible_s", "target_gap_time_s", "solve_wall_s",
        "total_worker_wall_s", "driver_observed_wall_s", "final_gap",
        "incumbent_cycles", "bound_cycles", "public_T_cycles",
        "protocol_completion_cycles", "legacy_T_task_cycles", "nodes", "area_mm2",
        "package_peak_power_W",
        "gpu_l_popularity_share", "top_hot_groups_on_gpu_l_fraction",
        "peak_rss_mb", "worker_returncode",
    ]
    qap_fields = [
        "contract_version", "formulation", "comparison_role", "objective",
        "mip_focus",
        "experts", "mesh", "sites", "tasks", "edges",
        "status", "feasible", "assignment_binaries",
        "auxiliary_variables_constructed", "model_variables",
        "model_binary_variables", "model_continuous_variables",
        "model_constraints", "model_nonzeros", "model_build_s",
        "first_feasible_s", "target_gap_time_s", "solve_wall_s",
        "total_worker_wall_s", "driver_observed_wall_s", "final_gap",
        "objective_byte_hops", "bound_byte_hops", "nodes", "peak_rss_mb",
        "worker_returncode",
    ]
    trace_fields = [
        "family", "formulation", "scenario", "experts", "mesh", "sequence", "wall_s",
        "incumbent", "bound", "gap", "nodes", "event",
    ]
    _write_csv(os.path.join(out_dir, "scaling_summary.csv"),
               scaling_rows, scaling_fields)
    _write_csv(os.path.join(out_dir, "scaling_gap_trace.csv"),
               scaling_trace, trace_fields)
    _write_csv(os.path.join(out_dir, "qap_summary.csv"), qap_rows, qap_fields)
    _write_csv(os.path.join(out_dir, "qap_gap_trace.csv"),
               qap_trace, trace_fields)
    projection = _projection_rows()
    _write_csv(os.path.join(out_dir, "product_preflight.csv"), projection,
               ["experts", "mesh", "tasks", "edges", "sites",
                "assignment_binaries", "product_auxiliary_binaries",
                "product_link_constraints", "constructed", "skip_reason"])
    metadata["plotting"] = _make_plots(
        out_dir, scaling_rows, qap_rows, scaling_trace + qap_trace)
    _write_json(os.path.join(out_dir, "metadata.json"), metadata)
    _write_readme(out_dir, metadata, scaling_rows, qap_rows)
    print("[done] wrote %s" % out_dir)
    return 0


def _parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("out", "scaling_qap_experiment"))
    ap.add_argument("--gurobi-python", default=DEFAULT_GUROBI_PYTHON)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--mipgap", type=float, default=0.01)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--mip-focus", type=int, choices=[MIP_FOCUS], default=MIP_FOCUS,
        help="fixed at 1 to prioritize finding/improving min-T incumbents; "
             "this solver setting does not change the objective")
    ap.add_argument("--reuse-qap", action="store_true",
                    help="reuse existing QAP worker JSONs while rerunning scaling")
    ap.add_argument("--reuse-scaling", action="store_true",
                    help="reuse existing scaling worker JSONs and logs")
    ap.add_argument("--scaling-scenarios",
                    choices=["both", "heterogeneous-main", "homogeneous-control"],
                    default="both",
                    help="full-model scenarios to include in the output CSV")
    # Internal isolated-worker interface.
    ap.add_argument("--worker", choices=["full", "qap"])
    ap.add_argument("--formulation", choices=["separable", "kb", "product"])
    ap.add_argument("--scaling-scenario",
                    choices=["heterogeneous-main", "homogeneous-control"],
                    default="homogeneous-control")
    ap.add_argument("--params")
    ap.add_argument("--result")
    ap.add_argument("--experts", type=int)
    ap.add_argument("--mesh", type=int)
    return ap


def main():
    args = _parser().parse_args()
    if args.mip_focus != MIP_FOCUS:
        raise SystemExit("this experiment contract requires --mip-focus 1")
    if args.worker == "full":
        return _worker_full(args)
    if args.worker == "qap":
        return _worker_qap(args)
    return _driver(args)


if __name__ == "__main__":
    sys.exit(main())
