# -*- coding: utf-8 -*-
"""Shared helpers for the experiment scripts (run with a gurobipy interpreter).

Every helper calls the unmodified ``src/ilp_layout.py``, so the analytical T
reported by these scripts is the paper's T.  Inputs live under ``data/`` and
outputs are written under ``results/``.
"""
from __future__ import print_function
import csv
import io
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
PARAMS_DIR = os.path.join(DATA, "params")
LAYOUTS = os.path.join(DATA, "layouts")
REALTRACE = os.path.join(DATA, "realtrace")
MEASURED = os.path.join(RESULTS, "legosim", "measured.csv")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import ilp_layout as il  # noqa: E402

EXPERT = re.compile(r"^e\((\d+)\)$")

# Table-II instances: id -> (experts, mesh side, A_max, P_max, primary time limit)
CASES = {
    "e008_m06": (8, 6, 1440.0, 1120.0, 45.0),
    "e016_m08": (16, 8, 2880.0, 2240.0, 45.0),
    "e032_m12": (32, 12, 5760.0, 4480.0, 45.0),
    "e064_m12": (64, 12, 11520.0, 8960.0, 45.0),
    "e064_m16": (64, 16, 11520.0, 8960.0, 240.0),
    "e128_m16": (128, 16, 23040.0, 17920.0, 45.0),
}


def read_json(path):
    with io.open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, value):
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        os.makedirs(d)
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(value, indent=1, sort_keys=True))
    os.replace(tmp, path)


def case_params(cid):
    return read_json(os.path.join(PARAMS_DIR, "experts_%03d.json" % CASES[cid][0]))


def case_paths(cid):
    """Replayed Table-II layouts of one instance (PopChip, proxies, blind)."""
    d = os.path.join(LAYOUTS, cid)
    out = {m: os.path.join(d, m + ".json") for m in ("popchip", "floorplet", "tap")}
    for s in (1, 2, 3):
        out["blind%d" % s] = os.path.join(d, "blind%d.json" % s)
    return out


def read_measured():
    """Rows of results/legosim/measured.csv with absolute layout/params paths."""
    rows = []
    with io.open(MEASURED, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["measured_cycles"] = int(r["measured_cycles"])
            r["layout"] = os.path.join(ROOT, r["layout"])
            r["params"] = os.path.join(ROOT, r["params"])
            rows.append(r)
    return rows


def budgets(a_max, p_max, nmin=None, nmax=None):
    b = {"a_max": float(a_max), "a_max_layer": None, "p_max": float(p_max),
         "nmin": {"CPU": 1}, "nmax": {"CPU": 1}}
    for k, v in (nmin or {}).items():
        b["nmin"][k] = int(v)
    for k, v in (nmax or {}).items():
        b["nmax"][k] = int(v)
    return b


def solver_opts(time_limit, mipgap=0.01, seed=1, geom=False, hop=None,
                flit=None, extra=None):
    o = {"time_limit": float(time_limit), "mipgap": float(mipgap),
         "mipfocus": 1, "seed": seed, "symmetry_break": True,
         "mode": "ilp-perf", "noc_clock_hz": None,
         "noc_traffic": "deterministic-dispatch",
         "dispatch_tokens_per_source": 32, "dispatch_hidden": 128,
         "dispatch_bytes_elem": 2, "threads": 1,
         "stage0_protocol_evaluations": None, "stage0_protocol_wall_s": None,
         "stage0_protocol_rounds": None, "cpu_anchor": "free",
         "bom_counts": None, "placement_bundles": None,
         "warm_start_solution": None, "performance_reference_solution": None,
         "reference_source": None, "geom_symmetry_break": bool(geom)}
    if hop is not None:
        o["protocol_hop_cycles"] = float(hop)
    if flit is not None:
        o["protocol_flit_bytes"] = float(flit)
    if extra:
        o.update(extra)
    return o


def counts(sol):
    c = {}
    for s in sol["sites"]:
        c[s["type"]] = c.get(s["type"], 0) + 1
    return c


def summary(sol, wall=None):
    if sol is None:
        return {"feasible": False}
    c = counts(sol)
    pm = sol.get("protocol_model") or {}
    ph = pm.get("phase_completion") or {}
    return {
        "feasible": True,
        "nL": c.get("GPU-L", 0), "nS": c.get("GPU-S", 0),
        "T": sol.get("predicted_makespan_cycles"),
        "bound": sol.get("primary_objective_bound_cycles"),
        "gap": sol.get("mip_gap"),
        "ttff": sol.get("time_to_first_feasible_s"),
        "wall": sol.get("solver_wall_s") if wall is None else wall,
        "nodes": sol.get("primary_node_count"),
        "area": sol.get("total_area_mm2"), "power": sol.get("peak_power_W"),
        "status": sol.get("solver_status"),
        "link_cycles": {k: v.get("link_cycles") for k, v in ph.items()},
        "endpoint_cycles": {k: v.get("endpoint_cycles") for k, v in ph.items()},
        "l_experts": sorted(
            int(EXPERT.match(t).group(1)) for s in sol["sites"]
            if s["type"] == "GPU-L" for t in s.get("tasks", []) if EXPERT.match(t)),
    }


def solve(params, X, a_max, p_max, tl, nmin=None, nmax=None, **kw):
    b = budgets(a_max, p_max, nmin, nmax)
    o = solver_opts(tl, **kw)
    t0 = time.time()
    sol, stats = il.build_and_solve(params, 1, X, X, b, o, verbose=False)
    return sol, time.time() - t0


def evaluate_fixed(params, sol, hop=None, flit=None):
    """Analytical T of a complete serialized layout (same Gurobi model)."""
    X = int(sol["grid"]["X"])
    o = solver_opts(900.0, mipgap=1e-9, hop=hop, flit=flit,
                    extra={"fixed_solution": sol})
    s, stats = il.build_and_solve(params, 1, X, X, budgets(1e9, 1e9), o,
                                  verbose=False)
    if s is None:
        return None
    return s.get("predicted_makespan_cycles")
