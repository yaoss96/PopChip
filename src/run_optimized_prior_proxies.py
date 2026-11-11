#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimize the two transparent prior-art-inspired proxy objectives.

The protocol is identical to ``prior_layout_baseline.py``: preserve the source
ILP solution's exact BOM and every complete task bundle, and change physical
placement only.  Unlike the old best-of-N candidate selector:

* Floorplet-inspired F_FP is encoded as a MILP and solved with a bound/gap.
* TAP-Spread is directly optimized by wall-clock simulated annealing.

These remain inspired proxies on this repository's discrete Mesh.  They are
not executions of the original Floorplet or TAP-2.5D software.
"""

from __future__ import print_function

import argparse
import collections
import copy
import csv
import json
import math
import os
import random
import statistics
import sys
import time

try:
    from pathlib import Path
except ImportError:
    # The bundled Gurobi 10 Python 3.7 runtime on Windows omits urllib, which
    # makes its stdlib pathlib import fail.  Keep the experiment runnable in
    # that supported solver interpreter with the tiny path surface used here.
    class Path(object):
        def __init__(self, value):
            self.value = os.path.normpath(str(value))

        def __str__(self):
            return self.value

        def __truediv__(self, other):
            return Path(os.path.join(self.value, str(other)))

        def __fspath__(self):
            return self.value

        def is_absolute(self):
            return os.path.isabs(self.value)

        def resolve(self):
            return Path(os.path.abspath(self.value))

        def exists(self):
            return os.path.exists(self.value)

        def mkdir(self, parents=False, exist_ok=False):
            if parents:
                if exist_ok and os.path.isdir(self.value):
                    return
                try:
                    os.makedirs(self.value)
                except OSError:
                    if not (exist_ok and os.path.isdir(self.value)):
                        raise
            else:
                try:
                    os.mkdir(self.value)
                except OSError:
                    if not (exist_ok and os.path.isdir(self.value)):
                        raise

        @property
        def name(self):
            return os.path.basename(self.value)

        @property
        def parent(self):
            return Path(os.path.dirname(self.value))

        def with_name(self, name):
            return Path(os.path.join(os.path.dirname(self.value), name))

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:
    sys.stderr.write("ERROR: gurobipy is required: %s\n" % exc)
    sys.exit(2)

import evaluate as ev
import matched_layout_baseline as matched
import prior_layout_baseline as prior


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_EXPERIMENT = Path(
    "out/budget_mesh_experiment/newrepo_mesh_budget_full_20260731")
DEFAULT_OUTPUT = Path("out/optimized_prior_proxy_audit_20260731")
EPS = 1.0e-12


def read_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(str(temporary), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def atomic_csv(path, fieldnames, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(str(temporary), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def footprint(kind, anchor, library):
    return set(matched._footprint(kind, anchor, library))


def distance(left, right):
    return sum(abs(int(left[axis]) - int(right[axis])) for axis in range(3))


def bundle_tasks(bundle):
    tasks = list(bundle["base_tasks"])
    if bundle.get("original_expert") is not None:
        tasks.append(bundle["original_expert"])
    return sorted(tasks)


def bundle_signature(solution):
    return sorted((str(site["type"]), tuple(sorted(site.get("tasks") or [])))
                  for site in solution.get("sites", []))


def build_solution(params, source, state, bundle_anchor, method, metadata):
    task_site = {}
    sites = []
    for bundle in state["bundles"]:
        anchor = tuple(bundle_anchor[bundle["id"]])
        names = bundle_tasks(bundle)
        for task in names:
            task_site[task] = list(anchor)
        sites.append({"l": anchor[0], "i": anchor[1], "j": anchor[2],
                      "type": bundle["type"], "tasks": names})
    sites.sort(key=lambda site: (site["l"], site["i"], site["j"],
                                 site["type"]))
    solution = {
        "method": method,
        "proxy_not_original_tool": True,
        "search_space": "exact BOM and immutable complete task bundles",
        "sites": sites,
        "task_site": dict((task, task_site[task])
                          for task in sorted(task_site)),
        "grid": copy.deepcopy(state["grid"]),
        "budgets": copy.deepcopy(state["budgets"]),
        "metadata": metadata,
    }
    types = ev.site_types_from_solution(solution)
    score, reasons = matched._strict_score(
        params, solution["task_site"], types, state["grid"], state["budgets"])
    if reasons:
        raise RuntimeError("%s failed strict evaluation: %s" %
                           (method, "; ".join(reasons)))
    if matched._type_counts(types) != state["type_counts"]:
        raise RuntimeError("%s changed exact BOM" % method)
    if bundle_signature(solution) != bundle_signature(source):
        raise RuntimeError("%s changed a complete task bundle" % method)
    solution["strict_evaluate"] = score
    return solution


def packing_feasible(params, state, bundle_anchor):
    library = params["library"]
    grid = state["grid"]
    occupied = set()
    layer_area = collections.defaultdict(float)
    column_power = collections.defaultdict(float)
    for bundle in state["bundles"]:
        kind = bundle["type"]
        anchor = tuple(bundle_anchor[bundle["id"]])
        if anchor not in matched._legal_anchors(kind, grid, library):
            return False
        cells = footprint(kind, anchor, library)
        if cells & occupied:
            return False
        occupied.update(cells)
        layer_area[anchor[0]] += float(library[kind]["area"])
        density = float(library[kind]["p_peak"]) / float(len(cells))
        for cell in cells:
            column_power[(cell[1], cell[2])] += density
    budgets = state["budgets"]
    if budgets.get("a_max_layer") is not None:
        cap = float(budgets["a_max_layer"])
        if any(value > cap + 1.0e-6 for value in layer_area.values()):
            return False
    if budgets.get("p_col") is not None:
        cap = float(budgets["p_col"])
        if any(value > cap + 1.0e-6 for value in column_power.values()):
            return False
    return True


def exact_floorplet(params, source, state, time_limit, seed, verbose=False,
                    threads=None):
    library = params["library"]
    bundles = state["bundles"]
    grid = state["grid"]
    bundle_by_task = {}
    for bundle in bundles:
        for task in bundle_tasks(bundle):
            bundle_by_task[task] = bundle["id"]
    pair_weight = collections.defaultdict(float)
    for edge in params["edges"]:
        left = bundle_by_task[edge["src"]]
        right = bundle_by_task[edge["dst"]]
        if left == right:
            continue
        key = tuple(sorted((left, right)))
        pair_weight[key] += float(edge["vol"])
    total_flow = max(sum(pair_weight.values()), 1.0)

    anchors = {}
    for bundle in bundles:
        anchors[bundle["id"]] = matched._legal_anchors(
            bundle["type"], grid, library)
    model = gp.Model("exact_floorplet_proxy")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = 0.0
    model.Params.Seed = int(seed)
    model.Params.Symmetry = 2
    if threads is not None:
        model.Params.Threads = int(threads)
    z = {}
    for bundle in bundles:
        bid = bundle["id"]
        for anchor in anchors[bid]:
            z[(bid, anchor)] = model.addVar(vtype=GRB.BINARY,
                                             name="z_%d_%d_%d_%d" %
                                             ((bid,) + anchor))
    model.update()
    # The source layout is always a legal fixed-BOM placement.  Supplying it
    # as a MIP start guarantees a feasible incumbent even for tight, large
    # scale-out meshes where a short Floorplet run may not discover one from
    # scratch.
    for bundle in bundles:
        bid = bundle["id"]
        old_anchor = tuple(bundle["old_anchor"])
        for anchor in anchors[bid]:
            z[(bid, anchor)].Start = 1.0 if anchor == old_anchor else 0.0
    for bundle in bundles:
        bid = bundle["id"]
        model.addConstr(gp.quicksum(z[(bid, anchor)]
                                    for anchor in anchors[bid]) == 1,
                        name="one_%d" % bid)

    for layer in range(int(grid["L"])):
        for row in range(int(grid["X"])):
            for col in range(int(grid["Y"])):
                cell = (layer, row, col)
                terms = []
                for bundle in bundles:
                    bid = bundle["id"]
                    kind = bundle["type"]
                    for anchor in anchors[bid]:
                        if cell in footprint(kind, anchor, library):
                            terms.append(z[(bid, anchor)])
                if terms:
                    model.addConstr(gp.quicksum(terms) <= 1,
                                    name="occupy_%d_%d_%d" % cell)

    if state["budgets"].get("a_max_layer") is not None:
        cap = float(state["budgets"]["a_max_layer"])
        for layer in range(int(grid["L"])):
            model.addConstr(gp.quicksum(
                float(library[bundle["type"]]["area"]) * z[(bundle["id"], anchor)]
                for bundle in bundles for anchor in anchors[bundle["id"]]
                if anchor[0] == layer) <= cap,
                name="layer_area_%d" % layer)
    if state["budgets"].get("p_col") is not None:
        cap = float(state["budgets"]["p_col"])
        for row in range(int(grid["X"])):
            for col in range(int(grid["Y"])):
                terms = []
                for bundle in bundles:
                    bid = bundle["id"]
                    kind = bundle["type"]
                    density = (float(library[kind]["p_peak"]) /
                               float(int(library[kind]["w"]) *
                                     int(library[kind]["h"])))
                    for anchor in anchors[bid]:
                        cells = footprint(kind, anchor, library)
                        if any(cell[1] == row and cell[2] == col
                               for cell in cells):
                            terms.append(density * z[(bid, anchor)])
                if terms:
                    model.addConstr(gp.quicksum(terms) <= cap,
                                    name="pcol_%d_%d" % (row, col))

    def coordinate(bid, axis):
        return gp.quicksum(anchor[axis] * z[(bid, anchor)]
                           for anchor in anchors[bid])

    objective_terms = []
    distance_vars = {}
    for pair, weight in sorted(pair_weight.items()):
        left, right = pair
        for axis in range(3):
            variable = model.addVar(lb=0.0, name="d_%d_%d_%d" %
                                    (left, right, axis))
            distance_vars[(pair, axis)] = variable
            model.addConstr(variable >= coordinate(left, axis) -
                            coordinate(right, axis))
            model.addConstr(variable >= coordinate(right, axis) -
                            coordinate(left, axis))
            objective_terms.append((weight / total_flow) * variable)
    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)
    trace = []

    def callback(native, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            trace.append({
                "t_s": float(native.cbGet(GRB.Callback.RUNTIME)),
                "incumbent_F_FP": float(
                    native.cbGet(GRB.Callback.MIPSOL_OBJ)) * total_flow,
                "bound_F_FP": float(
                    native.cbGet(GRB.Callback.MIPSOL_OBJBND)) * total_flow,
            })
        except Exception:
            pass

    started = time.perf_counter()
    model.optimize(callback)
    wall = time.perf_counter() - started
    if model.SolCount == 0:
        return None, {
            "status": int(model.Status), "wall_s": wall,
            "certified_exact": False, "trace": trace,
        }
    selected = {}
    for bundle in bundles:
        bid = bundle["id"]
        choices = [anchor for anchor in anchors[bid]
                   if z[(bid, anchor)].X > 0.5]
        if len(choices) != 1:
            raise RuntimeError("MILP bundle %d has %d selected anchors" %
                               (bid, len(choices)))
        selected[bid] = choices[0]
    raw_objective = 0.0
    for pair, weight in pair_weight.items():
        raw_objective += weight * distance(selected[pair[0]],
                                           selected[pair[1]])
    try:
        gap = float(model.MIPGap)
        bound = float(model.ObjBound) * total_flow
    except Exception:
        gap, bound = None, None
    metadata = {
        "optimizer": "exact MILP encoding",
        "objective": "F_FP traffic-weighted Manhattan distance",
        "F_FP": raw_objective,
        "bound_F_FP": bound,
        "mip_gap": gap,
        "status": int(model.Status),
        "wall_s": wall,
        "time_to_first_feasible_s": (
            trace[0]["t_s"] if trace else None),
        "time_limit_s": time_limit,
        "certified_exact": (int(model.Status) == int(GRB.OPTIMAL) and
                            gap is not None and gap <= 1.0e-10),
        "trace": trace,
    }
    solution = build_solution(params, source, state, selected,
                              "floorplet-inspired-exact-MILP", metadata)
    recomputed = prior._traffic_wirelength(params, solution)
    if abs(recomputed - raw_objective) > 1.0e-6:
        raise RuntimeError("F_FP objective extraction mismatch")
    return solution, metadata


def tap_terms(params, state, bundle_anchor):
    library = params["library"]
    grid = state["grid"]
    bundle_of_task = {}
    sources = []
    for bundle in state["bundles"]:
        bid = bundle["id"]
        anchor = tuple(bundle_anchor[bid])
        for task in bundle_tasks(bundle):
            bundle_of_task[task] = bid
        cells = footprint(bundle["type"], anchor, library)
        density = float(library[bundle["type"]]["p_peak"]) / float(len(cells))
        for cell in cells:
            sources.append((cell, density))
    peak = 0.0
    for layer in range(int(grid["L"])):
        for row in range(int(grid["X"])):
            for col in range(int(grid["Y"])):
                query = (layer, row, col)
                value = sum(density / float((1 + distance(query, cell)) ** 2)
                            for cell, density in sources)
                peak = max(peak, value)
    wire = 0.0
    for edge in params["edges"]:
        left = bundle_anchor[bundle_of_task[edge["src"]]]
        right = bundle_anchor[bundle_of_task[edge["dst"]]]
        wire += distance(left, right)
    return peak, wire


def deterministic_tap_ranges(params, state):
    library = params["library"]
    densities = []
    for bundle in state["bundles"]:
        kind = bundle["type"]
        cells = int(library[kind]["w"]) * int(library[kind]["h"])
        densities.extend([float(library[kind]["p_peak"]) / float(cells)] *
                         cells)
    h_min = max(densities) if densities else 0.0
    h_max = sum(densities)
    grid = state["grid"]
    diameter = ((int(grid["L"]) - 1) + (int(grid["X"]) - 1) +
                (int(grid["Y"]) - 1))
    w_min = 0.0
    w_max = float(len(params["edges"]) * diameter)
    return {"thermal": [h_min, max(h_max, h_min + EPS)],
            "wirelength": [w_min, max(w_max, EPS)],
            "method": "deterministic valid bounds from fixed BOM and grid"}


def tap_cost(thermal, wire, ranges, alpha):
    h_lo, h_hi = ranges["thermal"]
    w_lo, w_hi = ranges["wirelength"]
    h_norm = (thermal - h_lo) / max(h_hi - h_lo, EPS)
    w_norm = (wire - w_lo) / max(w_hi - w_lo, EPS)
    return alpha * h_norm + (1.0 - alpha) * w_norm, h_norm, w_norm


def mutate_packing(params, state, current, rng):
    bundles = state["bundles"]
    library = params["library"]
    for _ in range(48):
        candidate = dict(current)
        if rng.random() < 0.72:
            bundle = rng.choice(bundles)
            candidate[bundle["id"]] = rng.choice(matched._legal_anchors(
                bundle["type"], state["grid"], library))
        else:
            left, right = rng.sample(bundles, 2)
            candidate[left["id"]], candidate[right["id"]] = (
                candidate[right["id"]], candidate[left["id"]])
        if packing_feasible(params, state, candidate):
            return candidate
    return None


def tap_sa(params, source, state, wall_budget, seed, alpha, ranges):
    rng = random.Random(seed)
    started = time.perf_counter()
    current, packing_strategy = matched._random_packing(params, state, rng)
    thermal, wire = tap_terms(params, state, current)
    current_cost, h_norm, w_norm = tap_cost(thermal, wire, ranges, alpha)
    best = dict(current)
    best_tuple = (current_cost, thermal, wire, h_norm, w_norm)
    evaluations = 1
    trace = [{"t_s": time.perf_counter() - started,
              "evaluations": evaluations, "best_cost": current_cost,
              "thermal": thermal, "wirelength": wire}]
    while time.perf_counter() - started < wall_budget:
        progress = min((time.perf_counter() - started) / wall_budget, 1.0)
        temperature = 0.05 * (1.0e-5 ** progress)
        candidate = mutate_packing(params, state, current, rng)
        if candidate is None:
            continue
        cand_thermal, cand_wire = tap_terms(params, state, candidate)
        cand_cost, cand_h_norm, cand_w_norm = tap_cost(
            cand_thermal, cand_wire, ranges, alpha)
        evaluations += 1
        delta = cand_cost - current_cost
        if delta <= 0.0 or rng.random() < math.exp(
                -delta / max(temperature, EPS)):
            current, current_cost = candidate, cand_cost
        if cand_cost < best_tuple[0] - EPS:
            best = dict(candidate)
            best_tuple = (cand_cost, cand_thermal, cand_wire,
                          cand_h_norm, cand_w_norm)
            trace.append({"t_s": time.perf_counter() - started,
                          "evaluations": evaluations, "best_cost": cand_cost,
                          "thermal": cand_thermal, "wirelength": cand_wire})
    wall = time.perf_counter() - started
    cost, thermal, wire, h_norm, w_norm = best_tuple
    metadata = {
        "optimizer": "wall-clock simulated annealing",
        "objective": "0.7 normalized thermal coupling + 0.3 normalized wirelength",
        "alpha": alpha,
        "normalization": copy.deepcopy(ranges),
        "selection_cost": cost,
        "thermal_coupling_surrogate": thermal,
        "unweighted_net_wirelength": wire,
        "normalized_thermal": h_norm,
        "normalized_wirelength": w_norm,
        "wall_budget_s": wall_budget,
        "wall_s": wall,
        "time_to_first_feasible_s": (
            trace[0]["t_s"] if trace else None),
        "seed": seed,
        "evaluations": evaluations,
        "packing_strategy": packing_strategy,
        "trace": trace,
    }
    solution = build_solution(params, source, state, best,
                              "tap-spread-SA", metadata)
    return solution, metadata


def legacy_records(path):
    if path is None:
        return {}
    raw_path = Path(path) / "raw_results.json"
    if not raw_path.exists():
        return {}
    raw = read_json(raw_path)
    records = {}
    for group in raw.get("layout_groups", []):
        for case_id in group.get("mapped_case_ids", []):
            records[case_id] = group.get("layouts", {})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exact Floorplet proxy MILP and TAP-Spread SA")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--experiment", default=str(DEFAULT_EXPERIMENT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--area-only", action="store_true")
    parser.add_argument("--floorplet-time-limit", type=float, default=30.0)
    parser.add_argument("--tap-wall-budget", type=float, default=2.0)
    parser.add_argument("--tap-seeds", default="11,29,47")
    parser.add_argument("--tap-alpha", type=float, default=0.7)
    parser.add_argument("--legacy-proxy-experiment", default=None)
    parser.add_argument("--verbose-gurobi", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    experiment = Path(args.experiment)
    if not experiment.is_absolute():
        experiment = project_root / experiment
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir = out_dir / "solutions"
    solutions_dir.mkdir(exist_ok=True)
    raw = read_json(experiment / "raw_results.json")
    params = read_json(experiment / "params_mixtral.json")
    seeds = [int(value) for value in args.tap_seeds.replace(" ", ",").split(",")
             if value]
    old = legacy_records(args.legacy_proxy_experiment)
    requested = set(args.case)
    result_cases = []
    rows = []
    anytime_rows = []

    for case in raw["cases"]:
        case_id = case["case_id"]
        if requested and case_id not in requested:
            continue
        if args.area_only and case.get("sweep_dimension") != "area":
            continue
        source_path = experiment / "cases" / case_id / "ilp_solution.json"
        source = read_json(source_path)
        state = matched._source_state(params, source)
        ranges = deterministic_tap_ranges(params, state)
        case_record = {"case_id": case_id,
                       "sweep_dimension": case.get("sweep_dimension"),
                       "source_solution": str(source_path.resolve()),
                       "exact_bom": copy.deepcopy(state["type_counts"]),
                       "complete_bundle_signature": bundle_signature(source),
                       "tap_normalization": ranges,
                       "floorplet": None, "tap_runs": [],
                       "legacy_best_of_20000": old.get(case_id)}
        print("[case] %s" % case_id, flush=True)
        floor_solution, floor_meta = exact_floorplet(
            params, source, state, args.floorplet_time_limit, seed=1,
            verbose=args.verbose_gurobi)
        if floor_solution is not None:
            floor_path = solutions_dir / (case_id + "_floorplet_exact.json")
            atomic_json(floor_path, floor_solution)
            floor_meta["solution_path"] = str(floor_path.resolve())
        case_record["floorplet"] = floor_meta
        rows.append({
            "case_id": case_id, "method": "floorplet-exact-MILP",
            "seed": 1, "objective": floor_meta.get("F_FP"),
            "secondary_objective": None,
            "bound": floor_meta.get("bound_F_FP"),
            "mip_gap": floor_meta.get("mip_gap"),
            "certified_exact": floor_meta.get("certified_exact"),
            "wall_s": floor_meta.get("wall_s"),
            "time_to_first_feasible_s": floor_meta.get(
                "time_to_first_feasible_s"),
            "evaluations": None,
            "strict_feasible": (floor_solution or {}).get(
                "strict_evaluate", {}).get("feasible"),
            "solution_path": floor_meta.get("solution_path"),
        })
        print("  Floorplet F_FP=%s gap=%s wall=%.3fs" %
              (floor_meta.get("F_FP"), floor_meta.get("mip_gap"),
               floor_meta.get("wall_s", 0.0)), flush=True)

        for seed in seeds:
            tap_solution, tap_meta = tap_sa(
                params, source, state, args.tap_wall_budget, seed,
                args.tap_alpha, ranges)
            tap_path = solutions_dir / ("%s_tap_sa_seed%d.json" %
                                         (case_id, seed))
            atomic_json(tap_path, tap_solution)
            tap_meta["solution_path"] = str(tap_path.resolve())
            case_record["tap_runs"].append(tap_meta)
            rows.append({
                "case_id": case_id, "method": "tap-spread-SA",
                "seed": seed, "objective": tap_meta["selection_cost"],
                "secondary_objective": tap_meta["thermal_coupling_surrogate"],
                "bound": None, "mip_gap": None, "certified_exact": False,
                "wall_s": tap_meta["wall_s"],
                "time_to_first_feasible_s": tap_meta.get(
                    "time_to_first_feasible_s"),
                "evaluations": tap_meta["evaluations"],
                "strict_feasible": tap_solution["strict_evaluate"]["feasible"],
                "solution_path": str(tap_path.resolve()),
            })
            for point in tap_meta["trace"]:
                record = dict(point)
                record.update({"case_id": case_id, "method": "tap-spread-SA",
                               "seed": seed})
                anytime_rows.append(record)
            print("  TAP seed=%d cost=%.8f H=%.6f W=%.1f eval=%d" %
                  (seed, tap_meta["selection_cost"],
                   tap_meta["thermal_coupling_surrogate"],
                   tap_meta["unweighted_net_wirelength"],
                   tap_meta["evaluations"]), flush=True)
        result_cases.append(case_record)

    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": {
            "source_experiment": str(experiment.resolve()),
            "claim_boundary": (
                "prior-art-inspired objectives on discrete Mesh; original "
                "Floorplet/TAP-2.5D tools were not executed"),
            "protocol": (
                "exact source ILP BOM and immutable complete task bundles; "
                "physical placement only"),
            "floorplet": (
                "MILP minimizes F_FP and reports incumbent/bound/gap"),
            "tap": (
                "SA directly minimizes deterministic-normalized TAP-Spread; "
                "same wall-clock for every seed"),
            "tap_alpha": args.tap_alpha,
            "tap_wall_budget_s": args.tap_wall_budget,
            "tap_seeds": seeds,
            "floorplet_time_limit_s": args.floorplet_time_limit,
            "legacy_proxy_experiment": args.legacy_proxy_experiment,
        },
        "cases": result_cases,
    }
    atomic_json(out_dir / "summary.json", summary)
    atomic_csv(out_dir / "results.csv", [
        "case_id", "method", "seed", "objective", "secondary_objective",
        "bound", "mip_gap", "certified_exact", "wall_s", "evaluations",
        "time_to_first_feasible_s", "strict_feasible", "solution_path"], rows)
    atomic_csv(out_dir / "tap_anytime.csv", [
        "case_id", "method", "seed", "t_s", "evaluations", "best_cost",
        "thermal", "wirelength"], anytime_rows)
    lines = [
        "# Optimized prior-inspired proxy audit", "",
        "Floorplet-inspired is an exact MILP encoding of F_FP. TAP-Spread is "
        "now directly optimized by SA. Both preserve the exact source BOM and "
        "every complete task bundle.", "",
        "These are still transparent inspired proxies, not runs of the original "
        "authors' tools.", "",
        "Any newly selected coordinates require fresh code generation and "
        "LEGOSim/PopNet replay before replacing end-to-end cycles in a paper; "
        "proxy-objective improvements alone do not imply performance gains.", "",
        "| case | exact F_FP | certified? | TAP median cost |", 
        "|---|---:|---:|---:|",
    ]
    for case in result_cases:
        tap_values = [run["selection_cost"] for run in case["tap_runs"]]
        lines.append("| %s | %.3f | %s | %.8f |" % (
            case["case_id"], case["floorplet"].get("F_FP", float("nan")),
            "yes" if case["floorplet"].get("certified_exact") else "no",
            statistics.median(tap_values)))
    lines.append("")
    with open(str(out_dir / "README.md"), "w", encoding="utf-8",
              newline="\n") as handle:
        handle.write("\n".join(lines))
    print("[done] %s" % (out_dir / "summary.json"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
