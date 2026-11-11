#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feed a serialized layout back into the native C1--C13 Gurobi model."""

from __future__ import print_function

import argparse
import json
import os
import sys

import evaluate
import ilp_layout


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def independent_c11_c13(params, solution):
    tasks = params["tasks"]
    edges = params["edges"]
    ic = params["interconnect"]
    site_types = evaluate.site_types_from_solution(solution)
    task_site = dict((task, tuple(anchor))
                     for task, anchor in solution["task_site"].items())
    predecessors = dict((task, []) for task in tasks)
    successors = dict((task, []) for task in tasks)
    indegree = dict((task, 0) for task in tasks)
    for edge in edges:
        predecessors[edge["dst"]].append(edge)
        successors[edge["src"]].append(edge["dst"])
        indegree[edge["dst"]] += 1
    queue = [task for task in tasks if indegree[task] == 0]
    order = []
    while queue:
        task = queue.pop()
        order.append(task)
        for target in successors[task]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    finish = {}
    for task in order:
        anchor = task_site[task]
        kind = site_types[anchor]
        start = 0.0
        for edge in predecessors[task]:
            source_anchor = task_site[edge["src"]]
            horizontal = (abs(source_anchor[1] - anchor[1]) +
                          abs(source_anchor[2] - anchor[2]))
            vertical = abs(source_anchor[0] - anchor[0])
            link = (float(ic["t0"]) + float(ic["tau_h"]) * horizontal +
                    float(ic["tau_v"]) * vertical +
                    float(edge["vol"]) / float(ic["beta"]))
            start = max(start, finish[edge["src"]] + link)
        finish[task] = start + float(tasks[task]["cyc"][kind])
    busy = {}
    for task, anchor in task_site.items():
        kind = site_types[anchor]
        busy[anchor] = (busy.get(anchor, 0.0) +
                        float(tasks[task]["cyc"][kind]))
    return max(max(finish.values()), max(busy.values()))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--time-limit", type=float, default=30.0)
    args = parser.parse_args(argv)
    params = read_json(args.params)
    candidate = read_json(args.solution)
    grid = candidate["grid"]
    budgets = candidate["budgets"]
    site_types = evaluate.site_types_from_solution(candidate)
    strict = evaluate.evaluate(
        params, candidate["task_site"], site_types, grid,
        strict=True, budgets=budgets)
    independent = independent_c11_c13(params, candidate)
    solver_options = {
        "time_limit": args.time_limit,
        "mipgap": 0.0,
        "seed": 1,
        "symmetry_break": True,
        "mode": "certified-latency",
        "secondary_time": 0.0,
        "secondary_mipgap": 0.0,
        "noc_traffic": "task-dag",
        "dispatch_tokens_per_source": 32,
        "dispatch_hidden": 128,
        "dispatch_bytes_elem": params.get("meta", {}).get("bytes_elem", 2),
        "fixed_solution": candidate,
    }
    native, stats = ilp_layout.build_and_solve(
        params, int(grid["L"]), int(grid["X"]), int(grid["Y"]),
        budgets, solver_options, T_hat=None, verbose=False)
    output = {
        "params": os.path.abspath(args.params),
        "candidate": os.path.abspath(args.solution),
        "strict_evaluate": strict,
        "independent_c11_c13_cycles": independent,
        "native_model_stats": stats,
        "native_model_feasible": native is not None,
        "native_fixed_c11_c13_cycles": (
            native.get("primary_incumbent_cycles") if native else None),
        "native_fixed_primary_bound_cycles": (
            native.get("primary_objective_bound_cycles") if native else None),
        "native_fixed_primary_gap": (
            native.get("primary_mip_gap") if native else None),
        "absolute_difference_cycles": (
            abs(float(native.get("primary_incumbent_cycles")) - independent)
            if native else None),
        "all_C1_through_C13_native_constraints_satisfied": native is not None,
    }
    write_json(args.out, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if native is not None else 1


if __name__ == "__main__":
    sys.exit(main())
