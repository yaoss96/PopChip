#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the optimized prior-inspired baselines for one fresh ILP-Perf.

This is a small generic wrapper around :mod:`run_optimized_prior_proxies`.
Unlike that module's historical batch entry point, it accepts an arbitrary
parameter file and source solution.  Both generated layouts preserve the
source ILP-Perf chiplet BOM and every complete task bundle:

* Floorplet minimizes the transparent traffic-weighted Manhattan proxy by
  MILP; and
* TAP-Spread runs the transparent thermal/wirelength proxy by wall-clock SA
  and retains the best of the requested seeds.

The script must be launched with the Gurobi Python interpreter because the
Floorplet proxy is itself a MILP.  It performs no LEGOSim replay.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import shutil
import sys

import matched_layout_baseline as matched
import run_optimized_prior_proxies as proxies


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path, value):
    temporary = path + ".tmp.%d" % os.getpid()
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _parse_seeds(raw):
    values = []
    for item in str(raw).replace(" ", ",").split(","):
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("--tap-seeds must contain at least one integer")
    if len(set(values)) != len(values):
        raise ValueError("--tap-seeds contains duplicates")
    return values


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate Floorplet and TAP layouts from one ILP-Perf")
    parser.add_argument("--params", required=True)
    parser.add_argument("--source", required=True,
                        help="fresh ILP-Perf solution JSON")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--floorplet-time-limit", type=float, default=30.0)
    parser.add_argument("--tap-wall-budget", type=float, default=10.0)
    parser.add_argument("--tap-seeds", default="11,29,47")
    parser.add_argument("--tap-alpha", type=float, default=0.7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verbose-gurobi", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.floorplet_time_limit <= 0.0:
        raise ValueError("--floorplet-time-limit must be positive")
    if args.tap_wall_budget <= 0.0:
        raise ValueError("--tap-wall-budget must be positive")
    if not 0.0 <= args.tap_alpha <= 1.0:
        raise ValueError("--tap-alpha must be in [0,1]")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    seeds = _parse_seeds(args.tap_seeds)

    params_path = os.path.abspath(args.params)
    source_path = os.path.abspath(args.source)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isfile(params_path):
        raise ValueError("missing params: %s" % params_path)
    if not os.path.isfile(source_path):
        raise ValueError("missing source ILP-Perf: %s" % source_path)
    if os.path.exists(out_dir):
        raise ValueError("refusing to overwrite output directory: %s" % out_dir)

    params = _read_json(params_path)
    source = _read_json(source_path)
    if source.get("optimization_mode") != "ilp-perf":
        raise ValueError("--source must have optimization_mode=ilp-perf")
    state = matched._source_state(params, source)
    os.makedirs(out_dir)

    floor_solution, floor_meta = proxies.exact_floorplet(
        params, source, state, args.floorplet_time_limit, seed=1,
        verbose=args.verbose_gurobi, threads=args.threads)
    if floor_solution is None:
        raise RuntimeError("Floorplet proxy returned no feasible incumbent")
    floor_path = os.path.join(out_dir, "floorplet.json")
    _atomic_json(floor_path, floor_solution)

    ranges = proxies.deterministic_tap_ranges(params, state)
    tap_runs = []
    for seed in seeds:
        solution, metadata = proxies.tap_sa(
            params, source, state, args.tap_wall_budget, seed,
            args.tap_alpha, ranges)
        path = os.path.join(out_dir, "tap_seed%d.json" % seed)
        _atomic_json(path, solution)
        tap_runs.append((float(metadata["selection_cost"]), int(seed),
                         path, metadata))
    tap_runs.sort(key=lambda item: (item[0], item[1]))
    _, selected_seed, selected_source, selected_meta = tap_runs[0]
    tap_path = os.path.join(out_dir, "tap.json")
    shutil.copyfile(selected_source, tap_path)

    # Re-read and independently validate the selected artifacts through the
    # same strict evaluator used by the matched-random generator.
    validation = {}
    for method, path in (("floorplet", floor_path), ("tap", tap_path)):
        solution = _read_json(path)
        site_types = {
            (int(site["l"]), int(site["i"]), int(site["j"])): site["type"]
            for site in solution["sites"]}
        score, reasons = matched._strict_score(
            params, solution["task_site"], site_types, solution["grid"],
            solution.get("budgets") or {})
        if reasons:
            raise RuntimeError("%s strict validation failed: %s" %
                               (method, "; ".join(reasons)))
        if matched._type_counts(site_types) != state["type_counts"]:
            raise RuntimeError("%s changed the source chiplet BOM" % method)
        validation[method] = {
            "strict_feasible": bool(score.get("feasible")),
            "solution_path": os.path.basename(path),
            "solution_sha256": _sha256(path),
            "total_area_mm2": score.get("total_area_mm2"),
            "peak_power_W": score.get("peak_power_W"),
        }

    summary = {
        "schema_version": 1,
        "status": "complete",
        "case_id": args.case_id,
        "source_method": "ilp-perf",
        "source_solution": source_path,
        "source_solution_sha256": _sha256(source_path),
        "params": params_path,
        "params_sha256": _sha256(params_path),
        "protocol": (
            "exact source ILP-Perf BOM and immutable complete task bundles; "
            "physical placement only"),
        "claim_boundary": (
            "transparent prior-art-inspired objectives; original Floorplet "
            "and TAP-2.5D tools were not executed"),
        "floorplet": {
            "solution": "floorplet.json",
            "metadata": floor_meta,
            "validation": validation["floorplet"],
        },
        "tap": {
            "solution": "tap.json",
            "selected_seed": selected_seed,
            "selected_metadata": selected_meta,
            "runs": [{
                "seed": seed,
                "selection_cost": metadata["selection_cost"],
                "wall_s": metadata["wall_s"],
                "evaluations": metadata["evaluations"],
                "solution": os.path.basename(path),
                "solution_sha256": _sha256(path),
            } for _, seed, path, metadata in tap_runs],
            "validation": validation["tap"],
        },
    }
    summary_path = os.path.join(out_dir, "summary.json")
    _atomic_json(summary_path, summary)
    print(json.dumps({
        "status": "complete",
        "case_id": args.case_id,
        "floorplet": floor_path,
        "tap": tap_path,
        "tap_selected_seed": selected_seed,
        "summary": summary_path,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        sys.exit(2)
