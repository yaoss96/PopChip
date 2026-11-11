#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate exact-BOM prior-art-objective placement proxies.

The original Floorplet and TAP-2.5D tools target different physical-design
inputs from this repository and cannot be invoked directly on an MoE task
graph.  This script therefore implements *prior-art-inspired proxies*:

* ``floorplet-inspired`` minimizes traffic-weighted Manhattan wirelength,
  matching Floorplet's performance-aware communication-cost term.
* ``tap-2.5d-inspired`` minimizes a normalized combination of a static
  power-coupling thermal surrogate and unweighted net wirelength, matching
  TAP-2.5D's temperature/wirelength design intent.

Both proxies search the same deterministic pool of legal layouts.  Every
candidate preserves the source ILP solution's exact chiplet BOM, the complete
task bundle on every chiplet, area, package peak-power-model value, and budget
constraints.  Only whole chiplet bundles move.  This deliberately gives the
prior-placement proxies the ILP mapping, so the comparison isolates physical
placement and is conservative with respect to the proposed joint optimizer.
The emitted artifacts are explicitly labeled as proxies and must not be
described as runs of the authors' original implementations.
"""

from __future__ import print_function

import argparse
import copy
import json
import os
import sys
import time

import evaluate as ev
import matched_layout_baseline as matched


_EPS = 1.0e-12


def _distance(a, b):
    return (abs(int(a[0]) - int(b[0])) +
            abs(int(a[1]) - int(b[1])) +
            abs(int(a[2]) - int(b[2])))


def _traffic_wirelength(params, solution):
    task_site = solution["task_site"]
    total = 0.0
    for edge in params["edges"]:
        total += (float(edge["vol"]) *
                  _distance(task_site[edge["src"]],
                            task_site[edge["dst"]]))
    return total


def _net_wirelength(params, solution):
    task_site = solution["task_site"]
    return sum(_distance(task_site[edge["src"]],
                         task_site[edge["dst"]])
               for edge in params["edges"])


def _thermal_coupling(params, solution):
    """Static placement-dependent peak power-coupling surrogate.

    Chiplet power is spread uniformly over its occupied Mesh cells.  At each
    grid cell q we accumulate every occupied cell's density with a
    1/(1+d)^2 Manhattan decay and report the maximum.  This is deliberately
    called a surrogate: it encourages the high-power-die separation central
    to TAP-2.5D, but is not a HotSpot temperature.
    """
    library = params["library"]
    grid = solution["grid"]
    sources = []
    for site in solution["sites"]:
        anchor = (int(site["l"]), int(site["i"]), int(site["j"]))
        kind = site["type"]
        cells = matched._footprint(kind, anchor, library)
        density = float(library[kind]["p_peak"]) / float(len(cells))
        for cell in cells:
            sources.append((cell, density))

    peak = 0.0
    for layer in range(int(grid["L"])):
        for i in range(int(grid["X"])):
            for j in range(int(grid["Y"])):
                q = (layer, i, j)
                value = 0.0
                for cell, density in sources:
                    distance = _distance(q, cell)
                    value += density / float((1 + distance) ** 2)
                peak = max(peak, value)
    return peak


def _normalize(value, minimum, maximum):
    if maximum <= minimum + _EPS:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def _decorate(solution, method, pool_size, record, alpha):
    output = copy.deepcopy(solution)
    output["method"] = method
    output["baseline"]["method"] = method
    output["baseline"]["proxy_not_original_tool"] = True
    output["baseline"]["complete_task_bundles_preserved"] = True
    output["baseline"]["candidate_pool_size"] = int(pool_size)
    output["baseline"]["objective"] = {
        "traffic_weighted_wirelength": record["traffic_wirelength"],
        "unweighted_net_wirelength": record["net_wirelength"],
        "thermal_coupling_surrogate": record["thermal_coupling"],
        "normalized_thermal": record.get("normalized_thermal"),
        "normalized_wirelength": record.get("normalized_wirelength"),
        "tap_alpha": alpha if method == "tap-2.5d-inspired" else None,
        "selection_cost": record["selection_cost"],
    }
    return output


def _bundle_signature(solution):
    """Return a placement-independent multiset of complete typed bundles."""
    return sorted(
        (str(site["type"]), tuple(sorted(site.get("tasks") or [])))
        for site in solution.get("sites", []))


def run(params_path, solution_path, out_dir, candidates=4096,
        seed_offset=100000, tap_alpha=0.7):
    if candidates < 2:
        raise ValueError("candidates must be at least 2")
    if not 0.0 <= tap_alpha <= 1.0:
        raise ValueError("tap_alpha must be in [0,1]")

    params = matched._read_json(params_path)
    source_solution = matched._read_json(solution_path)
    state = matched._source_state(params, source_solution)
    os.makedirs(out_dir, exist_ok=True)

    started = time.monotonic()
    pool = []
    for index in range(candidates):
        seed = int(seed_offset) + index
        candidate, _ = matched._candidate_solution(
            params, source_solution, state, seed, permute_experts=False)
        pool.append({
            "seed": seed,
            "traffic_wirelength": _traffic_wirelength(params, candidate),
            "net_wirelength": _net_wirelength(params, candidate),
            "thermal_coupling": _thermal_coupling(params, candidate),
        })

    thermal_values = [item["thermal_coupling"] for item in pool]
    wire_values = [item["net_wirelength"] for item in pool]
    thermal_min, thermal_max = min(thermal_values), max(thermal_values)
    wire_min, wire_max = min(wire_values), max(wire_values)
    for item in pool:
        item["normalized_thermal"] = _normalize(
            item["thermal_coupling"], thermal_min, thermal_max)
        item["normalized_wirelength"] = _normalize(
            item["net_wirelength"], wire_min, wire_max)

    floorplet = min(
        pool,
        key=lambda item: (
            item["traffic_wirelength"],
            item["net_wirelength"],
            item["seed"]))
    floorplet["selection_cost"] = floorplet["traffic_wirelength"]

    for item in pool:
        item["tap_cost"] = (
            tap_alpha * item["normalized_thermal"] +
            (1.0 - tap_alpha) * item["normalized_wirelength"])
    tap = min(
        pool,
        key=lambda item: (
            item["tap_cost"],
            item["thermal_coupling"],
            item["net_wirelength"],
            item["seed"]))
    tap["selection_cost"] = tap["tap_cost"]

    outputs = {}
    selections = (
        ("floorplet-inspired", floorplet),
        ("tap-2.5d-inspired", tap),
    )
    for method, record in selections:
        # Keep the candidate pool compact: at 20k candidates, retaining every
        # full solution would waste hundreds of MB.  Candidate construction is
        # deterministic, so regenerate only the two selected layouts.
        selected_solution, _ = matched._candidate_solution(
            params, source_solution, state, record["seed"],
            permute_experts=False)
        artifact = _decorate(
            selected_solution, method, candidates, record, tap_alpha)
        artifact_path = os.path.join(out_dir, method + ".json")
        matched._write_json_atomic(artifact_path, artifact)

        # Re-read and validate the serialized artifact.
        written = matched._read_json(artifact_path)
        site_types = ev.site_types_from_solution(written)
        score, reasons = matched._strict_score(
            params, written["task_site"], site_types, written["grid"],
            written.get("budgets") or {})
        if reasons:
            raise RuntimeError("%s failed strict evaluation: %s" %
                               (method, "; ".join(reasons)))
        if matched._type_counts(site_types) != state["type_counts"]:
            raise RuntimeError("%s changed the source BOM" % method)
        if _bundle_signature(written) != _bundle_signature(source_solution):
            raise RuntimeError(
                "%s changed a chiplet's complete task bundle" % method)
        outputs[method] = {
            "path": os.path.abspath(artifact_path),
            "seed": record["seed"],
            "traffic_weighted_wirelength": record["traffic_wirelength"],
            "unweighted_net_wirelength": record["net_wirelength"],
            "thermal_coupling_surrogate": record["thermal_coupling"],
            "selection_cost": record["selection_cost"],
            "predicted_makespan_cycles": score["makespan_cycles"],
            "total_area_mm2": score["total_area_mm2"],
            "peak_power_W_model": score["peak_power_W"],
            "chiplet_type_counts": matched._type_counts(site_types),
            "complete_task_bundles_preserved": True,
            "strict_feasible": True,
        }

    summary = {
        "schema_version": 1,
        "method_class": "prior-art-objective placement proxies",
        "original_tools_executed": False,
        "params": os.path.abspath(params_path),
        "source_ilp_solution": os.path.abspath(solution_path),
        "candidate_pool_size": int(candidates),
        "seed_offset": int(seed_offset),
        "tap_alpha": float(tap_alpha),
        "search_wall_s": time.monotonic() - started,
        "grid": copy.deepcopy(state["grid"]),
        "budgets": copy.deepcopy(state["budgets"]),
        "source_chiplet_type_counts": copy.deepcopy(state["type_counts"]),
        "source_total_area_mm2": state["source_score"]["total_area_mm2"],
        "source_peak_power_W_model": state["source_score"]["peak_power_W"],
        "complete_task_bundles_preserved": True,
        "pool_ranges": {
            "thermal_coupling_surrogate": [thermal_min, thermal_max],
            "unweighted_net_wirelength": [wire_min, wire_max],
        },
        "outputs": outputs,
        "claim_boundary": (
            "Prior-art-inspired proxies on the repository's discrete Mesh; "
            "not executions of the original Floorplet or TAP-2.5D tools."),
    }
    summary_path = os.path.join(out_dir, "summary.json")
    matched._write_json_atomic(summary_path, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate exact-BOM prior-art-objective proxies")
    parser.add_argument("--params", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidates", type=int, default=4096)
    parser.add_argument("--seed-offset", type=int, default=100000)
    parser.add_argument("--tap-alpha", type=float, default=0.7)
    args = parser.parse_args(argv)
    summary = run(
        args.params, args.solution, args.out_dir,
        candidates=args.candidates,
        seed_offset=args.seed_offset,
        tap_alpha=args.tap_alpha)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
