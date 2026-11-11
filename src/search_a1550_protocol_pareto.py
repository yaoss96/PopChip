#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Search the fixed-A1550 bundle-placement protocol Pareto frontier.

This experiment deliberately does not call or modify ``ilp_layout.py``.  It
takes the formal A1550 ILP-Perf JSON as a bundle template, keeps every
chiplet's type and complete task set immutable, and only changes its legal
anchor on the 6x6 mesh.  GPU-L anchors occupy their complete 2x2 footprint;
GPU-S and CPU anchors occupy one cell.

The search objective is evaluated with the exact ordered q=1 no-contention
protocol recurrence in ``evaluate_legosim_protocol_surrogate.py``.  The final
artifacts are independently re-evaluated by that public evaluator before they
are written.  Full-protocol byte-hops include init, dispatch, combine, and
summary traffic.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import time

from evaluate_legosim_protocol_surrogate import (
    DEFAULT_A1550_PROFILE,
    evaluate_protocol_surrogate,
    logical_dispatch_matrix,
)


EXPERT_RE = re.compile(r"^e\((\d+)\)$")
PHASES = ("init", "dispatch", "combine", "summary")


def read_json(path):
    with open(path, "r") as fh:
        return json.load(fh)


def write_json(path, value):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expert_group(site):
    groups = []
    for task in site.get("tasks", []):
        match = EXPERT_RE.match(task)
        if match:
            groups.append(int(match.group(1)))
    if len(groups) > 1:
        raise ValueError("bundle has more than one expert group: %s" % site)
    return groups[0] if groups else None


def extract_bundles(template, params):
    library = params["library"]
    cpu = []
    by_group = {}
    for site in template.get("sites", []):
        chiplet_type = site["type"]
        if chiplet_type not in library:
            raise ValueError("unknown chiplet type %s" % chiplet_type)
        record = {
            "id": None,
            "type": chiplet_type,
            "tasks": list(site.get("tasks", [])),
            "w": int(library[chiplet_type]["w"]),
            "h": int(library[chiplet_type]["h"]),
            "initial_anchor": (int(site["i"]), int(site["j"])),
        }
        group = expert_group(site)
        if chiplet_type == "CPU":
            if group is not None:
                raise ValueError("CPU bundle unexpectedly owns an expert")
            record["id"] = "cpu"
            cpu.append(record)
        elif group is not None:
            record["id"] = "g%d" % group
            if group in by_group:
                raise ValueError("duplicate expert group %d" % group)
            by_group[group] = record
        else:
            raise ValueError("non-CPU bundle has no logical expert: %s" % site)
    groups = int(params.get("meta", {}).get("groups", 0))
    if len(cpu) != 1 or sorted(by_group) != list(range(groups)):
        raise ValueError("expected one CPU and expert groups 0..%d" %
                         (groups - 1))
    return cpu + [by_group[group] for group in range(groups)]


def footprint(bundle, anchor):
    i, j = anchor
    return set((i + di, j + dj)
               for di in range(bundle["h"])
               for dj in range(bundle["w"]))


class PlacementSearch(object):
    """Fast objective oracle plus immutable-bundle placement operations."""

    def __init__(self, params, template, rng):
        self.params = params
        self.template = template
        self.rng = rng
        self.bundles = extract_bundles(template, params)
        grid = template["grid"]
        if int(grid["L"]) != 1:
            raise ValueError("this A1550 search requires a one-layer mesh")
        self.rows = int(grid["X"])
        self.cols = int(grid["Y"])
        if (self.rows, self.cols) != (6, 6):
            raise ValueError("this experiment requires the formal 6x6 mesh")
        self.anchors = []
        for bundle in self.bundles:
            self.anchors.append([
                (i, j)
                for i in range(self.rows - bundle["h"] + 1)
                for j in range(self.cols - bundle["w"] + 1)
            ])
        self.initial_state = tuple(bundle["initial_anchor"]
                                   for bundle in self.bundles)
        matrix, _ = logical_dispatch_matrix(params, 32)
        self.matrix = matrix
        self.compute = []
        typed = DEFAULT_A1550_PROFILE["group_type_cycles"]
        for group, bundle in enumerate(self.bundles[1:]):
            self.compute.append(int(typed[str(group)][bundle["type"]]))
        # Pre-expand the protocol loops.  delay is F-1, exactly ceil(bytes/32).
        self.dispatch_events = []
        self.bytehop_weights = []
        for source in range(len(matrix)):
            for destination in range(len(matrix)):
                assignments = int(matrix[source][destination])
                if source == destination or assignments <= 0:
                    continue
                nbytes = assignments * 128 * 2
                delay = (nbytes + 31) // 32
                self.dispatch_events.append(
                    (source, destination, delay, nbytes))
                self.bytehop_weights.append(
                    (source, destination, 2 * nbytes))
        self.evaluations = 0
        self.frontier = []

    def is_legal(self, state):
        if len(state) != len(self.bundles):
            return False
        occupied = set()
        for bundle, anchor in zip(self.bundles, state):
            i, j = anchor
            if (i < 0 or j < 0 or
                    i + bundle["h"] > self.rows or
                    j + bundle["w"] > self.cols):
                return False
            cells = footprint(bundle, anchor)
            if occupied.intersection(cells):
                return False
            occupied.update(cells)
        return True

    def metrics(self, state, track=True):
        """Exact fast recurrence; final records are checked by the public oracle."""
        self.evaluations += 1
        coords = state[1:]
        cpu = state[0]
        clocks = [0] * 9                 # groups 0..7, then CPU
        byte_hops = 0

        def dist(a, b):
            pa = cpu if a == 8 else coords[a]
            pb = cpu if b == 8 else coords[b]
            return abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])

        def send(source, destination, delay):
            write_cycle = clocks[source] + 1000
            read_cycle = clocks[destination] + 1000
            clocks[source] = write_cycle + delay
            clocks[destination] = max(
                write_cycle + delay + 104 * dist(source, destination),
                read_cycle)

        # init: (4 + 2*8)*4 = 80 bytes, F-1 = ceil(80/32) = 3.
        for group in range(8):
            hops = dist(8, group)
            byte_hops += 80 * hops
            send(8, group, 3)

        for source, destination, delay, nbytes in self.dispatch_events:
            byte_hops += nbytes * dist(source, destination)
            send(source, destination, delay)

        for group, cycles in enumerate(self.compute):
            clocks[group] += cycles

        for source, destination, delay, nbytes in self.dispatch_events:
            byte_hops += nbytes * dist(source, destination)
            send(destination, source, delay)

        # summary: 4 bytes, F-1 = ceil(4/32) = 1.
        for group in range(8):
            hops = dist(group, 8)
            byte_hops += 4 * hops
            send(group, 8, 1)

        result = (int(clocks[8]), int(byte_hops))
        if track:
            self.update_frontier(state, result)
        return result

    def update_frontier(self, state, metrics):
        cycles, byte_hops = metrics
        for old_cycles, old_hops, _ in self.frontier:
            if (old_cycles <= cycles and old_hops <= byte_hops and
                    (old_cycles < cycles or old_hops < byte_hops)):
                return
            if old_cycles == cycles and old_hops == byte_hops:
                return
        self.frontier = [entry for entry in self.frontier
                         if not (cycles <= entry[0] and
                                 byte_hops <= entry[1] and
                                 (cycles < entry[0] or
                                  byte_hops < entry[1]))]
        self.frontier.append((cycles, byte_hops, tuple(state)))

    def random_state(self):
        # Place larger footprints first; randomize ties to avoid structural bias.
        order = list(range(len(self.bundles)))
        self.rng.shuffle(order)
        order.sort(key=lambda index:
                   self.bundles[index]["w"] * self.bundles[index]["h"],
                   reverse=True)
        for _ in range(1000):
            placed = {}
            occupied = set()
            ok = True
            for index in order:
                choices = list(self.anchors[index])
                self.rng.shuffle(choices)
                legal = [anchor for anchor in choices
                         if not occupied.intersection(
                             footprint(self.bundles[index], anchor))]
                if not legal:
                    ok = False
                    break
                anchor = legal[0]
                placed[index] = anchor
                occupied.update(footprint(self.bundles[index], anchor))
            if ok:
                return tuple(placed[index]
                             for index in range(len(self.bundles)))
        raise RuntimeError("could not construct a legal random placement")

    def propose(self, state):
        # Same-footprint swaps are always geometrically legal and are important
        # for assigning the hot logical groups to the best local positions.
        if self.rng.random() < 0.32:
            first = self.rng.randrange(len(self.bundles))
            same = [index for index, bundle in enumerate(self.bundles)
                    if index != first and
                    bundle["w"] == self.bundles[first]["w"] and
                    bundle["h"] == self.bundles[first]["h"]]
            if same:
                second = self.rng.choice(same)
                proposal = list(state)
                proposal[first], proposal[second] = (
                    proposal[second], proposal[first])
                proposal = tuple(proposal)
                if self.is_legal(proposal):
                    return proposal

        index = self.rng.randrange(len(self.bundles))
        choices = self.anchors[index]
        for _ in range(24):
            anchor = self.rng.choice(choices)
            if anchor == state[index]:
                continue
            proposal = list(state)
            proposal[index] = anchor
            proposal = tuple(proposal)
            if self.is_legal(proposal):
                return proposal
        return state

    def score(self, metrics, weight, cycle_scale, hop_scale, cap):
        cycles, byte_hops = metrics
        score = (weight * cycles / float(cycle_scale) +
                 (1.0 - weight) * byte_hops / float(hop_scale))
        if cycles > cap:
            score += 1000.0 * (cycles - cap) / float(cycle_scale)
        return score

    def anneal(self, start, weight, cycle_scale, hop_scale, cap,
               iterations):
        state = tuple(start)
        metrics = self.metrics(state)
        score = self.score(metrics, weight, cycle_scale, hop_scale, cap)
        best = (score, metrics, state)
        hot, cold = 0.035, 0.000015
        for iteration in range(iterations):
            proposal = self.propose(state)
            if proposal == state:
                continue
            candidate_metrics = self.metrics(proposal)
            candidate_score = self.score(
                candidate_metrics, weight, cycle_scale, hop_scale, cap)
            fraction = iteration / float(max(1, iterations - 1))
            temperature = hot * ((cold / hot) ** fraction)
            delta = candidate_score - score
            if delta <= 0.0 or self.rng.random() < math.exp(-delta / temperature):
                state, metrics, score = (
                    proposal, candidate_metrics, candidate_score)
                if (score, metrics) < (best[0], best[1]):
                    best = (score, metrics, state)
        return best[2]

    def local_descent(self, start, weight, cycle_scale, hop_scale, cap):
        state = tuple(start)
        metrics = self.metrics(state)
        score = self.score(metrics, weight, cycle_scale, hop_scale, cap)
        improved = True
        while improved:
            improved = False
            best = (score, metrics, state)
            # Exhaust all legal one-bundle relocations.
            for index in range(len(self.bundles)):
                for anchor in self.anchors[index]:
                    if anchor == state[index]:
                        continue
                    proposal = list(state)
                    proposal[index] = anchor
                    proposal = tuple(proposal)
                    if not self.is_legal(proposal):
                        continue
                    candidate_metrics = self.metrics(proposal)
                    candidate_score = self.score(
                        candidate_metrics, weight, cycle_scale,
                        hop_scale, cap)
                    if (candidate_score, candidate_metrics) < (best[0], best[1]):
                        best = (candidate_score, candidate_metrics, proposal)
            # Exhaust logical assignments among equal-footprint bundles.
            for first in range(len(self.bundles)):
                for second in range(first + 1, len(self.bundles)):
                    if (self.bundles[first]["w"], self.bundles[first]["h"]) != \
                            (self.bundles[second]["w"],
                             self.bundles[second]["h"]):
                        continue
                    proposal = list(state)
                    proposal[first], proposal[second] = (
                        proposal[second], proposal[first])
                    proposal = tuple(proposal)
                    candidate_metrics = self.metrics(proposal)
                    candidate_score = self.score(
                        candidate_metrics, weight, cycle_scale,
                        hop_scale, cap)
                    if (candidate_score, candidate_metrics) < (best[0], best[1]):
                        best = (candidate_score, candidate_metrics, proposal)
            if best[0] + 1e-15 < score:
                score, metrics, state = best
                improved = True
        return state

    def make_solution(self, state, label, metrics, source_template):
        sites = []
        task_site = {}
        for bundle, anchor in zip(self.bundles, state):
            i, j = anchor
            site = {
                "l": 0,
                "i": int(i),
                "j": int(j),
                "type": bundle["type"],
                "tasks": list(bundle["tasks"]),
            }
            sites.append(site)
            for task in bundle["tasks"]:
                task_site[task] = [0, int(i), int(j)]
        return {
            "sites": sites,
            "task_site": task_site,
            "optimization_mode": "independent-protocol-pareto-search",
            "mode_label": label,
            "protocol_surrogate_cycles": int(metrics[0]),
            "protocol_full_byte_hops": int(metrics[1]),
            "total_area_mm2": self.template.get("total_area_mm2"),
            "peak_power_W": self.template.get("peak_power_W"),
            "grid": copy.deepcopy(self.template["grid"]),
            "budgets": copy.deepcopy(self.template.get("budgets", {})),
            "search_provenance": {
                "source_template": os.path.abspath(source_template),
                "source_template_sha256": canonical_hash(self.template),
                "immutable_bundle_ids": [bundle["id"]
                                         for bundle in self.bundles],
                "immutable_bundle_types": [bundle["type"]
                                           for bundle in self.bundles],
                "immutable_bundle_tasks": [bundle["tasks"]
                                           for bundle in self.bundles],
                "footprint_rule": "full h-by-w non-overlap",
                "protocol_traffic": list(PHASES),
            },
        }


def full_byte_hops(evaluation):
    return int(sum(evaluation["phases"][phase]["byte_hops"]
                   for phase in PHASES))


def verify_solution(params, search, solution, expected):
    evaluation = evaluate_protocol_surrogate(params, solution)
    observed = (int(evaluation["cycles"]), full_byte_hops(evaluation))
    if observed != tuple(expected):
        raise AssertionError("fast/public evaluator mismatch: %s != %s" %
                             (observed, expected))
    state = tuple((int(site["i"]), int(site["j"]))
                  for site in solution["sites"])
    if not search.is_legal(state):
        raise AssertionError("output solution is not footprint-legal")
    observed_bundles = [(site["type"], tuple(site.get("tasks", [])))
                        for site in solution["sites"]]
    expected_bundles = [(bundle["type"], tuple(bundle["tasks"]))
                        for bundle in search.bundles]
    if observed_bundles != expected_bundles:
        raise AssertionError("output changed an immutable bundle")
    return evaluation


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        default=os.path.join(
            "out", "full_constraint_smoke_a1550", "solutions", "budget",
            "area_1550", "ilp_perf.json"))
    parser.add_argument(
        "--power-seed",
        default=os.path.join(
            "out", "full_constraint_smoke_a1550", "solutions", "budget",
            "area_1550", "ilp_power.json"))
    parser.add_argument(
        "--params",
        default=os.path.join(
            "out", "a1550_workerid_diagnostic_20260801_v3", "layouts",
            "area_1550", "ilp-perf", "codegen", "params.input.json"))
    parser.add_argument(
        "--out", default=os.path.join(
            "out", "a1550_protocol_pareto_search_20260801"))
    parser.add_argument("--seed", type=int, default=15502026)
    parser.add_argument("--restarts", type=int, default=8,
                        help="annealing restarts per scalarization")
    parser.add_argument("--iterations", type=int, default=7000,
                        help="iterations per annealing restart")
    return parser.parse_args(argv)


def state_from_solution(search, solution):
    by_id = {}
    for site in solution.get("sites", []):
        group = expert_group(site)
        bundle_id = "cpu" if site.get("type") == "CPU" else "g%d" % group
        by_id[bundle_id] = (int(site["i"]), int(site["j"]))
    return tuple(by_id[bundle["id"]] for bundle in search.bundles)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    params = read_json(args.params)
    template = read_json(args.template)
    rng = random.Random(args.seed)
    search = PlacementSearch(params, template, rng)
    if not search.is_legal(search.initial_state):
        raise ValueError("formal ILP-Perf template is footprint-illegal")

    template_metrics = search.metrics(search.initial_state)
    # The first sweep uses the formal gate's Perf layout as a conservative
    # search cap.  The reported 1.2*Tperf cap is recomputed from the searched
    # minimum-cycle endpoint below.
    provisional_cap = int(math.floor(1.2 * template_metrics[0] + 1e-9))
    fixed_seeds = [search.initial_state]
    seed_rows = [{"name": "formal-ilp-perf", "metrics": template_metrics}]
    if args.power_seed and os.path.isfile(args.power_seed):
        power_state = state_from_solution(search, read_json(args.power_seed))
        if not search.is_legal(power_state):
            raise ValueError("formal ILP-Power seed is footprint-illegal")
        fixed_seeds.append(power_state)
        seed_rows.append({"name": "formal-ilp-power",
                          "metrics": search.metrics(power_state)})

    cycle_scale = float(template_metrics[0])
    hop_scale = float(template_metrics[1])
    # Dense near the two endpoints, plus balanced intermediate searches.
    weights = [0.0, 0.03, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97, 1.0]
    run_rows = []
    for weight in weights:
        for restart in range(args.restarts):
            if restart < len(fixed_seeds):
                start = fixed_seeds[restart]
            else:
                start = search.random_state()
                search.metrics(start)
            annealed = search.anneal(
                start, weight, cycle_scale, hop_scale, provisional_cap,
                args.iterations)
            refined = search.local_descent(
                annealed, weight, cycle_scale, hop_scale, provisional_cap)
            result = search.metrics(refined)
            run_rows.append({
                "weight_on_cycles": weight,
                "restart": restart,
                "cycles": result[0],
                "full_protocol_byte_hops": result[1],
            })
        print("[pareto-search] weight=%4.2f frontier=%d evaluations=%d" %
              (weight, len(search.frontier), search.evaluations))

    frontier = sorted(search.frontier, key=lambda entry: (entry[0], entry[1]))
    # Defensive final filtering because the streaming frontier intentionally
    # ignores metric-identical geometric symmetries.
    final_frontier = []
    for entry in frontier:
        if any(other[0] <= entry[0] and other[1] <= entry[1] and
               (other[0] < entry[0] or other[1] < entry[1])
               for other in frontier):
            continue
        final_frontier.append(entry)
    frontier = final_frontier
    min_cycles = min(frontier, key=lambda entry: (entry[0], entry[1]))
    performance_cap = int(math.floor(1.2 * min_cycles[0] + 1e-9))
    feasible = [entry for entry in frontier if entry[0] <= performance_cap]
    if not feasible:
        raise AssertionError("search found no layout under the 1.2T cap")
    min_hops = min(feasible, key=lambda entry: (entry[1], entry[0]))
    strict_tradeoff = (min_cycles[0] < min_hops[0] and
                       min_cycles[1] > min_hops[1])

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    artifact_rows = []
    selected = {
        "min_cycles": min_cycles,
        "min_byte_hops_under_1p2_tperf": min_hops,
    }
    for name, entry in selected.items():
        label = ("Protocol-Min-Cycles" if name == "min_cycles" else
                 "Protocol-Min-ByteHop-1.2Tperf")
        solution = search.make_solution(
            entry[2], label, entry[:2], args.template)
        evaluation = verify_solution(params, search, solution, entry[:2])
        solution_path = os.path.join(args.out, "candidate_%s.json" % name)
        eval_path = os.path.join(args.out, "evaluation_%s.json" % name)
        write_json(solution_path, solution)
        write_json(eval_path, evaluation)
        artifact_rows.append({
            "name": name,
            "solution": os.path.abspath(solution_path),
            "evaluation": os.path.abspath(eval_path),
            "solution_sha256": canonical_hash(solution),
            "cycles": entry[0],
            "full_protocol_byte_hops": entry[1],
            "phase_byte_hops": {
                phase: evaluation["phases"][phase]["byte_hops"]
                for phase in PHASES
            },
            "anchors_by_bundle": {
                bundle["id"]: [0, entry[2][index][0], entry[2][index][1]]
                for index, bundle in enumerate(search.bundles)
            },
        })

    frontier_rows = []
    for index, entry in enumerate(frontier):
        frontier_rows.append({
            "index": index,
            "cycles": entry[0],
            "full_protocol_byte_hops": entry[1],
            "under_1p2_tperf": entry[0] <= performance_cap,
            "anchors_by_bundle": {
                bundle["id"]: [0, entry[2][bundle_index][0],
                               entry[2][bundle_index][1]]
                for bundle_index, bundle in enumerate(search.bundles)
            },
        })

    elapsed = time.time() - started
    summary = {
        "schema_version": 1,
        "scope": (
            "independent coordinate search over nine immutable A1550 bundles; "
            "not a MILP solve and not measured LEGOSim"),
        "objective_oracle": "legosim-ordered-protocol-max-plus-v1",
        "source_template": os.path.abspath(args.template),
        "params": os.path.abspath(args.params),
        "seed": args.seed,
        "search": {
            "scalarization_weights": weights,
            "restarts_per_weight": args.restarts,
            "annealing_iterations_per_restart": args.iterations,
            "provisional_search_cap_from_formal_template": provisional_cap,
            "objective_evaluations": search.evaluations,
            "elapsed_s": elapsed,
            "full_gpu_l_footprints_enforced": True,
            "immutable_type_and_task_bundles": True,
        },
        "reference": {
            "formal_ilp_perf_cycles": template_metrics[0],
            "formal_ilp_perf_full_protocol_byte_hops": template_metrics[1],
            "searched_tperf_cycles": min_cycles[0],
            "cycle_cap_definition": "floor(1.2 * searched_tperf_cycles)",
            "cycle_cap": performance_cap,
            "seed_metrics": [
                {"name": row["name"], "cycles": row["metrics"][0],
                 "full_protocol_byte_hops": row["metrics"][1]}
                for row in seed_rows
            ],
        },
        "selected": artifact_rows,
        "strict_cycles_vs_byte_hops_tradeoff_found": strict_tradeoff,
        "pareto_point_count": len(frontier_rows),
        "pareto_frontier": frontier_rows,
        "all_final_artifacts_verified_with_public_evaluator": True,
        "runs": run_rows,
    }
    write_json(os.path.join(args.out, "summary.json"), summary)

    csv_path = os.path.join(args.out, "pareto_frontier.csv")
    with open(csv_path, "w") as fh:
        fh.write("index,cycles,full_protocol_byte_hops,under_1p2_tperf\n")
        for row in frontier_rows:
            fh.write("%d,%d,%d,%s\n" %
                     (row["index"], row["cycles"],
                      row["full_protocol_byte_hops"],
                      str(row["under_1p2_tperf"]).lower()))

    print("[pareto-search] template cycles=%d byte_hops=%d cap=%d" %
          (template_metrics[0], template_metrics[1], performance_cap))
    print("[pareto-search] min cycles=%d byte_hops=%d" % min_cycles[:2])
    print("[pareto-search] min byte_hops<=cap: cycles=%d byte_hops=%d" %
          min_hops[:2])
    print("[pareto-search] strict tradeoff=%s frontier=%d elapsed=%.2fs" %
          (strict_tradeoff, len(frontier), elapsed))
    print("[pareto-search] wrote %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
