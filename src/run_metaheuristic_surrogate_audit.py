#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wall-clock-matched SA/GA baselines for the sole joint min-T model.

Each heuristic receives a fresh canonical ILP-Perf BOM, but independently
searches task mapping and physical placement, including the CPU coordinate.
Every candidate is a legal C1--C8 integer design.  Its score is the current
public T: the replay-aligned fixed-order deterministic-dispatch completion,
epigraphed with directed-Mesh link capacity.  A deterministic non-preemptive
single-lane list schedule remains a hard C6b/C11--C13 feasibility witness, but
its legacy whole-task makespan is deliberately not scored.

The schedule and routing policies are restricted subsets of the choices open
to the MILP.  This is therefore an honest same-objective, fixed-BOM baseline,
not the same search space.  Results are reported relative to the saved ILP
incumbent and, separately, to any available solver lower bound; no value is
called a gap to optimum.  SA and GA/NSGA-style search each receive one 45-s
wall-clock budget per method/case, split across fixed seeds 11/29/47; every
seed's initialization is charged to its 15-s share.
"""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import heapq
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = Path("out/metaheuristic_mint_c6b_flow_20260802")
EPS = 1.0e-8
FIXED_SEEDS = (11, 29, 47)
METHOD_CASE_WALL_BUDGET_S = 45.0


def fixed_seed_wall_budgets():
    """Split the 45-s method/case budget across the fixed seed set."""
    if not FIXED_SEEDS:
        raise AssertionError("the fixed metaheuristic seed set is empty")
    share = METHOD_CASE_WALL_BUDGET_S / float(len(FIXED_SEEDS))
    allocations = []
    allocated = 0.0
    for offset, seed in enumerate(FIXED_SEEDS):
        budget = (METHOD_CASE_WALL_BUDGET_S - allocated
                  if offset == len(FIXED_SEEDS) - 1 else share)
        allocations.append({"seed": int(seed), "wall_budget_s": budget})
        allocated += budget
    if allocated > METHOD_CASE_WALL_BUDGET_S + EPS:
        raise AssertionError("per-seed budgets exceed the method/case budget")
    return allocations


def wall_budget_contract():
    """Return the machine-readable cross-method wall-clock fairness contract."""
    allocations = fixed_seed_wall_budgets()
    return {
        "basis": "45 s total per method and case",
        "wall_budget_s_per_method_case": METHOD_CASE_WALL_BUDGET_S,
        "fixed_seeds": list(FIXED_SEEDS),
        "fixed_seed_allocations": allocations,
        "configured_seed_budget_sum_s": sum(
            allocation["wall_budget_s"] for allocation in allocations),
        "initialization_charged_to_seed_wall_time": True,
        "comparison_methods": [
            "ILP-Perf", "Floorplet", "TAP-2.5D", "SA", "GA"],
        "metaheuristic_seed_aggregation": "three fixed 15-s runs",
    }


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


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def footprint(kind, anchor, library):
    spec = library[kind]
    layer, row, col = anchor
    return {(layer, row + di, col + dj)
            for di in range(int(spec["w"]))
            for dj in range(int(spec["h"]))}


def legal_anchors(kind, grid, library):
    spec = library[kind]
    return [(layer, row, col)
            for layer in range(int(grid["L"]))
            for row in range(int(grid["X"]) - int(spec["w"]) + 1)
            for col in range(int(grid["Y"]) - int(spec["h"]) + 1)]


def is_cpu(kind, library):
    return ("CPU" in str(kind).upper() or
            "sniper" in str(library[kind].get("simlet", "")).lower())


def state_copy(state):
    return {
        "types": list(state["types"]),
        "anchors": list(state["anchors"]),
        "assign": dict(state["assign"]),
    }


def largest_remainder(weights, total):
    """Codegen-equivalent deterministic integer apportionment."""
    if total < 0:
        raise ValueError("assignment total must be nonnegative")
    if not weights:
        return []
    norm = float(sum(weights))
    if norm <= 0.0:
        weights = [1.0] * len(weights)
        norm = float(len(weights))
    exact = [total * float(weight) / norm for weight in weights]
    base = [int(math.floor(value)) for value in exact]
    order = sorted(range(len(weights)),
                   key=lambda index: (-(exact[index] - base[index]), index))
    for index in order[:total - sum(base)]:
        base[index] += 1
    return base


def balanced_transport(column_totals, row_count, row_total):
    """Codegen-equivalent cyclic transport with exact row/column margins."""
    totals = [int(value) for value in column_totals]
    if row_count <= 0 or row_total < 0:
        raise ValueError("invalid balanced-transport dimensions")
    if sum(totals) != row_count * row_total:
        raise ValueError("balanced-transport margins disagree")
    matrix = [[0] * len(totals) for _ in range(row_count)]
    cursor = 0
    for column, total in enumerate(totals):
        quotient, remainder = divmod(total, row_count)
        for source in range(row_count):
            matrix[source][column] = quotient
        for _ in range(remainder):
            matrix[cursor][column] += 1
            cursor = (cursor + 1) % row_count
    if ([sum(row) for row in matrix] != [row_total] * row_count or
            [sum(matrix[row][column] for row in range(row_count))
             for column in range(len(totals))] != totals):
        raise AssertionError("deterministic transport failed")
    return matrix


def deterministic_dispatch(params, tokens_per_source):
    meta = params.get("meta", {})
    groups = int(meta.get("groups", len(params.get("expert_tasks", []))))
    topk = int(meta.get("topk", 1))
    tokens = int(tokens_per_source)
    if groups <= 0 or topk <= 0 or tokens <= 0:
        raise ValueError("dispatch groups/topk/tokens must be positive")
    weights = list(meta.get("group_pi", []))
    if len(weights) != groups:
        weights = [1.0] * groups
    row_total = tokens * topk
    totals = largest_remainder(weights, groups * row_total)
    return balanced_transport(totals, groups, row_total)


def dimension_order_route(source, target):
    """Yield a legal directed shortest path in layer/row/column order."""
    current = [int(value) for value in source]
    target = tuple(int(value) for value in target)
    for axis in (0, 1, 2):
        while current[axis] != target[axis]:
            before = tuple(current)
            current[axis] += 1 if current[axis] < target[axis] else -1
            yield before, tuple(current)


def relative_pct(value, reference):
    if reference is None or float(reference) <= 0.0:
        return None
    return 100.0 * (float(value) / float(reference) - 1.0)


class CaseModel:
    def __init__(self, case_id, params, solution, budgets, grid):
        self.case_id = case_id
        self.params = params
        self.library = params["library"]
        self.tasks = params["tasks"]
        self.edges = params["edges"]
        self.ic = params["interconnect"]
        self.grid = {name: int(grid[name]) for name in ("L", "X", "Y")}
        contract = solution.get("objective_contract") or {}
        if (solution.get("optimization_mode") != "ilp-perf" or
                contract.get("active_objective_count") != 1 or
                contract.get("active_objective") != "min T" or
                contract.get("protocol_completion_in_T") is not True):
            raise ValueError("%s: source is not the sole protocol min-T solve" %
                             case_id)
        if (contract.get("energy_or_byte_hop_tie_break") is not False or
                solution.get("performance_cap_enforced") is not False):
            raise ValueError(
                "%s: power/capped/tie-break modes are forbidden" % case_id)
        if "p_col" in budgets:
            raise ValueError("%s: retired C9/p_col budget is forbidden" %
                             case_id)
        self.budgets = copy.deepcopy(budgets)
        self.experts = set(params.get("expert_tasks") or [])
        self.expert_slots = {}
        for kind, spec in self.library.items():
            raw_slots = spec.get("expert_slots", 1)
            slots = int(raw_slots)
            if slots < 0 or float(raw_slots) != float(slots):
                raise ValueError(
                    "%s: library.%s.expert_slots is invalid" %
                    (case_id, kind))
            self.expert_slots[kind] = slots

        source_types = [str(site["type"]) for site in solution["sites"]]
        cpu_types = [kind for kind in source_types
                     if is_cpu(kind, self.library)]
        if len(cpu_types) != 1:
            raise ValueError("%s: source BOM must contain exactly one CPU" %
                             case_id)
        # Keep a stable anonymous-instance order, but never fix any coordinate.
        cpu_kind = cpu_types[0]
        remaining = list(source_types)
        remaining.remove(cpu_kind)
        remaining.sort(key=lambda kind: (
            -int(self.library[kind]["w"]) * int(self.library[kind]["h"]),
            kind))
        self.types = [cpu_kind] + remaining
        self.anchors_by_type = {
            kind: legal_anchors(kind, self.grid, self.library)
            for kind in self.library
        }
        self.topological = self._topological_order()
        self.predecessors = {task: [] for task in self.tasks}
        for edge in self.edges:
            self.predecessors[edge["dst"]].append(edge)
        self.source_solution = solution
        proxy = solution.get("noc_power_proxy") or {}
        self.noc_traffic_kind = proxy.get("traffic_kind")
        if self.noc_traffic_kind != "deterministic-dispatch":
            raise ValueError(
                "%s: metaheuristic audit requires deterministic-dispatch "
                "because the public T is the replay-aligned protocol" %
                case_id)
        self.dispatch_tokens_per_source = int(
            proxy.get("dispatch_tokens_per_source") or 32)
        self.dispatch_token_bytes_per_phase = proxy.get(
            "dispatch_token_bytes_per_phase")
        if self.dispatch_token_bytes_per_phase is None:
            raise ValueError(
                "%s: canonical solution lacks dispatch bytes per phase" %
                case_id)
        self.dispatch_token_bytes_per_phase = float(
            self.dispatch_token_bytes_per_phase)
        if (self.dispatch_tokens_per_source <= 0 or
                self.dispatch_token_bytes_per_phase <= 0.0):
            raise ValueError("%s: invalid deterministic-dispatch parameters" %
                             case_id)
        self.dispatch_matrix = deterministic_dispatch(
            params, self.dispatch_tokens_per_source)
        groups = len(self.dispatch_matrix)
        expert_by_group = {}
        for task in self.tasks:
            match = re.match(r"^e\((\d+)\)$", task)
            if match:
                expert_by_group[int(match.group(1))] = task
        if sorted(expert_by_group) != list(range(groups)):
            raise ValueError(
                "%s: deterministic dispatch needs e(0)..e(%d)" %
                (case_id, groups - 1))
        self.expert_by_group = [expert_by_group[index]
                                for index in range(groups)]

        protocol = solution.get("protocol_model") or {}
        if protocol.get("kind") != "fixed-order-max-plus-event-DAG":
            raise ValueError(
                "%s: canonical solution lacks the fixed-order protocol model" %
                case_id)
        if protocol.get("included_in_T") is not True:
            raise ValueError("%s: protocol completion is not included in T" %
                             case_id)
        self.protocol_flit_bytes = float(protocol.get("flit_bytes"))
        self.protocol_hop_cycles = float(protocol.get("hop_cycles"))
        self.protocol_sync_cycles = float(
            protocol.get("sync_overhead_cycles"))
        if (self.protocol_flit_bytes <= 0.0 or
                self.protocol_hop_cycles < 0.0 or
                self.protocol_sync_cycles < 0.0):
            raise ValueError("%s: invalid fixed-order protocol timing" %
                             case_id)
        raw_compute = protocol.get("compute_cycles_by_expert_type") or {}
        self.protocol_compute_cycles = {}
        available_types = set(self.types)
        for expert in self.expert_by_group:
            by_type = raw_compute.get(expert) or {}
            normalized = {}
            for kind in self.tasks[expert]["allowed"]:
                if kind not in available_types:
                    continue
                try:
                    cycles = float(by_type[kind])
                except (KeyError, TypeError, ValueError):
                    raise ValueError(
                        "%s: protocol compute profile lacks %s on %s" %
                        (case_id, expert, kind))
                if not math.isfinite(cycles) or cycles < 0.0:
                    raise ValueError(
                        "%s: invalid protocol compute cycles for %s on %s" %
                        (case_id, expert, kind))
                normalized[kind] = cycles
            if not normalized:
                raise ValueError(
                    "%s: no usable protocol compute profile for %s" %
                    (case_id, expert))
            self.protocol_compute_cycles[expert] = normalized
        self.protocol_init_bytes = 4.0 * (4 + 2 * groups)
        self.protocol_done_bytes = 4.0

    def _topological_order(self):
        successors = {task: [] for task in self.tasks}
        indegree = {task: 0 for task in self.tasks}
        for edge in self.edges:
            successors[edge["src"]].append(edge["dst"])
            indegree[edge["dst"]] += 1
        queue = [task for task, degree in indegree.items() if degree == 0]
        heapq.heapify(queue)
        order = []
        while queue:
            task = heapq.heappop(queue)
            order.append(task)
            for target in successors[task]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    heapq.heappush(queue, target)
        if len(order) != len(self.tasks):
            raise ValueError("%s: task graph is cyclic" % self.case_id)
        return order

    def random_packing(self, rng):
        anchors = [None] * len(self.types)
        occupied = set()
        order = list(range(len(self.types)))
        order.sort(key=lambda index: (
            -int(self.library[self.types[index]]["w"]) *
            int(self.library[self.types[index]]["h"]), rng.random()))

        def visit(position):
            if position == len(order):
                return True
            index = order[position]
            kind = self.types[index]
            choices = list(self.anchors_by_type[kind])
            rng.shuffle(choices)
            for anchor in choices:
                cells = footprint(kind, anchor, self.library)
                if cells & occupied:
                    continue
                anchors[index] = anchor
                occupied.update(cells)
                if visit(position + 1):
                    return True
                occupied.difference_update(cells)
                anchors[index] = None
            return False

        if not visit(0):
            raise RuntimeError("%s: could not repack exact BOM" % self.case_id)
        return anchors

    def random_assignment(self, rng):
        assignment = {}
        memory = collections.defaultdict(float)
        expert_count = collections.Counter()
        task_count = collections.Counter()
        order = list(self.tasks)
        rng.shuffle(order)
        order.sort(key=lambda task: (
            len(self.tasks[task]["allowed"]),
            0 if task in self.experts else 1))
        for task in order:
            choices = []
            for index, kind in enumerate(self.types):
                if kind not in self.tasks[task]["allowed"]:
                    continue
                if (memory[index] +
                        float(self.tasks[task].get("mem_mb", 0.0)) >
                        float(self.library[kind]["mem"]) + 1.0e-6):
                    continue
                if (task in self.experts and
                        expert_count[index] >= self.expert_slots[kind]):
                    continue
                choices.append(index)
            if not choices:
                raise RuntimeError("%s: no capacity-compatible chiplet for %s" %
                                   (self.case_id, task))
            rng.shuffle(choices)
            choices.sort(key=lambda index: (task_count[index],
                                             expert_count[index]))
            index = choices[0]
            assignment[task] = index
            memory[index] += float(self.tasks[task].get("mem_mb", 0.0))
            task_count[index] += 1
            if task in self.experts:
                expert_count[index] += 1
        return assignment

    def random_state(self, rng):
        for _ in range(64):
            try:
                assignment = self.random_assignment(rng)
                anchors = self.random_packing(rng)
            except RuntimeError:
                continue
            state = {"types": list(self.types), "anchors": anchors,
                     "assign": assignment}
            if self.feasible(state)[0]:
                return state
        raise RuntimeError("%s: failed to initialize a legal state" % self.case_id)

    def feasible(self, state):
        reasons = []
        types = state["types"]
        anchors = state["anchors"]
        assignment = state["assign"]
        if types != self.types:
            reasons.append("exact BOM/type order changed")
        if len(anchors) != len(types) or any(anchor is None for anchor in anchors):
            reasons.append("anchor vector malformed")
            return False, reasons
        occupied = {}
        layer_area = {}
        for index, (kind, anchor) in enumerate(zip(types, anchors)):
            if anchor not in self.anchors_by_type[kind]:
                reasons.append("illegal anchor %s for %s" % (anchor, kind))
                continue
            cells = footprint(kind, anchor, self.library)
            for cell in cells:
                if cell in occupied:
                    reasons.append("overlap %s" % (cell,))
                occupied[cell] = index
            layer_area[anchor[0]] = (layer_area.get(anchor[0], 0.0) +
                                     float(self.library[kind]["area"]))

        area = sum(float(self.library[kind]["area"]) for kind in types)
        power = sum(float(self.library[kind]["p_peak"]) for kind in types)
        if self.budgets.get("a_max") is not None and area > float(
                self.budgets["a_max"]) + 1.0e-6:
            reasons.append("area budget")
        if self.budgets.get("p_max") is not None and power > float(
                self.budgets["p_max"]) + 1.0e-6:
            reasons.append("power budget")
        if self.budgets.get("a_max_layer") is not None:
            cap = float(self.budgets["a_max_layer"])
            if any(value > cap + 1.0e-6 for value in layer_area.values()):
                reasons.append("layer area budget")
        if set(assignment) != set(self.tasks):
            reasons.append("task set mismatch")
            return False, reasons
        memory = collections.defaultdict(float)
        expert_count = collections.Counter()
        task_count = collections.Counter()
        for task, index in assignment.items():
            if not isinstance(index, int) or not 0 <= index < len(types):
                reasons.append("bad task index")
                continue
            kind = types[index]
            if kind not in self.tasks[task]["allowed"]:
                reasons.append("type incompatibility")
            memory[index] += float(self.tasks[task].get("mem_mb", 0.0))
            task_count[index] += 1
            if task in self.experts:
                expert_count[index] += 1
        for index, kind in enumerate(types):
            if memory[index] > float(self.library[kind]["mem"]) + 1.0e-6:
                reasons.append("memory capacity")
            if expert_count[index] > self.expert_slots[kind]:
                reasons.append("resident-expert capacity")
            if task_count[index] < 1:
                reasons.append("selected chiplet has no assigned task")
        return not reasons, reasons

    def _edge_latency(self, edge, state):
        source = state["anchors"][state["assign"][edge["src"]]]
        target = state["anchors"][state["assign"][edge["dst"]]]
        if source == target:
            return 0.0
        horizontal = (abs(source[1] - target[1]) +
                      abs(source[2] - target[2]))
        vertical = abs(source[0] - target[0])
        return (float(self.ic["t0"]) +
                float(self.ic["tau_h"]) * horizontal +
                float(self.ic["tau_v"]) * vertical +
                float(edge["vol"]) / float(self.ic["beta"]))

    def _list_schedule(self, state):
        """Construct a deterministic feasible C6b/C11--C13 schedule."""
        assignment = state["assign"]
        types = state["types"]
        rank = {task: index for index, task in enumerate(self.topological)}
        remaining = set(self.tasks)
        chiplet_free = collections.defaultdict(float)
        start = {}
        finish = {}
        while remaining:
            ready = []
            for task in remaining:
                if any(edge["src"] not in finish
                       for edge in self.predecessors[task]):
                    continue
                release = 0.0
                for edge in self.predecessors[task]:
                    release = max(
                        release,
                        finish[edge["src"]] + self._edge_latency(edge, state))
                index = assignment[task]
                earliest = max(release, chiplet_free[index])
                ready.append((earliest, rank[task], str(task), task, release))
            if not ready:
                raise RuntimeError("%s: list scheduling deadlock" % self.case_id)
            earliest, _, _, task, release = min(ready)
            index = assignment[task]
            duration = float(self.tasks[task]["cyc"][types[index]])
            start[task] = earliest
            finish[task] = earliest + duration
            chiplet_free[index] = finish[task]
            remaining.remove(task)
        return {
            "start": start,
            "finish": finish,
            "makespan_cycles": max(finish.values()) if finish else 0.0,
            "policy": "global-earliest-start deterministic list schedule",
            "single_execution_lane_per_chiplet": True,
            "non_preemptive": True,
        }

    def _traffic_demands(self):
        demands = []
        for source in range(len(self.dispatch_matrix)):
            for target in range(len(self.dispatch_matrix)):
                assignments = int(self.dispatch_matrix[source][target])
                if source == target or assignments <= 0:
                    continue
                volume = assignments * self.dispatch_token_bytes_per_phase
                source_task = self.expert_by_group[source]
                target_task = self.expert_by_group[target]
                demands.append(("dispatch", source_task, target_task, volume))
                demands.append(("combine", target_task, source_task, volume))
        return demands

    def _protocol_completion(self, state, schedule):
        """Evaluate the exact fixed-order max-plus recurrence used by min T.

        Message order and endpoint-clock semantics match ``ilp_layout.py``.
        If several experts share one chiplet, their protocol-scale kernels use
        the same deterministic relative order as the hard task-schedule witness.
        """
        assignment = state["assign"]
        types = state["types"]
        anchors = state["anchors"]
        groups = len(self.expert_by_group)
        cpu_site = anchors[0]
        expert_sites = [anchors[assignment[expert]]
                        for expert in self.expert_by_group]
        resource_ids = [assignment[expert]
                        for expert in self.expert_by_group]
        compute_cycles = [
            self.protocol_compute_cycles[expert][types[assignment[expert]]]
            for expert in self.expert_by_group]
        compute_order = sorted(
            range(groups),
            key=lambda group: (
                schedule["start"][self.expert_by_group[group]], group))

        clocks = {"cpu": 0.0}
        clocks.update({group: 0.0 for group in range(groups)})
        phase_events = collections.Counter()
        event_count = 0

        def distance(first, second):
            return sum(abs(float(first[axis]) - float(second[axis]))
                       for axis in range(3))

        def message(phase, source, target, volume, route_distance):
            nonlocal event_count
            serialization = float(int(math.ceil(
                float(volume) / self.protocol_flit_bytes)))
            write_time = clocks[source] + self.protocol_sync_cycles
            sender_finish = write_time + serialization
            receiver_finish = max(
                write_time + serialization +
                self.protocol_hop_cycles * float(route_distance),
                clocks[target] + self.protocol_sync_cycles)
            clocks[source] = sender_finish
            clocks[target] = receiver_finish
            phase_events[phase] += 1
            event_count += 1

        for group in range(groups):
            message("init", "cpu", group, self.protocol_init_bytes,
                    distance(cpu_site, expert_sites[group]))

        for source in range(groups):
            for target in range(groups):
                assignments = int(self.dispatch_matrix[source][target])
                if source == target or assignments <= 0:
                    continue
                message(
                    "dispatch", source, target,
                    assignments * self.dispatch_token_bytes_per_phase,
                    distance(expert_sites[source], expert_sites[target]))

        resource_ready = {}
        compute_start = {}
        compute_finish = {}
        for group in compute_order:
            resource = resource_ids[group]
            start = max(clocks[group], resource_ready.get(resource, 0.0))
            finish = start + compute_cycles[group]
            compute_start[group] = start
            compute_finish[group] = finish
            clocks[group] = finish
            resource_ready[resource] = finish

        for source in range(groups):
            for target in range(groups):
                assignments = int(self.dispatch_matrix[source][target])
                if source == target or assignments <= 0:
                    continue
                message(
                    "combine", target, source,
                    assignments * self.dispatch_token_bytes_per_phase,
                    distance(expert_sites[source], expert_sites[target]))

        for group in range(groups):
            message("summary", group, "cpu", self.protocol_done_bytes,
                    distance(expert_sites[group], cpu_site))

        return {
            "completion_cycles": float(clocks["cpu"]),
            "worker_finish_cycles": [float(clocks[group])
                                     for group in range(groups)],
            "compute_start_cycles": {
                self.expert_by_group[group]: float(compute_start[group])
                for group in range(groups)},
            "compute_finish_cycles": {
                self.expert_by_group[group]: float(compute_finish[group])
                for group in range(groups)},
            "compute_cycles": {
                self.expert_by_group[group]: float(compute_cycles[group])
                for group in range(groups)},
            "compute_order": [self.expert_by_group[group]
                              for group in compute_order],
            "resource_ids": {
                self.expert_by_group[group]: int(resource_ids[group])
                for group in range(groups)},
            "event_count": event_count,
            "phase_event_count": dict(phase_events),
            "policy": "replay-aligned fixed-order max-plus event DAG",
            "message_order": (
                "worker-ID init; lexicographic dispatch; reverse-direction "
                "combine in the same pair order; worker-ID summaries"),
            "co_resident_compute_order": (
                "hard-schedule relative order on each chiplet"),
        }

    def _route_capacity(self, state):
        phase_load = collections.defaultdict(float)
        aggregate_load = collections.defaultdict(float)
        routed_messages = 0
        routed_byte_hop = 0.0
        for phase, source_task, target_task, volume in self._traffic_demands():
            source = state["anchors"][state["assign"][source_task]]
            target = state["anchors"][state["assign"][target_task]]
            hops = 0
            for link in dimension_order_route(source, target):
                phase_load[(phase, link)] += float(volume)
                aggregate_load[link] += float(volume)
                hops += 1
            routed_byte_hop += float(volume) * hops
            routed_messages += 1
        beta = float(self.ic["beta"])
        if beta <= 0.0:
            raise ValueError("%s: interconnect beta must be positive" %
                             self.case_id)
        maximum = max(aggregate_load.values()) if aggregate_load else 0.0
        return {
            "capacity_bound_cycles": maximum / beta,
            "maximum_directed_link_load_bytes": maximum,
            "routed_message_count": routed_messages,
            "routed_byte_hop": routed_byte_hop,
            "nonzero_directed_link_count": len(aggregate_load),
            "phase_link_load_bytes": phase_load,
            "aggregate_link_load_bytes": aggregate_load,
            "policy": "unsplittable dimension-order shortest paths L-X-Y",
            "milp_relation": (
                "restricted feasible subset of C10d/e source-aggregated "
                "splittable shortest-path flows"),
        }

    def evaluate_candidate(self, state):
        ok, reasons = self.feasible(state)
        if not ok:
            raise ValueError("%s: infeasible candidate: %s" %
                             (self.case_id, "; ".join(reasons)))
        schedule = self._list_schedule(state)
        protocol = self._protocol_completion(state, schedule)
        routing = self._route_capacity(state)
        value = max(protocol["completion_cycles"],
                    routing["capacity_bound_cycles"])
        return {"T_cycles": value, "protocol": protocol,
                "schedule": schedule, "routing": routing}

    def objective_T(self, state):
        return self.evaluate_candidate(state)["T_cycles"]

    def source_state(self):
        # Map source anchors to our same-type anonymous instance indices.
        source_by_type = collections.defaultdict(list)
        for site in self.source_solution["sites"]:
            source_by_type[str(site["type"])].append(
                (int(site["l"]), int(site["i"]), int(site["j"])))
        for values in source_by_type.values():
            values.sort()
        anchors = [None] * len(self.types)
        for index in range(len(self.types)):
            kind = self.types[index]
            if not source_by_type[kind]:
                raise RuntimeError("%s: source BOM/type count mismatch" %
                                   self.case_id)
            anchors[index] = source_by_type[kind].pop(0)
        index_at = {anchor: index for index, anchor in enumerate(anchors)}
        assignment = {
            task: index_at[tuple(anchor)]
            for task, anchor in self.source_solution["task_site"].items()
        }
        state = {"types": list(self.types), "anchors": anchors,
                 "assign": assignment}
        ok, reasons = self.feasible(state)
        if not ok:
            raise RuntimeError("%s: serialized ILP layout fails audit: %s" %
                               (self.case_id, "; ".join(reasons)))
        return state

    def solution(self, state, method, seed, score):
        evaluated = self.evaluate_candidate(state)
        if not math.isclose(float(score), evaluated["T_cycles"],
                            rel_tol=1.0e-10, abs_tol=1.0e-6):
            raise ValueError("serialized score disagrees with candidate T")
        tasks_at = collections.defaultdict(list)
        for task, index in state["assign"].items():
            tasks_at[index].append(task)
        sites = []
        for index, (kind, anchor) in enumerate(zip(state["types"],
                                                   state["anchors"])):
            sites.append({
                "l": anchor[0], "i": anchor[1], "j": anchor[2],
                "type": kind, "tasks": sorted(tasks_at[index]),
            })
        task_site = {
            task: list(state["anchors"][index])
            for task, index in sorted(state["assign"].items())
        }
        schedule = {
            task: {
                "start": evaluated["schedule"]["start"][task],
                "finish": evaluated["schedule"]["finish"][task],
                "chiplet_index": state["assign"][task],
            }
            for task in sorted(self.tasks)}
        link_loads = []
        for link, load in sorted(
                evaluated["routing"]["aggregate_link_load_bytes"].items()):
            link_loads.append({"src": list(link[0]), "dst": list(link[1]),
                               "load_bytes": load})
        return {
            "method": method,
            "seed": seed,
            "search_space": (
                "fixed fresh ILP-Perf BOM; independent task mapping and "
                "placement including movable CPU"),
            "objective": (
                "minimize T = max(fixed-order protocol completion, "
                "directed-Mesh capacity bound)"),
            "objective_scope": {
                "same_scalar_objective_as_ilp_perf": True,
                "protocol_policy": evaluated["protocol"]["policy"],
                "legacy_task_schedule_policy": evaluated["schedule"][
                    "policy"],
                "legacy_task_makespan_scored": False,
                "routing_policy": evaluated["routing"]["policy"],
                "routing_relation_to_milp": evaluated["routing"][
                    "milp_relation"],
                "same_search_space_as_milp": False,
                "fixed_bom": True,
                "energy_or_byte_hop_tie_break": False,
                "power_mode": False,
            },
            "T_cycles": score,
            "protocol_completion_cycles": evaluated["protocol"][
                "completion_cycles"],
            "protocol_model": copy.deepcopy(evaluated["protocol"]),
            "hard_feasibility_task_schedule": schedule,
            "hard_feasibility_task_makespan_cycles": evaluated["schedule"][
                "makespan_cycles"],
            "expert_slots": copy.deepcopy(self.expert_slots),
            "routing": {
                "traffic_kind": self.noc_traffic_kind,
                "capacity_bound_cycles": evaluated["routing"][
                    "capacity_bound_cycles"],
                "maximum_directed_link_load_bytes": evaluated["routing"][
                    "maximum_directed_link_load_bytes"],
                "routed_message_count": evaluated["routing"][
                    "routed_message_count"],
                "routed_byte_hop_diagnostic_only": evaluated["routing"][
                    "routed_byte_hop"],
                "policy": evaluated["routing"]["policy"],
                "milp_relation": evaluated["routing"]["milp_relation"],
                "link_loads": link_loads,
            },
            "sites": sites,
            "task_site": task_site,
            "grid": copy.deepcopy(self.grid),
            "budgets": copy.deepcopy(self.budgets),
        }

    def strict_evaluate(self, state):
        ok, reasons = self.feasible(state)
        if not ok:
            return {"feasible": False, "reasons": reasons}
        evaluated = self.evaluate_candidate(state)
        schedule = evaluated["schedule"]
        tolerance = 1.0e-6
        for edge in self.edges:
            required = (schedule["finish"][edge["src"]] +
                        self._edge_latency(edge, state))
            if schedule["start"][edge["dst"]] + tolerance < required:
                reasons.append("DAG/communication precedence")
        by_chiplet = collections.defaultdict(list)
        for task, index in state["assign"].items():
            by_chiplet[index].append((schedule["start"][task],
                                      schedule["finish"][task], task))
        for intervals in by_chiplet.values():
            intervals.sort()
            for previous, following in zip(intervals, intervals[1:]):
                if following[0] + tolerance < previous[1]:
                    reasons.append("co-resident execution overlap")
        protocol = evaluated["protocol"]
        protocol_by_chiplet = collections.defaultdict(list)
        for expert, index in protocol["resource_ids"].items():
            protocol_by_chiplet[index].append((
                protocol["compute_start_cycles"][expert],
                protocol["compute_finish_cycles"][expert], expert))
        for intervals in protocol_by_chiplet.values():
            intervals.sort()
            for previous, following in zip(intervals, intervals[1:]):
                if following[0] + tolerance < previous[1]:
                    reasons.append("co-resident protocol compute overlap")
        if (evaluated["T_cycles"] + tolerance <
                protocol["completion_cycles"]):
            reasons.append("fixed-order protocol completion")
        if (evaluated["T_cycles"] + tolerance <
                evaluated["routing"]["capacity_bound_cycles"]):
            reasons.append("directed-link capacity")
        return {
            "feasible": not reasons,
            "reasons": reasons,
            "T_cycles": evaluated["T_cycles"],
            "protocol_completion_cycles": protocol["completion_cycles"],
            "hard_feasibility_task_makespan_cycles": schedule[
                "makespan_cycles"],
            "hard_feasibility_task_makespan_scored": False,
            "route_capacity_bound_cycles": evaluated["routing"][
                "capacity_bound_cycles"],
        }

    def mutate(self, state, rng):
        for _ in range(32):
            candidate = state_copy(state)
            move = rng.choices(
                ["expert_swap", "task_move", "anchor_move", "bundle_swap"],
                weights=[0.30, 0.25, 0.30, 0.15], k=1)[0]
            if move == "expert_swap" and len(self.experts) >= 2:
                left, right = rng.sample(sorted(self.experts), 2)
                candidate["assign"][left], candidate["assign"][right] = (
                    candidate["assign"][right], candidate["assign"][left])
            elif move == "task_move":
                movable = sorted(self.tasks)
                if not movable:
                    continue
                task = rng.choice(movable)
                old = candidate["assign"][task]
                choices = [index for index, kind in enumerate(self.types)
                           if index != old and
                           kind in self.tasks[task]["allowed"]]
                if not choices:
                    continue
                candidate["assign"][task] = rng.choice(choices)
            elif move == "anchor_move":
                choices = list(range(len(self.types)))
                if not choices:
                    continue
                index = rng.choice(choices)
                candidate["anchors"][index] = rng.choice(
                    self.anchors_by_type[self.types[index]])
            elif move == "bundle_swap":
                choices = list(range(len(self.types)))
                if len(choices) < 2:
                    continue
                left, right = rng.sample(choices, 2)
                candidate["anchors"][left], candidate["anchors"][right] = (
                    candidate["anchors"][right],
                    candidate["anchors"][left])
            ok, _ = self.feasible(candidate)
            if ok:
                return candidate
        return None

    def crossover(self, left, right, rng):
        # Inherit one complete legal geometry.  Uniform mapping crossover is
        # capacity-checked, so it remains valid when expert_slots is greater
        # than one; an invalid combination falls back to a legal parent.
        child = state_copy(left if rng.random() < 0.5 else right)
        for task in sorted(self.tasks):
            if rng.random() < 0.5:
                candidate_index = right["assign"][task]
                if self.types[candidate_index] in self.tasks[task]["allowed"]:
                    child["assign"][task] = candidate_index
        ok, _ = self.feasible(child)
        return child if ok else state_copy(left)


def append_trace(trace, start, evaluations, best, ilp_incumbent,
                 lower_bound, force=False):
    elapsed = time.perf_counter() - start
    record = {"t_s": elapsed, "evaluations": evaluations,
              "best_T_cycles": best,
              "relative_vs_ilp_incumbent_pct": relative_pct(
                  best, ilp_incumbent),
              "relative_excess_vs_available_lower_bound_pct": relative_pct(
                  best, lower_bound)}
    if (force or not trace or best < trace[-1]["best_T_cycles"] - EPS or
            elapsed - trace[-1]["t_s"] >= 0.05):
        trace.append(record)


def run_sa(model, seed, seed_wall_budget, ilp_incumbent, lower_bound=None):
    rng = random.Random(seed)
    start = time.perf_counter()
    current = model.random_state(rng)
    current_score = model.objective_T(current)
    evaluations = 1
    best = state_copy(current)
    best_score = current_score
    trace = []
    first_one = 0.0 if best_score <= 1.01 * ilp_incumbent else None
    append_trace(trace, start, evaluations, best_score, ilp_incumbent,
                 lower_bound, force=True)
    initial_temperature = max(0.05 * current_score, 1.0)
    while time.perf_counter() - start < seed_wall_budget:
        progress = min(
            (time.perf_counter() - start) / seed_wall_budget, 1.0)
        temperature = initial_temperature * (1.0e-4 ** progress)
        candidate = model.mutate(current, rng)
        if candidate is None:
            continue
        candidate_score = model.objective_T(candidate)
        evaluations += 1
        delta = candidate_score - current_score
        if delta <= 0.0 or rng.random() < math.exp(
                -delta / max(temperature, 1.0e-12)):
            current, current_score = candidate, candidate_score
            if current_score < best_score - EPS:
                best, best_score = state_copy(current), current_score
                if (first_one is None and
                        best_score <= 1.01 * ilp_incumbent):
                    first_one = time.perf_counter() - start
        append_trace(trace, start, evaluations, best_score, ilp_incumbent,
                     lower_bound)
    append_trace(trace, start, evaluations, best_score, ilp_incumbent,
                 lower_bound, force=True)
    return best, best_score, evaluations, time.perf_counter() - start, first_one, trace


def tournament(population, rng, count=3):
    choices = rng.sample(population, min(count, len(population)))
    return min(choices, key=lambda item: item[0])[1]


def run_ga(model, seed, seed_wall_budget, ilp_incumbent, lower_bound=None,
           population_size=24):
    rng = random.Random(seed)
    start = time.perf_counter()
    population = []
    evaluations = 0
    while (len(population) < population_size and
           time.perf_counter() - start < seed_wall_budget):
        state = model.random_state(rng)
        population.append((model.objective_T(state), state))
        evaluations += 1
    if not population:
        raise RuntimeError("GA initialization exhausted its wall budget")
    population.sort(key=lambda item: item[0])
    best_score, best = population[0][0], state_copy(population[0][1])
    trace = []
    first_one = (time.perf_counter() - start
                 if best_score <= 1.01 * ilp_incumbent else None)
    append_trace(trace, start, evaluations, best_score, ilp_incumbent,
                 lower_bound, force=True)
    while time.perf_counter() - start < seed_wall_budget:
        left = tournament(population, rng)
        right = tournament(population, rng)
        child = model.crossover(left, right, rng)
        mutations = 1 + int(rng.random() < 0.35)
        for _ in range(mutations):
            changed = model.mutate(child, rng)
            if changed is not None:
                child = changed
        score = model.objective_T(child)
        evaluations += 1
        population.append((score, child))
        population.sort(key=lambda item: item[0])
        population = population[:population_size]
        if score < best_score - EPS:
            best_score, best = score, state_copy(child)
            if (first_one is None and
                    best_score <= 1.01 * ilp_incumbent):
                first_one = time.perf_counter() - start
        append_trace(trace, start, evaluations, best_score, ilp_incumbent,
                     lower_bound)
    append_trace(trace, start, evaluations, best_score, ilp_incumbent,
                 lower_bound, force=True)
    return best, best_score, evaluations, time.perf_counter() - start, first_one, trace


def source_reference(solution):
    incumbent = solution.get("primary_incumbent_cycles")
    if incumbent is None:
        incumbent = solution.get("predicted_makespan_cycles")
    lower_bound = solution.get("primary_objective_bound_cycles")
    incumbent = float(incumbent) if incumbent is not None else None
    lower_bound = (float(lower_bound)
                   if lower_bound is not None and
                   math.isfinite(float(lower_bound)) and
                   float(lower_bound) > 0.0 else None)
    if incumbent is None or not math.isfinite(incumbent) or incumbent <= 0.0:
        raise ValueError("fresh ILP-Perf solution lacks a finite incumbent T")
    bound_consistent = (lower_bound is None or
                        lower_bound <= incumbent +
                        max(1.0e-6, 1.0e-8 * incumbent))
    if not bound_consistent:
        raise ValueError("ILP lower bound exceeds its incumbent")
    return {
        "incumbent_cycles": incumbent,
        "lower_bound_cycles": lower_bound,
        "lower_bound_available": lower_bound is not None,
    }


def validate_canonical_root(canonical_root):
    """Return a usable canonical root, rejecting retired data fail-closed."""
    canonical_root = Path(canonical_root).resolve()
    if not canonical_root.is_dir():
        raise FileNotFoundError(
            "canonical root does not exist: %s" % canonical_root)
    superseded = [path for path in canonical_root.iterdir()
                  if path.name.upper().startswith("SUPERSEDED")]
    if superseded:
        raise ValueError(
            "canonical root is marked SUPERSEDED and must not be audited: %s" %
            canonical_root)
    manifest_path = canonical_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "canonical root lacks manifest.json: %s" % canonical_root)
    return canonical_root


def resolve_canonical_root(project_root, requested_root=None):
    """Resolve the explicitly named, non-retired canonical result root."""
    project_root = Path(project_root).resolve()
    if requested_root is None:
        raise ValueError(
            "--canonical-root is required; implicit or stale experiment roots "
            "are forbidden")
    candidate = Path(requested_root)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return validate_canonical_root(candidate)


def load_formal_cases(canonical_root):
    canonical_root = validate_canonical_root(canonical_root)
    manifest_path = canonical_root / "manifest.json"
    manifest = read_json(manifest_path)
    protocol = manifest.get("protocol") or {}
    if not protocol.get("fresh_ilp_perf"):
        raise ValueError("manifest does not attest fresh ILP-Perf solutions")
    if protocol.get("ilp_perf_objective") != "sole-joint-min-T":
        raise ValueError("manifest is not the sole-joint-min-T protocol")
    cases = []
    for case in manifest.get("cases", []):
        record = case.get("ilp_perf") or {}
        if (record.get("status") != "complete" or
                record.get("reused") is not False):
            raise ValueError("%s is not a fresh complete ILP-Perf solve" %
                             case.get("key", case.get("case_id")))
        command = list(record.get("command") or [])
        try:
            command_params_path = Path(command[command.index("--params") + 1])
        except (ValueError, IndexError):
            raise ValueError("%s lacks a recorded --params input" %
                             case.get("key", case.get("case_id")))
        params_path = Path(case.get("params_path", command_params_path))
        solution_path = Path(record["solution_path"])
        if not params_path.is_absolute():
            params_path = PROJECT_ROOT / params_path
        if not command_params_path.is_absolute():
            command_params_path = PROJECT_ROOT / command_params_path
        if params_path.resolve() != command_params_path.resolve():
            raise ValueError("%s params path disagrees with the solve command" %
                             case.get("key", case.get("case_id")))
        if not solution_path.is_absolute():
            solution_path = canonical_root / solution_path
        params_sha = case.get("params_sha256") or record.get("params_sha256")
        if not params_sha:
            raise ValueError("%s lacks a recorded params SHA-256" %
                             params_path)
        if sha256(params_path) != params_sha:
            raise ValueError("%s does not match the params SHA-256" %
                             params_path)
        recorded_sha = record.get("solution_sha256")
        if not recorded_sha:
            raise ValueError("%s lacks a recorded solution SHA-256" %
                             solution_path)
        if sha256(solution_path) != recorded_sha:
            raise ValueError("%s does not match the manifest SHA-256" %
                             solution_path)
        solution = read_json(solution_path)
        if solution.get("optimization_mode") != "ilp-perf":
            raise ValueError("%s is not an ILP-Perf solution" % solution_path)
        contract = solution.get("objective_contract") or {}
        if (contract.get("active_objective_count") != 1 or
                contract.get("active_objective") != "min T"):
            raise ValueError("%s violates the sole-min-T contract" %
                             solution_path)
        if contract.get("protocol_completion_in_T") is not True:
            raise ValueError("%s does not optimize protocol completion in T" %
                             solution_path)
        if contract.get("energy_or_byte_hop_tie_break") is not False:
            raise ValueError("%s has an energy/byte-hop tie-break" %
                             solution_path)
        if solution.get("performance_cap_enforced") is not False:
            raise ValueError("%s is not an uncapped performance-only solve" %
                             solution_path)
        proxy = solution.get("noc_power_proxy") or {}
        if (proxy.get("traffic_kind") != "deterministic-dispatch" or
                proxy.get("used_as_objective") is not False):
            raise ValueError(
                "%s is not a deterministic-dispatch performance-only solve" %
                solution_path)
        protocol_model = solution.get("protocol_model") or {}
        if (protocol_model.get("kind") !=
                "fixed-order-max-plus-event-DAG" or
                protocol_model.get("included_in_T") is not True):
            raise ValueError("%s lacks the active fixed-order protocol T" %
                             solution_path)
        if "p_col" in (case.get("budgets") or {}):
            raise ValueError("%s contains retired C9/p_col" % solution_path)
        params_value = read_json(params_path)
        groups = int(params_value.get("meta", {}).get("groups", 0))
        cases.append({
            "case_id": case["case_id"],
            "case_key": case.get("key", case["case_id"]),
            "workload": "moe-%d-experts" % groups,
            "params": params_value,
            "params_path": params_path,
            "solution": solution,
            "solution_path": solution_path,
            "budgets": case["budgets"],
            "grid": case["grid"],
        })
    return cases


def csv_write(path, fieldnames, rows):
    """Atomically replace *path*, emitting each input record exactly once."""
    temporary = Path(str(path) + ".tmp")
    with open(str(temporary), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(str(temporary), str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Wall-clock SA/GA fixed-BOM baselines for sole min-T")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--canonical-root", required=True,
        help="explicit fresh canonical result root (SUPERSEDED roots rejected)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args(argv)
    budget_contract = wall_budget_contract()
    seed_allocations = budget_contract["fixed_seed_allocations"]
    seeds = [allocation["seed"] for allocation in seed_allocations]
    project_root = Path(args.project_root).resolve()
    try:
        canonical_root = resolve_canonical_root(
            project_root, args.canonical_root)
        cases = load_formal_cases(canonical_root)
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    solution_dir = out_dir / "solutions"
    solution_dir.mkdir(exist_ok=True)

    if args.case:
        requested = set(args.case)
        cases = [case for case in cases
                 if case["case_id"] in requested or
                 case["case_key"] in requested]
        matched = {name for case in cases
                   for name in (case["case_id"], case["case_key"])}
        missing = requested - matched
        if missing:
            parser.error("unknown case(s): %s" % ", ".join(sorted(missing)))

    started = time.time()
    summary_cases = []
    run_rows = []
    trace_rows = []
    for case in cases:
        model = CaseModel(case["case_id"], case["params"], case["solution"],
                          case["budgets"], case["grid"])
        reference = source_reference(case["solution"])
        incumbent = reference["incumbent_cycles"]
        lower_bound = reference["lower_bound_cycles"]
        source_state = model.source_state()
        source_score = model.objective_T(source_state)
        source_vs_incumbent = relative_pct(source_score, incumbent)
        source_vs_bound = relative_pct(source_score, lower_bound)
        bound_consistent = (
            lower_bound is None or
            source_score + max(1.0e-6, 1.0e-8 * source_score) >= lower_bound)
        if not bound_consistent:
            raise RuntimeError(
                "%s: restricted rescore falls below the saved ILP lower bound; "
                "the objectives are not comparable" % case["case_id"])
        case_record = {
            "case_id": case["case_id"],
            "case_key": case["case_key"],
            "workload": case["workload"],
            "grid": case["grid"],
            "budgets": case["budgets"],
            "exact_bom": dict(collections.Counter(model.types)),
            "params_path": str(Path(case["params_path"]).resolve()),
            "params_sha256": sha256(case["params_path"]),
            "source_solution_path": str(Path(case["solution_path"]).resolve()),
            "source_solution_sha256": sha256(case["solution_path"]),
            "saved_ilp_incumbent_cycles": incumbent,
            "available_ilp_lower_bound_cycles": lower_bound,
            "saved_primary_wall_s": case["solution"].get("primary_wall_s"),
            "saved_first_feasible_s": case["solution"].get(
                "time_to_first_feasible_s"),
            "rescored_ilp_layout_restricted_T_cycles": source_score,
            "rescored_ilp_layout_relative_vs_incumbent_pct": (
                source_vs_incumbent),
            "rescored_ilp_layout_relative_excess_vs_lower_bound_pct": (
                source_vs_bound),
            "available_lower_bound_consistent_with_rescore": bound_consistent,
            "wall_budget_s_per_method_case": METHOD_CASE_WALL_BUDGET_S,
            "wall_budget_alignment": copy.deepcopy(budget_contract),
            "fixed_seed_allocations": copy.deepcopy(seed_allocations),
            "method_aggregates": {},
            "runs": [],
        }
        print("[case] %s incumbent=%.6f lower_bound=%s source=%.6f" %
              (case["case_id"], incumbent,
               "unavailable" if lower_bound is None else "%.6f" % lower_bound,
               source_score), flush=True)
        for algorithm, runner in (("sa", run_sa), ("ga", run_ga)):
            method_block_started = time.perf_counter()
            method_search_walls = []
            for allocation in seed_allocations:
                seed = allocation["seed"]
                seed_wall_budget = allocation["wall_budget_s"]
                result = runner(model, seed, seed_wall_budget, incumbent,
                                lower_bound)
                state, score, evaluations, wall, first_one, trace = result
                method_search_walls.append(wall)
                strict = model.strict_evaluate(state)
                if not strict.get("feasible"):
                    raise RuntimeError(
                        "%s %s seed %d emitted an infeasible candidate: %s" %
                        (case["case_id"], algorithm, seed,
                         "; ".join(strict.get("reasons", []))))
                if (lower_bound is not None and
                        score + max(1.0e-6, 1.0e-8 * score) < lower_bound):
                    raise RuntimeError(
                        "%s %s seed %d beats the solver lower bound; refusing "
                        "a misleading same-objective report" %
                        (case["case_id"], algorithm, seed))
                relative_incumbent = relative_pct(score, incumbent)
                relative_bound = relative_pct(score, lower_bound)
                solution = model.solution(state, algorithm, seed, score)
                solution["strict_evaluate"] = strict
                solution["reference_comparison"] = {
                    "ilp_incumbent_cycles": incumbent,
                    "available_lower_bound_cycles": lower_bound,
                    "relative_vs_ilp_incumbent_pct": relative_incumbent,
                    "relative_excess_vs_available_lower_bound_pct": (
                        relative_bound),
                }
                solution_path = solution_dir / (
                    "%s_%s_seed%d.json" % (case["case_id"], algorithm, seed))
                atomic_json(solution_path, solution)
                record = {
                    "algorithm": algorithm,
                    "seed": seed,
                    "seed_wall_budget_s": seed_wall_budget,
                    "method_case_wall_budget_s": METHOD_CASE_WALL_BUDGET_S,
                    "wall_s": wall,
                    "initialization_included_in_wall_s": True,
                    "evaluations": evaluations,
                    "best_T_cycles": score,
                    "relative_vs_ilp_incumbent_pct": relative_incumbent,
                    "relative_excess_vs_available_lower_bound_pct": (
                        relative_bound),
                    "time_to_within_1pct_ilp_incumbent_s": first_one,
                    "strict_feasible": bool(strict.get("feasible")),
                    "strict_reasons": strict.get("reasons", []),
                    "cpu_anchor": list(state["anchors"][0]),
                    "cpu_coordinate_is_search_variable": True,
                    "solution_path": str(solution_path.resolve()),
                    "trace": trace,
                }
                case_record["runs"].append(record)
                run_rows.append({
                    "case_id": case["case_id"],
                    "workload": case["workload"],
                    "algorithm": algorithm,
                    "seed": seed,
                    "seed_wall_budget_s": seed_wall_budget,
                    "method_case_wall_budget_s": METHOD_CASE_WALL_BUDGET_S,
                    "wall_s": wall,
                    "evaluations": evaluations,
                    "best_T_cycles": score,
                    "saved_ilp_incumbent_cycles": incumbent,
                    "available_ilp_lower_bound_cycles": lower_bound,
                    "relative_vs_ilp_incumbent_pct": relative_incumbent,
                    "relative_excess_vs_available_lower_bound_pct": (
                        relative_bound),
                    "time_to_within_1pct_ilp_incumbent_s": first_one,
                    "strict_feasible": bool(strict.get("feasible")),
                    "solution_path": str(solution_path.resolve()),
                })
                for point in trace:
                    row = dict(point)
                    row.update({"case_id": case["case_id"],
                                "workload": case["workload"],
                                "algorithm": algorithm, "seed": seed,
                                "seed_wall_budget_s": seed_wall_budget,
                                "method_case_wall_budget_s": (
                                    METHOD_CASE_WALL_BUDGET_S)})
                    trace_rows.append(row)
                print("  %-2s seed=%d best=%.6f vs-inc=%+.4f%% eval=%d" %
                      (algorithm.upper(), seed, score, relative_incumbent,
                       evaluations),
                      flush=True)
            configured_total = sum(
                allocation["wall_budget_s"]
                for allocation in seed_allocations)
            if configured_total > METHOD_CASE_WALL_BUDGET_S + EPS:
                raise AssertionError(
                    "%s configured seed budgets exceed %.3f seconds" %
                    (algorithm, METHOD_CASE_WALL_BUDGET_S))
            case_record["method_aggregates"][algorithm] = {
                "algorithm_family": (
                    "simulated annealing" if algorithm == "sa" else
                    "single-objective GA with NSGA-style population operators"),
                "fixed_seeds": list(seeds),
                "configured_total_wall_budget_s": configured_total,
                "configured_wall_budget_cap_s": METHOD_CASE_WALL_BUDGET_S,
                "configured_budget_exceeds_cap": False,
                "observed_search_wall_s_sum": sum(method_search_walls),
                "observed_method_block_wall_s_including_audit_output": (
                    time.perf_counter() - method_block_started),
                "initialization_included_in_each_seed_wall_s": True,
            }
        summary_cases.append(case_record)

    summary = {
        "schema_version": 4,
        "status": "complete",
        "started_epoch_s": started,
        "finished_epoch_s": time.time(),
        "experiment": {
            "source_canonical_root": str(canonical_root),
            "objective": "same scalar objective as ILP-Perf: minimize T",
            "T_definition": (
                "max(replay-aligned deterministic-dispatch fixed-order "
                "max-plus completion, C10d/e aggregate directed-link "
                "capacity bound)"),
            "legacy_task_schedule_role": (
                "hard C6b/C11-C13 feasibility witness only; its whole-task "
                "makespan is not scored"),
            "search_space": (
                "fresh serialized ILP-Perf BOM fixed; randomized legal task "
                "mapping and placement; CPU coordinate is movable"),
            "routing_policy": (
                "deterministic unsplittable dimension-order shortest paths; a "
                "restricted feasible subset of the MILP's source-aggregated "
                "splittable shortest-path flow choices"),
            "expert_sharing": (
                "resident expert capacity read from library expert_slots; "
                "co-resident tasks execute on one non-preemptive lane"),
            "local_power_density_C9": "removed; p_col is neither read nor scored",
            "power_mode": False,
            "performance_cap": False,
            "energy_or_byte_hop_tie_break": False,
            "fairness": (
                "Each method receives one 45-s wall-clock budget per case, "
                "split into three fixed 15-s seeds (11,29,47). Initialization "
                "is charged; no ILP task mapping/coordinates seed either "
                "method."),
            "claim_boundary": (
                "Same T objective but a fixed-BOM and restricted schedule/route "
                "policy; the only reported comparisons are relative to the "
                "saved ILP incumbent and, separately, its available lower "
                "bound."),
            "wall_budget_s_per_method_case": METHOD_CASE_WALL_BUDGET_S,
            "fixed_seed_allocations": copy.deepcopy(seed_allocations),
            "configured_seed_budget_sum_s": sum(
                allocation["wall_budget_s"]
                for allocation in seed_allocations),
            "fixed_seeds": seeds,
            "algorithms": ["sa", "ga"],
            "algorithm_families": {
                "sa": "simulated annealing",
                "ga": (
                    "single-objective GA with NSGA-style population "
                    "operators")},
            "python": sys.version,
        },
        "implementation_audit": {
            "old_baselines_py_reused": False,
            "old_baselines_reason": (
                "it hard-codes expert exclusivity/C9 semantics and omits the "
                "current directed-link capacity score"),
            "comparison_references": [
                "saved ILP incumbent", "available ILP lower bound"],
        },
        "cases": summary_cases,
    }
    atomic_json(out_dir / "summary.json", summary)
    csv_write(out_dir / "runs.csv", [
        "case_id", "workload", "algorithm", "seed",
        "seed_wall_budget_s", "method_case_wall_budget_s", "wall_s",
        "evaluations", "best_T_cycles",
        "saved_ilp_incumbent_cycles", "available_ilp_lower_bound_cycles",
        "relative_vs_ilp_incumbent_pct",
        "relative_excess_vs_available_lower_bound_pct",
        "time_to_within_1pct_ilp_incumbent_s", "strict_feasible",
        "solution_path"], run_rows)
    csv_write(out_dir / "anytime.csv", [
        "case_id", "workload", "algorithm", "seed",
        "seed_wall_budget_s", "method_case_wall_budget_s", "t_s",
        "evaluations", "best_T_cycles",
        "relative_vs_ilp_incumbent_pct",
        "relative_excess_vs_available_lower_bound_pct"], trace_rows)

    # Compact machine-derived conclusion; detailed values remain in CSV/JSON.
    lines = [
        "# Metaheuristic min-T audit",
        "",
        "SA and GA minimize the replay-aligned fixed-order protocol T with "
        "co-resident single-lane feasibility and C10 directed-link capacity. "
        "The legacy whole-task makespan is not scored. They receive "
        "an identical 45-s total wall budget per method/case, split across "
        "fixed seeds 11/29/47 (15 s each), and the fresh ILP-selected BOM; "
        "initialization is included in each seed's wall time.",
        "",
        "Deterministic dimension-order routing is a restricted legal subset of "
        "the MILP flow space. This is not variable-BOM search or LEGOSim.",
        "",
        "| case | SA median vs ILP incumbent | GA median vs ILP incumbent |",
        "|---|---:|---:|",
    ]
    for case in summary_cases:
        by_algorithm = collections.defaultdict(list)
        for record in case["runs"]:
            by_algorithm[record["algorithm"]].append(
                record["relative_vs_ilp_incumbent_pct"])
        sa_gap = statistics.median(by_algorithm["sa"])
        ga_gap = statistics.median(by_algorithm["ga"])
        lines.append("| %s | %+.4f%% | %+.4f%% |" %
                     (case["case_id"], sa_gap, ga_gap))
    lines.extend([
        "",
        "Negative values versus the ILP incumbent are legitimate incumbent "
        "improvements. The lower-bound column is reported separately when "
        "available; neither comparison is called a gap to optimum.",
        "",
    ])
    with open(str(out_dir / "README.md"), "w", encoding="utf-8",
              newline="\n") as handle:
        handle.write("\n".join(lines))
    print("[done] %s" % (out_dir / "summary.json"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
