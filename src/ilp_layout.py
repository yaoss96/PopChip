#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ilp_layout.py -- Compact ILP for 3D chiplet layout of a MoE decoder layer.

Implements Section 4 of the design doc:
  C1  footprint non-overlap / single occupancy
  C2  instance-count bounds (optional BOM)
  C3  each task placed exactly once
  C4  type compatibility
  C5  memory capacity
  C6  configurable resident-expert capacity plus exact co-resident scheduling
  C7  area (total + per-layer)
  C8  package power (TDP)
  C10 separable rectilinear distance linearization (the KEY compact trick:
      O(|V||P|) binaries instead of O(|E||P|^2))
  C11 timing / precedence
  C12 legacy task-schedule feasibility horizon
  C13 chiplet busy-time relaxation
  Objective: minimize replay-aligned protocol makespan T subject to area and
             package-power budgets
  Symmetry handling: anchor-indexed activation has no labelled same-type
  instances; Gurobi's internal symmetry detection is enabled by default.

Runs under Gurobi 10.0.1 / py37 (only gurobipy + stdlib). No numpy/yaml.

Usage:
  python ilp_layout.py --params params.json --L 4 --X 4 --Y 4 \
        --out solution.json [--p-max 800] [--a-max 2500] \
        [--time-limit 3600] [--mipgap 0.005] [--mip-focus 1]
"""

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as ex:      # pragma: no cover
    sys.stderr.write("ERROR: gurobipy import failed: %s\n" % ex)
    sys.stderr.write("Run with a Python interpreter that provides gurobipy, e.g.\n"
                     "  python "
                     "ilp_layout.py ...\n")
    sys.exit(2)


def _format_optional_float(value, pattern="%.1f", missing="unavailable"):
    """Format optional diagnostics without turning a valid solve into failure."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(number):
        return missing
    return pattern % number


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def make_sites(L, X, Y):
    """P = {(l,i,j)}. Flat index idx = l*X*Y + i*Y + j."""
    P = [(l, i, j) for l in range(L) for i in range(X) for j in range(Y)]
    idx = {p: n for n, p in enumerate(P)}
    return P, idx


def anchors_and_footprints(L, X, Y, library):
    """For each type k, the set of legal anchors A(k) and, for each anchor,
    its footprint F(k,anchor) as a set of sites. Anchor = lowest-index corner,
    footprint stays within one layer and inside the grid."""
    anchors = {}
    footprint = {}
    for k, spec in library.items():
        w, h = spec["w"], spec["h"]
        A = []
        F = {}
        for l in range(L):
            for i in range(X - w + 1):
                for j in range(Y - h + 1):
                    pa = (l, i, j)
                    cells = {(l, i + di, j + dj)
                             for di in range(w) for dj in range(h)}
                    A.append(pa)
                    F[pa] = cells
        anchors[k] = A
        footprint[k] = F
    return anchors, footprint


def reference_bom(solution, library):
    """Return and validate the chiplet-type counts in a serialized solution."""
    counts = {k: 0 for k in library}
    seen = set()
    for number, site in enumerate(solution.get("sites", [])):
        try:
            anchor = (int(site["l"]), int(site["i"]), int(site["j"]))
            kind = str(site["type"])
        except Exception:
            raise ValueError("reference site %d is malformed" % number)
        if anchor in seen:
            raise ValueError("reference solution has duplicate anchor %s" %
                             (anchor,))
        if kind not in library:
            raise ValueError("reference solution uses unknown type %s" % kind)
        seen.add(anchor)
        counts[kind] += 1
    if not seen:
        raise ValueError("reference solution has no occupied sites")
    return counts


def reference_bundles(solution, tasks, library):
    """Validate complete reference bundles for placement-only optimization.

    A bundle is the type and complete task set of one occupied reference
    chiplet.  Its original coordinate is deliberately discarded: the new
    model may move the whole bundle, but may neither split nor merge it.
    """
    task_site = solution.get("task_site", {})
    expected = set(tasks)
    if set(task_site) != expected:
        missing = sorted(expected - set(task_site))
        extra = sorted(set(task_site) - expected)
        raise ValueError("reference task set mismatch; missing=%s extra=%s" %
                         (missing, extra))

    by_anchor = {}
    for number, site in enumerate(solution.get("sites", [])):
        try:
            anchor = (int(site["l"]), int(site["i"]), int(site["j"]))
            kind = str(site["type"])
        except Exception:
            raise ValueError("reference site %d is malformed" % number)
        if kind not in library:
            raise ValueError("reference solution uses unknown type %s" % kind)
        if anchor in by_anchor:
            raise ValueError("reference solution has duplicate anchor %s" %
                             (anchor,))
        by_anchor[anchor] = {"type": kind, "tasks": []}

    for task, raw_anchor in task_site.items():
        try:
            anchor = tuple(int(value) for value in raw_anchor)
        except Exception:
            raise ValueError("reference task %s has malformed anchor" % task)
        if anchor not in by_anchor:
            raise ValueError("reference task %s maps to absent anchor %s" %
                             (task, anchor))
        kind = by_anchor[anchor]["type"]
        if kind not in tasks[task]["allowed"]:
            raise ValueError("reference task %s is incompatible with %s" %
                             (task, kind))
        by_anchor[anchor]["tasks"].append(task)

    bundles = []
    for bundle_id, (anchor, info) in enumerate(sorted(by_anchor.items())):
        if not info["tasks"]:
            raise ValueError("reference bundle at %s is empty" % (anchor,))
        bundles.append({
            "id": bundle_id,
            "type": info["type"],
            "tasks": sorted(info["tasks"]),
            "source_anchor": anchor,
        })
    return bundles


def bundle_signature(bundles):
    """Return a coordinate-independent digest of typed complete bundles."""
    canonical = sorted(
        ({"type": str(bundle["type"]),
          "tasks": sorted(str(task) for task in bundle.get("tasks", []))}
         for bundle in bundles),
        key=lambda bundle: (bundle["type"], bundle["tasks"]))
    payload = json.dumps(canonical, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _largest_remainder(weights, total):
    """Match codegen.py's deterministic integer apportionment exactly."""
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
    left = total - sum(base)
    order = sorted(range(len(weights)),
                   key=lambda i: (-(exact[i] - base[i]), i))
    for index in order[:left]:
        base[index] += 1
    return base


def _balanced_transport(column_totals, nrows, row_total):
    """Match codegen.py's cyclic balanced transport exactly."""
    totals = [int(value) for value in column_totals]
    if nrows <= 0 or row_total < 0:
        raise ValueError("invalid balanced-transport dimensions")
    if sum(totals) != nrows * row_total:
        raise ValueError("balanced-transport margins disagree")
    matrix = [[0] * len(totals) for _ in range(nrows)]
    cursor = 0
    for column, total in enumerate(totals):
        quotient, remainder = divmod(total, nrows)
        for source in range(nrows):
            matrix[source][column] = quotient
        for _ in range(remainder):
            matrix[cursor][column] += 1
            cursor = (cursor + 1) % nrows
    if ([sum(row) for row in matrix] != [row_total] * nrows or
            [sum(matrix[row][column] for row in range(nrows))
             for column in range(len(totals))] != totals):
        raise AssertionError("deterministic transport failed")
    return matrix


def deterministic_dispatch(params, tokens_per_source):
    """Return codegen-equivalent logical expert-to-expert token traffic."""
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
    targets = _largest_remainder(weights, groups * row_total)
    return _balanced_transport(targets, groups, row_total)


def _evaluate_fixed_order_protocol(
        cpu_site, expert_sites, dispatch_matrix, compute_cycles,
        token_bytes_per_phase, init_bytes, done_bytes,
        sync_overhead_cycles, flit_bytes, hop_cycles,
        expert_resource_ids=None, compute_order=None):
    """Evaluate the deterministic blocking protocol at fixed coordinates.

    This is the coordinate-level counterpart of the max-plus event DAG built
    in :func:`build_and_solve`.  Evaluating the recurrence directly is
    important after a lexicographic solve: lower-bounded event-clock variables
    may otherwise retain harmless slack up to a performance cap even though
    the selected coordinates imply a strictly smaller completion time.
    """
    group_count = len(expert_sites)
    if len(dispatch_matrix) != group_count:
        raise ValueError("dispatch row count and expert count disagree")
    if len(compute_cycles) != group_count:
        raise ValueError("compute-cycle count and expert count disagree")
    if any(len(row) != group_count for row in dispatch_matrix):
        raise ValueError("dispatch matrix must be square")
    if float(flit_bytes) <= 0.0:
        raise ValueError("protocol flit size must be positive")

    def distance(first, second):
        if len(first) != 3 or len(second) != 3:
            raise ValueError("protocol coordinates must be three-dimensional")
        return sum(abs(float(first[axis]) - float(second[axis]))
                   for axis in range(3))

    clocks = {"cpu": 0.0}
    clocks.update({group: 0.0 for group in range(group_count)})
    event_finish_cycles = []
    compute_start_cycles = [None] * group_count
    compute_finish_cycles = [None] * group_count

    def process_message(source, target, volume_bytes, route_distance):
        serialization = float(int(math.ceil(
            float(volume_bytes) / float(flit_bytes))))
        write_time = clocks[source] + float(sync_overhead_cycles)
        sender_finish = write_time + serialization
        receiver_finish = max(
            write_time + serialization +
            float(hop_cycles) * float(route_distance),
            clocks[target] + float(sync_overhead_cycles))
        clocks[source] = sender_finish
        clocks[target] = receiver_finish
        event_finish_cycles.append(float(receiver_finish))

    # CPU initialization is serialized in stable logical-worker order.
    for group in range(group_count):
        process_message(
            "cpu", group, init_bytes,
            distance(cpu_site, expert_sites[group]))

    # Dispatch uses stable (source, target) order.
    for source in range(group_count):
        for target in range(group_count):
            assignments = int(dispatch_matrix[source][target])
            if source == target or assignments <= 0:
                continue
            process_message(
                source, target,
                assignments * float(token_bytes_per_phase),
                distance(expert_sites[source], expert_sites[target]))

    if expert_resource_ids is None:
        for group in range(group_count):
            compute_start_cycles[group] = float(clocks[group])
            clocks[group] += float(compute_cycles[group])
            compute_finish_cycles[group] = float(clocks[group])
    else:
        if len(expert_resource_ids) != group_count:
            raise ValueError("expert resource count and expert count disagree")
        order = (list(range(group_count)) if compute_order is None
                 else list(compute_order))
        if sorted(order) != list(range(group_count)):
            raise ValueError("compute_order must be a permutation of workers")
        resource_ready = {}
        for group in order:
            resource = expert_resource_ids[group]
            start = max(clocks[group], resource_ready.get(resource, 0.0))
            compute_start_cycles[group] = float(start)
            clocks[group] = start + float(compute_cycles[group])
            compute_finish_cycles[group] = float(clocks[group])
            resource_ready[resource] = clocks[group]

    # Combine traverses the same logical pairs in the reverse direction.
    for source in range(group_count):
        for target in range(group_count):
            assignments = int(dispatch_matrix[source][target])
            if source == target or assignments <= 0:
                continue
            process_message(
                target, source,
                assignments * float(token_bytes_per_phase),
                distance(expert_sites[source], expert_sites[target]))

    # Worker summaries are consumed by the CPU in stable worker order.
    for group in range(group_count):
        process_message(
            group, "cpu", done_bytes,
            distance(expert_sites[group], cpu_site))

    return {
        "completion_cycles": float(clocks["cpu"]),
        "worker_finish_cycles": [float(clocks[group])
                                 for group in range(group_count)],
        "event_finish_cycles": event_finish_cycles,
        "compute_start_cycles": compute_start_cycles,
        "compute_finish_cycles": compute_finish_cycles,
    }


def _deterministic_protocol_placement_descent(
        initial_state, bundle_ids, allowed_anchors, placement_is_legal,
        protocol_cycles, max_evaluations=1000, max_rounds=2,
        max_wall_s=5.0):
    """Improve a typed-bundle MIP start using only exact protocol cycles.

    The deterministic neighborhood contains legal pair swaps followed by legal
    one-bundle relocations.  A bounded number of neighbors is evaluated, and a
    move is accepted only when its exact fixed-order max-plus completion is
    strictly smaller.  Coordinates and BOM remain unfixed; this is only primal
    start construction.
    """
    started = time.time()
    ordered_ids = list(bundle_ids)
    allowed = {
        bid: [tuple(anchor) for anchor in allowed_anchors[bid]]
        for bid in ordered_ids}
    allowed_sets = {bid: set(values) for bid, values in allowed.items()}
    state = {bid: tuple(initial_state[bid]) for bid in ordered_ids}
    if not placement_is_legal(state):
        raise ValueError("protocol placement descent received illegal state")
    evaluation_limit = max(0, int(max_evaluations))
    round_limit = max(0, int(max_rounds))
    wall_limit = max(0.0, float(max_wall_s))
    initial_cycles = float(protocol_cycles(state))
    current_cycles = initial_cycles
    evaluations = 0
    legal_neighbors = 0
    accepted_moves = []
    stopped_reason = "round-limit"
    wall_exhausted = False
    tolerance = 1.0e-9

    for round_number in range(round_limit):
        best_state = None
        best_cycles = current_cycles
        best_move = None

        def consider(proposal, move):
            nonlocal best_state, best_cycles, best_move
            nonlocal evaluations, legal_neighbors
            nonlocal wall_exhausted
            if evaluations >= evaluation_limit:
                return False
            if time.time() - started >= wall_limit:
                wall_exhausted = True
                return False
            if not placement_is_legal(proposal):
                return True
            legal_neighbors += 1
            cycles = float(protocol_cycles(proposal))
            evaluations += 1
            if cycles < best_cycles - tolerance:
                best_state = proposal
                best_cycles = cycles
                best_move = move
            return evaluations < evaluation_limit

        move_evaluation_limit = max(1, evaluation_limit // 2)
        keep_scanning = evaluation_limit > 0
        # Relocations come first so the scale cap always examines the CPU and
        # hottest expert bundles supplied first by the caller.
        for bid in ordered_ids:
            if not keep_scanning:
                break
            for anchor in allowed[bid]:
                if anchor == state[bid]:
                    continue
                proposal = dict(state)
                proposal[bid] = anchor
                keep_scanning = consider(proposal, {
                    "kind": "relocate", "first_bundle": bid,
                    "second_bundle": None,
                    "new_anchor": list(anchor)})
                if evaluations >= move_evaluation_limit:
                    keep_scanning = False
                if not keep_scanning:
                    break

        keep_scanning = evaluations < evaluation_limit
        if keep_scanning:
            for first_offset, first in enumerate(ordered_ids):
                if not keep_scanning:
                    break
                for second in ordered_ids[first_offset + 1:]:
                    if (state[second] not in allowed_sets[first] or
                            state[first] not in allowed_sets[second]):
                        continue
                    proposal = dict(state)
                    proposal[first], proposal[second] = (
                        state[second], state[first])
                    keep_scanning = consider(proposal, {
                        "kind": "swap", "first_bundle": first,
                        "second_bundle": second, "new_anchor": None})
                    if not keep_scanning:
                        break

        if best_state is None:
            stopped_reason = (
                "wall-limit" if wall_exhausted else
                "evaluation-limit" if evaluations >= evaluation_limit else
                "protocol-local-optimum")
            break
        previous = current_cycles
        state = best_state
        current_cycles = best_cycles
        move_record = dict(best_move)
        move_record.update({
            "round": round_number + 1,
            "cycles_before": previous,
            "cycles_after": current_cycles,
        })
        accepted_moves.append(move_record)
        if wall_exhausted or evaluations >= evaluation_limit:
            stopped_reason = (
                "wall-limit" if wall_exhausted else "evaluation-limit")
            break

    return state, {
        "attempted": True,
        "kind": "bounded exact-protocol typed-bundle descent",
        "objective": "fixed-order max-plus protocol cycles only",
        "deterministic": True,
        "coordinates_fixed": False,
        "bom_fixed_by_refinement": False,
        "evaluation_limit": evaluation_limit,
        "wall_limit_s": wall_limit,
        "wall_limit_reached": wall_exhausted,
        "relocation_evaluation_share": "at most first half",
        "evaluations": evaluations,
        "legal_neighbors": legal_neighbors,
        "round_limit": round_limit,
        "rounds_completed": len(accepted_moves),
        "initial_cycles": initial_cycles,
        "final_cycles": current_cycles,
        "improved": current_cycles < initial_cycles - tolerance,
        "stopped_reason": stopped_reason,
        "moves": accepted_moves,
        "wall_s": time.time() - started,
    }


def _deterministic_power_placement_descent(
        initial_state, bundle_ids, allowed_anchors, bundle_footprints,
        pair_axis_weights, protocol_cycles, deadline_cycles,
        max_rounds=4, max_protocol_checks_per_round=64):
    """Apply deterministic best-improvement energy moves under a cycle cap.

    The neighborhood contains every legal pair swap and every legal
    one-bundle relocation.  Pairwise axis-weighted Manhattan deltas cost O(n)
    each, so one exhaustive round costs O(n^3 + n^2 E), where ``E`` is the
    number of empty/legal anchor choices per bundle.  Candidate protocol cycles
    are evaluated in increasing energy order and must satisfy ``deadline``.

    This helper is deliberately independent of Gurobi.  It is used only to
    construct a primal MIP start; the MILP remains authoritative.
    """
    ordered_ids = list(bundle_ids)
    state = {bid: tuple(initial_state[bid]) for bid in ordered_ids}
    allowed_sets = {
        bid: set(tuple(anchor) for anchor in allowed_anchors[bid])
        for bid in ordered_ids}
    rounds_limit = max(0, int(max_rounds))
    checks_limit = max(1, int(max_protocol_checks_per_round))
    tolerance = 1.0e-9

    def placement_is_legal(candidate):
        occupied = set()
        for bid in ordered_ids:
            anchor = tuple(candidate[bid])
            if anchor not in allowed_sets[bid]:
                return False
            cells = bundle_footprints[bid].get(anchor)
            if cells is None or occupied.intersection(cells):
                return False
            occupied.update(cells)
        return True

    def pair_cost(first, second, first_anchor, second_anchor):
        key = ((first, second) if first < second else (second, first))
        weights = pair_axis_weights.get(key)
        if weights is None:
            return 0.0
        return sum(float(weights[axis]) *
                   abs(first_anchor[axis] - second_anchor[axis])
                   for axis in range(3))

    def objective(candidate):
        return sum(
            pair_cost(first, second,
                      candidate[first], candidate[second])
            for offset, first in enumerate(ordered_ids)
            for second in ordered_ids[offset + 1:])

    def swap_delta(candidate, first, second):
        first_anchor = candidate[first]
        second_anchor = candidate[second]
        delta = 0.0
        for other in ordered_ids:
            if other == first or other == second:
                continue
            other_anchor = candidate[other]
            delta += pair_cost(
                first, other, second_anchor, other_anchor)
            delta -= pair_cost(
                first, other, first_anchor, other_anchor)
            delta += pair_cost(
                second, other, first_anchor, other_anchor)
            delta -= pair_cost(
                second, other, second_anchor, other_anchor)
        return delta

    def move_delta(candidate, bid, new_anchor):
        old_anchor = candidate[bid]
        delta = 0.0
        for other in ordered_ids:
            if other == bid:
                continue
            other_anchor = candidate[other]
            delta += pair_cost(bid, other, new_anchor, other_anchor)
            delta -= pair_cost(bid, other, old_anchor, other_anchor)
        return delta

    if not placement_is_legal(state):
        raise ValueError("deterministic Power descent received illegal state")
    current_objective = objective(state)
    initial_objective = current_objective
    neighbor_evaluations = 0
    protocol_checks = 0
    moves = []
    stopped_reason = "round-limit"
    for round_number in range(rounds_limit):
        candidates = []
        for first_offset, first in enumerate(ordered_ids):
            for second in ordered_ids[first_offset + 1:]:
                first_anchor = state[first]
                second_anchor = state[second]
                if (second_anchor not in allowed_sets[first] or
                        first_anchor not in allowed_sets[second]):
                    continue
                proposal = dict(state)
                proposal[first], proposal[second] = (
                    second_anchor, first_anchor)
                if not placement_is_legal(proposal):
                    continue
                delta = swap_delta(state, first, second)
                neighbor_evaluations += 1
                if delta < -tolerance:
                    candidates.append((
                        delta, 0, first, second, None))
        for bid in ordered_ids:
            old_anchor = state[bid]
            for new_anchor in allowed_anchors[bid]:
                new_anchor = tuple(new_anchor)
                if new_anchor == old_anchor:
                    continue
                proposal = dict(state)
                proposal[bid] = new_anchor
                if not placement_is_legal(proposal):
                    continue
                delta = move_delta(state, bid, new_anchor)
                neighbor_evaluations += 1
                if delta < -tolerance:
                    candidates.append((
                        delta, 1, bid, -1, new_anchor))
        candidates.sort(key=lambda entry: (
            entry[0], entry[1], entry[2], entry[3],
            entry[4] if entry[4] is not None else (-1, -1, -1)))
        if not candidates:
            stopped_reason = "energy-local-optimum"
            break

        accepted = None
        for candidate in candidates[:checks_limit]:
            delta, move_kind, first, second, new_anchor = candidate
            proposal = dict(state)
            if move_kind == 0:
                proposal[first], proposal[second] = (
                    state[second], state[first])
            else:
                proposal[first] = new_anchor
            candidate_cycles = float(protocol_cycles(proposal))
            protocol_checks += 1
            if candidate_cycles <= float(deadline_cycles) + 1.0e-6:
                accepted = (candidate, proposal, candidate_cycles)
                break
        if accepted is None:
            stopped_reason = (
                "no deadline-feasible improving neighbor within protocol-"
                "check limit")
            break
        candidate, proposal, candidate_cycles = accepted
        delta, move_kind, first, second, new_anchor = candidate
        previous_objective = current_objective
        state = proposal
        current_objective = objective(state)
        if current_objective >= previous_objective - tolerance:
            raise AssertionError(
                "deterministic Power descent accepted a non-improving move")
        moves.append({
            "round": round_number + 1,
            "kind": "swap" if move_kind == 0 else "relocate",
            "first_bundle": first,
            "second_bundle": second if move_kind == 0 else None,
            "new_anchor": (list(new_anchor)
                           if new_anchor is not None else None),
            "delta_objective": float(delta),
            "objective_before": float(previous_objective),
            "objective_after": float(current_objective),
            "protocol_cycles": candidate_cycles,
        })

    return state, {
        "enabled": rounds_limit > 0,
        "deterministic": True,
        "neighborhood": "all legal pair swaps and one-bundle relocations",
        "round_limit": rounds_limit,
        "rounds_completed": len(moves),
        "protocol_checks_per_round_limit": checks_limit,
        "neighbor_evaluations": neighbor_evaluations,
        "protocol_checks": protocol_checks,
        "initial_objective": float(initial_objective),
        "final_objective": float(current_objective),
        "strictly_improved": current_objective < initial_objective - tolerance,
        "stopped_reason": stopped_reason,
        "moves": moves,
        "complexity": (
            "O(n^3+n^2E) per round using O(n) pairwise deltas; exact "
            "protocol checks are bounded separately"),
    }


# ----------------------------------------------------------------------------
# Model builder
# ----------------------------------------------------------------------------
def build_and_solve(params, L, X, Y, budgets, solver_opts, T_hat=None,
                    verbose=True):
    build_start = time.time()
    # Energy is reported only as a post-solve diagnostic.  Reject an archived
    # epsilon-constraint input instead of silently solving a different model
    # from the paper's sole min-T contract.
    if budgets.get("e_bar") is not None:
        raise ValueError(
            "the Ebar energy budget has been retired; ILP-Perf minimizes T "
            "under area and package peak-power budgets")
    budgets = dict(budgets)
    budgets.pop("e_bar", None)
    mode = solver_opts.get("mode", "ilp-perf")
    if mode != "ilp-perf":
        raise ValueError(
            "%s has been retired; build_and_solve accepts only ilp-perf "
            "with the single joint objective min T" % mode)
    lib = params["library"]
    tasks = params["tasks"]
    edges = params["edges"]
    expert_tasks = set(params["expert_tasks"])
    ic = params["interconnect"]
    V = list(tasks.keys())
    K = list(lib.keys())
    expert_slots = {}
    for k in K:
        raw_slots = lib[k].get("expert_slots", 1)
        slots = int(raw_slots)
        if slots < 0 or float(raw_slots) != float(slots):
            raise ValueError("library.%s.expert_slots must be a nonnegative integer" % k)
        expert_slots[k] = slots
    P, idx = make_sites(L, X, Y)
    anchors, footprint = anchors_and_footprints(L, X, Y, lib)

    # A proved-safe serial horizon supplies finite bounds for the conditional
    # non-overlap disjunctions.  It serializes every task and every DAG transfer
    # at the maximum simple-path distance, then adds a one-link serialization
    # allowance for the larger of DAG and deterministic-dispatch traffic.
    beta = float(ic["beta"])
    if beta <= 0.0:
        raise ValueError("interconnect.beta must be positive")
    tau_h = float(ic["tau_h"])
    tau_v = float(ic["tau_v"])
    t0 = float(ic["t0"])
    max_route_delay = (tau_h * ((X - 1) + (Y - 1)) +
                       tau_v * (L - 1))
    compute_serial = sum(
        max(float(value) for value in tasks[v]["cyc"].values()) for v in V)
    dag_bytes = sum(float(edge["vol"]) for edge in edges)
    dispatch_bytes_for_horizon = 0.0
    meta = params.get("meta", {})
    groups_for_horizon = int(meta.get("groups", len(expert_tasks)))
    if groups_for_horizon > 0:
        try:
            horizon_matrix = deterministic_dispatch(
                params, solver_opts.get("dispatch_tokens_per_source", 32))
            horizon_token_bytes = (
                2 * int(solver_opts.get("dispatch_hidden", 128)) *
                int(solver_opts.get(
                    "dispatch_bytes_elem", meta.get("bytes_elem", 2))))
            dispatch_bytes_for_horizon = sum(
                horizon_matrix[source][target] * horizon_token_bytes
                for source in range(len(horizon_matrix))
                for target in range(len(horizon_matrix))
                if source != target)
        except (KeyError, TypeError, ValueError):
            dispatch_bytes_for_horizon = 0.0
    transfer_serial = sum(
        t0 + max_route_delay + float(edge["vol"]) / beta
        for edge in edges)
    traffic_serial = max(dag_bytes, dispatch_bytes_for_horizon) / beta
    time_horizon = max(1.0, compute_serial + transfer_serial + traffic_serial)

    # site -> list of (k, anchor) whose footprint covers it (for C1 and ktype)
    cover = {p: [] for p in P}
    for k in K:
        for pa in anchors[k]:
            for cell in footprint[k][pa]:
                cover[cell].append((k, pa))

    # Directed nearest-neighbour Mesh links.  Horizontal directions are kept
    # distinct, matching full-duplex link-capacity accounting.
    directed_links = []
    link_axis = []
    outgoing = {p: [] for p in P}
    incoming = {p: [] for p in P}
    for p in P:
        for axis, bound in ((0, L), (1, X), (2, Y)):
            for step in (-1, 1):
                q_list = list(p)
                q_list[axis] += step
                if not (0 <= q_list[axis] < bound):
                    continue
                q = tuple(q_list)
                link_id = len(directed_links)
                directed_links.append((p, q))
                link_axis.append(axis)
                outgoing[p].append(link_id)
                incoming[q].append(link_id)

    m = gp.Model("moe_3d_chiplet_layout")
    if not verbose:
        m.Params.OutputFlag = 0

    # ---- Variables ----------------------------------------------------------
    x = m.addVars([(k, pa) for k in K for pa in anchors[k]],
                  vtype=GRB.BINARY, name="x")      # chiplet type at anchor
    y = m.addVars([(v, p) for v in V for p in P],
                  vtype=GRB.BINARY, name="y")   # task on site (anchor); key=(v,(l,i,j))
    # y^(k): task v on a type-k chiplet at anchor pa (needed for cyc/eta which
    # depend on the type). Only create where task allows type k and anchor legal.
    yk = {}
    for v in V:
        for k in tasks[v]["allowed"]:
            for pa in anchors[k]:
                yk[(v, k, pa)] = m.addVar(vtype=GRB.BINARY,
                                          name="yk_%s_%s_%d" % (v, k, idx[pa]))
    s = m.addVars(V, lb=0.0, ub=time_horizon, name="s")
    f = m.addVars(V, lb=0.0, ub=time_horizon, name="f")
    ei = list(range(len(edges)))
    dI = m.addVars(ei, lb=0.0, name="dI")
    dJ = m.addVars(ei, lb=0.0, name="dJ")
    dL = m.addVars(ei, lb=0.0, name="dL")
    # ``T_task`` retains the legacy full-task analytical scale used by C11--
    # C13.  The public objective ``T`` is intentionally separate: the evaluated
    # LEGOSim program uses the fixed-order blocking protocol below, whose cycle
    # scale is not commensurate with the legacy whole-layer task profile.
    T_task = m.addVar(lb=0.0, ub=time_horizon, name="T_task")
    T = m.addVar(lb=0.0, name="T")

    # ---- C1: footprint non-overlap / single occupancy ----------------------
    for p in P:
        if cover[p]:
            m.addConstr(gp.quicksum(x[kp] for kp in cover[p]) <= 1,
                        name="C1_%d" % idx[p])

    # ---- C2: instance-count bounds (optional) -------------------------------
    nmin = budgets.get("nmin", {})
    nmax = budgets.get("nmax", {})
    for k in K:
        if k in nmax:
            m.addConstr(gp.quicksum(x[(k, pa)] for pa in anchors[k]) <= nmax[k],
                        name="C2max_%s" % k)
        if k in nmin:
            m.addConstr(gp.quicksum(x[(k, pa)] for pa in anchors[k]) >= nmin[k],
                        name="C2min_%s" % k)
    # ---- C2s: geometric (D4) symmetry break via CPU fundamental domain ------
    # With exactly one freely placed CPU on a square Mesh, every layout has a
    # dihedral (8-fold) image whose CPU anchor lies in the fundamental domain
    # {2i <= X-1, j <= i}.  Restricting the CPU there removes the Mesh symmetry
    # without losing optimality, because Manhattan distances and directed-link
    # capacities are D4-invariant.  Off by default so the reported baseline
    # runs are unchanged; enabled via solver_opts["geom_symmetry_break"].
    if (solver_opts.get("geom_symmetry_break") and X == Y and
            "CPU" in anchors):
        disallowed_cpu = [pa for pa in anchors["CPU"]
                          if not (2 * pa[1] <= X - 1 and pa[2] <= pa[1])]
        if disallowed_cpu:
            m.addConstr(
                gp.quicksum(x[("CPU", pa)] for pa in disallowed_cpu) == 0,
                name="C2sym_cpu_domain")
    # A same-BOM comparison needs equality, not only the ordinary budget/count
    # bounds.  The counts come from an already serialized reference solution.
    # They remain hard constraints for same-BOM ILP-Perf comparisons.
    bom_counts = solver_opts.get("bom_counts")
    if bom_counts is not None:
        unknown = sorted(set(bom_counts) - set(K))
        if unknown:
            raise ValueError("fixed BOM contains unknown types: %s" % unknown)
        for k in K:
            count = int(bom_counts.get(k, 0))
            if count < 0 or float(bom_counts.get(k, 0)) != float(count):
                raise ValueError("fixed BOM count for %s is not a nonnegative integer" % k)
            m.addConstr(
                gp.quicksum(x[(k, pa)] for pa in anchors[k]) == count,
                name="C2fixed_%s" % k)

    # ---- link y and yk ------------------------------------------------------
    # y[v,pa] is 1 iff task v sits on the chiplet anchored at pa.
    # A task lives at an ANCHOR of its chiplet. So y is only meaningful on
    # anchors. Tie y to yk: y[v,pa] = sum_k yk[v,k,pa]; and require the type
    # be present: yk[v,k,pa] <= x[k,pa].
    for v in V:
        for pa in P:
            terms = [yk[(v, k, pa)] for k in tasks[v]["allowed"]
                     if (v, k, pa) in yk]
            if terms:
                m.addConstr(y[v, pa] == gp.quicksum(terms),
                            name="link_%s_%d" % (v, idx[pa]))
            else:
                m.addConstr(y[v, pa] == 0, name="linkz_%s_%d" % (v, idx[pa]))
    for (v, k, pa), var in yk.items():
        m.addConstr(var <= x[(k, pa)], name="C4_%s_%s_%d" % (v, k, idx[pa]))
    # Activation is bidirectional: an instantiated chiplet must host at least
    # one compatible task.  Without this valid-use link, the pure-T primary
    # stage may turn on arbitrary idle dies, creating large symmetric branches
    # that cannot improve makespan and making equivalent area/power tiers solve
    # very differently.  Requiring each selected instance to be used therefore
    # strengthens the min-T formulation without excluding a useful layout.
    for k in K:
        for pa in anchors[k]:
            users = [yk[(v, k, pa)] for v in V if (v, k, pa) in yk]
            if users:
                m.addConstr(x[(k, pa)] <= gp.quicksum(users),
                            name="C4use_%s_%d" % (k, idx[pa]))
            else:
                m.addConstr(x[(k, pa)] == 0,
                            name="C4unused_%s_%d" % (k, idx[pa]))

    # ---- C3: each task placed exactly once ----------------------------------
    m.addConstrs((gp.quicksum(y[v, p] for p in P) == 1 for v in V), name="C3")

    # ---- Optional placement-only bundle fixing ------------------------------
    # z[b,p] moves complete reference chiplet bundles.  Linking x and yk to z
    # preserves the exact BOM, type, and task grouping while leaving physical
    # anchors free.  This is stricter than merely fixing type counts and is the
    # appropriate search space for a same-BOM, placement-only comparison.
    placement_bundles = solver_opts.get("placement_bundles")
    bundle_z = None
    if placement_bundles is not None:
        bundle_tasks = set()
        for bundle in placement_bundles:
            kind = bundle["type"]
            if kind not in K:
                raise ValueError("placement bundle uses unknown type %s" % kind)
            if not bundle["tasks"]:
                raise ValueError("placement bundle %s is empty" % bundle["id"])
            for v in bundle["tasks"]:
                if v in bundle_tasks:
                    raise ValueError("task %s occurs in multiple bundles" % v)
                if v not in tasks or kind not in tasks[v]["allowed"]:
                    raise ValueError("task %s cannot remain on bundle type %s" %
                                     (v, kind))
                bundle_tasks.add(v)
        if bundle_tasks != set(V):
            raise ValueError("placement bundles do not cover the model task set")

        z_keys = [(bundle["id"], pa)
                  for bundle in placement_bundles
                  for pa in anchors[bundle["type"]]]
        bundle_z = m.addVars(z_keys, vtype=GRB.BINARY, name="bundle_z")
        for bundle in placement_bundles:
            bid = bundle["id"]
            kind = bundle["type"]
            source_anchor = tuple(bundle["source_anchor"])
            if source_anchor not in anchors[kind]:
                raise ValueError(
                    "reference bundle %s has illegal source anchor %s" %
                    (bid, source_anchor))
            for pa in anchors[kind]:
                bundle_z[(bid, pa)].Start = int(pa == source_anchor)
            m.addConstr(
                gp.quicksum(bundle_z[(bid, pa)] for pa in anchors[kind]) == 1,
                name="BUNDLE_once_%d" % bid)

        for k in K:
            typed = [bundle for bundle in placement_bundles
                     if bundle["type"] == k]
            for pa in anchors[k]:
                m.addConstr(
                    x[(k, pa)] == gp.quicksum(
                        bundle_z[(bundle["id"], pa)] for bundle in typed),
                    name="BUNDLE_type_%s_%d" % (k, idx[pa]))

        task_bundle = {}
        for bundle in placement_bundles:
            for v in bundle["tasks"]:
                task_bundle[v] = bundle
        for v in V:
            bundle = task_bundle[v]
            bid, fixed_kind = bundle["id"], bundle["type"]
            for k in tasks[v]["allowed"]:
                for pa in anchors[k]:
                    if k == fixed_kind:
                        m.addConstr(
                            yk[(v, k, pa)] == bundle_z[(bid, pa)],
                            name="BUNDLE_task_%s_%s_%d" % (v, k, idx[pa]))
                    else:
                        m.addConstr(
                            yk[(v, k, pa)] == 0,
                            name="BUNDLE_task_zero_%s_%s_%d" %
                                 (v, k, idx[pa]))

    # ---- Optional audit-only integer-layout fixing --------------------------
    # This hook is intentionally inactive in normal runs.  It lets an external
    # verifier feed a complete serialized layout back into the *same* Gurobi
    # model and minimize only its continuous timing variables.  That is a
    # stronger cross-check than duplicating C1--C13 in an independent scorer.
    fixed_solution = solver_opts.get("fixed_solution")
    warm_start_solution = solver_opts.get("warm_start_solution")
    audit_solution = (fixed_solution if fixed_solution is not None
                      else warm_start_solution)
    if audit_solution is not None:
        fixed_site_types = {}
        for site in audit_solution.get("sites", []):
            pa = (int(site["l"]), int(site["i"]), int(site["j"]))
            if pa in fixed_site_types:
                raise ValueError("audit solution has duplicate anchor %s" %
                                 (pa,))
            fixed_site_types[pa] = site["type"]
        fixed_task_site = {
            v: tuple(pa) for v, pa in
            audit_solution.get("task_site", {}).items()
        }
        if set(fixed_task_site) != set(V):
            raise ValueError("audit solution task set does not match model")
        for k in K:
            for pa in anchors[k]:
                value = 1 if fixed_site_types.get(pa) == k else 0
                if fixed_solution is not None:
                    m.addConstr(x[(k, pa)] == value,
                                name="AUDIT_fix_x_%s_%d" % (k, idx[pa]))
                else:
                    x[(k, pa)].Start = value
        for v in V:
            if fixed_task_site[v] not in idx:
                raise ValueError("fixed task %s has invalid site %s" %
                                 (v, fixed_task_site[v]))
            for pa in P:
                value = 1 if fixed_task_site[v] == pa else 0
                if fixed_solution is not None:
                    m.addConstr(y[v, pa] == value,
                                name="AUDIT_fix_y_%s_%d" % (v, idx[pa]))
                else:
                    y[v, pa].Start = value
        if warm_start_solution is not None:
            for (v, k, pa), var in yk.items():
                var.Start = int(fixed_task_site[v] == pa and
                                fixed_site_types.get(pa) == k)

    # ---- C5: memory capacity ------------------------------------------------
    for pa in P:
        cap = gp.quicksum(lib[k]["mem"] * x[(k, pa)]
                          for k in K if (k, pa) in x)
        m.addConstr(gp.quicksum(tasks[v]["mem_mb"] * y[v, pa] for v in V)
                    <= cap, name="C5_%d" % idx[pa])

    # ---- C6: configurable resident-expert capacity --------------------------
    # ``expert_slots`` defaults to one for the evaluated one-worker-per-expert
    # setup, but larger values permit multiple expert bodies to share a die.
    # C5 still accounts for their aggregate state memory; the conditional
    # non-overlap constraints below serialize incomparable co-resident tasks.
    for pa in P:
        et = [y[v, pa] for v in V if v in expert_tasks]
        if et:
            capacity = gp.quicksum(
                expert_slots[k] * x[(k, pa)]
                for k in K if pa in anchors[k])
            m.addConstr(gp.quicksum(et) <= capacity,
                        name="C6_%d" % idx[pa])

    # ---- C6b: conditional non-overlap for co-resident tasks ----------------
    # A single site-id disjunction per incomparable task pair avoids the
    # O(|V|^2|P|) o[u,v,p] construction.  The four states mean u is on a lower
    # site, u is on a higher site, same site with u first, or same site with v
    # first.  Only the same-site states activate temporal ordering.
    successors = {v: set() for v in V}
    for edge in edges:
        successors[edge["src"]].add(edge["dst"])
    reachable = {}
    for source in V:
        seen = set()
        stack = list(successors[source])
        while stack:
            target = stack.pop()
            if target in seen:
                continue
            seen.add(target)
            stack.extend(successors[target] - seen)
        reachable[source] = seen

    site_expr = {
        v: gp.quicksum(idx[p] * y[v, p] for p in P) for v in V}
    order_states = {}
    site_count = len(P)
    for u_index, u in enumerate(V):
        for v in V[u_index + 1:]:
            common_types = set(tasks[u]["allowed"]) & set(tasks[v]["allowed"])
            if not common_types:
                continue
            if u in reachable[v] or v in reachable[u]:
                continue
            if u in expert_tasks and v in expert_tasks and not any(
                    expert_slots[k] >= 2 for k in common_types):
                continue
            lower = m.addVar(vtype=GRB.BINARY,
                             name="C6b_lower_%s_%s" % (u, v))
            higher = m.addVar(vtype=GRB.BINARY,
                              name="C6b_higher_%s_%s" % (u, v))
            u_first = m.addVar(vtype=GRB.BINARY,
                               name="C6b_ufirst_%s_%s" % (u, v))
            v_first = m.addVar(vtype=GRB.BINARY,
                               name="C6b_vfirst_%s_%s" % (u, v))
            order_states[(u, v)] = (lower, higher, u_first, v_first)
            difference = site_expr[u] - site_expr[v]
            m.addConstr(lower + higher + u_first + v_first == 1,
                        name="C6b_state_%s_%s" % (u, v))
            m.addConstr(difference <= -1 + site_count * (1 - lower),
                        name="C6b_lower_if_%s_%s" % (u, v))
            m.addConstr(difference >= 1 - site_count * (1 - higher),
                        name="C6b_higher_if_%s_%s" % (u, v))
            m.addConstr(
                difference >= -(site_count - 1) * (lower + higher),
                name="C6b_same_lo_%s_%s" % (u, v))
            m.addConstr(
                difference <= (site_count - 1) * (lower + higher),
                name="C6b_same_hi_%s_%s" % (u, v))
            m.addConstr(s[v] >= f[u] - time_horizon * (1 - u_first),
                        name="C6b_sched_uv_%s_%s" % (u, v))
            m.addConstr(s[u] >= f[v] - time_horizon * (1 - v_first),
                        name="C6b_sched_vu_%s_%s" % (u, v))

    # ---- C7: area (total + per-layer) ---------------------------------------
    m.addConstr(gp.quicksum(lib[k]["area"] * x[(k, pa)]
                            for k in K for pa in anchors[k]) <= budgets["a_max"],
                name="C7_total")
    a_layer = budgets.get("a_max_layer")
    if a_layer:
        for l in range(L):
            m.addConstr(gp.quicksum(lib[k]["area"] * x[(k, pa)]
                                    for k in K for pa in anchors[k]
                                    if pa[0] == l) <= a_layer,
                        name="C7_layer%d" % l)

    # ---- C8: package power (TDP) --------------------------------------------
    m.addConstr(gp.quicksum(lib[k]["p_peak"] * x[(k, pa)]
                            for k in K for pa in anchors[k]) <= budgets["p_max"],
                name="C8")

    # ---- C10: separable rectilinear distance --------------------------------
    def coord_expr(v, axis):
        # axis 0=l, 1=i, 2=j ; linear in y (one-hot)
        return gp.quicksum(p[axis] * y[v, p] for p in P)

    edge_remote = {}
    edge_location_states = {}
    for n, e in enumerate(edges):
        u, w_ = e["src"], e["dst"]
        for dvar, ax in ((dL, 0), (dI, 1), (dJ, 2)):
            m.addConstr(dvar[n] >= coord_expr(u, ax) - coord_expr(w_, ax),
                        name="C10p_%d_%d" % (n, ax))
            m.addConstr(dvar[n] >= coord_expr(w_, ax) - coord_expr(u, ax),
                        name="C10m_%d_%d" % (n, ax))

        # Gate fixed startup/serialization latency when both endpoints share
        # one physical chiplet.  Three exclusive site-id states make the
        # remote indicator exact rather than relying on a floating distance.
        lower = m.addVar(vtype=GRB.BINARY, name="C10edge_lower_%d" % n)
        same = m.addVar(vtype=GRB.BINARY, name="C10edge_same_%d" % n)
        higher = m.addVar(vtype=GRB.BINARY, name="C10edge_higher_%d" % n)
        edge_location_states[n] = (lower, same, higher)
        difference = site_expr[u] - site_expr[w_]
        m.addConstr(lower + same + higher == 1,
                    name="C10edge_state_%d" % n)
        m.addConstr(difference <= -1 + site_count * (1 - lower),
                    name="C10edge_lower_if_%d" % n)
        m.addConstr(difference >= 1 - site_count * (1 - higher),
                    name="C10edge_higher_if_%d" % n)
        m.addConstr(difference >= -(site_count - 1) * (lower + higher),
                    name="C10edge_same_lo_%d" % n)
        m.addConstr(difference <= (site_count - 1) * (lower + higher),
                    name="C10edge_same_hi_%d" % n)
        edge_remote[n] = lower + higher

    # ---- C11: timing / precedence -------------------------------------------
    tau_h = ic["tau_h"]
    tau_v = ic["tau_v"]
    t0 = ic["t0"]
    beta = ic["beta"]

    def dur_expr(v):
        # sum over (k,pa) of cyc[v][k] * yk[v,k,pa]
        return gp.quicksum(tasks[v]["cyc"][k] * yk[(v, k, pa)]
                           for k in tasks[v]["allowed"]
                           for pa in anchors[k] if (v, k, pa) in yk)

    for v in V:
        m.addConstr(f[v] >= s[v] + dur_expr(v), name="C11dur_%s" % v)
    for n, e in enumerate(edges):
        u, w_ = e["src"], e["dst"]
        vol = e["vol"]
        m.addConstr(s[w_] >= f[u] +
                    (t0 + vol / beta) * edge_remote[n] +
                    tau_h * (dI[n] + dJ[n]) + tau_v * dL[n],
                    name="C11prec_%d" % n)

    # ---- C12: legacy task-schedule feasibility horizon ---------------------
    m.addConstrs((T_task >= f[v] for v in V), name="C12")

    # ---- C13: chiplet busy-time relaxation ----------------------------------
    for pa in P:
        busy = gp.quicksum(tasks[v]["cyc"][k] * yk[(v, k, pa)]
                           for v in V for k in tasks[v]["allowed"]
                           if (v, k, pa) in yk)
        m.addConstr(busy <= T_task, name="C13_%d" % idx[pa])

    # ---- Post-solve energy diagnostics --------------------------------------
    # These expressions are not an objective and impose no energy budget.
    eps_h = ic["eps_h"]
    eps_v = ic["eps_v"]
    E_comm = 8 * gp.quicksum(e["vol"] * (eps_h * (dI[n] + dJ[n]) + eps_v * dL[n])
                             for n, e in enumerate(edges))
    E_comp = gp.quicksum(tasks[v]["eta"][k] * yk[(v, k, pa)]
                         for v in V for k in tasks[v]["allowed"]
                         for pa in anchors[k] if (v, k, pa) in yk)
    # ---- C10b: explicit packet-flow routing and directed-link capacity -------
    # Traffic is splittable at packet granularity.  Aggregating commodities by
    # logical source is exact for total link load and dynamic hop energy, while
    # avoiding one route variable for every source/destination pair.
    #
    # The LEGOSim worker executes two *sequential and oppositely directed*
    # phases: dispatch sends source-token payloads to experts and combine sends
    # those payloads back.  Keeping the phases separate is essential for a
    # full-duplex Mesh: doubling dispatch bytes in the original direction gives
    # the wrong directed-link bottleneck.  C10capacity below deliberately keeps
    # the earlier whole-application capacity horizon T.  The phase-local
    # completion variables supply the task-DAG fallback and phase-level reports;
    # the sole objective remains T.
    dispatch_matrix = None
    dispatch_token_bytes = None
    dispatch_token_bytes_per_phase = None
    dispatch_expert_tasks = None
    noc_traffic_kind = solver_opts.get(
        "noc_traffic", "deterministic-dispatch")
    traffic_by_phase = {}
    route_phases = []
    if noc_traffic_kind == "deterministic-dispatch":
        groups = int(meta.get("groups", len(expert_tasks)))
        expert_by_group = {}
        for task in V:
            match = re.match(r"^e\((\d+)\)$", task)
            if match:
                group = int(match.group(1))
                if group in expert_by_group:
                    raise ValueError("expert group %d has multiple tasks" % group)
                expert_by_group[group] = task
        missing_groups = [group for group in range(groups)
                          if group not in expert_by_group]
        if missing_groups:
            raise ValueError(
                "deterministic-dispatch proxy requires e(g) for every group; "
                "missing %s (use --noc-traffic task-dag explicitly to fall "
                "back)" % missing_groups)
        dispatch_matrix = deterministic_dispatch(
            params, solver_opts.get("dispatch_tokens_per_source", 32))
        dispatch_hidden = int(solver_opts.get("dispatch_hidden", 128))
        bytes_elem = int(solver_opts.get(
            "dispatch_bytes_elem", meta.get("bytes_elem", 2)))
        if dispatch_hidden <= 0 or bytes_elem <= 0:
            raise ValueError("dispatch hidden/bytes-per-element must be positive")
        # One payload copy is sent in each direction.  ``dispatch_token_bytes``
        # retains its historical two-phase meaning for JSON compatibility;
        # flow demands use the per-phase value.
        dispatch_token_bytes_per_phase = dispatch_hidden * bytes_elem
        dispatch_token_bytes = 2 * dispatch_token_bytes_per_phase
        dispatch_expert_tasks = [expert_by_group[group]
                                 for group in range(groups)]
        route_phases = ["dispatch", "combine"]
        traffic_by_phase = {phase: {} for phase in route_phases}

        def add_phase_demand(phase, source_task, target_task, volume):
            destinations = traffic_by_phase[phase].setdefault(source_task, {})
            destinations[target_task] = (
                destinations.get(target_task, 0.0) + float(volume))

        for source in range(groups):
            source_task = dispatch_expert_tasks[source]
            for target in range(groups):
                if source == target or dispatch_matrix[source][target] <= 0:
                    continue
                target_task = dispatch_expert_tasks[target]
                volume = (dispatch_matrix[source][target] *
                          dispatch_token_bytes_per_phase)
                add_phase_demand("dispatch", source_task, target_task, volume)
                add_phase_demand("combine", target_task, source_task, volume)
    elif noc_traffic_kind == "task-dag":
        noc_traffic_kind = "task-dag"
        route_phases = ["task-dag"]
        traffic_by_phase = {"task-dag": {}}
        for edge in edges:
            destinations = traffic_by_phase["task-dag"].setdefault(
                edge["src"], {})
            destinations[edge["dst"]] = (
                destinations.get(edge["dst"], 0.0) + float(edge["vol"]))
    else:
        raise ValueError("noc_traffic must be deterministic-dispatch or task-dag")

    flow_scale_bytes = 1024.0
    route_sources = {
        phase: sorted(traffic_by_phase[phase]) for phase in route_phases}
    route_flow = m.addVars(
        [(phase, source, link_id) for phase in route_phases
         for source in route_sources[phase]
         for link_id in range(len(directed_links))],
        lb=0.0, name="route_kib")
    scaled_demands = {}
    for phase in route_phases:
        scaled_demands[phase] = {}
        for source in route_sources[phase]:
            scaled_demands[phase][source] = {
                target: float(volume) / flow_scale_bytes
                for target, volume in
                traffic_by_phase[phase][source].items()}
            total = sum(scaled_demands[phase][source].values())
            for node in P:
                balance = (total * y[source, node] - gp.quicksum(
                    volume * y[target, node]
                    for target, volume in
                    scaled_demands[phase][source].items()))
                m.addConstr(
                    gp.quicksum(route_flow[(phase, source, link_id)]
                                for link_id in outgoing[node]) -
                    gp.quicksum(route_flow[(phase, source, link_id)]
                                for link_id in incoming[node]) == balance,
                    name="C10flow_%s_%s_%d" %
                         (phase, source, idx[node]))

    link_load = {}
    aggregate_link_load = {}
    for phase in route_phases:
        for link_id in range(len(directed_links)):
            link_load[(phase, link_id)] = gp.quicksum(
                route_flow[(phase, source, link_id)]
                for source in route_sources[phase])
    for link_id in range(len(directed_links)):
        aggregate_link_load[link_id] = gp.quicksum(
            link_load[(phase, link_id)] for phase in route_phases)
        m.addConstr(
            aggregate_link_load[link_id] <=
            (beta / flow_scale_bytes) * T,
            name="C10capacity_%d" % link_id)

    # A phase-completion lower bound is the maximum of endpoint serialization
    # and the time for its most-loaded directed link to drain.  LEGOSim's
    # blocking send/receive wrappers add
    # 1000 cycles per non-local message.  For the legacy task-DAG option, t0 is
    # the matching analytical startup term.  The endpoint component is the
    # larger of every source's send sequence and every destination's receive
    # sequence; the link component is a standard bottleneck lower bound.
    dispatch_sync_overhead = float(solver_opts.get(
        "dispatch_sync_overhead_cycles", 1000.0))
    if dispatch_sync_overhead < 0.0:
        raise ValueError("dispatch sync overhead must be nonnegative")
    phase_link_cycles = m.addVars(route_phases, lb=0.0,
                                  name="noc_phase_link_cycles")
    phase_endpoint_cycles = m.addVars(route_phases, lb=0.0,
                                      name="noc_phase_endpoint_cycles")
    phase_completion_cycles = m.addVars(route_phases, lb=0.0,
                                        name="noc_phase_completion_cycles")
    message_axis_distance = {}
    message_pair_number = {}

    def message_distances(source, target):
        """Exact-at-optimum Manhattan components for a protocol message."""
        pair = (source, target)
        if pair not in message_pair_number:
            message_pair_number[pair] = len(message_pair_number)
            number = message_pair_number[pair]
            for axis in range(3):
                distance = m.addVar(
                    lb=0.0, name="protocol_dist_%d_%d" % (number, axis))
                difference = (coord_expr(source, axis) -
                              coord_expr(target, axis))
                m.addConstr(distance >= difference,
                            name="protocol_dist_p_%d_%d" % (number, axis))
                m.addConstr(distance >= -difference,
                            name="protocol_dist_m_%d_%d" % (number, axis))
                message_axis_distance[(pair, axis)] = distance
        return tuple(message_axis_distance[(pair, axis)]
                     for axis in range(3))

    phase_endpoint_bounds = {}
    for phase in route_phases:
        for link_id in range(len(directed_links)):
            m.addConstr(
                phase_link_cycles[phase] >=
                (flow_scale_bytes / beta) * link_load[(phase, link_id)],
                name="C10phase_link_%s_%d" % (phase, link_id))

        startup = (dispatch_sync_overhead
                   if noc_traffic_kind == "deterministic-dispatch" else t0)
        source_bounds = {}
        destination_terms = {}
        for source in route_sources[phase]:
            destinations = traffic_by_phase[phase][source]
            source_terms = []
            shortest_path_kib_hops = gp.LinExpr(0.0)
            for target, volume in destinations.items():
                if float(volume) <= 0.0:
                    continue
                d_l, d_i, d_j = message_distances(source, target)
                shortest_path_kib_hops += (
                    float(volume) / flow_scale_bytes) * (d_l + d_i + d_j)
                message_cycles = (
                    startup + float(volume) / beta +
                    tau_h * (d_i + d_j) + tau_v * d_l)
                source_terms.append(message_cycles)
                destination_terms.setdefault(target, []).append(
                    message_cycles)
            # Flow conservation alone permits cycles and non-minimal detours.
            # Equality to the exact Manhattan demand distance restricts the
            # splittable relaxation to shortest paths while retaining scalable
            # source aggregation.
            m.addConstr(
                gp.quicksum(route_flow[(phase, source, link_id)]
                            for link_id in range(len(directed_links))) ==
                shortest_path_kib_hops,
                name="C10phase_shortest_%s_%s" % (phase, source))
            source_bounds[source] = gp.quicksum(source_terms)
            m.addConstr(
                phase_endpoint_cycles[phase] >= source_bounds[source],
                name="C10phase_src_%s_%s" % (phase, source))
        destination_bounds = {}
        for target in sorted(destination_terms):
            destination_bounds[target] = gp.quicksum(
                destination_terms[target])
            m.addConstr(
                phase_endpoint_cycles[phase] >= destination_bounds[target],
                name="C10phase_dst_%s_%s" % (phase, target))
        phase_endpoint_bounds[phase] = {
            "source": source_bounds,
            "destination": destination_bounds,
        }
        m.addConstr(
            phase_completion_cycles[phase] >= phase_link_cycles[phase],
            name="C10phase_complete_link_%s" % phase)
        m.addConstr(
            phase_completion_cycles[phase] >= phase_endpoint_cycles[phase],
            name="C10phase_complete_endpoint_%s" % phase)

    C_noc_completion = gp.quicksum(
        phase_completion_cycles[phase] for phase in route_phases)

    # The bare-metal router sends one dynamically-sized row/column header to
    # every worker and later receives one 32-bit completion value from every
    # worker.  Both loops are sequential in moe_router_cpu.cpp, so their full
    # Manhattan latency belongs to the protocol critical path.  Exactly one
    # freely placed CPU is already the evaluated architecture assumption.
    C_cpu_init = gp.LinExpr(0.0)
    C_cpu_done = gp.LinExpr(0.0)
    B_cpu_control_proxy = gp.LinExpr(0.0)
    E_cpu_control_proxy = gp.LinExpr(0.0)
    cpu_control_bytes = 0.0
    cpu_control_distances = {}
    if noc_traffic_kind == "deterministic-dispatch":
        if "CPU" not in K:
            raise ValueError(
                "deterministic dispatch requires one explicit CPU router")
        cpu_instances = gp.quicksum(x[("CPU", pa)]
                                    for pa in anchors["CPU"])
        m.addConstr(cpu_instances == 1,
                    name="C10protocol_exactly_one_cpu")
        cpu_coordinates = {
            axis: gp.quicksum(pa[axis] * x[("CPU", pa)]
                              for pa in anchors["CPU"])
            for axis in range(3)}
        init_bytes = 4.0 * (4 + 2 * len(dispatch_expert_tasks))
        done_bytes = 4.0
        cpu_control_bytes = len(dispatch_expert_tasks) * (
            init_bytes + done_bytes)
        for number, expert in enumerate(dispatch_expert_tasks):
            distances = []
            for axis in range(3):
                distance = m.addVar(
                    lb=0.0,
                    name="protocol_cpu_dist_%d_%d" % (number, axis))
                difference = (cpu_coordinates[axis] -
                              coord_expr(expert, axis))
                m.addConstr(
                    distance >= difference,
                    name="protocol_cpu_dist_p_%d_%d" % (number, axis))
                m.addConstr(
                    distance >= -difference,
                    name="protocol_cpu_dist_m_%d_%d" % (number, axis))
                distances.append(distance)
            cpu_control_distances[expert] = tuple(distances)
            d_l, d_i, d_j = distances
            hop_cycles = tau_h * (d_i + d_j) + tau_v * d_l
            C_cpu_init += (dispatch_sync_overhead + init_bytes / beta +
                           hop_cycles)
            C_cpu_done += (dispatch_sync_overhead + done_bytes / beta +
                           hop_cycles)
            control_bytes = init_bytes + done_bytes
            B_cpu_control_proxy += control_bytes * (d_l + d_i + d_j)
            E_cpu_control_proxy += (
                8.0 * control_bytes *
                (eps_v * d_l + eps_h * (d_i + d_j)))

    # Exact max-plus event DAG for the deterministic blocking protocol.  Stable
    # q=1 worker IDs are logical expert IDs (enforced by codegen.py), hence the
    # nested (source,destination) order below is placement-independent.  For a
    # message a->b, the sender advances from its prior clock by the 1000-cycle
    # synchronization and source serialization; the receiver advances to the
    # maximum of its own readiness and the routed packet arrival.
    protocol_event_count = 0
    protocol_compute_profile = None
    protocol_compute_cycles = {}
    protocol_compute_start = {}
    protocol_compute_finish = {}
    protocol_worker_finish_expr = {}
    protocol_flit_bytes = None
    protocol_hop_cycles = None
    if noc_traffic_kind == "deterministic-dispatch":
        protocol_clock = {"cpu": gp.LinExpr(0.0)}
        for group in range(len(dispatch_expert_tasks)):
            protocol_clock[group] = gp.LinExpr(0.0)

        protocol_flit_bytes = float(solver_opts.get(
            "protocol_flit_bytes", 32.0))
        protocol_hop_cycles = float(solver_opts.get(
            "protocol_hop_cycles", 104.0))
        if protocol_flit_bytes <= 0.0 or protocol_hop_cycles < 0.0:
            raise ValueError("invalid protocol flit/hop timing")

        def add_protocol_message(source_process, target_process, volume_bytes,
                                 distances, label):
            """Append one fixed-order blocking message to the event DAG."""
            number = len(protocol_event_vars)
            flits = int(math.ceil(float(volume_bytes) /
                                  protocol_flit_bytes)) + 1
            serialization = float(flits - 1)
            d_l, d_i, d_j = distances
            sender_ready = protocol_clock[source_process]
            receiver_ready = protocol_clock[target_process]
            write_time = sender_ready + dispatch_sync_overhead
            sender_finish = write_time + serialization
            receiver_finish = m.addVar(
                lb=0.0, name="protocol_event_%d_%s" % (number, label))
            m.addConstr(
                receiver_finish >= write_time + serialization +
                protocol_hop_cycles * (d_l + d_i + d_j),
                name="protocol_arrival_%d" % number)
            m.addConstr(
                receiver_finish >= receiver_ready + dispatch_sync_overhead,
                name="protocol_ready_%d" % number)
            protocol_clock[source_process] = sender_finish
            protocol_clock[target_process] = receiver_finish
            protocol_event_vars.append(receiver_finish)

        protocol_event_vars = []
        # CPU initialization is ordered by stable logical worker ID.
        for group, expert in enumerate(dispatch_expert_tasks):
            add_protocol_message(
                "cpu", group, init_bytes, cpu_control_distances[expert],
                "init_%d" % group)

        # Dispatch: source s sends q[s,d] hidden-state payloads to expert d.
        for source in range(len(dispatch_expert_tasks)):
            for target in range(len(dispatch_expert_tasks)):
                assignments = dispatch_matrix[source][target]
                if source == target or assignments <= 0:
                    continue
                distances = message_distances(
                    dispatch_expert_tasks[source],
                    dispatch_expert_tasks[target])
                add_protocol_message(
                    source, target,
                    assignments * dispatch_token_bytes_per_phase,
                    distances, "dispatch_%d_%d" % (source, target))

        # Profiled expert compute is inserted independently at each worker's
        # post-dispatch clock, so unrelated workers may overlap exactly as in
        # the process-level replay.  E8 uses measured GPGPU-Sim values; other
        # scales use token-count regressions and are explicitly tagged.
        column_tokens = [
            sum(dispatch_matrix[source][target]
                for source in range(len(dispatch_expert_tasks)))
            for target in range(len(dispatch_expert_tasks))]
        exact_s = [42450.0, 29390.0, 23726.0, 20619.0,
                   18338.0, 17002.0, 16488.0, 16120.0]
        exact_l = [4663.0, 4670.0, 4615.0, 4598.0,
                   4573.0, 4570.0, 4559.0, 4557.0]
        exact_e8 = (
            len(dispatch_expert_tasks) == 8 and
            column_tokens == [144, 89, 67, 54, 47, 41, 37, 33] and
            int(solver_opts.get("dispatch_hidden", 128)) == 128)
        protocol_compute_profile = (
            "measured-e8-h128-i64-repeat1"
            if exact_e8 else
            "token-linear-fallback:GPU-S=7431.145+243.523*n;"
            "GPU-L=4575.206+0.6926*n")
        for group, expert in enumerate(dispatch_expert_tasks):
            by_type = {}
            compute_terms = []
            for kind in tasks[expert]["allowed"]:
                if kind == "GPU-S":
                    cycles = (exact_s[group] if exact_e8 else
                              7431.145 + 243.523 * column_tokens[group])
                elif kind == "GPU-L":
                    cycles = (exact_l[group] if exact_e8 else
                              4575.206 + 0.6926 * column_tokens[group])
                elif "proto_cycles" in lib[kind]:
                    # Optional replay-scale token-linear profile a + b*n for
                    # library types beyond the profiled GPU-L/S pair.
                    intercept, slope = lib[kind]["proto_cycles"]
                    cycles = (float(intercept) +
                              float(slope) * column_tokens[group])
                else:
                    cycles = float(tasks[expert]["cyc"][kind])
                by_type[kind] = float(cycles)
                compute_terms.extend(
                    cycles * yk[(expert, kind, pa)]
                    for pa in anchors[kind]
                    if (expert, kind, pa) in yk)
            protocol_compute_cycles[expert] = by_type
            compute_start = m.addVar(
                lb=0.0, name="protocol_compute_start_%d" % group)
            compute_finish = m.addVar(
                lb=0.0, name="protocol_compute_finish_%d" % group)
            m.addConstr(
                compute_start >= protocol_clock[group],
                name="protocol_compute_ready_%d" % group)
            m.addConstr(
                compute_finish == compute_start + gp.quicksum(compute_terms),
                name="protocol_compute_duration_%d" % group)
            protocol_compute_start[expert] = compute_start
            protocol_compute_finish[expert] = compute_finish
            protocol_clock[group] = compute_finish

        # When a library permits several experts to share one chiplet, their
        # replay-profiled kernels must not overlap.  Reuse the exact C6b site/
        # order states so this protocol-scale schedule and the legacy C11 task
        # schedule select the same non-preemptive order.  Indicators avoid a
        # second, protocol-specific big-M horizon.
        for (first, second), states in order_states.items():
            if (first not in protocol_compute_start or
                    second not in protocol_compute_start):
                continue
            u_first, v_first = states[2], states[3]
            m.addGenConstrIndicator(
                u_first, True,
                protocol_compute_start[second] >=
                protocol_compute_finish[first],
                name="protocol_compute_order_uv_%s_%s" % (first, second))
            m.addGenConstrIndicator(
                v_first, True,
                protocol_compute_start[first] >=
                protocol_compute_finish[second],
                name="protocol_compute_order_vu_%s_%s" % (first, second))

        # Combine traverses the same logical (s,d) order in reverse direction.
        for source in range(len(dispatch_expert_tasks)):
            for target in range(len(dispatch_expert_tasks)):
                assignments = dispatch_matrix[source][target]
                if source == target or assignments <= 0:
                    continue
                distances = message_distances(
                    dispatch_expert_tasks[source],
                    dispatch_expert_tasks[target])
                add_protocol_message(
                    target, source,
                    assignments * dispatch_token_bytes_per_phase,
                    distances, "combine_%d_%d" % (source, target))

        # Workers issue summaries independently; CPU consumes them in stable
        # worker-ID order, matching moe_router_cpu.cpp.
        for group, expert in enumerate(dispatch_expert_tasks):
            add_protocol_message(
                group, "cpu", done_bytes, cpu_control_distances[expert],
                "done_%d" % group)
        protocol_event_count = len(protocol_event_vars)
        protocol_worker_finish_expr = {
            dispatch_expert_tasks[group]: protocol_clock[group]
            for group in range(len(dispatch_expert_tasks))}
        C_protocol_proxy = protocol_clock["cpu"]
    else:
        # ``task-dag`` is a compatibility fallback, not the LEGOSim replay
        # protocol.  Its public T is the legacy task schedule followed by the
        # two phase-level NoC drain/endpoint lower bounds.  Keeping this branch
        # explicit avoids a self-reference through the public objective T.
        C_protocol_proxy = T_task + C_noc_completion

    # The replay-oriented fixed-order event DAG is the performance model used
    # by the deterministic-dispatch experiments.  Energy remains diagnostic.
    E_noc_proxy = E_cpu_control_proxy + 8.0 * flow_scale_bytes * gp.quicksum(
        (eps_v if link_axis[link_id] == 0 else eps_h) *
        route_flow[(phase, source, link_id)]
        for phase in route_phases for source in route_sources[phase]
        for link_id in range(len(directed_links)))
    routed_volume = cpu_control_bytes + sum(
        sum(destinations.values())
        for phase in route_phases
        for destinations in traffic_by_phase[phase].values())
    e_noc_proxy_ref = max(
        8.0 * routed_volume *
        (eps_h * ((X - 1) + (Y - 1)) + eps_v * (L - 1)),
        1.0e-30)
    noc_objective_kind = (
        "phase-separated source-aggregated splittable packet flow with "
        "directed-link capacity, endpoint serialization, and dynamic hop "
        "energy")

    # CPU placement is optimized like every other legal chiplet anchor; no
    # package-corner condition is part of the active formulation.
    cpu_anchor_policy = "free"

    # ---- Objective ----------------------------------------------------------
    # ILP-Perf has exactly one joint MILP objective.  T is the epigraph of the
    # fixed-order replay protocol for deterministic dispatch (or the documented
    # task-DAG compatibility bound above).  The max-plus event recurrence is
    # linear because its order is fixed: every receiver clock has two affine
    # lower bounds, and minimizing the final CPU clock makes the recurrence
    # exact.  C11--C13 remain hard schedule-feasibility constraints; protocol-
    # scale compute variables additionally serialize co-resident experts.
    # A unit cycle coefficient avoids the numerical false-zero-gap behavior
    # observed when 1e7--1e8 cycle values were scaled down to 1e-7 objective
    # coefficients.  There is deliberately no resource, energy, or placement
    # tie-break and no alternate internal objective mode.
    certified_T_ref = 1.0
    m.addConstr(T >= C_protocol_proxy, name="C12_protocol_completion")
    primary_performance_expr = T
    m.setObjective(primary_performance_expr, GRB.MINIMIZE)
    m.Params.MIPGap = solver_opts.get("mipgap", 0.005)
    requested_total_time_limit = float(solver_opts.get("time_limit", 3600))
    if requested_total_time_limit <= 0.0:
        raise ValueError("time_limit must be positive")
    m.Params.TimeLimit = requested_total_time_limit
    m.Params.Symmetry = 2 if solver_opts.get("symmetry_break", True) else 0
    # Optional diagnostics for numerical/cut audits.  Normal experiment runs
    # leave every one of these unset and retain Gurobi defaults.
    for option_name in ("Cuts", "NumericFocus", "Presolve", "ScaleFlag",
                        "Threads"):
        option_key = option_name.lower()
        if solver_opts.get(option_key) is not None:
            setattr(m.Params, option_name, solver_opts[option_key])
    # The sole objective is still min T.  MIPFocus=1 changes only Gurobi's
    # search emphasis toward quickly improving feasible incumbents, which is
    # useful for the short large-instance limits used by the scaling sweep.
    raw_mip_focus = solver_opts.get("mipfocus", 1)
    mip_focus = int(raw_mip_focus)
    if float(raw_mip_focus) != float(mip_focus) or mip_focus not in (0, 1, 2, 3):
        raise ValueError("Gurobi MIPFocus must be one of 0, 1, 2, or 3")
    m.Params.MIPFocus = mip_focus
    if solver_opts.get("seed") is not None:
        m.Params.Seed = solver_opts["seed"]

    # ---- Deterministic primary-solve feasibility start ---------------------
    # A wider area/power budget can enlarge the symmetric BOM/type search so
    # much that Gurobi spends the whole short primary limit without producing
    # any incumbent.  Build one legal *performance-first* typed packing up
    # front: start by upgrading every expert from its least-resource compatible
    # type to its fastest compatible type, then retract the lowest-benefit
    # upgrades until area, package peak power, count/capacity, memory, and
    # footprint packing are all legal.  The result is submitted only as a MIP
    # start.  No variable is fixed and the MILP remains free to select a better
    # BOM, type assignment, grouping, schedule, or placement.
    stage0_feasible_start = {
        "attempted": False,
        "accepted_for_submission": False,
        "reason": None,
        "strategy": None,
        "selection_policy": None,
        "packing_policy": None,
        "all_gpu_s": None,
        "type_counts": None,
        "bundle_count": 0,
        "total_area_mm2": None,
        "peak_power_W": None,
        "analytical_T_start_cycles": None,
        "legacy_task_T_start_cycles": None,
        "protocol_T_start_cycles": None,
        "upgrade_candidate_count": 0,
        "requested_upgrade_count": None,
        "realized_upgrade_count": None,
        "upgrade_attempts": [],
        "expert_upgrade_order": [],
        "expert_type_assignment": None,
        "cpu_anchor": None,
        "protocol_placement_refinement": None,
        "mip_start_variable_count": 0,
        "mip_start_undefined_variable_count": None,
        "mip_start_undefined_variables": [],
        "bundles": [],
    }

    def build_stage0_feasible_start():
        evidence = dict(stage0_feasible_start)
        evidence["attempted"] = True
        if fixed_solution is not None:
            evidence["reason"] = "explicit fixed_solution already supplied"
            return evidence
        if warm_start_solution is not None or placement_bundles is not None:
            evidence["reason"] = "serialized placement warm start already supplied"
            return evidence
        if bom_counts is not None:
            evidence["reason"] = "same-BOM caller requires its own typed start"
            return evidence

        def resource_score(kind):
            spec = lib[kind]
            normalized_cost = (
                float(spec["area"]) / max(float(budgets["a_max"]), 1.0) +
                float(spec["p_peak"]) /
                max(float(budgets["p_max"]), 1.0) +
                1.0e-3 * len(footprint[kind][anchors[kind][0]]))
            return normalized_cost, len(footprint[kind][anchors[kind][0]]), kind

        task_traffic = {task: 0.0 for task in V}
        for edge in edges:
            volume = float(edge["vol"])
            task_traffic[edge["src"]] += volume
            task_traffic[edge["dst"]] += volume

        def expert_ordinal(task):
            match = re.match(r"^e\((\d+)\)$", task)
            return (int(match.group(1)) if match else len(V), task)

        ordered_experts = sorted(
            expert_tasks, key=expert_ordinal)
        expert_candidates = {}
        for task in ordered_experts:
            candidates = [
                kind for kind in tasks[task]["allowed"]
                if (expert_slots.get(kind, 0) >= 1 and anchors.get(kind) and
                    float(tasks[task]["mem_mb"]) <=
                    float(lib[kind]["mem"]))]
            if not candidates:
                evidence["reason"] = (
                    "expert %s has no feasible resident type" % task)
                return evidence
            expert_candidates[task] = list(candidates)

        def start_performance_cycles(task, kind):
            if (noc_traffic_kind == "deterministic-dispatch" and
                    task in protocol_compute_cycles and
                    kind in protocol_compute_cycles[task]):
                return float(protocol_compute_cycles[task][kind])
            return float(tasks[task]["cyc"][kind])

        # The baseline is the least-resource compatible resident type.  Each
        # upgrade goes directly to the lowest-cycle compatible type on the
        # active performance scale (the replay profile for deterministic
        # dispatch).  Incident traffic is a deterministic tie-break only.
        baseline_plan = {}
        upgrade_records = []
        for task in ordered_experts:
            candidates = expert_candidates[task]
            baseline = min(
                candidates,
                key=lambda kind: (resource_score(kind),
                                  start_performance_cycles(task, kind), kind))
            fastest = min(
                candidates,
                key=lambda kind: (start_performance_cycles(task, kind),
                                  resource_score(kind), kind))
            baseline_plan[task] = baseline
            reduction = (start_performance_cycles(task, baseline) -
                         start_performance_cycles(task, fastest))
            if fastest != baseline and reduction > 1.0e-12:
                upgrade_records.append({
                    "task": task,
                    "baseline_type": baseline,
                    "fastest_type": fastest,
                    "baseline_cycles": start_performance_cycles(
                        task, baseline),
                    "fastest_cycles": start_performance_cycles(
                        task, fastest),
                    "cycle_reduction": reduction,
                    "incident_traffic_bytes": float(task_traffic[task]),
                })
        upgrade_records.sort(key=lambda record: (
            -record["cycle_reduction"],
            -record["incident_traffic_bytes"],
            expert_ordinal(record["task"])))
        evidence["selection_policy"] = (
            "performance-first fastest-type expert upgrades; descending "
            "active-scale cycle reduction with incident-traffic tie-break; "
            "retract "
            "lowest-benefit upgrades until fully feasible")
        evidence["upgrade_candidate_count"] = len(upgrade_records)
        evidence["expert_upgrade_order"] = [dict(record)
                                             for record in upgrade_records]

        # Try the largest hot/benefit prefix first.  Because every smaller
        # prefix is also tried, tight budget tiers deterministically fall back
        # from all-fast to the most profitable legal mixed BOM instead of
        # accepting an all-small start merely because it is cheap.
        type_plans = []
        for upgrade_count in range(len(upgrade_records), -1, -1):
            plan = dict(baseline_plan)
            for record in upgrade_records[:upgrade_count]:
                plan[record["task"]] = record["fastest_type"]
            type_plans.append((
                "performance-first %d/%d fastest expert upgrades" %
                (upgrade_count, len(upgrade_records)),
                plan, upgrade_count))

        required_counts = {
            kind: int(nmin.get(kind, 0)) for kind in K}
        maximum_counts = {
            kind: (int(nmax[kind]) if kind in nmax else None)
            for kind in K}

        def assemble_bundles(label, expert_plan):
            bundles = []

            def new_bundle(kind, required=False):
                bundle = {
                    "id": len(bundles), "type": kind, "tasks": [],
                    "memory_mb": 0.0, "expert_count": 0,
                    "required_instance": bool(required)}
                bundles.append(bundle)
                return bundle

            # One bundle per expert when expert_slots=1; types with a larger
            # declared capacity are filled deterministically up to that limit.
            for task in ordered_experts:
                kind = expert_plan[task]
                memory = float(tasks[task]["mem_mb"])
                target = None
                for bundle in bundles:
                    if (bundle["type"] == kind and
                            bundle["expert_count"] < expert_slots[kind] and
                            bundle["memory_mb"] + memory <=
                            float(lib[kind]["mem"])):
                        target = bundle
                        break
                if target is None:
                    target = new_bundle(kind)
                target["tasks"].append(task)
                target["memory_mb"] += memory
                target["expert_count"] += 1

            # Materialize required chiplet counts (normally one CPU).  C4use
            # means every required instance must receive at least one task.
            for kind in K:
                present = sum(bundle["type"] == kind for bundle in bundles)
                for _ in range(max(0, required_counts[kind] - present)):
                    new_bundle(kind, required=True)

            assigned = set(ordered_experts)
            auxiliary = [task for task in V if task not in assigned]

            # Populate empty required bundles first with the least-flexible
            # compatible auxiliary task.
            for bundle in [item for item in bundles if not item["tasks"]]:
                compatible = [
                    task for task in auxiliary if task not in assigned and
                    bundle["type"] in tasks[task]["allowed"] and
                    float(tasks[task]["mem_mb"]) <=
                    float(lib[bundle["type"]]["mem"])]
                compatible.sort(key=lambda task: (
                    len(tasks[task]["allowed"]), V.index(task)))
                if not compatible:
                    return None, (
                        "required %s instance cannot host an auxiliary task" %
                        bundle["type"])
                task = compatible[0]
                bundle["tasks"].append(task)
                bundle["memory_mb"] += float(tasks[task]["mem_mb"])
                assigned.add(task)

            # Co-locate remaining non-expert tasks whenever capacity and type
            # compatibility permit; otherwise create the cheapest legal die.
            for task in auxiliary:
                if task in assigned:
                    continue
                memory = float(tasks[task]["mem_mb"])
                compatible_bundles = [
                    bundle for bundle in bundles
                    if (bundle["type"] in tasks[task]["allowed"] and
                        bundle["memory_mb"] + memory <=
                        float(lib[bundle["type"]]["mem"]))]
                compatible_bundles.sort(key=lambda bundle: (
                    0 if bundle["type"] == "CPU" else 1,
                    bundle["id"]))
                if compatible_bundles:
                    target = compatible_bundles[0]
                else:
                    candidate_types = [
                        kind for kind in tasks[task]["allowed"]
                        if anchors.get(kind) and memory <=
                        float(lib[kind]["mem"])]
                    candidate_types.sort(key=resource_score)
                    if not candidate_types:
                        return None, (
                            "auxiliary task %s has no feasible type" % task)
                    target = new_bundle(candidate_types[0])
                target["tasks"].append(task)
                target["memory_mb"] += memory
                assigned.add(task)

            counts = {kind: sum(bundle["type"] == kind
                                for bundle in bundles) for kind in K}
            for kind in K:
                if counts[kind] < required_counts[kind]:
                    return None, "required count for %s not met" % kind
                if (maximum_counts[kind] is not None and
                        counts[kind] > maximum_counts[kind]):
                    return None, "maximum count for %s exceeded" % kind
            area_value = sum(float(lib[bundle["type"]]["area"])
                             for bundle in bundles)
            power_value = sum(float(lib[bundle["type"]]["p_peak"])
                              for bundle in bundles)
            if area_value > float(budgets["a_max"]) + 1.0e-9:
                return None, "area budget exceeded"
            if power_value > float(budgets["p_max"]) + 1.0e-9:
                return None, "power budget exceeded"

            a_layer_limit = budgets.get("a_max_layer")

            package_center = ((L - 1) / 2.0, (X - 1) / 2.0,
                              (Y - 1) / 2.0)

            def physical_center(kind, anchor):
                cells = footprint[kind][anchor]
                denominator = float(len(cells))
                return tuple(sum(cell[axis] for cell in cells) / denominator
                             for axis in range(3))

            def center_distance(kind, anchor, reference):
                center = physical_center(kind, anchor)
                return sum(abs(center[axis] - reference[axis])
                           for axis in range(3))

            def bundle_priority(bundle):
                # Potential cycle range is independent of the selected type and
                # therefore keeps both upgraded and residual slow hot experts
                # close to the CPU.  Footprint size remains the first key so a
                # large die cannot be fragmented out by earlier 1x1 bundles.
                potential_gain = 0.0
                traffic = 0.0
                for task in bundle["tasks"]:
                    traffic += task_traffic[task]
                    values = [start_performance_cycles(task, kind)
                              for kind in tasks[task]["allowed"]]
                    potential_gain += max(values) - min(values)
                bundle["performance_priority_cycle_gain"] = potential_gain
                bundle["traffic_priority_bytes"] = traffic
                footprint_size = len(footprint[bundle["type"]][
                    anchors[bundle["type"]][0]])
                return (-footprint_size, -potential_gain, -traffic,
                        bundle["id"])

            def anchor_is_legal(bundle, anchor, occupied, layer_area):
                kind = bundle["type"]
                if occupied & footprint[kind][anchor]:
                    return False
                if (a_layer_limit is not None and
                        layer_area[anchor[0]] + float(lib[kind]["area"]) >
                        float(a_layer_limit) + 1.0e-9):
                    return False
                return True

            def place(bundle, anchor, occupied, layer_area):
                kind = bundle["type"]
                bundle["anchor"] = anchor
                occupied.update(footprint[kind][anchor])
                layer_area[anchor[0]] += float(lib[kind]["area"])

            def performance_first_pack(primary_cpu_anchor=None):
                for bundle in bundles:
                    bundle.pop("anchor", None)
                occupied = set()
                layer_area = {layer: 0.0 for layer in range(L)}
                cpu_bundles = sorted(
                    [bundle for bundle in bundles
                     if bundle["type"] == "CPU"],
                    key=lambda bundle: bundle["id"])
                if cpu_bundles:
                    primary = cpu_bundles[0]
                    if (primary_cpu_anchor is None or
                            not anchor_is_legal(primary, primary_cpu_anchor,
                                                occupied, layer_area)):
                        return False
                    place(primary, primary_cpu_anchor, occupied, layer_area)
                    cpu_reference = physical_center(
                        primary["type"], primary_cpu_anchor)
                    for bundle in cpu_bundles[1:]:
                        choices = sorted(
                            anchors[bundle["type"]],
                            key=lambda anchor: (
                                center_distance(bundle["type"], anchor,
                                                package_center),
                                idx[anchor]))
                        chosen = next((anchor for anchor in choices
                                       if anchor_is_legal(
                                           bundle, anchor, occupied,
                                           layer_area)), None)
                        if chosen is None:
                            return False
                        place(bundle, chosen, occupied, layer_area)
                else:
                    cpu_reference = package_center

                remaining = [bundle for bundle in bundles
                             if "anchor" not in bundle]
                remaining.sort(key=bundle_priority)
                for bundle in remaining:
                    kind = bundle["type"]
                    choices = sorted(
                        anchors[kind],
                        key=lambda anchor: (
                            center_distance(kind, anchor, cpu_reference),
                            center_distance(kind, anchor, package_center),
                            idx[anchor]))
                    chosen = next((anchor for anchor in choices
                                   if anchor_is_legal(
                                       bundle, anchor, occupied, layer_area)),
                                  None)
                    if chosen is None:
                        return False
                    place(bundle, chosen, occupied, layer_area)
                return True

            cpu_bundles = [bundle for bundle in bundles
                           if bundle["type"] == "CPU"]
            if cpu_bundles:
                primary_cpu = min(cpu_bundles, key=lambda bundle: bundle["id"])
                cpu_anchor_trials = sorted(
                    anchors["CPU"],
                    key=lambda anchor: (
                        center_distance("CPU", anchor, package_center),
                        idx[anchor]))
            else:
                primary_cpu = None
                cpu_anchor_trials = [None]

            packed = False
            for cpu_anchor in cpu_anchor_trials:
                if performance_first_pack(cpu_anchor):
                    packed = True
                    break
            if not packed:
                return None, (
                    "performance-first center-out footprint packing failed")

            protocol_refinement = {
                "attempted": False,
                "kind": "bounded exact-protocol typed-bundle descent",
                "initial_cycles": None,
                "final_cycles": None,
                "improved": False,
                "wall_s": 0.0,
            }
            if (noc_traffic_kind == "deterministic-dispatch" and
                    len(cpu_bundles) == 1):
                bundles_by_id = {
                    bundle["id"]: bundle for bundle in bundles}
                expert_bundle = {}
                for bundle in bundles:
                    for task in bundle["tasks"]:
                        if task in expert_tasks:
                            expert_bundle[task] = bundle
                if all(expert in expert_bundle
                       for expert in dispatch_expert_tasks):
                    compute_values = [
                        protocol_compute_cycles[expert][expert_plan[expert]]
                        for expert in dispatch_expert_tasks]
                    cpu_bid = cpu_bundles[0]["id"]

                    def exact_protocol_for_state(state):
                        expert_sites_value = [
                            state[expert_bundle[expert]["id"]]
                            for expert in dispatch_expert_tasks]
                        return _evaluate_fixed_order_protocol(
                            state[cpu_bid], expert_sites_value, dispatch_matrix,
                            compute_values, dispatch_token_bytes_per_phase,
                            init_bytes, done_bytes, dispatch_sync_overhead,
                            protocol_flit_bytes, protocol_hop_cycles,
                            expert_resource_ids=expert_sites_value,
                            compute_order=list(range(
                                len(dispatch_expert_tasks))))

                    def exact_protocol_cycles_for_state(state):
                        return exact_protocol_for_state(state)[
                            "completion_cycles"]

                    def state_is_legal(state):
                        occupied_state = set()
                        area_by_layer = {
                            layer: 0.0 for layer in range(L)}
                        for bid, anchor in state.items():
                            bundle = bundles_by_id[bid]
                            kind = bundle["type"]
                            if anchor not in anchors[kind]:
                                return False
                            cells = footprint[kind][anchor]
                            if occupied_state & cells:
                                return False
                            occupied_state.update(cells)
                            area_by_layer[anchor[0]] += float(
                                lib[kind]["area"])
                        if a_layer_limit is not None and any(
                                area > float(a_layer_limit) + 1.0e-9
                                for area in area_by_layer.values()):
                            return False
                        return True

                    initial_state = {
                        bundle["id"]: bundle["anchor"]
                        for bundle in bundles}
                    # The fixed replay order makes late-finishing workers the
                    # most likely members of the max-plus critical tail.  Put
                    # them first so bounded relocation/swap scans spend their
                    # budget on objective-critical experts, rather than on the
                    # already center-placed CPU or low-index workers.
                    initial_protocol = exact_protocol_for_state(initial_state)
                    critical_experts = sorted(
                        enumerate(dispatch_expert_tasks),
                        key=lambda item: (
                            -float(initial_protocol[
                                "worker_finish_cycles"][item[0]]),
                            item[0]))
                    search_ids = []
                    for _, expert in critical_experts:
                        bid = expert_bundle[expert]["id"]
                        if bid not in search_ids:
                            search_ids.append(bid)
                    if cpu_bid not in search_ids:
                        search_ids.append(cpu_bid)
                    for bid in sorted(bundles_by_id):
                        if bid not in search_ids:
                            search_ids.append(bid)
                    allowed_by_bundle = {
                        bid: list(anchors[bundles_by_id[bid]["type"]])
                        for bid in search_ids}
                    default_evaluations = (
                        600 if len(search_ids) > 64 else
                        300 if len(search_ids) > 24 else 2000)
                    default_refinement_wall = (
                        6.0 if len(search_ids) > 64 else
                        2.0)
                    evaluation_override = solver_opts.get(
                        "stage0_protocol_evaluations")
                    round_override = solver_opts.get(
                        "stage0_protocol_rounds")
                    wall_override = solver_opts.get(
                        "stage0_protocol_wall_s")
                    evaluation_limit = int(
                        default_evaluations
                        if evaluation_override is None else
                        evaluation_override)
                    refined_state, protocol_refinement = \
                        _deterministic_protocol_placement_descent(
                            initial_state, search_ids, allowed_by_bundle,
                            state_is_legal, exact_protocol_cycles_for_state,
                            max_evaluations=evaluation_limit,
                            max_rounds=int(
                                2 if round_override is None else
                                round_override),
                            max_wall_s=float(
                                default_refinement_wall
                                if wall_override is None else wall_override))
                    protocol_refinement["search_order_policy"] = (
                        "descending initial worker-finish criticality; CPU "
                        "after experts")
                    protocol_refinement["critical_expert_order"] = [
                        expert for _, expert in critical_experts]
                    for bid, anchor in refined_state.items():
                        bundles_by_id[bid]["anchor"] = anchor
            return {
                "strategy": label,
                "bundles": bundles,
                "counts": counts,
                "area": area_value,
                "power": power_value,
                "packing_policy": (
                    "CPU center-first; footprint-safe bundles ordered by "
                    "cycle-gain/traffic heat and placed center-out"),
                "cpu_anchor": (primary_cpu.get("anchor")
                               if primary_cpu is not None else None),
                "protocol_placement_refinement": protocol_refinement,
            }, None

        candidate = None
        failures = []
        upgrade_attempts = []
        selected_upgrade_count = None
        for label, plan, requested_upgrades in type_plans:
            candidate, failure = assemble_bundles(label, plan)
            upgrade_attempts.append({
                "requested_upgrade_count": requested_upgrades,
                "accepted": candidate is not None,
                "failure": failure,
            })
            if candidate is not None:
                selected_upgrade_count = requested_upgrades
                break
            failures.append("%s: %s" % (label, failure))
        evidence["upgrade_attempts"] = upgrade_attempts
        if candidate is None:
            evidence["reason"] = "; ".join(failures)
            return evidence

        bundles = candidate["bundles"]
        task_assignment = {}
        selected_x = set()
        for bundle in bundles:
            kind, anchor = bundle["type"], bundle["anchor"]
            selected_x.add((kind, anchor))
            for task in bundle["tasks"]:
                task_assignment[task] = (kind, anchor)
        if set(task_assignment) != set(V):
            evidence["reason"] = "constructed bundles do not cover all tasks"
            return evidence

        # Complete every placement/type binary family.
        for kind in K:
            for anchor in anchors[kind]:
                x[(kind, anchor)].Start = int(
                    (kind, anchor) in selected_x)
        for task in V:
            assigned_kind, assigned_anchor = task_assignment[task]
            for anchor in P:
                y[(task, anchor)].Start = int(anchor == assigned_anchor)
            for kind in tasks[task]["allowed"]:
                for anchor in anchors[kind]:
                    yk[(task, kind, anchor)].Start = int(
                        kind == assigned_kind and anchor == assigned_anchor)

        task_site_start = {
            task: assignment[1] for task, assignment in task_assignment.items()}
        task_kind_start = {
            task: assignment[0] for task, assignment in task_assignment.items()}

        # Stable topological order supplies consistent same-site disjunctions.
        # A non-preemptive list schedule then advances one cursor per selected
        # chiplet, allowing unrelated tasks on different chiplets to overlap.
        successors_start = {task: [] for task in V}
        indegree = {task: 0 for task in V}
        for edge in edges:
            successors_start[edge["src"]].append(edge["dst"])
            indegree[edge["dst"]] += 1
        ready = [task for task in V if indegree[task] == 0]
        ready.sort(key=V.index)
        topo = []
        while ready:
            task = ready.pop(0)
            topo.append(task)
            for target in successors_start[task]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort(key=V.index)
        if len(topo) != len(V):
            evidence["reason"] = "task graph is cyclic"
            return evidence
        topo_rank = {task: rank for rank, task in enumerate(topo)}

        for (first, second), states in order_states.items():
            first_site = task_site_start[first]
            second_site = task_site_start[second]
            values = [0, 0, 0, 0]
            if idx[first_site] < idx[second_site]:
                values[0] = 1
            elif idx[first_site] > idx[second_site]:
                values[1] = 1
            elif topo_rank[first] < topo_rank[second]:
                values[2] = 1
            else:
                values[3] = 1
            for state, value in zip(states, values):
                state.Start = value

        predecessor_edges = {task: [] for task in V}
        for number, edge in enumerate(edges):
            predecessor_edges[edge["dst"]].append((number, edge))
            source_site = task_site_start[edge["src"]]
            target_site = task_site_start[edge["dst"]]
            delta = tuple(abs(source_site[axis] - target_site[axis])
                          for axis in range(3))
            dL[number].Start, dI[number].Start, dJ[number].Start = delta
            lower, same, higher = edge_location_states[number]
            source_index = idx[source_site]
            target_index = idx[target_site]
            lower.Start = int(source_index < target_index)
            same.Start = int(source_index == target_index)
            higher.Start = int(source_index > target_index)

        schedule_start = {}
        schedule_finish = {}
        site_cursor = {site: 0.0 for site in set(task_site_start.values())}
        for task in topo:
            dependency_ready = 0.0
            for number, edge in predecessor_edges[task]:
                source = edge["src"]
                source_site = task_site_start[source]
                target_site = task_site_start[task]
                d_l = abs(source_site[0] - target_site[0])
                d_i = abs(source_site[1] - target_site[1])
                d_j = abs(source_site[2] - target_site[2])
                remote = source_site != target_site
                transfer = (
                    ((t0 + float(edge["vol"]) / beta) if remote else 0.0) +
                    tau_h * (d_i + d_j) + tau_v * d_l)
                dependency_ready = max(
                    dependency_ready, schedule_finish[source] + transfer)
            task_site_value = task_site_start[task]
            start_value = max(site_cursor[task_site_value], dependency_ready)
            finish_value = start_value + float(
                tasks[task]["cyc"][task_kind_start[task]])
            schedule_start[task] = start_value
            schedule_finish[task] = finish_value
            site_cursor[task_site_value] = finish_value
            s[task].Start = start_value
            f[task].Start = finish_value

        # Exact shortest paths in deterministic axis order complete the flow
        # start and expose the required aggregate-capacity horizon.
        link_lookup = {
            (source, target): link_id
            for link_id, (source, target) in enumerate(directed_links)}
        route_start_values = {
            (phase, source, link_id): 0.0
            for phase in route_phases for source in route_sources[phase]
            for link_id in range(len(directed_links))}
        phase_completion_start_total = 0.0
        for phase in route_phases:
            for source in route_sources[phase]:
                source_site = task_site_start[source]
                for target, volume in scaled_demands[phase][source].items():
                    current = list(source_site)
                    destination = task_site_start[target]
                    for axis in range(3):
                        while current[axis] != destination[axis]:
                            step = (1 if destination[axis] > current[axis]
                                    else -1)
                            next_site = list(current)
                            next_site[axis] += step
                            first = tuple(current)
                            second = tuple(next_site)
                            link_id = link_lookup[(first, second)]
                            route_start_values[(phase, source, link_id)] += (
                                float(volume))
                            current = next_site
        phase_load_start = {
            (phase, link_id): sum(
                route_start_values[(phase, source, link_id)]
                for source in route_sources[phase])
            for phase in route_phases
            for link_id in range(len(directed_links))}
        aggregate_capacity_T = 0.0
        for key, value in route_start_values.items():
            route_flow[key].Start = value
        for link_id in range(len(directed_links)):
            aggregate_load = sum(
                phase_load_start[(phase, link_id)]
                for phase in route_phases)
            aggregate_capacity_T = max(
                aggregate_capacity_T,
                aggregate_load * flow_scale_bytes / beta)

        for phase in route_phases:
            link_cycles_value = max(
                [0.0] + [
                    phase_load_start[(phase, link_id)] *
                    flow_scale_bytes / beta
                    for link_id in range(len(directed_links))])
            startup = (dispatch_sync_overhead
                       if noc_traffic_kind == "deterministic-dispatch"
                       else t0)
            source_endpoint = {}
            destination_endpoint = {}
            for source in route_sources[phase]:
                value = 0.0
                for target, volume in traffic_by_phase[phase][source].items():
                    first = task_site_start[source]
                    second = task_site_start[target]
                    d_l = abs(first[0] - second[0])
                    d_i = abs(first[1] - second[1])
                    d_j = abs(first[2] - second[2])
                    message_value = (
                        startup + float(volume) / beta +
                        tau_h * (d_i + d_j) + tau_v * d_l)
                    value += message_value
                    destination_endpoint[target] = (
                        destination_endpoint.get(target, 0.0) +
                        message_value)
                source_endpoint[source] = value
            endpoint_value = max(
                [0.0] + list(source_endpoint.values()) +
                list(destination_endpoint.values()))
            phase_link_cycles[phase].Start = link_cycles_value
            phase_endpoint_cycles[phase].Start = endpoint_value
            phase_completion_value = max(link_cycles_value, endpoint_value)
            phase_completion_cycles[phase].Start = phase_completion_value
            phase_completion_start_total += phase_completion_value

        for (pair, axis), variable in message_axis_distance.items():
            first, second = pair
            variable.Start = abs(
                task_site_start[first][axis] -
                task_site_start[second][axis])
        for expert, distances in cpu_control_distances.items():
            cpu_tasks = [task for task in V
                         if task_kind_start[task] == "CPU"]
            if cpu_tasks:
                cpu_site = task_site_start[cpu_tasks[0]]
                for axis, variable in enumerate(distances):
                    variable.Start = abs(
                        cpu_site[axis] - task_site_start[expert][axis])

        task_T_start = max(schedule_finish.values())
        if task_T_start > time_horizon + 1.0e-6:
            evidence["reason"] = (
                "constructed timing start exceeds proved horizon")
            return evidence
        T_task.Start = task_T_start

        protocol_T_start = None
        if noc_traffic_kind == "deterministic-dispatch":
            cpu_sites_start = [
                bundle["anchor"] for bundle in bundles
                if bundle["type"] == "CPU"]
            if len(cpu_sites_start) != 1:
                evidence["reason"] = (
                    "deterministic protocol start requires exactly one CPU")
                return evidence
            expert_sites_start = [
                task_site_start[expert] for expert in dispatch_expert_tasks]
            selected_compute_start = [
                protocol_compute_cycles[expert][task_kind_start[expert]]
                for expert in dispatch_expert_tasks]
            group_by_expert = {
                expert: group for group, expert in
                enumerate(dispatch_expert_tasks)}
            protocol_compute_order_start = [
                group_by_expert[task] for task in topo
                if task in group_by_expert]
            evaluated_start = _evaluate_fixed_order_protocol(
                cpu_sites_start[0], expert_sites_start, dispatch_matrix,
                selected_compute_start, dispatch_token_bytes_per_phase,
                init_bytes, done_bytes, dispatch_sync_overhead,
                protocol_flit_bytes, protocol_hop_cycles,
                expert_resource_ids=expert_sites_start,
                compute_order=protocol_compute_order_start)
            if len(evaluated_start["event_finish_cycles"]) != len(
                    protocol_event_vars):
                evidence["reason"] = (
                    "protocol evaluator/event-variable count mismatch")
                return evidence
            for variable, value in zip(
                    protocol_event_vars,
                    evaluated_start["event_finish_cycles"]):
                variable.Start = value
            for group, expert in enumerate(dispatch_expert_tasks):
                protocol_compute_start[expert].Start = evaluated_start[
                    "compute_start_cycles"][group]
                protocol_compute_finish[expert].Start = evaluated_start[
                    "compute_finish_cycles"][group]
            protocol_T_start = evaluated_start["completion_cycles"]
            T_start = max(protocol_T_start, aggregate_capacity_T)
        else:
            protocol_T_start = task_T_start + phase_completion_start_total
            T_start = max(protocol_T_start, aggregate_capacity_T)
        T.Start = T_start
        # The start is intentionally complete: with only a small fraction of a
        # 30-second total budget left at E128, Gurobi must be able to validate
        # the incumbent without solving a partial-start completion sub-MIP.
        m.update()
        undefined_variables = []
        all_model_variables = m.getVars()
        for variable in all_model_variables:
            try:
                start_value = float(variable.Start)
            except (TypeError, ValueError):
                undefined_variables.append(variable.VarName)
                continue
            if (not math.isfinite(start_value) or
                    abs(start_value) >= 0.5 * float(GRB.UNDEFINED)):
                undefined_variables.append(variable.VarName)
        evidence["mip_start_variable_count"] = len(all_model_variables)
        evidence["mip_start_undefined_variable_count"] = len(
            undefined_variables)
        evidence["mip_start_undefined_variables"] = undefined_variables[:32]
        if undefined_variables:
            evidence["reason"] = (
                "constructed MIP start leaves %d variables undefined" %
                len(undefined_variables))
            return evidence
        m.Params.StartNodeLimit = max(
            int(solver_opts.get("start_node_limit", 0)), 0)

        evidence.update({
            "accepted_for_submission": True,
            "reason": "complete all-variable feasible MIP start submitted",
            "strategy": candidate["strategy"],
            "packing_policy": candidate["packing_policy"],
            "all_gpu_s": all(
                task_kind_start[task] == "GPU-S"
                for task in ordered_experts),
            "type_counts": candidate["counts"],
            "bundle_count": len(bundles),
            "total_area_mm2": candidate["area"],
            "peak_power_W": candidate["power"],
            "analytical_T_start_cycles": T_start,
            "legacy_task_T_start_cycles": task_T_start,
            "protocol_T_start_cycles": protocol_T_start,
            "requested_upgrade_count": selected_upgrade_count,
            "realized_upgrade_count": sum(
                task_kind_start[record["task"]] == record["fastest_type"]
                for record in upgrade_records),
            "expert_type_assignment": {
                task: task_kind_start[task] for task in ordered_experts},
            "cpu_anchor": (list(candidate["cpu_anchor"])
                            if candidate["cpu_anchor"] is not None else None),
            "protocol_placement_refinement": candidate[
                "protocol_placement_refinement"],
            "bundles": [
                {"id": bundle["id"], "type": bundle["type"],
                  "tasks": list(bundle["tasks"]),
                  "anchor": list(bundle["anchor"]),
                  "performance_priority_cycle_gain": float(
                      bundle.get("performance_priority_cycle_gain", 0.0)),
                  "traffic_priority_bytes": float(
                      bundle.get("traffic_priority_bytes", 0.0))}
                for bundle in bundles],
        })
        return evidence

    stage0_started = time.time()
    stage0_feasible_start = build_stage0_feasible_start()
    stage0_total_s = time.time() - stage0_started

    model_build_s = time.time() - build_start
    formulation_build_s = max(0.0, stage0_started - build_start)
    # The advertised time budget covers model construction and primal-start
    # refinement as well as branch-and-bound.  Leave a tiny solve allowance so
    # Gurobi can accept a submitted start even if construction exhausted the
    # requested budget.
    primary_time_limit = max(
        0.01, requested_total_time_limit - model_build_s)
    m.Params.TimeLimit = primary_time_limit
    anytime_trace = []

    def expression_value(expression):
        """Read a scalar Gurobi Var or expression on Gurobi 10/Python 3.7."""
        if hasattr(expression, "getValue"):
            return float(expression.getValue())
        return float(expression.X)

    def _record_incumbent(model, where):
        if where != GRB.Callback.MIPSOL:
            return
        try:
            runtime = float(model.cbGet(GRB.Callback.RUNTIME))
            obj = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
            bound = float(model.cbGet(GRB.Callback.MIPSOL_OBJBND))
            gap = abs(obj - bound) / max(abs(obj), 1e-12)
            anytime_trace.append({
                "wall_s": runtime,
                "incumbent_cycles": obj,
                "bound_cycles": bound,
                "gap": gap,
            })
        except Exception:
            # Instrumentation must never change the optimization result.
            pass

    t_start = time.time()
    m.optimize(_record_incumbent)
    primary_wall = time.time() - t_start

    def normalize_solver_gap(raw_gap, has_incumbent):
        """Return a JSON-safe gap and an explicit certificate record.

        Gurobi reports ``GRB.INFINITY`` for a time-limited stage that has an
        incumbent but no finite bound.  That is a valid *uncertified*
        incumbent, not a numeric MIP-gap certificate, and must never escape as
        the non-standard JSON token ``Infinity``.
        """
        if not has_incumbent:
            return None, {
                "certified": False,
                "reason": "stage-has-no-incumbent",
            }
        try:
            gap = float(raw_gap)
        except (TypeError, ValueError):
            return None, {
                "certified": False,
                "reason": "solver-mip-gap-unavailable",
            }
        if not math.isfinite(gap):
            return None, {
                "certified": False,
                "reason": "solver-reported-nonfinite-mip-gap",
            }
        if gap < 0.0:
            # Preserve a finite negative value so strict downstream validators
            # can distinguish solver corruption from a missing certificate.
            return gap, {
                "certified": False,
                "reason": "solver-reported-negative-mip-gap",
            }
        return gap, {"certified": True, "reason": None}

    primary_status = int(m.Status)
    primary_gap = None
    primary_gap_certificate = {
        "certified": False, "reason": "stage-has-no-incumbent"}
    primary_bound = None
    primary_incumbent = None
    primary_nodes = None
    if m.SolCount > 0:
        primary_incumbent = expression_value(primary_performance_expr)
        try:
            raw_primary_gap = m.MIPGap
        except Exception:
            raw_primary_gap = None
        primary_gap, primary_gap_certificate = normalize_solver_gap(
            raw_primary_gap, True)
        try:
            primary_bound = float(m.ObjBound)
            if not math.isfinite(primary_bound):
                primary_bound = None
        except Exception:
            primary_bound = None
        try:
            primary_nodes = float(m.NodeCount)
        except Exception:
            primary_nodes = None
        final_primary = {
            "wall_s": primary_wall,
            "incumbent_cycles": primary_incumbent,
            "bound_cycles": primary_bound,
            "gap": primary_gap,
        }
        if (not anytime_trace or
                abs(anytime_trace[-1]["incumbent_cycles"] -
                    primary_incumbent) > 1.0e-6 or
                anytime_trace[-1]["gap"] != primary_gap):
            anytime_trace.append(final_primary)

    if m.SolCount == 0:
        return None, {
            "status": primary_status,
            "wall": model_build_s + primary_wall,
            "model_build_s": model_build_s,
            "formulation_build_s": formulation_build_s,
            "stage0_total_s": stage0_total_s,
            "stage0_protocol_refinement_wall_s": float(
                (stage0_feasible_start.get(
                    "protocol_placement_refinement") or {}).get(
                        "wall_s", 0.0)),
            "requested_total_time_limit_s": requested_total_time_limit,
            "primary_solver_time_limit_s": primary_time_limit,
            "feasible": False,
            "anytime_trace": anytime_trace,
            "stage0_feasible_start": stage0_feasible_start,
            "num_vars": int(m.NumVars),
            "num_bin_vars": int(m.NumBinVars),
            "num_constrs": int(m.NumConstrs),
            "num_genconstrs": int(m.NumGenConstrs),
        }

    # ---- Optional makespan-preserving byte-hop compaction -------------------
    # Root cause of the large-instance placement gap: within the short primary
    # limit the min-T incumbent is only gap-certified, so among the many
    # layouts sharing an (essentially) equal makespan epigraph Gurobi may keep
    # one that scatters experts to the mesh corners.  The scattered layout has
    # the same modeled T but a materially higher *average* packet latency, which
    # is what a cycle-accurate replay actually measures.  When a caller opts in,
    # run one extra stage on the SAME joint model: cap T at the primary
    # incumbent (so the reported makespan never regresses) and re-minimize total
    # routed KiB-hops -- a linear proxy for average latency -- warm-started from
    # the primary incumbent.  The result is the most compact layout that still
    # achieves the optimized makespan.  Off by default: the sole objective stays
    # min T unless ``byte_hop_tiebreak`` is set, so every existing run is
    # bit-for-bit unchanged.
    byte_hop_tiebreak_record = {"enabled": False}
    if solver_opts.get("byte_hop_tiebreak") and m.SolCount > 0:
        tie_t0 = time.time()
        t_star = expression_value(T)
        makespan_slack = float(solver_opts.get("byte_hop_makespan_slack", 0.0))
        tie_cap_value = t_star * (1.0 + makespan_slack) + 1.0e-6
        m.addConstr(T <= tie_cap_value, name="Ctiebreak_makespan")
        byte_hop_expr = gp.quicksum(
            route_flow[(phase, source, link_id)]
            for phase in route_phases for source in route_sources[phase]
            for link_id in range(len(directed_links)))
        m.setObjective(byte_hop_expr, GRB.MINIMIZE)
        m.Params.TimeLimit = float(
            solver_opts.get("byte_hop_time_limit", primary_time_limit))
        m.Params.MIPGap = float(solver_opts.get("byte_hop_mipgap", 0.01))
        m.optimize()
        byte_hop_tiebreak_record = {
            "enabled": True,
            "makespan_cap_cycles": tie_cap_value,
            "primary_makespan_cycles": t_star,
            "post_makespan_cycles": (
                expression_value(T) if m.SolCount > 0 else None),
            "byte_hop_objective": (
                float(m.ObjVal) if m.SolCount > 0 else None),
            "byte_hop_bound": (
                float(m.ObjBound) if m.SolCount > 0 else None),
            "status": int(m.Status),
            "wall_s": time.time() - tie_t0,
        }
        # Leave the model holding the compacted incumbent; every primary_*
        # statistic reported below was already captured before this stage.

    secondary_wall = 0.0
    secondary_status = None
    secondary_gap = None
    secondary_has_incumbent = False
    secondary_gap_certificate = {
        "certified": False, "reason": "stage-not-run"}
    secondary_incumbent = None
    secondary_objective = None
    tertiary_wall = 0.0
    tertiary_status = None
    tertiary_gap = None
    tertiary_has_incumbent = False
    tertiary_gap_certificate = {
        "certified": False, "reason": "stage-not-run"}
    tertiary_incumbent = None
    tertiary_objective = None
    quaternary_wall = 0.0
    quaternary_status = None
    quaternary_gap = None
    quaternary_has_incumbent = False
    quaternary_gap_certificate = {
        "certified": False, "reason": "stage-not-run"}
    quaternary_incumbent = None
    quaternary_objective = None
    reference_probe_wall = 0.0
    compute_selection_relaxation = None
    compute_selection_cap = None
    performance_relaxation = None
    performance_reference = None
    performance_reference_kind = None
    performance_cap = None
    performance_cap_enforced = False
    secondary_time_limit = None
    tertiary_time_limit = None
    internal_bundle_refinement = {
        "enabled": False,
        "source_stage": None,
        "coordinates_fixed": False,
        "bundle_count": 0,
        "type_counts": None,
        "bundle_signature": None,
        "external_bundle_signature": (
            bundle_signature(placement_bundles)
            if placement_bundles is not None else None),
        "matches_external_bundles": None,
        "bundles": [],
    }
    internal_bundle_z = None
    internal_stage0_bundles = None
    constructive_perf_start = {
        "attempted": False,
        "accepted": False,
        "reason": None,
    }
    constructive_power_start = {
        "attempted": False,
        "accepted": False,
        "reason": None,
    }

    def freeze_stage0_bundles():
        """Restrict later stages to Stage 0's typed task bundles.

        The fresh z variables reproduce the exact placement-only formulation
        used when a serialized solution is supplied through --placement-from:
        type counts and complete co-resident task groups are fixed, while every
        bundle remains free to move to any legal anchor of its selected type.
        """
        task_assignment = {}
        for (task, kind, anchor), variable in yk.items():
            if variable.X > 0.5:
                if task in task_assignment:
                    raise RuntimeError(
                        "Stage 0 assigned task %s more than once" % task)
                task_assignment[task] = (kind, anchor)
        if set(task_assignment) != set(V):
            missing = sorted(set(V) - set(task_assignment))
            raise RuntimeError(
                "Stage 0 bundle capture missed tasks %s" % missing)

        stage0_bundles = []
        selected_x = set()
        for kind in K:
            for anchor in anchors[kind]:
                if x[(kind, anchor)].X <= 0.5:
                    continue
                selected_x.add((kind, anchor))
                residents = sorted(
                    task for task, assignment in task_assignment.items()
                    if assignment == (kind, anchor))
                if not residents:
                    raise RuntimeError(
                        "Stage 0 selected empty %s chiplet at %s" %
                        (kind, anchor))
                stage0_bundles.append({
                    "id": len(stage0_bundles),
                    "type": kind,
                    "tasks": residents,
                    "source_anchor": anchor,
                })

        stage0_signature = bundle_signature(stage0_bundles)
        external_signature = internal_bundle_refinement[
            "external_bundle_signature"]
        if (external_signature is not None and
                external_signature != stage0_signature):
            raise RuntimeError(
                "Stage 0 bundle signature disagrees with --placement-from")

        z_keys = [(bundle["id"], anchor)
                  for bundle in stage0_bundles
                  for anchor in anchors[bundle["type"]]]
        internal_z = m.addVars(
            z_keys, vtype=GRB.BINARY, name="internal_bundle_z")
        constraint_count = 0
        for bundle in stage0_bundles:
            bid = bundle["id"]
            kind = bundle["type"]
            source_anchor = bundle["source_anchor"]
            for anchor in anchors[kind]:
                internal_z[(bid, anchor)].Start = int(
                    anchor == source_anchor)
            m.addConstr(
                gp.quicksum(internal_z[(bid, anchor)]
                            for anchor in anchors[kind]) == 1,
                name="INTERNAL_BUNDLE_once_%d" % bid)
            constraint_count += 1

        for kind in K:
            typed = [bundle for bundle in stage0_bundles
                     if bundle["type"] == kind]
            for anchor in anchors[kind]:
                m.addConstr(
                    x[(kind, anchor)] == gp.quicksum(
                        internal_z[(bundle["id"], anchor)]
                        for bundle in typed),
                    name="INTERNAL_BUNDLE_type_%s_%d" %
                         (kind, idx[anchor]))
                constraint_count += 1
                x[(kind, anchor)].Start = int(
                    (kind, anchor) in selected_x)

        task_bundle = {}
        for bundle in stage0_bundles:
            for task in bundle["tasks"]:
                task_bundle[task] = bundle
        for task in V:
            bundle = task_bundle[task]
            bid = bundle["id"]
            fixed_kind = bundle["type"]
            for kind in tasks[task]["allowed"]:
                for anchor in anchors[kind]:
                    variable = yk[(task, kind, anchor)]
                    variable.Start = int(
                        task_assignment[task] == (kind, anchor))
                    if kind == fixed_kind:
                        m.addConstr(
                            variable == internal_z[(bid, anchor)],
                            name="INTERNAL_BUNDLE_task_%s_%s_%d" %
                                 (task, kind, idx[anchor]))
                    else:
                        m.addConstr(
                            variable == 0,
                            name="INTERNAL_BUNDLE_task_zero_%s_%s_%d" %
                                 (task, kind, idx[anchor]))
                    constraint_count += 1
        m.update()

        type_counts = {
            kind: sum(1 for bundle in stage0_bundles
                      if bundle["type"] == kind)
            for kind in K}
        evidence = {
            "enabled": True,
            "source_stage": "Stage 0 min-T incumbent",
            "coordinates_fixed": False,
            "search_space": "exact typed complete bundles; free coordinates",
            "bundle_count": len(stage0_bundles),
            "type_counts": type_counts,
            "bundle_signature": stage0_signature,
            "external_bundle_signature": external_signature,
            "matches_external_bundles": (
                external_signature == stage0_signature
                if external_signature is not None else None),
            "constraint_count": constraint_count,
            "bundles": [
                {"id": bundle["id"], "type": bundle["type"],
                 "tasks": bundle["tasks"],
                 "stage0_anchor": list(bundle["source_anchor"])}
                for bundle in stage0_bundles],
        }
        return evidence, internal_z, stage0_bundles

    def build_constructive_placement_start(
            intent, search_bundles, search_bundle_z,
            reference_cycles=None, deadline_cycles=None):
        """Build a deterministic protocol-aware placement-only MIP start.

        This searches only the already fixed, type-labelled complete bundles.
        Perf ranks candidates only by the exact max-plus protocol recurrence.
        Power ranks them by closed-form full-protocol dynamic hop energy and
        accepts candidates no slower than its caller-supplied deadline.  This is a
        primal-start heuristic only; the subsequent MILP stages remain
        authoritative.
        """
        if intent not in ("performance", "power"):
            raise ValueError("unknown constructive placement intent %s" %
                             intent)
        constructive_started = time.time()
        evidence = {
            "attempted": True,
            "accepted": False,
            "reason": None,
            "intent": intent,
            "seed": None,
            "iterations": 0,
            "restarts": 0,
            "requires_protocol_floor": False,
            "reference_cycles": (
                float(reference_cycles)
                if reference_cycles is not None else None),
            "deadline_cycles": (
                float(deadline_cycles)
                if deadline_cycles is not None else None),
        }

        def finish_evidence():
            evidence["wall_s"] = time.time() - constructive_started
            return evidence
        if (search_bundles is None or search_bundle_z is None or
                noc_traffic_kind != "deterministic-dispatch"):
            evidence["reason"] = (
                "requires deterministic traffic and fixed complete bundles")
            return finish_evidence()
        if not search_bundles:
            evidence["reason"] = "no placement bundles"
            return finish_evidence()

        bundles_by_id = {int(bundle["id"]): bundle
                         for bundle in search_bundles}
        bundle_ids = sorted(bundles_by_id)
        task_bundle = {}
        for bid in bundle_ids:
            for task in bundles_by_id[bid]["tasks"]:
                task_bundle[task] = bid
        try:
            expert_bundle = {
                group: task_bundle[dispatch_expert_tasks[group]]
                for group in range(len(dispatch_expert_tasks))}
        except KeyError:
            evidence["reason"] = "expert task missing from placement bundles"
            return finish_evidence()
        cpu_bundle_ids = [bid for bid in bundle_ids
                          if bundles_by_id[bid]["type"] == "CPU"]
        if len(cpu_bundle_ids) != 1:
            evidence["reason"] = "constructive search requires one CPU bundle"
            return finish_evidence()
        cpu_bid = cpu_bundle_ids[0]

        initial = {
            bid: tuple(bundles_by_id[bid]["source_anchor"])
            for bid in bundle_ids}
        allowed = {
            bid: list(anchors[bundles_by_id[bid]["type"]])
            for bid in bundle_ids}
        allowed_sets = {bid: set(values) for bid, values in allowed.items()}

        def placement_is_legal(state):
            occupied = set()
            for bid, anchor in state.items():
                kind = bundles_by_id[bid]["type"]
                if anchor not in allowed_sets[bid]:
                    return False
                cells = footprint[kind][anchor]
                if occupied & cells:
                    return False
                occupied.update(cells)
            return True

        def axis_distances(first, second):
            return tuple(abs(first[axis] - second[axis])
                         for axis in range(3))

        # Aggregate symmetric distance weights; direction does not affect
        # Manhattan dynamic hop energy.  Traffic already contains both phases.
        distance_terms = []
        for phase in route_phases:
            for source, destinations in traffic_by_phase[phase].items():
                source_bid = task_bundle[source]
                for target, volume in destinations.items():
                    target_bid = task_bundle[target]
                    if source_bid != target_bid and float(volume) > 0.0:
                        distance_terms.append(
                            (source_bid, target_bid, float(volume)))
        control_volume = float(init_bytes + done_bytes)
        for group in range(len(dispatch_expert_tasks)):
            target_bid = expert_bundle[group]
            if cpu_bid != target_bid:
                distance_terms.append(
                    (cpu_bid, target_bid, control_volume))

        energy_axis_scale = max(float(eps_h), float(eps_v), 1.0e-30)
        pair_axis_weights = {}
        for first, second, volume in distance_terms:
            key = ((first, second) if first < second else (second, first))
            weights = pair_axis_weights.setdefault(key, [0.0, 0.0, 0.0])
            weights[0] += float(volume) * float(eps_v) / energy_axis_scale
            weights[1] += float(volume) * float(eps_h) / energy_axis_scale
            weights[2] += float(volume) * float(eps_h) / energy_axis_scale

        def direct_metrics(state):
            weighted_byte_hop = 0.0
            byte_hop_value = 0.0
            for first, second, volume in distance_terms:
                d_l, d_i, d_j = axis_distances(
                    state[first], state[second])
                byte_hop_value += volume * (d_l + d_i + d_j)
                weighted_byte_hop += volume * (
                    float(eps_v) / energy_axis_scale * d_l +
                    float(eps_h) / energy_axis_scale * (d_i + d_j))
            energy_value = 8.0 * energy_axis_scale * weighted_byte_hop
            return weighted_byte_hop, byte_hop_value, energy_value

        def exact_protocol_cycles(state):
            selected_compute_cycles = []
            expert_sites = []
            for group, expert in enumerate(dispatch_expert_tasks):
                bundle = expert_bundle[group]
                kind = bundles_by_id[bundle]["type"]
                expert_sites.append(state[bundle])
                selected_compute_cycles.append(
                    protocol_compute_cycles[expert][kind])
            evaluated = _evaluate_fixed_order_protocol(
                state[cpu_bid], expert_sites, dispatch_matrix,
                selected_compute_cycles, dispatch_token_bytes_per_phase,
                init_bytes, done_bytes, dispatch_sync_overhead,
                protocol_flit_bytes, protocol_hop_cycles)
            return evaluated["completion_cycles"]

        if not placement_is_legal(initial):
            evidence["reason"] = "serialized reference placement is illegal"
            return finish_evidence()
        initial_weighted, initial_byte_hop, initial_energy = direct_metrics(
            initial)
        initial_protocol = exact_protocol_cycles(initial)
        best = dict(initial)
        best_weighted = initial_weighted
        best_byte_hop = initial_byte_hop
        best_energy = initial_energy
        best_protocol = initial_protocol

        option_prefix = ("perf" if intent == "performance" else "power")
        iteration_budget = int(solver_opts.get(
            option_prefix + "_constructive_iterations", 50000))
        iteration_budget = max(0, iteration_budget)
        restart_count = int(solver_opts.get(
            option_prefix + "_constructive_restarts", 10))
        restart_count = max(1, restart_count)
        # Quadratic traffic evaluation is intentionally bounded for scaling
        # sweeps; the exact MILP remains active regardless of heuristic size.
        if len(bundle_ids) > 64:
            iteration_budget = min(iteration_budget, 1000)
            restart_count = min(restart_count, 2)
        elif len(bundle_ids) > 24:
            iteration_budget = min(iteration_budget, 5000)
            restart_count = min(restart_count, 5)
        random_seed = int(solver_opts.get("seed", 1)) * 1009 + 731
        rng = random.Random(random_seed)
        evidence["seed"] = random_seed
        evidence["iterations"] = iteration_budget
        evidence["restarts"] = restart_count
        per_restart = max(1, int(math.ceil(
            float(iteration_budget) / restart_count)))

        footprint_area = {
            bid: len(footprint[bundles_by_id[bid]["type"]][
                allowed[bid][0]])
            for bid in bundle_ids}

        def random_legal_state():
            order = sorted(bundle_ids,
                           key=lambda bid: (-footprint_area[bid], bid))
            for _ in range(100):
                state = {}
                possible = True
                for bid in order:
                    candidates = list(allowed[bid])
                    rng.shuffle(candidates)
                    placed = False
                    for anchor in candidates:
                        state[bid] = anchor
                        if placement_is_legal(state):
                            placed = True
                            break
                    if not placed:
                        possible = False
                        break
                if possible:
                    return state
            return dict(initial)

        upper_cycles = (float(deadline_cycles)
                        if deadline_cycles is not None else float("inf"))
        tolerance = 1.0e-6
        evidence["deterministic_greedy"] = {
            "enabled": False,
            "reason": "Power-only primal-start refinement",
        }
        if intent == "power":
            default_greedy_rounds = 4 if len(bundle_ids) <= 65 else 2
            greedy_rounds = int(solver_opts.get(
                "power_greedy_rounds", default_greedy_rounds))
            greedy_protocol_checks = int(solver_opts.get(
                "power_greedy_protocol_checks_per_round", 64))
            bundle_footprints = {
                bid: {
                    anchor: set(footprint[
                        bundles_by_id[bid]["type"]][anchor])
                    for anchor in allowed[bid]}
                for bid in bundle_ids}
            greedy_state, greedy_evidence = \
                _deterministic_power_placement_descent(
                    initial, bundle_ids, allowed, bundle_footprints,
                    pair_axis_weights, exact_protocol_cycles, upper_cycles,
                    max_rounds=greedy_rounds,
                    max_protocol_checks_per_round=greedy_protocol_checks)
            (greedy_weighted, greedy_byte_hop,
             greedy_energy) = direct_metrics(greedy_state)
            greedy_protocol = exact_protocol_cycles(greedy_state)
            greedy_evidence.update({
                "direct_metric_objective": greedy_weighted,
                "byte_hop": greedy_byte_hop,
                "energy_J": greedy_energy,
                "protocol_cycles": greedy_protocol,
                "within_deadline": (
                    greedy_protocol <= upper_cycles + tolerance),
            })
            if not math.isclose(
                    greedy_weighted, greedy_evidence["final_objective"],
                    rel_tol=1.0e-10, abs_tol=1.0e-6):
                raise AssertionError(
                    "incremental Power objective disagrees with direct metric")
            evidence["deterministic_greedy"] = greedy_evidence
            if (greedy_weighted < best_weighted - 1.0e-9 and
                    greedy_protocol <= upper_cycles + tolerance):
                best = dict(greedy_state)
                best_weighted = greedy_weighted
                best_byte_hop = greedy_byte_hop
                best_energy = greedy_energy
                best_protocol = greedy_protocol

        for restart in range(restart_count):
            state = (dict(best) if restart == 0
                     else random_legal_state())
            current_weighted = direct_metrics(state)[0]
            current_protocol = exact_protocol_cycles(state)
            for iteration in range(per_restart):
                first = rng.choice(bundle_ids)
                old_first = state[first]
                second = None
                old_second = None
                if len(bundle_ids) > 1 and rng.random() < 0.25:
                    second = rng.choice(
                        [bid for bid in bundle_ids if bid != first])
                    old_second = state[second]
                    if (old_second in allowed_sets[first] and
                            old_first in allowed_sets[second]):
                        state[first], state[second] = old_second, old_first
                    else:
                        second = None
                        state[first] = rng.choice(allowed[first])
                else:
                    state[first] = rng.choice(allowed[first])

                if not placement_is_legal(state):
                    state[first] = old_first
                    if second is not None:
                        state[second] = old_second
                    continue
                (candidate_weighted, candidate_byte_hop,
                 candidate_energy) = direct_metrics(state)
                candidate_protocol = (
                    exact_protocol_cycles(state)
                    if intent == "performance" else None)
                fraction = float(iteration) / max(per_restart - 1, 1)
                if intent == "performance":
                    current_objective = current_protocol
                    candidate_objective = candidate_protocol
                    temperature = max(
                        1.0, 0.02 * max(initial_protocol, 1.0) *
                        (1.0 - fraction))
                else:
                    current_objective = current_weighted
                    candidate_objective = candidate_weighted
                    temperature = max(
                        1.0, 0.10 * max(initial_weighted, 1.0) *
                        (1.0 - fraction))
                accept = candidate_objective <= current_objective
                if not accept:
                    exponent = ((current_objective - candidate_objective) /
                                temperature)
                    accept = rng.random() < math.exp(max(-50.0, exponent))
                if accept:
                    current_weighted = candidate_weighted
                    if intent == "performance":
                        current_protocol = candidate_protocol
                else:
                    state[first] = old_first
                    if second is not None:
                        state[second] = old_second
                    continue

                if intent == "performance":
                    # Equal-cycle layouts are intentionally left untouched:
                    # selecting the lower-energy/byte-hop candidate here would
                    # silently reintroduce a non-performance tie-break.
                    better = candidate_protocol < best_protocol - tolerance
                else:
                    if candidate_weighted >= best_weighted - 1.0e-9:
                        continue
                    candidate_protocol = exact_protocol_cycles(state)
                    better = candidate_protocol <= upper_cycles + tolerance
                if not better:
                    continue
                best = dict(state)
                best_weighted = candidate_weighted
                best_byte_hop = candidate_byte_hop
                best_energy = candidate_energy
                best_protocol = candidate_protocol

        # Remove translation symmetry from the submitted representative.
        minima = [min(anchor[axis] for anchor in best.values())
                  for axis in range(3)]
        translated = {
            bid: tuple(anchor[axis] - minima[axis]
                       for axis in range(3))
            for bid, anchor in best.items()}
        if placement_is_legal(translated):
            best = translated

        improved = (
            best_protocol < initial_protocol - tolerance
            if intent == "performance" else
            best_energy < initial_energy - 1.0e-18)
        if not improved:
            evidence.update({
                "reason": (
                    "no lower-cycle candidate found"
                    if intent == "performance" else
                    "no lower-energy deadline-feasible candidate found"),
                "initial_protocol_cycles": initial_protocol,
                "initial_byte_hop": initial_byte_hop,
                "initial_energy_J": initial_energy,
            })
            return finish_evidence()

        # Submit a complete discrete placement start.  Gurobi completes the
        # schedule, flow, and event clocks under the existing hard constraints.
        for bid in bundle_ids:
            kind = bundles_by_id[bid]["type"]
            for anchor in allowed[bid]:
                search_bundle_z[(bid, anchor)].Start = int(
                    anchor == best[bid])
        occupied_type_anchor = {
            (bundles_by_id[bid]["type"], best[bid]) for bid in bundle_ids}
        for kind in K:
            for anchor in anchors[kind]:
                x[(kind, anchor)].Start = int(
                    (kind, anchor) in occupied_type_anchor)
        for task in V:
            bid = task_bundle[task]
            fixed_kind = bundles_by_id[bid]["type"]
            for anchor in P:
                y[(task, anchor)].Start = int(anchor == best[bid])
            for kind in tasks[task]["allowed"]:
                for anchor in anchors[kind]:
                    yk[(task, kind, anchor)].Start = int(
                        kind == fixed_kind and anchor == best[bid])

        evidence.update({
            "accepted": True,
            "reason": "submitted as placement-only partial MIP start",
            "initial_protocol_cycles": initial_protocol,
            "candidate_protocol_cycles": best_protocol,
            "initial_byte_hop": initial_byte_hop,
            "candidate_byte_hop": best_byte_hop,
            "initial_energy_J": initial_energy,
            "candidate_energy_J": best_energy,
            "cycle_reduction_fraction": (
                (initial_protocol - best_protocol) / initial_protocol
                if initial_protocol > 0.0 else 0.0),
            "energy_reduction_fraction": (
                (initial_energy - best_energy) / initial_energy
                if initial_energy > 0.0 else 0.0),
            "candidate_anchors": {
                str(bid): list(best[bid]) for bid in bundle_ids},
            "submitted_as_mip_start": True,
        })
        return finish_evidence()

    stage0_protocol_refinement_wall = float(
        (stage0_feasible_start.get("protocol_placement_refinement") or {}).get(
            "wall_s", 0.0))
    constructive_wall = (stage0_protocol_refinement_wall +
        float(constructive_perf_start.get("wall_s", 0.0)) +
        float(constructive_power_start.get("wall_s", 0.0)))
    post_build_constructive_wall = max(
        0.0, constructive_wall - stage0_protocol_refinement_wall)
    wall = (model_build_s + primary_wall + post_build_constructive_wall +
            reference_probe_wall +
            secondary_wall + tertiary_wall + quaternary_wall)

    # ---- Extract solution ---------------------------------------------------
    site_alloc = {}   # pa -> {"type":k, "tasks":[...]}
    for k in K:
        for pa in anchors[k]:
            if x[(k, pa)].X > 0.5:
                site_alloc[pa] = {"type": k, "tasks": []}
    task_site = {}
    for v in V:
        for p in P:
            if y[v, p].X > 0.5:
                task_site[v] = p
                if p in site_alloc:
                    site_alloc[p]["tasks"].append(v)
                break

    # energies -- compute distances DIRECTLY from placed coordinates (exact).
    # (The delta vars are only tight on the makespan-critical path; off-path
    #  deltas may float above the true |coord diff|, so we must not read them
    #  back for reporting.)
    def hops(a_site, b_site):
        return (abs(a_site[1] - b_site[1]) + abs(a_site[2] - b_site[2]),  # h
                abs(a_site[0] - b_site[0]))                               # v
    e_comm_val = 0.0
    byte_hop = 0.0
    for e in edges:
        ps, pd = task_site[e["src"]], task_site[e["dst"]]
        dh, dv = hops(ps, pd)
        e_comm_val += 8 * e["vol"] * (eps_h * dh + eps_v * dv)
        byte_hop += e["vol"] * (dh + dv)
    # Evaluate NoC energy from the selected coordinates and packet flows.
    # Energy is not an objective even though protocol completion and the link-
    # capacity horizon contribute to T.
    routed_noc_energy_val = 8.0 * flow_scale_bytes * sum(
        (eps_v if link_axis[link_id] == 0 else eps_h) *
        float(route_flow[(phase, source, link_id)].X)
        for phase in route_phases for source in route_sources[phase]
        for link_id in range(len(directed_links)))
    cpu_control_energy_val = 0.0
    if noc_traffic_kind == "deterministic-dispatch":
        selected_cpu_sites = [
            anchor for anchor, allocation in site_alloc.items()
            if allocation["type"] == "CPU"]
        if len(selected_cpu_sites) != 1:
            raise RuntimeError(
                "NoC diagnostic requires one selected CPU")
        for expert in dispatch_expert_tasks:
            dh, dv = hops(selected_cpu_sites[0], task_site[expert])
            cpu_control_energy_val += 8.0 * (init_bytes + done_bytes) * (
                eps_h * dh + eps_v * dv)
    e_noc_proxy_val = routed_noc_energy_val + cpu_control_energy_val
    dispatch_assignment_hops = None
    dispatch_byte_hop = None
    if dispatch_matrix is not None:
        dispatch_assignment_hops = 0.0
        dispatch_byte_hop = 0.0
        groups = len(dispatch_matrix)
        for source in range(groups):
            for target in range(groups):
                assignments = dispatch_matrix[source][target]
                if source == target or assignments <= 0:
                    continue
                ps = task_site[dispatch_expert_tasks[source]]
                pd = task_site[dispatch_expert_tasks[target]]
                dh, dv = hops(ps, pd)
                dispatch_assignment_hops += assignments * (dh + dv)
                dispatch_byte_hop += (assignments * dispatch_token_bytes *
                                      (dh + dv))
    e_comp_val = sum(tasks[v]["eta"][k] * yk[(v, k, pa)].X
                     for v in V for k in tasks[v]["allowed"]
                     for pa in anchors[k] if (v, k, pa) in yk)
    total_area = sum(lib[k]["area"] * x[(k, pa)].X
                     for k in K for pa in anchors[k])
    total_power = sum(lib[k]["p_peak"] * x[(k, pa)].X
                      for k in K for pa in anchors[k])
    selected_type_counts = {
        k: sum(1 for allocation in site_alloc.values()
               if allocation["type"] == k)
        for k in K}

    solver_task_schedule = {
        v: {"start": float(s[v].X), "finish": float(f[v].X)} for v in V}
    selected_order_states = []
    selected_same_site_orders = []
    for (u, v), states in sorted(order_states.items()):
        names = ("lower-site", "higher-site", "u-before-v", "v-before-u")
        chosen = max(range(4), key=lambda index: states[index].X)
        if chosen == 2:
            selected_same_site_orders.append((u, v))
        elif chosen == 3:
            selected_same_site_orders.append((v, u))
        selected_order_states.append({
            "u": u, "v": v, "state": names[chosen],
            "values": [float(state.X) for state in states],
        })

    # Lower-bounded schedule variables may contain harmless slack when the
    # replay protocol dominates T.  Reconstruct the earliest feasible schedule
    # for the selected placement and selected C6b disjunctions, rather than
    # serializing that slack as an analytical result.
    schedule_successors = {task: [] for task in V}
    schedule_indegree = {task: 0 for task in V}
    for edge in edges:
        first_site = task_site[edge["src"]]
        second_site = task_site[edge["dst"]]
        d_h, d_v = hops(first_site, second_site)
        remote = first_site != second_site
        transfer = (
            ((t0 + float(edge["vol"]) / beta) if remote else 0.0) +
            tau_h * d_h + tau_v * d_v)
        schedule_successors[edge["src"]].append(
            (edge["dst"], transfer))
        schedule_indegree[edge["dst"]] += 1
    for first, second in selected_same_site_orders:
        schedule_successors[first].append((second, 0.0))
        schedule_indegree[second] += 1
    schedule_ready = sorted(
        [task for task in V if schedule_indegree[task] == 0], key=V.index)
    selected_schedule_order = []
    while schedule_ready:
        task = schedule_ready.pop(0)
        selected_schedule_order.append(task)
        for target, _ in schedule_successors[task]:
            schedule_indegree[target] -= 1
            if schedule_indegree[target] == 0:
                schedule_ready.append(target)
                schedule_ready.sort(key=V.index)
    if len(selected_schedule_order) != len(V):
        raise RuntimeError("selected task/order precedence graph is cyclic")
    schedule_predecessors = {task: [] for task in V}
    for source, successors_value in schedule_successors.items():
        for target, delay in successors_value:
            schedule_predecessors[target].append((source, delay))
    task_schedule = {}
    for task in selected_schedule_order:
        start_value = max(
            [0.0] + [task_schedule[source]["finish"] + delay
                     for source, delay in schedule_predecessors[task]])
        selected_kind = site_alloc[task_site[task]]["type"]
        finish_value = start_value + float(tasks[task]["cyc"][selected_kind])
        task_schedule[task] = {
            "start": start_value, "finish": finish_value}
    task_schedule_completion_value = max(
        schedule["finish"] for schedule in task_schedule.values())
    legacy_task_horizon_value = float(T_task.X)

    sparse_route_flow = []
    route_link_loads = []
    routed_payload_byte_hop = 0.0
    capacity_kib = (beta / flow_scale_bytes) * float(T.X)
    maximum_link_utilization = 0.0
    maximum_phase_link_utilization = 0.0
    phase_max_link_load_kib = {phase: 0.0 for phase in route_phases}
    for link_id, (source_site, target_site) in enumerate(directed_links):
        phase_load_values = {}
        for phase in route_phases:
            phase_load = sum(
                float(route_flow[(phase, source, link_id)].X)
                for source in route_sources[phase])
            phase_load_values[phase] = phase_load
            phase_max_link_load_kib[phase] = max(
                phase_max_link_load_kib[phase], phase_load)
            # The exact phase horizon is the busiest link's drain time.  A
            # lower-bounded auxiliary phase clock may contain slack because
            # ILP-Perf does not optimize diagnostics.
            phase_peak = phase_max_link_load_kib[phase]
            phase_utilization = (phase_load / phase_peak
                                 if phase_peak > 0.0 else 0.0)
            maximum_phase_link_utilization = max(
                maximum_phase_link_utilization, phase_utilization)
        load_value = sum(phase_load_values.values())
        routed_payload_byte_hop += load_value * flow_scale_bytes
        utilization = (load_value / capacity_kib
                        if capacity_kib > 0.0 else 0.0)
        maximum_link_utilization = max(maximum_link_utilization, utilization)
        route_link_loads.append({
            "id": link_id, "src": list(source_site),
            "dst": list(target_site), "axis": int(link_axis[link_id]),
            "load_KiB": load_value, "capacity_KiB": capacity_kib,
            "phase_load_KiB": phase_load_values,
            "utilization": utilization,
        })
        for phase in route_phases:
            for source in route_sources[phase]:
                value = float(route_flow[(phase, source, link_id)].X)
                if value > 1.0e-9:
                    sparse_route_flow.append({
                        "phase": phase, "source": source,
                        "link_id": link_id, "flow_KiB": value})

    maximum_flow_residual = 0.0
    maximum_shortest_path_residual = 0.0
    for phase in route_phases:
        for source in route_sources[phase]:
            total = sum(scaled_demands[phase][source].values())
            for node in P:
                out_value = sum(
                    float(route_flow[(phase, source, link_id)].X)
                    for link_id in outgoing[node])
                in_value = sum(
                    float(route_flow[(phase, source, link_id)].X)
                    for link_id in incoming[node])
                expected = (total if task_site[source] == node else 0.0)
                expected -= sum(
                    volume for target, volume in
                    scaled_demands[phase][source].items()
                    if task_site[target] == node)
                maximum_flow_residual = max(
                    maximum_flow_residual,
                    abs(out_value - in_value - expected))

            actual_shortest_kib_hops = 0.0
            for target, volume in scaled_demands[phase][source].items():
                dh, dv = hops(task_site[source], task_site[target])
                actual_shortest_kib_hops += volume * (dh + dv)
            routed_source_kib_hops = sum(
                float(route_flow[(phase, source, link_id)].X)
                for link_id in range(len(directed_links)))
            maximum_shortest_path_residual = max(
                maximum_shortest_path_residual,
                abs(routed_source_kib_hops - actual_shortest_kib_hops))

    # Report exact lower-bound values instead of raw auxiliary-variable X
    # values.  Those variables are only lower bounded and are diagnostic in
    # ILP-Perf, so their solver values may contain harmless slack.  Converting
    # every LinExpr here also keeps the result JSON serializable.
    phase_completion_values = {}
    for phase in route_phases:
        startup = (dispatch_sync_overhead
                   if noc_traffic_kind == "deterministic-dispatch" else t0)
        source_endpoint_values = {}
        destination_endpoint_values = {}
        for source in route_sources[phase]:
            source_value = 0.0
            for target, volume in traffic_by_phase[phase][source].items():
                dh, dv = hops(task_site[source], task_site[target])
                message_value = (startup + float(volume) / beta +
                                 tau_h * dh + tau_v * dv)
                source_value += message_value
                destination_endpoint_values[target] = (
                    destination_endpoint_values.get(target, 0.0) +
                    message_value)
            source_endpoint_values[source] = source_value
        evaluated_endpoint_bounds = {
            "source": source_endpoint_values,
            "destination": destination_endpoint_values,
        }
        endpoint_value = max(
            [0.0] + [value
                     for bounds in evaluated_endpoint_bounds.values()
                     for value in bounds.values()])
        link_value = ((flow_scale_bytes / beta) *
                      phase_max_link_load_kib[phase])
        phase_completion_values[phase] = {
            "link_cycles": link_value,
            "endpoint_cycles": endpoint_value,
            "completion_cycles": max(link_value, endpoint_value),
            "max_link_load_KiB": phase_max_link_load_kib[phase],
            "endpoint_lower_bounds_cycles": evaluated_endpoint_bounds,
        }
    public_T_value = float(T.X)
    noc_completion_value = sum(
        values["completion_cycles"]
        for values in phase_completion_values.values())
    protocol_makespan_value = (
        legacy_task_horizon_value + noc_completion_value)
    protocol_extraction_kind = (
        "exact-coordinate task-DAG plus phase lower bounds")
    protocol_worker_finish_values = {
        expert: expression_value(expression)
        for expert, expression in protocol_worker_finish_expr.items()}
    protocol_compute_schedule_values = {
        expert: {
            "start": float(protocol_compute_start[expert].X),
            "finish": float(protocol_compute_finish[expert].X),
            "resource_site": list(task_site[expert]),
        }
        for expert in protocol_compute_start}
    if noc_traffic_kind == "deterministic-dispatch":
        cpu_sites = [anchor for anchor, allocation in site_alloc.items()
                     if allocation["type"] == "CPU"]
        if len(cpu_sites) != 1:
            raise RuntimeError(
                "exact protocol extraction requires one selected CPU")
        selected_compute_cycles = []
        final_expert_sites = []
        for expert in dispatch_expert_tasks:
            if expert not in task_site:
                raise RuntimeError(
                    "exact protocol extraction missed expert %s" % expert)
            expert_site = task_site[expert]
            if expert_site not in site_alloc:
                raise RuntimeError(
                    "expert %s is not assigned to a selected chiplet" %
                    expert)
            expert_kind = site_alloc[expert_site]["type"]
            try:
                compute_cycles = protocol_compute_cycles[expert][expert_kind]
            except KeyError:
                raise RuntimeError(
                    "no protocol compute profile for %s on %s" %
                    (expert, expert_kind))
            final_expert_sites.append(expert_site)
            selected_compute_cycles.append(compute_cycles)
        protocol_compute_order = sorted(
            range(len(dispatch_expert_tasks)),
            key=lambda group: (
                float(protocol_compute_start[
                    dispatch_expert_tasks[group]].X), group))
        exact_protocol = _evaluate_fixed_order_protocol(
            cpu_sites[0], final_expert_sites, dispatch_matrix,
            selected_compute_cycles, dispatch_token_bytes_per_phase,
            init_bytes, done_bytes, dispatch_sync_overhead,
            protocol_flit_bytes, protocol_hop_cycles,
            expert_resource_ids=final_expert_sites,
            compute_order=protocol_compute_order)
        protocol_makespan_value = exact_protocol["completion_cycles"]
        protocol_worker_finish_values = {
            expert: exact_protocol["worker_finish_cycles"][group]
            for group, expert in enumerate(dispatch_expert_tasks)}
        protocol_extraction_kind = (
            "exact-coordinate fixed-order max-plus recurrence")

    # The public operating metric is the sole MILP objective T.  For the main
    # deterministic-dispatch path it is the replay-aligned max-plus completion
    # epigraph, additionally lower bounded by aggregate Mesh link capacity.
    performance_proxy_value = public_T_value
    performance_extraction_kind = "MILP makespan variable T"
    performance_metric_kind = (
        "replay-aligned protocol makespan T with exact co-resident compute "
        "ordering and directed-link capacity")

    cpu_control_byte_hop = expression_value(B_cpu_control_proxy)
    routed_byte_hop = routed_payload_byte_hop + cpu_control_byte_hop
    if constructive_perf_start.get("accepted"):
        constructive_perf_start["final_protocol_cycles"] = (
            protocol_makespan_value)
        constructive_perf_start["final_energy_J"] = e_noc_proxy_val
        constructive_perf_start["candidate_retained_or_improved"] = (
            protocol_makespan_value <=
            constructive_perf_start["candidate_protocol_cycles"] + 1.0e-6)
    if constructive_power_start.get("accepted"):
        constructive_power_start["final_protocol_cycles"] = (
            protocol_makespan_value)
        constructive_power_start["final_energy_J"] = e_noc_proxy_val
        constructive_power_start["candidate_retained_or_improved"] = (
            e_noc_proxy_val <=
            constructive_power_start["candidate_energy_J"] + max(
                1.0e-15,
                constructive_power_start["candidate_energy_J"] * 1.0e-8))
        constructive_power_start["final_within_deadline"] = (
            protocol_makespan_value <=
            constructive_power_start["deadline_cycles"] + 1.0e-6)

    sites_out = []
    for pa, info in sorted(site_alloc.items(), key=lambda kv: idx[kv[0]]):
        l, i, j = pa
        sites_out.append({"l": l, "i": i, "j": j, "type": info["type"],
                          "tasks": info["tasks"]})
    final_bundle_signature = bundle_signature(sites_out)
    if internal_bundle_refinement["enabled"]:
        internal_bundle_refinement["final_bundle_signature"] = (
            final_bundle_signature)
        internal_bundle_refinement["signature_preserved"] = (
            final_bundle_signature ==
            internal_bundle_refinement["bundle_signature"])
        if not internal_bundle_refinement["signature_preserved"]:
            raise RuntimeError(
                "final ILP-Perf solution escaped the Stage 0 bundle space")
    else:
        internal_bundle_refinement["final_bundle_signature"] = None
        internal_bundle_refinement["signature_preserved"] = None

    def _safe_gap():
        if primary_gap is not None:
            return primary_gap
        for attr in ("MIPGap",):
            try:
                value = float(getattr(m, attr))
                if math.isfinite(value) and value >= 0.0:
                    return value
            except Exception:
                pass
        # multi-objective: derive from best bound of the top objective if any
        try:
            ob = m.ObjBound
            ov = float(T.X)
            value = abs(ov - ob) / max(abs(ov), 1e-9)
            return value if math.isfinite(value) else None
        except Exception:
            return None
    gap_val = _safe_gap()

    noc_clock_hz = solver_opts.get("noc_clock_hz")
    if noc_clock_hz is not None:
        noc_clock_hz = float(noc_clock_hz)
        if noc_clock_hz <= 0.0:
            raise ValueError("noc_clock_hz must be positive")
    noc_power_proxy_W = None
    if (noc_clock_hz is not None and protocol_makespan_value > 0.0):
        # Protocol-duration-normalized dynamic NoC power.  This is an
        # analytical proxy, not PopNet/ORION total power
        # (static/router/queue activity excluded).
        noc_power_proxy_W = (e_noc_proxy_val * noc_clock_hz /
                             protocol_makespan_value)

    mode_label = "ILP-Perf"
    reference_kind = ("placement-only-bundles" if placement_bundles is not None
                      else ("same-BOM" if bom_counts is not None else None))
    serialized_reference = solver_opts.get("performance_reference_solution")
    serialized_reference_bundle_signature = (
        serialized_reference.get("bundle_signature")
        if isinstance(serialized_reference, dict) else None)

    sol = {
        "sites": sites_out,
        "task_site": {v: list(p) for v, p in task_site.items()},
        # The public performance fields all report the sole joint objective T.
        "predicted_makespan_cycles": performance_proxy_value,
        "performance_proxy_cycles": performance_proxy_value,
        "tightened_performance_proxy_cycles": performance_proxy_value,
        "protocol_makespan_cycles": protocol_makespan_value,
        "performance_extraction_kind": performance_extraction_kind,
        "performance_metric_kind": performance_metric_kind,
        "analytical_task_makespan_cycles": task_schedule_completion_value,
        "legacy_task_horizon_cycles": task_schedule_completion_value,
        "legacy_task_horizon_variable_cycles": legacy_task_horizon_value,
        "analytical_noc_phase_bound_cycles": noc_completion_value,
        "mip_gap": gap_val,
        "solver_wall_s": wall,
        "solver_status": int(m.Status),
        "optimization_mode": mode,
        "mode_label": mode_label,
        "certified_objective_scale_cycles": certified_T_ref,
        "model_build_s": model_build_s,
        "formulation_build_s": formulation_build_s,
        "stage0_total_s": stage0_total_s,
        "primary_wall_s": primary_wall,
        "requested_total_time_limit_s": requested_total_time_limit,
        "primary_solver_time_limit_s": primary_time_limit,
        "constructive_search_wall_s": constructive_wall,
        "primary_status": primary_status,
        "primary_mip_gap": primary_gap,
        "primary_has_incumbent": primary_incumbent is not None,
        "primary_gap_certificate": primary_gap_certificate,
        "primary_incumbent_cycles": primary_incumbent,
        "primary_objective_bound_cycles": primary_bound,
        "primary_node_count": primary_nodes,
        "primary_objective": "minimize joint MILP makespan T",
        "byte_hop_tiebreak": byte_hop_tiebreak_record,
        "objective_contract": {
            "active_objective_count": 1,
            "active_objective": "min T",
            "joint_selection_placement_scheduling": True,
            "chiplet_count_definition": "N_c=sum_p x[c,p]",
            "gpu_type_counts_fixed": bom_counts is not None,
            "protocol_completion_in_T": True,
            "protocol_used_as_separate_objective": False,
            "legacy_task_schedule_role": (
                "hard-feasibility-only"
                if noc_traffic_kind == "deterministic-dispatch" else
                "included-in-task-DAG-fallback-T"),
            "T_definition": (
                "fixed-order replay max-plus completion plus Mesh capacity "
                "epigraph"
                if noc_traffic_kind == "deterministic-dispatch" else
                "legacy task horizon plus phase NoC completion bound"),
            "energy_or_byte_hop_tie_break": False,
        },
        "solver_configuration": {
            "gurobi_mip_focus": mip_focus,
            "mip_focus_meaning": (
                "1: prioritize finding and improving feasible incumbents"
                if mip_focus == 1 else
                "explicit Gurobi MIPFocus override"),
            "changes_objective": False,
        },
        "compute_selection_reference_cycles": None,
        "compute_selection_relaxation": compute_selection_relaxation,
        "compute_selection_cap_cycles": compute_selection_cap,
        "secondary_wall_s": secondary_wall,
        "secondary_time_limit_s": secondary_time_limit,
        "secondary_status": secondary_status,
        "secondary_mip_gap": secondary_gap,
        "secondary_has_incumbent": secondary_has_incumbent,
        "secondary_gap_certificate": secondary_gap_certificate,
        "secondary_objective": secondary_objective,
        "secondary_objective_value": secondary_incumbent,
        "tertiary_wall_s": tertiary_wall,
        "tertiary_time_limit_s": tertiary_time_limit,
        "tertiary_status": tertiary_status,
        "tertiary_mip_gap": tertiary_gap,
        "tertiary_has_incumbent": tertiary_has_incumbent,
        "tertiary_gap_certificate": tertiary_gap_certificate,
        "tertiary_objective": tertiary_objective,
        "tertiary_objective_value": tertiary_incumbent,
        "quaternary_wall_s": quaternary_wall,
        "quaternary_status": quaternary_status,
        "quaternary_mip_gap": quaternary_gap,
        "quaternary_has_incumbent": quaternary_has_incumbent,
        "quaternary_gap_certificate": quaternary_gap_certificate,
        "quaternary_objective": quaternary_objective,
        "quaternary_objective_value": quaternary_incumbent,
        "performance_reference_cycles": performance_reference,
        "performance_reference_kind": performance_reference_kind,
        "performance_relaxation": performance_relaxation,
        "performance_cap_cycles": performance_cap,
        "performance_cap_enforced": performance_cap_enforced,
        "time_to_first_feasible_s": (
            model_build_s + (
                anytime_trace[0]["wall_s"]
                if anytime_trace else primary_wall)),
        "anytime_trace": anytime_trace,
        "energy_comm_J": e_comm_val,
        "energy_comp_J": e_comp_val,
        "energy_total_J": e_comm_val + e_comp_val,
        "time_horizon_cycles": time_horizon,
        "task_schedule": task_schedule,
        "solver_task_schedule_witness": solver_task_schedule,
        "conditional_order_states": selected_order_states,
        "total_area_mm2": total_area,
        "peak_power_W": total_power,
        "selected_type_counts": selected_type_counts,
        "resource_selection": {
            "count_definition": "N_c=sum_p x[c,p]",
            "joint_with_mapping_and_placement": True,
            "fixed_bom": bom_counts is not None,
            "fixed_complete_bundles": placement_bundles is not None,
            "selected_counts": selected_type_counts,
            "area_budget_mm2": float(budgets["a_max"]),
            "area_used_mm2": total_area,
            "area_slack_mm2": float(budgets["a_max"]) - total_area,
            "peak_power_budget_W": float(budgets["p_max"]),
            "peak_power_used_W": total_power,
            "peak_power_slack_W": float(budgets["p_max"]) - total_power,
        },
        "noc_power_proxy": {
            "objective_kind": noc_objective_kind,
            "diagnostic_only": True,
            "used_as_objective": False,
            "traffic_kind": noc_traffic_kind,
            "energy_J": e_noc_proxy_val,
            "task_dag_energy_J": e_comm_val,
            "dispatch_tokens_per_source": (
                solver_opts.get("dispatch_tokens_per_source", 32)
                if dispatch_matrix is not None else None),
            "dispatch_token_bytes_two_phases": dispatch_token_bytes,
            "dispatch_token_bytes_per_phase": (
                dispatch_token_bytes_per_phase),
            "dispatch_assignment_hops": dispatch_assignment_hops,
            "dispatch_byte_hop": dispatch_byte_hop,
            "payload_routed_byte_hop": routed_payload_byte_hop,
            "cpu_control_byte_hop": cpu_control_byte_hop,
            "full_protocol_routed_byte_hop": routed_byte_hop,
            "cpu_control_energy_included": (
                noc_traffic_kind == "deterministic-dispatch"),
            "deadline_cycles": performance_cap,
            "normalization_cycles": protocol_makespan_value,
            "noc_clock_hz": noc_clock_hz,
            "deadline_normalized_dynamic_power_W": (
                noc_power_proxy_W if performance_cap is not None else None),
            "protocol_normalized_dynamic_power_W": noc_power_proxy_W,
            "phase_completion": phase_completion_values,
            "excludes": ["NoC static power", "router/buffer activity",
                          "queueing beyond the horizon-volume capacity bound",
                          "CPU/GPU power"],
            "measured_orion_power": False,
        },
        "routing": {
            "kind": "phase-separated-source-aggregated-splittable-packet-flow",
            "flow_scale_bytes": flow_scale_bytes,
            "directed_full_duplex": True,
            "capacity_horizon": "T",
            "phases": route_phases,
            "source_count": sum(len(sources)
                                for sources in route_sources.values()),
            "directed_link_count": len(directed_links),
            "sparse_flow": sparse_route_flow,
            "link_loads": route_link_loads,
            "maximum_link_utilization": maximum_link_utilization,
            "maximum_phase_link_utilization": (
                maximum_phase_link_utilization),
            "maximum_flow_residual_KiB": maximum_flow_residual,
            "maximum_shortest_path_residual_KiB_hops": (
                maximum_shortest_path_residual),
            "payload_routed_byte_hop": routed_payload_byte_hop,
            "cpu_control_byte_hop": cpu_control_byte_hop,
            "routed_byte_hop": routed_byte_hop,
        },
        "protocol_model": {
            "kind": ("fixed-order-max-plus-event-DAG"
                     if noc_traffic_kind == "deterministic-dispatch"
                     else "task-DAG-plus-phase-bound"),
            "event_count": protocol_event_count,
            "flit_bytes": protocol_flit_bytes,
            "hop_cycles": protocol_hop_cycles,
            "sync_overhead_cycles": dispatch_sync_overhead,
            "compute_profile": protocol_compute_profile,
            "compute_cycles_by_expert_type": protocol_compute_cycles,
            "completion_extraction": protocol_extraction_kind,
            "completion_cycles": protocol_makespan_value,
            "diagnostic_only": False,
            "used_as_objective": True,
            "included_in_T": True,
            "used_as_separate_objective": False,
            "co_resident_compute_nonoverlap": (
                noc_traffic_kind == "deterministic-dispatch"),
            "compute_schedule": protocol_compute_schedule_values,
            "worker_finish_cycles": protocol_worker_finish_values,
            "phase_completion": phase_completion_values,
        },
        "bundle_signature": final_bundle_signature,
        "stage0_feasible_start": stage0_feasible_start,
        "internal_bundle_refinement": internal_bundle_refinement,
        "constructive_perf_start": constructive_perf_start,
        "cpu_anchor_policy": cpu_anchor_policy,
        "expert_slots": expert_slots,
        "reference_coordinates_warm_started": (
            solver_opts.get("warm_start_solution") is not None),
        "comparison_reference": {
            "kind": reference_kind,
            "source": solver_opts.get("reference_source"),
            "fixed_bom": bom_counts,
            "fixed_complete_bundles": placement_bundles is not None,
            "source_bundle_signature": (
                serialized_reference_bundle_signature),
            "final_bundle_signature": final_bundle_signature,
            "bundle_signature_preserved": (
                serialized_reference_bundle_signature ==
                final_bundle_signature
                if serialized_reference_bundle_signature is not None
                else None),
        },
        "byte_hop": byte_hop,
        "grid": {"L": L, "X": X, "Y": Y},
        "budgets": budgets,
    }
    stats = {
        "status": int(m.Status),
        "wall": wall,
        "model_build_s": model_build_s,
        "feasible": True,
        "obj": float(T.X),
        "gap": gap_val,
        "primary_bound_cycles": primary_bound,
        "anytime_trace": anytime_trace,
        "num_vars": int(m.NumVars),
        "num_bin_vars": int(m.NumBinVars),
        "num_constrs": int(m.NumConstrs),
        "num_genconstrs": int(m.NumGenConstrs),
    }
    return sol, stats


def main():
    ap = argparse.ArgumentParser(description="Compact 3D chiplet layout ILP")
    ap.add_argument("--params", required=True)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--X", type=int, default=4)
    ap.add_argument("--Y", type=int, default=4)
    ap.add_argument("--a-max", type=float, default=2500.0)
    ap.add_argument("--a-max-layer", type=float, default=None)
    ap.add_argument("--p-max", type=float, default=1200.0)
    # Compatibility-only parser shim for archived commands.  It is never put
    # into ``budgets`` and therefore cannot create the removed C9 constraint.
    # Emit an explicit warning below instead of silently accepting it.
    ap.add_argument("--p-col", type=float, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--n-gpu-l-max", type=int, default=None)
    ap.add_argument("--n-cpu", type=int, default=1,
                    help="required freely placed CPU count (default: 1)")
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--mipgap", type=float, default=0.005)
    ap.add_argument("--mip-focus", type=int, choices=[0, 1, 2, 3], default=1,
                    help="Gurobi search emphasis (default: 1, prioritize "
                         "finding/improving feasible min-T incumbents); this "
                         "does not change the objective")
    ap.add_argument("--mode",
                    choices=["ilp-perf"], default="ilp-perf",
                    help="single-objective joint MILP: minimize makespan T "
                         "under exact scheduling, Mesh link-capacity, area, "
                         "and package peak-power constraints")
    ap.add_argument("--noc-clock-hz", type=float, default=None,
                    help="optional explicit NoC clock for converting the "
                         "protocol-normalized energy proxy to watts")
    ap.add_argument("--noc-traffic",
                    choices=["deterministic-dispatch", "task-dag"],
                    default="deterministic-dispatch",
                    help="performance traffic model: replay-aligned fixed-order "
                         "protocol (default), or the legacy task-DAG plus phase "
                         "completion fallback")
    ap.add_argument("--dispatch-tokens-per-source", type=int, default=32,
                    help="tokens/source used by the deterministic dispatch "
                         "proxy; must match subsequent codegen/replay")
    ap.add_argument("--dispatch-hidden", type=int, default=128,
                    help="hidden width used to scope dispatch proxy bytes")
    ap.add_argument("--dispatch-bytes-elem", type=int, default=None,
                    help="bytes/element for dispatch proxy; default is "
                         "params.meta.bytes_elem")
    reference = ap.add_mutually_exclusive_group()
    reference.add_argument("--bom-from", default=None,
                           help="fix exact chiplet-type counts from a reference "
                                "solution; task mapping remains free")
    reference.add_argument("--placement-from", default=None,
                           help="placement-only mode: fix the reference BOM and "
                                "every complete chiplet task bundle")
    ap.add_argument("--cpu-anchor", choices=["free"], default="free",
                    help="CPU placement policy (the active model is free)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None,
                    help="explicit Gurobi thread count (paper runs use 1)")
    ap.add_argument("--stage0-protocol-evaluations", type=int, default=None,
                    help="override the deterministic exact-protocol Stage-0 "
                         "neighbor-evaluation cap")
    ap.add_argument("--stage0-protocol-wall-s", type=float, default=None,
                    help="override the deterministic exact-protocol Stage-0 "
                         "wall-time cap in seconds")
    ap.add_argument("--stage0-protocol-rounds", type=int, default=None,
                    help="override the deterministic exact-protocol Stage-0 "
                         "best-improvement round cap")
    ap.add_argument("--no-symmetry-break", action="store_true",
                    help="disable Gurobi's internal symmetry detection")
    ap.add_argument("-o", "--out", default="solution.json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.p_col is not None:
        sys.stderr.write(
            "WARNING: --p-col/C9 is retired and ignored; only the package "
            "peak-power budget --p-max is enforced.\n")

    with open(a.params) as fh:
        params = json.load(fh)

    reference_solution = None
    reference_path = a.placement_from or a.bom_from
    if reference_path is not None:
        with open(reference_path) as fh:
            reference_solution = json.load(fh)
    fixed_bom = (reference_bom(reference_solution, params["library"])
                 if reference_solution is not None else None)
    fixed_bundles = (reference_bundles(reference_solution, params["tasks"],
                                       params["library"])
                     if a.placement_from is not None else None)
    if a.dispatch_tokens_per_source <= 0 or a.dispatch_hidden <= 0:
        ap.error("dispatch tokens/source and hidden width must be positive")
    if a.dispatch_bytes_elem is not None and a.dispatch_bytes_elem <= 0:
        ap.error("--dispatch-bytes-elem must be positive")
    if a.threads is not None and a.threads <= 0:
        ap.error("--threads must be positive")
    if (a.stage0_protocol_evaluations is not None and
            a.stage0_protocol_evaluations < 0):
        ap.error("--stage0-protocol-evaluations must be nonnegative")
    if (a.stage0_protocol_wall_s is not None and
            a.stage0_protocol_wall_s < 0.0):
        ap.error("--stage0-protocol-wall-s must be nonnegative")
    if (a.stage0_protocol_rounds is not None and
            a.stage0_protocol_rounds < 0):
        ap.error("--stage0-protocol-rounds must be nonnegative")
    if a.n_cpu < 0:
        ap.error("--n-cpu must be nonnegative")
    budgets = {
        "a_max": a.a_max, "a_max_layer": a.a_max_layer,
        "p_max": a.p_max,
        "nmax": {}, "nmin": {},
    }
    if a.n_gpu_l_max is not None:
        budgets["nmax"]["GPU-L"] = a.n_gpu_l_max
    if "CPU" in params.get("library", {}):
        budgets["nmin"]["CPU"] = a.n_cpu
        budgets["nmax"]["CPU"] = a.n_cpu

    solver_opts = {"time_limit": a.time_limit, "mipgap": a.mipgap,
                   "mipfocus": a.mip_focus,
                   "seed": a.seed, "symmetry_break": not a.no_symmetry_break,
                   "mode": a.mode,
                   "noc_clock_hz": a.noc_clock_hz,
                   "noc_traffic": a.noc_traffic,
                   "dispatch_tokens_per_source": a.dispatch_tokens_per_source,
                   "dispatch_hidden": a.dispatch_hidden,
                   "dispatch_bytes_elem": (
                       a.dispatch_bytes_elem
                       if a.dispatch_bytes_elem is not None
                       else params.get("meta", {}).get("bytes_elem", 2)),
                   "threads": a.threads,
                   "stage0_protocol_evaluations": (
                       a.stage0_protocol_evaluations
                       if a.stage0_protocol_evaluations is not None else
                       None),
                   "stage0_protocol_wall_s": (
                       a.stage0_protocol_wall_s
                       if a.stage0_protocol_wall_s is not None else None),
                   "stage0_protocol_rounds": (
                       a.stage0_protocol_rounds
                       if a.stage0_protocol_rounds is not None else None),
                   "cpu_anchor": a.cpu_anchor,
                   "bom_counts": fixed_bom,
                   "placement_bundles": fixed_bundles,
                   "warm_start_solution": (
                       reference_solution
                       if a.placement_from is not None else None),
                   # Retained only as inert provenance for archived inputs.
                   "performance_reference_solution": reference_solution,
                   "reference_source": reference_path}

    sol, stats = build_and_solve(params, a.L, a.X, a.Y, budgets, solver_opts,
                                 verbose=not a.quiet)
    if sol is None:
        print("[ilp] INFEASIBLE or no solution. status=%d total=%.2fs "
              "build=%.2fs formulation=%.2fs stage0=%.2fs "
              "refinement=%.2fs primary=%.2fs allowance=%.2fs"
              % (stats["status"], stats["wall"],
                 stats.get("model_build_s", 0.0),
                 stats.get("formulation_build_s", 0.0),
                 stats.get("stage0_total_s", 0.0),
                 stats.get("stage0_protocol_refinement_wall_s", 0.0),
                 max(0.0, stats["wall"] -
                     stats.get("model_build_s", 0.0)),
                 stats.get("primary_solver_time_limit_s", 0.0)))
        sys.exit(1)

    with open(a.out, "w") as fh:
        # RFC 8259 JSON has no NaN/Infinity literals.  Solver stages without a
        # finite bound serialize gap=None plus an explicit certificate reason.
        json.dump(sol, fh, indent=2, allow_nan=False)
    print("\n[ilp] ==== SOLUTION ====")
    print("[ilp] makespan (pred cycles) = %.1f" % sol["predicted_makespan_cycles"])
    gap_text = ("%.4f%%" % (sol["mip_gap"] * 100)
                if sol.get("mip_gap") is not None else "uncertified")
    print("[ilp] MIP gap = %s   wall = %.2fs   status=%d"
          % (gap_text, sol["solver_wall_s"], sol["solver_status"]))
    bound_text = _format_optional_float(
        sol.get("primary_objective_bound_cycles"))
    print("[ilp] min-T bound = %s cycles  primary wall = %.2fs"
          % (bound_text, sol.get("primary_wall_s", 0.0)))
    print("[ilp] ILP-Perf objective = min replay-aligned T; exact protocol "
          "completion = %.1f cycles" % sol["protocol_makespan_cycles"])
    print("[ilp] area=%.0f mm2  peak_power=%.0f W  energy=%.3e J  byte_hop=%.3e"
          % (sol["total_area_mm2"], sol["peak_power_W"],
             sol["energy_total_J"], sol["byte_hop"]))
    print("[ilp] occupied sites: %d" % len(sol["sites"]))
    for st in sol["sites"]:
        print("      (l=%d,i=%d,j=%d) %-6s  %s"
              % (st["l"], st["i"], st["j"], st["type"], ",".join(st["tasks"])))
    print("[ilp] wrote %s" % a.out)


if __name__ == "__main__":
    main()
