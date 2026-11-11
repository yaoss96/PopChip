#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resource-matched random-layout baseline for an ILP solution.

The baseline changes placement, not provisioning:

* every chiplet instance in the ILP solution is retained (including idle or
  dedicated CPU/router chiplets);
* chiplets are packed at random legal anchors with no footprint overlap;
* every non-expert task stays in its original chiplet bundle; and
* expert tasks are randomly permuted among the originally expert-bearing
  chiplet bundles.

Every emitted solution is checked with evaluate.evaluate(..., strict=True)
under the exact budgets copied from the ILP solution.  That legacy task-DAG
schedule is a hard-feasibility witness only.  Candidate ranking uses the same
public performance metric as the current ILP-Perf experiment:

    T = max(fixed-order protocol completion,
            directed-Mesh link-capacity epigraph).

The script uses only the Python standard library plus the repository's
stdlib-only evaluators and is compatible with Python 3.7.
"""

from __future__ import print_function

import argparse
import collections
import copy
import json
import os
import random
import shutil
import sys

import evaluate as ev


_FLOAT_TOL = 1.0e-6
_MAX_PACK_NODES = 250000
_MAX_PACK_RESTARTS = 24
_MAX_LAYOUT_ATTEMPTS = 128
_MAX_MATCH_NODES = 500000

_PUBLIC_T_METRIC = "public_T_cycles"
_PUBLIC_T_DEFINITION = (
    "T = max(fixed-order protocol completion, directed-Mesh capacity bound)")
_PUBLIC_T_IMPLEMENTATION = (
    "run_metaheuristic_surrogate_audit.CaseModel.evaluate_candidate")
_REPRESENTATIVE_POLICY = (
    "nearest observed candidate to the mathematical median public T; "
    "ties use lower public T, then lower seed")


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json_atomic(path, obj):
    """Write deterministic JSON without leaving a partially written file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def parse_seeds(values):
    """Parse ``--seeds 0 1 2`` and ``--seeds 0,1,2`` forms."""
    seeds = []
    for value in values:
        for item in value.replace(",", " ").split():
            try:
                seeds.append(int(item, 10))
            except ValueError:
                raise ValueError("invalid seed %r (expected an integer)" % item)
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicates; seed filenames must be unique")
    return seeds


def _as_anchor(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("%s must be a three-element [l, i, j] coordinate" % label)
    try:
        return (int(value[0]), int(value[1]), int(value[2]))
    except (TypeError, ValueError):
        raise ValueError("%s contains a non-integer coordinate" % label)


def _grid_tuple(grid):
    try:
        dims = (int(grid["L"]), int(grid["X"]), int(grid["Y"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("solution.grid must contain integer L, X, and Y")
    if min(dims) <= 0:
        raise ValueError("solution.grid dimensions must all be positive")
    return dims


def _footprint(chiplet_type, anchor, library):
    spec = library[chiplet_type]
    l, i, j = anchor
    w, h = int(spec["w"]), int(spec["h"])
    return tuple((l, i + di, j + dj)
                 for di in range(w) for dj in range(h))


def _legal_anchors(chiplet_type, grid, library):
    L, X, Y = _grid_tuple(grid)
    spec = library[chiplet_type]
    w, h = int(spec["w"]), int(spec["h"])
    if w <= 0 or h <= 0:
        raise ValueError("chiplet %s has a non-positive footprint" % chiplet_type)
    if w > X or h > Y:
        return []
    return [(l, i, j)
            for l in range(L)
            for i in range(X - w + 1)
            for j in range(Y - h + 1)]


def _type_counts(site_types):
    counts = collections.Counter(site_types.values())
    return {name: counts[name] for name in sorted(counts)}


def _is_cpu_type(name, spec):
    upper_name = str(name).upper()
    simlet = str(spec.get("simlet", "")).lower()
    return "CPU" in upper_name or "sniper" in simlet


def _cpu_counts(site_types, library):
    counts = collections.Counter()
    for chiplet_type in site_types.values():
        if _is_cpu_type(chiplet_type, library[chiplet_type]):
            counts[chiplet_type] += 1
    return {name: counts[name] for name in sorted(counts)}


def _budget_reasons(params, site_types, score, budgets):
    """Check ILP budgets that evaluate.py does not currently implement."""
    reasons = []
    library = params["library"]
    counts = collections.Counter(site_types.values())

    nmin = budgets.get("nmin") or {}
    nmax = budgets.get("nmax") or {}
    for chiplet_type, minimum in sorted(nmin.items()):
        if counts.get(chiplet_type, 0) < minimum:
            reasons.append("count %s=%d < nmin %s" %
                           (chiplet_type, counts.get(chiplet_type, 0), minimum))
    for chiplet_type, maximum in sorted(nmax.items()):
        if counts.get(chiplet_type, 0) > maximum:
            reasons.append("count %s=%d > nmax %s" %
                           (chiplet_type, counts.get(chiplet_type, 0), maximum))

    layer_cap = budgets.get("a_max_layer")
    if layer_cap is not None:
        layer_area = {}
        for anchor, chiplet_type in site_types.items():
            layer_area[anchor[0]] = (layer_area.get(anchor[0], 0.0)
                                     + float(library[chiplet_type]["area"]))
        for layer, area in sorted(layer_area.items()):
            if area > float(layer_cap) + _FLOAT_TOL:
                reasons.append("layer %d area %.6g > a_max_layer %.6g" %
                               (layer, area, float(layer_cap)))

    energy_cap = budgets.get("e_bar")
    if energy_cap is not None and "energy_total_J" in score:
        if score["energy_total_J"] > float(energy_cap) + _FLOAT_TOL:
            reasons.append("energy %.9g > e_bar %.9g" %
                           (score["energy_total_J"], float(energy_cap)))
    return reasons


def _strict_score(params, task_site, site_types, grid, budgets):
    """Run the repository's canonical evaluator in strict mode."""
    score = ev.evaluate(params, task_site, site_types, grid,
                        strict=True, budgets=budgets)
    if not score.get("feasible"):
        return score, list(score.get("reasons", ["strict evaluation failed"]))
    extra = _budget_reasons(params, site_types, score, budgets)
    return score, extra


class _PublicTScorer(object):
    """Score fixed layouts with the current sole ILP-Perf objective.

    The implementation is deliberately shared with the SA/GA audit instead of
    reimplementing either the protocol recurrence or the directed-Mesh
    capacity epigraph here.  The source ILP artifact supplies the exact timing
    contract and deterministic-dispatch parameters for the case.
    """

    def __init__(self, params, source_solution, state):
        # Keep this import lazy.  The Floorplet/TAP wrapper reuses the strict
        # feasibility helpers in this module under Gurobi's minimal Python 3.7
        # runtime, whose standard library omits urllib and therefore cannot
        # import pathlib.  Matched-Random itself runs under the system Python
        # and is the only path that needs the shared public-T scorer.
        import run_metaheuristic_surrogate_audit as public_t_model
        self.model = public_t_model.CaseModel(
            "matched-random", params, source_solution,
            state["budgets"], state["grid"])
        self.source_evaluation = self.model.evaluate_candidate(
            self.model.source_state())

        # Fail closed if the shared scorer is not reproducing the public T in
        # the source artifact.  A small absolute/relative tolerance only
        # accommodates JSON/solver floating-point serialization.
        try:
            source_T = float(source_solution["predicted_makespan_cycles"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "source ILP solution lacks predicted_makespan_cycles")
        rescored_T = float(self.source_evaluation["T_cycles"])
        tolerance = max(1.0e-4,
                        1.0e-9 * max(abs(source_T), abs(rescored_T)))
        if abs(source_T - rescored_T) > tolerance:
            raise ValueError(
                "public-T scorer disagrees with source ILP: %.12g vs %.12g" %
                (rescored_T, source_T))

    def _state_from_solution(self, solution):
        """Translate serialized anonymous chiplets to CaseModel indices."""
        anchors_by_type = collections.defaultdict(list)
        for ordinal, site in enumerate(solution.get("sites") or []):
            try:
                chiplet_type = str(site["type"])
                anchor = (int(site["l"]), int(site["i"]), int(site["j"]))
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    "candidate sites[%d] is malformed" % ordinal)
            anchors_by_type[chiplet_type].append(anchor)
        for anchors in anchors_by_type.values():
            anchors.sort()

        expected = collections.Counter(self.model.types)
        actual = collections.Counter({
            chiplet_type: len(anchors)
            for chiplet_type, anchors in anchors_by_type.items()})
        if actual != expected:
            raise ValueError(
                "candidate BOM differs from public-T source: %s vs %s" %
                (dict(actual), dict(expected)))

        anchors = []
        next_by_type = collections.Counter()
        for chiplet_type in self.model.types:
            offset = next_by_type[chiplet_type]
            anchors.append(anchors_by_type[chiplet_type][offset])
            next_by_type[chiplet_type] += 1
        index_at = {anchor: index for index, anchor in enumerate(anchors)}
        if len(index_at) != len(anchors):
            raise ValueError("candidate has duplicate chiplet anchors")

        assignment = {}
        for task, raw_anchor in (solution.get("task_site") or {}).items():
            anchor = _as_anchor(raw_anchor, "task_site[%r]" % task)
            if anchor not in index_at:
                raise ValueError(
                    "candidate task %s is assigned to empty anchor %s" %
                    (task, anchor))
            assignment[str(task)] = index_at[anchor]
        return {
            "types": list(self.model.types),
            "anchors": anchors,
            "assign": assignment,
        }

    def evaluate(self, solution):
        return self.model.evaluate_candidate(
            self._state_from_solution(solution))

    @staticmethod
    def provenance(evaluation):
        return {
            "metric": _PUBLIC_T_METRIC,
            "definition": _PUBLIC_T_DEFINITION,
            "implementation": _PUBLIC_T_IMPLEMENTATION,
            "protocol_policy": evaluation["protocol"]["policy"],
            "directed_mesh_capacity_policy": evaluation["routing"]["policy"],
            "routing_relation_to_milp": evaluation["routing"][
                "milp_relation"],
            "active_objective": "min T",
            "protocol_completion_in_T": True,
            "directed_mesh_capacity_in_T": True,
            "legacy_task_makespan_role": "hard-feasibility-only",
            "legacy_task_makespan_scored": False,
            "energy_or_byte_hop_tie_break": False,
        }


def _apply_public_T_score(solution, public_score, legacy_score, provenance):
    """Replace legacy ranking fields with the sole public performance T."""
    public_T = float(public_score["T_cycles"])
    protocol_T = float(public_score["protocol"]["completion_cycles"])
    capacity_T = float(public_score["routing"]["capacity_bound_cycles"])
    task_T = float(public_score["schedule"]["makespan_cycles"])
    legacy_T = float(legacy_score["makespan_cycles"])

    solution["public_T_cycles"] = public_T
    solution["predicted_makespan_cycles"] = public_T
    solution["performance_proxy_cycles"] = public_T
    solution["makespan_cycles"] = public_T
    solution["evaluated_makespan_cycles"] = public_T
    solution["protocol_makespan_cycles"] = protocol_T
    solution["protocol_completion_cycles"] = protocol_T
    solution["directed_mesh_capacity_bound_cycles"] = capacity_T
    solution["hard_feasibility_task_makespan_cycles"] = task_T
    solution["legacy_evaluate_task_makespan_cycles"] = legacy_T
    solution["public_T_scoring"] = copy.deepcopy(provenance)
    baseline = solution.setdefault("baseline", {})
    baseline["representative_selection_metric"] = _PUBLIC_T_METRIC
    baseline["public_T_scoring"] = copy.deepcopy(provenance)
    baseline["legacy_task_makespan_scored"] = False


def _source_state(params, solution):
    """Validate the source ILP solution and derive immutable chiplet bundles."""
    if "sites" not in solution or "task_site" not in solution:
        raise ValueError("solution must contain sites and task_site")
    if "grid" not in solution:
        raise ValueError("solution must contain grid")
    grid = copy.deepcopy(solution["grid"])
    _grid_tuple(grid)
    budgets = copy.deepcopy(solution.get("budgets") or {})
    library = params.get("library")
    tasks = params.get("tasks")
    if not isinstance(library, dict) or not library:
        raise ValueError("params.library must be a non-empty object")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("params.tasks must be a non-empty object")

    site_records = []
    seen_anchors = set()
    for ordinal, site in enumerate(solution["sites"]):
        try:
            anchor = (int(site["l"]), int(site["i"]), int(site["j"]))
            chiplet_type = site["type"]
        except (KeyError, TypeError, ValueError):
            raise ValueError("solution.sites[%d] is malformed" % ordinal)
        if anchor in seen_anchors:
            raise ValueError("solution has duplicate chiplet anchor %s" %
                             (anchor,))
        if chiplet_type not in library:
            raise ValueError("solution uses unknown chiplet type %r" %
                             chiplet_type)
        seen_anchors.add(anchor)
        site_records.append((anchor, chiplet_type))
    site_records.sort(key=lambda item: item[0])
    site_types = {anchor: chiplet_type
                  for anchor, chiplet_type in site_records}

    source_task_site = solution["task_site"]
    task_names = set(tasks)
    mapped_names = set(source_task_site)
    if task_names != mapped_names:
        missing = sorted(task_names - mapped_names)
        extra = sorted(mapped_names - task_names)
        raise ValueError("solution.task_site task mismatch; missing=%s extra=%s" %
                         (missing, extra))

    task_anchor = {}
    for task_name in sorted(tasks):
        anchor = _as_anchor(source_task_site[task_name],
                            "task_site[%r]" % task_name)
        if anchor not in site_types:
            raise ValueError("task %s is assigned to empty anchor %s" %
                             (task_name, anchor))
        task_anchor[task_name] = anchor

    source_score, reasons = _strict_score(
        params,
        {name: list(anchor) for name, anchor in task_anchor.items()},
        site_types, grid, budgets)
    if reasons:
        raise ValueError("source ILP solution is not strictly feasible: %s" %
                         "; ".join(reasons))

    expert_names = set(params.get("expert_tasks") or [])
    unknown_experts = expert_names - task_names
    if unknown_experts:
        raise ValueError("params.expert_tasks contains unknown tasks: %s" %
                         sorted(unknown_experts))

    tasks_at = {anchor: [] for anchor in site_types}
    for task_name in sorted(tasks):
        tasks_at[task_anchor[task_name]].append(task_name)

    bundles = []
    for bundle_id, (anchor, chiplet_type) in enumerate(site_records):
        names = tasks_at[anchor]
        experts = [name for name in names if name in expert_names]
        if len(experts) > 1:
            raise ValueError("source bundle at %s contains multiple experts" %
                             (anchor,))
        base_tasks = tuple(name for name in names if name not in expert_names)
        bundles.append({
            "id": bundle_id,
            "old_anchor": anchor,
            "type": chiplet_type,
            "base_tasks": base_tasks,
            "original_expert": experts[0] if experts else None,
        })

    return {
        "grid": grid,
        "budgets": budgets,
        "site_types": site_types,
        "task_anchor": task_anchor,
        "bundles": bundles,
        "source_score": source_score,
        "type_counts": _type_counts(site_types),
        "cpu_counts": _cpu_counts(site_types, library),
    }


def _random_expert_assignment(params, bundles, rng):
    """Random feasible matching of experts to expert-bearing bundles."""
    expert_bundles = [bundle for bundle in bundles
                      if bundle["original_expert"] is not None]
    if not expert_bundles:
        return {}, False

    expert_bundles.sort(key=lambda bundle: bundle["id"])
    expert_names = sorted(bundle["original_expert"]
                          for bundle in expert_bundles)
    tasks = params["tasks"]
    library = params["library"]
    original = {bundle["id"]: bundle["original_expert"]
                for bundle in expert_bundles}

    candidates = {}
    for bundle in expert_bundles:
        chiplet_type = bundle["type"]
        base_mem = sum(float(tasks[name]["mem_mb"])
                       for name in bundle["base_tasks"])
        cap = float(library[chiplet_type]["mem"])
        legal = []
        for expert in expert_names:
            if chiplet_type not in tasks[expert]["allowed"]:
                continue
            if base_mem + float(tasks[expert]["mem_mb"]) > cap + _FLOAT_TOL:
                continue
            legal.append(expert)
        if not legal:
            raise RuntimeError("no expert fits source bundle %d (%s)" %
                               (bundle["id"], chiplet_type))
        candidates[bundle["id"]] = legal

    # Precomputed random priorities avoid dependence on set/dict iteration.
    bundle_priority = {bundle["id"]: rng.random()
                       for bundle in expert_bundles}
    candidate_priority = {}
    for bundle in expert_bundles:
        for expert in candidates[bundle["id"]]:
            candidate_priority[(bundle["id"], expert)] = rng.random()

    def solve(require_change):
        assignment = {}
        used = set()
        nodes = [0]

        def visit(changed):
            nodes[0] += 1
            if nodes[0] > _MAX_MATCH_NODES:
                return False
            if len(assignment) == len(expert_bundles):
                return changed or not require_change

            remaining = []
            for bundle in expert_bundles:
                bundle_id = bundle["id"]
                if bundle_id in assignment:
                    continue
                available = [expert for expert in candidates[bundle_id]
                             if expert not in used]
                remaining.append((len(available),
                                  bundle_priority[bundle_id],
                                  bundle_id, available))
            remaining.sort(key=lambda item: (item[0], item[1], item[2]))
            if not remaining or remaining[0][0] == 0:
                return False
            _, _, bundle_id, available = remaining[0]
            available.sort(
                key=lambda expert: candidate_priority[(bundle_id, expert)])
            for expert in available:
                assignment[bundle_id] = expert
                used.add(expert)
                if visit(changed or expert != original[bundle_id]):
                    return True
                used.remove(expert)
                del assignment[bundle_id]
            return False

        if visit(False):
            return dict(assignment)
        return None

    # With two or more experts, make "shuffle" observable whenever any legal
    # non-identity matching exists.  A singleton or type-forced identity is
    # retained because no other strictly feasible permutation exists.
    result = solve(require_change=(len(expert_bundles) > 1))
    if result is None:
        result = solve(require_change=False)
    if result is None:
        raise RuntimeError("expert-to-GPU permutation has no feasible matching")
    changed = any(result[bundle_id] != original[bundle_id]
                  for bundle_id in result)
    return result, changed


def _random_packing(params, state, rng):
    """Pack the exact source chiplet multiset at random non-overlapping anchors."""
    library = params["library"]
    grid = state["grid"]
    budgets = state["budgets"]
    bundles = state["bundles"]
    counts = collections.Counter(bundle["type"] for bundle in bundles)
    chiplet_types = sorted(counts)

    all_anchors = {}
    for chiplet_type in chiplet_types:
        anchors = _legal_anchors(chiplet_type, grid, library)
        if len(anchors) < counts[chiplet_type]:
            raise RuntimeError("grid has too few anchors for %d %s chiplets" %
                               (counts[chiplet_type], chiplet_type))
        all_anchors[chiplet_type] = anchors

    layer_cap = budgets.get("a_max_layer")
    if layer_cap is not None:
        layer_cap = float(layer_cap)
    column_cap = budgets.get("p_col")
    if column_cap is not None:
        column_cap = float(column_cap)

    for unused_restart in range(_MAX_PACK_RESTARTS):
        del unused_restart
        # Large footprints go first.  Random tie priorities make equally sized
        # types and their anchor combinations seed-dependent.
        type_priority = {name: rng.random() for name in chiplet_types}
        group_order = sorted(
            chiplet_types,
            key=lambda name: (
                -(int(library[name]["w"]) * int(library[name]["h"])),
                -float(library[name]["area"]),
                type_priority[name],
                name))
        anchor_order = {}
        footprint_cache = {}
        for chiplet_type in chiplet_types:
            values = list(all_anchors[chiplet_type])
            rng.shuffle(values)
            anchor_order[chiplet_type] = values
            for anchor in values:
                footprint_cache[(chiplet_type, anchor)] = _footprint(
                    chiplet_type, anchor, library)

        occupied = set()
        layer_area = {}
        column_power = {}
        selected = {name: [] for name in chiplet_types}
        nodes = [0]
        exhausted = [False]

        def can_add(chiplet_type, anchor):
            cells = footprint_cache[(chiplet_type, anchor)]
            if any(cell in occupied for cell in cells):
                return False
            spec = library[chiplet_type]
            if layer_cap is not None:
                new_area = (layer_area.get(anchor[0], 0.0)
                            + float(spec["area"]))
                if new_area > layer_cap + _FLOAT_TOL:
                    return False
            if column_cap is not None:
                density = (float(spec["p_peak"]) /
                           (int(spec["w"]) * int(spec["h"])))
                for cell in cells:
                    key = (cell[1], cell[2])
                    if (column_power.get(key, 0.0) + density >
                            column_cap + _FLOAT_TOL):
                        return False
            return True

        def add(chiplet_type, anchor):
            spec = library[chiplet_type]
            cells = footprint_cache[(chiplet_type, anchor)]
            occupied.update(cells)
            layer_area[anchor[0]] = (layer_area.get(anchor[0], 0.0)
                                     + float(spec["area"]))
            density = (float(spec["p_peak"]) /
                       (int(spec["w"]) * int(spec["h"])))
            for cell in cells:
                key = (cell[1], cell[2])
                column_power[key] = column_power.get(key, 0.0) + density
            selected[chiplet_type].append(anchor)

        def remove(chiplet_type, anchor):
            spec = library[chiplet_type]
            cells = footprint_cache[(chiplet_type, anchor)]
            for cell in cells:
                occupied.remove(cell)
            layer_area[anchor[0]] -= float(spec["area"])
            density = (float(spec["p_peak"]) /
                       (int(spec["w"]) * int(spec["h"])))
            for cell in cells:
                key = (cell[1], cell[2])
                column_power[key] -= density
            selected[chiplet_type].pop()

        def place_group(group_index):
            if exhausted[0]:
                return False
            if group_index == len(group_order):
                return True
            chiplet_type = group_order[group_index]
            anchors = anchor_order[chiplet_type]
            required = counts[chiplet_type]

            # Identical chiplets are anonymous during geometry search.  Taking
            # anchors in increasing randomized rank eliminates factorially many
            # equivalent search branches; bundle identities are shuffled onto
            # the selected anchors after a packing is found.
            def choose(start_index, left):
                nodes[0] += 1
                if nodes[0] > _MAX_PACK_NODES:
                    exhausted[0] = True
                    return False
                if left == 0:
                    return place_group(group_index + 1)
                if len(anchors) - start_index < left:
                    return False
                last_start = len(anchors) - left
                for position in range(start_index, last_start + 1):
                    anchor = anchors[position]
                    if not can_add(chiplet_type, anchor):
                        continue
                    add(chiplet_type, anchor)
                    if choose(position + 1, left - 1):
                        return True
                    remove(chiplet_type, anchor)
                    if exhausted[0]:
                        return False
                return False

            return choose(0, required)

        if not place_group(0):
            continue

        bundle_anchor = {}
        for chiplet_type in chiplet_types:
            typed_bundles = sorted(
                (bundle for bundle in bundles
                 if bundle["type"] == chiplet_type),
                key=lambda bundle: bundle["id"])
            anchors = list(selected[chiplet_type])
            rng.shuffle(anchors)
            for bundle, anchor in zip(typed_bundles, anchors):
                bundle_anchor[bundle["id"]] = anchor
        return bundle_anchor, "random-backtracking"

    # The input itself proves that a packing exists.  Coordinate reflections
    # preserve overlap, Manhattan distance, per-layer area, and column density,
    # so this deterministic-seed fallback remains a valid randomized relocation
    # even for pathological tightly packed instances that hit the node limit.
    return _reflected_source_packing(params, state, rng), "source-reflection"


def _reflected_source_packing(params, state, rng):
    L, X, Y = _grid_tuple(state["grid"])
    library = params["library"]
    transforms = [(flip_l, flip_i, flip_j)
                  for flip_l in (False, True)
                  for flip_i in (False, True)
                  for flip_j in (False, True)]
    rng.shuffle(transforms)
    # Prefer a non-identity reflection.  Degenerate one-cell dimensions may
    # still make the resulting coordinates identical, which is unavoidable.
    transforms.sort(key=lambda value: value == (False, False, False))
    flip_l, flip_i, flip_j = transforms[0]

    by_type = {}
    for bundle in state["bundles"]:
        chiplet_type = bundle["type"]
        l, i, j = bundle["old_anchor"]
        spec = library[chiplet_type]
        if flip_l:
            l = L - 1 - l
        if flip_i:
            i = X - int(spec["w"]) - i
        if flip_j:
            j = Y - int(spec["h"]) - j
        by_type.setdefault(chiplet_type, []).append((l, i, j))

    result = {}
    for chiplet_type in sorted(by_type):
        anchors = sorted(by_type[chiplet_type])
        rng.shuffle(anchors)
        typed_bundles = sorted(
            (bundle for bundle in state["bundles"]
             if bundle["type"] == chiplet_type),
            key=lambda bundle: bundle["id"])
        for bundle, anchor in zip(typed_bundles, anchors):
            result[bundle["id"]] = anchor
    return result


def _candidate_solution(params, source_solution, state, seed,
                        permute_experts=True):
    rng = random.Random(seed)
    tasks = params["tasks"]
    task_rank = {name: rank for rank, name in
                 enumerate(sorted(tasks))}
    last_reasons = []

    for attempt in range(_MAX_LAYOUT_ATTEMPTS):
        bundle_anchor, packing_strategy = _random_packing(params, state, rng)
        if permute_experts:
            expert_assignment, expert_changed = _random_expert_assignment(
                params, state["bundles"], rng)
        else:
            expert_assignment = {
                bundle["id"]: bundle["original_expert"]
                for bundle in state["bundles"]
                if bundle["original_expert"] is not None
            }
            expert_changed = False

        task_site = {}
        bundle_tasks = {}
        site_types = {}
        for bundle in state["bundles"]:
            bundle_id = bundle["id"]
            anchor = bundle_anchor[bundle_id]
            site_types[anchor] = bundle["type"]
            names = list(bundle["base_tasks"])
            if bundle["original_expert"] is not None:
                names.append(expert_assignment[bundle_id])
            names.sort(key=lambda name: (task_rank[name], name))
            bundle_tasks[bundle_id] = names
            for name in names:
                task_site[name] = list(anchor)

        score, reasons = _strict_score(
            params, task_site, site_types, state["grid"], state["budgets"])
        if reasons:
            last_reasons = reasons
            continue

        if _type_counts(site_types) != state["type_counts"]:
            raise AssertionError("internal error: chiplet type counts changed")
        if _cpu_counts(site_types, params["library"]) != state["cpu_counts"]:
            raise AssertionError("internal error: dedicated CPU count changed")
        if (abs(score["total_area_mm2"]
                - state["source_score"]["total_area_mm2"]) > _FLOAT_TOL):
            raise AssertionError("internal error: total chiplet area changed")
        if (abs(score["peak_power_W"]
                - state["source_score"]["peak_power_W"]) > _FLOAT_TOL):
            raise AssertionError("internal error: package power changed")

        sites = []
        for bundle in sorted(state["bundles"],
                             key=lambda item: bundle_anchor[item["id"]]):
            anchor = bundle_anchor[bundle["id"]]
            sites.append({
                "l": anchor[0],
                "i": anchor[1],
                "j": anchor[2],
                "type": bundle["type"],
                "tasks": bundle_tasks[bundle["id"]],
            })

        output = {
            "method": "resource-matched-random-layout",
            "seed": seed,
            "sites": sites,
            "task_site": {name: task_site[name] for name in sorted(task_site)},
            "grid": copy.deepcopy(state["grid"]),
            "budgets": copy.deepcopy(state["budgets"]),
            # codegen.py expects this conventional field.  For a baseline it is
            # the exact shared-evaluator value, not an ILP model prediction.
            "predicted_makespan_cycles": score["makespan_cycles"],
            "makespan_cycles": score["makespan_cycles"],
            "evaluated_makespan_cycles": score["makespan_cycles"],
            "energy_total_J": score["energy_total_J"],
            "energy_comm_J": score["energy_comm_J"],
            "energy_comp_J": score["energy_comp_J"],
            "byte_hop": score["byte_hop"],
            "total_area_mm2": score["total_area_mm2"],
            "peak_power_W": score["peak_power_W"],
            "feasible": True,
            "reasons": [],
            "baseline": {
                "method": "resource-matched-random-layout",
                "seed": seed,
                "layout_attempt": attempt + 1,
                "packing_strategy": packing_strategy,
                "expert_mapping_changed": expert_changed,
                "strict_evaluate": True,
                "source_ilp_predicted_makespan_cycles":
                    source_solution.get("predicted_makespan_cycles"),
            },
        }
        return output, score

    detail = "; ".join(last_reasons) if last_reasons else "no legal packing"
    raise RuntimeError("seed %d: failed to produce a budget-feasible matched "
                       "layout after %d attempts (%s)" %
                       (seed, _MAX_LAYOUT_ATTEMPTS, detail))


def _median(values):
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _select_public_T_representative(runs):
    """Return the mathematical public-T median and an observed candidate."""
    median_public_T = _median([record["public_T_cycles"]
                              for record in runs])
    # For an even seed count the mathematical median is between two observed
    # runs.  representative.json must be an actual layout, so choose the closer
    # central observation; ties go to lower public T, then lower seed.
    representative = min(
        runs,
        key=lambda record: (
            abs(record["public_T_cycles"] - median_public_T),
            record["public_T_cycles"],
            record["seed"]))
    return median_public_T, representative


def _public_T_components(evaluation):
    """Return the JSON-safe scalar decomposition used in result metadata."""
    return {
        "public_T_cycles": float(evaluation["T_cycles"]),
        "protocol_completion_cycles": float(
            evaluation["protocol"]["completion_cycles"]),
        "directed_mesh_capacity_bound_cycles": float(
            evaluation["routing"]["capacity_bound_cycles"]),
        "hard_feasibility_task_makespan_cycles": float(
            evaluation["schedule"]["makespan_cycles"]),
    }


def run(params_path, solution_path, seeds, out_dir):
    params = _read_json(params_path)
    source_solution = _read_json(solution_path)
    state = _source_state(params, source_solution)
    public_scorer = _PublicTScorer(params, source_solution, state)
    scoring_provenance = public_scorer.provenance(
        public_scorer.source_evaluation)

    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        raise ValueError("--out-dir exists but is not a directory: %s" % out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    runs = []
    generated = {}
    for seed in seeds:
        solution, legacy_score = _candidate_solution(
            params, source_solution, state, seed)
        public_score = public_scorer.evaluate(solution)
        _apply_public_T_score(
            solution, public_score, legacy_score, scoring_provenance)
        filename = "seed_%d.json" % seed
        path = os.path.join(out_dir, filename)
        _write_json_atomic(path, solution)

        # Validate the serialized artifact, not only the in-memory candidate.
        written = _read_json(path)
        written_types = ev.site_types_from_solution(written)
        written_score, reasons = _strict_score(
            params, written["task_site"], written_types,
            written["grid"], written.get("budgets") or {})
        if reasons:
            raise RuntimeError("serialized %s failed strict evaluation: %s" %
                               (filename, "; ".join(reasons)))
        if _type_counts(written_types) != state["type_counts"]:
            raise RuntimeError("serialized %s changed chiplet type counts" %
                               filename)
        if written.get("budgets") != state["budgets"]:
            raise RuntimeError("serialized %s changed the ILP budgets" % filename)

        written_public = public_scorer.evaluate(written)
        written_components = _public_T_components(written_public)
        for field, expected in written_components.items():
            try:
                actual = float(written[field])
            except (KeyError, TypeError, ValueError):
                raise RuntimeError(
                    "serialized %s lacks public-T field %s" %
                    (filename, field))
            tolerance = max(1.0e-6,
                            1.0e-10 * max(abs(actual), abs(expected)))
            if abs(actual - expected) > tolerance:
                raise RuntimeError(
                    "serialized %s public-T field %s changed" %
                    (filename, field))

        generated[seed] = path
        run_record = {
            "seed": seed,
            "solution": filename,
            "strict_feasible": True,
            "public_T_cycles": written_components["public_T_cycles"],
            # Backward-compatible name; this is now explicitly public T.
            "makespan_cycles": written_components["public_T_cycles"],
            "protocol_completion_cycles": written_components[
                "protocol_completion_cycles"],
            "directed_mesh_capacity_bound_cycles": written_components[
                "directed_mesh_capacity_bound_cycles"],
            "hard_feasibility_task_makespan_cycles": written_components[
                "hard_feasibility_task_makespan_cycles"],
            "legacy_evaluate_task_makespan_cycles": written_score[
                "makespan_cycles"],
            "energy_total_J": written_score["energy_total_J"],
            "energy_comm_J": written_score["energy_comm_J"],
            "energy_comp_J": written_score["energy_comp_J"],
            "byte_hop": written_score["byte_hop"],
            "total_area_mm2": written_score["total_area_mm2"],
            "peak_power_W": written_score["peak_power_W"],
            "expert_mapping_changed":
                solution["baseline"]["expert_mapping_changed"],
            "packing_strategy": solution["baseline"]["packing_strategy"],
        }
        runs.append(run_record)
        print("[matched] seed=%d public_T=%.6f task_feas=%.6f "
              "area=%.6f power=%.6f" %
              (seed, written_components["public_T_cycles"],
               written_components["hard_feasibility_task_makespan_cycles"],
               written_score["total_area_mm2"],
               written_score["peak_power_W"]))

    median_public_T, representative = _select_public_T_representative(runs)
    representative_path = os.path.join(out_dir, "representative.json")
    shutil.copyfile(generated[representative["seed"]], representative_path)

    rep_solution = _read_json(representative_path)
    rep_types = ev.site_types_from_solution(rep_solution)
    rep_score, rep_reasons = _strict_score(
        params, rep_solution["task_site"], rep_types,
        rep_solution["grid"], rep_solution.get("budgets") or {})
    if rep_reasons:
        raise RuntimeError("representative.json failed strict evaluation: %s" %
                           "; ".join(rep_reasons))
    rep_public = public_scorer.evaluate(rep_solution)
    rep_components = _public_T_components(rep_public)

    summary = {
        "schema_version": 2,
        "method": "resource-matched-random-layout",
        "params": os.path.abspath(params_path),
        "source_solution": os.path.abspath(solution_path),
        "grid": copy.deepcopy(state["grid"]),
        "budgets": copy.deepcopy(state["budgets"]),
        "seeds": list(seeds),
        "chiplet_type_counts": copy.deepcopy(state["type_counts"]),
        "dedicated_cpu_type_counts": copy.deepcopy(state["cpu_counts"]),
        "source_evaluation": copy.deepcopy(state["source_score"]),
        "source_public_T_evaluation": _public_T_components(
            public_scorer.source_evaluation),
        "representative_selection_metric": _PUBLIC_T_METRIC,
        "representative_selection_policy": _REPRESENTATIVE_POLICY,
        "candidate_count": len(runs),
        "all_candidates_scored_with_public_T": True,
        "public_T_scoring": copy.deepcopy(scoring_provenance),
        "runs": runs,
        "parsed_public_T_cycles":
            [record["public_T_cycles"] for record in runs],
        "median_public_T_cycles": median_public_T,
        # Backward-compatible aliases; both are public T, never task-DAG T.
        "parsed_makespans_cycles":
            [record["public_T_cycles"] for record in runs],
        "median_makespan_cycles": median_public_T,
        "representative_seed": representative["seed"],
        "representative_source": representative["solution"],
        "representative_solution": "representative.json",
        "representative_public_T_cycles": rep_components[
            "public_T_cycles"],
        "representative_makespan_cycles": rep_components[
            "public_T_cycles"],
        "representative_protocol_completion_cycles": rep_components[
            "protocol_completion_cycles"],
        "representative_directed_mesh_capacity_bound_cycles": rep_components[
            "directed_mesh_capacity_bound_cycles"],
        "representative_hard_feasibility_task_makespan_cycles":
            rep_components["hard_feasibility_task_makespan_cycles"],
        "representative_legacy_evaluate_task_makespan_cycles": rep_score[
            "makespan_cycles"],
        "strict_evaluate": True,
        "resource_match": {
            "chiplet_type_counts_equal": True,
            "dedicated_cpu_counts_equal": True,
            "budgets_copied_exactly": True,
            "total_area_mm2": state["source_score"]["total_area_mm2"],
            "peak_power_W": state["source_score"]["peak_power_W"],
        },
    }
    summary_path = os.path.join(out_dir, "summary.json")
    _write_json_atomic(summary_path, summary)
    print("[matched] representative seed=%d public_T=%.6f -> %s" %
          (representative["seed"], rep_components["public_T_cycles"],
           representative_path))
    print("[matched] summary -> %s" % summary_path)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Random layout baseline matched to an ILP chiplet BOM")
    parser.add_argument("--params", required=True,
                        help="workload/library params JSON")
    parser.add_argument("--solution", required=True,
                        help="source ILP solution JSON")
    parser.add_argument("--seeds", required=True, nargs="+",
                        help="integer seeds (space- or comma-separated)")
    parser.add_argument("--out-dir", required=True,
                        help="directory for per-seed solutions and summary")
    args = parser.parse_args(argv)
    try:
        seeds = parse_seeds(args.seeds)
        run(args.params, args.solution, seeds, args.out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
