#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the LEGOSim MoE protocol on one fixed layout.

This is an independent, pure-Python oracle for reviewing the MILP performance
objective.  It mirrors the ordering implemented by ``moe_router_cpu.cpp`` and
``moe_gpu_worker.cu``:

  CPU init (worker-ID order) -> ordered dispatch -> measured GPU compute ->
  reverse combine -> worker summaries -> CPU completion (worker-ID order).

The current formal q=1 ABI uses stable logical IDs: worker ``g`` owns ``e(g)``.
Every send/read API advances its local process clock by 1,000 cycles.  For a
message with F flits, the no-contention timing envelope calibrated from the
audited PopNet traces is

  source delay      = F - 1
  destination delay = F - 1 + hop_cycles * Manhattan_distance.

The two endpoint clocks are asymmetric, matching InterChiplet's
``NetworkDelayList::getEndCycle``: the source completes from its write time,
while the destination completes at max(write + destination_delay, read time).
No solver, simulator, code generator, or third-party package is imported.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import re
import sys


EXPERT_RE = re.compile(r"^e\((\d+)\)$")

# Audited A1550 profile: 32 tokens/source, top-2, hidden=128,
# intermediate=64, ffn_repeat=1.  Entries are deliberately type-scoped so a
# layout that maps a group to an unprofiled GPU type fails closed.
DEFAULT_A1550_PROFILE = {
    "source": "audited-a1550-stable-worker-id-gpgpusim",
    "scope": {
        "groups": 8,
        "tokens_per_source": 32,
        "topk": 2,
        "hidden": 128,
        "intermediate": 64,
        "ffn_repeat": 1,
    },
    "group_type_cycles": {
        "0": {"GPU-L": 4663},
        "1": {"GPU-L": 4670},
        "2": {"GPU-S": 23726},
        "3": {"GPU-S": 20619},
        "4": {"GPU-S": 18338},
        "5": {"GPU-S": 17002},
        "6": {"GPU-S": 16488},
        "7": {"GPU-S": 16120},
    },
    "group_cycles": {},
}


def _read_json(path):
    with open(path, "r") as fh:
        return json.load(fh)


def _write_json(path, value):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _positive_int(value, name):
    try:
        integer = int(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a positive integer" % name)
    if integer <= 0 or float(value) != float(integer):
        raise ValueError("%s must be a positive integer" % name)
    return integer


def _nonnegative_cycle(value, name):
    try:
        cycle = int(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a nonnegative integer cycle count" % name)
    if cycle < 0 or float(value) != float(cycle):
        raise ValueError("%s must be a nonnegative integer cycle count" % name)
    return cycle


def _largest_remainder(weights, total):
    """Deterministically apportion integer assignments while preserving sum."""
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
                   key=lambda index: (-(exact[index] - base[index]), index))
    for index in order[:left]:
        base[index] += 1
    return base


def _balanced_transport(column_totals, nrows, row_total):
    """Construct the same cyclic logical transport used by codegen.py."""
    totals = [int(value) for value in column_totals]
    if nrows <= 0 or row_total < 0 or any(value < 0 for value in totals):
        raise ValueError("invalid balanced-transport margins")
    if sum(totals) != nrows * row_total:
        raise ValueError("balanced-transport row/column totals disagree")
    matrix = [[0] * len(totals) for _ in range(nrows)]
    cursor = 0
    for column, total in enumerate(totals):
        quotient, remainder = divmod(total, nrows)
        if quotient:
            for source in range(nrows):
                matrix[source][column] = quotient
        for _ in range(remainder):
            matrix[cursor][column] += 1
            cursor = (cursor + 1) % nrows
    if ([sum(row) for row in matrix] != [row_total] * nrows or
            [sum(matrix[row][column] for row in range(nrows))
             for column in range(len(totals))] != totals):
        raise AssertionError("balanced transport failed to preserve margins")
    return matrix


def logical_dispatch_matrix(params, tokens_per_source):
    meta = params.get("meta", {})
    groups = int(meta.get("groups", len(params.get("expert_tasks", []))))
    topk = _positive_int(meta.get("topk", 1), "meta.topk")
    tokens = _positive_int(tokens_per_source, "tokens_per_source")
    if groups <= 0:
        raise ValueError("formal protocol requires at least one logical group")
    weights = list(meta.get("group_pi", []))
    if len(weights) != groups:
        weights = [1.0] * groups
    assignments_per_source = tokens * topk
    targets = _largest_remainder(
        weights, groups * assignments_per_source)
    return (_balanced_transport(targets, groups, assignments_per_source),
            targets)


def _expert_group(task):
    match = EXPERT_RE.match(task)
    return int(match.group(1)) if match else None


def _flatten_site(site, grid):
    # Same Option-A mapping used by codegen.py and the C++ coordinate file:
    # x = l*Y+j, y = i.
    return (int(site["l"]) * int(grid["Y"]) + int(site["j"]),
            int(site["i"]))


def _resolve_workers_and_router(solution, groups):
    grid = solution.get("grid", {})
    for key in ("L", "X", "Y"):
        if int(grid.get(key, 0)) <= 0:
            raise ValueError("solution grid.%s must be positive" % key)

    workers = {}
    expert_sites = []
    for site in solution.get("sites", []):
        logical = [_expert_group(task) for task in site.get("tasks", [])]
        logical = [group for group in logical if group is not None]
        if not logical:
            continue
        if not str(site.get("type", "")).startswith("GPU"):
            raise ValueError("expert-bearing site is not a GPU: %s" % site)
        if len(logical) != 1:
            raise ValueError(
                "q=1 protocol surrogate requires exactly one logical expert "
                "group per GPU worker; site has %s" % sorted(logical))
        group = logical[0]
        if group in workers:
            raise ValueError("logical expert group %d has multiple sites" % group)
        workers[group] = {
            "worker_id": group,
            "group": group,
            "type": site["type"],
            "site": [int(site["l"]), int(site["i"]), int(site["j"])],
            "network_xy": list(_flatten_site(site, grid)),
            "tasks": list(site.get("tasks", [])),
        }
        expert_sites.append(site)

    expected = list(range(groups))
    missing = [group for group in expected if group not in workers]
    extra = sorted(set(workers) - set(expected))
    if missing or extra or len(workers) != groups:
        raise ValueError(
            "solution must map groups 0..%d bijectively to q=1 GPU workers; "
            "missing=%s extra=%s" % (groups - 1, missing, extra))

    cpu_sites = [site for site in solution.get("sites", [])
                 if site.get("type") == "CPU"]
    if cpu_sites:
        router_site = cpu_sites[0]
        router_kind = "solution-cpu-site"
    else:
        occupied = set((int(site["l"]), int(site["i"]), int(site["j"]))
                       for site in expert_sites)
        router_site = None
        # Preserve codegen.resolve_router scan order: i, then l, then j.
        for i in range(int(grid["X"])):
            for layer in range(int(grid["L"])):
                for j in range(int(grid["Y"])):
                    if (layer, i, j) not in occupied:
                        router_site = {"l": layer, "i": i, "j": j,
                                       "type": "synthetic-router",
                                       "tasks": []}
                        break
                if router_site is not None:
                    break
            if router_site is not None:
                break
        if router_site is None:
            router_site = {"l": 0, "i": 0, "j": 0,
                           "type": "shared-fallback-router", "tasks": []}
        router_kind = "codegen-compatible-free-node"

    router = {
        "kind": router_kind,
        "site": [int(router_site["l"]), int(router_site["i"]),
                 int(router_site["j"])],
        "network_xy": list(_flatten_site(router_site, grid)),
    }
    return [workers[group] for group in expected], router


def _normalize_profile(profile, source):
    if profile is None:
        profile = DEFAULT_A1550_PROFILE
        source = DEFAULT_A1550_PROFILE["source"]
    if not isinstance(profile, dict):
        raise ValueError("GPU profile must be a mapping")

    normalized = {
        "source": str(profile.get("source", source or "provided-mapping")),
        "scope": dict(profile.get("scope", {})),
        "group_cycles": {},
        "group_type_cycles": {},
    }

    if "group_cycles" in profile or "group_type_cycles" in profile:
        direct = profile.get("group_cycles", {})
        typed = profile.get("group_type_cycles", {})
    else:
        # Compact accepted form: {group: cycles} or
        # {group: {GPU-L: cycles, GPU-S: cycles}}.
        direct, typed = {}, {}
        for group, value in profile.items():
            if group in ("source", "scope"):
                continue
            if isinstance(value, dict):
                typed[group] = value
            else:
                direct[group] = value

    for group, value in direct.items():
        gid = str(int(group))
        normalized["group_cycles"][gid] = _nonnegative_cycle(
            value, "GPU profile group %s" % gid)
    for group, by_type in typed.items():
        gid = str(int(group))
        if not isinstance(by_type, dict):
            raise ValueError("typed GPU profile group %s is not a mapping" % gid)
        normalized["group_type_cycles"][gid] = {}
        for chiplet_type, value in by_type.items():
            normalized["group_type_cycles"][gid][str(chiplet_type)] = \
                _nonnegative_cycle(
                    value, "GPU profile group %s type %s" %
                    (gid, chiplet_type))
    return normalized


def load_gpu_profile(path):
    """Load JSON, existing gpu_cycles.tsv, or ``group type cycles`` TSV."""
    if str(path).lower().endswith(".json"):
        return _normalize_profile(_read_json(path), os.path.abspath(path))

    direct = {}
    typed = {}
    with open(path, "r") as fh:
        for number, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) == 1:
                fields = line.split()
            if len(fields) == 2:
                label, value = fields
                match = re.search(r"gpgpusim\.0\.(\d+)\.log", label)
                group = int(match.group(1)) if match else int(label)
                direct[str(group)] = _nonnegative_cycle(
                    value, "%s:%d" % (path, number))
            elif len(fields) == 3:
                group, chiplet_type, value = fields
                gid = str(int(group))
                typed.setdefault(gid, {})[chiplet_type] = _nonnegative_cycle(
                    value, "%s:%d" % (path, number))
            else:
                raise ValueError("%s:%d has unsupported GPU profile format" %
                                 (path, number))
    return _normalize_profile({
        "source": os.path.abspath(path),
        "group_cycles": direct,
        "group_type_cycles": typed,
    }, os.path.abspath(path))


def _profile_cycles(profile, worker):
    group = str(worker["group"])
    chiplet_type = worker["type"]
    typed = profile["group_type_cycles"].get(group, {})
    if chiplet_type in typed:
        return typed[chiplet_type]
    if "*" in typed:
        return typed["*"]
    if group in profile["group_cycles"]:
        return profile["group_cycles"][group]
    raise ValueError(
        "GPU profile %s has no cycle entry for group %s on %s" %
        (profile["source"], group, chiplet_type))


def _canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_protocol_surrogate(params, solution, gpu_profile=None,
                                tokens_per_source=32, hidden=128,
                                element_bytes=2, flit_payload_bytes=32,
                                operation_overhead_cycles=1000,
                                hop_cycles=104, include_events=False,
                                profile_source=None):
    """Return a deterministic fixed-layout protocol timing decomposition."""
    tokens = _positive_int(tokens_per_source, "tokens_per_source")
    hidden = _positive_int(hidden, "hidden")
    element_bytes = _positive_int(element_bytes, "element_bytes")
    flit_payload_bytes = _positive_int(
        flit_payload_bytes, "flit_payload_bytes")
    overhead = _positive_int(
        operation_overhead_cycles, "operation_overhead_cycles")
    hop_cycles = _positive_int(hop_cycles, "hop_cycles")

    matrix, targets = logical_dispatch_matrix(params, tokens)
    groups = len(matrix)
    workers, router = _resolve_workers_and_router(solution, groups)
    profile = _normalize_profile(gpu_profile, profile_source)
    compute_cycles = [_profile_cycles(profile, worker) for worker in workers]
    compute_tokens = [sum(matrix[source][group]
                          for source in range(groups))
                      for group in range(groups)]

    process_clock = {"cpu": 0}
    for group in range(groups):
        process_clock[group] = 0

    event_rows = []
    phase_stats = {}

    def xy(process):
        if process == "cpu":
            return tuple(router["network_xy"])
        return tuple(workers[int(process)]["network_xy"])

    def distance(source, destination):
        source_xy = xy(source)
        destination_xy = xy(destination)
        return (abs(source_xy[0] - destination_xy[0]) +
                abs(source_xy[1] - destination_xy[1]))

    def start_phase(name):
        phase_stats[name] = {
            "message_count": 0,
            "bytes": 0,
            "flits": 0,
            "hop_count": 0,
            "byte_hops": 0,
            "source_service_cycles": 0,
        }

    def send(phase, label, source, destination, nbytes,
             token_assignments=None):
        if nbytes <= 0 or source == destination:
            return
        hops = distance(source, destination)
        flits = int(math.ceil(float(nbytes) / flit_payload_bytes)) + 1
        source_delay = flits - 1
        destination_delay = source_delay + hop_cycles * hops
        write_cycle = process_clock[source] + overhead
        read_cycle = process_clock[destination] + overhead
        source_finish = write_cycle + source_delay
        destination_finish = max(
            write_cycle + destination_delay, read_cycle)
        process_clock[source] = source_finish
        process_clock[destination] = destination_finish

        stats = phase_stats[phase]
        stats["message_count"] += 1
        stats["bytes"] += nbytes
        stats["flits"] += flits
        stats["hop_count"] += hops
        stats["byte_hops"] += nbytes * hops
        stats["source_service_cycles"] += source_delay
        if token_assignments is not None:
            stats["token_assignments"] = \
                stats.get("token_assignments", 0) + token_assignments
            stats["assignment_hops"] = \
                stats.get("assignment_hops", 0) + token_assignments * hops

        if include_events:
            event_rows.append({
                "index": len(event_rows),
                "phase": phase,
                "label": label,
                "source": source,
                "destination": destination,
                "source_xy": list(xy(source)),
                "destination_xy": list(xy(destination)),
                "bytes": nbytes,
                "flits": flits,
                "hops": hops,
                "write_cycle": write_cycle,
                "read_cycle": read_cycle,
                "source_delay": source_delay,
                "destination_delay": destination_delay,
                "source_finish": source_finish,
                "destination_finish": destination_finish,
            })

    def finish_phase(name, previous_tail):
        worker_clocks = [process_clock[group] for group in range(groups)]
        tail = max([process_clock["cpu"]] + worker_clocks)
        phase_stats[name].update({
            "cpu_clock": process_clock["cpu"],
            "worker_clocks": worker_clocks,
            "tail_cycles": tail,
            "tail_increment_from_previous_snapshot": tail - previous_tail,
        })
        return tail

    previous_tail = 0
    start_phase("init")
    init_bytes = (4 + 2 * groups) * 4
    for group in range(groups):
        send("init", "init(%d)" % group, "cpu", group, init_bytes)
    previous_tail = finish_phase("init", previous_tail)

    start_phase("dispatch")
    for source in range(groups):
        for destination in range(groups):
            assignments = matrix[source][destination]
            if source == destination or assignments <= 0:
                continue
            send("dispatch", "dispatch(%d,%d)" % (source, destination),
                 source, destination,
                 assignments * hidden * element_bytes,
                 token_assignments=assignments)
    previous_tail = finish_phase("dispatch", previous_tail)

    start_phase("compute")
    before_compute = [process_clock[group] for group in range(groups)]
    for group in range(groups):
        process_clock[group] += compute_cycles[group]
    phase_stats["compute"].update({
        "message_count": 0,
        "compute_cycles_by_worker": compute_cycles,
        "compute_tokens_by_worker": compute_tokens,
        "worker_clocks_before": before_compute,
    })
    previous_tail = finish_phase("compute", previous_tail)

    start_phase("combine")
    for source in range(groups):
        for destination in range(groups):
            assignments = matrix[source][destination]
            if source == destination or assignments <= 0:
                continue
            # The expert destination returns the hidden states to the original
            # logical source: reverse direction, same payload.
            send("combine", "combine(%d,%d)" % (source, destination),
                 destination, source,
                 assignments * hidden * element_bytes,
                 token_assignments=assignments)
    previous_tail = finish_phase("combine", previous_tail)

    start_phase("summary")
    for group in range(groups):
        send("summary", "summary(%d)" % group,
             group, "cpu", 4)
    previous_tail = finish_phase("summary", previous_tail)

    protocol_cycles = process_clock["cpu"]
    row_sums = [sum(row) for row in matrix]
    column_sums = [sum(matrix[source][destination]
                       for source in range(groups))
                   for destination in range(groups)]
    workers_out = []
    for group, worker in enumerate(workers):
        record = dict(worker)
        record["compute_tokens"] = compute_tokens[group]
        record["compute_cycles"] = compute_cycles[group]
        workers_out.append(record)

    output = {
        "schema_version": 1,
        "model": "legosim-ordered-protocol-max-plus-v1",
        "scope": (
            "fixed-layout q=1 no-contention PopNet envelope; stable logical "
            "worker IDs; not measured LEGOSim cycles"),
        "protocol_cycles": protocol_cycles,
        "cycles": protocol_cycles,
        "critical_process": "cpu-after-ordered-summaries",
        "config": {
            "groups": groups,
            "topk": int(params.get("meta", {}).get("topk", 1)),
            "tokens_per_source": tokens,
            "hidden": hidden,
            "element_bytes": element_bytes,
            "flit_payload_bytes": flit_payload_bytes,
            "operation_overhead_cycles": overhead,
            "hop_cycles": hop_cycles,
        },
        "semantics": {
            "worker_id": "logical group ID g for e(g)",
            "phase_order": ["init", "dispatch", "compute", "combine",
                            "summary"],
            "dispatch_order": "lexicographic source,destination",
            "combine_order": "same logical pair order, reverse direction",
            "endpoint_finish": (
                "src=write+F-1; dst=max(write+F-1+hop_cycles*hops,read)"),
            "local_self_traffic": "no network message; included in compute",
        },
        "router": router,
        "workers": workers_out,
        "gpu_profile": profile,
        "dispatch": {
            "matrix": matrix,
            "sha256": _canonical_hash(matrix),
            "row_sums": row_sums,
            "column_sums": column_sums,
            "target_column_sums": targets,
            "total_assignments": sum(row_sums),
        },
        "phases": phase_stats,
        "message_count": sum(phase_stats[name].get("message_count", 0)
                             for name in ("init", "dispatch", "combine",
                                          "summary")),
        "final_process_clocks": {
            "cpu": process_clock["cpu"],
            "workers": [process_clock[group] for group in range(groups)],
        },
    }
    if include_events:
        output["events"] = event_rows
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed-layout LEGOSim ordered-protocol surrogate")
    parser.add_argument("--params", required=True)
    parser.add_argument("--solution", required=True)
    parser.add_argument("--gpu-profile", default=None,
                        help="JSON, gpu_cycles.tsv, or group/type/cycles TSV; "
                             "default is the audited A1550 profile")
    parser.add_argument("--tokens-per-source", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--element-bytes", type=int, default=None,
                        help="default: params.meta.bytes_elem or 2")
    parser.add_argument("--flit-payload-bytes", type=int, default=32)
    parser.add_argument("--operation-overhead-cycles", type=int, default=1000)
    parser.add_argument("--hop-cycles", type=int, default=104)
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("-o", "--out", default=None,
                        help="write JSON here; omit to print JSON to stdout")
    args = parser.parse_args(argv)

    params = _read_json(args.params)
    solution = _read_json(args.solution)
    profile = (load_gpu_profile(args.gpu_profile)
               if args.gpu_profile else None)
    element_bytes = (args.element_bytes if args.element_bytes is not None
                     else int(params.get("meta", {}).get("bytes_elem", 2)))
    result = evaluate_protocol_surrogate(
        params, solution, gpu_profile=profile,
        tokens_per_source=args.tokens_per_source,
        hidden=args.hidden, element_bytes=element_bytes,
        flit_payload_bytes=args.flit_payload_bytes,
        operation_overhead_cycles=args.operation_overhead_cycles,
        hop_cycles=args.hop_cycles,
        include_events=args.include_events,
        profile_source=args.gpu_profile)
    if args.out:
        _write_json(args.out, result)
        print("[protocol-surrogate] cycles=%d  messages=%d  wrote %s" %
              (result["protocol_cycles"], result["message_count"], args.out))
    else:
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
