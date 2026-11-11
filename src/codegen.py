#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
codegen.py -- turn an ILP solution.json into LEGOSim artifacts.

Option-A coordinate flattening (design doc Section 6): our ILP works in a 3D
grid (l, i, j) with l in [0,L), i in [0,X), j in [0,Y). LEGOSim / popnet works
on a 2D mesh addressed by (row, col) node ids. We flatten:

        (l, i, j)  ->  row = i ,  col = l * Y + j
        node_id    =   row * (L*Y) + col

i.e. the L vertical layers are unrolled side-by-side into one wide 2D mesh of
shape X rows x (L*Y) cols.  The current LEGOSim/popnet runner does not consume
our model-side per-link delay reference, so L>1 is only a visualization/export
mapping and is not topology-equivalent to a physical 3D network.  Paper-grade
default-Mesh comparisons therefore use L=1, where Manhattan distance is exact.

Emitted into <out_dir>/:
  topology/mesh_<R>x<C>.gv   generic 2D mesh graphviz for popnet
  bench.yml                  phase1 (router+workers) + phase2 (popnet) manifest
  coords.json                task_id -> {node, row, col, layer, type} (the header)
  layout_report.txt          human-readable placement + predicted metrics

This is a *generator*: it does not require LEGOSim to be installed. The paths
inside bench.yml use $SIMULATOR_ROOT / $BENCHMARK_ROOT exactly like the stock
moe_e2e.yml so it drops into benchmark/All2all-MoE-E2E unchanged.
"""

import argparse
import json
import math
import os
import re


def flatten(l, i, j, L, X, Y):
    row = i
    col = l * Y + j
    C = L * Y
    return row, col, row * C + col


def resolve_router(sol, L, X, Y):
    """Return (l, i, j) for the baremetal router node, GUARANTEED valid for ANY
    solution (ILP or baseline).

    LEGOSim's moe_gpu_worker/moe_router_cpu require a 'router' entry in
    moe_coords.txt; without it every worker blocks forever waiting for a
    handshake that never arrives. ILP solutions carry a dedicated CPU site that
    serves as the router, but heuristic/floorplanner baselines may place all
    control tasks on GPU tiles and leave no CPU site at all. To keep all 30
    (workload x method) LEGOSim runs on identical footing -- a lightweight
    control-flow router on its own NoC node, off the makespan critical path --
    we resolve the router node in priority order:

      1. a CPU site (with or without tasks), or
      2. a free mesh node not occupied by any GPU worker (prefer (0,0)).

    Never returns a node shared with a GPU worker, so the router process gets
    its own NoC endpoint."""
    for s in sol["sites"]:
        if s["type"] == "CPU":
            return s["l"], s["i"], s["j"]
    # No CPU site: synthesize a router at a free mesh node.
    occupied = set()
    for s in sol["sites"]:
        if s["type"].startswith("GPU") and s["tasks"]:
            occupied.add((s["l"], s["i"], s["j"]))
    for i in range(X):
        for l in range(L):
            for j in range(Y):
                if (l, i, j) not in occupied:
                    return l, i, j
    # Degenerate fallback (grid full): reuse (0,0,0).
    return 0, 0, 0


def _expert_group(task_id):
    """Return the numeric logical group carried by an ``e(g)`` task."""
    match = re.match(r"^e\((\d+)\)$", task_id)
    return int(match.group(1)) if match else None


def gpu_worker_sites(sol, params):
    """Return the GPU sites that own MoE experts, in the one canonical worker
    order used by bench.yml, moe_coords.txt, and dispatch_matrix.txt.

    Older code started a worker for every GPU site with any task and only wrote
    the e(g) ownership in a comment.  The C++ router therefore had no way to
    honor the ILP's expert-to-chiplet mapping.  A formal MoE run has one worker
    per expert-bearing site; control-only tasks are handled by the router
    process and are not fake expert workers.

    Worker IDs are protocol IDs, not placement ranks.  For the current q=1
    formal ABI, worker ``g`` owns logical expert group ``e(g)`` regardless of
    that expert's coordinates.  Sorting by numeric logical group here keeps
    dispatch rows, combine columns, bench process IDs, and coordinate-file IDs
    stable when a placer moves experts.  The coordinate key is only a
    deterministic fallback for non-MoE workloads whose task names have no
    ``e(g)`` identity.
    """
    expert_tasks = set(params.get("expert_tasks", []))
    sites = [
        s for s in sol["sites"]
        if s["type"].startswith("GPU")
        and any(t in expert_tasks for t in s.get("tasks", []))
    ]
    if not sites:
        # Backward-compatible fallback for non-MoE inputs.
        sites = [
            s for s in sol["sites"]
            if s["type"].startswith("GPU") and s.get("tasks")
        ]

    def worker_key(site):
        logical_groups = sorted(
            group for group in
            (_expert_group(task) for task in site.get("tasks", []))
            if group is not None)
        if logical_groups:
            return (0, tuple(logical_groups), site["l"], site["i"], site["j"],
                    site["type"], ",".join(site.get("tasks", [])))
        return (1, (), site["l"], site["i"], site["j"], site["type"],
                ",".join(site.get("tasks", [])))

    return sorted(
        sites,
        key=worker_key)


def write_mesh_gv(path, R, C):
    lines = ["graph mesh_%dx%d" % (R, C), "{",
             "    edge[weight=100]",
             "    node[pipeline_stage_delay=1]"]
    # horizontal chains
    for r in range(R):
        row_nodes = [str(r * C + c) for c in range(C)]
        lines.append("    " + "--".join(row_nodes))
    # vertical chains
    for c in range(C):
        col_nodes = [str(r * C + c) for r in range(R)]
        lines.append("    " + "--".join(col_nodes))
    lines.append("}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# worker command templates keyed by chiplet type -> (binary, simulator hint)
WORKER_CMD = {
    "GPU-L": ("$BENCHMARK_ROOT/build/bin/moe_gpu_worker", "gpgpu"),
    "GPU-S": ("$BENCHMARK_ROOT/build/bin/moe_gpu_worker", "gpgpu"),
    "CPU":   ("$SIMULATOR_ROOT/snipersim/run-sniper", "sniper"),
}
# GPGPU-Sim config per GPU class (large vs small SM count)
GPU_CFG = {
    "GPU-L": "$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM7_TITANV/*",
    "GPU-S": "$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM6_TITANX/*",
}


def emit_bench_yaml(sol, params, R, C, out_dir, router_mode="baremetal",
                    coord_file="$BENCHMARK_ROOT/moe_coords.txt",
                    sim_cfg=None):
    L = sol["grid"]["L"]; X = sol["grid"]["X"]; Y = sol["grid"]["Y"]
    meta = params.get("meta", {})
    workers = gpu_worker_sites(sol, params)
    ngpu = len(workers)
    sim_cfg = sim_cfg or {}
    # Resolve the router node robustly (CPU site or a free mesh node) so the
    # bench manifest ALWAYS starts a router process -- baseline layouts with no
    # CPU site would otherwise deadlock every GPU worker.
    r_l, r_i, r_j = resolve_router(sol, L, X, Y)
    router = {"l": r_l, "i": r_i, "j": r_j}

    lines = []
    lines.append("# Auto-generated by codegen.py from an ILP layout solution.")
    lines.append("# config=%s experts=%s groups=%s topk=%s"
                 % (meta.get("config"), meta.get("experts"),
                    meta.get("groups"), meta.get("topk")))
    lines.append("# 3D grid (L=%d,X=%d,Y=%d) flattened to %dx%d 2D mesh."
                 % (L, X, Y, R, C))
    lines.append("# router_mode=%s  coord_file=%s" % (router_mode, coord_file))
    lines.append("# NOTE: workers/router read layout coords from $MOE_COORD_FILE")
    lines.append("#       (set it to %s before running interchiplet)." % coord_file)
    lines.append("phase1:")

    # router: 'sniper' (design-doc canonical CPU model) or 'baremetal' (router
    # runs directly without Sniper -- used on hosts where Sniper's Pin/SIFT is
    # broken; the router is a lightweight gating process off the makespan
    # critical path, documented honestly in the design doc).
    if router is not None:
        rr, rc, rn = flatten(router["l"], router["i"], router["j"], L, X, Y)
        experts = meta.get("experts", 8)
        topk = meta.get("topk", 2)
        skew = meta.get("skew", 1.0)
        # NoC/compute scale (tokens_per_gpu, hidden, inter). Defaults are the
        # smoke values GPGPU-Sim runs fastest on; env overrides let us dial the
        # simulated scale toward production so that inter-chiplet all-to-all
        # traffic (which grows as tokens*hidden) becomes a meaningful fraction
        # of makespan and the LAYOUT (hop distance) actually matters. The SAME
        # scale is applied to every method/workload so speedup ratios stay a
        # fair, apples-to-apples comparison.
        tok = str(sim_cfg.get(
            "tokens_per_source", os.environ.get("MOE_SIM_TOKENS", "4")))
        hid = str(sim_cfg.get(
            "hidden", os.environ.get("MOE_SIM_HIDDEN", "128")))
        itr = str(sim_cfg.get(
            "intermediate", os.environ.get("MOE_SIM_INTER", "64")))
        # router argv: cpu_x cpu_y ngpu tokens_per_gpu experts topk hidden inter alpha
        # (cpu_x/cpu_y/ngpu are overridden by the coord file at runtime).
        rargs = ('"%d", "%d", "%d", "%s", "%d", "%d", "%s", "%s", "%.2f"'
                 % (rc, rr, ngpu, tok, max(experts, ngpu), topk, hid, itr, skew))
        if router_mode == "sniper":
            lines.append('  - cmd: "$SIMULATOR_ROOT/snipersim/run-sniper"')
            lines.append('    args: ["--", "$BENCHMARK_ROOT/build/bin/moe_router_cpu", '
                         + rargs + ']')
            lines.append('    log: "sniper.router.%d.%d.log"' % (rc, rr))
            lines.append('    is_to_stdout: false')
            lines.append('    clock_rate: 3')
        else:  # baremetal
            lines.append('  - cmd: "$BENCHMARK_ROOT/build/bin/moe_router_cpu"')
            lines.append('    args: [' + rargs + ']')
            lines.append('    log: "router.%d.%d.log"' % (rc, rr))
            lines.append('    is_to_stdout: false')
            lines.append('    clock_rate: 1')

    # GPU/expert workers
    widx = 0
    repeat = int(sim_cfg.get(
        "ffn_repeat", os.environ.get("MOE_SIM_FFN_REPEAT", "2")))
    for s in workers:
        rr, rc, rn = flatten(s["l"], s["i"], s["j"], L, X, Y)
        binp = WORKER_CMD[s["type"]][0]
        lines.append('  - cmd: "%s"' % binp)
        lines.append('    args: ["%d", "%d", "%d", "%d"]  # worker_id col row repeat  (tasks: %s)'
                     % (widx, rc, rr, repeat, ",".join(s["tasks"])))
        lines.append('    log: "gpgpusim.0.%d.log"' % widx)
        lines.append('    is_to_stdout: false')
        lines.append('    clock_rate: 1')
        lines.append('    pre_copy: "%s"' % GPU_CFG.get(s["type"], GPU_CFG["GPU-S"]))
        widx += 1

    # phase2: popnet on the flattened mesh
    A = R * C
    # Per-hop link length (wire latency).  Default 1000 reproduces the nominal
    # compute-bound operating point; MOE_POPNET_LINK_LEN raises it to move into
    # the latency/communication-bound (decode/online-serving) regime.
    link_len = int(os.environ.get("MOE_POPNET_LINK_LEN", "1000"))
    lines.append("")
    lines.append("phase2:")
    lines.append('  - cmd: "$SIMULATOR_ROOT/popnet_chiplet/build/popnet"')
    lines.append('    args: ["-A", "%d", "-c", "1", "-V", "3", "-B", "12", '
                 '"-O", "12", "-F", "4", "-L", "%d", "-T", "1000000000000", '
                 '"-r", "1", "-I", "../bench.txt", "-R", "4", '
                 '"-G", "$BENCHMARK_ROOT/topology/mesh_%dx%d.gv", '
                 '"-D", "../delayInfo.txt", "-P"]' % (A, link_len, R, C))
    lines.append('    log: "popnet_0.log"')
    lines.append('    is_to_stdout: false')
    lines.append('    clock_rate: 1')
    lines.append("")
    lines.append('bench_file: "./bench.txt"')
    lines.append('delayinfo_file: "./delayInfo.txt"')

    with open(os.path.join(out_dir, "bench.yml"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def emit_coords(sol, params, R, C, out_dir):
    L = sol["grid"]["L"]; X = sol["grid"]["X"]; Y = sol["grid"]["Y"]
    coords = {"grid": sol["grid"], "mesh": {"rows": R, "cols": C},
              "flatten": "row=i, col=l*Y+j, node=row*(L*Y)+col",
              "tasks": {}, "sites": []}
    for s in sol["sites"]:
        rr, rc, rn = flatten(s["l"], s["i"], s["j"], L, X, Y)
        coords["sites"].append({"layer": s["l"], "i": s["i"], "j": s["j"],
                                "row": rr, "col": rc, "node": rn,
                                "type": s["type"], "tasks": s["tasks"]})
        for t in s["tasks"]:
            coords["tasks"][t] = {"node": rn, "row": rr, "col": rc,
                                  "layer": s["l"], "type": s["type"]}
    with open(os.path.join(out_dir, "coords.json"), "w") as fh:
        json.dump(coords, fh, indent=2)


def emit_delay_info(sol, params, R, C, out_dir):
    """Write a model-side per-link delay reference.

    The current run_ilp_e2e.sh/popnet flow produces its own delayInfo.txt and
    does not consume this file.  It is retained for audit/debugging only.  L=1
    experiments do not need a vertical-delay override and directly match the
    generated 2D Mesh.
    """
    ic = params["interconnect"]
    L = sol["grid"]["L"]; Y = sol["grid"]["Y"]
    tau_h = ic["tau_h"]; tau_v = ic["tau_v"]
    # scale to integer cycles; popnet delays are relative, keep >=1
    def dly(is_vertical):
        base = tau_v if is_vertical else tau_h
        return max(1, int(round(base * 100)))  # 100 = per-hop unit
    lines = [
        "# MODEL-SIDE REFERENCE ONLY; current popnet runner does not read this.",
        "# src dst delay  (auto: cross-layer=tau_v, in-layer=tau_h)",
    ]
    for r in range(R):
        for c in range(C):
            n = r * C + c
            # right neighbor
            if c + 1 < C:
                m = r * C + (c + 1)
                # a horizontal step that crosses a layer boundary: col//Y differs
                is_v = (c // Y) != ((c + 1) // Y)
                d = dly(is_v)
                lines.append("%d %d %d" % (n, m, d))
                lines.append("%d %d %d" % (m, n, d))
            # down neighbor (always in-layer, same col)
            if r + 1 < R:
                m = (r + 1) * C + c
                d = dly(False)
                lines.append("%d %d %d" % (n, m, d))
                lines.append("%d %d %d" % (m, n, d))
    with open(os.path.join(out_dir, "linkDelay.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def emit_coord_file(sol, params, R, C, out_dir):
    """Emit moe_coords.txt consumed by moe_router_cpu / moe_gpu_worker via
    $MOE_COORD_FILE (design doc Section 7, Task 3). This is what actually makes
    the ILP layout drive the NoC: without it the binaries fall back to the
    legacy hard-coded 4-GPU column.

    InterChiplet linearizes a NoC node as node = x + y*width (net_bench.h,
    width == interchiplet's -w flag). Our Option-A flatten gives
    (l,i,j)->row=i,col=l*Y+j,node=row*C+col with C=L*Y. Equating node numbers
    forces  x=col, y=row, width=C. We therefore write width=C and, per site,
    (x,y)=(col,row)."""
    L = sol["grid"]["L"]; X = sol["grid"]["X"]; Y = sol["grid"]["Y"]
    # Resolve router robustly (CPU site or a free mesh node); baseline layouts
    # may have no CPU site, in which case we place the router on a free node so
    # the coord file ALWAYS carries a 'router' line (workers deadlock without it).
    r_l, r_i, r_j = resolve_router(sol, L, X, Y)
    router = {"l": r_l, "i": r_i, "j": r_j}

    gpu_sites = gpu_worker_sites(sol, params)
    lines = ["# Auto-generated by codegen.py -- ILP layout -> NoC coordinates.",
             "# Consumed by moe_router_cpu / moe_gpu_worker via $MOE_COORD_FILE.",
             "# node = x + y*width  (InterChiplet), width = C = L*Y = %d." % C,
             "# Mapping: x=col=l*Y+j, y=row=i.",
             "width %d" % C]
    if router is not None:
        rr, rc, rn = flatten(router["l"], router["i"], router["j"], L, X, Y)
        lines.append("router %d %d   # node=%d (l=%d,i=%d,j=%d)"
                     % (rc, rr, rn, router["l"], router["i"], router["j"]))
    lines.append("ngpu %d" % len(gpu_sites))
    for wid, s in enumerate(gpu_sites):
        rr, rc, rn = flatten(s["l"], s["i"], s["j"], L, X, Y)
        lines.append("gpu %d %d %d   # node=%d type=%s tasks=%s"
                     % (wid, rc, rr, rn, s["type"], ",".join(s["tasks"])))
    with open(os.path.join(out_dir, "moe_coords.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _largest_remainder(weights, total):
    """Deterministically apportion integer assignments while preserving sum."""
    if total < 0:
        raise ValueError("assignment total must be non-negative")
    if not weights:
        return []
    norm = float(sum(weights))
    if norm <= 0:
        weights = [1.0] * len(weights)
        norm = float(len(weights))
    exact = [total * float(w) / norm for w in weights]
    base = [int(math.floor(v)) for v in exact]
    left = total - sum(base)
    order = sorted(range(len(weights)),
                   key=lambda i: (-(exact[i] - base[i]), i))
    for i in order[:left]:
        base[i] += 1
    return base


def _balanced_transport(column_totals, nrows, row_total):
    """Build a deterministic integer matrix with exact row/column margins.

    Columns are logical expert groups and rows are logical source groups.  A
    full quotient is first placed in every row; each column's remainder is
    then assigned by one continuous cyclic cursor.  Continuing the cursor
    across columns (instead of restarting at source zero) makes every row sum
    exact whenever ``sum(column_totals) == nrows * row_total``.
    """
    if nrows <= 0:
        raise ValueError("transport row count must be positive")
    if row_total < 0:
        raise ValueError("transport row total must be non-negative")
    totals = [int(value) for value in column_totals]
    if any(value < 0 for value in totals):
        raise ValueError("transport column totals must be non-negative")
    expected_total = nrows * row_total
    if sum(totals) != expected_total:
        raise ValueError(
            "transport margins disagree: columns sum to %d, rows require %d"
            % (sum(totals), expected_total))

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

    row_sums = [sum(row) for row in matrix]
    actual_columns = [
        sum(matrix[row][column] for row in range(nrows))
        for column in range(len(totals))
    ]
    if row_sums != [row_total] * nrows or actual_columns != totals:
        raise AssertionError("deterministic transport failed to preserve margins")
    return matrix


def emit_dispatch_matrix(sol, params, out_dir, sim_cfg):
    """Emit a formal, deterministic dispatch trace that consumes the ILP's
    expert-to-worker assignment.

    Each logical source group contributes the same fixed number of tokens.
    First, one global largest-remainder apportionment fixes the destination
    group totals.  A deterministic balanced transport then constructs the
    logical source-group x destination-group matrix with exact row and column
    margins.  Only after that is the logical matrix permuted on both axes via
    the solution's e(g)->worker ownership.  Thus layouts see byte-identical
    logical traffic even when their physical coordinates differ.  Under the
    current q=1 ABI, canonical worker ID equals logical group ID, so the
    emitted protocol matrix itself is coordinate-invariant.
    """
    meta = params.get("meta", {})
    workers = gpu_worker_sites(sol, params)
    ngpu = len(workers)
    if ngpu <= 0:
        raise ValueError("formal MoE dispatch requires at least one GPU worker")
    if ngpu > 128:
        raise ValueError(
            "formal LEGOSim MoE ABI supports at most 128 GPU workers; got %d"
            % ngpu)

    owner = {}
    for wid, site in enumerate(workers):
        for task_id in site.get("tasks", []):
            group = _expert_group(task_id)
            if group is not None:
                if group in owner:
                    raise ValueError("expert group %d has multiple workers" % group)
                owner[group] = wid

    groups = int(meta.get("groups", len(params.get("expert_tasks", []))))
    expected = list(range(groups))
    missing = [g for g in expected if g not in owner]
    if missing:
        raise ValueError(
            "solution does not map every expert group to a worker; missing %s"
            % missing)

    # A physical source worker has one fixed tokens-per-source budget.  A
    # stable logical-source interpretation is therefore unambiguous only for
    # a one-to-one group/worker ownership relation.  Fail closed instead of
    # letting a solution-dependent worker order change the logical workload.
    worker_groups = [[] for _ in range(ngpu)]
    for group in expected:
        worker_groups[owner[group]].append(group)
    non_bijective = [
        {"worker": wid, "groups": owned}
        for wid, owned in enumerate(worker_groups) if len(owned) != 1
    ]
    if groups != ngpu or non_bijective:
        raise ValueError(
            "formal dispatch requires exactly one logical expert group per "
            "GPU worker (groups=%d, workers=%d, non_bijective=%s)"
            % (groups, ngpu, non_bijective))

    # gpu_worker_sites() is the single ordering authority used by the worker
    # processes, coordinate file, and protocol matrix.  With one logical group
    # per worker and contiguous groups 0..G-1, its logical sort must make the
    # mapping the identity.  Fail closed if a future code change silently
    # reintroduces a placement-derived protocol ID.
    unstable_ids = [group for group in expected if owner[group] != group]
    if unstable_ids:
        raise AssertionError(
            "logical worker IDs are not stable for groups %s" % unstable_ids)

    topk = int(meta.get("topk", 1))
    tokens = int(sim_cfg["tokens_per_source"])
    hidden = int(sim_cfg["hidden"])
    intermediate = int(sim_cfg["intermediate"])
    repeat = int(sim_cfg["ffn_repeat"])
    for name, value in (
            ("tokens_per_source", tokens), ("topk", topk),
            ("hidden", hidden), ("intermediate", intermediate),
            ("ffn_repeat", repeat)):
        if value <= 0:
            raise ValueError("%s must be positive" % name)

    weights = list(meta.get("group_pi", []))
    if len(weights) != groups:
        weights = [1.0] * groups
    assignments_per_source = tokens * topk
    total_assignments = ngpu * assignments_per_source
    global_group_targets = _largest_remainder(weights, total_assignments)
    logical_matrix = _balanced_transport(
        global_group_targets, groups, assignments_per_source)

    # Map logical source and destination groups to protocol worker IDs.  This
    # is intentionally the identity for q=1, but retaining the explicit owner
    # lookup documents the ABI and keeps the invariant auditable.
    matrix = [[0] * ngpu for _ in range(ngpu)]
    for source_group in expected:
        source_worker = owner[source_group]
        for destination_group in expected:
            destination_worker = owner[destination_group]
            matrix[source_worker][destination_worker] = (
                logical_matrix[source_group][destination_group])

    column_totals = [
        sum(matrix[r][c] for r in range(ngpu)) for c in range(ngpu)
    ]
    logical_actual = [
        column_totals[owner[group]] for group in expected
    ]
    logical_error = [
        actual - target for actual, target in
        zip(logical_actual, global_group_targets)
    ]
    zero_workers = [
        worker for worker, count in enumerate(column_totals) if count == 0
    ]
    if zero_workers:
        raise ValueError(
            "formal dispatch produced globally zero-token GPU workers: %s"
            % zero_workers)

    lines = [
        "# Auto-generated model-derived deterministic MoE dispatch.",
        "# File-backed/reproducible; not a captured production-model trace.",
        "# Runtime synthetic Zipf fallback is disabled.",
        "# Logical group traffic is identical across layouts; rows and",
        "# columns use stable logical worker IDs: worker g owns e(g).",
        "ngpu %d" % ngpu,
        "experts %d" % int(meta.get("experts", groups)),
        "topk %d" % topk,
        "n_rounds 1",
        "tokens_per_source %d" % tokens,
        "sim_hidden %d" % hidden,
        "sim_intermediate %d" % intermediate,
        "sim_ffn_repeat %d" % repeat,
        "round 0 layer all phase all",
    ]
    lines.extend(" ".join(str(v) for v in values) for values in matrix)
    with open(os.path.join(out_dir, "dispatch_matrix.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    dispatch_meta = {
        "trace_kind": "model_derived_deterministic",
        "source": "global largest-remainder apportionment of workload "
                  "meta.group_pi followed by deterministic cyclic balanced "
                  "transport in logical group space",
        "dispatch_algorithm": "global_largest_remainder_cyclic_transport_v1",
        "logical_matrix_axes": "source_group_x_destination_group",
        "source_rows_constructed_individually": True,
        "per_source_rows_replicated": False,
        "physical_mapping": "matrix[owner(src_group)][owner(dst_group)]",
        "worker_id_semantics": "worker_id_equals_logical_group_id_q1",
        "protocol_matrix_axes": "source_worker_group_x_destination_worker_group",
        "runtime_sampling": False,
        "note": "file-backed and reproducible, but not a trace captured from "
                "a deployed model",
        "ngpu": ngpu,
        "groups": groups,
        "topk": topk,
        "tokens_per_source": tokens,
        "total_route_assignments": sum(sum(values) for values in matrix),
        "sim_hidden": hidden,
        "sim_intermediate": intermediate,
        "sim_ffn_repeat": repeat,
        "per_source_route_assignments": [sum(row) for row in matrix],
        "global_group_target_assignments": global_group_targets,
        "global_group_actual_assignments": logical_actual,
        "global_group_error_assignments": logical_error,
        "global_group_max_abs_error": max(
            [abs(value) for value in logical_error] or [0]),
        "group_to_worker": {str(g): owner[g] for g in expected},
        "worker_token_assignments": column_totals,
        "zero_worker_count": len(zero_workers),
        "zero_workers": zero_workers,
        "workers": [
            {
                "worker": wid,
                "type": site["type"],
                "site": [site["l"], site["i"], site["j"]],
                "tasks": list(site.get("tasks", [])),
            }
            for wid, site in enumerate(workers)
        ],
    }
    with open(os.path.join(out_dir, "dispatch_meta.json"), "w") as fh:
        json.dump(dispatch_meta, fh, indent=2)
    return dispatch_meta


def emit_report(sol, params, R, C, out_dir, sim_cfg=None):
    L = sol["grid"]["L"]; X = sol["grid"]["X"]; Y = sol["grid"]["Y"]
    lines = []
    lines.append("ILP 3D-chiplet MoE layout -> LEGOSim mapping report")
    lines.append("=" * 55)
    m = params.get("meta", {})
    lines.append("workload: config=%s experts=%s groups=%s topk=%s d_model=%s d_ff=%s"
                 % (m.get("config"), m.get("experts"), m.get("groups"),
                    m.get("topk"), m.get("d_model"), m.get("d_ff")))
    lines.append("grid: L=%d X=%d Y=%d  -> flattened mesh %dx%d (%d nodes)"
                 % (L, X, Y, R, C, R * C))
    if L > 1:
        lines.append(
            "WARNING: L>1 flattening is not a physical-3D-equivalent popnet "
            "topology; use L=1 for default-Mesh experiments.")
    lines.append("predicted makespan (cycles): %.1f"
                 % sol.get("predicted_makespan_cycles", -1))
    raw_gap = sol.get("mip_gap")
    gap_text = ("%.4f%%" % (float(raw_gap) * 100.0)
                if raw_gap is not None else "uncertified")
    lines.append("MIP gap: %s   solver wall: %.2fs"
                 % (gap_text, sol.get("solver_wall_s", 0)))
    lines.append("energy total: %.3e J (comm %.3e + comp %.3e)"
                 % (sol.get("energy_total_J", 0), sol.get("energy_comm_J", 0),
                    sol.get("energy_comp_J", 0)))
    lines.append("area: %.0f mm2   peak power: %.0f W   byte-hop: %.3e"
                 % (sol.get("total_area_mm2", 0), sol.get("peak_power_W", 0),
                    sol.get("byte_hop", 0)))
    if sim_cfg:
        lines.append(
            "LEGOSim input: tokens/source=%d hidden=%d intermediate=%d "
            "ffn_repeat=%d"
            % (sim_cfg["tokens_per_source"], sim_cfg["hidden"],
               sim_cfg["intermediate"], sim_cfg["ffn_repeat"]))
    lines.append("")
    lines.append("%-4s %-6s %-14s %-6s %s" %
                 ("node", "type", "(l,i,j)", "(r,c)", "tasks"))
    for s in sol["sites"]:
        rr, rc, rn = flatten(s["l"], s["i"], s["j"], L, X, Y)
        lines.append("%-4d %-6s (%d,%d,%d)%s %-6s %s" %
                     (rn, s["type"], s["l"], s["i"], s["j"],
                      " " * (14 - len("(%d,%d,%d)" % (s["l"], s["i"], s["j"]))),
                      "(%d,%d)" % (rr, rc), ",".join(s["tasks"])))
    with open(os.path.join(out_dir, "layout_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="ILP solution -> LEGOSim artifacts")
    ap.add_argument("--params", required=True)
    ap.add_argument("--sol", required=True)
    ap.add_argument("--out-dir", default="out/legosim")
    ap.add_argument("--router-mode", choices=["baremetal", "sniper"],
                    default="baremetal",
                    help="baremetal: router runs without Sniper (off the "
                         "makespan critical path; use where Sniper's Pin/SIFT "
                         "is unavailable). sniper: design-doc canonical CPU "
                         "model via run-sniper.")
    ap.add_argument("--sim-tokens-per-source", type=int,
                    default=int(os.environ.get("MOE_SIM_TOKENS", "4")),
                    help="fixed tokens contributed by each source worker")
    ap.add_argument("--sim-hidden", type=int,
                    default=int(os.environ.get("MOE_SIM_HIDDEN", "128")))
    ap.add_argument("--sim-intermediate", type=int,
                    default=int(os.environ.get("MOE_SIM_INTER", "64")))
    ap.add_argument("--sim-ffn-repeat", type=int,
                    default=int(os.environ.get("MOE_SIM_FFN_REPEAT", "2")))
    a = ap.parse_args()

    params = json.load(open(a.params))
    sol = json.load(open(a.sol))
    L = sol["grid"]["L"]; X = sol["grid"]["X"]; Y = sol["grid"]["Y"]
    R, C = X, L * Y
    sim_cfg = {
        "tokens_per_source": a.sim_tokens_per_source,
        "hidden": a.sim_hidden,
        "intermediate": a.sim_intermediate,
        "ffn_repeat": a.sim_ffn_repeat,
    }
    for name, value in sim_cfg.items():
        if value <= 0:
            ap.error("--sim-%s must be positive" % name.replace("_", "-"))

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "topology"), exist_ok=True)

    write_mesh_gv(os.path.join(a.out_dir, "topology",
                               "mesh_%dx%d.gv" % (R, C)), R, C)
    emit_bench_yaml(sol, params, R, C, a.out_dir,
                    router_mode=a.router_mode, sim_cfg=sim_cfg)
    emit_coords(sol, params, R, C, a.out_dir)
    emit_coord_file(sol, params, R, C, a.out_dir)
    dispatch_meta = emit_dispatch_matrix(
        sol, params, a.out_dir, sim_cfg=sim_cfg)
    emit_delay_info(sol, params, R, C, a.out_dir)
    report = emit_report(sol, params, R, C, a.out_dir, sim_cfg=sim_cfg)

    print(report)
    print("\n[codegen] wrote artifacts to %s/  (router_mode=%s)"
          % (a.out_dir, a.router_mode))
    print("[codegen]   bench.yml, coords.json, moe_coords.txt, linkDelay.txt,")
    print("[codegen]   dispatch_matrix.txt, dispatch_meta.json "
          "(%d route assignments)" %
          dispatch_meta["total_route_assignments"])
    print("[codegen]   layout_report.txt, topology/mesh_%dx%d.gv" % (R, C))
    print("[codegen]   RUN: export MOE_COORD_FILE=%s/moe_coords.txt  before"
          " interchiplet" % os.path.abspath(a.out_dir))


if __name__ == "__main__":
    main()
