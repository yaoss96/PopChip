#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit and aggregate paper-grade scaling LEGOSim replay evidence.

This program never edits a replay directory.  It reads the ``results.json``
files produced by :mod:`replay_legosim_solutions`, independently checks every
mirrored evidence bundle, and writes a compact JSON/CSV report.  A row is
publishable only when all synchronization, dispatch, worker, PopNet, and
cryptographic mirror checks pass.

The reported ``legosim_cycles`` is deliberately the synchronized end-to-end
makespan, not the earlier maximum timestamp in ``bench.txt``.  Likewise,
``orion_noc_power_W`` is only PopNet/ORION's NoC power; it is not package or
wall-plug power.  Nominal dynamic NoC energy uses the final packet completion
in ``delayInfo.txt`` at a nominal 1-GHz clock, rather than multiplying power by
the end-to-end application makespan.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath


DEFAULT_EXPERTS = (8, 16, 32, 64, 128)
# Public/reporting names.  ``ilp_power`` is optional until its replays exist;
# callers can require it with ``--expected-methods``.  Historical replay
# manifests used ``ilp``, so aliases are normalized before completeness and
# duplicate checks instead of forcing old evidence to be renamed.
SUPPORTED_METHODS = ("ilp_perf", "ilp_power", "floorplet", "tap")
DEFAULT_METHODS = ("ilp_perf", "floorplet", "tap")
METHOD_ALIASES = {
    "ilp": "ilp_perf",
    "ilp-perf": "ilp_perf",
    "ilp_perf": "ilp_perf",
    "ilp-power": "ilp_power",
    "ilp_power": "ilp_power",
    "floorplet": "floorplet",
    "floorplet-perf": "floorplet",
    "floorplet_perf": "floorplet",
    "tap": "tap",
    "tap-spread": "tap",
    "tap_spread": "tap",
}
CASE_RE = re.compile(r"(?:^|_)e(?P<experts>[0-9]+)(?:_|$)", re.IGNORECASE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CSV_FIELDS = [
    "case_id",
    "method",
    "method_raw",
    "experts",
    "mesh",
    "cpu_count",
    "gpu_l_count",
    "gpu_s_count",
    "gpu_total_count",
    "auxiliary_gpu_count",
    "solution_gpu_chiplets",
    "audit_status",
    "legosim_validation",
    "legosim_cycles",
    "bench_cycles",
    "completion_delta_cycles",
    "gpu_cycle_max",
    "gpu_cycle_rows",
    "orion_noc_power_W",
    "noc_active_cycles",
    "nominal_orion_dynamic_noc_energy_uJ",
    "popnet_average_delay",
    "popnet_total_finished",
    "popnet_expected_finished",
    "rounds_requested",
    "n_gpu_workers",
    "gpu_init_bytes_per_worker",
    "simulator_binaries_immutable",
    "mirror_hashes_verified",
    "evidence_dir",
    "results_json",
    "errors",
]


def normalize_method(value):
    """Return the stable reporting name while retaining unknown labels.

    Unknown labels are returned in a lowercase/underscore form so the normal
    unexpected-method check can report them.  This keeps the audit fail closed
    without breaking historical ``ilp``/proxy result manifests.
    """
    raw = str(value or "").strip().lower()
    return METHOD_ALIASES.get(raw, raw.replace("-", "_"))

MIRROR_REQUIRED_MEMBERS = {
    "bench.txt",
    "command.txt",
    "delayInfo.txt",
    "dispatch_matrix.diff",
    "effective_dispatch_matrix.log",
    "expected_dispatch_matrix.log",
    "git_revisions.txt",
    "gpu_cycles.tsv",
    "harness_sources.sha256",
    "ilp_e2e_summary.txt",
    "interchiplet.stderr",
    "interchiplet.stdout",
    "raw_output_manifest.sha256",
    "run_evidence.tar.gz",
    "run_manifest.txt",
    "runner_exit_code.txt",
    "simulator_binaries.sha256",
    "simulator_binaries.end.sha256",
    "simulator_binaries.diff",
    "source_input_manifest.sha256",
    "staged_input_manifest.sha256",
}


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "Verify mirrored evidence for the 8--128 expert LEGOSim "
            "scaling replay and export only audited synchronized cycles."))
    ap.add_argument(
        "input_root",
        help=(
            "final scaling directory containing per-size results.json files, "
            "or one results.json file"))
    ap.add_argument(
        "--out-prefix", default=None,
        help=(
            "output prefix (default: <input_root>/scaling_legosim_audit); "
            ".json and .csv are appended"))
    ap.add_argument(
        "--expected-experts", nargs="+", type=int,
        default=list(DEFAULT_EXPERTS),
        help="required expert counts (default: 8 16 32 64 128)")
    ap.add_argument(
        "--expected-methods", nargs="+", default=list(DEFAULT_METHODS),
        help=(
            "required normalized method names per size (default: ilp_perf "
            "floorplet tap; add ilp_power after those replays exist)"))
    ap.add_argument(
        "--expected-rounds", type=int, default=2,
        help="required requested/completed Phase-1/Phase-2 rounds (default: 2)")
    return ap.parse_args(argv)


def _load_json(path):
    with open(str(path), "r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object: %s" % path)
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _parse_key_values(path):
    values = {}
    with open(str(path), "r", encoding="utf-8", errors="strict") as fh:
        for number, raw in enumerate(fh, 1):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            if "=" not in line:
                raise ValueError("%s:%d is not key=value" % (path, number))
            key, value = line.split("=", 1)
            if not key or key in values:
                raise ValueError("%s:%d has an empty/duplicate key" %
                                 (path, number))
            values[key] = value
    return values


def _integer(value, label, errors, minimum=None):
    try:
        # Reject a silent truncation such as int(2.5).
        text = str(value).strip()
        if not re.match(r"^-?[0-9]+$", text):
            raise ValueError
        result = int(text)
    except (TypeError, ValueError):
        errors.append("%s is not an integer: %r" % (label, value))
        return None
    if minimum is not None and result < minimum:
        errors.append("%s=%d is below %d" % (label, result, minimum))
    return result


def _number(value, label, errors, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError):
        errors.append("%s is not numeric: %r" % (label, value))
        return None
    if positive and not result > 0.0:
        errors.append("%s=%r is not positive" % (label, value))
    return result


def _same(label, actual, expected, errors):
    if actual != expected:
        errors.append("%s=%r, expected %r" % (label, actual, expected))


def _resolve_record_path(value, results_path, fallback=None):
    if value:
        path = Path(str(value))
        if not path.is_absolute():
            path = results_path.parent / path
        return path.resolve()
    return Path(fallback).resolve() if fallback is not None else None


def _manifest_entries(path):
    entries = []
    seen = set()
    with open(str(path), "r", encoding="utf-8", errors="strict") as fh:
        for number, raw in enumerate(fh, 1):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            match = re.match(r"^([0-9A-Fa-f]{64})  (.+)$", line)
            if not match:
                raise ValueError("%s:%d is not a sha256sum record" %
                                 (path, number))
            expected = match.group(1).lower()
            raw_name = match.group(2)
            if raw_name.startswith("./"):
                raw_name = raw_name[2:]
            posix = PurePosixPath(raw_name)
            if (not raw_name or posix.is_absolute() or
                    any(part in ("", ".", "..") for part in posix.parts)):
                raise ValueError("%s:%d has unsafe path %r" %
                                 (path, number, raw_name))
            normalized = "/".join(posix.parts)
            if normalized in seen:
                raise ValueError("%s:%d duplicates %s" %
                                 (path, number, normalized))
            seen.add(normalized)
            entries.append((normalized, expected))
    if not entries:
        raise ValueError("empty hash manifest: %s" % path)
    return entries


def _verify_mirror(evidence_dir, errors):
    complete = evidence_dir / "MIRROR_COMPLETE"
    if not complete.is_file():
        errors.append("missing MIRROR_COMPLETE")
    else:
        try:
            _same("MIRROR_COMPLETE", complete.read_text(
                encoding="utf-8").strip(), "complete", errors)
        except (OSError, UnicodeError) as ex:
            errors.append("cannot read MIRROR_COMPLETE: %s" % ex)

    manifest = evidence_dir / "mirror_manifest.sha256"
    if not manifest.is_file():
        errors.append("missing mirror_manifest.sha256")
        return 0
    try:
        entries = _manifest_entries(manifest)
    except (OSError, UnicodeError, ValueError) as ex:
        errors.append("invalid mirror manifest: %s" % ex)
        return 0

    names = {name for name, _ in entries}
    for missing in sorted(MIRROR_REQUIRED_MEMBERS - names):
        errors.append("mirror manifest omits %s" % missing)

    verified = 0
    root = evidence_dir.resolve()
    for name, expected in entries:
        candidate = root.joinpath(*PurePosixPath(name).parts)
        try:
            candidate.resolve().relative_to(root)
        except ValueError:
            errors.append("mirror member escapes evidence directory: %s" %
                          name)
            continue
        if not candidate.is_file():
            errors.append("mirror member missing: %s" % name)
            continue
        try:
            actual = _sha256(candidate)
        except OSError as ex:
            errors.append("cannot hash mirror member %s: %s" % (name, ex))
            continue
        if actual != expected:
            errors.append("mirror hash mismatch: %s" % name)
            continue
        verified += 1
    return verified


def _verify_summary_manifest_hash(summary, evidence_dir, key, filename,
                                  errors):
    expected = str(summary.get(key, "")).lower()
    path = evidence_dir / filename
    if not SHA256_RE.match(expected):
        errors.append("%s is not a SHA-256 digest" % key)
        return
    if not path.is_file():
        errors.append("missing %s" % filename)
        return
    try:
        actual = _sha256(path)
    except OSError as ex:
        errors.append("cannot hash %s: %s" % (filename, ex))
        return
    _same(key, actual, expected, errors)


def _read_gpu_cycles(path, errors):
    rows = []
    names = set()
    if not path.is_file():
        errors.append("missing gpu_cycles.tsv")
        return rows
    try:
        with open(str(path), "r", encoding="utf-8", errors="strict") as fh:
            for number, raw in enumerate(fh, 1):
                line = raw.rstrip("\r\n")
                parts = line.split("\t")
                if len(parts) != 2 or not parts[0]:
                    errors.append("gpu_cycles.tsv:%d is malformed" % number)
                    continue
                cycle = _integer(parts[1], "gpu_cycles.tsv:%d" % number,
                                 errors, minimum=1)
                if parts[0] in names:
                    errors.append("gpu_cycles.tsv duplicates %s" % parts[0])
                names.add(parts[0])
                if cycle is not None and cycle > 0:
                    rows.append((parts[0], cycle))
    except (OSError, UnicodeError) as ex:
        errors.append("cannot read gpu_cycles.tsv: %s" % ex)
    return rows


def _read_delay_info_active_cycles(path, errors):
    """Return the last PopNet packet-completion cycle in ``delayInfo.txt``.

    PopNet emits one whitespace-delimited record per transaction as

    ``srcTime src dst desc delayCount packetDelay...``.

    The last delay value is the receive-side completion delay for the final
    packet belonging to that transaction.  Consequently, the NoC active
    interval ends at ``max(srcTime + packetDelay[-1])``.  The producer casts
    both fields to integral cycles, so accepting anything else here would hide
    a malformed or incompatible evidence file.
    """
    if not path.is_file():
        errors.append("missing delayInfo.txt")
        return None

    active_cycles = None
    valid_rows = 0
    try:
        with open(str(path), "r", encoding="utf-8", errors="strict") as fh:
            for number, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) < 6:
                    errors.append(
                        "delayInfo.txt:%d has fewer than 6 fields" % number)
                    continue
                if any(not re.match(r"^-?[0-9]+$", value)
                       for value in fields):
                    errors.append(
                        "delayInfo.txt:%d contains a non-integer field" %
                        number)
                    continue
                values = [int(value) for value in fields]
                source_time = values[0]
                delay_count = values[4]
                if source_time < 0:
                    errors.append(
                        "delayInfo.txt:%d has negative srcTime" % number)
                    continue
                if delay_count <= 0:
                    errors.append(
                        "delayInfo.txt:%d has non-positive delayCount" %
                        number)
                    continue
                expected_fields = 5 + delay_count
                if len(values) != expected_fields:
                    errors.append(
                        "delayInfo.txt:%d has %d fields, expected %d from "
                        "delayCount" %
                        (number, len(values), expected_fields))
                    continue
                delays = values[5:]
                if any(delay < 0 for delay in delays):
                    errors.append(
                        "delayInfo.txt:%d has a negative packetDelay" %
                        number)
                    continue
                completion = source_time + delays[-1]
                if completion <= 0:
                    errors.append(
                        "delayInfo.txt:%d has non-positive completion" %
                        number)
                    continue
                active_cycles = (completion if active_cycles is None else
                                 max(active_cycles, completion))
                valid_rows += 1
    except (OSError, UnicodeError) as ex:
        errors.append("cannot read delayInfo.txt: %s" % ex)
        return None

    if valid_rows == 0:
        errors.append("delayInfo.txt has no valid records")
        return None
    return active_cycles


def _mesh_label(codegen_dir, errors):
    path = codegen_dir / "coords.json"
    if not path.is_file():
        errors.append("missing codegen coords.json")
        return ""
    try:
        coords = _load_json(path)
        mesh = coords.get("mesh") or {}
        rows = int(mesh["rows"])
        cols = int(mesh["cols"])
        if rows <= 0 or cols <= 0:
            raise ValueError("non-positive mesh")
        return "%dx%d" % (rows, cols)
    except (OSError, ValueError, KeyError, TypeError) as ex:
        errors.append("invalid codegen coords.json mesh: %s" % ex)
        return ""


def _params_experts(params_path, errors, label):
    if not params_path or not params_path.is_file():
        errors.append("missing %s params JSON: %s" % (label, params_path))
        return None
    try:
        params = _load_json(params_path)
        experts = params.get("meta", {}).get("experts")
        return _integer(experts, "%s meta.experts" % label, errors,
                        minimum=1)
    except (OSError, ValueError, TypeError) as ex:
        errors.append("cannot read %s params JSON: %s" % (label, ex))
        return None


def _audit_layout(results_path, manifest, manifest_experts, record,
                  expected_rounds):
    errors = []
    case_id = str(record.get("case_id", ""))
    method_raw = str(record.get("method", ""))
    method = normalize_method(method_raw)
    case_match = CASE_RE.search(case_id)
    case_experts = (int(case_match.group("experts"))
                    if case_match else None)
    experts = manifest_experts if manifest_experts is not None else case_experts
    if experts is None:
        errors.append("cannot determine expert count")
        experts = 0
    if case_experts is None:
        errors.append("case_id does not encode eNNN: %r" % case_id)
    elif case_experts != experts:
        errors.append("case_id expert count %d differs from params %d" %
                      (case_experts, experts))

    codegen = record.get("codegen") or {}
    fallback_codegen = (results_path.parent / "layouts" / case_id /
                        method_raw /
                        "codegen")
    codegen_dir = _resolve_record_path(
        codegen.get("codegen_dir"), results_path, fallback_codegen)
    if not codegen_dir.is_dir():
        errors.append("missing codegen directory: %s" % codegen_dir)
    mesh = _mesh_label(codegen_dir, errors) if codegen_dir.is_dir() else ""

    copied_experts = _params_experts(
        codegen_dir / "params.input.json", errors, "codegen")
    if copied_experts is not None:
        _same("codegen expert count", copied_experts, experts, errors)

    dispatch_meta = codegen.get("dispatch_meta") or {}
    if not dispatch_meta and (codegen_dir / "dispatch_meta.json").is_file():
        try:
            dispatch_meta = _load_json(codegen_dir / "dispatch_meta.json")
        except (OSError, ValueError) as ex:
            errors.append("cannot read dispatch_meta.json: %s" % ex)
    dispatch_ngpu = _integer(
        dispatch_meta.get("ngpu"), "dispatch_meta.ngpu", errors, minimum=1)
    if dispatch_ngpu is not None:
        _same("dispatch_meta.ngpu", dispatch_ngpu, experts, errors)
    _same("codegen.status", codegen.get("status"), "ok", errors)
    _same("codegen.dispatch_formal", codegen.get("dispatch_formal"), True,
          errors)

    metrics = record.get("solution_metrics") or {}
    metric_gpu_chiplets = _integer(
        metrics.get("gpu_workers"), "solution gpu_workers", errors, minimum=1)
    gpu_l = _integer(metrics.get("gpu_l_count"), "gpu_l_count", errors,
                     minimum=0)
    gpu_s = _integer(metrics.get("gpu_s_count"), "gpu_s_count", errors,
                     minimum=0)
    cpu_count = _integer(metrics.get("cpu_count"), "cpu_count", errors,
                         minimum=1)
    if cpu_count is not None:
        _same("CPU count", cpu_count, 1, errors)
    gpu_total = None
    auxiliary_gpu_count = None
    if gpu_l is not None and gpu_s is not None:
        gpu_total = gpu_l + gpu_s
        # q=1 requires at least one resident GPU per expert, but the joint MILP
        # may legally instantiate an additional GPU for a non-expert task such
        # as dispatch/attention.  Equality here would silently re-introduce the
        # fixed-GPU-count assumption that the experiment is designed to remove.
        if experts is not None and gpu_total < experts:
            errors.append(
                "GPU-L + GPU-S=%d is below the %d resident expert workers" %
                (gpu_total, experts))
        if experts is not None:
            auxiliary_gpu_count = gpu_total - experts
        if metric_gpu_chiplets is not None:
            _same("solution physical GPU chiplets", metric_gpu_chiplets,
                  gpu_total, errors)

    legosim = record.get("legosim") or {}
    summary_path = _resolve_record_path(
        legosim.get("summary_path"), results_path,
        results_path.parent / "layouts" / case_id / method_raw /
        "legosim_result" / "ilp_e2e_summary.txt")
    evidence_dir = summary_path.parent
    summary = {}
    if not summary_path.is_file():
        errors.append("missing ilp_e2e_summary.txt: %s" % summary_path)
    else:
        try:
            summary = _parse_key_values(summary_path)
        except (OSError, UnicodeError, ValueError) as ex:
            errors.append("invalid ilp_e2e_summary.txt: %s" % ex)

    _same("record.status", record.get("status"), "ok", errors)
    _same("legosim.status", legosim.get("status"), "ok", errors)
    _same("summary validation", summary.get("validation"), "PASS", errors)
    _same("JSON validation", legosim.get("validation"), "PASS", errors)
    _same("dispatch_mode", summary.get("dispatch_mode"),
          "real_dispatch_file", errors)
    _same("interchiplet_rc", summary.get("interchiplet_rc"), "0", errors)
    _same("simulator_binaries_immutable",
          summary.get("simulator_binaries_immutable"), "PASS", errors)

    bench = _integer(summary.get("makespan_cycles"), "bench cycles", errors,
                     minimum=1)
    sync = _integer(summary.get("synchronized_makespan_cycles"),
                    "synchronized cycles", errors, minimum=1)
    delta = _integer(summary.get("completion_delta_cycles"),
                     "completion delta", errors, minimum=0)
    if bench is not None and sync is not None:
        if sync < bench:
            errors.append("synchronized cycles %d < bench cycles %d" %
                          (sync, bench))
        if delta is not None:
            _same("completion delta", delta, sync - bench, errors)
    for key, json_value in (
            ("bench cycles", legosim.get("bench_makespan_cycles")),
            ("synchronized cycles",
             legosim.get("synchronized_makespan_cycles")),
            ("measured cycles", legosim.get("measured_makespan_cycles"))):
        parsed = _integer(json_value, "JSON %s" % key, errors, minimum=1)
        target = bench if key == "bench cycles" else sync
        if parsed is not None and target is not None:
            _same("JSON %s" % key, parsed, target, errors)

    rounds_requested = _integer(
        summary.get("rounds_requested"), "rounds_requested", errors,
        minimum=1)
    phase1 = _integer(summary.get("rounds_phase1"), "rounds_phase1", errors,
                      minimum=1)
    phase2 = _integer(summary.get("rounds_phase2"), "rounds_phase2", errors,
                      minimum=1)
    for label, value in (("rounds_requested", rounds_requested),
                         ("rounds_phase1", phase1),
                         ("rounds_phase2", phase2)):
        if value is not None:
            _same(label, value, expected_rounds, errors)

    workers = _integer(summary.get("n_gpu_workers"), "n_gpu_workers", errors,
                       minimum=1)
    if workers is not None:
        _same("n_gpu_workers", workers, experts, errors)
    _same("gpu_init_protocol", summary.get("gpu_init_protocol"), "row_col",
          errors)
    init_bytes = _integer(summary.get("gpu_init_bytes_per_worker"),
                          "gpu_init_bytes_per_worker", errors, minimum=1)
    expected_init_bytes = (4 + 2 * experts) * 4
    if init_bytes is not None:
        _same("gpu_init_bytes_per_worker", init_bytes, expected_init_bytes,
              errors)

    gpu_rows = _read_gpu_cycles(evidence_dir / "gpu_cycles.tsv", errors)
    _same("positive gpu_cycles.tsv rows", len(gpu_rows), experts, errors)
    gpu_max = max((value for _, value in gpu_rows), default=None)
    summary_gpu_max = _integer(summary.get("gpu_cycle_max"),
                               "summary gpu_cycle_max", errors, minimum=1)
    if gpu_max is not None and summary_gpu_max is not None:
        _same("gpu_cycle_max", summary_gpu_max, gpu_max, errors)

    diff_path = evidence_dir / "dispatch_matrix.diff"
    if not diff_path.is_file():
        errors.append("missing dispatch_matrix.diff")
    elif diff_path.stat().st_size != 0:
        errors.append("dispatch_matrix.diff is not empty")
    expected_dispatch = evidence_dir / "expected_dispatch_matrix.log"
    effective_dispatch = evidence_dir / "effective_dispatch_matrix.log"
    if not expected_dispatch.is_file() or not effective_dispatch.is_file():
        errors.append("missing expected/effective dispatch evidence")
    else:
        try:
            if _sha256(expected_dispatch) != _sha256(effective_dispatch):
                errors.append("expected/effective dispatch hashes differ")
        except OSError as ex:
            errors.append("cannot hash dispatch evidence: %s" % ex)

    popnet_total = _integer(summary.get("popnet_total_finished"),
                            "popnet_total_finished", errors, minimum=1)
    popnet_expected = _integer(summary.get("popnet_expected_finished"),
                               "popnet_expected_finished", errors, minimum=1)
    if popnet_total is not None and popnet_expected is not None:
        _same("PopNet total finished", popnet_total, popnet_expected, errors)
    noc_power = _number(summary.get("popnet_total_power"),
                        "PopNet/ORION NoC power", errors, positive=True)
    popnet_delay = _number(summary.get("popnet_average_delay"),
                           "PopNet average delay", errors, positive=True)
    noc_active_cycles = _read_delay_info_active_cycles(
        evidence_dir / "delayInfo.txt", errors)
    noc_energy_uJ = (noc_power * noc_active_cycles / 1000.0
                     if noc_power is not None and
                     noc_active_cycles is not None else None)

    mirror_verified = (_verify_mirror(evidence_dir, errors)
                       if evidence_dir.is_dir() else 0)
    if evidence_dir.is_dir():
        _verify_summary_manifest_hash(
            summary, evidence_dir, "source_input_manifest_sha256",
            "source_input_manifest.sha256", errors)
        _verify_summary_manifest_hash(
            summary, evidence_dir, "staged_input_manifest_sha256",
            "staged_input_manifest.sha256", errors)
        _verify_summary_manifest_hash(
            summary, evidence_dir,
            "simulator_binaries_start_manifest_sha256",
            "simulator_binaries.sha256", errors)
        _verify_summary_manifest_hash(
            summary, evidence_dir,
            "simulator_binaries_end_manifest_sha256",
            "simulator_binaries.end.sha256", errors)
        _verify_summary_manifest_hash(
            summary, evidence_dir, "simulator_binaries_diff_sha256",
            "simulator_binaries.diff", errors)
        binary_start = evidence_dir / "simulator_binaries.sha256"
        binary_end = evidence_dir / "simulator_binaries.end.sha256"
        binary_diff = evidence_dir / "simulator_binaries.diff"
        if binary_start.is_file() and binary_end.is_file():
            try:
                if _sha256(binary_start) != _sha256(binary_end):
                    errors.append("start/end simulator binary manifests differ")
            except OSError as ex:
                errors.append("cannot hash simulator binary manifests: %s" % ex)
        if binary_diff.is_file() and binary_diff.stat().st_size != 0:
            errors.append("simulator_binaries.diff is not empty")
        exit_path = evidence_dir / "runner_exit_code.txt"
        if not exit_path.is_file():
            errors.append("missing runner_exit_code.txt")
        else:
            try:
                exit_values = _parse_key_values(exit_path)
                _same("runner_exit_code", exit_values.get("runner_exit_code"),
                      "0", errors)
            except (OSError, UnicodeError, ValueError) as ex:
                errors.append("invalid runner_exit_code.txt: %s" % ex)

    return {
        "case_id": case_id,
        "method": method,
        "method_raw": method_raw,
        "experts": experts or None,
        "mesh": mesh,
        "cpu_count": cpu_count,
        "gpu_l_count": gpu_l,
        "gpu_s_count": gpu_s,
        "gpu_total_count": gpu_total,
        "auxiliary_gpu_count": auxiliary_gpu_count,
        "solution_gpu_chiplets": metric_gpu_chiplets,
        "audit_status": "PASS" if not errors else "FAIL",
        "legosim_validation": summary.get("validation"),
        "legosim_cycles": sync,
        "bench_cycles": bench,
        "completion_delta_cycles": delta,
        "gpu_cycle_max": gpu_max,
        "gpu_cycle_rows": len(gpu_rows),
        "orion_noc_power_W": noc_power,
        "noc_active_cycles": noc_active_cycles,
        "nominal_orion_dynamic_noc_energy_uJ": noc_energy_uJ,
        "popnet_average_delay": popnet_delay,
        "popnet_total_finished": popnet_total,
        "popnet_expected_finished": popnet_expected,
        "rounds_requested": rounds_requested,
        "n_gpu_workers": workers,
        "gpu_init_bytes_per_worker": init_bytes,
        "simulator_binaries_immutable": summary.get(
            "simulator_binaries_immutable"),
        "mirror_hashes_verified": mirror_verified,
        "evidence_dir": str(evidence_dir),
        "results_json": str(results_path),
        "errors": errors,
    }


def _discover_results(input_path):
    if input_path.is_file():
        return [input_path.resolve()] if input_path.name == "results.json" else []
    return sorted(path.resolve() for path in input_path.rglob("results.json"))


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with open(str(temporary), "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with open(str(temporary), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["errors"] = " | ".join(row.get("errors") or [])
            writer.writerow(flat)
    os.replace(str(temporary), str(path))


def main(argv=None):
    args = _parse_args(argv)
    if args.expected_rounds <= 0:
        raise ValueError("--expected-rounds must be positive")
    if not args.expected_experts or any(value <= 0 for value in
                                        args.expected_experts):
        raise ValueError("--expected-experts must contain positive values")
    if len(set(args.expected_experts)) != len(args.expected_experts):
        raise ValueError("--expected-experts contains duplicates")
    if not args.expected_methods or any(not value for value in
                                        args.expected_methods):
        raise ValueError("--expected-methods must not be empty")
    expected_methods = [normalize_method(value)
                        for value in args.expected_methods]
    if len(set(expected_methods)) != len(expected_methods):
        raise ValueError("--expected-methods contains duplicates")
    unsupported = sorted(set(expected_methods) - set(SUPPORTED_METHODS))
    if unsupported:
        raise ValueError("--expected-methods contains unsupported names: %s" %
                         ", ".join(unsupported))

    input_path = Path(args.input_root).resolve()
    if not input_path.exists():
        raise ValueError("input root does not exist: %s" % input_path)
    results_paths = _discover_results(input_path)
    if not results_paths:
        raise ValueError("no results.json found below %s" % input_path)

    rows = []
    global_errors = []
    seen_pairs = set()
    for results_path in results_paths:
        try:
            manifest = _load_json(results_path)
        except (OSError, ValueError) as ex:
            global_errors.append("cannot read %s: %s" % (results_path, ex))
            continue
        if manifest.get("status") != "complete":
            global_errors.append("%s status=%r, expected 'complete'" %
                                 (results_path, manifest.get("status")))
        if manifest.get("finished_utc") in (None, ""):
            global_errors.append("%s has no finished_utc" % results_path)
        if manifest.get("rounds") != args.expected_rounds:
            global_errors.append("%s rounds=%r, expected %d" %
                                 (results_path, manifest.get("rounds"),
                                  args.expected_rounds))
        if manifest.get("codegen_only") is not False:
            global_errors.append("%s is a codegen-only manifest" %
                                 results_path)

        params_path = _resolve_record_path(
            manifest.get("params_path"), results_path)
        params_errors = []
        manifest_experts = _params_experts(
            params_path, params_errors, "manifest")
        global_errors.extend("%s: %s" % (results_path, error)
                             for error in params_errors)
        if params_path and params_path.is_file():
            expected_hash = str(manifest.get("params_sha256", "")).lower()
            if not SHA256_RE.match(expected_hash):
                global_errors.append("%s has invalid params_sha256" %
                                     results_path)
            elif _sha256(params_path) != expected_hash:
                global_errors.append("%s params SHA-256 mismatch" %
                                     results_path)

        layouts = manifest.get("layouts")
        if not isinstance(layouts, list):
            global_errors.append("%s layouts is not a list" % results_path)
            continue
        for record in layouts:
            if not isinstance(record, dict):
                global_errors.append("%s contains a non-object layout" %
                                     results_path)
                continue
            row = _audit_layout(results_path, manifest, manifest_experts,
                                record, args.expected_rounds)
            pair = (row.get("experts"), row.get("method"))
            if pair in seen_pairs:
                global_errors.append("duplicate expert/method result: %r" %
                                     (pair,))
                row["errors"].append("duplicate expert/method result")
                row["audit_status"] = "FAIL"
            seen_pairs.add(pair)
            rows.append(row)

    expected_pairs = {(experts, method)
                      for experts in args.expected_experts
                      for method in expected_methods}
    missing_pairs = sorted(expected_pairs - seen_pairs)
    unexpected_pairs = sorted(seen_pairs - expected_pairs,
                              key=lambda item: (item[0] or -1, item[1]))
    if missing_pairs:
        global_errors.append("missing expert/method results: %s" %
                             ", ".join("E%d/%s" % pair
                                       for pair in missing_pairs))
    if unexpected_pairs:
        global_errors.append("unexpected expert/method results: %s" %
                             ", ".join("%s/%s" % pair
                                       for pair in unexpected_pairs))

    rows.sort(key=lambda row: (row.get("experts") or -1,
                               row.get("method") or ""))
    failed_rows = [row for row in rows if row["audit_status"] != "PASS"]
    overall = "PASS" if not failed_rows and not global_errors else "FAIL"
    report = {
        "schema_version": 1,
        "audit_status": overall,
        "input_root": str(input_path),
        "protocol": {
            "expected_experts": args.expected_experts,
            "expected_methods": expected_methods,
            "supported_methods": list(SUPPORTED_METHODS),
            "method_aliases": dict(sorted(METHOD_ALIASES.items())),
            "expected_rounds": args.expected_rounds,
            "cycle_metric": "synchronized_makespan_cycles",
            "power_metric": "PopNet/ORION NoC-only total power (W)",
            "noc_active_cycle_metric": (
                "max(srcTime + final packetDelay) from delayInfo.txt"),
            "noc_dynamic_energy_metric": (
                "orion_noc_power_W * noc_active_cycles / 1000 (uJ, "
                "nominal 1-GHz clock)"),
            "gpu_init_bytes_formula": "(4 + 2*experts) * 4",
        },
        "results_manifests": [str(path) for path in results_paths],
        "row_count": len(rows),
        "pass_count": len(rows) - len(failed_rows),
        "fail_count": len(failed_rows),
        "global_errors": global_errors,
        "rows": rows,
    }

    if args.out_prefix:
        prefix = Path(args.out_prefix).resolve()
    else:
        base = input_path if input_path.is_dir() else input_path.parent
        prefix = base / "scaling_legosim_audit"
    json_path = Path(str(prefix) + ".json")
    csv_path = Path(str(prefix) + ".csv")
    _atomic_json(json_path, report)
    _atomic_csv(csv_path, rows)
    print(json.dumps({
        "audit_status": overall,
        "rows": len(rows),
        "passed": len(rows) - len(failed_rows),
        "failed": len(failed_rows),
        "global_errors": len(global_errors),
        "json": str(json_path),
        "csv": str(csv_path),
    }, indent=2, sort_keys=True))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as ex:
        print("error: %s" % ex, file=sys.stderr)
        sys.exit(2)
