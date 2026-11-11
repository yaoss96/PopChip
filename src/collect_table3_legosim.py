#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Table III only from fully audited large-scale LEGOSim replays.

The collector is intentionally fail closed.  It accepts exactly one replay for
each member of the fixed 5-scale x 3-method experiment, delegates the raw
LEGOSim/evidence checks to :mod:`audit_scaling_legosim_results`, and adds
solution/parameter/grid/BOM provenance checks.  Publication artifacts are
written only when all fifteen records pass.

No analytical latency, analytical power, or solution surrogate value is read
into the table.  ``cycles`` is LEGOSim's synchronized end-to-end makespan and
``noc_power_W`` is PopNet/ORION NoC-only total power (not CPU, GPU, package, or
wall-plug power).
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

import audit_scaling_legosim_results as audit


EXPECTED_SCALES = {
    8: {"mesh_side": 6, "gpu_l": 2, "gpu_s": 6},
    16: {"mesh_side": 8, "gpu_l": 4, "gpu_s": 12},
    32: {"mesh_side": 12, "gpu_l": 8, "gpu_s": 24},
    64: {"mesh_side": 12, "gpu_l": 16, "gpu_s": 48},
    128: {"mesh_side": 16, "gpu_l": 32, "gpu_s": 96},
}
# Stable reporting names.  The immutable 2026-08-01 input catalog contains
# only the three REQUIRED_METHODS; ``ilp_power`` is reserved in the output
# schema and becomes populated after a separately audited replay is supplied.
SUPPORTED_METHODS = ("ilp_perf", "ilp_power", "floorplet", "tap")
METHODS = ("ilp_perf", "floorplet", "tap")
MANIFEST_METHOD = {
    "ilp_perf": "ilp",
    "floorplet": "floorplet",
    "tap": "tap",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_INPUT_MANIFEST = "final_input_manifest.json"
CANONICAL_ILP_ROOT = "out/scaling_ilp_secondary_20260801"
CANONICAL_PROXY_ROOT = "out/scaling_secondary_proxy_experiment_20260801"
CANONICAL_PARAMS_ROOT = "out/scaling_qap_experiment/params"
FORBIDDEN_CANONICAL_MARKERS = (
    "secondary300",
    "scaling_prior_proxy_experiment_20260731",
    "scaling_prior_proxy_experiment",
)
TABLE_CSV_FIELDS = [
    "experts",
    "mesh",
    "gpu_l_s",
    "ilp_perf_synchronized_cycles",
    "ilp_perf_orion_noc_power_W",
    "ilp_power_synchronized_cycles",
    "ilp_power_orion_noc_power_W",
    "floorplet_synchronized_cycles",
    "floorplet_orion_noc_power_W",
    "tap_synchronized_cycles",
    "tap_orion_noc_power_W",
]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Strictly audit the fixed E8--E128 LEGOSim scaling suite and "
            "emit Table III JSON, CSV, and LaTeX rows only on a full PASS."))
    parser.add_argument(
        "input_root", nargs="?",
        default="out/scaling_legosim_final_twostage_20260801",
        help=(
            "root recursively containing results.json manifests "
            "(default: out/scaling_legosim_final_twostage_20260801)"))
    parser.add_argument(
        "--out-prefix", default=None,
        help=(
            "publication output prefix; .json, .csv, and .tex are appended "
            "(default: <input_root>/table3_legosim_real)"))
    return parser.parse_args(argv)


def _add_error(errors, message):
    if message not in errors:
        errors.append(message)


def _expect_equal(errors, label, actual, expected):
    if actual != expected:
        _add_error(errors, "%s=%r, expected %r" %
                   (label, actual, expected))


def _load_object(path, label, errors):
    try:
        return audit._load_json(path)
    except (OSError, ValueError) as exc:
        _add_error(errors, "cannot read %s %s: %s" % (label, path, exc))
        return None


def _require_within(path, parent, label, errors):
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        _add_error(errors, "%s is outside its results directory: %s" %
                   (label, path))


def _exact_bom(experts):
    scale = EXPECTED_SCALES[experts]
    return {"CPU": 1, "GPU-L": scale["gpu_l"], "GPU-S": scale["gpu_s"]}


def _pair_name(experts):
    return "e%03d_m%02d" % (experts, EXPECTED_SCALES[experts]["mesh_side"])


def _expected_solution_relative(experts, method):
    method = audit.normalize_method(method)
    pair = _pair_name(experts)
    if method == "ilp_perf":
        return "%s/solutions/%s_ilp_secondary.json" % (
            CANONICAL_ILP_ROOT, pair)
    return "%s/solutions/%s_%s.json" % (
        CANONICAL_PROXY_ROOT, pair, method)


def _expected_params_relative(experts):
    return "%s/experts_%03d.json" % (CANONICAL_PARAMS_ROOT, experts)


def _canonical_path(raw_value, label, errors):
    """Resolve an absolute or canonical project-relative manifest path."""
    if not isinstance(raw_value, str) or not raw_value:
        _add_error(errors, "%s is not a non-empty path" % label)
        return None
    lowered = raw_value.replace("\\", "/").lower()
    for marker in FORBIDDEN_CANONICAL_MARKERS:
        if marker in lowered:
            _add_error(errors, "%s uses forbidden legacy path %r" %
                       (label, raw_value))
    raw_path = Path(raw_value)
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        posix = PurePosixPath(raw_value)
        normalized = posix.as_posix()
        if (raw_value != normalized or
                any(part in ("", ".", "..") for part in posix.parts)):
            _add_error(errors, "%s is not a canonical relative path: %r" %
                       (label, raw_value))
        resolved = PROJECT_ROOT.joinpath(*posix.parts).resolve()
    _require_within(resolved, PROJECT_ROOT, label, errors)
    return resolved


def _reject_legacy_manifest_strings(value, label, errors):
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_legacy_manifest_strings(
                child, "%s.%s" % (label, key), errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_legacy_manifest_strings(
                child, "%s[%d]" % (label, index), errors)
    elif isinstance(value, str):
        lowered = value.replace("\\", "/").lower()
        for marker in FORBIDDEN_CANONICAL_MARKERS:
            if marker in lowered:
                _add_error(errors,
                           "%s contains forbidden legacy input marker %r" %
                           (label, marker))


def _canonical_value_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _solution_contract(solution, experts, label, errors):
    """Return the independently derived BOM and immutable bundle hash."""
    if not isinstance(solution, dict):
        _add_error(errors, "%s is not an object" % label)
        return None, None
    _validate_grid(solution.get("grid"), experts, "%s.grid" % label, errors)
    sites = solution.get("sites")
    task_site = solution.get("task_site")
    if not isinstance(sites, list) or not isinstance(task_site, dict):
        _add_error(errors, "%s lacks sites/task_site objects" % label)
        return None, None

    site_types = {}
    declared = []
    for index, site in enumerate(sites):
        if not isinstance(site, dict):
            _add_error(errors, "%s.sites[%d] is not an object" %
                       (label, index))
            continue
        try:
            anchor = tuple(int(site[key]) for key in ("l", "i", "j"))
        except (KeyError, TypeError, ValueError):
            _add_error(errors, "%s.sites[%d] has invalid anchor" %
                       (label, index))
            continue
        if anchor in site_types:
            _add_error(errors, "%s has duplicate site anchor %r" %
                       (label, anchor))
        kind = str(site.get("type"))
        site_types[anchor] = kind
        tasks = site.get("tasks")
        if not isinstance(tasks, list):
            _add_error(errors, "%s.sites[%d].tasks is not a list" %
                       (label, index))
            tasks = []
        declared.append([kind, sorted(str(task) for task in tasks)])

    mapped_at = {anchor: [] for anchor in site_types}
    for task, raw_anchor in task_site.items():
        try:
            anchor = tuple(int(value) for value in raw_anchor)
        except (TypeError, ValueError):
            _add_error(errors, "%s.task_site[%s] has invalid anchor" %
                       (label, task))
            continue
        if anchor not in mapped_at:
            _add_error(errors, "%s maps %s to undeclared site %r" %
                       (label, task, anchor))
            continue
        mapped_at[anchor].append(str(task))
    mapped = [[site_types[anchor], sorted(tasks)]
              for anchor, tasks in mapped_at.items()]
    declared.sort(key=lambda item: (item[0], item[1]))
    mapped.sort(key=lambda item: (item[0], item[1]))
    _expect_equal(errors, "%s declared/mapped bundle signature" % label,
                  declared, mapped)

    counts = Counter(site_types.values())
    bom = {kind: int(counts.get(kind, 0))
           for kind in ("CPU", "GPU-L", "GPU-S")}
    _expect_equal(errors, "%s BOM" % label, bom, _exact_bom(experts))
    unexpected = sorted(kind for kind in counts if kind not in bom)
    if unexpected:
        _add_error(errors, "%s has unexpected site types: %s" %
                   (label, ", ".join(unexpected)))
    return bom, _canonical_value_hash(mapped)


def _verify_manifest_file(path_value, digest_value, expected_relative,
                          label, errors):
    path = _canonical_path(path_value, "%s path" % label, errors)
    expected_path = (PROJECT_ROOT.joinpath(
        *PurePosixPath(expected_relative).parts).resolve())
    if path is not None:
        _expect_equal(errors, "%s canonical path" % label,
                      path, expected_path)
    digest = str(digest_value or "").lower()
    if not audit.SHA256_RE.match(digest):
        _add_error(errors, "%s SHA-256 is invalid" % label)
    if path is None or not path.is_file():
        _add_error(errors, "%s file is missing: %s" % (label, path))
        return None, path, digest
    try:
        actual = audit._sha256(path)
        _expect_equal(errors, "%s SHA-256" % label, digest, actual)
    except OSError as exc:
        _add_error(errors, "cannot hash %s: %s" % (label, exc))
    value = _load_object(path, label, errors)
    return value, path, digest


def _validate_record_canonical_identity(solution_path, solution_digest,
                                        canonical, errors):
    """Bind a self-labelled result row to its immutable catalog identity."""
    canonical = canonical if isinstance(canonical, dict) else {}
    _expect_equal(errors, "record/canonical solution path", solution_path,
                  canonical.get("solution_path"))
    _expect_equal(errors, "record/canonical solution SHA-256",
                  str(solution_digest or "").lower(),
                  canonical.get("solution_sha256"))


def _validate_grid(value, experts, label, errors):
    side = EXPECTED_SCALES[experts]["mesh_side"]
    if not isinstance(value, dict):
        _add_error(errors, "%s is not an object" % label)
        return
    _expect_equal(errors, "%s.L" % label, value.get("L"), 1)
    _expect_equal(errors, "%s.X" % label, value.get("X"), side)
    _expect_equal(errors, "%s.Y" % label, value.get("Y"), side)


def _validate_bom_mapping(value, experts, label, errors):
    expected = _exact_bom(experts)
    if not isinstance(value, dict):
        _add_error(errors, "%s is not an object" % label)
        return
    normalized = {key: value.get(key) for key in expected}
    _expect_equal(errors, label, normalized, expected)
    unexpected = sorted(set(value) - set(expected))
    if unexpected:
        _add_error(errors, "%s contains unexpected types: %s" %
                   (label, ", ".join(unexpected)))


def _site_bom(sites, experts, label, errors):
    if not isinstance(sites, list):
        _add_error(errors, "%s is not a list" % label)
        return
    counts = Counter()
    occupied = set()
    expert_tasks = []
    for index, site in enumerate(sites):
        if not isinstance(site, dict):
            _add_error(errors, "%s[%d] is not an object" % (label, index))
            continue
        kind = site.get("type")
        counts[kind] += 1
        coordinate = tuple(site.get(key) for key in ("l", "i", "j"))
        if coordinate == (None, None, None):
            coordinate = tuple(site.get(key) for key in
                               ("layer", "row", "col"))
        if coordinate in occupied:
            _add_error(errors, "%s has duplicate occupied coordinate %r" %
                       (label, coordinate))
        occupied.add(coordinate)
        tasks = site.get("tasks")
        if not isinstance(tasks, list):
            _add_error(errors, "%s[%d].tasks is not a list" %
                       (label, index))
            continue
        expert_tasks.extend(task for task in tasks
                            if isinstance(task, str) and
                            task.startswith("e(") and task.endswith(")"))
    expected = _exact_bom(experts)
    actual = {key: counts.get(key, 0) for key in expected}
    _expect_equal(errors, "%s type counts" % label, actual, expected)
    unexpected_types = sorted(str(key) for key in counts if key not in expected)
    if unexpected_types:
        _add_error(errors, "%s has unexpected types: %s" %
                   (label, ", ".join(unexpected_types)))
    expected_tasks = ["e(%d)" % index for index in range(experts)]
    if sorted(expert_tasks) != sorted(expected_tasks):
        _add_error(errors, "%s does not contain each expert task exactly once" %
                   label)


def _validation_flags(value, label, errors):
    if not isinstance(value, dict):
        _add_error(errors, "%s is not an object" % label)
        return
    for key in ("pass", "exact_BOM_equal",
                "immutable_bundle_multiset_equal",
                "candidate_serialization_consistent"):
        _expect_equal(errors, "%s.%s" % (label, key), value.get(key), True)


def _scale_audit_bindings(scale_entry, experts, errors):
    """Verify the immutable input audits and extract their path/hash bindings."""
    pair = _pair_name(experts)
    side = EXPECTED_SCALES[experts]["mesh_side"]
    if not isinstance(scale_entry, dict):
        _add_error(errors, "missing scale audit for %s" % pair)
        return {}
    _expect_equal(errors, "%s scale pair" % pair,
                  scale_entry.get("pair"), pair)
    _expect_equal(errors, "%s scale experts" % pair,
                  scale_entry.get("experts"), experts)
    _expect_equal(errors, "%s scale mesh" % pair,
                  scale_entry.get("mesh"), "%dx%d" % (side, side))
    for key in ("ilp_audit_pass", "proxy_audit_pass",
                "exact_bom_match_across_methods",
                "exact_bundle_match_across_methods"):
        _expect_equal(errors, "%s scale %s" % (pair, key),
                      scale_entry.get(key), True)
    _expect_equal(errors, "%s primary time" % pair,
                  scale_entry.get("primary_time_limit_s"), 30.0)
    _expect_equal(errors, "%s secondary time" % pair,
                  scale_entry.get("secondary_time_limit_s"), 30.0)
    _expect_equal(errors, "%s scale BOM" % pair,
                  scale_entry.get("bom"), _exact_bom(experts))
    scale_bundle = str(scale_entry.get("bundle_signature_sha256", "")).lower()
    if not audit.SHA256_RE.match(scale_bundle):
        _add_error(errors, "%s scale bundle signature is invalid" % pair)

    ilp_rel = "%s/audits/%s_audit.json" % (CANONICAL_ILP_ROOT, pair)
    proxy_rel = "%s/audits/%s_proxy_audit.json" % (
        CANONICAL_PROXY_ROOT, pair)
    ilp_audit, _, _ = _verify_manifest_file(
        scale_entry.get("ilp_audit_path"),
        scale_entry.get("ilp_audit_sha256"), ilp_rel,
        "%s ILP input audit" % pair, errors)
    proxy_audit, _, _ = _verify_manifest_file(
        scale_entry.get("proxy_audit_path"),
        scale_entry.get("proxy_audit_sha256"), proxy_rel,
        "%s proxy input audit" % pair, errors)
    if ilp_audit is None or proxy_audit is None:
        return {}

    for label, value in (("ILP", ilp_audit), ("proxy", proxy_audit)):
        _expect_equal(errors, "%s %s audit pass" % (pair, label),
                      value.get("pass"), True)
        _expect_equal(errors, "%s %s audit pair" % (pair, label),
                      value.get("pair"), pair)
        _expect_equal(errors, "%s %s audit experts" % (pair, label),
                      value.get("experts"), experts)
        _expect_equal(errors, "%s %s audit mesh" % (pair, label),
                      value.get("mesh"), side)

    ilp_validation = ilp_audit.get("validation")
    if not isinstance(ilp_validation, dict):
        _add_error(errors, "%s ILP validation is missing" % pair)
    else:
        _expect_equal(errors, "%s ILP validation.pass" % pair,
                      ilp_validation.get("pass"), True)
        _expect_equal(errors, "%s ILP validation BOM" % pair,
                      ilp_validation.get("exact_BOM_counts"),
                      _exact_bom(experts))

    artifacts = ilp_audit.get("artifacts") or {}
    ilp_solution_path = _canonical_path(
        artifacts.get("solution"), "%s ILP-audit solution" % pair, errors)
    ilp_params_path = _canonical_path(
        artifacts.get("params"), "%s ILP-audit params" % pair, errors)
    expected_ilp_path = _expected_solution_relative(experts, "ilp")
    expected_params_path = _expected_params_relative(experts)
    _expect_equal(errors, "%s ILP-audit solution path" % pair,
                  ilp_solution_path,
                  (PROJECT_ROOT / expected_ilp_path).resolve())
    _expect_equal(errors, "%s ILP-audit params path" % pair,
                  ilp_params_path,
                  (PROJECT_ROOT / expected_params_path).resolve())
    ilp_solution_sha = str(artifacts.get("solution_sha256", "")).lower()
    params_sha = str(artifacts.get("params_sha256", "")).lower()
    if not audit.SHA256_RE.match(ilp_solution_sha):
        _add_error(errors, "%s ILP-audit solution SHA is invalid" % pair)
    if not audit.SHA256_RE.match(params_sha):
        _add_error(errors, "%s ILP-audit params SHA is invalid" % pair)

    command_rel = "%s/commands/%s_command.json" % (CANONICAL_ILP_ROOT, pair)
    command_path = _canonical_path(
        artifacts.get("command"), "%s ILP command" % pair, errors)
    _expect_equal(errors, "%s ILP command path" % pair, command_path,
                  (PROJECT_ROOT / command_rel).resolve())
    command = (_load_object(command_path, "%s ILP command" % pair, errors)
               if command_path and command_path.is_file() else None)
    if command is None:
        _add_error(errors, "%s ILP command is missing" % pair)
    else:
        options = command.get("solver_options") or {}
        _expect_equal(errors, "%s command primary time" % pair,
                      options.get("time_limit"), 30.0)
        _expect_equal(errors, "%s command secondary time" % pair,
                      options.get("secondary_time"), 30.0)
        _expect_equal(errors, "%s command mode" % pair,
                      options.get("mode"), "certified-latency")
        _expect_equal(errors, "%s command solution path" % pair,
                      _canonical_path(command.get("solution"),
                                      "%s command solution" % pair, errors),
                      ilp_solution_path)
        _expect_equal(errors, "%s command solution SHA" % pair,
                      str(command.get("solution_sha256", "")).lower(),
                      ilp_solution_sha)

    provenance = proxy_audit.get("provenance") or {}
    _expect_equal(errors, "%s proxy params path" % pair,
                  _canonical_path(provenance.get("params_path"),
                                  "%s proxy params" % pair, errors),
                  ilp_params_path)
    _expect_equal(errors, "%s proxy params SHA" % pair,
                  str(provenance.get("params_sha256", "")).lower(),
                  params_sha)
    _expect_equal(errors, "%s proxy source ILP path" % pair,
                  _canonical_path(provenance.get("source_ilp_path"),
                                  "%s proxy source ILP" % pair, errors),
                  ilp_solution_path)
    _expect_equal(errors, "%s proxy source ILP SHA" % pair,
                  str(provenance.get("source_ilp_sha256", "")).lower(),
                  ilp_solution_sha)

    source_ilp = proxy_audit.get("source_ilp") or {}
    _expect_equal(errors, "%s source-ILP path" % pair,
                  _canonical_path(source_ilp.get("path"),
                                  "%s source-ILP" % pair, errors),
                  ilp_solution_path)
    _expect_equal(errors, "%s source-ILP SHA" % pair,
                  str(source_ilp.get("sha256", "")).lower(),
                  ilp_solution_sha)
    _validation_flags(source_ilp.get("validation"),
                      "%s source-ILP validation" % pair, errors)
    source_bundle = str(source_ilp.get("bundle_multiset_sha256", "")).lower()
    _expect_equal(errors, "%s source-ILP bundle" % pair,
                  source_bundle, scale_bundle)
    _expect_equal(errors, "%s source-ILP BOM" % pair,
                  source_ilp.get("BOM"), _exact_bom(experts))

    bindings = {
        "ilp": {
            "solution_path": ilp_solution_path,
            "solution_sha256": ilp_solution_sha,
            "params_path": ilp_params_path,
            "params_sha256": params_sha,
            "bom": _exact_bom(experts),
            "bundle_signature_sha256": scale_bundle,
        }
    }
    proxy_rows = proxy_audit.get("proxies")
    if not isinstance(proxy_rows, list):
        _add_error(errors, "%s proxy audit rows are missing" % pair)
        proxy_rows = []
    proxy_by_method = {}
    for proxy in proxy_rows:
        if not isinstance(proxy, dict):
            _add_error(errors, "%s proxy audit has non-object row" % pair)
            continue
        method = proxy.get("method")
        if method in proxy_by_method:
            _add_error(errors, "%s proxy audit duplicates %s" % (pair, method))
        proxy_by_method[method] = proxy
    _expect_equal(errors, "%s proxy audit methods" % pair,
                  set(proxy_by_method), {"floorplet", "tap"})
    for method in ("floorplet", "tap"):
        proxy = proxy_by_method.get(method) or {}
        expected_rel = _expected_solution_relative(experts, method)
        solution_path = _canonical_path(
            proxy.get("solution_path"),
            "%s %s audit solution" % (pair, method), errors)
        _expect_equal(errors, "%s %s audit solution path" % (pair, method),
                      solution_path, (PROJECT_ROOT / expected_rel).resolve())
        solution_sha = str(proxy.get("solution_sha256", "")).lower()
        if not audit.SHA256_RE.match(solution_sha):
            _add_error(errors, "%s %s audit solution SHA is invalid" %
                       (pair, method))
        validation = proxy.get("validation")
        _validation_flags(validation,
                          "%s %s validation" % (pair, method), errors)
        validation = validation if isinstance(validation, dict) else {}
        proxy_bundle = str(validation.get("bundle_multiset_sha256", "")).lower()
        _expect_equal(errors, "%s %s bundle" % (pair, method),
                      proxy_bundle, scale_bundle)
        _expect_equal(errors, "%s %s BOM" % (pair, method),
                      validation.get("candidate_BOM"), _exact_bom(experts))
        bindings[method] = {
            "solution_path": solution_path,
            "solution_sha256": solution_sha,
            "params_path": ilp_params_path,
            "params_sha256": params_sha,
            "bom": _exact_bom(experts),
            "bundle_signature_sha256": scale_bundle,
        }
    return bindings


def _validate_final_input_manifest(root, errors):
    """Load the canonical input catalog and return 15 immutable bindings."""
    path = root / FINAL_INPUT_MANIFEST
    if not path.is_file():
        _add_error(errors, "missing canonical final_input_manifest.json: %s" %
                   path)
        return {}, None
    manifest = _load_object(path, "canonical final input manifest", errors)
    if manifest is None:
        return {}, path
    _reject_legacy_manifest_strings(manifest, "input manifest", errors)

    expected_result_root = (PROJECT_ROOT /
                            "out/scaling_legosim_final_twostage_20260801").resolve()
    _expect_equal(errors, "collector input root", root.resolve(),
                  expected_result_root)
    _expect_equal(errors, "input manifest schema_version",
                  manifest.get("schema_version"), 1)
    _expect_equal(errors, "input manifest expected_input_count",
                  manifest.get("expected_input_count"), 15)
    _expect_equal(errors, "input manifest actual_input_count",
                  manifest.get("actual_input_count"), 15)
    for key in ("all_input_audits_pass", "all_scales_exact_bom_match",
                "all_scales_exact_bundle_match"):
        _expect_equal(errors, "input manifest %s" % key,
                      manifest.get(key), True)

    policy = manifest.get("input_policy") or {}
    for key, expected_rel in (
            ("canonical_ilp_root", CANONICAL_ILP_ROOT),
            ("canonical_proxy_root", CANONICAL_PROXY_ROOT),
            ("result_root", "out/scaling_legosim_final_twostage_20260801")):
        actual_path = _canonical_path(
            policy.get(key), "input_policy.%s" % key, errors)
        _expect_equal(errors, "input_policy.%s" % key, actual_path,
                      (PROJECT_ROOT / expected_rel).resolve())
    _expect_equal(
        errors, "input_policy.ilp_source", policy.get("ilp_source"),
        "30 s primary plus 30 s communication-energy secondary stage")
    _expect_equal(
        errors, "input_policy.proxy_source", policy.get("proxy_source"),
        "proxies regenerated from the canonical two-stage ILP bundles")

    global_audits = manifest.get("global_audits") or {}
    _expect_equal(errors, "global two-stage audit pass",
                  global_audits.get("two_stage_ilp_audit_pass"), True)
    _expect_equal(errors, "global proxy audits pass",
                  global_audits.get("all_proxy_audits_pass"), True)
    secondary, _, _ = _verify_manifest_file(
        global_audits.get("two_stage_ilp_audit_path"),
        global_audits.get("two_stage_ilp_audit_sha256"),
        "%s/secondary_audit.json" % CANONICAL_ILP_ROOT,
        "global two-stage ILP audit", errors)
    proxy_metadata, _, _ = _verify_manifest_file(
        global_audits.get("proxy_metadata_path"),
        global_audits.get("proxy_metadata_sha256"),
        "%s/metadata.json" % CANONICAL_PROXY_ROOT,
        "global proxy metadata", errors)
    if secondary is not None:
        _expect_equal(errors, "global ILP audit pass", secondary.get("pass"), True)
        _expect_equal(errors, "global ILP completed pairs",
                      secondary.get("completed_pairs"), 5)
    if proxy_metadata is not None:
        _expect_equal(errors, "global proxy metadata pass",
                      proxy_metadata.get("pass"), True)
        _expect_equal(errors, "global proxy metadata status",
                      proxy_metadata.get("status"), "complete")
        _expect_equal(errors, "global proxy completed count",
                      proxy_metadata.get("completed_count"), 5)

    scale_rows = manifest.get("scale_audits")
    if not isinstance(scale_rows, list):
        _add_error(errors, "input manifest scale_audits is not a list")
        scale_rows = []
    scale_by_experts = {}
    for row in scale_rows:
        if not isinstance(row, dict):
            _add_error(errors, "input manifest has non-object scale audit")
            continue
        experts = row.get("experts")
        if experts in scale_by_experts:
            _add_error(errors, "input manifest duplicates E%s scale audit" %
                       experts)
        scale_by_experts[experts] = row
    _expect_equal(errors, "input manifest scale audit experts",
                  set(scale_by_experts), set(EXPECTED_SCALES))

    audit_bindings = {}
    for experts in sorted(EXPECTED_SCALES):
        for method, binding in _scale_audit_bindings(
                scale_by_experts.get(experts), experts, errors).items():
            audit_bindings[(experts, audit.normalize_method(method))] = binding

    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        _add_error(errors, "input manifest inputs is not a list")
        inputs = []
    _expect_equal(errors, "input manifest inputs length", len(inputs), 15)
    bindings = {}
    bundle_by_scale = {}
    for index, entry in enumerate(inputs):
        if not isinstance(entry, dict):
            _add_error(errors, "input manifest inputs[%d] is not an object" %
                       index)
            continue
        experts = entry.get("experts")
        raw_method = entry.get("method")
        method = audit.normalize_method(raw_method)
        pair_key = (experts, method)
        if pair_key in bindings:
            _add_error(errors, "input manifest duplicates E%s/%s" % pair_key)
            continue
        if experts not in EXPECTED_SCALES or method not in METHODS:
            _add_error(errors, "input manifest has unexpected pair E%s/%s" %
                       pair_key)
            continue
        pair = _pair_name(experts)
        side = EXPECTED_SCALES[experts]["mesh_side"]
        _expect_equal(errors, "%s/%s pair" % (pair, method),
                      entry.get("pair"), pair)
        _expect_equal(errors, "%s/%s mesh" % (pair, method),
                      entry.get("mesh"), "%dx%d" % (side, side))
        _expect_equal(errors, "%s/%s mesh_rows" % (pair, method),
                      entry.get("mesh_rows"), side)
        _expect_equal(errors, "%s/%s mesh_columns" % (pair, method),
                      entry.get("mesh_columns"), side)
        expected_kind = ("two_stage_ilp_30s_primary_30s_secondary"
                         if method == "ilp_perf" else
                         "regenerated_proxy_from_two_stage_ilp")
        _expect_equal(errors, "%s/%s source_kind" % (pair, method),
                      entry.get("source_kind"), expected_kind)
        for key in ("ilp_audit_pass", "proxy_audit_pass",
                    "exact_bom_match_across_methods",
                    "exact_bundle_match_across_methods"):
            _expect_equal(errors, "%s/%s %s" % (pair, method, key),
                          entry.get(key), True)

        expected_solution_rel = _expected_solution_relative(experts, method)
        solution, solution_path, solution_sha = _verify_manifest_file(
            entry.get("solution_path"), entry.get("solution_sha256"),
            expected_solution_rel, "%s/%s canonical solution" %
            (pair, method), errors)
        params, params_path, params_sha = _verify_manifest_file(
            entry.get("params_path"), entry.get("params_sha256"),
            _expected_params_relative(experts),
            "%s/%s canonical params" % (pair, method), errors)
        if params is not None:
            meta = params.get("meta") or {}
            _expect_equal(errors, "%s/%s params experts" % (pair, method),
                          meta.get("experts"), experts)
        bom, bundle_sha = _solution_contract(
            solution, experts, "%s/%s canonical solution" % (pair, method),
            errors)
        _expect_equal(errors, "%s/%s manifest BOM" % (pair, method),
                      entry.get("bom"), bom)
        _expect_equal(errors, "%s/%s GPU-L" % (pair, method),
                      entry.get("gpu_l"), EXPECTED_SCALES[experts]["gpu_l"])
        _expect_equal(errors, "%s/%s GPU-S" % (pair, method),
                      entry.get("gpu_s"), EXPECTED_SCALES[experts]["gpu_s"])
        _expect_equal(errors, "%s/%s GPU-L/S" % (pair, method),
                      entry.get("gpu_l_s"), "%d/%d" % (
                          EXPECTED_SCALES[experts]["gpu_l"],
                          EXPECTED_SCALES[experts]["gpu_s"]))
        manifest_bundle = str(
            entry.get("bundle_signature_sha256", "")).lower()
        _expect_equal(errors, "%s/%s bundle signature" % (pair, method),
                      manifest_bundle, bundle_sha)
        bundle_by_scale.setdefault(experts, set()).add(manifest_bundle)

        audited = audit_bindings.get(pair_key)
        if audited is None:
            _add_error(errors, "%s/%s lacks an immutable audit binding" %
                       (pair, method))
        else:
            for key, actual in (
                    ("solution_path", solution_path),
                    ("solution_sha256", solution_sha),
                    ("params_path", params_path),
                    ("params_sha256", params_sha),
                    ("bom", bom),
                    ("bundle_signature_sha256", manifest_bundle)):
                _expect_equal(errors, "%s/%s audit binding %s" %
                              (pair, method, key), actual, audited.get(key))

        result = entry.get("result") or {}
        expected_result_dir = root / ("e%03d_%s" %
                                      (experts, raw_method))
        _expect_equal(errors, "%s/%s canonical result_dir" % (pair, method),
                      _canonical_path(result.get("result_dir"),
                                      "%s/%s result_dir" % (pair, method),
                                      errors), expected_result_dir.resolve())
        _expect_equal(errors, "%s/%s canonical results_json_path" %
                      (pair, method),
                      _canonical_path(result.get("results_json_path"),
                                      "%s/%s results_json_path" %
                                      (pair, method), errors),
                      (expected_result_dir / "results.json").resolve())
        bindings[pair_key] = {
            "pair": pair,
            "method": method,
            "experts": experts,
            "solution_path": solution_path,
            "solution_sha256": solution_sha,
            "params_path": params_path,
            "params_sha256": params_sha,
            "bom": bom,
            "bundle_signature_sha256": manifest_bundle,
            "results_json_path": (expected_result_dir / "results.json").resolve(),
        }

    expected_pairs = {(experts, method) for experts in EXPECTED_SCALES
                      for method in METHODS}
    _expect_equal(errors, "canonical input expert/method pairs",
                  set(bindings), expected_pairs)
    for experts in sorted(EXPECTED_SCALES):
        _expect_equal(errors, "E%d canonical bundle signature count" % experts,
                      len(bundle_by_scale.get(experts, set())), 1)
    return bindings, path


def _bind_codegen_to_staged_manifests(codegen_dir, evidence_dir, errors):
    """Bind verified codegen files to the runner's staged-input hash catalog.

    The evidence auditor cryptographically verifies both manifest files in the
    mirrored run bundle.  Requiring every semantic entry to match and rehashing
    every listed source file establishes the staged semantic inputs without
    decompressing multi-gigabyte GPGPU traces from run_evidence.tar.gz.  A
    runner may legitimately rewrite non-semantic ``bench.yml`` paths, so that
    file is hashed but is not required to be byte-identical after staging.
    """
    catalogs = []
    for name in ("source_input_manifest.sha256",
                 "staged_input_manifest.sha256"):
        path = evidence_dir / name
        if not path.is_file():
            _add_error(errors, "missing %s" % name)
            catalogs.append({})
            continue
        try:
            entries = audit._manifest_entries(path)
            catalogs.append(dict(entries))
        except (OSError, UnicodeError, ValueError) as exc:
            _add_error(errors, "invalid %s: %s" % (name, exc))
            catalogs.append({})
    source, staged = catalogs
    required = {
        "coords.json",
        "dispatch_matrix.txt",
        "dispatch_meta.json",
        "moe_coords.txt",
        "params.input.json",
        "solution.input.json",
    }
    semantic = required | {
        name for name in source if name.startswith("topology/mesh_") and
        name.endswith(".gv")}
    for label, catalog in (("source", source), ("staged", staged)):
        for missing in sorted(semantic - set(catalog)):
            _add_error(errors, "%s input manifest omits %s" %
                       (label, missing))
    for relative in sorted(semantic & set(source) & set(staged)):
        _expect_equal(errors, "%s source/staged SHA-256" % relative,
                      source[relative], staged[relative])
    verified = 0
    root = codegen_dir.resolve()
    for relative, expected_sha in sorted(source.items()):
        candidate = root.joinpath(*PurePosixPath(relative).parts)
        try:
            candidate.resolve().relative_to(root)
        except ValueError:
            _add_error(errors, "staged input path escapes codegen: %s" %
                       relative)
            continue
        if not candidate.is_file():
            _add_error(errors, "codegen file listed by staged manifest is "
                       "missing: %s" % relative)
            continue
        try:
            _expect_equal(errors, "%s staged SHA-256" % relative,
                          audit._sha256(candidate), expected_sha)
            verified += 1
        except OSError as exc:
            _add_error(errors, "cannot hash staged input %s: %s" %
                       (relative, exc))
    return verified


def _expected_workers(solution, params, experts, errors):
    expert_tasks = set(params.get("expert_tasks", []))
    sites = [site for site in solution.get("sites", [])
             if str(site.get("type", "")).startswith("GPU") and
             any(task in expert_tasks for task in site.get("tasks", []))]
    try:
        sites = sorted(
            sites,
            key=lambda site: (int(site["l"]), int(site["i"]),
                              int(site["j"]), str(site["type"]),
                              ",".join(str(task) for task in
                                       site.get("tasks", []))))
    except (KeyError, TypeError, ValueError) as exc:
        _add_error(errors, "canonical worker ordering is invalid: %s" % exc)
        return []
    _expect_equal(errors, "canonical expert-worker count", len(sites), experts)
    return sites


def _parse_moe_coords(path, experts, rows, cols, errors):
    width = None
    router = None
    ngpu = None
    gpu = {}
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _add_error(errors, "cannot read moe_coords.txt: %s" % exc)
        return {}
    for number, raw in enumerate(raw_lines, 1):
        payload = raw.split("#", 1)[0].strip()
        if not payload:
            continue
        parts = payload.split()
        try:
            if parts[0] == "width" and len(parts) == 2:
                if width is not None:
                    raise ValueError("duplicate width")
                width = int(parts[1])
            elif parts[0] == "router" and len(parts) == 3:
                if router is not None:
                    raise ValueError("duplicate router")
                router = (int(parts[1]), int(parts[2]))
            elif parts[0] == "ngpu" and len(parts) == 2:
                if ngpu is not None:
                    raise ValueError("duplicate ngpu")
                ngpu = int(parts[1])
            elif parts[0] == "gpu" and len(parts) == 4:
                worker = int(parts[1])
                if worker in gpu:
                    raise ValueError("duplicate gpu %d" % worker)
                gpu[worker] = (int(parts[2]), int(parts[3]))
            else:
                raise ValueError("unknown/malformed record")
        except ValueError as exc:
            _add_error(errors, "moe_coords.txt:%d %s" % (number, exc))
    _expect_equal(errors, "moe_coords width", width, cols)
    _expect_equal(errors, "moe_coords ngpu", ngpu, experts)
    _expect_equal(errors, "moe_coords GPU IDs", sorted(gpu),
                  list(range(experts)))
    if router is None:
        _add_error(errors, "moe_coords has no router")
    elif not (0 <= router[0] < cols and 0 <= router[1] < rows):
        _add_error(errors, "moe_coords router is outside the mesh: %r" %
                   (router,))
    coords = list(gpu.values())
    if len(set(coords)) != len(coords):
        _add_error(errors, "moe_coords GPU coordinates are not unique")
    for worker, (x_coord, y_coord) in gpu.items():
        if not (0 <= x_coord < cols and 0 <= y_coord < rows):
            _add_error(errors, "moe_coords GPU %d is outside the mesh" % worker)
    if router in set(coords):
        _add_error(errors, "moe_coords router overlaps a GPU worker")
    return {"width": width, "router": router, "ngpu": ngpu, "gpu": gpu}


def _validate_topology(path, rows, cols, errors):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _add_error(errors, "cannot read topology %s: %s" % (path, exc))
        return
    expected_name = "mesh_%dx%d" % (rows, cols)
    names = []
    actual_edges = set()
    duplicate_edges = set()
    for number, raw in enumerate(lines, 1):
        payload = raw.split("//", 1)[0].strip()
        match = re.match(r"^graph\s+([^\s{]+)", payload)
        if match:
            names.append(match.group(1))
        if "--" not in payload:
            continue
        if not re.match(r"^[0-9]+(?:--[0-9]+)+\s*;?$", payload):
            _add_error(errors, "topology line %d is malformed: %s" %
                       (number, payload))
            continue
        nodes = [int(value) for value in re.findall(r"[0-9]+", payload)]
        for left, right in zip(nodes, nodes[1:]):
            edge = tuple(sorted((left, right)))
            if edge in actual_edges:
                duplicate_edges.add(edge)
            actual_edges.add(edge)
    _expect_equal(errors, "topology graph name", names, [expected_name])
    if duplicate_edges:
        _add_error(errors, "topology duplicates edges: %s" %
                   sorted(duplicate_edges))
    expected_edges = set()
    for row in range(rows):
        for column in range(cols):
            node = row * cols + column
            if column + 1 < cols:
                expected_edges.add((node, node + 1))
            if row + 1 < rows:
                expected_edges.add((node, node + cols))
    _expect_equal(errors, "topology undirected edge set", actual_edges,
                  expected_edges)
    actual_nodes = {node for edge in actual_edges for node in edge}
    _expect_equal(errors, "topology node set", actual_nodes,
                  set(range(rows * cols)))


def _parse_dispatch_matrix(path, experts, errors):
    headers = {}
    matrices = []
    current = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _add_error(errors, "cannot read dispatch_matrix.txt: %s" % exc)
        return headers, []
    for number, raw in enumerate(lines, 1):
        payload = raw.split("#", 1)[0].strip()
        if not payload:
            continue
        parts = payload.split()
        if parts[0] == "round":
            current = []
            matrices.append(current)
            continue
        if current is None:
            if len(parts) != 2 or parts[0] in headers:
                _add_error(errors, "dispatch_matrix.txt:%d malformed header" %
                           number)
                continue
            try:
                headers[parts[0]] = int(parts[1])
            except ValueError:
                _add_error(errors, "dispatch_matrix.txt:%d non-integer header" %
                           number)
        else:
            try:
                values = [int(value) for value in parts]
            except ValueError:
                _add_error(errors, "dispatch_matrix.txt:%d non-integer row" %
                           number)
                continue
            if len(values) != experts or any(value < 0 for value in values):
                _add_error(errors, "dispatch_matrix.txt:%d invalid row" % number)
            current.append(values)
    _expect_equal(errors, "dispatch matrix round count", len(matrices), 1)
    matrix = matrices[0] if len(matrices) == 1 else []
    _expect_equal(errors, "dispatch matrix row count", len(matrix), experts)
    return headers, matrix


def _largest_remainder_targets(weights, total):
    norm = float(sum(weights))
    exact = [total * float(weight) / norm for weight in weights]
    base = [int(math.floor(value)) for value in exact]
    remaining = total - sum(base)
    order = sorted(range(len(weights)),
                   key=lambda index: (-(exact[index] - base[index]), index))
    for index in order[:remaining]:
        base[index] += 1
    return base


def _validate_dispatch_semantics(matrix, meta, params, experts, errors):
    if not isinstance(meta, dict):
        _add_error(errors, "dispatch_meta.json is not an object")
        return []
    for key, expected in (("ngpu", experts), ("groups", experts),
                          ("topk", 2), ("tokens_per_source", 32)):
        _expect_equal(errors, "dispatch_meta.%s" % key,
                      meta.get(key), expected)
    if len(matrix) != experts or any(len(row) != experts for row in matrix):
        return []
    row_sums = [sum(row) for row in matrix]
    _expect_equal(errors, "physical dispatch row sums", row_sums,
                  [64] * experts)
    _expect_equal(errors, "dispatch_meta per-source assignments",
                  meta.get("per_source_route_assignments"), row_sums)
    _expect_equal(errors, "dispatch total route assignments",
                  meta.get("total_route_assignments"), experts * 64)

    raw_mapping = meta.get("group_to_worker")
    mapping = {}
    if not isinstance(raw_mapping, dict):
        _add_error(errors, "dispatch_meta.group_to_worker is not an object")
    else:
        try:
            mapping = {int(group): int(worker)
                       for group, worker in raw_mapping.items()}
        except (TypeError, ValueError):
            _add_error(errors, "dispatch_meta.group_to_worker is non-integer")
            mapping = {}
    _expect_equal(errors, "logical group IDs", set(mapping),
                  set(range(experts)))
    _expect_equal(errors, "group_to_worker worker IDs", set(mapping.values()),
                  set(range(experts)))
    if len(mapping) != experts or len(set(mapping.values())) != experts:
        return []

    logical = [[matrix[mapping[source]][mapping[destination]]
                for destination in range(experts)]
               for source in range(experts)]
    _expect_equal(errors, "logical dispatch row sums",
                  [sum(row) for row in logical], [64] * experts)
    logical_columns = [sum(logical[row][column] for row in range(experts))
                       for column in range(experts)]
    physical_columns = [sum(matrix[row][column] for row in range(experts))
                        for column in range(experts)]
    if any(value <= 0 for value in logical_columns):
        _add_error(errors, "logical dispatch has a globally zero expert")
    _expect_equal(errors, "dispatch_meta worker column totals",
                  meta.get("worker_token_assignments"), physical_columns)
    _expect_equal(errors, "dispatch_meta logical actual totals",
                  meta.get("global_group_actual_assignments"), logical_columns)
    _expect_equal(errors, "dispatch_meta logical target totals",
                  meta.get("global_group_target_assignments"), logical_columns)
    _expect_equal(errors, "dispatch_meta logical errors",
                  meta.get("global_group_error_assignments"), [0] * experts)
    _expect_equal(errors, "dispatch_meta logical max error",
                  meta.get("global_group_max_abs_error"), 0)
    _expect_equal(errors, "dispatch_meta zero worker count",
                  meta.get("zero_worker_count"), 0)
    _expect_equal(errors, "dispatch_meta zero workers",
                  meta.get("zero_workers"), [])

    meta_params = params.get("meta") or {}
    weights = meta_params.get("group_pi")
    if not isinstance(weights, list) or len(weights) != experts:
        _add_error(errors, "params meta.group_pi does not cover all experts")
    else:
        independently_targeted = _largest_remainder_targets(
            weights, experts * 64)
        _expect_equal(errors, "independent global apportionment",
                      logical_columns, independently_targeted)
    return logical


def _validate_codegen_semantics(codegen_dir, evidence_dir, solution, params,
                                experts, errors):
    side = EXPECTED_SCALES[experts]["mesh_side"]
    verified_inputs = _bind_codegen_to_staged_manifests(
        codegen_dir, evidence_dir, errors)
    coords = _parse_moe_coords(
        codegen_dir / "moe_coords.txt", experts, side, side, errors)
    workers = _expected_workers(solution, params, experts, errors)
    expected_gpu = {}
    for worker, site in enumerate(workers):
        x_coord = int(site["l"]) * int(solution["grid"]["Y"]) + int(site["j"])
        y_coord = int(site["i"])
        expected_gpu[worker] = (x_coord, y_coord)
    _expect_equal(errors, "moe_coords/canonical worker mapping",
                  coords.get("gpu"), expected_gpu)
    cpu_sites = [site for site in solution.get("sites", [])
                 if site.get("type") == "CPU"]
    if len(cpu_sites) == 1:
        cpu = cpu_sites[0]
        expected_router = (int(cpu["l"]) * int(solution["grid"]["Y"]) +
                           int(cpu["j"]), int(cpu["i"]))
        _expect_equal(errors, "moe_coords/canonical router",
                      coords.get("router"), expected_router)

    topology_rel = "topology/mesh_%dx%d.gv" % (side, side)
    source_manifest = evidence_dir / "source_input_manifest.sha256"
    if source_manifest.is_file():
        try:
            entries = dict(audit._manifest_entries(source_manifest))
            if topology_rel not in entries:
                _add_error(errors, "staged manifest omits %s" % topology_rel)
        except (OSError, UnicodeError, ValueError):
            pass
    _validate_topology(codegen_dir / topology_rel, side, side, errors)

    headers, matrix = _parse_dispatch_matrix(
        codegen_dir / "dispatch_matrix.txt", experts, errors)
    for key, expected in (("ngpu", experts), ("experts", experts),
                          ("topk", 2), ("n_rounds", 1),
                          ("tokens_per_source", 32)):
        _expect_equal(errors, "dispatch header %s" % key,
                      headers.get(key), expected)
    dispatch_meta = _load_object(
        codegen_dir / "dispatch_meta.json", "dispatch_meta.json", errors)
    logical = _validate_dispatch_semantics(
        matrix, dispatch_meta, params, experts, errors)

    if isinstance(dispatch_meta, dict):
        meta_workers = dispatch_meta.get("workers")
        if not isinstance(meta_workers, list):
            _add_error(errors, "dispatch_meta.workers is not a list")
        else:
            expected_meta_workers = [
                {"worker": worker, "type": site["type"],
                 "site": [site["l"], site["i"], site["j"]],
                 "tasks": list(site.get("tasks", []))}
                for worker, site in enumerate(workers)
            ]
            _expect_equal(errors, "dispatch_meta/canonical workers",
                          meta_workers, expected_meta_workers)
    return {
        "verified_staged_input_files": verified_inputs,
        "logical_dispatch_matrix": logical,
        "logical_dispatch_sha256": (
            _canonical_value_hash(logical) if logical else None),
        "topology": "%dx%d exact four-neighbor mesh" % (side, side),
        "staged_binding": (
            "verified codegen copy plus matching semantic entries in the "
            "source/staged SHA-256 catalogs; large run_evidence.tar.gz not "
            "decompressed"),
    }


def _safe_archive_name(raw_name, label, errors, allow_root=False):
    if not isinstance(raw_name, str) or not raw_name or "\\" in raw_name:
        _add_error(errors, "%s has invalid archive name %r" %
                   (label, raw_name))
        return None
    name = raw_name
    while name.startswith("./"):
        name = name[2:]
    if allow_root and name in ("", "."):
        return "."
    posix = PurePosixPath(name)
    if (not name or posix.is_absolute() or
            any(part in ("", ".", "..") for part in posix.parts)):
        _add_error(errors, "%s has unsafe archive path %r" %
                   (label, raw_name))
        return None
    normalized = "/".join(posix.parts)
    if normalized != name:
        _add_error(errors, "%s is not normalized: %r" % (label, raw_name))
    return normalized


def _manifest_from_bytes(payload, label, errors):
    try:
        text = payload.decode("utf-8", "strict")
    except UnicodeError as exc:
        _add_error(errors, "%s is not UTF-8: %s" % (label, exc))
        return {}
    entries = {}
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw:
            continue
        match = re.match(r"^([0-9A-Fa-f]{64})  (.+)$", raw)
        if not match:
            _add_error(errors, "%s:%d is not a sha256sum record" %
                       (label, number))
            continue
        name = _safe_archive_name(
            match.group(2), "%s:%d" % (label, number), errors)
        if name is None:
            continue
        if name in entries:
            _add_error(errors, "%s:%d duplicates %s" %
                       (label, number, name))
        entries[name] = match.group(1).lower()
    if not entries:
        _add_error(errors, "%s is empty" % label)
    return entries


def _raw_metric_state():
    return {
        "interchiplet_sync_values": [],
        "interchiplet_phases": set(),
        "bench_max": 0,
        "bench_rows": 0,
        "bench_bad_rows": 0,
        "router_cycles": [],
        "router_done": False,
        "popnet": {},
        "gpu": {},
        "text_errors": [],
    }


def _consume_raw_metric_line(name, raw_line, state):
    try:
        line = raw_line.decode("utf-8", "strict").rstrip("\r\n")
    except UnicodeError as exc:
        state["text_errors"].append("%s is not UTF-8: %s" % (name, exc))
        return
    if name == "interchiplet.stdout":
        match = re.search(r"Benchmark elapses\s+([0-9]+)\s+cycle", line)
        if match:
            state["interchiplet_sync_values"].append(int(match.group(1)))
        match = re.search(r"Round\s+([0-9]+)\s+Phase\s+([12])", line)
        if match:
            state["interchiplet_phases"].add(
                (int(match.group(1)), int(match.group(2))))
    elif name == "bench.txt":
        if not line.strip():
            return
        parts = line.split()
        if len(parts) < 2 or not all(re.match(r"^-?[0-9]+$", value)
                                    for value in parts):
            state["bench_bad_rows"] += 1
            return
        values = [int(value) for value in parts]
        state["bench_rows"] += 1
        state["bench_max"] = max(state["bench_max"], values[0], values[1])
    elif re.match(r"^proc_r2_p1_t0/router\.[^/]+\.log$", name):
        match = re.search(r"\[INTERCMD\]\s+CYCLE\s+([0-9]+)", line)
        if match:
            state["router_cycles"].append(int(match.group(1)))
        if "[CPU router] done" in line:
            state["router_done"] = True
    else:
        pop_match = re.match(
            r"^proc_r([12])_p2_t0/popnet_0\.log$", name)
        if pop_match:
            round_id = int(pop_match.group(1))
            metrics = state["popnet"].setdefault(
                round_id, {"finished": [], "delay": [], "power": []})
            patterns = (
                ("finished", r"^\s*total finished:\s*([0-9]+)\s*$", int),
                ("delay", r"^\s*average delay:\s*"
                 r"([0-9.eE+\-]+)\s*$", float),
                ("power", r"^\s*total power:\s*"
                 r"([0-9.eE+\-]+)\s*$", float),
            )
            for key, pattern, conversion in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        metrics[key].append(conversion(match.group(1)))
                    except ValueError:
                        state["text_errors"].append(
                            "%s has invalid %s" % (name, key))
        gpu_match = re.match(
            r"^proc_r2_p1_t([0-9]+)/gpgpusim\.[0-9]+\.([0-9]+)\.log$",
            name)
        if gpu_match:
            task_id = int(gpu_match.group(1))
            worker_id = int(gpu_match.group(2))
            gpu = state["gpu"].setdefault(
                name, {"task": task_id, "worker": worker_id,
                       "cycles": [], "runner_ok": False, "done": False})
            match = re.match(
                r"^\s*gpu_tot_sim_cycle\s*=\s*([0-9]+)\s*$", line)
            if match:
                gpu["cycles"].append(int(match.group(1)))
            if "external_gpgpusim_runner_rc=0" in line:
                gpu["runner_ok"] = True
            if "] done local_tokens=" in line:
                gpu["done"] = True


def _parse_gpu_cycles_bytes(payload, errors):
    try:
        lines = payload.decode("utf-8", "strict").splitlines()
    except UnicodeError as exc:
        _add_error(errors, "tar gpu_cycles.tsv is not UTF-8: %s" % exc)
        return {}
    values = {}
    for number, line in enumerate(lines, 1):
        parts = line.split("\t")
        if (len(parts) != 2 or not parts[0] or
                not re.match(r"^[0-9]+$", parts[1])):
            _add_error(errors, "tar gpu_cycles.tsv:%d is malformed" % number)
            continue
        name = _safe_archive_name(
            parts[0], "tar gpu_cycles.tsv:%d" % number, errors)
        if name in values:
            _add_error(errors, "tar gpu_cycles.tsv duplicates %s" % name)
        values[name] = int(parts[1])
    return values


def _stream_raw_evidence(evidence_dir, experts, audited_row, errors):
    """Recompute every publishable metric from raw, hash-complete tar data."""
    tar_path = evidence_dir / "run_evidence.tar.gz"
    if not tar_path.is_file():
        _add_error(errors, "missing run_evidence.tar.gz")
        return {}
    state = _raw_metric_state()
    actual_hashes = {}
    seen_members = set()
    ignored_links = []
    captured = {}
    capture_names = {
        "raw_output_manifest.sha256",
        "source_input_manifest.sha256",
        "staged_input_manifest.sha256",
        "gpu_cycles.tsv",
    }
    metric_member = re.compile(
        r"^(?:interchiplet\.stdout|bench\.txt|"
        r"proc_r2_p1_t0/router\.[^/]+\.log|"
        r"proc_r[12]_p2_t0/popnet_0\.log|"
        r"proc_r2_p1_t[0-9]+/gpgpusim\.[0-9]+\.[0-9]+\.log)$")
    try:
        with tarfile.open(str(tar_path), mode="r|gz") as archive:
            for member in archive:
                name = _safe_archive_name(
                    member.name, "tar member", errors, allow_root=True)
                if name is None:
                    continue
                if name in seen_members:
                    _add_error(errors, "tar duplicates member %s" % name)
                    continue
                seen_members.add(name)
                if member.isdir():
                    continue
                if member.issym() or member.islnk():
                    # Link payloads are never opened, followed, hashed, or
                    # accepted as evidence.  The runner's one infrastructure
                    # build link is tolerated solely as an ignored member.
                    if name != "input/build":
                        _add_error(errors, "tar contains unsupported link %s" %
                                   name)
                    ignored_links.append(name)
                    continue
                if not member.isfile():
                    _add_error(errors, "tar contains unsupported special "
                               "member %s" % name)
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    _add_error(errors, "cannot stream tar member %s" % name)
                    continue
                digest = hashlib.sha256()
                capture = bytearray() if name in capture_names else None
                if metric_member.match(name):
                    for raw_line in handle:
                        digest.update(raw_line)
                        if capture is not None:
                            capture.extend(raw_line)
                        _consume_raw_metric_line(name, raw_line, state)
                else:
                    while True:
                        block = handle.read(1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                        if capture is not None:
                            capture.extend(block)
                actual_hashes[name] = digest.hexdigest()
                if capture is not None:
                    captured[name] = bytes(capture)
    except (OSError, tarfile.TarError) as exc:
        _add_error(errors, "cannot stream run_evidence.tar.gz: %s" % exc)
        return {}
    for error in state["text_errors"]:
        _add_error(errors, error)

    raw_payload = captured.get("raw_output_manifest.sha256")
    if raw_payload is None:
        _add_error(errors, "tar omits raw_output_manifest.sha256")
        raw_catalog = {}
    else:
        raw_catalog = _manifest_from_bytes(
            raw_payload, "tar raw_output_manifest.sha256", errors)
    expected_regular = set(raw_catalog) | {"raw_output_manifest.sha256"}
    _expect_equal(errors, "tar/raw-manifest regular member set",
                  set(actual_hashes), expected_regular)
    for name, expected_sha in raw_catalog.items():
        _expect_equal(errors, "raw tar member %s SHA-256" % name,
                      actual_hashes.get(name), expected_sha)
    external_raw = evidence_dir / "raw_output_manifest.sha256"
    if external_raw.is_file() and "raw_output_manifest.sha256" in actual_hashes:
        _expect_equal(errors, "tar/mirrored raw manifest SHA-256",
                      actual_hashes["raw_output_manifest.sha256"],
                      audit._sha256(external_raw))

    staged_payload = captured.get("staged_input_manifest.sha256")
    staged_catalog = (_manifest_from_bytes(
        staged_payload, "tar staged_input_manifest.sha256", errors)
        if staged_payload is not None else {})
    if staged_payload is None:
        _add_error(errors, "tar omits staged_input_manifest.sha256")
    for relative, expected_sha in staged_catalog.items():
        _expect_equal(errors, "tar staged input %s SHA-256" % relative,
                      actual_hashes.get("input/" + relative), expected_sha)
    external_staged = evidence_dir / "staged_input_manifest.sha256"
    if (external_staged.is_file() and
            "staged_input_manifest.sha256" in actual_hashes):
        _expect_equal(errors, "tar/mirrored staged manifest SHA-256",
                      actual_hashes["staged_input_manifest.sha256"],
                      audit._sha256(external_staged))

    sync_values = state["interchiplet_sync_values"]
    sync = sync_values[-1] if sync_values else None
    if not sync_values:
        _add_error(errors, "raw interchiplet.stdout has no Benchmark elapses")
    _expect_equal(errors, "raw InterChiplet phase set",
                  state["interchiplet_phases"],
                  {(1, 1), (1, 2), (2, 1), (2, 2)})
    if state["bench_bad_rows"]:
        _add_error(errors, "raw bench.txt has %d malformed rows" %
                   state["bench_bad_rows"])
    if state["bench_rows"] <= 0:
        _add_error(errors, "raw bench.txt has no records")
    bench = state["bench_max"] or None
    router_cycle = (state["router_cycles"][-1]
                    if state["router_cycles"] else None)
    if not state["router_done"]:
        _add_error(errors, "latest raw router log has no completion marker")
    _expect_equal(errors, "raw router/interchiplet synchronized cycle",
                  router_cycle, sync)
    _expect_equal(errors, "raw synchronized/summary cycles", sync,
                  audited_row.get("legosim_cycles"))
    _expect_equal(errors, "raw bench/summary cycles", bench,
                  audited_row.get("bench_cycles"))
    if sync is not None and bench is not None:
        _expect_equal(errors, "raw completion/summary delta", sync - bench,
                      audited_row.get("completion_delta_cycles"))

    popnet_rounds = {}
    _expect_equal(errors, "raw PopNet round IDs", set(state["popnet"]), {1, 2})
    for round_id in (1, 2):
        metrics = state["popnet"].get(round_id) or {}
        result = {}
        for key in ("finished", "delay", "power"):
            values = metrics.get(key) or []
            _expect_equal(errors, "raw PopNet round %d %s count" %
                          (round_id, key), len(values), 1)
            result[key] = values[-1] if values else None
        _expect_equal(errors, "raw PopNet round %d finished" % round_id,
                      result["finished"],
                      audited_row.get("popnet_expected_finished"))
        popnet_rounds[round_id] = result
    latest_popnet = popnet_rounds.get(2, {})
    _expect_equal(errors, "raw PopNet/summary finished",
                  latest_popnet.get("finished"),
                  audited_row.get("popnet_total_finished"))
    _expect_equal(errors, "raw PopNet/summary average delay",
                  latest_popnet.get("delay"),
                  audited_row.get("popnet_average_delay"))
    _expect_equal(errors, "raw PopNet/summary NoC power",
                  latest_popnet.get("power"),
                  audited_row.get("orion_noc_power_W"))

    gpu_by_worker = {}
    gpu_by_name = {}
    for name, gpu in state["gpu"].items():
        worker = gpu["worker"]
        _expect_equal(errors, "raw GPU task/worker identity %s" % name,
                      gpu["task"], worker + 1)
        if worker in gpu_by_worker:
            _add_error(errors, "raw GPU logs duplicate worker %d" % worker)
        cycles = gpu["cycles"]
        if not cycles:
            _add_error(errors, "raw GPU log has no total cycles: %s" % name)
            cycle = None
        else:
            cycle = cycles[-1]
        if not gpu["runner_ok"]:
            _add_error(errors, "raw GPU runner did not report rc=0: %s" % name)
        if not gpu["done"]:
            _add_error(errors, "raw GPU worker did not complete: %s" % name)
        gpu_by_worker[worker] = cycle
        gpu_by_name[name] = cycle
    _expect_equal(errors, "raw GPU worker IDs", set(gpu_by_worker),
                  set(range(experts)))
    positive_cycles = [cycle for cycle in gpu_by_worker.values()
                       if isinstance(cycle, int) and cycle > 0]
    _expect_equal(errors, "raw positive GPU cycle rows",
                  len(positive_cycles), experts)
    gpu_max = max(positive_cycles) if positive_cycles else None
    _expect_equal(errors, "raw GPU/summary max cycles", gpu_max,
                  audited_row.get("gpu_cycle_max"))
    tsv_payload = captured.get("gpu_cycles.tsv")
    if tsv_payload is None:
        _add_error(errors, "tar omits gpu_cycles.tsv")
        tsv_cycles = {}
    else:
        tsv_cycles = _parse_gpu_cycles_bytes(tsv_payload, errors)
    _expect_equal(errors, "raw GPU logs/gpu_cycles.tsv", gpu_by_name,
                  tsv_cycles)

    return {
        "tar_regular_files_verified": len(actual_hashes),
        "raw_manifest_entries_verified": len(raw_catalog),
        "staged_input_entries_verified_in_tar": len(staged_catalog),
        "link_payloads_accepted": 0,
        "ignored_non_evidence_links": ignored_links,
        "synchronized_cycles_from_interchiplet": sync,
        "synchronized_cycles_from_router": router_cycle,
        "bench_cycles_from_raw_trace": bench,
        "gpu_cycle_rows_from_raw_logs": len(positive_cycles),
        "gpu_cycle_max_from_raw_logs": gpu_max,
        "popnet_round1": popnet_rounds.get(1),
        "popnet_round2_latest": popnet_rounds.get(2),
    }


def _validate_dispatch_expected(evidence_dir, experts, summary_expected,
                                errors):
    """Independently reproduce the runner's expected PopNet message count."""
    path = evidence_dir / "expected_dispatch_matrix.log"
    if not path.is_file():
        return  # The shared evidence audit already reports this precisely.
    rows = {}
    try:
        with open(str(path), "r", encoding="utf-8", errors="strict") as handle:
            for number, raw in enumerate(handle, 1):
                parts = raw.split()
                if len(parts) != experts + 2 or parts[0] != "src":
                    _add_error(errors,
                               "expected dispatch line %d is malformed" % number)
                    continue
                try:
                    source = int(parts[1])
                    values = [int(value) for value in parts[2:]]
                except ValueError:
                    _add_error(errors,
                               "expected dispatch line %d is non-integer" % number)
                    continue
                if source in rows:
                    _add_error(errors,
                               "expected dispatch duplicates source %d" % source)
                if any(value < 0 for value in values):
                    _add_error(errors,
                               "expected dispatch source %d has negative tokens" %
                               source)
                rows[source] = values
    except (OSError, UnicodeError) as exc:
        _add_error(errors, "cannot parse expected dispatch: %s" % exc)
        return
    _expect_equal(errors, "expected dispatch source IDs", sorted(rows),
                  list(range(experts)))
    if len(rows) != experts:
        return
    nonlocal_messages = 0
    for source, values in rows.items():
        nonlocal_messages += sum(
            1 for destination, tokens in enumerate(values)
            if source != destination and tokens > 0)
    independently_expected = 2 * nonlocal_messages + 2 * experts
    _expect_equal(errors, "independent PopNet expected-finished",
                  summary_expected, independently_expected)


def _validate_solution_and_codegen(results_path, manifest, record, experts,
                                   audited_row, canonical):
    errors = audited_row["errors"]
    config = EXPECTED_SCALES[experts]
    expected_case = "e%03d_m%02d" % (experts, config["mesh_side"])
    _expect_equal(errors, "case_id", record.get("case_id"), expected_case)
    if not isinstance(canonical, dict):
        _add_error(errors, "%s/%s has no canonical input-manifest binding" %
                   (expected_case, record.get("method")))
        canonical = {}

    solution_path = audit._resolve_record_path(
        record.get("solution_path"), results_path)
    if solution_path is None or not solution_path.is_file():
        _add_error(errors, "missing solution JSON: %s" % solution_path)
        solution = None
    else:
        recorded_digest = str(record.get("solution_sha256", "")).lower()
        if not audit.SHA256_RE.match(recorded_digest):
            _add_error(errors, "solution_sha256 is not a SHA-256 digest")
        else:
            try:
                _expect_equal(errors, "solution SHA-256",
                              audit._sha256(solution_path), recorded_digest)
            except OSError as exc:
                _add_error(errors, "cannot hash solution JSON: %s" % exc)
        solution = _load_object(solution_path, "solution JSON", errors)

    _validate_record_canonical_identity(
        solution_path, record.get("solution_sha256"), canonical, errors)

    if solution is not None:
        _validate_grid(solution.get("grid"), experts, "solution.grid", errors)
        budgets = solution.get("budgets")
        if not isinstance(budgets, dict):
            _add_error(errors, "solution.budgets is not an object")
        else:
            _validate_bom_mapping(budgets.get("nmin"), experts,
                                  "solution.budgets.nmin", errors)
            _validate_bom_mapping(budgets.get("nmax"), experts,
                                  "solution.budgets.nmax", errors)
        _site_bom(solution.get("sites"), experts, "solution.sites", errors)
        solution_bom, bundle_sha = _solution_contract(
            solution, experts, "record solution", errors)
        _expect_equal(errors, "record/canonical solution BOM", solution_bom,
                      canonical.get("bom"))
        _expect_equal(errors, "record/canonical bundle signature", bundle_sha,
                      canonical.get("bundle_signature_sha256"))

    manifest_params_path = audit._resolve_record_path(
        manifest.get("params_path"), results_path)
    _expect_equal(errors, "results/canonical params path",
                  manifest_params_path, canonical.get("params_path"))
    _expect_equal(errors, "results/canonical params SHA-256",
                  str(manifest.get("params_sha256", "")).lower(),
                  canonical.get("params_sha256"))

    codegen = record.get("codegen") or {}
    fallback = (results_path.parent / "layouts" / expected_case /
                str(record.get("method")) / "codegen")
    codegen_dir = audit._resolve_record_path(
        codegen.get("codegen_dir"), results_path, fallback)
    if not codegen_dir.is_dir():
        return
    _require_within(codegen_dir, results_path.parent, "codegen directory",
                    errors)

    params_input = codegen_dir / "params.input.json"
    solution_input = codegen_dir / "solution.input.json"
    manifest_params_digest = str(manifest.get("params_sha256", "")).lower()
    recorded_solution_digest = str(record.get("solution_sha256", "")).lower()
    for label, path, expected_digest in (
            ("codegen params.input.json", params_input,
             manifest_params_digest),
            ("codegen solution.input.json", solution_input,
             recorded_solution_digest)):
        if not path.is_file():
            _add_error(errors, "missing %s" % label)
        elif audit.SHA256_RE.match(expected_digest):
            try:
                _expect_equal(errors, "%s SHA-256" % label,
                              audit._sha256(path), expected_digest)
            except OSError as exc:
                _add_error(errors, "cannot hash %s: %s" % (label, exc))
    if solution_input.is_file():
        try:
            _expect_equal(errors, "codegen/canonical solution SHA-256",
                          audit._sha256(solution_input),
                          canonical.get("solution_sha256"))
        except OSError as exc:
            _add_error(errors, "cannot hash codegen solution.input.json: %s" %
                       exc)
    if params_input.is_file():
        try:
            _expect_equal(errors, "codegen/canonical params SHA-256",
                          audit._sha256(params_input),
                          canonical.get("params_sha256"))
        except OSError as exc:
            _add_error(errors, "cannot hash codegen params.input.json: %s" %
                       exc)

    coords = _load_object(codegen_dir / "coords.json", "coords JSON", errors)
    if coords is not None:
        _validate_grid(coords.get("grid"), experts, "coords.grid", errors)
        mesh = coords.get("mesh")
        if not isinstance(mesh, dict):
            _add_error(errors, "coords.mesh is not an object")
        else:
            side = config["mesh_side"]
            _expect_equal(errors, "coords.mesh.rows", mesh.get("rows"), side)
            _expect_equal(errors, "coords.mesh.cols", mesh.get("cols"), side)
        _site_bom(coords.get("sites"), experts, "coords.sites", errors)

    metric = record.get("solution_metrics") or {}
    _expect_equal(errors, "solution_metrics.gpu_l_count",
                  metric.get("gpu_l_count"), config["gpu_l"])
    _expect_equal(errors, "solution_metrics.gpu_s_count",
                  metric.get("gpu_s_count"), config["gpu_s"])
    _expect_equal(errors, "audited mesh", audited_row.get("mesh"),
                  "%dx%d" % (config["mesh_side"], config["mesh_side"]))
    _expect_equal(errors, "audited GPU-L count",
                  audited_row.get("gpu_l_count"), config["gpu_l"])
    _expect_equal(errors, "audited GPU-S count",
                  audited_row.get("gpu_s_count"), config["gpu_s"])

    power = audited_row.get("orion_noc_power_W")
    if not isinstance(power, (int, float)) or not math.isfinite(power) or power <= 0:
        _add_error(errors, "ORION NoC power is not finite and positive")
    cycles = audited_row.get("legosim_cycles")
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles <= 0:
        _add_error(errors, "synchronized LEGOSim cycles is not a positive integer")
    evidence_dir = Path(audited_row["evidence_dir"])
    _require_within(evidence_dir, results_path.parent, "evidence directory",
                    errors)

    canonical_params = None
    canonical_params_path = canonical.get("params_path")
    if canonical_params_path and Path(canonical_params_path).is_file():
        canonical_params = _load_object(
            Path(canonical_params_path), "canonical params", errors)
    else:
        _add_error(errors, "canonical params are unavailable for semantic audit")
    semantic = {}
    if (solution is not None and canonical_params is not None and
            codegen_dir.is_dir() and evidence_dir.is_dir()):
        semantic = _validate_codegen_semantics(
            codegen_dir, evidence_dir, solution, canonical_params, experts,
            errors)
    raw_evidence = (_stream_raw_evidence(
        evidence_dir, experts, audited_row, errors)
        if evidence_dir.is_dir() else {})

    # The text summary is the extraction source.  Cross-check its critical
    # fields against the independently serialized JSON summary so hand-editing
    # either representation cannot silently change a table cell.
    legosim_record = record.get("legosim") or {}
    _expect_equal(errors, "JSON completion_delta_cycles",
                  legosim_record.get("completion_delta_cycles"),
                  audited_row.get("completion_delta_cycles"))
    embedded = legosim_record.get("summary")
    if not isinstance(embedded, dict):
        _add_error(errors, "legosim.summary is not an object")
    else:
        for key, expected in (
                ("makespan_cycles", audited_row.get("bench_cycles")),
                ("synchronized_makespan_cycles", cycles),
                ("completion_delta_cycles",
                 audited_row.get("completion_delta_cycles")),
                ("gpu_cycle_max", audited_row.get("gpu_cycle_max")),
                ("popnet_total_power", power),
                ("popnet_average_delay",
                 audited_row.get("popnet_average_delay")),
                ("popnet_total_finished",
                 audited_row.get("popnet_total_finished")),
                ("popnet_expected_finished",
                 audited_row.get("popnet_expected_finished")),
                ("rounds_phase1", 2),
                ("rounds_phase2", 2),
                ("n_gpu_workers", experts)):
            _expect_equal(errors, "embedded legosim.summary.%s" % key,
                          embedded.get(key), expected)
    _validate_dispatch_expected(
        evidence_dir, experts,
        audited_row.get("popnet_expected_finished"), errors)

    audited_row.update({
        "solution_path": str(solution_path) if solution_path else "",
        "solution_sha256": recorded_solution_digest,
        "params_sha256": manifest_params_digest,
        "canonical_bundle_signature_sha256": canonical.get(
            "bundle_signature_sha256"),
        "semantic_input_audit": {
            key: value for key, value in semantic.items()
            if key != "logical_dispatch_matrix"
        },
        "raw_evidence_audit": raw_evidence,
        "_logical_dispatch_matrix": semantic.get("logical_dispatch_matrix"),
    })
    audited_row["audit_status"] = "PASS" if not errors else "FAIL"


def _validate_manifest(results_path, manifest, errors):
    if manifest.get("status") != "complete":
        _add_error(errors, "%s status=%r, expected 'complete'" %
                   (results_path, manifest.get("status")))
    if not manifest.get("finished_utc"):
        _add_error(errors, "%s has no finished_utc" % results_path)
    _expect_equal(errors, "%s rounds" % results_path,
                  manifest.get("rounds"), 2)
    _expect_equal(errors, "%s codegen_only" % results_path,
                  manifest.get("codegen_only"), False)

    params_path = audit._resolve_record_path(
        manifest.get("params_path"), results_path)
    if params_path is None or not params_path.is_file():
        _add_error(errors, "%s has no readable params JSON: %s" %
                   (results_path, params_path))
        return None, params_path
    params = _load_object(params_path, "manifest params", errors)
    experts = None
    if params is not None:
        meta = params.get("meta")
        experts = meta.get("experts") if isinstance(meta, dict) else None
        if not isinstance(experts, int) or isinstance(experts, bool):
            _add_error(errors, "%s params meta.experts is not an integer" %
                       results_path)
            experts = None
        elif experts not in EXPECTED_SCALES:
            _add_error(errors, "%s has unexpected expert count %d" %
                       (results_path, experts))
    digest = str(manifest.get("params_sha256", "")).lower()
    if not audit.SHA256_RE.match(digest):
        _add_error(errors, "%s has invalid params_sha256" % results_path)
    else:
        try:
            _expect_equal(errors, "%s params SHA-256" % results_path,
                          audit._sha256(params_path), digest)
        except OSError as exc:
            _add_error(errors, "%s cannot hash params: %s" %
                       (results_path, exc))
    return experts, params_path


def collect(input_root):
    root = Path(input_root).resolve()
    if not root.exists():
        raise ValueError("input root does not exist: %s" % root)
    manifests = audit._discover_results(root)
    if not manifests:
        raise ValueError("no results.json found below %s" % root)

    global_errors = []
    canonical_bindings, canonical_manifest_path = \
        _validate_final_input_manifest(root, global_errors)
    audited_rows = []
    by_pair = {}
    params_by_scale = {}
    for results_path in manifests:
        manifest = _load_object(results_path, "results manifest",
                                global_errors)
        if manifest is None:
            continue
        experts, _ = _validate_manifest(results_path, manifest, global_errors)
        layouts = manifest.get("layouts")
        if not isinstance(layouts, list):
            _add_error(global_errors, "%s layouts is not a list" % results_path)
            continue
        if not layouts:
            _add_error(global_errors, "%s has no completed layout record" %
                       results_path)
        for record in layouts:
            if not isinstance(record, dict):
                _add_error(global_errors,
                           "%s contains a non-object layout" % results_path)
                continue
            row = audit._audit_layout(results_path, manifest, experts, record, 2)
            method = row.get("method")
            row_experts = row.get("experts")
            if row_experts in EXPECTED_SCALES:
                canonical = canonical_bindings.get((row_experts, method))
                if canonical is not None:
                    _expect_equal(
                        row["errors"], "results.json canonical path",
                        results_path.resolve(),
                        canonical.get("results_json_path"))
                _validate_solution_and_codegen(
                    results_path, manifest, record, row_experts, row,
                    canonical)
            else:
                _add_error(row["errors"], "unexpected expert scale %r" %
                           row_experts)
                row["audit_status"] = "FAIL"
            pair = (row_experts, method)
            if pair in by_pair:
                _add_error(global_errors,
                           "duplicate expert/method result: E%s/%s" % pair)
                _add_error(row["errors"], "duplicate expert/method result")
                row["audit_status"] = "FAIL"
            else:
                by_pair[pair] = row
            audited_rows.append(row)
            if row_experts in EXPECTED_SCALES:
                params_by_scale.setdefault(row_experts, set()).add(
                    str(manifest.get("params_sha256", "")).lower())

    expected_pairs = {(experts, method) for experts in EXPECTED_SCALES
                      for method in METHODS}
    found_pairs = set(by_pair)
    missing = sorted(expected_pairs - found_pairs)
    unexpected = sorted(found_pairs - expected_pairs,
                        key=lambda pair: (pair[0] if isinstance(pair[0], int)
                                          else -1, str(pair[1])))
    if missing:
        _add_error(global_errors, "missing expert/method results: %s" %
                   ", ".join("E%d/%s" % pair for pair in missing))
    if unexpected:
        _add_error(global_errors, "unexpected expert/method results: %s" %
                   ", ".join("E%s/%s" % pair for pair in unexpected))
    for experts in sorted(EXPECTED_SCALES):
        digests = params_by_scale.get(experts, set())
        if len(digests) > 1:
            _add_error(global_errors,
                       "E%d methods do not share one params SHA-256" % experts)

    # The physical rows/columns differ with worker ordering.  Undo each
    # method's group_to_worker permutation and require one byte-identical
    # logical source-group x destination-group matrix per scale.
    for experts in sorted(EXPECTED_SCALES):
        scale_rows = [by_pair.get((experts, method)) for method in METHODS]
        if any(row is None for row in scale_rows):
            continue
        reference = scale_rows[0].get("_logical_dispatch_matrix")
        scale_equal = bool(reference)
        for method, row in zip(METHODS, scale_rows):
            logical = row.get("_logical_dispatch_matrix")
            if not logical:
                _add_error(row["errors"],
                           "logical dispatch matrix is unavailable")
                scale_equal = False
            elif reference and logical != reference:
                _add_error(row["errors"],
                           "logical dispatch differs across methods at E%d" %
                           experts)
                scale_equal = False
        for row in scale_rows:
            semantic = row.get("semantic_input_audit") or {}
            semantic["logical_dispatch_equal_across_methods"] = scale_equal
            row["semantic_input_audit"] = semantic
            row["audit_status"] = "PASS" if not row["errors"] else "FAIL"
            row.pop("_logical_dispatch_matrix", None)

    failed = [row for row in audited_rows
              if row.get("audit_status") != "PASS"]
    complete = (not global_errors and not failed and
                found_pairs == expected_pairs and len(audited_rows) == 15)
    report = {
        "schema_version": 1,
        "audit_status": "PASS" if complete else "FAIL",
        "source_root": str(root),
        "metric_contract": {
            "cycles": "LEGOSim synchronized_makespan_cycles",
            "power": "PopNet/ORION NoC-only total power (W)",
            "power_excludes": ["CPU", "GPU", "package", "wall-plug"],
            "analytical_fallback": False,
            "rounds_phase1": 2,
            "rounds_phase2": 2,
            "semantic_input_binding": (
                "verified codegen files rehashed against matching semantic "
                "entries in mirrored source/staged SHA-256 manifests; "
                "run_evidence.tar.gz is not decompressed"),
            "topology_contract": "exact R x C undirected four-neighbor mesh",
            "dispatch_contract": (
                "inverse-permuted logical matrices identical across methods; "
                "64 assignments/source and independently recomputed global "
                "largest-remainder margins"),
            "result_metric_source": (
                "streamed run_evidence.tar.gz raw logs, with every regular "
                "member checked against raw_output_manifest.sha256"),
            "raw_metric_cross_checks": (
                "InterChiplet plus router synchronized cycles; bench trace; "
                "both PopNet rounds; E latest-round GPGPU-Sim logs; summary "
                "and results JSON must agree"),
        },
        "required_experts": sorted(EXPECTED_SCALES),
        "required_methods": list(METHODS),
        "results_manifests": [str(path) for path in manifests],
        "canonical_input_manifest": (
            str(canonical_manifest_path) if canonical_manifest_path else None),
        "canonical_input_manifest_sha256": (
            audit._sha256(canonical_manifest_path)
            if canonical_manifest_path and canonical_manifest_path.is_file()
            else None),
        "record_count": len(audited_rows),
        "pass_count": len(audited_rows) - len(failed),
        "fail_count": len(failed),
        "global_errors": global_errors,
        "audit_rows": sorted(
            audited_rows,
            key=lambda row: (row.get("experts") or -1,
                             str(row.get("method")))),
    }
    if complete:
        report["table_rows"] = _build_table_rows(by_pair)
    return report


def _build_table_rows(by_pair):
    rows = []
    for experts in sorted(EXPECTED_SCALES):
        config = EXPECTED_SCALES[experts]
        row = {
            "experts": experts,
            "mesh": "%dx%d" % (config["mesh_side"], config["mesh_side"]),
            "gpu_l_s": "%d/%d" % (config["gpu_l"], config["gpu_s"]),
        }
        for method in METHODS:
            source = by_pair[(experts, method)]
            row["%s_synchronized_cycles" % method] = \
                source["legosim_cycles"]
            row["%s_orion_noc_power_W" % method] = \
                source["orion_noc_power_W"]
        rows.append(row)
    return rows


def _atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with open(str(temporary), "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    os.replace(str(temporary), str(path))


def _json_text(report):
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def _csv_text(rows):
    # csv.writer needs a file-like object, and StringIO works in Python 3.7+.
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=TABLE_CSV_FIELDS,
                            extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _latex_integer(value):
    return format(int(value), ",").replace(",", "{,}")


def _latex_text(rows):
    lines = [
        "% Auto-generated by collect_table3_legosim.py after a 15/15 audit.",
        "% Cycles: synchronized LEGOSim end-to-end makespan.",
        "% Power: PopNet/ORION NoC only (W); excludes CPU/GPU/package power.",
    ]
    for row in rows:
        side = row["mesh"].split("x")
        cells = [
            str(row["experts"]),
            "$%s\\times%s$" % (side[0], side[1]),
            row["gpu_l_s"],
        ]
        for method in METHODS:
            # Accept the pre-normalization ``ilp_*`` keys in small downstream
            # callers/tests while all newly generated CSV uses ``ilp_perf``.
            cycle_key = "%s_synchronized_cycles" % method
            power_key = "%s_orion_noc_power_W" % method
            if method == "ilp_perf" and cycle_key not in row:
                cycle_key = "ilp_synchronized_cycles"
                power_key = "ilp_orion_noc_power_W"
            cells.append(_latex_integer(
                row[cycle_key]))
            cells.append("%.6f" % row[power_key])
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = _parse_args(argv)
    report = collect(args.input_root)
    if report["audit_status"] != "PASS":
        print(json.dumps({
            "audit_status": "FAIL",
            "record_count": report["record_count"],
            "pass_count": report["pass_count"],
            "fail_count": report["fail_count"],
            "global_errors": report["global_errors"],
            "publication_outputs_written": False,
        }, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    root = Path(args.input_root).resolve()
    prefix = (Path(args.out_prefix).resolve() if args.out_prefix else
              root / "table3_legosim_real")
    json_path = Path(str(prefix) + ".json")
    csv_path = Path(str(prefix) + ".csv")
    tex_path = Path(str(prefix) + ".tex")
    rows = report["table_rows"]
    _atomic_text(json_path, _json_text(report))
    _atomic_text(csv_path, _csv_text(rows))
    _atomic_text(tex_path, _latex_text(rows))
    print(json.dumps({
        "audit_status": "PASS",
        "record_count": report["record_count"],
        "power_scope": "PopNet/ORION NoC only (W)",
        "analytical_fallback": False,
        "json": str(json_path),
        "csv": str(csv_path),
        "latex_rows": str(tex_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        sys.exit(2)
