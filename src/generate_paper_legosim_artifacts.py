#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper TeX data interfaces from one complete unified LEGOSim CSV.

The generator is fail closed.  Full mode requires exactly 5x3 scaling rows and
8x4 budget rows; scaling-only mode requires exactly the 5x3 fresh scaling rows
and deliberately leaves the existing audited budget artifact untouched.  Both
modes require accepted evidence tiers, PASS validation, positive
synchronized cycles/ORION NoC power, and dynamic energy consistent with the
measured power and active interval.  Figure 3 reports normalized cycles,
average NoC power, and a synchronized-cycle versus nominal dynamic-NoC-energy
trade-off.  No analytical or estimated fallback exists.  All three outputs
are staged before any destination is atomically replaced.

For every scale-out row, the CSV must bind the common replay BOM to a fresh
joint min-T MILP solution, prove that no GPU type count was fixed, and carry the
package caps and actual resource use.  Floorplet/TAP remain placement-only
comparisons which inherit that independently audited ILP-selected BOM.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path

from run_scaling_qap_experiment import (
    SCALING_AREA_MM2_PER_EXPERT,
    SCALING_POWER_W_PER_EXPERT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "out/unified_legosim_data/legosim_metrics.csv"
DEFAULT_PAPER_DIR = "paper_date"
SCALING_METHODS = ("ilp_perf", "floorplet", "tap")
BUDGET_METHODS = ("ilp_perf", "matched_random", "floorplet", "tap")
METHOD_MACRO = {
    "ilp_perf": "ILPPerf",
    "matched_random": "MatchedRandom",
    "floorplet": "Floorplet",
    "tap": "TAP",
}
SCALE_CONFIG = {
    # Only workload/grid identity is fixed here.  CPU/GPU-L/GPU-S counts are
    # read from the manifest-bound replay CSV and must match across all three
    # placement-only comparisons.  The ILP row is fresh joint selection; the
    # two prior-art proxies inherit its BOM to isolate placement quality.
    "e008_m06": (8, "6x6"),
    "e016_m08": (16, "8x8"),
    "e032_m12": (32, "12x12"),
    "e064_m12": (64, "12x12"),
    "e128_m16": (128, "16x16"),
}
BUDGET_CONFIG = {
    # Only workload identity and hard package budgets are configuration.  The
    # GPU-L/GPU-S counts are joint-MILP outputs read from the replay CSV; they
    # must never be encoded here as an expected tier BOM.
    "area_970": ("area", 970, 970, 1625),
    "area_1260": ("area", 1260, 1260, 1625),
    "area_1550": ("area", 1550, 1550, 1625),
    "area_2130": ("area", 2130, 2130, 1625),
    "power_785": ("power", 785, 2130, 785),
    "power_995": ("power", 995, 2130, 995),
    "power_1205": ("power", 1205, 2130, 1205),
    "power_1625": ("power", 1625, 2130, 1625),
}
EXPECTED_SCALING_ROWS = len(SCALE_CONFIG) * len(SCALING_METHODS)
EXPECTED_BUDGET_ROWS = len(BUDGET_CONFIG) * len(BUDGET_METHODS)
EXPECTED_FULL_ROWS = EXPECTED_SCALING_ROWS + EXPECTED_BUDGET_ROWS
OUTPUT_NAMES = (
    "table3-three-methods-data.tex",
    "scaling-tradeoff-data.tex",
    "budget-curves-data.tex",
)
CPU_PEAK_W = 65
GPU_L_PEAK_W = 300
GPU_S_PEAK_W = 90
CPU_AREA_MM2 = 90
GPU_L_AREA_MM2 = 400
GPU_S_AREA_MM2 = 110


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate strict paper TeX data from final unified CSV")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--paper-dir", default=DEFAULT_PAPER_DIR)
    parser.add_argument(
        "--check-only", action="store_true",
        help="validate completeness without writing paper artifacts")
    parser.add_argument(
        "--scaling-only", action="store_true",
        help=("accept exactly the 5x3 fresh scaling matrix and update only "
              "Table III/Fig. 3 data; leave the audited budget figure intact"))
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value, label):
    text = str(value or "").strip()
    try:
        number = int(text)
    except ValueError:
        raise ValueError("%s is not an integer: %r" % (label, value))
    if str(number) != text or number <= 0:
        raise ValueError("%s is not a positive integer: %r" % (label, value))
    return number


def _nonnegative_int(value, label):
    text = str(value or "").strip()
    try:
        number = int(text)
    except ValueError:
        raise ValueError("%s is not an integer: %r" % (label, value))
    if str(number) != text or number < 0:
        raise ValueError("%s is not a nonnegative integer: %r" %
                         (label, value))
    return number


def _positive_float(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s is not numeric: %r" % (label, value))
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("%s is not finite and positive: %r" % (label, value))
    return number


def _exact_float(value, expected, label):
    number = _positive_float(value, label)
    if not math.isclose(number, float(expected), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("%s=%r, expected %r" % (label, number, expected))


def _strict_false(value, label):
    if str(value).strip().lower() != "false":
        raise ValueError("%s=%r, expected False" % (label, value))
    return False


def _read_and_validate(path, suites=None):
    suites = frozenset(suites or ("scaling", "budget"))
    if not suites or not suites.issubset({"scaling", "budget"}):
        raise ValueError("invalid requested suites: %r" % sorted(suites))
    if not path.is_file():
        raise ValueError("unified CSV not found: %s" % path)
    with open(str(path), "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "suite", "case_id", "sweep_dimension", "budget_value",
            "area_budget_mm2", "power_budget_W", "area_used_mm2",
            "experts", "mesh",
            "method", "cpu_count", "gpu_l_count", "gpu_s_count",
            "shared_bom_peak_power_W", "bom_selection_provenance",
            "gpu_counts_fixed", "fixed_bom", "solver_mip_gap",
            "synchronized_cycles", "orion_noc_power_W",
            "noc_active_cycles", "nominal_orion_dynamic_noc_energy_uJ",
            "legosim_validation", "audit_status", "evidence_tier",
            "evidence_dir", "source_results_json", "errors",
        }
        missing_columns = sorted(required - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError("unified CSV missing columns: %s" %
                             ", ".join(missing_columns))
        raw_rows = list(reader)

    expected = set()
    if "scaling" in suites:
        expected.update({("scaling", case_id, method)
                         for case_id in SCALE_CONFIG
                         for method in SCALING_METHODS})
    if "budget" in suites:
        expected.update({("budget", case_id, method)
                         for case_id in BUDGET_CONFIG
                         for method in BUDGET_METHODS})
    if len(raw_rows) != len(expected):
        raise ValueError("unified CSV has %d rows, expected exactly %d" %
                         (len(raw_rows), len(expected)))
    by_key = {}
    for index, row in enumerate(raw_rows, 2):
        key = (row.get("suite"), row.get("case_id"), row.get("method"))
        if key not in expected:
            raise ValueError("CSV line %d has unexpected key %r" %
                             (index, key))
        if key in by_key:
            raise ValueError("duplicate row %r" % (key,))
        if row.get("audit_status") != "PASS":
            raise ValueError("%r audit_status is not PASS" % (key,))
        if row.get("legosim_validation") != "PASS":
            raise ValueError("%r legosim_validation is not PASS" % (key,))
        if str(row.get("errors") or "").strip():
            raise ValueError("%r contains audit errors" % (key,))
        row["_cycles"] = _positive_int(
            row.get("synchronized_cycles"), "%r cycles" % (key,))
        row["_noc_W"] = _positive_float(
            row.get("orion_noc_power_W"), "%r NoC W" % (key,))
        row["_noc_active_cycles"] = _positive_int(
            row.get("noc_active_cycles"), "%r NoC active cycles" % (key,))
        row["_noc_energy_uJ"] = _positive_float(
            row.get("nominal_orion_dynamic_noc_energy_uJ"),
            "%r dynamic NoC energy" % (key,))
        expected_energy = (row["_noc_W"] * row["_noc_active_cycles"] /
                           1000.0)
        if not math.isclose(row["_noc_energy_uJ"], expected_energy,
                            rel_tol=1.0e-9, abs_tol=1.0e-9):
            raise ValueError(
                "%r dynamic NoC energy %.12g is inconsistent with "
                "power*active_cycles/1000 = %.12g" %
                (key, row["_noc_energy_uJ"], expected_energy))
        row["_cpu"] = _positive_int(row.get("cpu_count"),
                                    "%r CPU count" % (key,))
        row["_gpu_l"] = _nonnegative_int(row.get("gpu_l_count"),
                                          "%r GPU-L count" % (key,))
        row["_gpu_s"] = _nonnegative_int(row.get("gpu_s_count"),
                                           "%r GPU-S count" % (key,))
        row["_area_budget_mm2"] = _positive_float(
            row.get("area_budget_mm2"), "%r area budget" % (key,))
        row["_power_budget_W"] = _positive_float(
            row.get("power_budget_W"), "%r power budget" % (key,))
        row["_area_used_mm2"] = _positive_float(
            row.get("area_used_mm2"), "%r area used" % (key,))
        row["_chip_peak_W"] = _positive_float(
            row.get("shared_bom_peak_power_W"),
            "%r selected chip peak" % (key,))
        if row.get("bom_selection_provenance") != "fresh-joint-min-T-MILP":
            raise ValueError("%r lacks fresh joint-BOM provenance" % (key,))
        _strict_false(row.get("gpu_counts_fixed"),
                      "%r gpu_counts_fixed" % (key,))
        _strict_false(row.get("fixed_bom"), "%r fixed_bom" % (key,))
        expected_tier = ("strict_scaling_joint_bom_v4"
                         if key[0] == "scaling" else
                         "fresh_constraint_replay_v1")
        if row.get("evidence_tier") != expected_tier:
            raise ValueError("%r evidence_tier=%r, expected %r" %
                             (key, row.get("evidence_tier"), expected_tier))
        for field in ("evidence_dir", "source_results_json"):
            if not str(row.get(field) or "").strip():
                raise ValueError("%r has no %s" % (key, field))
        by_key[key] = row
    missing = sorted(expected - set(by_key))
    if missing:
        raise ValueError("unified CSV lacks %d required rows" % len(missing))

    for case_id, config in (SCALE_CONFIG.items()
                            if "scaling" in suites else ()):
        experts, mesh = config
        selected_bom = None
        expected_area_budget = SCALING_AREA_MM2_PER_EXPERT * experts
        expected_power_budget = SCALING_POWER_W_PER_EXPERT * experts
        for method in SCALING_METHODS:
            row = by_key[("scaling", case_id, method)]
            if int(row.get("experts")) != experts or row.get("mesh") != mesh:
                raise ValueError("scaling identity mismatch for %s/%s" %
                                 (case_id, method))
            actual_bom = (row["_cpu"], row["_gpu_l"], row["_gpu_s"])
            if actual_bom[0] != 1 or actual_bom[1] + actual_bom[2] < experts:
                raise ValueError("invalid scaling BOM for %s/%s: %r" %
                                 (case_id, method, actual_bom))
            _exact_float(row["_area_budget_mm2"], expected_area_budget,
                         "%s/%s scaling area budget" % (case_id, method))
            _exact_float(row["_power_budget_W"], expected_power_budget,
                         "%s/%s scaling power budget" % (case_id, method))
            expected_area_used = (
                CPU_AREA_MM2 * actual_bom[0] +
                GPU_L_AREA_MM2 * actual_bom[1] +
                GPU_S_AREA_MM2 * actual_bom[2])
            expected_chip_peak = (
                CPU_PEAK_W * actual_bom[0] +
                GPU_L_PEAK_W * actual_bom[1] +
                GPU_S_PEAK_W * actual_bom[2])
            _exact_float(row["_area_used_mm2"], expected_area_used,
                         "%s/%s scaling area used" % (case_id, method))
            _exact_float(row["_chip_peak_W"], expected_chip_peak,
                         "%s/%s scaling peak power" % (case_id, method))
            if expected_area_used > expected_area_budget + 1.0e-9:
                raise ValueError("%s/%s exceeds scaling area cap" %
                                 (case_id, method))
            if expected_chip_peak > expected_power_budget + 1.0e-9:
                raise ValueError("%s/%s exceeds scaling power cap" %
                                 (case_id, method))
            if selected_bom is None:
                selected_bom = actual_bom
            elif actual_bom != selected_bom:
                raise ValueError(
                    "three-method scaling BOM mismatch for %s: %r != %r" %
                    (case_id, actual_bom, selected_bom))

    for case_id, config in (BUDGET_CONFIG.items()
                            if "budget" in suites else ()):
        dimension, value, area, power = config
        shared_bom = None
        for method in BUDGET_METHODS:
            row = by_key[("budget", case_id, method)]
            if int(row.get("experts")) != 8 or row.get("mesh") != "6x6":
                raise ValueError("budget identity mismatch for %s/%s" %
                                 (case_id, method))
            if row.get("sweep_dimension") != dimension:
                raise ValueError("budget sweep mismatch for %s/%s" %
                                 (case_id, method))
            _exact_float(row.get("budget_value"), value, "budget value")
            _exact_float(row.get("area_budget_mm2"), area, "area budget")
            _exact_float(row.get("power_budget_W"), power, "power budget")
            actual_bom = (row["_cpu"], row["_gpu_l"], row["_gpu_s"])
            if actual_bom[0] != 1 or actual_bom[1] + actual_bom[2] < 8:
                raise ValueError("invalid budget BOM for %s/%s: %r" %
                                 (case_id, method, actual_bom))
            if shared_bom is None:
                shared_bom = actual_bom
            elif actual_bom != shared_bom:
                raise ValueError(
                    "four-method budget BOM mismatch for %s: %r != %r" %
                    (case_id, actual_bom, shared_bom))
            expected_area_used = (
                CPU_AREA_MM2 * actual_bom[0] +
                GPU_L_AREA_MM2 * actual_bom[1] +
                GPU_S_AREA_MM2 * actual_bom[2])
            expected_chip_peak = (
                CPU_PEAK_W * actual_bom[0] +
                GPU_L_PEAK_W * actual_bom[1] +
                GPU_S_PEAK_W * actual_bom[2])
            _exact_float(row["_area_used_mm2"], expected_area_used,
                         "%s/%s budget area used" % (case_id, method))
            _exact_float(row["_chip_peak_W"], expected_chip_peak,
                         "%s/%s budget peak power" % (case_id, method))
            if expected_area_used > area + 1.0e-9:
                raise ValueError(
                    "%s/%s selected BOM uses %.3f mm2 above Amax %.3f" %
                    (case_id, method, expected_area_used, area))
            if expected_chip_peak > power + 1.0e-9:
                raise ValueError(
                    "%s/%s selected BOM uses %.3f W above Pmax %.3f" %
                    (case_id, method, expected_chip_peak, power))

    return by_key


def _header(source_path, source_hash, description):
    return (
        "%% Auto-generated by generate_paper_legosim_artifacts.py.\n"
        "%% %s\n"
        "%% Source CSV: %s\n"
        "%% Source SHA-256: %s\n"
        "%% Metrics: synchronized LEGOSim cycles; PopNet/ORION NoC-only W; "
        "nominal dynamic NoC energy over the audited active interval.\n" %
        (description, source_path.as_posix(), source_hash))


def _coords(points, digits):
    template = "%%.%df" % digits
    return "".join("(%s,%s)" % (x_value, template % y_value)
                   for x_value, y_value in points)


def _labeled_coords(points, x_digits, y_digits):
    """Format PGFPlots coordinates with explicit symbolic point metadata."""
    x_template = "%%.%df" % x_digits
    y_template = "%%.%df" % y_digits
    return " ".join("(%s,%s) [%s]" %
                    (x_template % x_value, y_template % y_value, label)
                    for x_value, y_value, label in points)


def _scale_norm_ymax(maximum_normalized_cycles):
    """Return a readable upper bound with 7.5--10% plot headroom.

    Normalized performance is at least one by construction.  Start from a
    conventional 1-2-5-derived quantum (0.05 for ordinary 1.x ratios), round
    a 7.5% target upward, and refine the quantum only if that rounding would
    exceed 10% headroom.  The result therefore cannot clip an observed point
    and remains a clean decimal axis limit over both small and large ratios.
    """
    maximum = float(maximum_normalized_cycles)
    if not math.isfinite(maximum) or maximum < 1.0:
        raise ValueError("maximum normalized cycles must be finite and >= 1")

    target = maximum * 1.075
    upper = maximum * 1.10
    exponent = math.floor(math.log10(maximum))
    quantum = 5.0 * (10.0 ** (exponent - 2))
    while True:
        # The small subtraction prevents an exact multiple represented a few
        # ulps high from being rounded up by an entire extra quantum.
        units = math.ceil(target / quantum - 1.0e-12)
        limit = units * quantum
        if limit <= upper + max(1.0e-12, abs(upper) * 1.0e-12):
            break
        quantum /= 2.0

    # Six decimal places comfortably cover the normalized ranges in this
    # experiment and avoid binary-float tails in the TeX interface.
    return float("%.6f" % limit)


def _scale_noc_power_ymax(maximum_noc_power_W):
    """Return a clean power-axis upper bound with 7.5--10% headroom."""
    maximum = float(maximum_noc_power_W)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum NoC power must be finite and positive")

    target = maximum * 1.075
    upper = maximum * 1.10
    exponent = math.floor(math.log10(maximum))
    quantum = 5.0 * (10.0 ** (exponent - 2))
    while True:
        units = math.ceil(target / quantum - 1.0e-12)
        limit = units * quantum
        if limit <= upper + max(1.0e-12, abs(upper) * 1.0e-12):
            break
        quantum /= 2.0
    return float("%.6f" % limit)


def _table_text(rows, source_path, source_hash):
    lines = [_header(source_path, source_hash,
                     "Table III fresh joint-BOM selection and three-method "
                     "placement replay data.")]
    for case_id, config in sorted(SCALE_CONFIG.items(),
                                  key=lambda item: item[1][0]):
        experts, mesh = config
        perf_row = rows[("scaling", case_id, "ilp_perf")]
        cpu = perf_row["_cpu"]
        gpu_l = perf_row["_gpu_l"]
        gpu_s = perf_row["_gpu_s"]
        area_budget = perf_row["_area_budget_mm2"]
        power_budget = perf_row["_power_budget_W"]
        area_used = perf_row["_area_used_mm2"]
        chip_peak = perf_row["_chip_peak_W"]
        mesh_tex = "$%s\\times%s$" % tuple(mesh.split("x"))
        cells = [str(experts), mesh_tex, "%.0f" % area_budget,
                 "%.0f" % power_budget, "%d/%d" % (gpu_l, gpu_s),
                 "%.0f" % area_used, "%.0f" % chip_peak]
        for method in SCALING_METHODS:
            row = rows[("scaling", case_id, method)]
            cells.extend(["%.3f" % (row["_cycles"] / 1000.0),
                          "%.6f" % row["_noc_W"]])
        lines.append(" & ".join(cells) + " \\\\")
    lines.append(r"\bottomrule")
    return "\n".join(lines) + "\n"


def _scaling_text(rows, source_path, source_hash):
    lines = [_header(source_path, source_hash,
                     "Scaling coordinates; normalized cycles, average NoC power, "
                     "and cycle-energy trade-off.")]
    normalized = {method: [] for method in SCALING_METHODS}
    noc_power = {method: [] for method in SCALING_METHODS}
    cycle_energy = {method: [] for method in SCALING_METHODS}
    for case_id, config in sorted(SCALE_CONFIG.items(),
                                  key=lambda item: item[1][0]):
        experts = config[0]
        minimum = min(rows[("scaling", case_id, method)]["_cycles"]
                      for method in SCALING_METHODS)
        for method in SCALING_METHODS:
            row = rows[("scaling", case_id, method)]
            normalized[method].append((experts, row["_cycles"] / float(minimum)))
            noc_power[method].append((experts, row["_noc_W"]))
            cycle_energy[method].append(
                (row["_cycles"] / 1000.0, row["_noc_energy_uJ"],
                 "E%d" % experts))
    maximum_normalized = max(
        value for points in normalized.values() for _, value in points)
    normalized_ymax = _scale_norm_ymax(maximum_normalized)
    maximum_noc_power = max(
        value for points in noc_power.values() for _, value in points)
    noc_power_ymax = _scale_noc_power_ymax(maximum_noc_power)
    lines.extend([
        r"\newif\ifScaleNormReady", r"\ScaleNormReadytrue",
        r"\newif\ifScaleNoCPowerReady", r"\ScaleNoCPowerReadytrue",
        r"\newif\ifScaleNoCEnergyReady", r"\ScaleNoCEnergyReadytrue",
        r"\providecommand{\ScaleNormYMax}{%.6f}" % normalized_ymax,
        r"\providecommand{\ScaleNoCPowerYMax}{%.6f}" % noc_power_ymax,
    ])
    for method in SCALING_METHODS:
        macro = METHOD_MACRO[method]
        lines.append(r"\providecommand{\%sNormCoords}{%s}" %
                     (macro, _coords(normalized[method], 6)))
    for method in SCALING_METHODS:
        macro = METHOD_MACRO[method]
        lines.append(r"\providecommand{\%sNoCPowerCoords}{%s}" %
                     (macro, _coords(noc_power[method], 6)))
    for method in SCALING_METHODS:
        macro = METHOD_MACRO[method]
        lines.append(r"\providecommand{\%sCycleEnergyCoords}{%s}" %
                     (macro, _labeled_coords(
                         cycle_energy[method], 3, 6)))
    return "\n".join(lines) + "\n"


def _budget_text(rows, source_path, source_hash):
    lines = [_header(source_path, source_hash,
                     "Budget-curve coordinates for four real replay methods.")]
    all_cycles_k = []
    for dimension, ordered_cases in (
            ("Area", ("area_970", "area_1260", "area_1550", "area_2130")),
            ("Power", ("power_785", "power_995", "power_1205", "power_1625"))):
        for method in BUDGET_METHODS:
            points = []
            for case_id in ordered_cases:
                row = rows[("budget", case_id, method)]
                x_value = BUDGET_CONFIG[case_id][1]
                y_value = row["_cycles"] / 1000.0
                all_cycles_k.append(y_value)
                points.append((x_value, y_value))
            lines.append(r"\renewcommand{\Budget%s%sCoords}{%s}" %
                         (dimension, METHOD_MACRO[method], _coords(points, 3)))
        selected_points = []
        for case_id in ordered_cases:
            row = rows[("budget", case_id, "ilp_perf")]
            selected_points.append((
                BUDGET_CONFIG[case_id][1], row["_cycles"] / 1000.0,
                "%d/%d" % (row["_gpu_l"], row["_gpu_s"])))
        lines.append(r"\renewcommand{\Budget%sILPSelectionCoords}{%s}" %
                     (dimension, _labeled_coords(
                         selected_points, 0, 3)))
    y_min = math.floor(min(all_cycles_k)) - 1
    y_max = math.ceil(max(all_cycles_k)) + 1
    lines.extend([
        r"\renewcommand{\BudgetCyclesYMin}{%d}" % y_min,
        r"\renewcommand{\BudgetCyclesYMax}{%d}" % y_max,
        r"\BudgetCurvesReadytrue",
    ])
    return "\n".join(lines) + "\n"


def _render(path, suites=None):
    suites = frozenset(suites or ("scaling", "budget"))
    rows = _read_and_validate(path, suites=suites)
    digest = _sha256(path)
    outputs = {}
    if "scaling" in suites:
        outputs[OUTPUT_NAMES[0]] = _table_text(rows, path, digest)
        outputs[OUTPUT_NAMES[1]] = _scaling_text(rows, path, digest)
    if "budget" in suites:
        outputs[OUTPUT_NAMES[2]] = _budget_text(rows, path, digest)
    return outputs


def _atomic_outputs(paper_dir, outputs):
    paper_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for name in OUTPUT_NAMES:
            if name not in outputs:
                continue
            destination = paper_dir / name
            temporary = destination.with_name(
                destination.name + ".tmp.%d" % os.getpid())
            with open(str(temporary), "w", encoding="utf-8", newline="") as handle:
                handle.write(outputs[name])
            staged.append((temporary, destination))
        for temporary, destination in staged:
            os.replace(str(temporary), str(destination))
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()


def main(argv=None):
    args = _parse_args(argv)
    input_path = _resolve(args.input)
    suites = (("scaling",) if args.scaling_only else
              ("scaling", "budget"))
    outputs = _render(input_path, suites=suites)
    paper_dir = _resolve(args.paper_dir)
    if not args.check_only:
        _atomic_outputs(paper_dir, outputs)
    print(json.dumps({
        "status": "PASS",
        "check_only": args.check_only,
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "scaling_rows": (EXPECTED_SCALING_ROWS
                         if "scaling" in suites else 0),
        "budget_rows": (EXPECTED_BUDGET_ROWS if "budget" in suites else 0),
        "outputs": [str(paper_dir / name) for name in OUTPUT_NAMES
                    if name in outputs],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        sys.exit(2)
