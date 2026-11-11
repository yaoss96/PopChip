#!/usr/bin/env python3
"""Build paper-facing tables and dependency-free SVGs for a budget sweep.

The input is an experiment directory produced by
``run_budget_mesh_experiment.py``.  The script reads both
``raw_results.csv`` (the flat result records) and ``raw_results.json`` (run
metadata and simulator-only fields), then writes:

* ``paper_summary.csv`` -- one paired ILP/matched-random row per budget case;
* ``REPORT.md`` -- a Chinese, paper-facing audit and result summary;
* ``area_budget_results.svg`` and ``power_budget_results.svg`` -- each contains
  real LEGOSim makespan, measured speedup, and search-time panels.

Only Python's standard library is used.  Incomplete/in-progress experiments are
valid inputs: missing values are left blank and shown as N/A instead of being
silently imputed.  A measured speedup is emitted only when *both* layouts have
a positive LEGOSim makespan and ``validation=PASS``.
"""

from __future__ import division, print_function

import argparse
import csv
import datetime
import html
import json
import math
import os
import sys
import tempfile
from pathlib import Path


SUMMARY_FIELDS = [
    "experiment_id",
    "experiment_status",
    "case_id",
    "workload",
    "topology",
    "sweep_dimension",
    "budget_value",
    "mesh_L",
    "mesh_X",
    "mesh_Y",
    "area_budget_mm2_model",
    "power_budget_W_model",
    "per_column_power_budget_W_model",
    "ilp_status",
    "matched_random_status",
    "ilp_budget_check",
    "matched_random_budget_check",
    "ilp_chiplet_bom",
    "matched_random_chiplet_bom",
    "chiplet_bom_match",
    "ilp_legosim_validation",
    "matched_random_legosim_validation",
    "ilp_legosim_makespan_cycles",
    "matched_random_legosim_makespan_cycles",
    "measured_speedup_matched_over_ilp",
    "ilp_legosim_wall_s",
    "matched_random_legosim_wall_s",
    "ilp_popnet_total_power_NoC_only",
    "matched_random_popnet_total_power_NoC_only",
    "ilp_popnet_average_delay",
    "matched_random_popnet_average_delay",
    "ilp_predicted_makespan_cycles",
    "matched_random_predicted_makespan_cycles",
    "predicted_speedup_matched_over_ilp",
    "ilp_time_to_first_feasible_s",
    "ilp_model_build_s",
    "ilp_primary_wall_s",
    "ilp_solver_wall_s",
    "ilp_orchestrator_search_wall_s",
    "matched_random_search_wall_s",
    "search_time_ratio_matched_over_ilp",
    "ilp_mip_gap",
    "ilp_primary_status",
    "ilp_certificate_status",
    "ilp_total_area_mm2",
    "matched_random_total_area_mm2",
    "ilp_peak_power_W_model",
    "matched_random_peak_power_W_model",
    "ilp_area_slack_mm2",
    "matched_random_area_slack_mm2",
    "ilp_power_slack_W_model",
    "matched_random_power_slack_W_model",
    "ilp_error",
    "matched_random_error",
]

METHOD_LABELS = {
    "ilp": "ILP",
    "matched_random": "Matched random",
}

ILP_COLOR = "#0072B2"
MATCHED_COLOR = "#D55E00"
SPEEDUP_COLOR = "#009E73"
PROOF_COLOR = "#CC79A7"
GRID_COLOR = "#DCE3EA"
TEXT_COLOR = "#17212B"
MUTED_COLOR = "#5F6B76"


def _utc_now():
    return datetime.datetime.now(
        datetime.timezone.utc).replace(
            microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n",
        prefix=path.name + ".", suffix=".tmp",
        dir=str(path.parent), delete=False)
    tmp_name = handle.name
    try:
        with handle:
            handle.write(text)
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="",
        prefix=path.name + ".", suffix=".tmp",
        dir=str(path.parent), delete=False)
    tmp_name = handle.name
    try:
        with handle:
            writer = csv.DictWriter(
                handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore",
                lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: _csv_value(row.get(key)) for key in SUMMARY_FIELDS
                })
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if value == 0:
            return "0"
        return "%.12g" % value
    return str(value)


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _positive(value):
    result = _number(value)
    return result if result is not None and result > 0 else None


def _text(value):
    if value is None:
        return ""
    return str(value).strip()


def _upper(value):
    return _text(value).upper()


def _normalize_method(value):
    value = _text(value).lower().replace("_", "-")
    if value == "ilp" or value.startswith("ilp-"):
        return "ilp"
    if ("matched" in value or "resource-matched" in value or
            value in ("random", "matched-layout")):
        return "matched_random"
    return value.replace("-", "_")


def _listify(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def _load_inputs(experiment_dir):
    warnings = []
    csv_path = experiment_dir / "raw_results.csv"
    json_path = experiment_dir / "raw_results.json"
    rows = []
    raw_json = {}

    if csv_path.exists():
        try:
            with open(str(csv_path), "r", encoding="utf-8-sig",
                      newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    warnings.append("raw_results.csv 没有表头。")
                else:
                    for index, row in enumerate(reader):
                        if not isinstance(row, dict):
                            continue
                        clean = {
                            _text(key): value
                            for key, value in row.items() if key is not None
                        }
                        if any(_text(value) for value in clean.values()):
                            clean["_source_index"] = index
                            rows.append(clean)
        except (OSError, csv.Error, UnicodeError) as exc:
            warnings.append("无法完整读取 raw_results.csv：%s" % exc)
    else:
        warnings.append("缺少 raw_results.csv。")

    if json_path.exists():
        try:
            with open(str(json_path), "r", encoding="utf-8") as handle:
                value = json.load(handle)
            if isinstance(value, dict):
                raw_json = value
            else:
                warnings.append("raw_results.json 顶层不是对象。")
        except (OSError, ValueError, UnicodeError) as exc:
            warnings.append("无法完整读取 raw_results.json：%s" % exc)
    else:
        warnings.append("缺少 raw_results.json。")

    if not rows and raw_json:
        rows = _rows_from_json(raw_json)
        if rows:
            warnings.append(
                "CSV 没有可用数据行；已从 raw_results.json 恢复汇总行。")

    if not csv_path.exists() and not json_path.exists():
        raise FileNotFoundError(
            "experiment directory contains neither raw_results.csv nor "
            "raw_results.json: %s" % experiment_dir)
    return rows, raw_json, warnings, csv_path, json_path


def _rows_from_json(raw_json):
    """Best-effort reconstruction for a header-only/truncated CSV."""
    rows = []
    experiment = raw_json.get("experiment", {})
    grid = experiment.get("grid", {})
    experiment_id = _text(experiment.get("experiment_id"))
    for case_index, case in enumerate(_listify(raw_json.get("cases"))):
        if not isinstance(case, dict):
            continue
        budgets = case.get("budgets", {})
        base = {
            "experiment_id": experiment_id,
            "case_id": case.get("case_id"),
            "workload": experiment.get("workload"),
            "sweep_dimension": case.get("sweep_dimension"),
            "budget_value": case.get("budget_value"),
            "L": grid.get("L"),
            "X": grid.get("X"),
            "Y": grid.get("Y"),
            "a_max_mm2": budgets.get("a_max"),
            "p_max_W": budgets.get("p_max"),
            "p_col_W": budgets.get("p_col"),
        }
        layouts = _listify(case.get("layouts"))
        if not layouts and isinstance(case.get("ilp"), dict):
            layouts = [{
                "method": "ilp",
                "metrics": case.get("ilp", {}).get("metrics", {}),
                "solution_path": case.get("ilp", {}).get("solution_path"),
            }]
        for layout_index, layout in enumerate(layouts):
            if not isinstance(layout, dict):
                continue
            method = _normalize_method(layout.get("method"))
            row = dict(base)
            row["_source_index"] = case_index * 100 + layout_index
            row["method"] = method
            row.update(layout.get("metrics", {}))
            codegen = layout.get("codegen") or {}
            legosim = layout.get("legosim") or {}
            row["codegen_status"] = codegen.get("status")
            row["dispatch_formal"] = codegen.get("dispatch_formal")
            row["legosim_status"] = (
                legosim.get("status") if legosim else "not_requested")
            row["legosim_validation"] = legosim.get("validation")
            row["measured_makespan_cycles"] = (
                legosim.get("measured_makespan_cycles"))
            row["legosim_wall_s"] = (
                (legosim.get("process") or {}).get("wall_s"))
            row["solution_path"] = layout.get("solution_path")
            if method == "ilp":
                process = (case.get("ilp") or {}).get("process") or {}
            else:
                process = (
                    (case.get("matched_baseline") or {}).get("process") or {})
            row["orchestrator_solve_wall_s"] = process.get("wall_s")
            row["returncode"] = process.get("returncode")
            row["timed_out"] = process.get("timed_out")
            row["error"] = (
                layout.get("error") or legosim.get("error") or
                codegen.get("error") or "")
            if not layout.get("solution_path"):
                row["status"] = "solve_failed"
            elif legosim and legosim.get("status") != "ok":
                row["status"] = "legosim_failed"
            elif codegen and codegen.get("status") != "ok":
                row["status"] = "codegen_failed"
            else:
                row["status"] = "ok"
            rows.append(row)
    return rows


def _json_indexes(raw_json):
    layout_index = {}
    case_index = {}
    for case in _listify(raw_json.get("cases")):
        if not isinstance(case, dict):
            continue
        case_id = _text(case.get("case_id"))
        if not case_id:
            continue
        case_index[case_id] = case
        for layout in _listify(case.get("layouts")):
            if not isinstance(layout, dict):
                continue
            method = _normalize_method(layout.get("method"))
            if method:
                layout_index[(case_id, method)] = layout
    return layout_index, case_index


def _row_quality(row):
    valid = _valid_legosim(row)
    measured = _positive(row.get("measured_makespan_cycles")) is not None
    status_ok = _text(row.get("status")).lower() == "ok"
    legosim_ok = _text(row.get("legosim_status")).lower() == "ok"
    populated = sum(
        1 for key, value in row.items()
        if not key.startswith("_") and _text(value))
    return (
        1 if valid else 0,
        1 if measured else 0,
        1 if legosim_ok else 0,
        1 if status_ok else 0,
        populated,
        int(_number(row.get("_source_index")) or 0),
    )


def _valid_legosim(row):
    return (
        isinstance(row, dict) and
        _upper(row.get("legosim_validation")) == "PASS" and
        _positive(row.get("measured_makespan_cycles")) is not None)


def _best_row(candidates):
    candidates = [row for row in candidates if isinstance(row, dict)]
    return max(candidates, key=_row_quality) if candidates else None


def _value(row, key):
    return row.get(key) if isinstance(row, dict) else None


def _first_value(rows, key, default=None):
    for row in rows:
        value = _value(row, key)
        if _text(value):
            return value
    return default


def _safe_ratio(numerator, denominator):
    numerator = _positive(numerator)
    denominator = _positive(denominator)
    if numerator is None or denominator is None:
        return None
    return numerator / denominator


def _json_layout_metrics(layout):
    if not isinstance(layout, dict):
        return {}
    legosim = layout.get("legosim") or {}
    summary = legosim.get("summary") or {}
    if not isinstance(summary, dict):
        return {}
    return {
        "popnet_total_power": _number(summary.get("popnet_total_power")),
        "popnet_average_delay": _number(summary.get(
            "popnet_average_delay")),
    }


def _chiplet_counts_from_case(case, method):
    if not isinstance(case, dict):
        return None
    if method == "matched_random":
        summary = (case.get("matched_baseline") or {}).get("summary") or {}
        counts = summary.get("chiplet_type_counts")
        if isinstance(counts, dict) and counts:
            return counts
    return None


def _format_bom(row, json_counts=None):
    counts = {}
    if isinstance(json_counts, dict):
        for key, value in json_counts.items():
            number = _number(value)
            if number is not None and number > 0:
                counts[_text(key)] = int(round(number))
    if not counts and isinstance(row, dict):
        known = [
            ("CPU", "cpu_count"),
            ("GPU-L", "gpu_l_count"),
            ("GPU-S", "gpu_s_count"),
        ]
        for label, key in known:
            number = _number(row.get(key))
            if number is not None and number > 0:
                counts[label] = int(round(number))
        occupied = _number(row.get("occupied_sites"))
        known_total = sum(counts.values())
        if occupied is not None and occupied > known_total + 0.5:
            counts["Other"] = int(round(occupied - known_total))
    if not counts:
        return ""
    preferred = ["CPU", "GPU-L", "GPU-S"]
    ordered = [key for key in preferred if key in counts]
    ordered.extend(sorted(key for key in counts if key not in preferred))
    return " + ".join("%s×%d" % (key, counts[key]) for key in ordered)


def _budget_check(row):
    if not isinstance(row, dict):
        return "N/A"
    area = _number(row.get("total_area_mm2"))
    area_max = _number(row.get("a_max_mm2"))
    power = _number(row.get("peak_power_W"))
    power_max = _number(row.get("p_max_W"))
    checks = []
    if area is not None and area_max is not None:
        checks.append(area <= area_max + max(1e-7, abs(area_max) * 1e-9))
    if power is not None and power_max is not None:
        checks.append(
            power <= power_max + max(1e-7, abs(power_max) * 1e-9))
    if not checks:
        return "N/A"
    return "PASS" if all(checks) else "FAIL"


def _pair_cases(rows, raw_json):
    layout_index, case_index = _json_indexes(raw_json)
    grouped = {}
    order = []
    for row in rows:
        case_id = _text(row.get("case_id"))
        method = _normalize_method(row.get("method"))
        if not case_id or method not in ("ilp", "matched_random"):
            continue
        if case_id not in grouped:
            grouped[case_id] = {"ilp": [], "matched_random": []}
            order.append(case_id)
        grouped[case_id][method].append(row)

    # JSON case records let an in-progress/header-only run remain visible.
    for case_id in case_index:
        if case_id not in grouped:
            grouped[case_id] = {"ilp": [], "matched_random": []}
            order.append(case_id)

    experiment = raw_json.get("experiment", {})
    experiment_status = _text(raw_json.get("status")) or "unknown"
    default_grid = experiment.get("grid", {})
    summaries = []
    for case_id in order:
        ilp = _best_row(grouped[case_id]["ilp"])
        matched = _best_row(grouped[case_id]["matched_random"])
        case = case_index.get(case_id, {})
        case_budgets = case.get("budgets", {}) if isinstance(case, dict) else {}
        candidates = [ilp, matched]
        dimension = _text(_first_value(
            candidates, "sweep_dimension",
            case.get("sweep_dimension") if isinstance(case, dict) else ""))
        budget_value = _number(_first_value(
            candidates, "budget_value",
            case.get("budget_value") if isinstance(case, dict) else None))
        area_budget = _number(_first_value(
            candidates, "a_max_mm2", case_budgets.get("a_max")))
        power_budget = _number(_first_value(
            candidates, "p_max_W", case_budgets.get("p_max")))
        p_col = _number(_first_value(
            candidates, "p_col_W", case_budgets.get("p_col")))
        grid_l = _number(_first_value(
            candidates, "L", default_grid.get("L")))
        grid_x = _number(_first_value(
            candidates, "X", default_grid.get("X")))
        grid_y = _number(_first_value(
            candidates, "Y", default_grid.get("Y")))

        ilp_valid = _valid_legosim(ilp)
        matched_valid = _valid_legosim(matched)
        ilp_measured = (
            _positive(ilp.get("measured_makespan_cycles"))
            if ilp_valid else None)
        matched_measured = (
            _positive(matched.get("measured_makespan_cycles"))
            if matched_valid else None)
        measured_speedup = _safe_ratio(matched_measured, ilp_measured)
        ilp_predicted = _positive(_value(
            ilp, "predicted_makespan_cycles"))
        matched_predicted = _positive(_value(
            matched, "predicted_makespan_cycles"))
        predicted_speedup = _safe_ratio(
            matched_predicted, ilp_predicted)
        ilp_search = _positive(_value(
            ilp, "orchestrator_solve_wall_s"))
        matched_search = _positive(_value(
            matched, "orchestrator_solve_wall_s"))

        ilp_layout = layout_index.get((case_id, "ilp"))
        matched_layout = layout_index.get((case_id, "matched_random"))
        ilp_json_metrics = _json_layout_metrics(ilp_layout)
        matched_json_metrics = _json_layout_metrics(matched_layout)
        ilp_bom = _format_bom(
            ilp, _chiplet_counts_from_case(case, "ilp"))
        matched_bom = _format_bom(
            matched, _chiplet_counts_from_case(case, "matched_random"))
        bom_match = (
            ilp_bom == matched_bom if ilp_bom and matched_bom else None)

        summary = {
            "experiment_id": _text(_first_value(
                candidates, "experiment_id",
                experiment.get("experiment_id"))),
            "experiment_status": experiment_status,
            "case_id": case_id,
            "workload": _text(_first_value(
                candidates, "workload", experiment.get("workload"))),
            "topology": _text(experiment.get("topology")) or "mesh",
            "sweep_dimension": dimension,
            "budget_value": budget_value,
            "mesh_L": grid_l,
            "mesh_X": grid_x,
            "mesh_Y": grid_y,
            "area_budget_mm2_model": area_budget,
            "power_budget_W_model": power_budget,
            "per_column_power_budget_W_model": p_col,
            "ilp_status": _text(_value(ilp, "status")) or "missing",
            "matched_random_status": (
                _text(_value(matched, "status")) or "missing"),
            "ilp_budget_check": _budget_check(ilp),
            "matched_random_budget_check": _budget_check(matched),
            "ilp_chiplet_bom": ilp_bom,
            "matched_random_chiplet_bom": matched_bom,
            "chiplet_bom_match": bom_match,
            "ilp_legosim_validation": (
                _text(_value(ilp, "legosim_validation")) or "N/A"),
            "matched_random_legosim_validation": (
                _text(_value(matched, "legosim_validation")) or "N/A"),
            "ilp_legosim_makespan_cycles": ilp_measured,
            "matched_random_legosim_makespan_cycles": matched_measured,
            "measured_speedup_matched_over_ilp": measured_speedup,
            "ilp_legosim_wall_s": _number(_value(
                ilp, "legosim_wall_s")),
            "matched_random_legosim_wall_s": _number(_value(
                matched, "legosim_wall_s")),
            "ilp_popnet_total_power_NoC_only": ilp_json_metrics.get(
                "popnet_total_power"),
            "matched_random_popnet_total_power_NoC_only": (
                matched_json_metrics.get("popnet_total_power")),
            "ilp_popnet_average_delay": ilp_json_metrics.get(
                "popnet_average_delay"),
            "matched_random_popnet_average_delay": (
                matched_json_metrics.get("popnet_average_delay")),
            "ilp_predicted_makespan_cycles": ilp_predicted,
            "matched_random_predicted_makespan_cycles": matched_predicted,
            "predicted_speedup_matched_over_ilp": predicted_speedup,
            "ilp_time_to_first_feasible_s": _number(_value(
                ilp, "time_to_first_feasible_s")),
            "ilp_model_build_s": _number(_value(ilp, "model_build_s")),
            "ilp_primary_wall_s": _number(_value(
                ilp, "primary_wall_s")),
            "ilp_solver_wall_s": _number(_value(ilp, "solver_wall_s")),
            "ilp_orchestrator_search_wall_s": ilp_search,
            "matched_random_search_wall_s": matched_search,
            "search_time_ratio_matched_over_ilp": _safe_ratio(
                matched_search, ilp_search),
            "ilp_mip_gap": _number(_value(ilp, "mip_gap")),
            "ilp_primary_status": _text(_value(
                ilp, "primary_status")),
            "ilp_certificate_status": _text(_value(
                ilp, "certificate_status")),
            "ilp_total_area_mm2": _number(_value(
                ilp, "total_area_mm2")),
            "matched_random_total_area_mm2": _number(_value(
                matched, "total_area_mm2")),
            "ilp_peak_power_W_model": _number(_value(
                ilp, "peak_power_W")),
            "matched_random_peak_power_W_model": _number(_value(
                matched, "peak_power_W")),
            "ilp_area_slack_mm2": _number(_value(
                ilp, "area_slack_mm2")),
            "matched_random_area_slack_mm2": _number(_value(
                matched, "area_slack_mm2")),
            "ilp_power_slack_W_model": _number(_value(
                ilp, "power_slack_W")),
            "matched_random_power_slack_W_model": _number(_value(
                matched, "power_slack_W")),
            "ilp_error": _text(_value(ilp, "error")),
            "matched_random_error": _text(_value(matched, "error")),
        }
        summaries.append(summary)

    dimension_order = {"area": 0, "power": 1}
    summaries.sort(key=lambda item: (
        dimension_order.get(item.get("sweep_dimension"), 9),
        item.get("budget_value")
        if item.get("budget_value") is not None else float("inf"),
        item.get("case_id", "")))
    return summaries


def _fmt_number(value, digits=3):
    value = _number(value)
    if value is None:
        return "—"
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e15:
        return "{:,}".format(int(round(value)))
    return ("{:,." + str(digits) + "f}").format(value).rstrip(
        "0").rstrip(".")


def _fmt_cycles(value):
    return _fmt_number(value, 1)


def _fmt_seconds(value):
    value = _number(value)
    if value is None:
        return "—"
    if value < 0.001:
        return "%.3g s" % value
    if value < 10:
        return "%.3f s" % value
    return "%.1f s" % value


def _fmt_speedup(value):
    value = _number(value)
    return ("%.3f×" % value) if value is not None else "—"


def _fmt_gap(value):
    value = _number(value)
    return ("%.4g" % value) if value is not None else "—"


def _median(values):
    values = sorted(
        value for value in (_number(item) for item in values)
        if value is not None)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def _geometric_mean(values):
    values = [
        value for value in (_positive(item) for item in values)
        if value is not None
    ]
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _md(value):
    return _text(value).replace("|", "\\|").replace("\r", " ").replace(
        "\n", " ")


def _relative_link(target, report_dir):
    try:
        value = os.path.relpath(str(target), str(report_dir))
    except ValueError:
        value = str(target)
    return value.replace("\\", "/")


def _status_cell(summary):
    return "%s / %s" % (
        _md(summary.get("ilp_legosim_validation") or "N/A"),
        _md(summary.get("matched_random_legosim_validation") or "N/A"))


def _build_report(experiment_dir, output_dir, summaries, raw_json, warnings,
                  csv_path, json_path, svg_paths):
    experiment = raw_json.get("experiment", {})
    experiment_status = _text(raw_json.get("status")) or "unknown"
    valid_ilp = sum(
        1 for item in summaries
        if _positive(item.get("ilp_legosim_makespan_cycles")) is not None)
    valid_matched = sum(
        1 for item in summaries
        if _positive(item.get(
            "matched_random_legosim_makespan_cycles")) is not None)
    speedups = [
        item.get("measured_speedup_matched_over_ilp")
        for item in summaries
        if _positive(item.get(
            "measured_speedup_matched_over_ilp")) is not None
    ]
    first_feasible = [
        item.get("ilp_time_to_first_feasible_s") for item in summaries]
    primary_times = [item.get("ilp_primary_wall_s") for item in summaries]
    ilp_search = [
        item.get("ilp_orchestrator_search_wall_s") for item in summaries]
    matched_search = [
        item.get("matched_random_search_wall_s") for item in summaries]

    lines = [
        "# Mesh 预算约束布局实验汇总",
        "",
        "> 本报告由 `report_budget_mesh_experiment.py` 从原始实验记录自动生成；"
        "缺失项不会被插值。生成时间：`%s`。" % _utc_now(),
        "",
        "## 结论摘要",
        "",
        "- 实验状态：`%s`；已汇总 %d 个预算 case。" % (
            _md(experiment_status), len(summaries)),
        "- 通过 LEGOSim 完整性验证的 ILP / matched-random 布局分别为 "
        "%d / %d；两者均有效、可计算真实加速比的配对为 %d。" % (
            valid_ilp, valid_matched, len(speedups)),
    ]
    if speedups:
        lines.extend([
            "- 真实 LEGOSim 加速比定义为 "
            "`matched-random makespan / ILP makespan`（大于 1 表示 ILP "
            "更快）：几何平均 %s，范围 %s–%s。" % (
                _fmt_speedup(_geometric_mean(speedups)),
                _fmt_speedup(min(speedups)),
                _fmt_speedup(max(speedups))),
        ])
    else:
        lines.extend([
            "- 当前没有同时满足 `validation=PASS` 的 ILP 与 "
            "matched-random 配对，因此**不报告真实加速比**。",
        ])
    lines.extend([
        "- ILP 首个可行解时间中位数：%s；主优化阶段耗时中位数：%s；"
        "ILP 编排端到端搜索耗时中位数：%s；matched-random 搜索耗时"
        "中位数：%s。" % (
            _fmt_seconds(_median(first_feasible)),
            _fmt_seconds(_median(primary_times)),
            _fmt_seconds(_median(ilp_search)),
            _fmt_seconds(_median(matched_search))),
        "",
        "这里的“真实”仅指经验证的 LEGOSim 同步 makespan；预测 makespan "
        "单独保留用于模型诊断，不参与真实加速比统计。",
        "",
        "## 实验配置与数据来源",
        "",
        "- 实验目录：`%s`" % _md(str(experiment_dir)),
        "- 工作负载：`%s`；拓扑：`%s`；网格：`%s×%s×%s`。" % (
            _md(experiment.get("workload") or
                (summaries[0].get("workload") if summaries else "N/A")),
            _md(experiment.get("topology") or
                (summaries[0].get("topology") if summaries else "mesh")),
            _fmt_number((experiment.get("grid") or {}).get(
                "L", summaries[0].get("mesh_L") if summaries else None)),
            _fmt_number((experiment.get("grid") or {}).get(
                "X", summaries[0].get("mesh_X") if summaries else None)),
            _fmt_number((experiment.get("grid") or {}).get(
                "Y", summaries[0].get("mesh_Y") if summaries else None))),
        "- 原始表：`%s`；原始元数据：`%s`。" % (
            _md(_relative_link(csv_path, output_dir)),
            _md(_relative_link(json_path, output_dir))),
        "- 论文表：[%s](%s)。" % (
            "paper_summary.csv",
            _relative_link(output_dir / "paper_summary.csv", output_dir)),
    ])
    if svg_paths:
        lines.append("- 图：%s。" % "；".join(
            "[%s](%s)" % (path.name, _relative_link(path, output_dir))
            for path in svg_paths))

    lines.extend([
        "",
        "## 预算扫描结果",
        "",
        "表内功耗预算、峰值功耗和余量均属于**芯粒库预算模型**。"
        "`验证` 一栏为 `ILP / matched-random` 的 LEGOSim 验证状态。",
    ])

    for dimension, title, unit in (
            ("area", "面积预算扫描", "mm²"),
            ("power", "功耗预算扫描", "W")):
        selected = [
            item for item in summaries
            if item.get("sweep_dimension") == dimension]
        lines.extend(["", "### %s" % title, ""])
        if not selected:
            lines.append("当前部分结果中尚无该扫描维度。")
            continue
        lines.extend([
            "| 预算 | A 上限 | P 上限* | ILP 芯粒清单 | "
            "matched-random 芯粒清单 | ILP LEGOSim cycles | "
            "matched cycles | 真实加速比 | ILP 首可行 / 主求解 / 搜索 | "
            "matched 搜索 | 验证 |",
            "|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|",
        ])
        for item in selected:
            lines.append(
                "| %s %s | %s | %s | %s | %s | %s | %s | %s | "
                "%s / %s / %s | %s | %s |" % (
                    _fmt_number(item.get("budget_value")), unit,
                    _fmt_number(item.get("area_budget_mm2_model")),
                    _fmt_number(item.get("power_budget_W_model")),
                    _md(item.get("ilp_chiplet_bom") or "—"),
                    _md(item.get("matched_random_chiplet_bom") or "—"),
                    _fmt_cycles(item.get(
                        "ilp_legosim_makespan_cycles")),
                    _fmt_cycles(item.get(
                        "matched_random_legosim_makespan_cycles")),
                    _fmt_speedup(item.get(
                        "measured_speedup_matched_over_ilp")),
                    _fmt_seconds(item.get(
                        "ilp_time_to_first_feasible_s")),
                    _fmt_seconds(item.get("ilp_primary_wall_s")),
                    _fmt_seconds(item.get(
                        "ilp_orchestrator_search_wall_s")),
                    _fmt_seconds(item.get(
                        "matched_random_search_wall_s")),
                    _status_cell(item)))

    lines.extend([
        "",
        "## 预算与验证审计",
        "",
        "| case | ILP 状态 | matched 状态 | 全局预算检查 "
        "(ILP / matched) | ILP 面积 / 上限 | ILP 模型峰值功耗 / 上限 | "
        "BOM 一致 | MIP gap |",
        "|---|---|---|---|---:|---:|---|---:|",
    ])
    if summaries:
        for item in summaries:
            bom_match = item.get("chiplet_bom_match")
            lines.append(
                "| %s | %s | %s | %s / %s | %s / %s | %s / %s | "
                "%s | %s |" % (
                    _md(item.get("case_id")),
                    _md(item.get("ilp_status")),
                    _md(item.get("matched_random_status")),
                    _md(item.get("ilp_budget_check")),
                    _md(item.get("matched_random_budget_check")),
                    _fmt_number(item.get("ilp_total_area_mm2")),
                    _fmt_number(item.get("area_budget_mm2_model")),
                    _fmt_number(item.get("ilp_peak_power_W_model")),
                    _fmt_number(item.get("power_budget_W_model")),
                    ("PASS" if bom_match is True else
                     "FAIL" if bom_match is False else "N/A"),
                    _fmt_gap(item.get("ilp_mip_gap"))))
    else:
        lines.append("| — | — | — | — | — | — | — | — |")

    noc_rows = [
        item for item in summaries
        if (item.get("ilp_popnet_total_power_NoC_only") is not None or
            item.get(
                "matched_random_popnet_total_power_NoC_only") is not None)]
    if noc_rows:
        lines.extend([
            "",
            "## PopNet 网络观测量（仅 NoC）",
            "",
            "这些值仅用于观察网络行为，不代表芯粒封装总功耗。",
            "",
            "| case | ILP PopNet total power (NoC-only) | "
            "matched NoC-only | ILP avg delay | matched avg delay |",
            "|---|---:|---:|---:|---:|",
        ])
        for item in noc_rows:
            lines.append("| %s | %s | %s | %s | %s |" % (
                _md(item.get("case_id")),
                _fmt_number(item.get(
                    "ilp_popnet_total_power_NoC_only"), 6),
                _fmt_number(item.get(
                    "matched_random_popnet_total_power_NoC_only"), 6),
                _fmt_number(item.get("ilp_popnet_average_delay"), 3),
                _fmt_number(item.get(
                    "matched_random_popnet_average_delay"), 3)))

    lines.extend([
        "",
        "## 指标口径（论文中应保留）",
        "",
        "1. `measured_*` / 图中的 real makespan 来自 LEGOSim 的同步端到端"
        "执行结果；只有 `legosim_validation=PASS` 的正值才进入加速比。",
        "2. `peak_power_W`、`p_max_W` 和 `p_col_W` 是芯粒库中的预算模型值，"
        "**不是 LEGOSim 测得的封装功耗**，不能写成“仿真测量总功耗”。",
        "3. `popnet_total_power`（CSV 中明确标为 `NoC_only`）仅覆盖 PopNet "
        "网络部分，不是封装/系统总功耗，也不能与芯粒库峰值功耗预算直接作"
        "等价比较。",
        "4. 搜索时间比使用两种方法各自的编排端到端搜索 wall time；"
        "`primary_wall_s` 和 `time_to_first_feasible_s` 是 ILP 求解器内部"
        "阶段指标，单独列出以展示快速找到可行/最优布局的过程。",
        "5. matched-random 保持芯粒物料清单和预算不变；只有 "
        "`chiplet_bom_match=PASS` 的行适合作为布局/专家映射对照。",
    ])

    if warnings or experiment_status not in ("complete", "completed"):
        lines.extend(["", "## 部分结果与告警", ""])
        if experiment_status not in ("complete", "completed"):
            lines.append(
                "- 原始 JSON 状态为 `%s`；本报告是当前快照，后续完整运行"
                "可能增加或替换结果。" % _md(experiment_status))
        for warning in warnings:
            lines.append("- %s" % _md(warning))
        failed = [
            item for item in summaries
            if (item.get("ilp_status") not in ("ok", "missing") or
                item.get("matched_random_status") not in ("ok", "missing"))
        ]
        if failed:
            lines.append("- 非成功 case：%s。" % ", ".join(
                "`%s`" % _md(item.get("case_id")) for item in failed))

    errors = []
    for item in summaries:
        for method, key in (
                ("ILP", "ilp_error"),
                ("matched-random", "matched_random_error")):
            value = _text(item.get(key))
            if value:
                errors.append((item.get("case_id"), method, value))
    if errors:
        lines.extend(["", "### 失败详情（截断显示）", ""])
        for case_id, method, value in errors:
            compact = " ".join(value.split())
            if len(compact) > 300:
                compact = compact[:297] + "..."
            lines.append("- `%s` / %s：`%s`" % (
                _md(case_id), method, _md(compact)))

    lines.append("")
    return "\n".join(lines)


def _svg_escape(value):
    return html.escape(_text(value), quote=True)


def _svg_text(parts, x, y, text, size=13, color=TEXT_COLOR,
              anchor="start", weight="400", extra=""):
    parts.append(
        '<text x="%.2f" y="%.2f" font-size="%s" fill="%s" '
        'text-anchor="%s" font-weight="%s" %s>%s</text>' % (
            x, y, size, color, anchor, weight, extra,
            _svg_escape(text)))


def _svg_line(parts, x1, y1, x2, y2, color=GRID_COLOR, width=1,
              dash=None):
    dash_attr = (' stroke-dasharray="%s"' % dash) if dash else ""
    parts.append(
        '<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" '
        'stroke="%s" stroke-width="%s"%s />' % (
            x1, y1, x2, y2, color, width, dash_attr))


def _axis_label(value):
    value = _number(value)
    if value is None:
        return "N/A"
    absolute = abs(value)
    if absolute >= 1e9:
        return "%.3gG" % (value / 1e9)
    if absolute >= 1e6:
        return "%.3gM" % (value / 1e6)
    if absolute >= 1e3:
        return "%.3gk" % (value / 1e3)
    if absolute >= 100:
        return "%.4g" % value
    if absolute >= 1:
        return "%.3g" % value
    return "%.2g" % value


def _linear_range(values, include_one=False):
    values = [
        value for value in (_number(item) for item in values)
        if value is not None]
    if include_one:
        values.append(1.0)
    if not values:
        return None
    low = min(values)
    high = max(values)
    if abs(high - low) <= max(1e-12, abs(high) * 1e-10):
        span = max(abs(high) * 0.08, 0.05)
        low -= span
        high += span
    else:
        span = high - low
        low -= span * 0.12
        high += span * 0.12
    if low < 0 and min(values) >= 0:
        low = 0.0
    return low, high


def _x_positions(items, x0, x1):
    if not items:
        return []
    if len(items) == 1:
        return [(x0 + x1) / 2.0]
    step = (x1 - x0) / (len(items) - 1)
    return [x0 + index * step for index in range(len(items))]


def _draw_panel_frame(parts, panel, title, subtitle=""):
    x, y, width, height = panel
    parts.append(
        '<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" rx="8" '
        'fill="#FBFCFE" stroke="#C9D3DD" />' % (
            x, y, width, height))
    _svg_text(parts, x + 18, y + 25, title, size=16, weight="600")
    if subtitle:
        _svg_text(parts, x + width - 18, y + 24, subtitle, size=11,
                  color=MUTED_COLOR, anchor="end")


def _draw_linear_series_panel(parts, panel, items, series, title,
                              y_note="", include_one=False,
                              empty_message="No available values"):
    _draw_panel_frame(parts, panel, title, y_note)
    x, y, width, height = panel
    plot_left = x + 78
    plot_right = x + width - 28
    plot_top = y + 48
    plot_bottom = y + height - 50
    x_values = _x_positions(items, plot_left + 12, plot_right - 12)
    all_values = []
    for spec in series:
        all_values.extend(
            _number(item.get(spec["key"])) for item in items)
    value_range = _linear_range(all_values, include_one=include_one)

    _svg_line(parts, plot_left, plot_bottom, plot_right, plot_bottom,
              color="#8996A3")
    _svg_line(parts, plot_left, plot_top, plot_left, plot_bottom,
              color="#8996A3")
    if value_range is None:
        _svg_text(parts, (plot_left + plot_right) / 2,
                  (plot_top + plot_bottom) / 2, empty_message,
                  size=15, color=MUTED_COLOR, anchor="middle")
    else:
        low, high = value_range
        for tick in range(5):
            fraction = tick / 4.0
            value = low + (high - low) * fraction
            tick_y = plot_bottom - fraction * (plot_bottom - plot_top)
            _svg_line(parts, plot_left, tick_y, plot_right, tick_y)
            _svg_text(parts, plot_left - 10, tick_y + 4,
                      _axis_label(value), size=11, color=MUTED_COLOR,
                      anchor="end")
        if include_one and low <= 1.0 <= high:
            one_y = plot_bottom - (
                (1.0 - low) / (high - low) *
                (plot_bottom - plot_top))
            _svg_line(parts, plot_left, one_y, plot_right, one_y,
                      color="#777777", width=1.2, dash="5 4")
            _svg_text(parts, plot_right - 2, one_y - 5, "1.0 baseline",
                      size=10, color=MUTED_COLOR, anchor="end")

        for spec_index, spec in enumerate(series):
            points = []
            for item_index, item in enumerate(items):
                value = _number(item.get(spec["key"]))
                if value is None:
                    continue
                point_x = x_values[item_index]
                point_y = plot_bottom - (
                    (value - low) / (high - low) *
                    (plot_bottom - plot_top))
                points.append((point_x, point_y, value, item_index))
            if len(points) > 1:
                parts.append(
                    '<polyline points="%s" fill="none" stroke="%s" '
                    'stroke-width="2.5" stroke-linejoin="round" />' % (
                        " ".join("%.2f,%.2f" % (px, py)
                                 for px, py, _, _ in points),
                        spec["color"]))
            for point_x, point_y, value, _ in points:
                parts.append(
                    '<circle cx="%.2f" cy="%.2f" r="5" fill="%s" '
                    'stroke="#FFFFFF" stroke-width="1.5"><title>%s: '
                    '%s</title></circle>' % (
                        point_x, point_y, spec["color"],
                        _svg_escape(spec["label"]),
                        _svg_escape(_axis_label(value))))
                label_y = point_y - 9 if spec_index == 0 else point_y + 18
                _svg_text(parts, point_x, label_y, _axis_label(value),
                          size=10, color=spec["color"], anchor="middle",
                          weight="600")

    for index, item in enumerate(items):
        _svg_line(parts, x_values[index], plot_bottom,
                  x_values[index], plot_bottom + 5, color="#8996A3")
        _svg_text(parts, x_values[index], plot_bottom + 20,
                  _axis_label(item.get("budget_value")),
                  size=11, color=MUTED_COLOR, anchor="middle")
    unit = "mm²" if (
        items and items[0].get("sweep_dimension") == "area") else "W"
    _svg_text(parts, (plot_left + plot_right) / 2, y + height - 10,
              "Budget (%s)" % unit, size=11, color=MUTED_COLOR,
              anchor="middle")

    legend_x = plot_left + 8
    for spec in series:
        parts.append(
            '<circle cx="%.2f" cy="%.2f" r="4" fill="%s" />' % (
                legend_x, y + 40, spec["color"]))
        _svg_text(parts, legend_x + 8, y + 44, spec["label"], size=11,
                  color=MUTED_COLOR)
        legend_x += 135


def _draw_log_time_panel(parts, panel, items):
    title = "ILP time to feasible layout and certified primary optimum"
    _draw_panel_frame(parts, panel, title, "seconds; logarithmic y-axis")
    x, y, width, height = panel
    plot_left = x + 78
    plot_right = x + width - 28
    plot_top = y + 48
    plot_bottom = y + height - 50
    x_values = _x_positions(items, plot_left + 12, plot_right - 12)
    series = [
        {
            "key": "ilp_time_to_first_feasible_s",
            "label": "First feasible",
            "color": SPEEDUP_COLOR,
            "label_dy": 18,
        },
        {
            "key": "ilp_primary_wall_s",
            "label": "Primary solve / proof",
            "color": ILP_COLOR,
            "label_dy": -9,
        },
        {
            "key": "ilp_orchestrator_search_wall_s",
            "label": "End-to-end incl. tie-break",
            "color": PROOF_COLOR,
            "label_dy": 18,
        },
    ]
    values = []
    for spec in series:
        values.extend(
            _positive(item.get(spec["key"])) for item in items)
    values = [value for value in values if value is not None]
    _svg_line(parts, plot_left, plot_bottom, plot_right, plot_bottom,
              color="#8996A3")
    _svg_line(parts, plot_left, plot_top, plot_left, plot_bottom,
              color="#8996A3")
    if not values:
        _svg_text(parts, (plot_left + plot_right) / 2,
                  (plot_top + plot_bottom) / 2,
                  "No search-time values in current partial results",
                  size=15, color=MUTED_COLOR, anchor="middle")
    else:
        log_low = math.floor(math.log10(min(values)))
        log_high = math.ceil(math.log10(max(values)))
        if log_high <= log_low:
            log_high = log_low + 1
        for exponent in range(int(log_low), int(log_high) + 1):
            fraction = ((exponent - log_low) /
                        float(log_high - log_low))
            tick_y = plot_bottom - fraction * (
                plot_bottom - plot_top)
            _svg_line(parts, plot_left, tick_y, plot_right, tick_y)
            _svg_text(parts, plot_left - 10, tick_y + 4,
                      _axis_label(10 ** exponent), size=11,
                      color=MUTED_COLOR, anchor="end")
        for spec in series:
            points = []
            for item_index, item in enumerate(items):
                value = _positive(item.get(spec["key"]))
                if value is None:
                    continue
                point_x = x_values[item_index]
                fraction = ((math.log10(value) - log_low) /
                            float(log_high - log_low))
                point_y = plot_bottom - fraction * (
                    plot_bottom - plot_top)
                points.append((point_x, point_y, value))
            if len(points) > 1:
                parts.append(
                    '<polyline points="%s" fill="none" stroke="%s" '
                    'stroke-width="2.5" stroke-linejoin="round" />' % (
                        " ".join("%.2f,%.2f" % (px, py)
                                 for px, py, _ in points),
                        spec["color"]))
            for point_x, point_y, value in points:
                parts.append(
                    '<circle cx="%.2f" cy="%.2f" r="5" fill="%s" '
                    'stroke="#FFFFFF" stroke-width="1.5"><title>%s: '
                    '%s s</title></circle>' % (
                        point_x, point_y, spec["color"],
                        _svg_escape(spec["label"]),
                        _svg_escape(_axis_label(value))))
                label_y = point_y + spec.get("label_dy", -9)
                _svg_text(parts, point_x, label_y,
                          "%ss" % _axis_label(value), size=10,
                          color=spec["color"], anchor="middle",
                          weight="600")

    for index, item in enumerate(items):
        _svg_line(parts, x_values[index], plot_bottom,
                  x_values[index], plot_bottom + 5, color="#8996A3")
        _svg_text(parts, x_values[index], plot_bottom + 20,
                  _axis_label(item.get("budget_value")),
                  size=11, color=MUTED_COLOR, anchor="middle")
    unit = "mm²" if (
        items and items[0].get("sweep_dimension") == "area") else "W"
    _svg_text(parts, (plot_left + plot_right) / 2, y + height - 10,
              "Budget (%s)" % unit, size=11, color=MUTED_COLOR,
              anchor="middle")
    legend_x = plot_left + 8
    for spec in series:
        parts.append(
            '<circle cx="%.2f" cy="%.2f" r="4" fill="%s" />' % (
                legend_x, y + 40, spec["color"]))
        _svg_text(parts, legend_x + 8, y + 44, spec["label"], size=11,
                  color=MUTED_COLOR)
        legend_x += 245


def _build_svg(dimension, summaries, raw_json):
    items = [
        item for item in summaries
        if item.get("sweep_dimension") == dimension]
    items.sort(key=lambda item: (
        item.get("budget_value")
        if item.get("budget_value") is not None else float("inf"),
        item.get("case_id", "")))
    title_dimension = "Area-budget sweep" if dimension == "area" else (
        "Power-budget sweep")
    experiment = raw_json.get("experiment", {})
    experiment_id = _text(experiment.get("experiment_id"))
    width = 1240
    height = 910
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
        'viewBox="0 0 %d %d" role="img" aria-labelledby="title desc">' % (
            width, height, width, height),
        '<title id="title">%s</title>' % _svg_escape(
            "%s: ILP vs matched-random" % title_dimension),
        '<desc id="desc">%s</desc>' % _svg_escape(
            "Three panels compare validated real LEGOSim makespan, "
            "matched-over-ILP speedup, and layout search time. Missing or "
            "unvalidated simulator results are not imputed."),
        '<rect width="100%" height="100%" fill="#F2F5F8" />',
        '<g font-family="Inter, Segoe UI, Arial, sans-serif">',
    ]
    _svg_text(parts, 32, 38,
              "%s — ILP vs resource-matched random layout" %
              title_dimension, size=23, weight="700")
    subtitle = (
        "Experiment: %s · Mesh L=1 · real makespan requires validation=PASS"
        % (experiment_id or "unknown"))
    _svg_text(parts, 32, 62, subtitle, size=12, color=MUTED_COLOR)

    if not items:
        parts.append(
            '<rect x="30" y="92" width="1180" height="760" rx="10" '
            'fill="#FBFCFE" stroke="#C9D3DD" />')
        _svg_text(parts, width / 2, height / 2,
                  "No %s cases in current partial results" % dimension,
                  size=20, color=MUTED_COLOR, anchor="middle")
    else:
        panels = [
            (30, 86, 1180, 245),
            (30, 350, 1180, 245),
            (30, 614, 1180, 245),
        ]
        _draw_linear_series_panel(
            parts, panels[0], items,
            [
                {
                    "key": "ilp_legosim_makespan_cycles",
                    "label": "ILP",
                    "color": ILP_COLOR,
                },
                {
                    "key": "matched_random_legosim_makespan_cycles",
                    "label": "Matched random",
                    "color": MATCHED_COLOR,
                },
            ],
            "Validated real LEGOSim makespan",
            "cycles; y-range follows observed values",
            empty_message=(
                "No validation=PASS LEGOSim makespan in current results"))
        _draw_linear_series_panel(
            parts, panels[1], items,
            [{
                "key": "measured_speedup_matched_over_ilp",
                "label": "Matched / ILP",
                "color": SPEEDUP_COLOR,
            }],
            "Measured speedup (matched-random / ILP)",
            "> 1 means ILP is faster", include_one=True,
            empty_message="No validated ILP/matched pair")
        _draw_log_time_panel(parts, panels[2], items)

    _svg_text(
        parts, 32, 892,
        "Power-budget fields are chiplet-library model values. "
        "PopNet power, when reported, is NoC-only.",
        size=11, color=MUTED_COLOR)
    parts.extend(["</g>", "</svg>", ""])
    return "\n".join(parts)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a run_budget_mesh_experiment.py directory into a "
            "paper CSV, Markdown report, and dependency-free SVG plots."))
    parser.add_argument(
        "experiment_dir",
        help="directory containing raw_results.csv/raw_results.json")
    parser.add_argument(
        "--output-dir",
        help="output directory (default: experiment_dir)")
    parser.add_argument(
        "--no-svg", action="store_true",
        help="write CSV/Markdown only")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    if not experiment_dir.is_dir():
        print("ERROR: experiment directory not found: %s" % experiment_dir,
              file=sys.stderr)
        return 2
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else experiment_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows, raw_json, warnings, csv_path, json_path = _load_inputs(
            experiment_dir)
    except (OSError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    summaries = _pair_cases(rows, raw_json)
    summary_path = output_dir / "paper_summary.csv"
    _atomic_csv(summary_path, summaries)

    svg_paths = []
    if not args.no_svg:
        for dimension in ("area", "power"):
            path = output_dir / ("%s_budget_results.svg" % dimension)
            _atomic_text(path, _build_svg(dimension, summaries, raw_json))
            svg_paths.append(path)

    report_path = output_dir / "REPORT.md"
    report = _build_report(
        experiment_dir, output_dir, summaries, raw_json, warnings,
        csv_path, json_path, svg_paths)
    _atomic_text(report_path, report)

    paired = sum(
        1 for item in summaries
        if item.get("measured_speedup_matched_over_ilp") is not None)
    print("Wrote %s" % summary_path)
    print("Wrote %s" % report_path)
    for path in svg_paths:
        print("Wrote %s" % path)
    print("Summary: cases=%d, validated_pairs=%d, warnings=%d" % (
        len(summaries), paired, len(warnings)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
