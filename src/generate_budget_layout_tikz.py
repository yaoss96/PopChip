#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the three-panel budget-layout TikZ artifact.

Coordinates and task assignments come from fresh canonical ILP-Perf solution
JSON files.  Measured cycles come from a unified LEGOSim CSV row whose
solution SHA-256 must match the JSON being drawn.  The generator fails closed:
all three inputs and all validation/provenance checks pass before an output is
atomically replaced.
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
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_CSV = "out/unified_legosim_data/legosim_metrics.csv"
DEFAULT_OUTPUT = "paper_date/budget-layouts-tikz.tex"

CASES = (
    {"case_id": "area_970", "panel": "a", "area_budget": 970,
     "gpu_l": 0, "gpu_s": 8},
    {"case_id": "area_1550", "panel": "b", "area_budget": 1550,
     "gpu_l": 2, "gpu_s": 6},
    {"case_id": "area_2130", "panel": "c", "area_budget": 2130,
     "gpu_l": 4, "gpu_s": 4},
)

TYPE_SPECS = {
    "CPU": {"width": 1, "height": 1, "area": 90.0, "peak_W": 65.0,
            "macro": "budgetcpuchip", "short": "CPU"},
    "GPU-L": {"width": 2, "height": 2, "area": 400.0, "peak_W": 300.0,
              "macro": "budgetlgpu", "short": "L"},
    "GPU-S": {"width": 1, "height": 1, "area": 110.0, "peak_W": 90.0,
              "macro": "budgetsgpu", "short": "S"},
}

TASK_ORDER = (
    "e(0)", "e(1)", "e(2)", "e(3)", "e(4)", "e(5)", "e(6)", "e(7)",
    "t_attn", "t_gate", "t_dispatch", "t_combine", "t_norm",
)
TASK_SHORT = {
    "e(0)": "e0", "e(1)": "e1", "e(2)": "e2", "e(3)": "e3",
    "e(4)": "e4", "e(5)": "e5", "e(6)": "e6", "e(7)": "e7",
    "t_attn": "At", "t_gate": "Gt", "t_dispatch": "Ds",
    "t_combine": "Cb", "t_norm": "Nm",
}
REQUIRED_CSV_COLUMNS = {
    "suite", "case_id", "sweep_dimension", "budget_value",
    "area_budget_mm2", "power_budget_W", "experts", "mesh", "method",
    "cpu_count", "gpu_l_count", "gpu_s_count", "synchronized_cycles",
    "legosim_validation", "audit_status", "solution_sha256", "errors",
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate strict three-panel budget-layout TikZ")
    parser.add_argument(
        "--canonical-root", required=True,
        help="root containing solutions/budget/<case>/ilp_perf.json")
    parser.add_argument(
        "--metrics-csv", "--input", dest="metrics_csv",
        default=DEFAULT_METRICS_CSV,
        help="final unified LEGOSim metrics CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only", action="store_true",
        help="validate and render in memory without replacing the output")
    return parser.parse_args(argv)


def _resolve(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _read_bytes(path, label):
    if not path.is_file():
        raise ValueError("%s not found: %s" % (label, path))
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError("cannot read %s %s: %s" % (label, path, exc))


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _strict_int(value, label, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s is not an integer: %r" % (label, value))
    if minimum is not None and value < minimum:
        raise ValueError("%s is below %d: %r" % (label, minimum, value))
    return value


def _csv_int(value, label, minimum=None):
    text = str(value or "").strip()
    try:
        number = int(text)
    except ValueError:
        raise ValueError("%s is not an integer: %r" % (label, value))
    if str(number) != text:
        raise ValueError("%s is not a canonical integer: %r" % (label, value))
    if minimum is not None and number < minimum:
        raise ValueError("%s is below %d: %r" % (label, minimum, value))
    return number


def _finite_float(value, label, positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s is not numeric: %r" % (label, value))
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError("%s is not finite%s: %r" %
                         (label, " and positive" if positive else "", value))
    return number


def _require_close(actual, expected, label):
    number = _finite_float(actual, label)
    if not math.isclose(number, float(expected), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("%s=%r, expected %r" % (label, number, expected))
    return number


def _solution_path(canonical_root, case_id):
    return (canonical_root / "solutions" / "budget" / case_id /
            "ilp_perf.json")


def _load_solution(canonical_root, config):
    case_id = config["case_id"]
    path = _solution_path(canonical_root, case_id)
    raw = _read_bytes(path, "%s canonical solution" % case_id)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("invalid JSON in %s: %s" % (path, exc))
    if not isinstance(value, dict):
        raise ValueError("%s solution is not a JSON object" % case_id)
    if value.get("optimization_mode") != "ilp-perf":
        raise ValueError("%s is not an ilp-perf solution" % case_id)
    if value.get("cpu_anchor_policy") != "free":
        raise ValueError("%s does not use free CPU placement" % case_id)

    grid = value.get("grid")
    if grid != {"L": 1, "X": 6, "Y": 6}:
        raise ValueError("%s grid is not exactly L=1, X=6, Y=6: %r" %
                         (case_id, grid))
    budgets = value.get("budgets")
    if not isinstance(budgets, dict):
        raise ValueError("%s lacks budgets" % case_id)
    _require_close(budgets.get("a_max"), config["area_budget"],
                   "%s a_max" % case_id)
    _require_close(budgets.get("p_max"), 1625.0, "%s p_max" % case_id)

    sites = value.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("%s lacks solution sites" % case_id)
    occupancy = {}
    seen_tasks = {}
    counts = {name: 0 for name in TYPE_SPECS}
    normalized_sites = []
    task_rank = {task: index for index, task in enumerate(TASK_ORDER)}
    for index, site in enumerate(sites):
        label = "%s site %d" % (case_id, index)
        if not isinstance(site, dict):
            raise ValueError("%s is not an object" % label)
        chip_type = site.get("type")
        if chip_type not in TYPE_SPECS:
            raise ValueError("%s has unsupported type %r" % (label, chip_type))
        spec = TYPE_SPECS[chip_type]
        layer = _strict_int(site.get("l"), "%s l" % label, 0)
        row = _strict_int(site.get("i"), "%s i" % label, 0)
        column = _strict_int(site.get("j"), "%s j" % label, 0)
        if layer != 0:
            raise ValueError("%s is outside the sole Mesh layer" % label)
        if (row + spec["height"] > 6 or
                column + spec["width"] > 6):
            raise ValueError("%s footprint leaves the 6x6 Mesh" % label)

        tasks = site.get("tasks")
        if (not isinstance(tasks, list) or not tasks or
                any(not isinstance(task, str) or not task for task in tasks)):
            raise ValueError("%s has an invalid/empty task list" % label)
        if len(set(tasks)) != len(tasks):
            raise ValueError("%s repeats a task" % label)
        for task in tasks:
            if task not in TASK_SHORT:
                raise ValueError("%s contains unexpected task %r" %
                                 (label, task))
            if task in seen_tasks:
                raise ValueError("task %s appears at multiple sites" % task)
            seen_tasks[task] = (layer, row, column)

        for occupied_row in range(row, row + spec["height"]):
            for occupied_column in range(column, column + spec["width"]):
                cell = (layer, occupied_row, occupied_column)
                if cell in occupancy:
                    raise ValueError("%s overlaps %s at %r" %
                                     (label, occupancy[cell], cell))
                occupancy[cell] = label
        counts[chip_type] += 1
        normalized_sites.append({
            "type": chip_type, "l": layer, "i": row, "j": column,
            "x": column, "y": 6 - row - spec["height"],
            "width": spec["width"], "height": spec["height"],
            "tasks": tuple(sorted(tasks, key=lambda task: task_rank[task])),
        })

    missing_tasks = sorted(set(TASK_ORDER) - set(seen_tasks))
    extra_tasks = sorted(set(seen_tasks) - set(TASK_ORDER))
    if missing_tasks or extra_tasks:
        raise ValueError("%s task set mismatch; missing=%r extra=%r" %
                         (case_id, missing_tasks, extra_tasks))
    if (counts["CPU"], counts["GPU-L"], counts["GPU-S"]) != \
            (1, config["gpu_l"], config["gpu_s"]):
        raise ValueError("%s BOM mismatch: %r" % (case_id, counts))

    task_site = value.get("task_site")
    if not isinstance(task_site, dict) or set(task_site) != set(TASK_ORDER):
        raise ValueError("%s task_site does not match the task set" % case_id)
    for task, expected_anchor in seen_tasks.items():
        anchor = task_site.get(task)
        if (not isinstance(anchor, list) or len(anchor) != 3 or
                any(isinstance(item, bool) or not isinstance(item, int)
                    for item in anchor) or tuple(anchor) != expected_anchor):
            raise ValueError("%s task_site[%s] disagrees with sites" %
                             (case_id, task))

    derived_area = sum(TYPE_SPECS[name]["area"] * count
                       for name, count in counts.items())
    derived_peak = sum(TYPE_SPECS[name]["peak_W"] * count
                       for name, count in counts.items())
    used_area = _require_close(value.get("total_area_mm2"), derived_area,
                               "%s total_area_mm2" % case_id)
    peak_power = _require_close(value.get("peak_power_W"), derived_peak,
                                "%s peak_power_W" % case_id)
    _require_close(used_area, config["area_budget"],
                   "%s used area" % case_id)

    normalized_sites.sort(key=lambda site: (site["i"], site["j"],
                                             site["type"]))
    return {
        "case_id": case_id,
        "config": config,
        "path": path,
        "sha256": _digest(raw),
        "sites": normalized_sites,
        "counts": counts,
        "used_area": used_area,
        "peak_power": peak_power,
    }


def _load_metrics(path, solutions):
    raw = _read_bytes(path, "unified metrics CSV")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid UTF-8 in metrics CSV %s: %s" % (path, exc))
    reader = csv.DictReader(io.StringIO(text, newline=""))
    missing_columns = sorted(REQUIRED_CSV_COLUMNS - set(reader.fieldnames or []))
    if missing_columns:
        raise ValueError("unified CSV missing columns: %s" %
                         ", ".join(missing_columns))

    wanted = set(solution["case_id"] for solution in solutions)
    selected = {}
    for line_number, row in enumerate(reader, 2):
        case_id = str(row.get("case_id") or "").strip()
        if (str(row.get("suite") or "").strip() != "budget" or
                str(row.get("method") or "").strip() != "ilp_perf" or
                case_id not in wanted):
            continue
        if case_id in selected:
            raise ValueError("duplicate budget/%s/ilp_perf CSV row" % case_id)
        selected[case_id] = (line_number, row)

    missing = sorted(wanted - set(selected))
    if missing:
        raise ValueError("unified CSV lacks required ilp_perf rows: %s" %
                         ", ".join(missing))

    metrics = {}
    for solution in solutions:
        case_id = solution["case_id"]
        config = solution["config"]
        line_number, row = selected[case_id]
        prefix = "CSV line %d (%s)" % (line_number, case_id)
        if str(row.get("legosim_validation") or "").strip() != "PASS":
            raise ValueError("%s legosim_validation is not PASS" % prefix)
        if str(row.get("audit_status") or "").strip() != "PASS":
            raise ValueError("%s audit_status is not PASS" % prefix)
        if str(row.get("errors") or "").strip():
            raise ValueError("%s contains audit errors" % prefix)
        if str(row.get("sweep_dimension") or "").strip() != "area":
            raise ValueError("%s is not an area sweep" % prefix)
        if str(row.get("mesh") or "").strip() != "6x6":
            raise ValueError("%s Mesh is not 6x6" % prefix)
        if _csv_int(row.get("experts"), "%s experts" % prefix, 1) != 8:
            raise ValueError("%s does not contain eight experts" % prefix)
        _require_close(row.get("budget_value"), config["area_budget"],
                       "%s budget_value" % prefix)
        _require_close(row.get("area_budget_mm2"), config["area_budget"],
                       "%s area_budget_mm2" % prefix)
        _require_close(row.get("power_budget_W"), 1625.0,
                       "%s power_budget_W" % prefix)
        csv_bom = (
            _csv_int(row.get("cpu_count"), "%s cpu_count" % prefix, 0),
            _csv_int(row.get("gpu_l_count"), "%s gpu_l_count" % prefix, 0),
            _csv_int(row.get("gpu_s_count"), "%s gpu_s_count" % prefix, 0),
        )
        solution_bom = (solution["counts"]["CPU"],
                        solution["counts"]["GPU-L"],
                        solution["counts"]["GPU-S"])
        if csv_bom != solution_bom:
            raise ValueError("%s BOM %r disagrees with solution %r" %
                             (prefix, csv_bom, solution_bom))
        recorded_sha = str(row.get("solution_sha256") or "").strip().lower()
        if recorded_sha != solution["sha256"]:
            raise ValueError("%s solution_sha256 does not match canonical JSON" %
                             prefix)
        metrics[case_id] = {
            "cycles": _csv_int(row.get("synchronized_cycles"),
                               "%s synchronized_cycles" % prefix, 1),
            "line_number": line_number,
        }
    return metrics, _digest(raw)


def _task_label(site):
    tokens = [TASK_SHORT[task] for task in site["tasks"]]
    chip = TYPE_SPECS[site["type"]]["short"]
    if site["type"] == "CPU":
        lines = [chip]
    else:
        lines = [chip + ":" + tokens.pop(0)]
    while tokens:
        lines.append("+".join(tokens[:2]))
        del tokens[:2]
    return r"\\".join(lines)


def _tex_integer(number):
    value = _finite_float(number, "TeX integer")
    if not value.is_integer():
        raise ValueError("TeX integer is not integral: %r" % number)
    return ("{:,}".format(int(value))).replace(",", "{,}")


def _safe_comment(value):
    return str(value).replace("\r", " ").replace("\n", " ")


def _tex_text(canonical_root, metrics_path, csv_sha256, solutions, metrics):
    lines = [
        "%% Auto-generated by generate_budget_layout_tikz.py; do not edit.",
        "%% Canonical root: %s" % _safe_comment(canonical_root.as_posix()),
        "%% Metrics CSV: %s" % _safe_comment(metrics_path.as_posix()),
        "%% Metrics CSV SHA-256: %s" % csv_sha256,
    ]
    for solution in solutions:
        lines.extend([
            "%% %s solution: %s" %
            (solution["case_id"], _safe_comment(solution["path"].as_posix())),
            "%% %s solution SHA-256: %s" %
            (solution["case_id"], solution["sha256"]),
        ])
    lines.extend([
        "%% Coordinates: source (l,i,j) maps to TikZ (x,y)=(j,6-i-height).",
        "%% Labels: At=t_attn, Gt=t_gate, Ds=t_dispatch, Cb=t_combine,",
        "%%         Nm=t_norm; e0..e7 denote e(0)..e(7).",
        r"\centering",
        r"\begingroup",
        r"\tikzset{",
        r"  budgetgrid/.style={draw=black!28,line width=0.25pt},",
        r"  budgetlarge/.style={draw=blue!65!black,fill=morandiA!62,line width=0.5pt},",
        r"  budgetsmall/.style={draw=green!45!black,fill=morandiC!62,line width=0.5pt},",
        r"  budgetcpu/.style={draw=orange!65!black,fill=morandiD!78,line width=0.5pt},",
        r"  budgetlabel/.style={font=\sffamily\fontsize{3.5}{3.6}\selectfont,",
        r"                     inner sep=0.1pt,align=center}",
        r"}",
        r"\newcommand{\budgetmesh}{%",
        r"  \draw[budgetgrid,step=1] (0,0) grid (6,6);",
        r"  \draw[black!65,line width=0.45pt] (0,0) rectangle (6,6);",
        r"}",
        r"\newcommand{\budgetsgpu}[3]{%",
        r"  \filldraw[budgetsmall] (#1,#2) rectangle ++(1,1);",
        r"  \node[budgetlabel] at (#1+0.5,#2+0.5) {#3};",
        r"}",
        r"\newcommand{\budgetlgpu}[3]{%",
        r"  \filldraw[budgetlarge] (#1,#2) rectangle ++(2,2);",
        r"  \node[budgetlabel] at (#1+1,#2+1) {#3};",
        r"}",
        r"\newcommand{\budgetcpuchip}[3]{%",
        r"  \filldraw[budgetcpu] (#1,#2) rectangle ++(1,1);",
        r"  \node[budgetlabel] at (#1+0.5,#2+0.5) {#3};",
        r"}",
    ])

    for index, solution in enumerate(solutions):
        config = solution["config"]
        case_id = solution["case_id"]
        if index:
            lines.append(r"\hfill")
        lines.extend([
            r"\begin{minipage}[t]{0.318\linewidth}",
            r"\centering",
            r"\begin{tikzpicture}[x=0.40cm,y=0.40cm]",
            r"  \budgetmesh",
        ])
        for site in solution["sites"]:
            tasks = ",".join(site["tasks"])
            lines.append(
                "  %% source (l,i,j)=(%d,%d,%d), type=%s, tasks=[%s]; "
                "TikZ=(%d,%d)" %
                (site["l"], site["i"], site["j"], site["type"], tasks,
                 site["x"], site["y"]))
            macro = TYPE_SPECS[site["type"]]["macro"]
            lines.append(r"  \%s{%d}{%d}{%s}" %
                         (macro, site["x"], site["y"], _task_label(site)))
        lines.extend([
            r"\end{tikzpicture}\\[-0.35mm]",
            (r"{\fontsize{4.8}{5.25}\selectfont"
             r"\textbf{(%s) $A_{\max}=%d\,\mathrm{mm}^2$}\\[-0.1mm]" %
             (config["panel"], config["area_budget"])),
            (r"GPU-L/S $=%d/%d$\\[-0.1mm]" %
             (solution["counts"]["GPU-L"], solution["counts"]["GPU-S"])),
            (r"used %s\,mm$^2$; BOM pk. %s\,W\\[-0.1mm]" %
             (_tex_integer(solution["used_area"]),
              _tex_integer(solution["peak_power"]))),
            (r"LEGOSim: %s measured cycles\par}" %
             _tex_integer(metrics[case_id]["cycles"])),
            r"\end{minipage}",
        ])
    lines.extend([
        r"\par\vspace{0.25mm}",
        (r"{\fontsize{4.8}{5.2}\selectfont At/Ds/Cb denote attention/dispatch/"
         r"combine; Gt/Nm denote gate/norm.}"),
        r"\endgroup",
    ])
    return "\n".join(lines) + "\n"


def _render(canonical_root, metrics_path):
    solutions = [_load_solution(canonical_root, config) for config in CASES]
    metrics, csv_sha256 = _load_metrics(metrics_path, solutions)
    text = _tex_text(canonical_root, metrics_path, csv_sha256,
                     solutions, metrics)
    return text, {
        "metrics_csv_sha256": csv_sha256,
        "solution_sha256": {
            solution["case_id"]: solution["sha256"]
            for solution in solutions
        },
        "cycles": {
            case_id: metrics[case_id]["cycles"] for case_id in metrics
        },
    }


def _atomic_write(destination, text):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".tmp.", dir=str(destination.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    args = _parse_args(argv)
    canonical_root = _resolve(args.canonical_root)
    if not canonical_root.is_dir():
        raise ValueError("canonical root not found: %s" % canonical_root)
    metrics_path = _resolve(args.metrics_csv)
    output_path = _resolve(args.output)
    text, metadata = _render(canonical_root, metrics_path)
    if not args.check_only:
        _atomic_write(output_path, text)
    print(json.dumps({
        "status": "PASS",
        "check_only": bool(args.check_only),
        "canonical_root": str(canonical_root),
        "metrics_csv": str(metrics_path),
        "output": str(output_path),
        "output_sha256": _digest(text.encode("utf-8")),
        "metrics_csv_sha256": metadata["metrics_csv_sha256"],
        "solution_sha256": metadata["solution_sha256"],
        "measured_cycles": metadata["cycles"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        sys.exit(2)
