#!/usr/bin/env python3
"""Build the Table III modeled-active-power rows from simulator outputs.

The reported quantity is deliberately *not* wall-plug or time-window-averaged
package power.  It is the sum of three component-model averages:

  Sniper/McPAT CPU active power
  + AccelWattch average power for each active GPU workload
  + PopNet/ORION average NoC power.

Keeping that definition explicit avoids relabeling the ILP's peak-power budget
or the chiplet-library TDP values as measured run-time power.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


CASE_ROWS = (
    ("Area", "area_970", 970, 1625),
    ("Area", "area_1260", 1260, 1625),
    ("Area", "area_1550", 1550, 1625),
    ("Area", "area_2130", 2130, 1625),
    ("Power", "power_785", 2130, 785),
    ("Power", "power_995", 2130, 995),
    ("Power", "power_1205", 2130, 1205),
    ("Power", "power_1625", 2130, 1625),
)


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_gpu_types(path: Path) -> list[str]:
    gpu_types: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "pre_copy:" not in raw:
            continue
        if "SM7_TITANV" in raw:
            gpu_types.append("SM7_TITANV")
        elif "SM6_TITANX" in raw:
            gpu_types.append("SM6_TITANX")
        else:
            raise ValueError("Unknown GPU configuration in %s: %s" % (path, raw))
    if len(gpu_types) != 8:
        raise ValueError("Expected 8 GPU workers in %s, found %d" % (path, len(gpu_types)))
    return gpu_types


def read_worker_tokens(path: Path) -> list[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    ngpu = None
    matrix_start = None
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("ngpu "):
            ngpu = int(stripped.split()[1])
        if stripped.startswith("round "):
            matrix_start = index + 1
            break
    if ngpu is None or matrix_start is None:
        raise ValueError("Missing ngpu/round header in %s" % path)
    matrix: list[list[int]] = []
    for raw in lines[matrix_start:]:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        row = [int(value) for value in stripped.split()]
        if len(row) != ngpu:
            raise ValueError("Dispatch row width mismatch in %s" % path)
        matrix.append(row)
        if len(matrix) == ngpu:
            break
    if len(matrix) != ngpu:
        raise ValueError("Expected %d dispatch rows in %s" % (ngpu, path))
    return [sum(matrix[source][worker] for source in range(ngpu)) for worker in range(ngpu)]


def read_gpu_power(path: Path) -> dict[tuple[str, int], float]:
    table: dict[tuple[str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["type"], int(row["tokens"]))
            table[key] = float(row["gpu_tot_avg_power_W"])
    return table


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_experiment = (
        repo_root
        / "out"
        / "budget_mesh_experiment"
        / "newrepo_mesh_budget_full_v2_20260731"
    )
    default_gpu_power = repo_root / "out" / "power_model_audit" / "accelwattch_qv100_model_replay.csv"
    default_output = repo_root / "out" / "power_model_audit" / "table3_modeled_active_power.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=default_experiment)
    parser.add_argument("--gpu-power-csv", type=Path, default=default_gpu_power)
    parser.add_argument("--cpu-power-w", type=float, default=13.926945)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    gpu_power = read_gpu_power(args.gpu_power_csv)
    rows: list[dict[str, object]] = []
    for constraint, case_name, area_max, power_max in CASE_ROWS:
        case_dir = args.experiment_root / "cases" / case_name / "ilp"
        summary = read_key_values(case_dir / "legosim_result" / "ilp_e2e_summary.txt")
        if summary.get("validation") != "PASS":
            raise ValueError("Formal run did not pass validation: %s" % case_name)
        types = read_gpu_types(case_dir / "codegen" / "bench.yml")
        tokens = read_worker_tokens(case_dir / "codegen" / "dispatch_matrix.txt")
        gpu_values: list[float] = []
        for gpu_type, worker_tokens in zip(types, tokens):
            key = (gpu_type, worker_tokens)
            if key not in gpu_power:
                raise KeyError("No AccelWattch result for %s tokens=%d" % key)
            gpu_values.append(gpu_power[key])
        gpu_total = sum(gpu_values)
        noc_power = float(summary["popnet_total_power"])
        modeled_total = args.cpu_power_w + gpu_total + noc_power
        large = sum(1 for gpu_type in types if gpu_type == "SM7_TITANV")
        small = sum(1 for gpu_type in types if gpu_type == "SM6_TITANX")
        rows.append(
            {
                "constraint": constraint,
                "case": case_name,
                "area_limit_mm2": area_max,
                "power_limit_W": power_max,
                "gpu_L_S": "%d/%d" % (large, small),
                "cpu_mcpat_active_W": "%.5f" % args.cpu_power_w,
                "gpu_accelwattch_active_sum_W": "%.5f" % gpu_total,
                "noc_orion_average_W": "%.6f" % noc_power,
                "modeled_active_power_W": "%.3f" % modeled_total,
                "e2e_cycles": int(summary["makespan_cycles"]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote %s" % args.output)
    for row in rows:
        print(
            "{constraint:5s} A={area_limit_mm2:4d} P={power_limit_W:4d} "
            "GPU-L/S={gpu_L_S} modeled={modeled_active_power_W} W cycles={e2e_cycles}".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
