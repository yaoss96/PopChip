#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_budget_layout_tikz as generator


CSV_FIELDS = [
    "suite", "case_id", "sweep_dimension", "budget_value",
    "area_budget_mm2", "power_budget_W", "experts", "mesh", "method",
    "cpu_count", "gpu_l_count", "gpu_s_count", "synchronized_cycles",
    "legosim_validation", "audit_status", "solution_sha256", "errors",
]


def _site(chip_type, row, column, tasks):
    return {"l": 0, "i": row, "j": column,
            "type": chip_type, "tasks": list(tasks)}


def _sites(case_id):
    auxiliary = ("e(0)", "t_attn", "t_gate", "t_dispatch", "t_combine")
    if case_id == "area_970":
        return [
            _site("CPU", 0, 0, ("t_norm",)),
            _site("GPU-S", 0, 1, auxiliary),
            _site("GPU-S", 0, 2, ("e(1)",)),
            _site("GPU-S", 0, 3, ("e(2)",)),
            _site("GPU-S", 0, 4, ("e(3)",)),
            _site("GPU-S", 0, 5, ("e(4)",)),
            _site("GPU-S", 1, 0, ("e(5)",)),
            _site("GPU-S", 1, 1, ("e(6)",)),
            _site("GPU-S", 1, 2, ("e(7)",)),
        ]
    if case_id == "area_1550":
        return [
            _site("CPU", 0, 0, ("t_norm",)),
            _site("GPU-L", 0, 1, auxiliary),
            _site("GPU-L", 0, 3, ("e(1)",)),
            _site("GPU-S", 0, 5, ("e(2)",)),
            _site("GPU-S", 1, 5, ("e(3)",)),
            _site("GPU-S", 2, 0, ("e(4)",)),
            _site("GPU-S", 2, 1, ("e(5)",)),
            _site("GPU-S", 2, 2, ("e(6)",)),
            _site("GPU-S", 2, 3, ("e(7)",)),
        ]
    if case_id == "area_2130":
        return [
            _site("CPU", 0, 0, ("t_norm",)),
            _site("GPU-L", 0, 1, auxiliary),
            _site("GPU-L", 0, 3, ("e(1)",)),
            _site("GPU-L", 2, 1, ("e(2)",)),
            _site("GPU-L", 2, 3, ("e(3)",)),
            _site("GPU-S", 0, 5, ("e(4)",)),
            _site("GPU-S", 1, 5, ("e(5)",)),
            _site("GPU-S", 2, 0, ("e(6)",)),
            _site("GPU-S", 3, 0, ("e(7)",)),
        ]
    raise AssertionError("unknown synthetic case %s" % case_id)


def _write_solutions(root):
    paths = {}
    for config in generator.CASES:
        case_id = config["case_id"]
        sites = _sites(case_id)
        task_site = {}
        for site in sites:
            anchor = [site["l"], site["i"], site["j"]]
            for task in site["tasks"]:
                task_site[task] = anchor
        counts = {name: sum(site["type"] == name for site in sites)
                  for name in generator.TYPE_SPECS}
        total_area = sum(generator.TYPE_SPECS[name]["area"] * count
                         for name, count in counts.items())
        peak_power = sum(generator.TYPE_SPECS[name]["peak_W"] * count
                         for name, count in counts.items())
        value = {
            "optimization_mode": "ilp-perf",
            "cpu_anchor_policy": "free",
            "grid": {"L": 1, "X": 6, "Y": 6},
            "budgets": {"a_max": config["area_budget"], "p_max": 1625.0},
            "sites": sites,
            "task_site": task_site,
            "total_area_mm2": total_area,
            "peak_power_W": peak_power,
        }
        path = (root / "solutions" / "budget" / case_id / "ilp_perf.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        paths[case_id] = path
    return paths


def _rows(solution_paths):
    cycles = {"area_970": 126019, "area_1550": 105511,
              "area_2130": 100938}
    rows = []
    for config in generator.CASES:
        case_id = config["case_id"]
        rows.append({
            "suite": "budget", "case_id": case_id,
            "sweep_dimension": "area",
            "budget_value": config["area_budget"],
            "area_budget_mm2": config["area_budget"],
            "power_budget_W": 1625,
            "experts": 8, "mesh": "6x6", "method": "ilp_perf",
            "cpu_count": 1, "gpu_l_count": config["gpu_l"],
            "gpu_s_count": config["gpu_s"],
            "synchronized_cycles": cycles[case_id],
            "legosim_validation": "PASS", "audit_status": "PASS",
            "solution_sha256": generator._digest(
                solution_paths[case_id].read_bytes()),
            "errors": "",
        })
    return rows


def _write_csv(path, rows):
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class BudgetLayoutTikzGeneratorTest(unittest.TestCase):
    def _fixture(self, root):
        canonical = root / "canonical"
        paths = _write_solutions(canonical)
        metrics = root / "unified.csv"
        rows = _rows(paths)
        _write_csv(metrics, rows)
        return canonical, metrics, rows

    def test_synthetic_inputs_generate_three_single_column_panels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical, metrics, _ = self._fixture(root)
            output = root / "paper" / "budget-layouts-tikz.tex"
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                status = generator.main([
                    "--canonical-root", str(canonical),
                    "--metrics-csv", str(metrics),
                    "--output", str(output),
                ])
            self.assertEqual(status, 0)
            text = output.read_text(encoding="utf-8")
            self.assertEqual(text.count(r"\begin{minipage}[t]{0.318\linewidth}"),
                             3)
            self.assertIn("Metrics CSV SHA-256: %s" %
                          generator._digest(metrics.read_bytes()), text)
            for case_id, path in _write_solutions(canonical).items():
                self.assertIn("%s solution SHA-256: %s" %
                              (case_id, generator._digest(path.read_bytes())), text)
            self.assertIn(
                "source (l,i,j)=(0,0,0), type=CPU, tasks=[t_norm]", text)
            self.assertIn(r"\budgetlgpu{1}{4}{L:e0\\At+Gt\\Ds+Cb}", text)
            self.assertIn("GPU-L/S $=4/4$", text)
            self.assertIn("used 2{,}130\\,mm$^2$; BOM pk. 1{,}625\\,W", text)
            self.assertIn("LEGOSim: 105{,}511 measured cycles", text)
            self.assertFalse(any(output.parent.glob(output.name + ".tmp.*")))

    def test_missing_csv_or_required_row_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical, metrics, rows = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "metrics CSV not found"):
                generator._render(canonical, root / "missing.csv")
            _write_csv(metrics, rows[:-1])
            with self.assertRaisesRegex(ValueError, "lacks required ilp_perf rows"):
                generator._render(canonical, metrics)

    def test_failed_validation_does_not_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical, metrics, rows = self._fixture(root)
            rows[1]["legosim_validation"] = "FAIL"
            _write_csv(metrics, rows)
            output = root / "paper" / "budget-layouts-tikz.tex"
            output.parent.mkdir(parents=True)
            output.write_text("sentinel\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError,
                                        "legosim_validation is not PASS"):
                generator.main([
                    "--canonical-root", str(canonical),
                    "--metrics-csv", str(metrics),
                    "--output", str(output),
                ])
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")
            self.assertFalse(any(output.parent.glob(output.name + ".tmp.*")))

    def test_csv_solution_hash_must_bind_the_drawn_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical, metrics, rows = self._fixture(root)
            rows[0]["solution_sha256"] = "0" * 64
            _write_csv(metrics, rows)
            with self.assertRaisesRegex(ValueError,
                                        "does not match canonical JSON"):
                generator._render(canonical, metrics)


if __name__ == "__main__":
    unittest.main()
