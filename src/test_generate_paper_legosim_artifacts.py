#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import csv
import re
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_paper_legosim_artifacts as generator


FIELDS = [
    "suite", "case_id", "sweep_dimension", "budget_value",
    "area_budget_mm2", "power_budget_W", "area_used_mm2", "experts",
    "mesh", "method",
    "cpu_count", "gpu_l_count", "gpu_s_count", "synchronized_cycles",
    "shared_bom_peak_power_W", "bom_selection_provenance",
    "gpu_counts_fixed", "fixed_bom", "solver_mip_gap",
    "orion_noc_power_W", "noc_active_cycles",
    "nominal_orion_dynamic_noc_energy_uJ", "legosim_validation",
    "audit_status", "evidence_tier", "evidence_dir",
    "source_results_json", "errors",
]


def synthetic_rows():
    rows = []
    for scale_index, (case_id, config) in enumerate(
            sorted(generator.SCALE_CONFIG.items(), key=lambda item: item[1][0])):
        experts, mesh = config
        cpu = 1
        if experts == 8:
            gpu_l, gpu_s = 1, 8
        else:
            # Exercise arbitrary shared evaluation BOMs rather than assuming
            # the former hard-coded 25/75 split.  These synthetic rows do not
            # claim joint-solver provenance.
            gpu_l, gpu_s = max(1, experts // 8), experts - max(
                1, experts // 8) + 1
        for method_index, method in enumerate(generator.SCALING_METHODS):
            noc_power = 0.3 + method_index * 0.01
            noc_active_cycles = 100000 + scale_index * 1000 + method_index * 100
            area_budget = generator.SCALING_AREA_MM2_PER_EXPERT * experts
            power_budget = generator.SCALING_POWER_W_PER_EXPERT * experts
            area_used = (generator.CPU_AREA_MM2 * cpu +
                         generator.GPU_L_AREA_MM2 * gpu_l +
                         generator.GPU_S_AREA_MM2 * gpu_s)
            chip_peak = (generator.CPU_PEAK_W * cpu +
                         generator.GPU_L_PEAK_W * gpu_l +
                         generator.GPU_S_PEAK_W * gpu_s)
            rows.append({
                "suite": "scaling", "case_id": case_id,
                "sweep_dimension": "experts", "budget_value": experts,
                "area_budget_mm2": area_budget,
                "power_budget_W": power_budget,
                "area_used_mm2": area_used,
                "experts": experts, "mesh": mesh, "method": method,
                "cpu_count": cpu, "gpu_l_count": gpu_l,
                "gpu_s_count": gpu_s,
                "shared_bom_peak_power_W": chip_peak,
                "bom_selection_provenance": "fresh-joint-min-T-MILP",
                "gpu_counts_fixed": False, "fixed_bom": False,
                "solver_mip_gap": 0.01,
                "synchronized_cycles": 100000 + scale_index * 10000 +
                                       method_index * 1000,
                "orion_noc_power_W": noc_power,
                "noc_active_cycles": noc_active_cycles,
                "nominal_orion_dynamic_noc_energy_uJ": (
                    noc_power * noc_active_cycles / 1000.0),
                "legosim_validation": "PASS", "audit_status": "PASS",
                "evidence_tier": "strict_scaling_joint_bom_v4",
                "evidence_dir": "evidence/%s/%s" % (case_id, method),
                "source_results_json": "replays/scaling/%s/results.json" %
                                       case_id,
                "errors": "",
            })
    # Deliberately use legal but not always budget-maximal GPU-L counts.  This
    # catches any regression that turns solver output back into a hard-coded
    # tier BOM in the paper generator.
    selected_budget_bom = {
        "area_970": (0, 8), "area_1260": (0, 8),
        "area_1550": (1, 7), "area_2130": (3, 5),
        "power_785": (0, 8), "power_995": (0, 8),
        "power_1205": (1, 7), "power_1625": (3, 5),
    }
    for case_index, (case_id, config) in enumerate(
            generator.BUDGET_CONFIG.items()):
        dimension, value, area, power = config
        cpu = 1
        gpu_l, gpu_s = selected_budget_bom[case_id]
        for method_index, method in enumerate(generator.BUDGET_METHODS):
            noc_power = 0.4 + method_index * 0.01
            noc_active_cycles = 90000 + case_index * 1000 + method_index * 100
            area_used = (generator.CPU_AREA_MM2 * cpu +
                         generator.GPU_L_AREA_MM2 * gpu_l +
                         generator.GPU_S_AREA_MM2 * gpu_s)
            chip_peak = (generator.CPU_PEAK_W * cpu +
                         generator.GPU_L_PEAK_W * gpu_l +
                         generator.GPU_S_PEAK_W * gpu_s)
            rows.append({
                "suite": "budget", "case_id": case_id,
                "sweep_dimension": dimension, "budget_value": value,
                "area_budget_mm2": area, "power_budget_W": power,
                "area_used_mm2": area_used,
                "experts": 8, "mesh": "6x6", "method": method,
                "cpu_count": cpu, "gpu_l_count": gpu_l,
                "gpu_s_count": gpu_s,
                "shared_bom_peak_power_W": chip_peak,
                "bom_selection_provenance": "fresh-joint-min-T-MILP",
                "gpu_counts_fixed": False, "fixed_bom": False,
                "solver_mip_gap": 0.0,
                "synchronized_cycles": 100000 + case_index * 1000 +
                                       method_index * 100,
                "orion_noc_power_W": noc_power,
                "noc_active_cycles": noc_active_cycles,
                "nominal_orion_dynamic_noc_energy_uJ": (
                    noc_power * noc_active_cycles / 1000.0),
                "legosim_validation": "PASS", "audit_status": "PASS",
                "evidence_tier": "fresh_constraint_replay_v1",
                "evidence_dir": "evidence/%s/%s" % (case_id, method),
                "source_results_json": "replays/budget/results.json",
                "errors": "",
            })
    return rows


class PaperArtifactGeneratorTest(unittest.TestCase):
    def _write_csv(self, path, rows):
        with open(str(path), "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_complete_synthetic_matrix_generates_all_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "unified.csv"
            paper = root / "paper"
            self._write_csv(source, synthetic_rows())
            outputs = generator._render(source)
            generator._atomic_outputs(paper, outputs)
            self.assertEqual(set(path.name for path in paper.iterdir()),
                             set(generator.OUTPUT_NAMES))
            table = (paper / generator.OUTPUT_NAMES[0]).read_text()
            scaling = (paper / generator.OUTPUT_NAMES[1]).read_text()
            budget = (paper / generator.OUTPUT_NAMES[2]).read_text()
            self.assertIn("Source SHA-256:", table)
            self.assertIn("fresh joint-BOM selection", table)
            self.assertNotIn("no fresh joint-selection provenance", table)
            self.assertIn("1085", table)  # 65 + 1*300 + 8*90
            self.assertIn("2/15", table)  # E16 synthetic evaluation BOM
            table_rows = [line for line in table.splitlines()
                          if line.strip().endswith(r"\\")]
            self.assertEqual(len(table_rows), 5)
            self.assertTrue(all(len(line.split(" & ")) == 13
                                 for line in table_rows))
            self.assertIn(r"\ILPPerfNormCoords", scaling)
            self.assertIn("(8,1.000000)", scaling)
            self.assertIn(r"\ScaleNoCPowerReadytrue", scaling)
            self.assertIn(
                r"\providecommand{\ScaleNormYMax}{1.100000}", scaling)
            self.assertIn(
                r"\providecommand{\ScaleNoCPowerYMax}{0.345000}", scaling)
            self.assertIn(r"\FloorpletNoCPowerCoords", scaling)
            self.assertIn("(8,0.310000)", scaling)
            self.assertIn(r"\ScaleNoCEnergyReadytrue", scaling)
            self.assertIn(r"\ILPPerfCycleEnergyCoords", scaling)
            self.assertIn(r"\FloorpletCycleEnergyCoords", scaling)
            self.assertIn(r"\TAPCycleEnergyCoords", scaling)
            self.assertIn("(100.000,30.000000) [E8]", scaling)
            self.assertIn("[E128]", scaling)
            self.assertIn("cycle-energy trade-off", scaling)
            self.assertIn("nominal dynamic NoC energy", scaling)
            self.assertNotIn("ILPPower", scaling)
            self.assertNotIn("MatchedRandom", scaling)
            self.assertIn(r"\BudgetAreaMatchedRandomCoords", budget)
            self.assertIn(r"\BudgetAreaILPSelectionCoords", budget)
            self.assertIn("[0/8]", budget)
            self.assertIn("[3/5]", budget)
            self.assertIn(r"\BudgetCurvesReadytrue", budget)
            self.assertNotIn("ILPPower", budget)

    def test_scaling_ymax_tracks_worst_ratio_with_safe_headroom(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "wide_ratio.csv"
            rows = synthetic_rows()
            # E128 has a 140000-cycle row minimum.  Make TAP exactly 2.25x
            # slower so the old fixed 1.30 limit would visibly clip it.
            tap = next(row for row in rows
                       if row["suite"] == "scaling" and
                       row["case_id"] == "e128_m16" and
                       row["method"] == "tap")
            tap["synchronized_cycles"] = 315000
            self._write_csv(source, rows)

            scaling = generator._render(source)[generator.OUTPUT_NAMES[1]]
            match = re.search(
                r"\\providecommand\{\\ScaleNormYMax\}\{([0-9.]+)\}",
                scaling)
            self.assertIsNotNone(match)
            ymax = float(match.group(1))
            self.assertGreaterEqual(ymax, 2.25 * 1.05)
            self.assertLessEqual(ymax, 2.25 * 1.10)
            self.assertGreater(ymax, 2.25)
            self.assertNotEqual(ymax, 1.30)

    def test_scaling_only_updates_table_and_scaling_but_preserves_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "scaling_only.csv"
            paper = root / "paper"
            scaling_rows = [row for row in synthetic_rows()
                            if row["suite"] == "scaling"]
            self._write_csv(source, scaling_rows)
            paper.mkdir()
            budget_path = paper / generator.OUTPUT_NAMES[2]
            budget_path.write_text("audited-budget-sentinel\n",
                                   encoding="utf-8")

            outputs = generator._render(source, suites=("scaling",))
            self.assertEqual(set(outputs), set(generator.OUTPUT_NAMES[:2]))
            generator._atomic_outputs(paper, outputs)

            self.assertEqual("audited-budget-sentinel\n",
                             budget_path.read_text(encoding="utf-8"))
            self.assertTrue((paper / generator.OUTPUT_NAMES[0]).is_file())
            self.assertTrue((paper / generator.OUTPUT_NAMES[1]).is_file())

    def test_scaling_figure_consumes_generated_ymax_macro(self):
        figure = (generator.PROJECT_ROOT / "paper_date" /
                  "scaling-tradeoff-fig.tex").read_text(encoding="utf-8")
        self.assertIn(r"ymax=\ScaleNormYMax", figure)
        self.assertIn(r"ymax=\ScaleNoCPowerYMax", figure)
        self.assertIn("NoC power (W)", figure)
        self.assertIn(r"\ILPPerfNoCPowerCoords", figure)
        self.assertIn(r"\ifScaleNoCEnergyReady", figure)
        self.assertIn(r"\ILPPerfCycleEnergyCoords", figure)
        self.assertIn(r"NoC energy ($\mu$J)", figure)
        self.assertIn("Cycle--energy trade-off", figure)
        self.assertIn("E labels show experts", figure)
        self.assertIn("point meta=explicit symbolic", figure)
        self.assertNotIn("ymax=1.30", figure)

    def test_scaling_power_ymax_tracks_measured_power(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "wide_power.csv"
            rows = synthetic_rows()
            tap = next(row for row in rows
                       if row["suite"] == "scaling" and
                       row["case_id"] == "e128_m16" and
                       row["method"] == "tap")
            tap["orion_noc_power_W"] = 2.25
            tap["nominal_orion_dynamic_noc_energy_uJ"] = (
                2.25 * tap["noc_active_cycles"] / 1000.0)
            self._write_csv(source, rows)

            scaling = generator._render(source)[generator.OUTPUT_NAMES[1]]
            match = re.search(
                r"\\providecommand\{\\ScaleNoCPowerYMax\}\{([0-9.]+)\}",
                scaling)
            self.assertIsNotNone(match)
            ymax = float(match.group(1))
            self.assertGreaterEqual(ymax, 2.25 * 1.05)
            self.assertLessEqual(ymax, 2.25 * 1.10)
            self.assertGreater(ymax, 2.25)

    def test_budget_legend_names_both_prior_art_methods_as_proxies(self):
        figure = (generator.PROJECT_ROOT / "paper_date" /
                  "budget-curves-fig.tex").read_text(encoding="utf-8")
        self.assertIn(r"Floorplet proxy~\cite{floorplet}", figure)
        self.assertIn(r"TAP-2.5D proxy~\cite{tap25d}", figure)

    def test_incomplete_matrix_refuses_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "incomplete.csv"
            rows = synthetic_rows()[:-1]
            self._write_csv(source, rows)
            with self.assertRaises(ValueError):
                generator._render(source)
            self.assertFalse((root / "paper").exists())

    def test_failed_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "failed.csv"
            rows = synthetic_rows()
            rows[0]["audit_status"] = "FAIL"
            self._write_csv(source, rows)
            with self.assertRaises(ValueError):
                generator._render(source)

    def test_scaling_methods_must_share_the_ilp_selected_bom(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bom_mismatch.csv"
            rows = synthetic_rows()
            row = next(value for value in rows
                       if value["suite"] == "scaling" and
                       value["case_id"] == "e016_m08" and
                       value["method"] == "tap")
            row["gpu_s_count"] += 1
            row["area_used_mm2"] += generator.GPU_S_AREA_MM2
            row["shared_bom_peak_power_W"] += generator.GPU_S_PEAK_W
            self._write_csv(source, rows)
            with self.assertRaisesRegex(ValueError, "scaling BOM mismatch"):
                generator._render(source)

    def test_budget_methods_must_share_the_ilp_selected_bom(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "budget_bom_mismatch.csv"
            rows = synthetic_rows()
            row = next(value for value in rows
                       if value["suite"] == "budget" and
                       value["case_id"] == "area_1550" and
                       value["method"] == "tap")
            row["gpu_l_count"] += 1
            row["gpu_s_count"] -= 1
            row["area_used_mm2"] += (
                generator.GPU_L_AREA_MM2 - generator.GPU_S_AREA_MM2)
            row["shared_bom_peak_power_W"] += (
                generator.GPU_L_PEAK_W - generator.GPU_S_PEAK_W)
            self._write_csv(source, rows)
            with self.assertRaisesRegex(ValueError, "budget BOM mismatch"):
                generator._render(source)

    def test_rejects_removed_ilp_power_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "removed_mode.csv"
            rows = synthetic_rows()
            removed = dict(rows[0])
            removed["method"] = "ilp_power"
            rows.append(removed)
            self._write_csv(source, rows)
            with self.assertRaisesRegex(ValueError, "expected exactly"):
                generator._render(source)

    def test_rejects_energy_inconsistent_with_measured_power_and_activity(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad_energy.csv"
            rows = synthetic_rows()
            rows[0]["nominal_orion_dynamic_noc_energy_uJ"] += 1.0
            self._write_csv(source, rows)
            with self.assertRaisesRegex(ValueError, "is inconsistent"):
                generator._render(source)

    def test_rejects_nonfresh_budget_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "legacy.csv"
            rows = synthetic_rows()
            budget = next(row for row in rows if row["suite"] == "budget")
            budget["evidence_tier"] = "legacy_budget_mirror_v1"
            self._write_csv(source, rows)
            with self.assertRaisesRegex(ValueError, "evidence_tier"):
                generator._render(source)

    def test_rejects_scaling_row_with_fixed_gpu_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "fixed_gpu_counts.csv"
            rows = synthetic_rows()
            row = next(value for value in rows
                       if value["suite"] == "scaling")
            row["gpu_counts_fixed"] = True
            self._write_csv(source, rows)
            with self.assertRaisesRegex(ValueError, "gpu_counts_fixed"):
                generator._render(source)

    def test_rejects_old_target_bom_derived_scaling_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "old_scaling_cap.csv"
            rows = synthetic_rows()
            for row in rows:
                if (row["suite"] == "scaling" and
                        row["case_id"] == "e008_m06"):
                    row["area_budget_mm2"] = 1550
            self._write_csv(source, rows)
            with self.assertRaisesRegex(ValueError, "scaling area budget"):
                generator._render(source)


if __name__ == "__main__":
    unittest.main()
