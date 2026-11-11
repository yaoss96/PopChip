#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_legosim_metrics as collector


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_min_t_solution(experts, a_max=None, p_max=None):
    completion = float(100000 + experts)
    area_used = float(90 + 110 * experts)
    power_used = float(65 + 90 * experts)
    a_max = float(a_max if a_max is not None else area_used)
    p_max = float(p_max if p_max is not None else power_used)
    selected_counts = {"CPU": 1, "GPU-S": experts}
    value = {
        "optimization_mode": "ilp-perf",
        "primary_objective": "minimize joint MILP makespan T",
        "primary_incumbent_cycles": completion,
        "predicted_makespan_cycles": completion,
        "performance_proxy_cycles": completion,
        "tightened_performance_proxy_cycles": completion,
        "protocol_makespan_cycles": completion,
        "primary_mip_gap": 0.0,
        "total_area_mm2": area_used,
        "peak_power_W": power_used,
        "budgets": {
            "a_max": a_max, "p_max": p_max,
            "nmin": {"CPU": 1}, "nmax": {"CPU": 1}},
        "selected_type_counts": dict(selected_counts),
        "resource_selection": {
            "count_definition": "N_c=sum_p x[c,p]",
            "joint_with_mapping_and_placement": True,
            "fixed_bom": False,
            "selected_counts": dict(selected_counts),
            "area_budget_mm2": a_max,
            "area_used_mm2": area_used,
            "peak_power_budget_W": p_max,
            "peak_power_used_W": power_used,
        },
        "sites": ([{"type": "CPU", "l": 0, "i": 0, "j": 0}] +
                  [{"type": "GPU-S", "l": 0, "i": 0, "j": index + 1}
                   for index in range(experts)]),
        "objective_contract": {
            "active_objective_count": 1,
            "active_objective": "min T",
            "joint_selection_placement_scheduling": True,
            "chiplet_count_definition": "N_c=sum_p x[c,p]",
            "gpu_type_counts_fixed": False,
            "protocol_completion_in_T": True,
            "protocol_used_as_separate_objective": False,
            "legacy_task_schedule_role": "hard-feasibility-only",
            "energy_or_byte_hop_tie_break": False,
        },
        "protocol_model": {
            "completion_cycles": completion,
            "included_in_T": True,
            "diagnostic_only": False,
            "used_as_objective": True,
            "used_as_separate_objective": False,
        },
        "secondary_time_limit_s": None,
        "tertiary_time_limit_s": None,
    }
    for stage in ("secondary", "tertiary", "quaternary"):
        value.update({
            stage + "_objective": None,
            stage + "_objective_value": None,
            stage + "_status": None,
            stage + "_mip_gap": None,
            stage + "_has_incumbent": False,
            stage + "_wall_s": 0.0,
            stage + "_gap_certificate": {
                "certified": False, "reason": "stage-not-run"},
        })
    return value


def _fresh_manifest_fixture(raw_root):
    root = Path(raw_root) / "fresh_run"
    cases = []
    solution_map = {}
    for suite, ids in (
            ("budget", collector.EXPECTED_BUDGET_CASES),
            ("scaling", collector.EXPECTED_SCALING_CASES)):
        for case_id in ids:
            experts = (8 if suite == "budget" else
                       collector.EXPECTED_SCALING_CASES[case_id][0])
            if suite == "scaling":
                a_max = collector.SCALING_AREA_MM2_PER_EXPERT * experts
                p_max = collector.SCALING_POWER_W_PER_EXPERT * experts
            elif case_id.startswith("area_"):
                a_max = float(case_id.split("_")[1])
                p_max = 1625.0
            else:
                a_max = 2130.0
                p_max = float(case_id.split("_")[1])
            methods = (collector.BUDGET_METHODS if suite == "budget" else
                       collector.OPTIMIZATION_METHODS)
            paths = {
                method: collector._fresh_solution_path(
                    root.resolve(), suite, case_id, method)
                for method in methods}
            _write_json(paths["ilp_perf"], _valid_min_t_solution(
                experts, a_max=a_max, p_max=p_max))
            for method in methods:
                if method != "ilp_perf":
                    _write_json(paths[method], {
                        "method": method, "case_id": case_id,
                        "fixture": "manifest-bound"})
                solution_map[(suite, case_id, method)] = {
                    "path": paths[method].resolve(),
                    "sha256": _sha256(paths[method]),
                }
            row = {
                "suite": suite,
                "case_id": case_id,
                "budgets": {"a_max": a_max, "p_max": p_max},
                "ilp_perf": {
                    "status": "complete",
                    "solution_path": str(paths["ilp_perf"].resolve()),
                    "solution_sha256": _sha256(paths["ilp_perf"]),
                },
                "prior": {
                    "status": "complete",
                    "floorplet_solution_path": str(
                        paths["floorplet"].resolve()),
                    "tap_solution_path": str(paths["tap"].resolve()),
                    "solution_sha256": {
                        "floorplet": _sha256(paths["floorplet"]),
                        "tap": _sha256(paths["tap"]),
                    },
                },
            }
            if suite == "scaling":
                row.update({
                    "scaling_budget_contract_version":
                        collector.SCALING_BUDGET_CONTRACT_VERSION,
                    "budget_policy": collector.SCALING_BUDGET_POLICY,
                    "budget_target_bom": None,
                })
            if suite == "budget":
                row["matched_random"] = {
                    "status": "complete",
                    "solution_path": str(paths["matched_random"].resolve()),
                    "solution_sha256": {
                        "matched-random": _sha256(paths["matched_random"]),
                    },
                }
            cases.append(row)

    def replay_layout(suite, case_id, method):
        bound = solution_map[(suite, case_id, method)]
        return {
            "case_id": case_id,
            "method": method,
            "status": "ok",
            "solution_path": str(bound["path"]),
            "solution_sha256": bound["sha256"],
        }

    budget_results = root / "replays" / "budget" / "results.json"
    budget_layouts = [
        replay_layout("budget", case_id, method)
        for case_id in collector.EXPECTED_BUDGET_CASES
        for method in collector.BUDGET_METHODS]
    _write_json(budget_results, {
        "status": "complete", "layouts": budget_layouts})
    invocations = [{
        "name": "budget",
        "cases": ["budget:%s" % case_id
                  for case_id in collector.EXPECTED_BUDGET_CASES],
        "expected_rows": 32,
        "expected_layout_keys": [
            [case_id, method]
            for case_id in collector.EXPECTED_BUDGET_CASES
            for method in collector.BUDGET_METHODS],
        "expected_solutions": [
            {"case_id": case_id, "method": method,
             "solution_path": str(
                 solution_map[("budget", case_id, method)]["path"])}
            for case_id in collector.EXPECTED_BUDGET_CASES
            for method in collector.BUDGET_METHODS],
        "status": "complete",
        "results_json": str(budget_results.resolve()),
        "results_sha256": _sha256(budget_results),
    }]
    for case_id in collector.EXPECTED_SCALING_CASES:
        results = root / "replays" / "scaling" / case_id / "results.json"
        layouts = [replay_layout("scaling", case_id, method)
                   for method in collector.OPTIMIZATION_METHODS]
        _write_json(results, {"status": "complete", "layouts": layouts})
        invocations.append({
            "name": "scaling_" + case_id,
            "cases": ["scaling:" + case_id],
            "expected_rows": 3,
            "expected_layout_keys": [
                [case_id, method]
                for method in collector.OPTIMIZATION_METHODS],
            "expected_solutions": [
                {"case_id": case_id, "method": method,
                 "solution_path": str(
                     solution_map[("scaling", case_id, method)]["path"])}
                for method in collector.OPTIMIZATION_METHODS],
            "status": "complete",
            "results_json": str(results.resolve()),
            "results_sha256": _sha256(results),
        })
    manifest = {
        "schema_version": 5,
        "status": "complete",
        "out_root": str(root.resolve()),
        "selected_case_count": 13,
        "expected_replay_rows": 47,
        "protocol": dict(collector.FRESH_PROTOCOL),
        "cases": cases,
        "replay_invocations": invocations,
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return root.resolve(), budget_results.resolve(), manifest_path.resolve(), manifest


def _complete_rows(fresh_run):
    rows = []
    budget_source = fresh_run["invocations"]["budget"]["results_path"]
    for case_id in collector.EXPECTED_BUDGET_CASES:
        for method in collector.BUDGET_METHODS:
            rows.append({
                "suite": "budget", "case_id": case_id,
                "method": method, "cpu_count": 1,
                "gpu_l_count": 0, "gpu_s_count": 8, "experts": 8,
                "mesh": "6x6", "source_results_json": str(budget_source),
                "evidence_dir": str(fresh_run["out_root"] / "evidence"),
                "audit_status": "PASS", "errors": [],
            })
    for case_id, (experts, mesh) in collector.EXPECTED_SCALING_CASES.items():
        source = fresh_run["invocations"][
            "scaling:" + case_id]["results_path"]
        for method in collector.OPTIMIZATION_METHODS:
            rows.append({
                "suite": "scaling", "case_id": case_id,
                "method": method, "cpu_count": 1,
                "gpu_l_count": 0, "gpu_s_count": experts,
                "experts": experts,
                "mesh": mesh, "source_results_json": str(source),
                "evidence_dir": str(fresh_run["out_root"] / "evidence" /
                                    case_id / method),
                "audit_status": "PASS", "errors": [],
            })
    return rows


class UnifiedCollectorTest(unittest.TestCase):
    def test_method_aliases_are_stable_and_power_is_not_published(self):
        self.assertEqual(collector.normalize_method("ilp"), "ilp_perf")
        self.assertEqual(collector.normalize_method("ILP-Power"), "ilp_power")
        self.assertEqual(collector.normalize_method("floorplet-perf"),
                         "floorplet")
        self.assertEqual(collector.normalize_method("tap-spread"), "tap")
        self.assertEqual(collector.normalize_method("matched"),
                         "matched_random")
        self.assertNotIn("ilp_power", collector.OPTIMIZATION_METHODS)
        self.assertEqual(
            collector.BUDGET_METHODS,
            ("ilp_perf", "matched_random", "floorplet", "tap"))
        self.assertEqual(collector.EXPECTED_BUDGET_ROWS, 32)
        self.assertEqual(collector.EXPECTED_SCALING_ROWS, 15)
        self.assertEqual(collector.EXPECTED_FULL_ROWS, 47)

    def test_csv_schema_separates_metric_scope_and_provenance(self):
        for field in ("synchronized_cycles", "orion_noc_power_W",
                      "noc_active_cycles",
                      "nominal_orion_dynamic_noc_energy_uJ",
                      "shared_bom_peak_power_W",
                      "area_used_mm2", "bom_selection_provenance",
                      "gpu_counts_fixed", "fixed_bom", "solver_mip_gap",
                      "evidence_tier", "method_raw", "source_results_json"):
            self.assertIn(field, collector.CSV_FIELDS)
        self.assertNotIn("total_power_W", collector.CSV_FIELDS)

    def test_delay_info_uses_latest_final_packet_completion(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "delayInfo.txt"
            path.write_text(
                "1000 0 1 0 2 33 324\n"
                "2033 0 3 0 4 33 534 55 900\n"
                "2800 3 1 0 2 40 100\n",
                encoding="utf-8")
            errors = []
            active = collector.scaling_audit._read_delay_info_active_cycles(
                path, errors)
        self.assertEqual(errors, [])
        self.assertEqual(active, 2933)
        self.assertAlmostEqual(0.75 * active / 1000.0, 2.19975)

    def test_delay_info_rejects_count_mismatch(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "delayInfo.txt"
            path.write_text("1000 0 1 0 4 33 324\n", encoding="utf-8")
            errors = []
            active = collector.scaling_audit._read_delay_info_active_cycles(
                path, errors)
        self.assertIsNone(active)
        self.assertTrue(any("expected 9" in error for error in errors))
        self.assertIn("delayInfo.txt has no valid records", errors)

    def test_existing_scaling_audit_is_normalized(self):
        fixture = (collector.PROJECT_ROOT / "out/unified_legosim_data/"
                   "scaling_audit.json")
        if not fixture.is_file():
            self.skipTest("generated scaling audit fixture is unavailable")
        rows = collector._scaling_rows(fixture)
        self.assertEqual(len(rows), 15)
        self.assertEqual({row["method"] for row in rows},
                         {"ilp_perf", "floorplet", "tap"})
        self.assertTrue(all(row["audit_status"] == "PASS" for row in rows))
        self.assertTrue(all(row["cpu_count"] == 1 for row in rows))

    def test_modern_budget_replay_uses_synchronized_measured_cycles(self):
        fixture = (collector.PROJECT_ROOT / "out/ilp_power_reval_20260801/"
                   "replays/budget/results.json")
        if not fixture.is_file():
            self.skipTest("modern budget replay fixture is unavailable")
        payload = collector._load_json(fixture)
        record = next(row for row in payload["layouts"]
                      if row["case_id"] == "area_2130")
        checked = collector._verify_budget_evidence(record, fixture)
        self.assertEqual(checked["errors"], [])
        self.assertEqual(checked["cycles"], 100077)
        self.assertGreater(checked["noc_active_cycles"], 0)
        self.assertAlmostEqual(
            checked["noc_energy_uJ"],
            checked["power"] * checked["noc_active_cycles"] / 1000.0)

    def test_fresh_manifest_binds_protocol_and_every_replay_hash(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root, budget, manifest_path, manifest = _fresh_manifest_fixture(
                raw_dir)
            checked = collector._verify_fresh_run_manifest(
                manifest_path, budget)
            self.assertEqual(checked["out_root"], root)
            self.assertEqual(len(checked["invocations"]), 6)

            expected_methods = manifest["protocol"]["production_methods"]
            manifest["protocol"]["production_methods"] = {
                "budget": expected_methods["budget"] + ["ilp-power"],
                "scaling": expected_methods["scaling"],
            }
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "production_methods"):
                collector._verify_fresh_run_manifest(manifest_path, budget)
            manifest["protocol"]["production_methods"] = expected_methods

            manifest["protocol"]["ilp_perf_objective"] = "wrong-objective"
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "ilp_perf_objective"):
                collector._verify_fresh_run_manifest(manifest_path, budget)

            manifest["protocol"]["ilp_perf_objective"] = \
                "sole-joint-min-T"
            _write_json(manifest_path, manifest)
            budget.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                collector._verify_fresh_run_manifest(manifest_path, budget)

    def test_fresh_manifest_rejects_schema4_and_superseded_roots(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root, budget, manifest_path, manifest = _fresh_manifest_fixture(
                raw_dir)
            manifest["schema_version"] = 4
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "schema_version=.*5"):
                collector._verify_fresh_run_manifest(manifest_path, budget)

            manifest["schema_version"] = 5
            _write_json(manifest_path, manifest)
            (root / "SUPERSEDED.txt").write_text(
                "retired fixture", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SUPERSEDED"):
                collector._verify_fresh_run_manifest(manifest_path, budget)

    def test_public_min_t_solution_requires_protocol_in_T_flags(self):
        original = _valid_min_t_solution(8)
        mutations = (
            (("objective_contract", "protocol_completion_in_T"), False,
             "protocol_completion_in_T"),
            (("protocol_model", "included_in_T"), False, "included_in_T"),
            (("protocol_model", "diagnostic_only"), True,
             "diagnostic_only"),
            (("protocol_model", "used_as_objective"), False,
             "used_as_objective"),
        )
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "ilp_perf.json"
            _write_json(path, original)
            checked = collector._verify_public_min_T_solution(
                path, _sha256(path), "fixture")
            self.assertEqual(checked["bom"]["shared_bom_peak_power_W"], 785)
            for key_path, bad_value, message in mutations:
                with self.subTest(field=".".join(key_path)):
                    value = copy.deepcopy(original)
                    value[key_path[0]][key_path[1]] = bad_value
                    _write_json(path, value)
                    with self.assertRaisesRegex(ValueError, message):
                        collector._verify_public_min_T_solution(
                            path, _sha256(path), "fixture")

    def test_public_min_t_solution_rejects_secondary_objective(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "ilp_perf.json"
            value = _valid_min_t_solution(8)
            value.update({
                "secondary_objective": "minimize NoC energy",
                "secondary_objective_value": 1.0,
                "secondary_status": 2,
                "secondary_mip_gap": 0.0,
                "secondary_has_incumbent": True,
                "secondary_wall_s": 0.5,
                "secondary_gap_certificate": {
                    "certified": True, "reason": "optimal"},
            })
            _write_json(path, value)
            with self.assertRaisesRegex(ValueError, "post-T objective"):
                collector._verify_public_min_T_solution(
                    path, _sha256(path), "fixture")

    def test_fresh_budget_file_requires_exact_8_by_4_matrix(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir).resolve()
            results = root / "replays" / "budget" / "results.json"
            layouts = []
            for case_id in collector.EXPECTED_BUDGET_CASES:
                for method in collector.BUDGET_METHODS:
                    base = root / "layouts" / case_id / method
                    layouts.append({
                        "case_id": case_id,
                        "method": method,
                        "status": "ok",
                        "solution_path": str(base / "solution.json"),
                        "codegen": {"codegen_dir": str(base / "codegen")},
                        "legosim": {"summary_path": str(
                            base / "legosim" / "ilp_e2e_summary.txt")},
                    })
            _write_json(results, {"status": "complete", "layouts": layouts})
            case_meta = {
                case_id: {"case_id": case_id,
                          "budgets": {"a_max": 1, "p_max": 1}}
                for case_id in collector.EXPECTED_BUDGET_CASES}
            fresh_run = {"out_root": root, "run_id": "fresh"}
            with mock.patch.object(
                    collector, "_budget_row",
                    side_effect=lambda record, *_args, **_kwargs: {
                        "case_id": record["case_id"],
                        "method": collector.normalize_method(
                            record["method"])}) as build_row:
                rows = collector._fresh_budget_rows(
                    results, case_meta, fresh_run)
            self.assertEqual(len(rows), 32)
            self.assertEqual(build_row.call_count, 32)

            layouts[-1] = dict(layouts[0])
            _write_json(results, {"status": "complete", "layouts": layouts})
            with self.assertRaisesRegex(ValueError, "32 unique rows"):
                collector._fresh_budget_rows(results, case_meta, fresh_run)

    def test_complete_47_matrix_rejects_old_scaling_source(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root, budget, manifest_path, _ = _fresh_manifest_fixture(raw_dir)
            fresh_run = collector._verify_fresh_run_manifest(
                manifest_path, budget)
            rows = _complete_rows(fresh_run)
            collector._require_exact_complete_matrix(rows, fresh_run)
            rows[-1]["source_results_json"] = str(budget)
            with self.assertRaisesRegex(ValueError, "non-fresh source"):
                collector._require_exact_complete_matrix(rows, fresh_run)

    def test_fresh_collect_uses_legacy_core_as_metadata_only(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root, budget, manifest_path, _ = _fresh_manifest_fixture(raw_dir)
            fresh_run = collector._verify_fresh_run_manifest(
                manifest_path, budget)
            all_rows = _complete_rows(fresh_run)
            scaling_rows = [row for row in all_rows
                            if row["suite"] == "scaling"]
            budget_rows = [row for row in all_rows
                           if row["suite"] == "budget"]
            cases = []
            for case_id in collector.EXPECTED_BUDGET_CASES:
                cases.append({
                    "case_id": case_id,
                    "sweep_dimension": case_id.split("_", 1)[0],
                    "budget_value": float(case_id.split("_", 1)[1]),
                    "budgets": {"a_max": 2130, "p_max": 1625},
                    # This deliberately invalid legacy payload proves fresh
                    # mode never attempts to collect old case layouts.
                    "layouts": "MUST_NOT_BE_READ",
                })
            core = Path(raw_dir) / "old_raw_results.json"
            _write_json(core, {"status": "complete", "cases": cases})
            args = SimpleNamespace(
                scaling_audit=[str(Path(raw_dir) / "fresh_audit.json")],
                budget_core=str(core),
                budget_proxies=str(Path(raw_dir) / "must_not_be_read.json"),
                additional_results=[],
                fresh_budget_results=str(budget),
                fresh_run_manifest=str(manifest_path),
                require_complete_47=True,
            )
            with mock.patch.object(
                    collector, "_scaling_rows", return_value=scaling_rows), \
                    mock.patch.object(
                        collector, "_fresh_budget_rows",
                        return_value=budget_rows):
                report = collector.collect(args)
            self.assertEqual(report["audit_status"], "PASS")
            self.assertEqual(report["row_count"], 47)
            self.assertEqual(report["budget_collection_mode"],
                             "fresh_manifest_bound")
            self.assertTrue(report["strict_complete_47"])
            self.assertEqual(report["production_methods"]["budget"],
                             list(collector.BUDGET_METHODS))
            self.assertEqual(report["production_methods"]["scaling"],
                             list(collector.OPTIMIZATION_METHODS))


if __name__ == "__main__":
    unittest.main()
