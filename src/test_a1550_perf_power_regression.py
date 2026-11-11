#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_a1550_perf_power_regression as gate
import run_full_constraint_revalidation as full_validation


@unittest.skip(
    "retired: ILP-Power/Pareto endpoint was removed; the active model has "
    "one ILP-Perf objective, min T")
class A1550PerfPowerRegressionTest(unittest.TestCase):
    def _solution(self, method, primary=100.0, raw_t=120.0,
                  gap=0.01, secondary_gap=0.02,
                  protocol_cycles=None, protocol_gap=0.03,
                  protocol_cap=None):
        if protocol_cycles is None:
            protocol_cycles = primary
        if protocol_cap is None:
            protocol_cap = protocol_cycles * (
                1.000001 if method == "ilp-perf" else 1.20)
        signature = "sha256:test-bundle-signature"
        frozen_bundles = [
            {"id": 0, "type": "CPU", "tasks": ["t_gate", "t_norm"]},
            {"id": 1, "type": "GPU-L",
             "tasks": ["t_attn", "t_dispatch", "e(0)", "t_combine"]},
            {"id": 2, "type": "GPU-S", "tasks": ["e(1)"]},
        ]
        return {
            "optimization_mode": method,
            "primary_status": 2,
            "primary_mip_gap": gap,
            "primary_has_incumbent": True,
            "primary_gap_certificate": {
                "certified": True, "reason": None},
            "primary_incumbent_cycles": primary,
            "primary_objective_bound_cycles": primary * (1.0 - gap),
            "primary_wall_s": 3.0,
            "secondary_status": 2,
            "secondary_mip_gap": secondary_gap,
            "secondary_wall_s": 2.0,
            "secondary_has_incumbent": True,
            "secondary_objective": (
                "fixed-order max-plus protocol completion"
                if method == "ilp-perf" else
                "NoC energy under 1.20x ILP-Perf protocol cap"),
            "secondary_objective_value": (
                protocol_cycles if method == "ilp-perf" else 1.0e-6),
            "secondary_gap_certificate": {
                "certified": True, "reason": None},
            "tertiary_status": 2,
            "tertiary_mip_gap": protocol_gap,
            "tertiary_wall_s": 1.0,
            "tertiary_objective": (
                "protocol-performance tie-break at fixed NoC energy"),
            "tertiary_objective_value": protocol_cycles,
            "tertiary_has_incumbent": True,
            "tertiary_gap_certificate": {
                "certified": True, "reason": None},
            "time_to_first_feasible_s": 0.1,
            "predicted_makespan_cycles": raw_t,
            "performance_proxy_cycles": protocol_cycles,
            "protocol_stage_mip_gap": protocol_gap,
            "protocol_model": {
                "kind": "fixed-order-max-plus-event-dag",
                "event_count": 128,
                "phases": ["init", "dispatch", "compute", "combine",
                           "done"],
                "compute_profile": "e8-legosim-calibrated-v1",
                "worker_id_policy": "logical-expert-group",
            },
            "performance_reference_cycles": protocol_cycles,
            "performance_cap_cycles": protocol_cap,
            "performance_relaxation": (
                1.0e-6 if method == "ilp-perf" else 0.20),
            "noc_power_proxy": {"energy_J": 1.0e-6},
            "routing": {
                "kind": (
                    "phase-separated-source-aggregated-splittable-"
                    "packet-flow"),
                "routed_byte_hop": 1000.0,
                "maximum_link_utilization": 0.1,
                "maximum_flow_residual_KiB": 0.0,
            },
            "budgets": {"a_max": 1550.0, "p_max": 1625.0},
            "total_area_mm2": 600.0,
            "peak_power_W": 455.0,
            "grid": {"L": 1, "X": 6, "Y": 6},
            "cpu_anchor_policy": "free",
            "expert_slots": {"CPU": 1, "GPU-L": 1, "GPU-S": 1},
            "task_schedule": {},
            "bundle_signature": signature,
            "internal_bundle_refinement": ({
                "enabled": True,
                "source_stage": "Stage 0 analytical-T incumbent",
                "coordinates_fixed": False,
                "search_space": (
                    "exact typed complete bundles; free coordinates"),
                "bundle_count": 3,
                "type_counts": {"CPU": 1, "GPU-L": 1, "GPU-S": 1},
                "bundle_signature": signature,
                "external_bundle_signature": None,
                "matches_external_bundles": None,
                "final_bundle_signature": signature,
                "signature_preserved": True,
                "bundles": frozen_bundles,
            } if method == "ilp-perf" else {
                "enabled": False,
                "coordinates_fixed": False,
                "external_bundle_signature": signature,
            }),
            "reference_coordinates_warm_started": method == "ilp-power",
            "comparison_reference": {
                "fixed_complete_bundles": method == "ilp-power",
                "source_bundle_signature": (
                    signature if method == "ilp-power" else None),
                "final_bundle_signature": signature,
                "bundle_signature_preserved": (
                    True if method == "ilp-power" else None),
            },
            "sites": [
                {"l": 0, "i": 0, "j": 0, "type": "CPU",
                 "tasks": ["t_gate", "t_norm"]},
                {"l": 0, "i": 1, "j": 1, "type": "GPU-L",
                 "tasks": ["t_attn", "t_dispatch", "e(0)",
                           "t_combine"]},
                {"l": 0, "i": 1, "j": 2, "type": "GPU-S",
                 "tasks": ["e(1)"]},
            ],
        }

    def _row(self, method, cycles, power, average_delay=10.0,
             finished=1):
        return {
            "case_id": gate.CASE_ID,
            "method": method,
            "status": "ok",
            "solution_metrics": {
                "total_area_mm2": 600.0,
                "peak_power_W": 455.0,
                "cpu_count": 1,
                "gpu_l_count": 1,
                "gpu_s_count": 1,
                # Deliberately unrelated to measured cycles: this guards
                # against accidentally reporting a solver surrogate.
                "predicted_makespan_cycles": 99999999.0,
            },
            "codegen": {"dispatch_formal": True},
            "legosim": {
                "status": "ok",
                "validation": "PASS",
                "measured_makespan_cycles": cycles,
                "bench_makespan_cycles": cycles,
                "synchronized_makespan_cycles": cycles,
                "completion_delta_cycles": 0,
                "summary": {
                    "makespan_cycles": cycles,
                    "synchronized_makespan_cycles": cycles,
                    "completion_delta_cycles": 0,
                    "gpu_cycle_max": max(1, int(cycles / 2)),
                    "n_gpu_workers": 2,
                    "popnet_total_power": power,
                    "popnet_average_delay": average_delay,
                    "popnet_total_finished": finished,
                },
            },
        }

    def _fixture(self, root, perf_cycles=100, power_cycles=105,
                 perf_power=1.0, power_power=0.8,
                 perf_active=100, power_active=100,
                 perf_gap=0.01, power_gap=0.01):
        root = Path(root)
        solution_dir = root / "solutions" / "budget" / gate.CASE_ID
        solution_dir.mkdir(parents=True)
        perf_path = solution_dir / "ilp_perf.json"
        power_path = solution_dir / "ilp_power.json"
        perf_path.write_text(json.dumps(self._solution(
            "ilp-perf", primary=100.0, raw_t=101.0, gap=perf_gap)),
            encoding="utf-8")
        power_path.write_text(json.dumps(self._solution(
            "ilp-power", primary=102.0, raw_t=122.4, gap=power_gap)),
            encoding="utf-8")

        params_path = root / "params.json"
        params_path.write_text(json.dumps({
            "library": {
                "CPU": {"area": 90.0, "p_peak": 65.0},
                "GPU-L": {"area": 400.0, "p_peak": 300.0},
                "GPU-S": {"area": 110.0, "p_peak": 90.0},
            },
        }), encoding="utf-8")

        replay = root / "replays" / "a1550_perf_power"
        rows = [
            self._row("ilp-perf", perf_cycles, perf_power),
            self._row("ilp-power", power_cycles, power_power),
        ]
        results_path = replay / "results.json"
        replay.mkdir(parents=True)
        for row, path in ((rows[0], perf_path), (rows[1], power_path)):
            row["solution_path"] = str(path.resolve())
            row["solution_sha256"] = gate._sha256(path)
        results_path.write_text(json.dumps({
            "status": "complete",
            "params_path": str(params_path.resolve()),
            "params_sha256": gate._sha256(params_path),
            "layouts": rows,
        }), encoding="utf-8")
        for method, active in (("ilp-perf", perf_active),
                               ("ilp-power", power_active)):
            result_dir = (replay / "layouts" / gate.CASE_ID / method /
                          "legosim_result")
            result_dir.mkdir(parents=True)
            # completion = start + final delay = active
            (result_dir / "delayInfo.txt").write_text(
                "10 0 1 0 2 1 %d\n" % (active - 10), encoding="utf-8")
        return perf_path, power_path, results_path

    def _refresh_replay_identity(self, results_path, method,
                                 solution_path):
        value = json.loads(Path(results_path).read_text(encoding="utf-8"))
        for row in value["layouts"]:
            if row.get("method") == method:
                row["solution_path"] = str(Path(solution_path).resolve())
                row["solution_sha256"] = gate._sha256(solution_path)
                break
        Path(results_path).write_text(json.dumps(value), encoding="utf-8")

    def test_pass_when_power_saves_energy_but_is_not_faster(self):
        with tempfile.TemporaryDirectory() as temporary:
            perf, power, results = self._fixture(temporary)
            report = gate.audit_pair(perf, power, results)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["comparison"][
            "perf_dominated_on_cycles_energy"])
        self.assertEqual(report["methods"]["ilp-perf"]["measured"][
            "noc_dynamic_energy_uJ_at_1GHz"], 0.1)

    def test_rejects_power_that_is_faster_and_lower_energy_and_power(self):
        with tempfile.TemporaryDirectory() as temporary:
            perf, power, results = self._fixture(
                temporary, perf_cycles=104401, power_cycles=104379,
                perf_power=0.437201, power_power=0.334435,
                perf_active=103887, power_active=103841)
            report = gate.audit_pair(perf, power, results)
        self.assertEqual(report["status"], "FAIL")
        comparison = report["comparison"]
        self.assertTrue(comparison["perf_dominated_on_all_three"])
        self.assertEqual(comparison["power_minus_perf"]["cycles"], -22)
        self.assertIn(
            "ILP-Perf is dominated on measured cycles and NoC energy",
            report["gate"]["failures"])
        self.assertIn(
            "ILP-Perf is dominated on measured cycles and average NoC power",
            report["gate"]["failures"])

    def test_rejects_cycles_plus_power_even_if_active_energy_is_higher(self):
        with tempfile.TemporaryDirectory() as temporary:
            perf, power, results = self._fixture(
                temporary, perf_cycles=100, power_cycles=99,
                perf_power=1.0, power_power=0.9,
                perf_active=100, power_active=200)
            report = gate.audit_pair(perf, power, results)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["comparison"][
            "perf_dominated_on_cycles_energy"])
        self.assertTrue(report["comparison"][
            "perf_dominated_on_cycles_power"])

    def test_rejects_equal_modes_as_no_measured_tradeoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            perf, power, results = self._fixture(
                temporary, perf_cycles=104961, power_cycles=104961,
                perf_power=0.354132, power_power=0.354132,
                perf_active=104564, power_active=104564)
            report = gate.audit_pair(perf, power, results)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["comparison"]["perf_faster"])
        self.assertFalse(report["comparison"]["power_saves_noc"])
        self.assertIn(
            "ILP-Perf does not have fewer measured cycles than ILP-Power",
            report["gate"]["failures"])
        self.assertIn(
            "ILP-Power does not reduce measured NoC energy or average power",
            report["gate"]["failures"])

    def test_primary_gap_threshold_is_an_optional_hard_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            perf, power, results = self._fixture(
                temporary, perf_gap=0.2, power_gap=0.001)
            report = gate.audit_pair(
                perf, power, results, max_primary_gap=0.05)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("Stage-0 T gap" in reason
                            for reason in report["gate"]["failures"]))

    def test_e128_style_uncertified_stage0_is_valid_for_both_modes(self):
        for method in ("ilp-perf", "ilp-power"):
            value = self._solution(method, primary=3220698.998)
            value["primary_mip_gap"] = None
            value["primary_has_incumbent"] = True
            value["primary_gap_certificate"] = {
                "certified": False,
                "reason": "solver-reported-nonfinite-mip-gap",
            }
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / (method + ".json")
                path.write_text(json.dumps(value), encoding="utf-8")
                full_validation._validate_solution(path, method)
            stage0 = gate._stage0_gap(value, method)
            self.assertIsNone(stage0["gap"])
            self.assertFalse(stage0["gap_certified"])
            self.assertEqual(
                stage0["gap_reason"],
                "solver-reported-nonfinite-mip-gap")
            metrics = gate._solver_metrics(value, method)
            self.assertFalse(metrics["solver"]["stage0_T"]
                             ["gap_certificate"]["certified"])

    def test_uncertified_stage0_requires_complete_evidence(self):
        base = self._solution("ilp-perf")
        base["primary_mip_gap"] = None
        base["primary_gap_certificate"] = {
            "certified": False,
            "reason": "solver-reported-nonfinite-mip-gap",
        }
        variants = []
        missing_field = dict(base)
        missing_field.pop("primary_mip_gap")
        variants.append((missing_field, "MIP gap field"))
        missing_incumbent = dict(base)
        missing_incumbent["primary_has_incumbent"] = False
        variants.append((missing_incumbent, "lacks an incumbent"))
        missing_cycles = dict(base)
        missing_cycles["primary_incumbent_cycles"] = None
        variants.append((missing_cycles, "finite positive incumbent"))
        missing_certificate = dict(base)
        missing_certificate.pop("primary_gap_certificate")
        variants.append((missing_certificate, "certificate evidence"))
        missing_reason = dict(base)
        missing_reason["primary_gap_certificate"] = {
            "certified": False, "reason": None}
        variants.append((missing_reason, "structured reason"))
        for value, message in variants:
            with self.assertRaisesRegex(RuntimeError, message):
                full_validation._stage0_gap_contract(value, "ilp-perf")

    def test_stage0_rejects_negative_and_infinite_numeric_gaps(self):
        for gap, message in ((-0.1, "negative"),
                             (float("inf"), "non-finite")):
            value = self._solution("ilp-perf", gap=gap)
            with self.assertRaisesRegex(RuntimeError, message):
                full_validation._stage0_gap_contract(value, "ilp-perf")

    def test_uncertified_stage0_fails_when_max_gap_is_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            perf, power, results = self._fixture(temporary)
            value = json.loads(perf.read_text(encoding="utf-8"))
            value["primary_mip_gap"] = None
            value["primary_has_incumbent"] = True
            value["primary_gap_certificate"] = {
                "certified": False,
                "reason": "solver-reported-nonfinite-mip-gap",
            }
            perf.write_text(json.dumps(value), encoding="utf-8")
            self._refresh_replay_identity(results, "ilp-perf", perf)
            report = gate.audit_pair(
                perf, power, results, max_primary_gap=0.05)
            solver_report = gate._solver_only_report(
                temporary, max_primary_gap=0.05)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn(
            "ilp-perf Stage-0 T has no finite gap certificate",
            report["gate"]["failures"])
        self.assertIsNone(report["methods"]["ilp-perf"]["solver"]
                          ["stage0_T"]["gap"])
        self.assertEqual(solver_report["status"], "FAIL")
        self.assertIn(
            "ilp-perf Stage-0 T has no finite gap certificate",
            solver_report["gate"]["failures"])

    def test_protocol_gap_threshold_is_a_separate_optional_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            perf, power, results = self._fixture(temporary)
            value = json.loads(perf.read_text(encoding="utf-8"))
            # Perf's Stage 1/secondary objective is protocol completion.
            value["secondary_mip_gap"] = 0.2
            perf.write_text(json.dumps(value), encoding="utf-8")
            self._refresh_replay_identity(results, "ilp-perf", perf)
            report = gate.audit_pair(
                perf, power, results, max_protocol_gap=0.05)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("protocol-stage gap" in reason
                            for reason in report["gate"]["failures"]))

    def test_surrogate_requires_protocol_field_and_ignores_floating_raw_T(self):
        value = self._solution(
            "ilp-perf", primary=45.0, raw_t=45000000.0, gap=0.1,
            protocol_cycles=104000.0)
        metric = gate._surrogate_metric(value, "ilp-perf")
        self.assertEqual(metric["field"], "performance_proxy_cycles")
        self.assertEqual(metric["cycles"], 104000.0)
        del value["performance_proxy_cycles"]
        with self.assertRaises(gate.RegressionFailure):
            gate._surrogate_metric(value, "ilp-perf")

    def test_power_cap_uses_protocol_cycles_not_predicted_makespan(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "power.json"
            value = self._solution(
                "ilp-power", raw_t=45000000.0,
                protocol_cycles=110000.0, protocol_cap=120000.0)
            path.write_text(json.dumps(value), encoding="utf-8")
            # The 45M legacy T exceeds 120k, but that is dimensionally
            # irrelevant and must not fail the protocol cap.
            full_validation._validate_solution(path, "ilp-power")
            value["performance_proxy_cycles"] = 120001.0
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                full_validation._validate_solution(path, "ilp-power")

    def test_protocol_profile_and_event_metadata_are_mandatory(self):
        value = self._solution("ilp-perf")
        del value["protocol_model"]
        with self.assertRaises(gate.RegressionFailure):
            gate._surrogate_metric(value, "ilp-perf")
        value = self._solution("ilp-perf")
        # protocol_model itself is the profile and its event_count is the
        # compatible event metadata; deleting the count must fail closed.
        del value["protocol_model"]["event_count"]
        with self.assertRaises(gate.RegressionFailure):
            gate._surrogate_metric(value, "ilp-perf")

    def test_solver_report_separates_stage0_and_protocol_gaps(self):
        value = self._solution(
            "ilp-perf", gap=0.19, secondary_gap=0.007,
            protocol_gap=0.009)
        value["analytical_task_makespan_cycles"] = 45000000.0
        metrics = gate._solver_metrics(value, "ilp-perf")
        self.assertEqual(metrics["solver"]["stage0_T"]["gap"], 0.19)
        self.assertEqual(
            metrics["solver"]["protocol_stage"]["gap"], 0.007)
        self.assertEqual(
            metrics["surrogate"]["stage0_T_diagnostic"]["cycles"],
            45000000.0)
        self.assertEqual(
            metrics["surrogate"]["stage0_T_diagnostic"]["field"],
            "analytical_task_makespan_cycles")
        self.assertEqual(
            metrics["surrogate"]["protocol_latency"]["cycles"], 100.0)

    def test_power_accepts_legacy_nonfinite_optional_tie_break_gap(self):
        """E32-style Infinity is an uncertified tie-break, not a failure."""
        value = self._solution(
            "ilp-power", secondary_gap=1.0, protocol_cycles=1135752.183)
        value.pop("tertiary_gap_certificate")
        value.pop("tertiary_has_incumbent")
        value["tertiary_mip_gap"] = float("inf")
        contract = full_validation._protocol_surrogate_contract(
            value, "ilp-power")
        self.assertEqual(contract["operating_objective_kind"],
                         "noc-dynamic-energy")
        self.assertEqual(contract["operating_gap"], 1.0)
        self.assertFalse(contract["operating_gap_certificate"]
                         ["is_protocol_certificate"])
        tie_break = contract["tie_break"]
        self.assertTrue(tie_break["has_incumbent"])
        self.assertIsNone(tie_break["gap"])
        self.assertFalse(tie_break["gap_certified"])
        self.assertEqual(tie_break["gap_reason"],
                         "legacy-nonfinite-tertiary-mip-gap")
        metrics = gate._solver_metrics(value, "ilp-power")
        self.assertEqual(metrics["solver"]["operating_objective_stage"]
                         ["kind"], "noc-dynamic-energy")
        self.assertFalse(metrics["solver"]["protocol_stage"]
                         ["is_protocol_certificate"])
        self.assertIsNone(metrics["solver"]["protocol_stage"]["gap"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy_e32_power.json"
            # Python's legacy encoder emits the exact non-standard Infinity
            # token present in the retained E32 temporary artifact.
            path.write_text(json.dumps(value), encoding="utf-8")
            full_validation._validate_solution(path, "ilp-power")

    def test_power_accepts_new_null_optional_tie_break_certificate(self):
        value = self._solution("ilp-power", secondary_gap=0.25)
        value["tertiary_mip_gap"] = None
        value["tertiary_gap_certificate"] = {
            "certified": False,
            "reason": "solver-reported-nonfinite-mip-gap",
        }
        contract = full_validation._protocol_surrogate_contract(
            value, "ilp-power")
        self.assertIsNone(contract["tie_break"]["gap"])
        self.assertEqual(
            contract["tie_break"]["gap_reason"],
            "solver-reported-nonfinite-mip-gap")

    def test_uncertified_operating_incumbents_are_accepted(self):
        for method in ("ilp-perf", "ilp-power"):
            value = self._solution(method)
            value["secondary_mip_gap"] = None
            value["secondary_has_incumbent"] = True
            value["secondary_gap_certificate"] = {
                "certified": False,
                "reason": "solver-reported-nonfinite-mip-gap",
            }
            contract = full_validation._protocol_surrogate_contract(
                value, method)
            self.assertIsNone(contract["operating_gap"])
            self.assertFalse(contract["operating_gap_certificate"]
                             ["certified"])
            self.assertTrue(contract["operating_has_incumbent"])

    def test_uncertified_operating_stage_requires_full_evidence(self):
        base = self._solution("ilp-perf")
        base["secondary_mip_gap"] = None
        base["secondary_gap_certificate"] = {
            "certified": False,
            "reason": "solver-reported-nonfinite-mip-gap",
        }
        variants = []
        missing_incumbent = dict(base)
        missing_incumbent["secondary_has_incumbent"] = False
        variants.append((missing_incumbent, "lacks an incumbent"))
        missing_objective = dict(base)
        missing_objective["secondary_objective_value"] = None
        variants.append((missing_objective, "lacks a finite objective"))
        missing_reason = dict(base)
        missing_reason["secondary_gap_certificate"] = {
            "certified": False, "reason": None}
        variants.append((missing_reason, "lacks a structured reason"))
        for value, message in variants:
            with self.assertRaisesRegex(RuntimeError, message):
                full_validation._protocol_surrogate_contract(
                    value, "ilp-perf")

    def test_perf_operating_gap_nonfinite_and_negative_are_distinct(self):
        value = self._solution("ilp-perf")
        value.pop("secondary_gap_certificate")
        value["secondary_mip_gap"] = float("inf")
        with self.assertRaisesRegex(RuntimeError, "non-finite protocol"):
            full_validation._protocol_surrogate_contract(value, "ilp-perf")
        value["secondary_mip_gap"] = -0.1
        with self.assertRaisesRegex(RuntimeError, "negative protocol"):
            full_validation._protocol_surrogate_contract(value, "ilp-perf")

    def test_power_rejects_negative_optional_tie_break_gap(self):
        value = self._solution("ilp-power")
        value["tertiary_mip_gap"] = -0.1
        with self.assertRaisesRegex(RuntimeError,
                                    "negative tertiary tie-break gap"):
            full_validation._protocol_surrogate_contract(value, "ilp-power")

    def test_measured_cycles_never_come_from_solution_surrogate(self):
        with tempfile.TemporaryDirectory() as temporary:
            perf, power, results = self._fixture(
                temporary, perf_cycles=321, power_cycles=333)
            report = gate.audit_pair(perf, power, results)
        measured = report["methods"]["ilp-perf"]["measured"]
        self.assertEqual(measured["cycles"], 321)
        self.assertEqual(
            measured["cycle_metric"],
            "LEGOSim synchronized_makespan_cycles")
        self.assertNotEqual(
            measured["cycles"],
            report["methods"]["ilp-perf"]["surrogate"][
                "protocol_latency"]["cycles"])

    def test_audit_rejects_stale_replay_after_solution_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            perf, power, results = self._fixture(temporary)
            value = json.loads(perf.read_text(encoding="utf-8"))
            value["primary_wall_s"] = 123.0
            perf.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(gate.RegressionFailure):
                gate.audit_pair(perf, power, results)

    def test_full_driver_resume_validation_rejects_stale_solution(self):
        with tempfile.TemporaryDirectory() as temporary:
            perf, power, results = self._fixture(temporary)
            manifest = json.loads(results.read_text(encoding="utf-8"))
            item = {
                "name": "a1550-test",
                "results_json": str(results),
                "expected_rows": 2,
                "expected_layout_keys": [
                    [gate.CASE_ID, "ilp-perf"],
                    [gate.CASE_ID, "ilp-power"],
                ],
                "expected_solutions": [{
                    "case_id": gate.CASE_ID,
                    "method": "ilp-perf",
                    "solution_path": str(perf.resolve()),
                }, {
                    "case_id": gate.CASE_ID,
                    "method": "ilp-power",
                    "solution_path": str(power.resolve()),
                }],
                "expected_params_path": manifest["params_path"],
            }
            full_validation._validate_replay_result(item)
            value = json.loads(power.read_text(encoding="utf-8"))
            value["primary_wall_s"] = 456.0
            power.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                full_validation._validate_replay_result(item)

    def test_shared_bom_is_reported_once_and_matches_legosim_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            perf, power, results = self._fixture(temporary)
            report = gate.audit_pair(perf, power, results)
        bom = report["shared_bom"]
        self.assertEqual(bom["type_counts"], {
            "CPU": 1, "GPU-L": 1, "GPU-S": 1})
        self.assertEqual(bom["cpu_plus_gpu_peak_power_W"], 455.0)
        self.assertIn("not LEGOSim dynamic power", bom["power_semantics"])
        for method in gate.METHODS:
            self.assertEqual(report["methods"][method]["measured"][
                "n_gpu_workers"], 2)

    def test_strict_replay_rejects_extra_baseline_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            perf, power, results = self._fixture(temporary)
            value = json.loads(results.read_text(encoding="utf-8"))
            value["layouts"].append(self._row("floorplet", 110, 0.7))
            results.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(gate.RegressionFailure):
                gate.audit_pair(perf, power, results)

    def test_replay_command_contains_exactly_two_layouts(self):
        args = gate._parse_args([
            "--out-root", "out/test_gate",
            "--system-python", sys.executable,
        ])
        out_root = gate._resolve(args.out_root)
        command = gate._replay_command(
            args, out_root, Path("params.json"),
            gate._default_replay_dir(out_root))
        self.assertEqual(command.count("--layout"), 2)
        joined = "\n".join(command)
        self.assertIn("area_1550:ilp-perf:", joined)
        self.assertIn("area_1550:ilp-power:", joined)
        self.assertNotIn("matched-random", joined)
        self.assertNotIn("floorplet", joined)
        self.assertNotIn(":tap:", joined)

    def test_solve_command_selects_only_canonical_a1550(self):
        args = gate._parse_args([
            "--out-root", "out/test_gate",
            "--system-python", sys.executable,
        ])
        command = gate._solve_command(args, gate._resolve(args.out_root))
        joined = "\n".join(command)
        self.assertIn("--phase\nsolve-only", joined)
        self.assertIn("--case\nbudget:area_1550", joined)

    def test_dry_run_has_two_replays_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary) / "must_not_exist"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = gate.main([
                    "--dry-run", "--out-root", str(out_root),
                    "--system-python", sys.executable,
                ])
            plan = json.loads(stream.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(plan["replay_layout_count"], 2)
        self.assertEqual(plan["replay_methods"], list(gate.METHODS))
        self.assertFalse(out_root.exists())


if __name__ == "__main__":
    unittest.main()
