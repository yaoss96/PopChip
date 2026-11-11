#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused contract tests for run_scaling_qap_experiment.py."""

from __future__ import print_function

import argparse
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import run_scaling_qap_experiment as experiment


class ScalingQapContractTests(unittest.TestCase):

    @staticmethod
    def _args():
        return argparse.Namespace(
            time_limit=30.0, mipgap=0.01, seed=7, threads=1,
            mip_focus=1)

    @staticmethod
    def _valid_solution():
        return {
            "optimization_mode": "ilp-perf",
            "objective_contract": {
                "active_objective_count": 1,
                "active_objective": "min T",
                "joint_selection_placement_scheduling": True,
                "protocol_completion_in_T": True,
                "protocol_used_as_separate_objective": False,
                "legacy_task_schedule_role": "hard-feasibility-only",
                "T_definition": (
                    "fixed-order replay max-plus completion plus Mesh "
                    "capacity epigraph"),
                "energy_or_byte_hop_tie_break": False,
            },
            "protocol_model": {
                "kind": "fixed-order-max-plus-event-DAG",
                "included_in_T": True,
                "used_as_separate_objective": False,
            },
            "performance_cap_enforced": False,
            "secondary_objective": None,
            "tertiary_objective": None,
            "quaternary_objective": None,
            "cpu_anchor_policy": "free",
            "solver_configuration": {
                "gurobi_mip_focus": 1,
                "changes_objective": False,
            },
        }

    def test_heterogeneous_envelope_keeps_gpu_selection_joint(self):
        budgets, opts, tier = experiment._full_run_contract(
            8, "heterogeneous-main", self._args())
        self.assertEqual(1440.0, budgets["a_max"])
        self.assertEqual(1120.0, budgets["p_max"])
        self.assertEqual({"CPU": 1}, budgets["nmin"])
        self.assertEqual({"CPU": 1}, budgets["nmax"])
        self.assertEqual(experiment.SCALING_BUDGET_POLICY, tier["policy"])
        self.assertEqual(180.0, tier["area_mm2_per_expert"])
        self.assertEqual(140.0, tier["power_W_per_expert"])
        self.assertIsNone(tier["target_bom"])
        self.assertNotIn("gpu_l", tier)
        self.assertNotIn("gpu_s", tier)
        self.assertNotIn("GPU-L", budgets["nmin"])
        self.assertNotIn("GPU-S", budgets["nmax"])
        self.assertNotIn("p_col", budgets)
        self.assertNotIn("e_bar", budgets)
        self.assertEqual("ilp-perf", opts["mode"])
        self.assertEqual(1, opts["mipfocus"])
        self.assertEqual("free", opts["cpu_anchor"])

    def test_heterogeneous_caps_follow_one_target_bom_independent_rule(self):
        expected = {
            8: (1440.0, 1120.0),
            16: (2880.0, 2240.0),
            32: (5760.0, 4480.0),
            64: (11520.0, 8960.0),
            128: (23040.0, 17920.0),
        }
        for experts, caps in sorted(expected.items()):
            with self.subTest(experts=experts):
                budgets, _, tier = experiment._full_run_contract(
                    experts, "heterogeneous-main", self._args())
                self.assertEqual(caps, (budgets["a_max"],
                                        budgets["p_max"]))
                self.assertIsNone(tier["target_bom"])
                self.assertEqual({"CPU"}, set(budgets["nmin"]))
                self.assertEqual({"CPU"}, set(budgets["nmax"]))

    def test_resource_only_gpu_l_ceilings_are_not_a_fixed_quarter_mix(self):
        # q=1 needs E GPUs.  Starting from CPU + E GPU-S, an L upgrade
        # consumes another 290 mm2 and 210 W.  These are resource ceilings,
        # not selected counts: footprint and min-T can make the MILP use less.
        expected_upper_bounds = {8: 1, 16: 3, 32: 7, 64: 14, 128: 30}
        for experts, expected_upper in sorted(expected_upper_bounds.items()):
            budgets, _, _ = experiment._full_run_contract(
                experts, "heterogeneous-main", self._args())
            all_s_area = 90.0 + 110.0 * experts
            all_s_power = 65.0 + 90.0 * experts
            area_upper = int((budgets["a_max"] - all_s_area) // 290.0)
            power_upper = int((budgets["p_max"] - all_s_power) // 210.0)
            self.assertEqual(expected_upper, min(area_upper, power_upper))
        self.assertNotEqual(
            [experts // 4 for experts in sorted(expected_upper_bounds)],
            [expected_upper_bounds[experts]
             for experts in sorted(expected_upper_bounds)])

    def test_homogeneous_envelope_is_budget_not_power_mode(self):
        budgets, opts, tier = experiment._full_run_contract(
            8, "homogeneous-control", self._args())
        self.assertEqual(970.0, budgets["a_max"])
        self.assertEqual(785.0, budgets["p_max"])
        self.assertEqual(experiment.HOMOGENEOUS_CONTROL_POLICY,
                         tier["policy"])
        self.assertIsNone(tier["target_bom"])
        self.assertIsNone(tier["area_mm2_per_expert"])
        self.assertIsNone(tier["power_W_per_expert"])
        self.assertEqual("ilp-perf", opts["mode"])
        forbidden = {
            "secondary_time", "secondary_mipgap", "certified_t_ref",
            "performance_cap", "latency_relax", "e_bar", "p_col",
        }
        self.assertFalse(forbidden.intersection(opts))

    def test_perf_solution_contract_accepts_exact_min_t_contract(self):
        self.assertTrue(experiment._validate_perf_solution_contract(
            self._valid_solution()))

    def test_perf_solution_contract_rejects_modes_caps_and_tie_breaks(self):
        mutations = [
            ("optimization_mode", "ilp-power"),
            ("performance_cap_enforced", True),
            ("cpu_anchor_policy", "corner"),
        ]
        for key, value in mutations:
            sol = self._valid_solution()
            sol[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment._validate_perf_solution_contract(sol)

        sol = self._valid_solution()
        sol["objective_contract"]["energy_or_byte_hop_tie_break"] = True
        with self.assertRaises(ValueError):
            experiment._validate_perf_solution_contract(sol)

        sol = self._valid_solution()
        sol["objective_contract"]["protocol_completion_in_T"] = False
        with self.assertRaises(ValueError):
            experiment._validate_perf_solution_contract(sol)

        sol = self._valid_solution()
        sol["objective_contract"]["legacy_task_schedule_role"] = (
            "optimized-task-makespan")
        with self.assertRaises(ValueError):
            experiment._validate_perf_solution_contract(sol)

        sol = self._valid_solution()
        sol["protocol_model"]["used_as_separate_objective"] = True
        with self.assertRaises(ValueError):
            experiment._validate_perf_solution_contract(sol)

        sol = self._valid_solution()
        sol["solver_configuration"]["gurobi_mip_focus"] = 0
        with self.assertRaises(ValueError):
            experiment._validate_perf_solution_contract(sol)

    def test_qap_scope_is_explicitly_restricted_representation_ablation(self):
        self.assertIn("fixed_block", experiment.QAP_RESULT_KIND)
        self.assertIn("representation ablation", experiment.QAP_SCOPE["role"])
        excluded = " ".join(experiment.QAP_SCOPE["excluded"])
        self.assertIn("non-preemptive task scheduling", excluded)
        self.assertIn("fixed-order replay protocol", excluded)
        self.assertIn("directed Mesh routing", excluded)
        self.assertIn("chiplet type/BOM selection", excluded)
        self.assertEqual(
            "min traffic-weighted Manhattan byte-hops",
            experiment.QAP_OBJECTIVE)

    def test_reuse_contract_rejects_legacy_rows(self):
        with self.assertRaises(ValueError):
            experiment._validate_result_record_contract(
                {"kind": "full_compact", "optimization_mode":
                 "certified-latency"}, "scaling")
        with self.assertRaises(ValueError):
            experiment._validate_result_record_contract(
                {"kind": "qap_ablation"}, "qap")

        qap = {
            "contract_version": experiment.EXPERIMENT_CONTRACT_VERSION,
            "kind": experiment.QAP_RESULT_KIND,
            "comparison_role": experiment.QAP_SCOPE["role"],
            "objective": experiment.QAP_OBJECTIVE,
            "mip_focus": 1,
        }
        self.assertTrue(experiment._validate_result_record_contract(qap, "qap"))

        scaling = {
            "contract_version": experiment.EXPERIMENT_CONTRACT_VERSION,
            "scaling_budget_contract_version": (
                experiment.SCALING_BUDGET_CONTRACT_VERSION),
            "kind": experiment.PERF_RESULT_KIND,
            "scenario": "heterogeneous-main",
            "optimization_mode": "ilp-perf",
            "active_objective": "min T",
            "T_definition": (
                "fixed-order replay max-plus completion plus Mesh capacity "
                "epigraph"),
            "legacy_task_schedule_role": "hard-feasibility-only",
            "protocol_completion_in_T": True,
            "protocol_used_as_separate_objective": False,
            "mip_focus": 1,
            "cpu_anchor_policy": "free",
            "gpu_counts_fixed": False,
            "budget_policy": experiment.SCALING_BUDGET_POLICY,
            "budget_area_mm2_per_expert": 180.0,
            "budget_power_W_per_expert": 140.0,
            "budget_target_bom": None,
            "experts": 8,
            "budgets": {
                "a_max": 1440.0, "p_max": 1120.0,
                "nmin": {"CPU": 1}, "nmax": {"CPU": 1},
            },
        }
        self.assertTrue(experiment._validate_result_record_contract(
            scaling, "scaling"))
        scaling["legacy_task_schedule_role"] = "objective"
        with self.assertRaises(ValueError):
            experiment._validate_result_record_contract(scaling, "scaling")

    def test_reuse_contract_rejects_old_or_bom_encoded_scaling_budgets(self):
        row = {
            "contract_version": experiment.EXPERIMENT_CONTRACT_VERSION,
            "scaling_budget_contract_version": (
                experiment.SCALING_BUDGET_CONTRACT_VERSION),
            "kind": experiment.PERF_RESULT_KIND,
            "scenario": "heterogeneous-main",
            "optimization_mode": "ilp-perf",
            "active_objective": "min T",
            "T_definition": (
                "fixed-order replay max-plus completion plus Mesh capacity "
                "epigraph"),
            "legacy_task_schedule_role": "hard-feasibility-only",
            "protocol_completion_in_T": True,
            "protocol_used_as_separate_objective": False,
            "mip_focus": 1,
            "cpu_anchor_policy": "free",
            "gpu_counts_fixed": False,
            "budget_policy": experiment.SCALING_BUDGET_POLICY,
            "budget_area_mm2_per_expert": 180.0,
            "budget_power_W_per_expert": 140.0,
            "budget_target_bom": None,
            "experts": 8,
            "budgets": {
                "a_max": 1440.0, "p_max": 1120.0,
                "nmin": {"CPU": 1}, "nmax": {"CPU": 1},
            },
        }
        stale = dict(row)
        stale["scaling_budget_contract_version"] = "old"
        with self.assertRaises(ValueError):
            experiment._validate_result_record_contract(stale, "scaling")
        encoded = dict(row)
        encoded["budgets"] = dict(row["budgets"])
        encoded["budgets"]["nmax"] = {"CPU": 1, "GPU-L": 2}
        with self.assertRaises(ValueError):
            experiment._validate_result_record_contract(encoded, "scaling")

    def test_readme_never_presents_qap_as_full_competitor(self):
        directory = tempfile.mkdtemp(prefix="qap_contract_test_")
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        metadata = {
            "generated_at": "2026-08-02T00:00:00+08:00",
            "time_limit_s": 30.0,
            "mipgap_target": 0.01,
        }
        scaling = [{
            "scenario": "heterogeneous-main", "experts": 8, "mesh": 6,
            "feasible": True, "first_feasible_s": 1.0,
            "target_gap_time_s": None, "final_gap": 0.2,
            "status": "TIME_LIMIT",
        }]
        qap = [{
            "formulation": "separable", "experts": 4, "feasible": True,
            "model_variables": 1, "model_constraints": 1,
            "total_worker_wall_s": 0.1, "final_gap": 0.0,
            "status": "OPTIMAL",
        }]
        experiment._write_readme(directory, metadata, scaling, qap)
        with open(os.path.join(directory, "README.md"), "r") as handle:
            text_value = handle.read()
        self.assertIn("Its sole objective is `min T`",
                      text_value)
        self.assertIn("legacy C11--C13 task horizon `T_task`", text_value)
        self.assertIn("not full-system competitors", text_value)
        self.assertIn("directed Mesh routing/link-capacity", text_value)
        self.assertNotIn("certified-latency", text_value)

    def test_parser_fixes_mip_focus_to_one(self):
        parser = experiment._parser()
        self.assertEqual(1, parser.parse_args([]).mip_focus)
        self.assertEqual(1, parser.parse_args(["--mip-focus", "1"]).mip_focus)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--mip-focus", "0"])


if __name__ == "__main__":
    unittest.main()
