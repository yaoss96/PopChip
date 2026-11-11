#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_full_constraint_revalidation as driver


class FullConstraintPerfOnlyProtocolTest(unittest.TestCase):
    @staticmethod
    def _sites(gpu_coordinate=(0, 1, 0), gpu_type="GPU-L",
               gpu_tasks=None):
        if gpu_tasks is None:
            gpu_tasks = ["e(0)"]
        return [
            {"l": 0, "i": 0, "j": 0, "type": "CPU",
             "tasks": ["t_gate", "t_norm"]},
            {"l": gpu_coordinate[0], "i": gpu_coordinate[1],
             "j": gpu_coordinate[2], "type": gpu_type,
             "tasks": list(gpu_tasks)},
        ]

    @classmethod
    def _solution(cls, primary_cycles=1000.0,
                  analytical_task_cycles=93460352.0,
                  certified=True, sites=None):
        if sites is None:
            sites = cls._sites()
        task_site = {}
        for site in sites:
            anchor = [site["l"], site["i"], site["j"]]
            for task in site["tasks"]:
                task_site[task] = list(anchor)
        selected_counts = {}
        for site in sites:
            kind = site["type"]
            selected_counts[kind] = selected_counts.get(kind, 0) + 1
        return {
            "optimization_mode": "ilp-perf",
            "primary_objective": "minimize joint MILP makespan T",
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
            "primary_has_incumbent": True,
            "primary_incumbent_cycles": float(primary_cycles),
            "primary_objective_bound_cycles": (
                float(primary_cycles) if certified else None),
            "primary_mip_gap": 0.0 if certified else None,
            "primary_gap_certificate": ({
                "certified": True, "reason": None,
            } if certified else {
                "certified": False,
                "reason": "solver-reported-nonfinite-mip-gap",
            }),
            "analytical_task_makespan_cycles": float(
                analytical_task_cycles),
            "legacy_task_horizon_cycles": float(analytical_task_cycles),
            "predicted_makespan_cycles": float(primary_cycles),
            "performance_proxy_cycles": float(primary_cycles),
            "tightened_performance_proxy_cycles": float(primary_cycles),
            "protocol_makespan_cycles": float(primary_cycles),
            "protocol_model": {
                "kind": "fixed-order-max-plus-event-DAG",
                "event_count": 16,
                "phases": ["dispatch", "compute", "combine"],
                "completion_cycles": float(primary_cycles),
                "diagnostic_only": False,
                "used_as_objective": True,
                "included_in_T": True,
                "used_as_separate_objective": False,
            },
            "noc_power_proxy": {
                "diagnostic_only": True,
                "used_as_objective": False,
            },
            "secondary_wall_s": 0.0,
            "secondary_status": None,
            "secondary_mip_gap": None,
            "secondary_objective": None,
            "secondary_has_incumbent": False,
            "secondary_objective_value": None,
            "secondary_gap_certificate": {
                "certified": False, "reason": "stage-not-run"},
            "secondary_time_limit_s": None,
            "tertiary_wall_s": 0.0,
            "tertiary_status": None,
            "tertiary_mip_gap": None,
            "tertiary_objective": None,
            "tertiary_has_incumbent": False,
            "tertiary_objective_value": None,
            "tertiary_gap_certificate": {
                "certified": False, "reason": "stage-not-run"},
            "tertiary_time_limit_s": None,
            "quaternary_wall_s": 0.0,
            "quaternary_status": None,
            "quaternary_mip_gap": None,
            "quaternary_objective": None,
            "quaternary_has_incumbent": False,
            "quaternary_objective_value": None,
            "quaternary_gap_certificate": {
                "certified": False, "reason": "stage-not-run"},
            "performance_reference_cycles": None,
            "performance_relaxation": None,
            "performance_cap_cycles": None,
            "performance_cap_enforced": False,
            "comparison_reference": {
                "kind": None, "fixed_bom": None,
                "fixed_complete_bundles": False},
            "internal_bundle_refinement": {
                "enabled": False, "coordinates_fixed": False},
            "reference_coordinates_warm_started": False,
            "cpu_anchor_policy": "free",
            "expert_slots": {"GPU-L": 1, "GPU-S": 1},
            "routing": {
                "kind": (
                    "phase-separated-source-aggregated-splittable-packet-flow"),
                "maximum_flow_residual_KiB": 0.0,
                "maximum_link_utilization": 0.5,
            },
            "task_schedule": {task: {"start": 0.0}
                              for task in task_site},
            "model_build_s": 0.5,
            "formulation_build_s": 0.3,
            "stage0_total_s": 0.2,
            "requested_total_time_limit_s": 45.0,
            "primary_solver_time_limit_s": 44.5,
            "primary_wall_s": 0.4,
            "anytime_trace": [{
                "wall_s": 0.1,
                "incumbent_cycles": float(primary_cycles),
            }],
            "time_to_first_feasible_s": 0.6,
            "constructive_search_wall_s": 0.2,
            "stage0_feasible_start": {
                "attempted": True,
                "accepted_for_submission": True,
                "selection_policy": "performance-first protocol start",
                "type_counts": {"CPU": 1, "GPU-L": 1},
                "protocol_T_start_cycles": float(primary_cycles) + 200.0,
                "legacy_task_T_start_cycles": float(
                    analytical_task_cycles),
                "protocol_placement_refinement": {"wall_s": 0.2},
            },
            "bundle_signature": "sha256:synthetic-complete-bundles",
            "sites": sites,
            "task_site": task_site,
            "grid": {"L": 1, "X": 6, "Y": 6},
            "budgets": {"a_max": 1550.0, "p_max": 1625.0},
            "total_area_mm2": 500.0,
            "peak_power_W": 400.0,
            "selected_type_counts": selected_counts,
            "resource_selection": {
                "count_definition": "N_c=sum_p x[c,p]",
                "joint_with_mapping_and_placement": True,
                "fixed_bom": False,
                "fixed_complete_bundles": False,
                "selected_counts": selected_counts,
                "area_budget_mm2": 1550.0,
                "area_used_mm2": 500.0,
                "area_slack_mm2": 1050.0,
                "peak_power_budget_W": 1625.0,
                "peak_power_used_W": 400.0,
                "peak_power_slack_W": 1225.0,
            },
        }

    @staticmethod
    def _case(key="budget:area_1550", suite="budget",
              case_id="area_1550", params=None, source=None):
        case = {
            "key": key,
            "suite": suite,
            "case_id": case_id,
            "params": Path(params or "params.json"),
            "source_metadata": Path(source or "source.json"),
            "grid": (1, 6, 6),
            "budgets": {"a_max": 1550.0, "p_max": 1625.0},
            "mipgap": 0.01,
        }
        if suite == "scaling":
            case.update({
                "scaling_budget_contract_version":
                    driver.SCALING_BUDGET_CONTRACT_VERSION,
                "budget_policy": driver.SCALING_BUDGET_POLICY,
                "budget_target_bom": None,
            })
        return case

    @staticmethod
    def _write_json(path, value):
        Path(path).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    def test_production_method_sets_total_47_rows(self):
        self.assertEqual(
            driver.BUDGET_METHODS,
            ("ilp-perf", "matched-random", "floorplet", "tap"))
        self.assertEqual(
            driver.SCALING_METHODS,
            ("ilp-perf", "floorplet", "tap"))
        self.assertEqual(
            8 * len(driver.BUDGET_METHODS) +
            5 * len(driver.SCALING_METHODS), 47)
        self.assertEqual(driver.PROTOCOL["expected_full_replay_rows"], 47)
        self.assertEqual(
            driver.PROTOCOL["ilp_perf_objective"], "sole-joint-min-T")
        self.assertEqual(
            driver.PROTOCOL["protocol_completion_role"],
            "sole-primary-objective")
        self.assertEqual(
            driver.PROTOCOL["primary_objective"],
            "minimize joint MILP makespan T")
        self.assertEqual(driver.PROTOCOL["active_objective_count"], 1)
        self.assertEqual(driver.PROTOCOL["active_objective"], "min T")
        self.assertTrue(
            driver.PROTOCOL["joint_selection_placement_scheduling"])
        self.assertFalse(driver.PROTOCOL["gpu_type_counts_fixed"])
        self.assertEqual(
            driver.PROTOCOL["gpu_type_count_definition"],
            "N_c=sum_p x[c,p]")
        self.assertTrue(
            driver.PROTOCOL["package_area_and_peak_power_budgets"])
        self.assertTrue(driver.PROTOCOL["explicit_task_schedule"])
        self.assertTrue(driver.PROTOCOL["mesh_flow_capacity"])
        self.assertTrue(driver.PROTOCOL["protocol_completion_in_T"])
        self.assertTrue(
            driver.PROTOCOL["protocol_completion_used_as_objective"])
        self.assertEqual(
            driver.PROTOCOL["legacy_analytical_task_makespan_role"],
            "hard-feasibility-only")
        self.assertEqual(
            driver.PROTOCOL["stage0_public_T_evidence"],
            "protocol_T_start_cycles")
        self.assertEqual(driver.PROTOCOL["ilp_total_time_limit_s"], 45.0)
        self.assertIn(
            "formulation", driver.PROTOCOL["ilp_budget_scope"])
        self.assertIn("Stage-0", driver.PROTOCOL["ilp_budget_scope"])
        self.assertEqual(
            driver.PROTOCOL["floorplet_time_limit_s"], 45.0)
        self.assertEqual(
            driver.PROTOCOL["tap_wall_budget_s_per_seed"], 15.0)
        self.assertEqual(
            driver.PROTOCOL["tap_total_wall_budget_s"], 45.0)
        self.assertEqual(driver.PROTOCOL["tap_seeds"], [11, 29, 47])
        self.assertEqual(
            driver.PROTOCOL["matched_random_seeds"], list(range(100)))
        self.assertEqual(
            driver.PROTOCOL["matched_random_selection_metric"],
            "public_T_cycles")
        self.assertFalse(
            driver.PROTOCOL["matched_random_legacy_task_makespan_scored"])
        self.assertNotIn("pareto_endpoint_gate", driver.PROTOCOL)

    def test_ilp_command_contains_only_the_joint_min_t_mode(self):
        case = self._case()
        # Historical case metadata may say 30 s; the formal protocol always
        # overrides it with one uniform end-to-end 45 s budget.
        case["source_time_limit_s"] = 30.0
        command = driver._ilp_command(
            case, "ilp-perf", Path("solution.json"),
            Path(sys.executable), Path("out"))
        joined = " ".join(str(value) for value in command)

        self.assertEqual(command[command.index("--mode") + 1], "ilp-perf")
        self.assertEqual(
            command[command.index("--time-limit") + 1], "45.0")
        self.assertNotIn("--placement-from", command)
        self.assertNotIn("--bom-from", command)
        self.assertNotIn("--n-gpu-l-max", command)
        self.assertNotIn("--power-perf-relax", command)
        self.assertNotIn("--perf-tie-relax", command)
        self.assertNotIn("--secondary-time", command)
        self.assertNotIn("--secondary-mipgap", command)
        self.assertNotIn("1.20", joined)
        with self.assertRaisesRegex(ValueError, "invalid ILP method"):
            driver._ilp_command(
                case, "ilp-power", Path("forbidden.json"),
                Path(sys.executable), Path("out"))

    def test_baseline_commands_use_equal_45_second_formal_budgets(self):
        case = self._case()
        prior = driver._prior_command(
            case, Path("out"), Path(sys.executable))
        self.assertEqual(
            prior[prior.index("--floorplet-time-limit") + 1], "45.0")
        self.assertEqual(
            prior[prior.index("--tap-wall-budget") + 1], "15.0")
        self.assertEqual(
            prior[prior.index("--tap-seeds") + 1], "11,29,47")
        self.assertEqual(
            3 * float(prior[prior.index("--tap-wall-budget") + 1]), 45.0)

        matched = driver._matched_command(
            case, Path("out"), Path(sys.executable))
        seed_start = matched.index("--seeds") + 1
        seed_end = matched.index("--out-dir")
        self.assertEqual(
            [int(seed) for seed in matched[seed_start:seed_end]],
            list(range(100)))

    def test_baseline_validation_rejects_old_optimizer_budgets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            floor = root / "floorplet.json"
            tap = root / "tap.json"
            self._write_json(floor, {
                "strict_evaluate": {"feasible": True},
                "metadata": {"time_limit_s": 30.0},
            })
            self._write_json(tap, {
                "strict_evaluate": {"feasible": True},
                "metadata": {"wall_budget_s": 10.0, "seed": 11},
            })
            with self.assertRaisesRegex(RuntimeError, "formal 45 s"):
                driver._validate_solution(floor, "floorplet")
            with self.assertRaisesRegex(RuntimeError, "formal 15 s"):
                driver._validate_solution(tap, "tap")

    def test_matched_validation_requires_100_public_t_scored_candidates(self):
        scoring = {
            "metric": "public_T_cycles",
            "active_objective": "min T",
            "protocol_completion_in_T": True,
            "directed_mesh_capacity_in_T": True,
            "legacy_task_makespan_role": "hard-feasibility-only",
            "legacy_task_makespan_scored": False,
            "energy_or_byte_hop_tie_break": False,
        }
        representative = {
            "feasible": True,
            "seed": 49,
            "public_T_cycles": 1049.0,
            "predicted_makespan_cycles": 1049.0,
            "protocol_completion_cycles": 1049.0,
            "directed_mesh_capacity_bound_cycles": 10.0,
            "baseline": {
                "strict_evaluate": True,
                "representative_selection_metric": "public_T_cycles",
                "public_T_scoring": scoring,
            },
        }
        runs = [{"seed": seed, "public_T_cycles": 1000.0 + seed}
                for seed in range(100)]
        summary = {
            "schema_version": 2,
            "seeds": list(range(100)),
            "candidate_count": 100,
            "all_candidates_scored_with_public_T": True,
            "representative_selection_metric": "public_T_cycles",
            "representative_seed": 49,
            "representative_public_T_cycles": 1049.0,
            "runs": runs,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "representative.json"
            self._write_json(path, representative)
            self._write_json(root / "summary.json", summary)
            driver._validate_solution(path, "matched-random")

            old_metric = dict(representative)
            old_metric["baseline"] = dict(representative["baseline"])
            old_metric["baseline"]["representative_selection_metric"] = \
                "legacy_task_makespan_cycles"
            self._write_json(path, old_metric)
            with self.assertRaisesRegex(RuntimeError, "public min-T"):
                driver._validate_solution(path, "matched-random")

            self._write_json(path, representative)
            incomplete = dict(summary)
            incomplete["candidate_count"] = 99
            self._write_json(root / "summary.json", incomplete)
            with self.assertRaisesRegex(RuntimeError, "100-candidate"):
                driver._validate_solution(path, "matched-random")

    def test_validate_accepts_certified_sole_min_protocol_t(self):
        solution = self._solution(
            primary_cycles=98437.0,
            analytical_task_cycles=93460352.0,
            certified=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            parsed = driver._validate_solution(path, "ilp-perf")

        protocol = driver._protocol_surrogate_contract(
            parsed, "ilp-perf")
        self.assertEqual(protocol["cycles"], 98437.0)
        self.assertEqual(protocol["role"], "sole-primary-objective")
        self.assertTrue(protocol["is_optimization_objective"])
        primary = driver._primary_min_t_contract(parsed, "synthetic")
        self.assertEqual(primary["objective"], "T")
        self.assertTrue(primary["gap_certified"])
        self.assertEqual(primary["incumbent_cycles"], 98437.0)
        contract = driver._joint_min_t_output_contract(
            parsed, "synthetic", primary["incumbent_cycles"])
        self.assertEqual(
            contract["primary_objective"],
            "minimize joint MILP makespan T")
        self.assertEqual(
            contract["public_performance_cycles"]
            ["predicted_makespan_cycles"], 98437.0)
        self.assertEqual(
            contract["feasibility_task_makespan_cycles"], 93460352.0)

    def test_validate_accepts_time_limited_primary_t_incumbent(self):
        solution = self._solution(certified=False)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            parsed = driver._validate_solution(path, "ilp-perf")

        primary = driver._primary_min_t_contract(parsed, "synthetic")
        self.assertFalse(primary["gap_certified"])
        self.assertEqual(
            primary["gap_reason"], "solver-reported-nonfinite-mip-gap")

    def test_joint_min_t_may_change_the_performance_start_bom(self):
        # The tier BOM is only a strong MIP start.  Under a sole min-T
        # objective the joint model may legally trade one large chiplet for
        # more small chiplets when the extra concurrency lowers T.
        sites = self._sites(
            gpu_coordinate=(0, 1, 0), gpu_type="GPU-S",
            gpu_tasks=["e(0)"])
        sites.append({
            "l": 0, "i": 2, "j": 0, "type": "GPU-S",
            "tasks": ["t_attn"],
        })
        solution = self._solution(
            primary_cycles=1000.0, certified=False, sites=sites)
        solution.update({
            "total_area_mm2": 500.0,
            "peak_power_W": 400.0,
            "stage0_feasible_start": {
                "accepted_for_submission": True,
                "selection_policy": "performance-first test start",
                "type_counts": {"CPU": 1, "GPU-L": 1},
                "protocol_T_start_cycles": 1200.0,
                "protocol_placement_refinement": {"wall_s": 0.2},
            },
            "solver_configuration": {"gurobi_mip_focus": 1},
        })
        case = self._case()
        case["performance_start_bom"] = {"CPU": 1, "GPU-L": 1}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            parsed = driver._validate_solution(path, "ilp-perf", case)

        actual_types = [site["type"] for site in parsed["sites"]]
        self.assertEqual(actual_types.count("GPU-L"), 0)
        self.assertEqual(actual_types.count("GPU-S"), 2)

    def test_joint_min_t_validation_rejects_package_budget_violation(self):
        solution = self._solution(certified=False)
        solution.update({
            "total_area_mm2": 1550.0001,
            "peak_power_W": 400.0,
            "stage0_feasible_start": {
                "accepted_for_submission": True,
                "selection_policy": "performance-first test start",
                "type_counts": {"CPU": 1, "GPU-L": 1},
                "protocol_T_start_cycles": 1200.0,
                "protocol_placement_refinement": {"wall_s": 0.2},
            },
            "solver_configuration": {"gurobi_mip_focus": 1},
        })
        case = self._case()
        case["performance_start_bom"] = {"CPU": 1, "GPU-L": 1}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            with self.assertRaisesRegex(RuntimeError, "exceeds a_max"):
                driver._validate_solution(path, "ilp-perf", case)

    def test_fresh_joint_solve_rejects_gpu_count_bounds(self):
        solution = self._solution(certified=False)
        solution["budgets"]["nmax"] = {"GPU-L": 1}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            with self.assertRaisesRegex(RuntimeError, "fixes/bounds GPU counts"):
                driver._validate_solution(path, "ilp-perf")

    def test_fresh_joint_solve_rejects_fixed_bom_reference(self):
        solution = self._solution(certified=False)
        solution["comparison_reference"].update({
            "kind": "same-BOM", "fixed_bom": {"CPU": 1, "GPU-L": 1}})
        solution["resource_selection"]["fixed_bom"] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            with self.assertRaisesRegex(RuntimeError, "not a fresh uncapped"):
                driver._validate_solution(path, "ilp-perf")

    def test_validate_rejects_active_post_t_objective(self):
        solution = self._solution()
        solution.update({
            "secondary_objective": "fixed-order protocol completion",
            "secondary_has_incumbent": True,
            "secondary_objective_value": 100.0,
        })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            with self.assertRaisesRegex(RuntimeError,
                                        "active post-T objective"):
                driver._validate_solution(path, "ilp-perf")

    def test_validate_accepts_feasibility_task_horizon_different_from_t(self):
        solution = self._solution(primary_cycles=1000.0)
        solution["analytical_task_makespan_cycles"] = 99999999.0
        solution["legacy_task_horizon_cycles"] = 99999999.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            parsed = driver._validate_solution(path, "ilp-perf")
        self.assertEqual(parsed["protocol_makespan_cycles"], 1000.0)
        self.assertEqual(
            parsed["analytical_task_makespan_cycles"], 99999999.0)

    def test_validate_rejects_primary_incumbent_not_equal_to_protocol_t(self):
        solution = self._solution(primary_cycles=1000.0)
        solution["protocol_makespan_cycles"] = 999.0
        solution["protocol_model"]["completion_cycles"] = 999.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            with self.assertRaisesRegex(RuntimeError,
                                        "differs from the primary incumbent T"):
                driver._validate_solution(path, "ilp-perf")

    def test_validate_rejects_wrong_objective_contract(self):
        solution = self._solution()
        solution["objective_contract"]["active_objective_count"] = 2
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            with self.assertRaisesRegex(
                    RuntimeError, "active_objective_count"):
                driver._validate_solution(path, "ilp-perf")

    def test_validate_rejects_old_diagnostic_only_protocol_output(self):
        solution = self._solution()
        solution["protocol_model"].update({
            "diagnostic_only": True,
            "used_as_objective": False,
            "included_in_T": False,
        })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old_final8.json"
            self._write_json(path, solution)
            with self.assertRaisesRegex(
                    RuntimeError, "sole T objective"):
                driver._validate_solution(path, "ilp-perf")

    def test_validate_rejects_first_feasible_excluding_model_build(self):
        solution = self._solution()
        solution["time_to_first_feasible_s"] = 0.1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            with self.assertRaisesRegex(
                    RuntimeError, "does not include model build"):
                driver._validate_solution(path, "ilp-perf")

    def test_validate_rejects_old_30_second_ilp_budget(self):
        solution = self._solution()
        solution["requested_total_time_limit_s"] = 30.0
        solution["primary_solver_time_limit_s"] = 29.5
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old_30s.json"
            self._write_json(path, solution)
            with self.assertRaisesRegex(
                    RuntimeError, "formal 45 s total ILP budget"):
                driver._validate_solution(path, "ilp-perf")

    def test_stage0_validation_uses_protocol_t_start(self):
        solution = self._solution(primary_cycles=1000.0, certified=False)
        solution["stage0_feasible_start"].update({
            "protocol_T_start_cycles": 900.0,
            # A large legacy task horizon must not mask the invalid public-T
            # start: T_task is feasibility evidence, not the objective.
            "analytical_T_start_cycles": 99999999.0,
        })
        case = self._case()
        case["performance_start_bom"] = {"CPU": 1, "GPU-L": 1}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            with self.assertRaisesRegex(
                    RuntimeError, "worse than.*protocol-T start"):
                driver._validate_solution(path, "ilp-perf", case)

    def test_validate_rejects_public_metric_not_equal_to_t(self):
        solution = self._solution(primary_cycles=1000.0)
        solution["performance_proxy_cycles"] = 999.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            with self.assertRaisesRegex(
                    RuntimeError,
                    "performance_proxy_cycles differs from the primary"):
                driver._validate_solution(path, "ilp-perf")

    def test_validate_rejects_quaternary_stage(self):
        solution = self._solution()
        solution["quaternary_objective"] = "forbidden tie-break"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ilp_perf.json"
            self._write_json(path, solution)
            with self.assertRaisesRegex(
                    RuntimeError, "active post-T objective.*quaternary"):
                driver._validate_solution(path, "ilp-perf")

    def test_validate_rejects_invalid_primary_bound(self):
        solution = self._solution(primary_cycles=1000.0)
        solution["primary_objective_bound_cycles"] = 1001.0
        with self.assertRaisesRegex(RuntimeError,
                                    "bound exceeds its incumbent"):
            driver._primary_min_t_contract(solution, "synthetic")

    def test_validate_accepts_zero_time_limited_lower_bound(self):
        solution = self._solution(primary_cycles=1000.0, certified=True)
        solution["primary_objective_bound_cycles"] = 0.0
        solution["primary_mip_gap"] = 1.0
        primary = driver._primary_min_t_contract(solution, "synthetic")
        self.assertEqual(primary["bound_cycles"], 0.0)

    def test_solve_launches_exactly_one_ilp_per_case(self):
        cases = ([{"key": "budget:case_%02d" % index}
                  for index in range(8)] +
                 [{"key": "scaling:case_%02d" % index}
                  for index in range(5)])
        rows = [{"key": case["key"], "ilp_perf": {"status": "pending"}}
                for case in cases]
        manifest = {
            "schema_version": 5,
            "status": "planned",
            "cases": rows,
            "cases_by_key": {row["key"]: row for row in rows},
        }
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(driver, "_run_ilp") as run_ilp:
            out_root = Path(temporary)
            driver._solve(
                cases, out_root, Path(sys.executable), manifest, False)
            persisted = json.loads(
                (out_root / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(run_ilp.call_count, 13)
        self.assertEqual(
            [call.args[1] for call in run_ilp.call_args_list],
            ["ilp-perf"] * 13)
        self.assertEqual(persisted["status"], "solve_complete")
        self.assertNotIn("pareto_endpoint_gate_summary", persisted)
        self.assertTrue(all("ilp_power" not in row
                            for row in persisted["cases"]))

    def test_manifest_schema_and_counts_exclude_power(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            params = root / "params.json"
            source = root / "source.json"
            self._write_json(params, {"params": True})
            self._write_json(source, {"source": True})
            cases = [
                self._case(params=params, source=source),
                self._case(
                    key="scaling:e008_m06", suite="scaling",
                    case_id="e008_m06", params=params, source=source),
            ]
            manifest = driver._new_manifest(
                cases, root / "out", Path(sys.executable),
                Path(sys.executable))

        self.assertEqual(manifest["schema_version"], 5)
        self.assertEqual(manifest["expected_ilp_solutions"], 2)
        self.assertEqual(manifest["expected_replay_rows"], 7)
        self.assertEqual(manifest["formal_wall_budgets_s_per_case"], {
            "ilp_perf_total": 45.0,
            "floorplet": 45.0,
            "tap_total": 45.0,
            "tap_per_seed": 15.0,
        })
        self.assertEqual(manifest["matched_random_seed_count"], 100)
        self.assertIn(
            "run_metaheuristic_surrogate_audit.py", manifest["drivers"])
        self.assertNotIn("pareto_endpoint_gate_summary", manifest)
        for row in manifest["cases"]:
            self.assertIn("ilp_perf", row)
            self.assertEqual(
                row["formal_wall_budgets_s"],
                manifest["formal_wall_budgets_s_per_case"])
            self.assertNotIn("ilp_power", row)
            self.assertNotIn("pareto_endpoint_gate", row)
            commands = json.dumps(row.get("ilp_perf", {}).get("command"))
            self.assertNotIn("ilp-power", commands)
            self.assertIn("45.0", commands)
        budget_row = next(
            row for row in manifest["cases"] if row["suite"] == "budget")
        self.assertEqual(budget_row["matched_random_seed_count"], 100)

    def test_load_manifest_rejects_old_final8_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_json(root / "manifest.json", {
                "schema_version": 4,
                "protocol": dict(driver.PROTOCOL),
                "cases": [],
            })
            with self.assertRaisesRegex(
                    RuntimeError, "manifest schema does not match"):
                driver._load_manifest(root, [])

    def test_load_manifest_rejects_changed_formal_wall_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            params = root / "params.json"
            source = root / "source.json"
            self._write_json(params, {"params": True})
            self._write_json(source, {"source": True})
            case = self._case(params=params, source=source)
            manifest = driver._new_manifest(
                [case], root, Path(sys.executable), Path(sys.executable))
            manifest["formal_wall_budgets_s_per_case"][
                "ilp_perf_total"] = 30.0
            driver._save_manifest(root, manifest)
            with self.assertRaisesRegex(
                    RuntimeError, "formal wall budgets do not match"):
                driver._load_manifest(root, [case])

    def test_prior_baselines_keep_perf_bom_and_bundles_but_may_relayout(self):
        case = self._case(
            key="scaling:e008_m06", suite="scaling", case_id="e008_m06")
        perf = self._solution()
        moved_sites = self._sites(gpu_coordinate=(0, 4, 4))
        baseline = {
            "strict_evaluate": {"feasible": True},
            "metadata": {
                "time_limit_s": 45.0,
                "wall_budget_s": 15.0,
                "seed": 11,
            },
            "sites": moved_sites,
            "grid": dict(perf["grid"]),
            "budgets": dict(perf["budgets"]),
        }
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary)
            for method, value in (
                    ("ilp-perf", perf),
                    ("floorplet", baseline),
                    ("tap", baseline)):
                path = driver._solution_path(out_root, case, method)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._write_json(path, value)

            driver._validate_reference_matching(
                case, out_root, driver.SCALING_METHODS)

            changed = dict(baseline)
            changed["sites"] = self._sites(
                gpu_coordinate=(0, 4, 4), gpu_tasks=["e(1)"])
            self._write_json(
                driver._solution_path(out_root, case, "tap"), changed)
            with self.assertRaisesRegex(RuntimeError,
                                        "changed an ILP-Perf task bundle"):
                driver._validate_reference_matching(
                    case, out_root, driver.SCALING_METHODS)


if __name__ == "__main__":
    unittest.main()
