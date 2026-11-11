#!/usr/bin/env python3
"""Small, solver-free tests for the sole-min-T SA/GA candidate scorer."""

import json
import csv
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_metaheuristic_surrogate_audit as audit
import evaluate_legosim_protocol_surrogate as protocol_oracle


def params(expert_slots=2, beta=64.0, legacy_cycles=500000.0):
    return {
        "meta": {"groups": 2, "topk": 2, "group_pi": [0.5, 0.5]},
        "library": {
            "CPU": {"w": 1, "h": 1, "area": 1.0, "p_peak": 1.0,
                    "mem": 100.0, "expert_slots": 0},
            "GPU-S": {"w": 1, "h": 1, "area": 1.0, "p_peak": 2.0,
                      "mem": 100.0, "expert_slots": expert_slots},
        },
        "tasks": {
            "control": {"allowed": ["CPU"], "mem_mb": 1.0,
                        "cyc": {"CPU": legacy_cycles},
                        "eta": {"CPU": 0.0}},
            "e(0)": {"allowed": ["GPU-S"], "mem_mb": 1.0,
                     "cyc": {"GPU-S": legacy_cycles},
                     "eta": {"GPU-S": 0.0}},
            "e(1)": {"allowed": ["GPU-S"], "mem_mb": 1.0,
                     "cyc": {"GPU-S": legacy_cycles},
                     "eta": {"GPU-S": 0.0}},
        },
        "expert_tasks": ["e(0)", "e(1)"],
        "edges": [
            {"src": "control", "dst": "e(0)", "vol": 0.0},
            {"src": "control", "dst": "e(1)", "vol": 0.0},
        ],
        "interconnect": {"tau_h": 0.0, "tau_v": 0.0, "t0": 0.0,
                         "beta": beta, "eps_h": 0.0, "eps_v": 0.0},
    }


def solution(shared=True):
    sites = [
        {"l": 0, "i": 1, "j": 1, "type": "CPU",
         "tasks": ["control"]},
        {"l": 0, "i": 0, "j": 0, "type": "GPU-S",
         "tasks": ["e(0)"] + (["e(1)"] if shared else [])},
    ]
    task_site = {"control": [0, 1, 1], "e(0)": [0, 0, 0]}
    if shared:
        task_site["e(1)"] = [0, 0, 0]
    else:
        sites.append({"l": 0, "i": 0, "j": 2, "type": "GPU-S",
                      "tasks": ["e(1)"]})
        task_site["e(1)"] = [0, 0, 2]
    return {
        "sites": sites,
        "task_site": task_site,
        "grid": {"L": 1, "X": 2, "Y": 3},
        "noc_power_proxy": {
            "traffic_kind": "deterministic-dispatch",
            "diagnostic_only": True,
            "used_as_objective": False,
            "dispatch_tokens_per_source": 1,
            "dispatch_token_bytes_per_phase": 32.0,
        },
        "protocol_model": {
            "kind": "fixed-order-max-plus-event-DAG",
            "included_in_T": True,
            "flit_bytes": 32.0,
            "hop_cycles": 2.0,
            "sync_overhead_cycles": 10.0,
            "compute_cycles_by_expert_type": {
                "e(0)": {"GPU-S": 5.0},
                "e(1)": {"GPU-S": 7.0},
            },
        },
        "optimization_mode": "ilp-perf",
        "objective_contract": {"active_objective_count": 1,
                               "active_objective": "min T",
                               "protocol_completion_in_T": True,
                               "energy_or_byte_hop_tie_break": False},
        "performance_cap_enforced": False,
        "primary_incumbent_cycles": 120.0,
        "primary_objective_bound_cycles": 100.0,
        "primary_mip_gap": 1.0 / 6.0,
    }


def model(expert_slots=2, beta=64.0, shared=True,
          legacy_cycles=500000.0):
    return audit.CaseModel(
        "tiny", params(expert_slots, beta, legacy_cycles), solution(shared),
        {"a_max": 3.0, "p_max": 5.0}, {"L": 1, "X": 2, "Y": 3})


def deterministic_dispatch_model():
    return model(expert_slots=1, beta=64.0, shared=False,
                 legacy_cycles=500000.0)


class MetaheuristicMinTTest(unittest.TestCase):
    def test_csv_write_emits_each_row_once_and_replaces_existing_file(self):
        with tempfile.TemporaryDirectory(prefix="meta_csv_write_") as raw:
            output = Path(raw) / "results.csv"
            rows = [
                {"case_id": "a970", "score": 10, "ignored": "x"},
                {"case_id": "a1260", "score": 9, "ignored": "y"},
            ]
            audit.csv_write(output, ["case_id", "score"], rows)
            with output.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [
                    {"case_id": "a970", "score": "10"},
                    {"case_id": "a1260", "score": "9"},
                ])

            audit.csv_write(
                output, ["case_id", "score"],
                [{"case_id": "replacement", "score": 8}])
            with output.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [
                    {"case_id": "replacement", "score": "8"},
                ])

    def test_fixed_seeds_split_one_45_second_method_case_budget(self):
        allocations = audit.fixed_seed_wall_budgets()
        self.assertEqual([row["seed"] for row in allocations], [11, 29, 47])
        self.assertEqual([row["wall_budget_s"] for row in allocations],
                         [15.0, 15.0, 15.0])
        self.assertAlmostEqual(
            sum(row["wall_budget_s"] for row in allocations), 45.0)
        self.assertLessEqual(
            sum(row["wall_budget_s"] for row in allocations),
            audit.METHOD_CASE_WALL_BUDGET_S)
        contract = audit.wall_budget_contract()
        self.assertEqual(contract["basis"],
                         "45 s total per method and case")
        self.assertEqual(contract["configured_seed_budget_sum_s"], 45.0)
        self.assertTrue(contract["initialization_charged_to_seed_wall_time"])
        self.assertEqual(contract["fixed_seeds"], [11, 29, 47])
        self.assertIn("ILP-Perf", contract["comparison_methods"])
        self.assertIn("TAP-2.5D", contract["comparison_methods"])

    def test_cpu_is_movable_and_expert_slots_control_sharing(self):
        shared = model(expert_slots=2)
        state = shared.source_state()
        self.assertEqual(state["anchors"][0], (0, 1, 1))
        self.assertTrue(shared.feasible(state)[0])
        state["anchors"][0] = (0, 1, 0)
        self.assertTrue(shared.feasible(state)[0])
        self.assertTrue(shared.strict_evaluate(state)["feasible"])
        random_cpu_anchors = {
            shared.random_state(random.Random(seed))["anchors"][0]
            for seed in range(8)}
        self.assertGreater(len(random_cpu_anchors), 1)
        random_state = shared.random_state(random.Random(19))
        self.assertEqual(random_state["assign"]["e(0)"],
                         random_state["assign"]["e(1)"])

        exclusive = audit.CaseModel(
            "exclusive", params(expert_slots=1), solution(shared=True),
            {"a_max": 3.0, "p_max": 5.0},
            {"L": 1, "X": 2, "Y": 3})
        ok, reasons = exclusive.feasible(audit.state_copy(shared.source_state()))
        self.assertFalse(ok)
        self.assertIn("resident-expert capacity", reasons)

    def test_public_T_is_protocol_completion_not_legacy_task_makespan(self):
        instance = model(expert_slots=2, beta=64.0, shared=True)
        state = instance.source_state()
        evaluated = instance.evaluate_candidate(state)
        self.assertAlmostEqual(
            evaluated["schedule"]["makespan_cycles"], 1500000.0)
        self.assertAlmostEqual(
            evaluated["protocol"]["completion_cycles"], 106.0)
        self.assertAlmostEqual(evaluated["T_cycles"], 106.0)
        first = (evaluated["schedule"]["start"]["e(0)"],
                 evaluated["schedule"]["finish"]["e(0)"])
        second = (evaluated["schedule"]["start"]["e(1)"],
                  evaluated["schedule"]["finish"]["e(1)"])
        self.assertTrue(first[1] <= second[0] or second[1] <= first[0])
        pfirst = (evaluated["protocol"]["compute_start_cycles"]["e(0)"],
                  evaluated["protocol"]["compute_finish_cycles"]["e(0)"])
        psecond = (evaluated["protocol"]["compute_start_cycles"]["e(1)"],
                   evaluated["protocol"]["compute_finish_cycles"]["e(1)"])
        self.assertLessEqual(pfirst[1], psecond[0])
        strict = instance.strict_evaluate(state)
        self.assertTrue(strict["feasible"])
        self.assertFalse(strict["hard_feasibility_task_makespan_scored"])
        emitted = instance.solution(
            state, "sa", 11, evaluated["T_cycles"])
        self.assertNotIn("surrogate_cycles", emitted)
        self.assertEqual(emitted["T_cycles"], 106.0)
        self.assertFalse(
            emitted["objective_scope"]["legacy_task_makespan_scored"])
        self.assertFalse(
            emitted["objective_scope"]["energy_or_byte_hop_tie_break"])

    def test_deterministic_mesh_route_capacity_enters_T(self):
        instance = model(expert_slots=1, beta=0.01, shared=False)
        evaluated = instance.evaluate_candidate(instance.source_state())
        routing = evaluated["routing"]
        self.assertEqual(routing["policy"],
                         "unsplittable dimension-order shortest paths L-X-Y")
        self.assertAlmostEqual(routing["maximum_directed_link_load_bytes"],
                               64.0)
        self.assertAlmostEqual(routing["capacity_bound_cycles"], 6400.0)
        self.assertAlmostEqual(
            evaluated["protocol"]["completion_cycles"], 105.0)
        self.assertAlmostEqual(evaluated["T_cycles"], 6400.0)

    def test_fixed_order_protocol_and_bidirectional_capacity_match_contract(self):
        instance = deterministic_dispatch_model()
        evaluated = instance.evaluate_candidate(instance.source_state())
        routing = evaluated["routing"]
        self.assertEqual(routing["routed_message_count"], 4)
        self.assertAlmostEqual(routing["maximum_directed_link_load_bytes"],
                               64.0)
        self.assertAlmostEqual(routing["capacity_bound_cycles"], 1.0)
        self.assertAlmostEqual(routing["routed_byte_hop"], 256.0)
        self.assertAlmostEqual(
            evaluated["protocol"]["completion_cycles"], 105.0)
        self.assertEqual(evaluated["protocol"]["event_count"], 8)
        self.assertAlmostEqual(evaluated["T_cycles"], 105.0)
        oracle = protocol_oracle.evaluate_protocol_surrogate(
            params(expert_slots=1), solution(shared=False),
            gpu_profile={
                "source": "unit-test",
                "group_type_cycles": {
                    "0": {"GPU-S": 5}, "1": {"GPU-S": 7}},
            },
            tokens_per_source=1, hidden=16, element_bytes=2,
            flit_payload_bytes=32, operation_overhead_cycles=10,
            hop_cycles=2)
        self.assertAlmostEqual(
            evaluated["protocol"]["completion_cycles"],
            oracle["protocol_cycles"])
        self.assertEqual(list(audit.dimension_order_route(
            (0, 0, 0), (0, 0, 2))), [
                ((0, 0, 0), (0, 0, 1)),
                ((0, 0, 1), (0, 0, 2))])

    def test_reference_reports_incumbent_and_bound_not_optimum_gap(self):
        reference = audit.source_reference(solution())
        self.assertEqual(reference["incumbent_cycles"], 120.0)
        self.assertEqual(reference["lower_bound_cycles"], 100.0)
        trace = []
        audit.append_trace(trace, time.perf_counter(), 1, 110.0,
                           reference["incumbent_cycles"],
                           reference["lower_bound_cycles"], force=True)
        self.assertAlmostEqual(trace[0]["relative_vs_ilp_incumbent_pct"],
                               -100.0 / 12.0)
        self.assertAlmostEqual(
            trace[0]["relative_excess_vs_available_lower_bound_pct"], 10.0)
        self.assertFalse(any("optimum" in key for key in trace[0]))

    def test_nonpositive_solver_bound_is_reported_unavailable(self):
        for bound in (0.0, -1.0, None):
            with self.subTest(bound=bound):
                value = solution()
                value["primary_objective_bound_cycles"] = bound
                reference = audit.source_reference(value)
                self.assertIsNone(reference["lower_bound_cycles"])
                self.assertFalse(reference["lower_bound_available"])
        invalid = solution()
        invalid["primary_objective_bound_cycles"] = 121.0
        with self.assertRaisesRegex(ValueError, "exceeds its incumbent"):
            audit.source_reference(invalid)

    def test_retired_objective_and_C9_inputs_fail_closed(self):
        capped = solution()
        capped["performance_cap_enforced"] = True
        with self.assertRaisesRegex(ValueError, "forbidden"):
            audit.CaseModel(
                "capped", params(), capped,
                {"a_max": 3.0, "p_max": 5.0},
                {"L": 1, "X": 2, "Y": 3})

        tied = solution()
        tied["objective_contract"]["energy_or_byte_hop_tie_break"] = True
        with self.assertRaisesRegex(ValueError, "forbidden"):
            audit.CaseModel(
                "tied", params(), tied,
                {"a_max": 3.0, "p_max": 5.0},
                {"L": 1, "X": 2, "Y": 3})

        task_dag = solution()
        task_dag["noc_power_proxy"]["traffic_kind"] = "task-dag"
        with self.assertRaisesRegex(ValueError, "deterministic-dispatch"):
            audit.CaseModel(
                "task-dag", params(), task_dag,
                {"a_max": 3.0, "p_max": 5.0},
                {"L": 1, "X": 2, "Y": 3})

        with self.assertRaisesRegex(ValueError, "C9/p_col"):
            audit.CaseModel(
                "c9", params(), solution(),
                {"a_max": 3.0, "p_max": 5.0, "p_col": 2.0},
                {"L": 1, "X": 2, "Y": 3})

    def test_canonical_root_is_explicit_and_superseded_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="meta_canonical_resolve_") as raw:
            project_root = Path(raw)
            with self.assertRaisesRegex(ValueError,
                                        "--canonical-root is required"):
                audit.resolve_canonical_root(project_root)
            canonical = project_root / "explicit-canonical"
            canonical.mkdir(parents=True)
            (canonical / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                audit.resolve_canonical_root(project_root,
                                             "explicit-canonical"),
                canonical.resolve())
            (canonical / "SUPERSEDED.txt").write_text(
                "retired", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SUPERSEDED"):
                audit.resolve_canonical_root(project_root,
                                             "explicit-canonical")

    def test_loader_reads_fresh_canonical_manifest_and_verifies_sha(self):
        with tempfile.TemporaryDirectory(prefix="meta_canonical_loader_") as raw:
            root = Path(raw)
            params_path = root / "params.json"
            solution_path = root / "ilp_perf.json"
            params_path.write_text(json.dumps(params()), encoding="utf-8")
            solution_path.write_text(json.dumps(solution()), encoding="utf-8")
            manifest = {
                "protocol": {"fresh_ilp_perf": True,
                             "ilp_perf_objective": "sole-joint-min-T"},
                "cases": [{
                    "case_id": "tiny", "key": "budget:tiny",
                    "params_path": str(params_path),
                    "params_sha256": audit.sha256(params_path),
                    "budgets": {"a_max": 2.0, "p_max": 3.0},
                    "grid": {"L": 1, "X": 2, "Y": 2},
                    "ilp_perf": {
                        "status": "complete", "reused": False,
                        "command": ["python", "ilp_layout.py", "--params",
                                    str(params_path)],
                        "solution_path": str(solution_path),
                        "solution_sha256": audit.sha256(solution_path),
                    },
                }],
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            cases = audit.load_formal_cases(root)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["case_key"], "budget:tiny")
            (root / "SUPERSEDED.txt").write_text(
                "retired", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SUPERSEDED"):
                audit.load_formal_cases(root)
            (root / "SUPERSEDED.txt").unlink()
            original_solution = solution_path.read_text(encoding="utf-8")
            solution_path.write_text(original_solution + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                audit.load_formal_cases(root)
            solution_path.write_text(original_solution, encoding="utf-8")
            params_path.write_text(params_path.read_text(encoding="utf-8") + " ",
                                   encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "params SHA-256"):
                audit.load_formal_cases(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
