#!/usr/bin/env python3
"""Focused tests for Matched-Random's public-T representative selection."""

import collections
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_layout_baseline as matched
import run_metaheuristic_surrogate_audit as public_t_model


def _tiny_params():
    return {
        "meta": {"groups": 2, "topk": 2, "group_pi": [0.5, 0.5]},
        "library": {
            "CPU": {"w": 1, "h": 1, "area": 1.0, "p_peak": 1.0,
                    "mem": 100.0, "expert_slots": 0},
            "GPU-S": {"w": 1, "h": 1, "area": 1.0, "p_peak": 2.0,
                      "mem": 100.0, "expert_slots": 1},
        },
        "tasks": {
            "control": {"allowed": ["CPU"], "mem_mb": 1.0,
                        "cyc": {"CPU": 500000.0},
                        "eta": {"CPU": 0.0}},
            "e(0)": {"allowed": ["GPU-S"], "mem_mb": 1.0,
                     "cyc": {"GPU-S": 500000.0},
                     "eta": {"GPU-S": 0.0}},
            "e(1)": {"allowed": ["GPU-S"], "mem_mb": 1.0,
                     "cyc": {"GPU-S": 500000.0},
                     "eta": {"GPU-S": 0.0}},
        },
        "expert_tasks": ["e(0)", "e(1)"],
        "edges": [
            {"src": "control", "dst": "e(0)", "vol": 0.0},
            {"src": "control", "dst": "e(1)", "vol": 0.0},
        ],
        "interconnect": {"tau_h": 0.0, "tau_v": 0.0, "t0": 0.0,
                         "beta": 64.0, "eps_h": 0.0, "eps_v": 0.0},
    }


def _tiny_solution():
    return {
        "sites": [
            {"l": 0, "i": 1, "j": 1, "type": "CPU",
             "tasks": ["control"]},
            {"l": 0, "i": 0, "j": 0, "type": "GPU-S",
             "tasks": ["e(0)"]},
            {"l": 0, "i": 0, "j": 2, "type": "GPU-S",
             "tasks": ["e(1)"]},
        ],
        "task_site": {
            "control": [0, 1, 1], "e(0)": [0, 0, 0],
            "e(1)": [0, 0, 2],
        },
        "grid": {"L": 1, "X": 2, "Y": 3},
        "budgets": {"a_max": 3.0, "p_max": 5.0},
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
        "objective_contract": {
            "active_objective_count": 1,
            "active_objective": "min T",
            "protocol_completion_in_T": True,
            "energy_or_byte_hop_tie_break": False,
        },
        "performance_cap_enforced": False,
    }


class MatchedPublicTTest(unittest.TestCase):
    def test_scorer_is_the_shared_protocol_and_mesh_capacity_objective(self):
        params = _tiny_params()
        solution = _tiny_solution()
        direct = public_t_model.CaseModel(
            "direct", params, solution, solution["budgets"], solution["grid"])
        direct_evaluation = direct.evaluate_candidate(direct.source_state())
        solution["predicted_makespan_cycles"] = direct_evaluation["T_cycles"]

        state = {"budgets": solution["budgets"], "grid": solution["grid"]}
        scorer = matched._PublicTScorer(params, solution, state)
        evaluated = scorer.evaluate(solution)

        self.assertEqual(
            matched._PUBLIC_T_IMPLEMENTATION,
            "run_metaheuristic_surrogate_audit.CaseModel.evaluate_candidate")
        self.assertAlmostEqual(evaluated["T_cycles"],
                               direct_evaluation["T_cycles"])
        self.assertAlmostEqual(
            evaluated["T_cycles"],
            max(evaluated["protocol"]["completion_cycles"],
                evaluated["routing"]["capacity_bound_cycles"]))
        self.assertNotEqual(evaluated["T_cycles"],
                            evaluated["schedule"]["makespan_cycles"])
        provenance = scorer.provenance(evaluated)
        self.assertEqual(provenance["metric"], "public_T_cycles")
        self.assertEqual(provenance["active_objective"], "min T")
        self.assertTrue(provenance["protocol_completion_in_T"])
        self.assertTrue(provenance["directed_mesh_capacity_in_T"])
        self.assertFalse(provenance["legacy_task_makespan_scored"])
        self.assertEqual(provenance["legacy_task_makespan_role"],
                         "hard-feasibility-only")

    def test_all_100_candidates_are_scored_and_public_T_selects_median(self):
        public_calls = []

        class FakePublicTScorer(object):
            def __init__(self, unused_params, unused_source, unused_state):
                del unused_params, unused_source, unused_state
                self.source_evaluation = self._evaluation(-1)

            @staticmethod
            def _evaluation(seed):
                public_T = 1000.0 + max(seed, 0)
                return {
                    "T_cycles": public_T,
                    "protocol": {
                        "completion_cycles": public_T,
                        "policy": "test fixed-order protocol",
                    },
                    "routing": {
                        "capacity_bound_cycles": 10.0,
                        "policy": "test directed-Mesh capacity",
                        "milp_relation": "test relation",
                    },
                    "schedule": {
                        "makespan_cycles": 9000.0 - max(seed, 0),
                        "policy": "test feasibility schedule",
                    },
                }

            def evaluate(self, solution):
                seed = int(solution["seed"])
                public_calls.append(seed)
                return self._evaluation(seed)

            @staticmethod
            def provenance(unused_evaluation):
                del unused_evaluation
                return {
                    "metric": "public_T_cycles",
                    "definition": matched._PUBLIC_T_DEFINITION,
                    "implementation": matched._PUBLIC_T_IMPLEMENTATION,
                    "protocol_policy": "test fixed-order protocol",
                    "directed_mesh_capacity_policy":
                        "test directed-Mesh capacity",
                    "routing_relation_to_milp": "test relation",
                    "legacy_task_makespan_role": "hard-feasibility-only",
                    "legacy_task_makespan_scored": False,
                    "energy_or_byte_hop_tie_break": False,
                }

        def fake_candidate(unused_params, unused_source, state, seed,
                           permute_experts=True):
            del unused_params, unused_source, permute_experts
            anchor = [0, 0, int(seed)]
            legacy = {
                "feasible": True,
                "reasons": [],
                "makespan_cycles": 2000.0 - seed,
                "energy_total_J": 0.0,
                "energy_comm_J": 0.0,
                "energy_comp_J": 0.0,
                "byte_hop": 0.0,
                "total_area_mm2": 1.0,
                "peak_power_W": 1.0,
            }
            return ({
                "method": "resource-matched-random-layout",
                "seed": int(seed),
                "sites": [{"l": 0, "i": 0, "j": int(seed),
                           "type": "GPU", "tasks": ["task"]}],
                "task_site": {"task": anchor},
                "grid": dict(state["grid"]),
                "budgets": dict(state["budgets"]),
                "feasible": True,
                "baseline": {
                    "strict_evaluate": True,
                    "expert_mapping_changed": False,
                    "packing_strategy": "test",
                },
            }, legacy)

        def fake_strict(unused_params, task_site, unused_types,
                        unused_grid, unused_budgets):
            del unused_params, unused_types, unused_grid, unused_budgets
            seed = int(task_site["task"][2])
            return ({
                "feasible": True,
                "reasons": [],
                "makespan_cycles": 2000.0 - seed,
                "energy_total_J": 0.0,
                "energy_comm_J": 0.0,
                "energy_comp_J": 0.0,
                "byte_hop": 0.0,
                "total_area_mm2": 1.0,
                "peak_power_W": 1.0,
            }, [])

        fake_state = {
            "grid": {"L": 1, "X": 1, "Y": 100},
            "budgets": {"a_max": 1.0, "p_max": 1.0},
            "source_score": {"total_area_mm2": 1.0,
                             "peak_power_W": 1.0},
            "type_counts": {"GPU": 1},
            "cpu_counts": {},
        }
        params = {"library": {"GPU": {"simlet": "gpu"}}}
        with tempfile.TemporaryDirectory(prefix="matched_public_t_") as raw:
            root = Path(raw)
            params_path = root / "params.json"
            source_path = root / "source.json"
            params_path.write_text(json.dumps(params), encoding="utf-8")
            source_path.write_text("{}", encoding="utf-8")
            output = root / "matched"

            with mock.patch.object(matched, "_source_state",
                                   return_value=fake_state), \
                    mock.patch.object(matched, "_candidate_solution",
                                      side_effect=fake_candidate), \
                    mock.patch.object(matched, "_strict_score",
                                      side_effect=fake_strict), \
                    mock.patch.object(matched, "_PublicTScorer",
                                      FakePublicTScorer):
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = matched.run(
                        str(params_path), str(source_path), list(range(100)),
                        str(output))

            # Public values 1000..1099 have median 1049.5; the documented
            # lower-central tie-break chooses seed 49.  Legacy task makespans
            # 2000..1901 would instead choose seed 50.
            self.assertEqual(summary["representative_seed"], 49)
            self.assertEqual(summary["median_public_T_cycles"], 1049.5)
            legacy_median = matched._median(
                [2000.0 - seed for seed in range(100)])
            legacy_choice = min(
                range(100),
                key=lambda seed: (abs((2000.0 - seed) - legacy_median),
                                  2000.0 - seed, seed))
            self.assertEqual(legacy_choice, 50)

            counts = collections.Counter(public_calls)
            self.assertEqual(set(counts), set(range(100)))
            # Each candidate is scored before and after serialization; the
            # selected representative receives one final independent rescore.
            self.assertEqual(counts[49], 3)
            self.assertTrue(all(counts[seed] == 2
                                for seed in range(100) if seed != 49))

            representative = json.loads(
                (output / "representative.json").read_text(encoding="utf-8"))
            self.assertEqual(representative["seed"], 49)
            self.assertEqual(representative["public_T_cycles"], 1049.0)
            self.assertEqual(
                representative["legacy_evaluate_task_makespan_cycles"],
                1951.0)
            self.assertEqual(summary["schema_version"], 2)
            self.assertEqual(summary["candidate_count"], 100)
            self.assertTrue(summary["all_candidates_scored_with_public_T"])
            self.assertEqual(summary["representative_selection_metric"],
                             "public_T_cycles")
            self.assertEqual(
                summary["public_T_scoring"]["implementation"],
                matched._PUBLIC_T_IMPLEMENTATION)
            self.assertFalse(summary["public_T_scoring"]
                             ["legacy_task_makespan_scored"])


if __name__ == "__main__":
    unittest.main()
