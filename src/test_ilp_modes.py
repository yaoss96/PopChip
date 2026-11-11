#!/usr/bin/env python3
"""Smoke-test the single-objective ILP-Perf contract and core constraints.

The test process itself needs no gurobipy.  It invokes the repository's
Gurobi Python 3.7 interpreter on a deliberately tiny three-bundle instance.
"""

import json
import sys
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ILP = ROOT / "src" / "ilp_layout.py"
DEFAULT_GUROBI_PYTHON = Path(os.environ.get("GUROBI_PYTHON", sys.executable))


def tiny_params():
    return {
        "meta": {"note": "unit-test analytical workload", "groups": 2,
                 "topk": 1, "group_pi": [0.8, 0.2], "bytes_elem": 2},
        "library": {
            "CPU": {"w": 1, "h": 1, "area": 1.0,
                    "p_peak": 1.0, "mem": 1024.0},
            "GPU-S": {"w": 1, "h": 1, "area": 1.0,
                      "p_peak": 2.0, "mem": 1024.0},
        },
        "tasks": {
            "control": {"allowed": ["CPU"], "mem_mb": 1.0,
                        "cyc": {"CPU": 100.0},
                        "eta": {"CPU": 1.0e-9}},
            "e(0)": {"allowed": ["GPU-S"], "mem_mb": 1.0,
                        "cyc": {"GPU-S": 1000.0},
                        "eta": {"GPU-S": 2.0e-9}},
            "e(1)": {"allowed": ["GPU-S"], "mem_mb": 1.0,
                        "cyc": {"GPU-S": 900.0},
                        "eta": {"GPU-S": 2.0e-9}},
        },
        "expert_tasks": ["e(0)", "e(1)"],
        "edges": [
            {"src": "control", "dst": "e(0)", "vol": 160.0},
            {"src": "control", "dst": "e(1)", "vol": 640.0},
        ],
        "interconnect": {"tau_h": 1.0, "tau_v": 0.2,
                         "eps_h": 1.0e-12, "eps_v": 1.0e-13,
                         "t0": 10.0, "beta": 16.0},
    }


def tiny_reference():
    return {
        "sites": [
            {"l": 0, "i": 0, "j": 0, "type": "CPU",
             "tasks": ["control"]},
            {"l": 0, "i": 0, "j": 2, "type": "GPU-S",
             "tasks": ["e(0)"]},
            {"l": 0, "i": 0, "j": 3, "type": "GPU-S",
             "tasks": ["e(1)"]},
        ],
        "task_site": {
            "control": [0, 0, 0],
            "e(0)": [0, 0, 2],
            "e(1)": [0, 0, 3],
        },
    }


def mixed_bom_params(experts=8):
    """Performance-skewed expert set for Stage-0 BOM/packing tests."""
    tasks = {
        "control": {"allowed": ["CPU"], "mem_mb": 1.0,
                    "cyc": {"CPU": 100.0},
                    "eta": {"CPU": 1.0e-9}},
    }
    edges = []
    expert_names = []
    weights = [float(experts - index) for index in range(experts)]
    weight_sum = sum(weights)
    for index in range(experts):
        name = "e(%d)" % index
        expert_names.append(name)
        # Absolute S->L reductions decrease with index, making the intended
        # hot/benefit upgrade order independently observable in JSON evidence.
        tasks[name] = {
            "allowed": ["GPU-L", "GPU-S"], "mem_mb": 16.0,
            "cyc": {"GPU-L": 1000.0 + 10.0 * index,
                    "GPU-S": 10000.0 - 100.0 * index},
            "eta": {"GPU-L": 1.0e-9, "GPU-S": 2.0e-9},
        }
        edges.append({"src": "control", "dst": name,
                      "vol": 1024.0 - 8.0 * index})
    return {
        "meta": {"note": "Stage-0 mixed-BOM unit test",
                 "groups": experts, "topk": 1,
                 "group_pi": [weight / weight_sum for weight in weights],
                 "bytes_elem": 2},
        "library": {
            "CPU": {"w": 1, "h": 1, "area": 90.0,
                    "p_peak": 65.0, "mem": 4096.0,
                    "expert_slots": 0},
            "GPU-L": {"w": 2, "h": 2, "area": 400.0,
                      "p_peak": 300.0, "mem": 8192.0,
                      "expert_slots": 1},
            "GPU-S": {"w": 1, "h": 1, "area": 110.0,
                      "p_peak": 90.0, "mem": 2048.0,
                      "expert_slots": 1},
        },
        "tasks": tasks,
        "expert_tasks": expert_names,
        "edges": edges,
        "interconnect": {"tau_h": 1.0, "tau_v": 0.2,
                         "eps_h": 1.0e-12, "eps_v": 1.0e-13,
                         "t0": 1.0, "beta": 64.0},
    }


class ILPModeSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("GUROBI_PYTHON")
        cls.python = Path(configured) if configured else DEFAULT_GUROBI_PYTHON
        if not cls.python.is_file():
            raise unittest.SkipTest("Gurobi Python not found: %s" % cls.python)

    def test_optional_bound_diagnostic_is_none_safe(self):
        source_dir = str(ILP.parent)
        script = (
            "import sys; sys.path.insert(0, %r); import ilp_layout as m; "
            "assert m._format_optional_float(None) == 'unavailable'; "
            "assert m._format_optional_float(float('inf')) == 'unavailable'; "
            "assert m._format_optional_float(12.34) == '12.3'" % source_dir)
        completed = subprocess.run(
            [str(self.python), "-c", script], cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=30)
        self.assertEqual(
            completed.returncode, 0,
            msg="format helper failed\nSTDOUT:\n%s\nSTDERR:\n%s" %
                (completed.stdout, completed.stderr))

    def test_public_cli_exposes_only_perf_and_no_energy_budget(self):
        completed = subprocess.run(
            [str(self.python), str(ILP), "--help"], cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=30)
        self.assertEqual(completed.returncode, 0)
        help_text = " ".join(completed.stdout.split())
        self.assertIn("ilp-perf", help_text)
        self.assertIn("single-objective joint MILP: minimize makespan T",
                      help_text)
        self.assertIn("--mip-focus", help_text)
        self.assertIn("prioritize finding/improving feasible min-T incumbents",
                      help_text)
        self.assertNotIn("ilp-power", help_text)
        self.assertNotIn("1.20", help_text)
        for retired_flag in (
                "--e-bar", "--secondary-time", "--secondary-mipgap",
                "--power-tertiary-time", "--perf-tie-relax",
                "--power-perf-relax", "--w-ecomm", "--w-resource",
                "--t-hat", "--p-col"):
            self.assertNotIn(retired_flag, help_text)

        for retired_mode in (
                "ilp-power", "latency", "efficiency", "certified-latency"):
            rejected = subprocess.run(
                [str(self.python), str(ILP), "--params", "unused.json",
                 "--mode", retired_mode], cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("invalid choice", rejected.stderr)

        rejected_energy = subprocess.run(
            [str(self.python), str(ILP), "--params", "unused.json",
             "--e-bar", "1e-3"], cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=30)
        self.assertNotEqual(rejected_energy.returncode, 0)
        self.assertIn("unrecognized arguments: --e-bar", rejected_energy.stderr)

        source_dir = str(ILP.parent)
        rejected_api = subprocess.run(
            [str(self.python), "-c",
             "import sys; sys.path.insert(0, %r); import ilp_layout as m; "
             "\ntry:\n m.build_and_solve({}, 1, 1, 1, "
             "{'a_max': 1, 'p_max': 1, 'e_bar': 1e-3}, "
             "{'mode': 'ilp-perf'}, verbose=False)"
             "\nexcept ValueError as ex:\n assert 'retired' in str(ex)"
             "\nelse:\n raise AssertionError('Ebar was accepted')" % source_dir],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=30)
        self.assertEqual(
            rejected_api.returncode, 0,
            msg="retired Ebar API was accepted\nSTDOUT:\n%s\nSTDERR:\n%s" %
                (rejected_api.stdout, rejected_api.stderr))

        rejected_modes_api = subprocess.run(
            [str(self.python), "-c",
             "import sys; sys.path.insert(0, %r); import ilp_layout as m\n"
             "for retired in ('ilp-power', 'latency', 'efficiency', "
             "'certified-latency'):\n"
             " try:\n"
             "  m.build_and_solve({}, 1, 1, 1, "
             "{'a_max': 1, 'p_max': 1}, {'mode': retired}, verbose=False)\n"
             " except ValueError as ex:\n"
             "  assert 'accepts only ilp-perf' in str(ex), str(ex)\n"
             " else:\n"
             "  raise AssertionError(retired + ' API mode was accepted')" %
             source_dir],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=30)
        self.assertEqual(
            rejected_modes_api.returncode, 0,
            msg="retired internal objective mode was accepted\nSTDOUT:\n%s\n"
                "STDERR:\n%s" %
                (rejected_modes_api.stdout, rejected_modes_api.stderr))

    def test_area_and_package_peak_power_budgets_are_hard(self):
        with tempfile.TemporaryDirectory(prefix="ilp_budget_smoke_") as raw:
            directory = Path(raw)
            params_path = directory / "params.json"
            reference_path = directory / "reference.json"
            with params_path.open("w") as stream:
                json.dump(tiny_params(), stream)
            with reference_path.open("w") as stream:
                json.dump(tiny_reference(), stream)

            for label, area_limit, power_limit in (
                    ("area", "2.9", "5"), ("power", "3", "4.9")):
                output_path = directory / (label + ".json")
                completed = subprocess.run(
                    [str(self.python), str(ILP),
                     "--params", str(params_path),
                     "--bom-from", str(reference_path),
                     "--L", "1", "--X", "1", "--Y", "4",
                     "--a-max", area_limit, "--p-max", power_limit,
                     "--noc-traffic", "task-dag", "--time-limit", "5",
                     "--threads", "1", "--quiet", "-o", str(output_path)],
                    cwd=str(ROOT), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, universal_newlines=True,
                    timeout=30)
                self.assertNotEqual(
                    completed.returncode, 0,
                    msg="%s budget unexpectedly feasible\nSTDOUT:\n%s\n"
                        "STDERR:\n%s" %
                        (label, completed.stdout, completed.stderr))
                self.assertFalse(output_path.exists())

    def run_mode(self, directory, mode, cpu_anchor="free",
                 no_symmetry=False, reference_name="reference.json",
                 reference_policy="placement"):
        output = directory / (mode + ".json")
        command = [
            str(self.python), str(ILP),
            "--params", str(directory / "params.json"),
            ("--placement-from" if reference_policy == "placement"
             else "--bom-from"), str(directory / reference_name),
            "--L", "1", "--X", "1", "--Y", "4",
            "--a-max", "3", "--p-max", "5",
            "--mode", mode, "--cpu-anchor", cpu_anchor,
            "--time-limit", "5", "--mipgap", "1e-9",
            "--noc-clock-hz", "1000000000",
            "--threads", "1",
            "--quiet", "-o", str(output),
        ]
        if no_symmetry:
            command.append("--no-symmetry-break")
        completed = subprocess.run(
            command, cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        self.assertEqual(
            completed.returncode, 0,
            msg="command failed\nSTDOUT:\n%s\nSTDERR:\n%s" %
                (completed.stdout, completed.stderr))
        with output.open() as stream:
            return json.load(stream)

    @staticmethod
    def bundle_signature(solution):
        return sorted((site["type"], tuple(sorted(site.get("tasks", []))))
                      for site in solution["sites"])

    @staticmethod
    def exact_tiny_protocol(solution):
        """Independently replay the tiny fixed-order max-plus recurrence."""
        protocol = solution["protocol_model"]
        expert_names = ["e(0)", "e(1)"]
        dispatch = [[26, 6], [25, 7]]
        token_bytes = solution["noc_power_proxy"][
            "dispatch_token_bytes_per_phase"]
        init_bytes = 4.0 * (4 + 2 * len(expert_names))
        done_bytes = 4.0
        sync = protocol["sync_overhead_cycles"]
        flit_bytes = protocol["flit_bytes"]
        hop_cycles = protocol["hop_cycles"]
        type_by_site = {
            (site["l"], site["i"], site["j"]): site["type"]
            for site in solution["sites"]}
        cpu_site = next(
            site for site, kind in type_by_site.items() if kind == "CPU")
        expert_sites = [tuple(solution["task_site"][expert])
                        for expert in expert_names]
        compute_cycles = [
            protocol["compute_cycles_by_expert_type"][expert][
                type_by_site[expert_sites[group]]]
            for group, expert in enumerate(expert_names)]

        def distance(first, second):
            return sum(abs(first[axis] - second[axis])
                       for axis in range(3))

        clocks = {"cpu": 0.0, 0: 0.0, 1: 0.0}

        def message(source, target, volume, route_distance):
            serialization = float(int(math.ceil(volume / flit_bytes)))
            write_time = clocks[source] + sync
            sender_finish = write_time + serialization
            receiver_finish = max(
                write_time + serialization + hop_cycles * route_distance,
                clocks[target] + sync)
            clocks[source] = sender_finish
            clocks[target] = receiver_finish

        for group in range(2):
            message("cpu", group, init_bytes,
                    distance(cpu_site, expert_sites[group]))
        for source in range(2):
            for target in range(2):
                if source != target and dispatch[source][target] > 0:
                    message(
                        source, target,
                        dispatch[source][target] * token_bytes,
                        distance(expert_sites[source], expert_sites[target]))
        for group in range(2):
            clocks[group] += compute_cycles[group]
        for source in range(2):
            for target in range(2):
                if source != target and dispatch[source][target] > 0:
                    message(
                        target, source,
                        dispatch[source][target] * token_bytes,
                        distance(expert_sites[source], expert_sites[target]))
        for group in range(2):
            message(group, "cpu", done_bytes,
                    distance(expert_sites[group], cpu_site))
        return clocks

    def test_perf_is_single_joint_min_replay_aligned_t(self):
        with tempfile.TemporaryDirectory(prefix="ilp_perf_smoke_") as raw:
            directory = Path(raw)
            with (directory / "params.json").open("w") as stream:
                json.dump(tiny_params(), stream)
            with (directory / "reference.json").open("w") as stream:
                json.dump(tiny_reference(), stream)

            result = self.run_mode(
                directory, "ilp-perf", cpu_anchor="free",
                no_symmetry=True, reference_policy="bom")

            self.assertEqual(result["mode_label"], "ILP-Perf")
            self.assertEqual(result["optimization_mode"], "ilp-perf")
            self.assertEqual(
                result["primary_objective"],
                "minimize joint MILP makespan T")
            contract = result["objective_contract"]
            self.assertEqual(contract["active_objective_count"], 1)
            self.assertEqual(contract["active_objective"], "min T")
            self.assertTrue(
                contract["joint_selection_placement_scheduling"])
            self.assertTrue(contract["protocol_completion_in_T"])
            self.assertFalse(contract["protocol_used_as_separate_objective"])
            self.assertEqual(
                contract["legacy_task_schedule_role"],
                "hard-feasibility-only")
            self.assertFalse(contract["energy_or_byte_hop_tie_break"])
            self.assertNotIn("e_bar", result["budgets"])

            makespan = result["predicted_makespan_cycles"]
            self.assertTrue(math.isclose(
                result["performance_proxy_cycles"], makespan,
                rel_tol=0.0, abs_tol=1e-9))
            self.assertTrue(math.isclose(
                result["tightened_performance_proxy_cycles"], makespan,
                rel_tol=0.0, abs_tol=1e-9))
            self.assertTrue(math.isclose(
                result["primary_incumbent_cycles"], makespan,
                rel_tol=0.0, abs_tol=1e-9))
            self.assertEqual(
                result["performance_extraction_kind"],
                "MILP makespan variable T")
            self.assertIn("replay-aligned protocol makespan T",
                          result["performance_metric_kind"])
            self.assertTrue(result["primary_has_incumbent"])
            self.assertTrue(result["primary_gap_certificate"]["certified"])
            if result["primary_objective_bound_cycles"] is not None:
                self.assertLessEqual(
                    result["primary_objective_bound_cycles"],
                    makespan + 1e-7)
            self.assertTrue(math.isclose(
                result["solver_wall_s"],
                result["model_build_s"] + result["primary_wall_s"],
                rel_tol=0.0, abs_tol=1e-9))
            self.assertTrue(math.isclose(
                result["model_build_s"],
                result["formulation_build_s"] + result["stage0_total_s"],
                rel_tol=0.0, abs_tol=1e-9))
            stage0_evidence = result["stage0_feasible_start"] or {}
            refinement_evidence = stage0_evidence.get(
                "protocol_placement_refinement") or {}
            self.assertLessEqual(
                refinement_evidence.get("wall_s", 0.0),
                result["stage0_total_s"] + 1e-9)
            self.assertTrue(math.isclose(
                result["time_to_first_feasible_s"],
                result["model_build_s"] + result["anytime_trace"][0]["wall_s"],
                rel_tol=0.0, abs_tol=1e-9))
            self.assertEqual(result["constructive_search_wall_s"], 0.0)

            self.assertIsNone(result["compute_selection_reference_cycles"])
            self.assertIsNone(result["compute_selection_relaxation"])
            self.assertIsNone(result["compute_selection_cap_cycles"])
            self.assertIsNone(result["performance_reference_cycles"])
            self.assertIsNone(result["performance_reference_kind"])
            self.assertIsNone(result["performance_relaxation"])
            self.assertIsNone(result["performance_cap_cycles"])
            self.assertFalse(result["performance_cap_enforced"])
            for stage in ("secondary", "tertiary", "quaternary"):
                self.assertIsNone(result[stage + "_objective"])
                self.assertFalse(result[stage + "_has_incumbent"])
                self.assertEqual(
                    result[stage + "_gap_certificate"]["reason"],
                    "stage-not-run")
                self.assertEqual(result[stage + "_wall_s"], 0.0)
            self.assertIsNone(result["secondary_time_limit_s"])
            self.assertIsNone(result["tertiary_time_limit_s"])

            self.assertFalse(result["internal_bundle_refinement"]["enabled"])
            self.assertFalse(result["constructive_perf_start"]["attempted"])
            self.assertNotIn("constructive_power_start", result)
            self.assertEqual(result["cpu_anchor_policy"], "free")
            comparison = result["comparison_reference"]
            self.assertEqual(comparison["kind"], "same-BOM")
            self.assertFalse(comparison["fixed_complete_bundles"])
            self.assertFalse(result["reference_coordinates_warm_started"])

            protocol = result["protocol_model"]
            self.assertEqual(protocol["kind"],
                             "fixed-order-max-plus-event-DAG")
            self.assertFalse(protocol["diagnostic_only"])
            self.assertTrue(protocol["used_as_objective"])
            self.assertTrue(protocol["included_in_T"])
            self.assertFalse(protocol["used_as_separate_objective"])
            self.assertTrue(protocol["co_resident_compute_nonoverlap"])
            self.assertEqual(protocol["event_count"], 8)
            self.assertEqual(protocol["flit_bytes"], 32.0)
            self.assertEqual(protocol["hop_cycles"], 104.0)
            self.assertEqual(
                protocol["completion_extraction"],
                "exact-coordinate fixed-order max-plus recurrence")
            self.assertTrue(math.isclose(
                protocol["completion_cycles"],
                result["protocol_makespan_cycles"],
                rel_tol=0.0, abs_tol=1e-9))
            self.assertTrue(math.isclose(
                protocol["completion_cycles"], makespan,
                rel_tol=1e-9, abs_tol=1e-6))
            exact_clocks = self.exact_tiny_protocol(result)
            self.assertTrue(math.isclose(
                result["protocol_makespan_cycles"],
                exact_clocks["cpu"], rel_tol=0.0, abs_tol=1e-9))
            self.assertGreaterEqual(
                makespan + 1e-4, result["protocol_makespan_cycles"])
            self.assertNotEqual(
                result["analytical_task_makespan_cycles"],
                result["protocol_makespan_cycles"])
            for group, expert in enumerate(("e(0)", "e(1)")):
                self.assertTrue(math.isclose(
                    protocol["worker_finish_cycles"][expert],
                    exact_clocks[group], rel_tol=0.0, abs_tol=1e-9))
            for phase in ("dispatch", "combine"):
                phase_data = protocol["phase_completion"][phase]
                self.assertTrue(math.isclose(
                    phase_data["completion_cycles"],
                    max(phase_data["link_cycles"],
                        phase_data["endpoint_cycles"]),
                    rel_tol=0.0, abs_tol=1e-9))

            proxy = result["noc_power_proxy"]
            self.assertFalse(proxy["measured_orion_power"])
            self.assertTrue(proxy["diagnostic_only"])
            self.assertFalse(proxy["used_as_objective"])
            self.assertIn("splittable packet flow",
                          proxy["objective_kind"])
            self.assertEqual(proxy["traffic_kind"],
                             "deterministic-dispatch")
            self.assertGreater(proxy["energy_J"], 0.0)
            self.assertIsNone(proxy["deadline_cycles"])
            self.assertIsNone(
                proxy["deadline_normalized_dynamic_power_W"])
            self.assertGreater(
                proxy["protocol_normalized_dynamic_power_W"], 0.0)
            self.assertTrue(proxy["cpu_control_energy_included"])

            routing = result["routing"]
            self.assertEqual(routing["capacity_horizon"], "T")
            self.assertEqual(routing["phases"], ["dispatch", "combine"])
            self.assertLessEqual(
                routing["maximum_link_utilization"], 1.0 + 1e-8)
            self.assertLessEqual(
                routing["maximum_flow_residual_KiB"], 1e-8)
            self.assertLessEqual(
                routing["maximum_shortest_path_residual_KiB_hops"], 1e-8)
            self.assertTrue(math.isclose(
                routing["payload_routed_byte_hop"],
                proxy["dispatch_byte_hop"],
                rel_tol=0.0, abs_tol=1e-8))
            self.assertTrue(math.isclose(
                routing["routed_byte_hop"],
                routing["payload_routed_byte_hop"] +
                routing["cpu_control_byte_hop"],
                rel_tol=0.0, abs_tol=1e-8))
            by_phase_source = {}
            for entry in routing["sparse_flow"]:
                key = (entry["phase"], entry["source"])
                by_phase_source[key] = (
                    by_phase_source.get(key, 0.0) + entry["flow_KiB"])
            expert_distance = sum(abs(first - second) for first, second in zip(
                result["task_site"]["e(0)"],
                result["task_site"]["e(1)"]))
            self.assertTrue(math.isclose(
                by_phase_source[("dispatch", "e(0)")],
                1.5 * expert_distance,
                rel_tol=0.0, abs_tol=1e-8))
            self.assertTrue(math.isclose(
                by_phase_source[("combine", "e(0)")],
                6.25 * expert_distance,
                rel_tol=0.0, abs_tol=1e-8))

    def test_reference_policies_do_not_add_an_internal_objective_stage(self):
        with tempfile.TemporaryDirectory(prefix="ilp_reference_modes_") as raw:
            directory = Path(raw)
            with (directory / "params.json").open("w") as stream:
                json.dump(tiny_params(), stream)
            reference = tiny_reference()
            with (directory / "reference.json").open("w") as stream:
                json.dump(reference, stream)

            bom_result = self.run_mode(
                directory, "ilp-perf", no_symmetry=True,
                reference_policy="bom")
            self.assertFalse(
                bom_result["internal_bundle_refinement"]["enabled"])
            self.assertEqual(
                bom_result["comparison_reference"]["kind"], "same-BOM")
            self.assertFalse(
                bom_result["comparison_reference"][
                    "fixed_complete_bundles"])
            self.assertFalse(
                bom_result["reference_coordinates_warm_started"])

            placement_result = self.run_mode(
                directory, "ilp-perf", no_symmetry=True,
                reference_policy="placement")
            self.assertFalse(
                placement_result["internal_bundle_refinement"]["enabled"])
            comparison = placement_result["comparison_reference"]
            self.assertEqual(comparison["kind"],
                             "placement-only-bundles")
            self.assertTrue(comparison["fixed_complete_bundles"])
            self.assertTrue(
                placement_result["reference_coordinates_warm_started"])
            self.assertEqual(
                self.bundle_signature(placement_result),
                self.bundle_signature(reference))
            for result in (bom_result, placement_result):
                self.assertGreaterEqual(
                    result["predicted_makespan_cycles"] + 1e-4,
                    result["protocol_makespan_cycles"])
                self.assertGreater(
                    result["analytical_task_makespan_cycles"], 0.0)
                self.assertEqual(
                    result["objective_contract"]["active_objective"],
                    "min T")
                self.assertTrue(
                    result["objective_contract"]["protocol_completion_in_T"])
                self.assertIsNone(result["secondary_objective"])

    def test_unreferenced_primary_start_supplies_immediate_incumbent(self):
        """A loose budget must not strand the min-T solve without an incumbent."""
        with tempfile.TemporaryDirectory(prefix="ilp_primary_start_") as raw:
            directory = Path(raw)
            params_path = directory / "params.json"
            output_path = directory / "solution.json"
            with params_path.open("w") as stream:
                json.dump(tiny_params(), stream)
            command = [
                str(self.python), str(ILP), "--params", str(params_path),
                "--L", "1", "--X", "1", "--Y", "4",
                "--a-max", "4", "--p-max", "8", "--n-cpu", "1",
                "--time-limit", "2",
                "--mipgap", "1e-9", "--threads", "1",
                "--quiet", "-o", str(output_path),
            ]
            completed = subprocess.run(
                command, cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
            self.assertEqual(
                completed.returncode, 0,
                msg="command failed\nSTDOUT:\n%s\nSTDERR:\n%s" %
                    (completed.stdout, completed.stderr))
            with output_path.open() as stream:
                result = json.load(stream)
            self.assertEqual(result["optimization_mode"], "ilp-perf")

            # ILP-Perf is a single solve: all legacy follow-up stages must
            # remain absent and their JSON diagnostics must be explicit.
            json.dumps(result, allow_nan=False)
            for stage in ("secondary", "tertiary", "quaternary"):
                self.assertFalse(result[stage + "_has_incumbent"])
                certificate = result[stage + "_gap_certificate"]
                self.assertFalse(certificate["certified"])
                self.assertEqual(certificate["reason"], "stage-not-run")
                self.assertIsNone(result.get(stage + "_mip_gap"))

            start = result["stage0_feasible_start"]
            self.assertTrue(start["attempted"])
            self.assertTrue(start["accepted_for_submission"])
            self.assertGreater(start["mip_start_variable_count"], 0)
            self.assertEqual(start["mip_start_undefined_variable_count"], 0)
            self.assertEqual(start["mip_start_undefined_variables"], [])
            self.assertEqual(
                start["strategy"],
                "performance-first 0/0 fastest expert upgrades")
            self.assertIn("performance-first", start["selection_policy"])
            self.assertIn("CPU center-first", start["packing_policy"])
            self.assertEqual(start["upgrade_candidate_count"], 0)
            self.assertEqual(start["requested_upgrade_count"], 0)
            self.assertEqual(start["realized_upgrade_count"], 0)
            self.assertTrue(start["all_gpu_s"])
            self.assertEqual(start["type_counts"], {"CPU": 1, "GPU-S": 2})
            self.assertEqual(start["bundle_count"], 3)
            self.assertGreater(start["legacy_task_T_start_cycles"], 0.0)
            self.assertGreater(start["protocol_T_start_cycles"], 0.0)
            self.assertGreaterEqual(
                start["analytical_T_start_cycles"] + 1e-7,
                start["protocol_T_start_cycles"])
            refinement = start["protocol_placement_refinement"]
            self.assertTrue(refinement["attempted"])
            self.assertGreater(refinement["evaluation_limit"], 0)
            self.assertGreater(refinement["evaluations"], 0)
            self.assertLessEqual(
                refinement["evaluations"], refinement["evaluation_limit"])
            self.assertFalse(refinement["coordinates_fixed"])
            self.assertFalse(refinement["bom_fixed_by_refinement"])
            self.assertGreaterEqual(refinement["wall_s"], 0.0)
            self.assertEqual(refinement["wall_limit_s"], 2.0)
            self.assertLess(refinement["wall_s"], 2.5)
            self.assertLessEqual(
                refinement["final_cycles"],
                refinement["initial_cycles"] + 1e-7)
            self.assertEqual(
                result["solver_configuration"]["gurobi_mip_focus"], 1)
            self.assertFalse(
                result["solver_configuration"]["changes_objective"])
            self.assertLess(result["time_to_first_feasible_s"], 1.0)

    def run_stage0_start_case(self, directory, params, label, grid,
                              area_limit, power_limit, time_limit="2"):
        params_path = directory / (label + "_params.json")
        output_path = directory / (label + "_solution.json")
        with params_path.open("w") as stream:
            json.dump(params, stream)
        command = [
            str(self.python), str(ILP), "--params", str(params_path),
            "--L", "1", "--X", str(grid), "--Y", str(grid),
            "--a-max", str(area_limit), "--p-max", str(power_limit),
            "--n-cpu", "1", "--mode", "ilp-perf",
            "--noc-traffic", "task-dag", "--time-limit", time_limit,
            "--mipgap", "1e-9", "--mip-focus", "1",
            "--threads", "1", "--quiet",
            "-o", str(output_path),
        ]
        completed = subprocess.run(
            command, cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        self.assertEqual(
            completed.returncode, 0,
            msg="%s failed\nSTDOUT:\n%s\nSTDERR:\n%s" %
                (label, completed.stdout, completed.stderr))
        with output_path.open() as stream:
            return json.load(stream)

    def test_stage0_performance_first_budget_ladder_and_hot_upgrades(self):
        params = mixed_bom_params(8)
        tiers = [
            ("a970", 970.0, 0),
            ("a1260", 1260.0, 1),
            ("a1550", 1550.0, 2),
            ("a2130", 2130.0, 4),
        ]
        observed_gpu_l = []
        with tempfile.TemporaryDirectory(prefix="ilp_stage0_ladder_") as raw:
            directory = Path(raw)
            for label, area_limit, expected_gpu_l in tiers:
                with self.subTest(label=label):
                    result = self.run_stage0_start_case(
                        directory, params, label, 6,
                        area_limit, 1625.0)
                    start = result["stage0_feasible_start"]
                    self.assertTrue(start["accepted_for_submission"])
                    self.assertEqual(start["upgrade_candidate_count"], 8)
                    self.assertEqual(start["requested_upgrade_count"],
                                     expected_gpu_l)
                    self.assertEqual(start["realized_upgrade_count"],
                                     expected_gpu_l)
                    self.assertEqual(start["type_counts"], {
                        "CPU": 1, "GPU-L": expected_gpu_l,
                        "GPU-S": 8 - expected_gpu_l})
                    self.assertEqual(result["selected_type_counts"], {
                        "CPU": 1, "GPU-L": expected_gpu_l,
                        "GPU-S": 8 - expected_gpu_l})
                    selection = result["resource_selection"]
                    self.assertEqual(
                        selection["count_definition"], "N_c=sum_p x[c,p]")
                    self.assertTrue(
                        selection["joint_with_mapping_and_placement"])
                    self.assertFalse(selection["fixed_bom"])
                    self.assertGreaterEqual(selection["area_slack_mm2"], -1e-6)
                    self.assertGreaterEqual(
                        selection["peak_power_slack_W"], -1e-6)
                    self.assertEqual(
                        start["all_gpu_s"], expected_gpu_l == 0)
                    assignment = start["expert_type_assignment"]
                    self.assertEqual(
                        [assignment["e(%d)" % index]
                         for index in range(8)],
                        (["GPU-L"] * expected_gpu_l +
                         ["GPU-S"] * (8 - expected_gpu_l)))
                    self.assertEqual(
                        start["upgrade_attempts"][0][
                            "requested_upgrade_count"], 8)
                    self.assertTrue(start["upgrade_attempts"][-1]["accepted"])
                    self.assertEqual(start["cpu_anchor"][0], 0)
                    self.assertLessEqual(
                        abs(start["cpu_anchor"][1] - 2.5) +
                        abs(start["cpu_anchor"][2] - 2.5), 2.0)
                    if expected_gpu_l == 0:
                        task_anchor = {}
                        for bundle in start["bundles"]:
                            for task in bundle["tasks"]:
                                task_anchor[task] = bundle["anchor"]
                        cpu = start["cpu_anchor"]
                        distance = lambda task: sum(
                            abs(task_anchor[task][axis] - cpu[axis])
                            for axis in range(3))
                        self.assertLessEqual(distance("e(0)"), distance("e(7)"))
                    self.assertEqual(
                        result["objective_contract"]["active_objective"],
                        "min T")
                    self.assertEqual(
                        result["solver_configuration"]["gurobi_mip_focus"],
                        1)
                    selected_serial = 100.0 + sum(
                        params["tasks"][expert]["cyc"][assignment[expert]]
                        for expert in params["expert_tasks"])
                    self.assertLess(
                        start["legacy_task_T_start_cycles"], selected_serial)
                    observed_gpu_l.append(start["type_counts"]["GPU-L"])
        self.assertEqual(observed_gpu_l, [0, 1, 2, 4])
        self.assertEqual(observed_gpu_l, sorted(observed_gpu_l))

    def test_stage0_larger_budget_is_not_all_small(self):
        # This is intentionally solver-light (16 experts, empty secondary
        # stages) but large enough to catch the former unconditional all-S
        # preference that poisoned scaling incumbents.
        params = mixed_bom_params(16)
        with tempfile.TemporaryDirectory(prefix="ilp_stage0_e16_") as raw:
            result = self.run_stage0_start_case(
                Path(raw), params, "e16", 8, 3010.0, 2345.0,
                time_limit="2")
        start = result["stage0_feasible_start"]
        self.assertTrue(start["accepted_for_submission"])
        self.assertFalse(start["all_gpu_s"])
        self.assertEqual(start["requested_upgrade_count"], 4)
        self.assertEqual(start["type_counts"], {
            "CPU": 1, "GPU-L": 4, "GPU-S": 12})
        self.assertEqual(start["expert_type_assignment"]["e(0)"], "GPU-L")

    def run_bom_case(self, directory, params, reference, extra=None,
                     expect_success=True):
        params_path = directory / "custom_params.json"
        reference_path = directory / "custom_reference.json"
        output_path = directory / "custom_solution.json"
        with params_path.open("w") as stream:
            json.dump(params, stream)
        with reference_path.open("w") as stream:
            json.dump(reference, stream)
        command = [
            str(self.python), str(ILP), "--params", str(params_path),
            "--bom-from", str(reference_path),
            "--L", "1", "--X", "1", "--Y", str(len(reference["sites"])),
            "--a-max", "20", "--p-max", "20", "--n-cpu", "1",
            "--mode", "ilp-perf", "--noc-traffic", "task-dag",
            "--time-limit", "5", "--mipgap", "1e-9",
            "--threads", "1", "--quiet", "-o", str(output_path),
        ]
        command.extend(extra or [])
        completed = subprocess.run(
            command, cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        self._last_bom_stderr = completed.stderr
        if not expect_success:
            self.assertNotEqual(completed.returncode, 0)
            return None
        self.assertEqual(
            completed.returncode, 0,
            msg="command failed\nSTDOUT:\n%s\nSTDERR:\n%s" %
                (completed.stdout, completed.stderr))
        with output_path.open() as stream:
            return json.load(stream)

    @staticmethod
    def sharing_params(expert_slots, two_gpus=False, edge_volume=None):
        library = {
            "CPU": {"w": 1, "h": 1, "area": 1.0, "p_peak": 1.0,
                    "mem": 1024.0, "expert_slots": 0},
            "GPU-S": {"w": 1, "h": 1, "area": 1.0, "p_peak": 2.0,
                      "mem": 1024.0, "expert_slots": expert_slots},
        }
        tasks = {
            "control": {"allowed": ["CPU"], "mem_mb": 1.0,
                        "cyc": {"CPU": 1.0}, "eta": {"CPU": 1e-9}},
            "e(0)": {"allowed": ["GPU-S"], "mem_mb": 1.0,
                     "cyc": {"GPU-S": 10.0},
                     "eta": {"GPU-S": 1e-9}},
            "e(1)": {"allowed": ["GPU-S"], "mem_mb": 1.0,
                     "cyc": {"GPU-S": 20.0},
                     "eta": {"GPU-S": 1e-9}},
        }
        edges = []
        if edge_volume is not None:
            tasks.pop("e(1)")
            edges = [{"src": "control", "dst": "e(0)",
                      "vol": float(edge_volume)}]
        return {
            "meta": {"note": "sharing/link-capacity unit test"},
            "library": library, "tasks": tasks,
            "expert_tasks": [name for name in tasks if name.startswith("e(")],
            "edges": edges,
            "interconnect": {"tau_h": 0.0, "tau_v": 0.0,
                             "eps_h": 1e-12, "eps_v": 1e-13,
                             "t0": 0.0,
                             "beta": 1.0 if edge_volume is not None else 16.0},
        }

    def test_expert_slots_and_conditional_colocated_order(self):
        reference = {
            "sites": [
                {"l": 0, "i": 0, "j": 0, "type": "CPU", "tasks": []},
                {"l": 0, "i": 0, "j": 1, "type": "GPU-S", "tasks": []},
            ], "task_site": {}}
        with tempfile.TemporaryDirectory(prefix="ilp_share_smoke_") as raw:
            directory = Path(raw)
            self.run_bom_case(
                directory, self.sharing_params(1), reference,
                expect_success=False)
            result = self.run_bom_case(
                directory, self.sharing_params(2), reference)
            self.assertEqual(result["expert_slots"]["GPU-S"], 2)
            self.assertEqual(result["task_site"]["e(0)"],
                             result["task_site"]["e(1)"])
            first = result["task_schedule"]["e(0)"]
            second = result["task_schedule"]["e(1)"]
            self.assertTrue(first["finish"] <= second["start"] + 1e-8 or
                            second["finish"] <= first["start"] + 1e-8)
            self.assertGreaterEqual(result["predicted_makespan_cycles"],
                                    30.0 - 1e-8)

    def test_replay_protocol_serializes_colocated_expert_compute(self):
        reference = {
            "sites": [
                {"l": 0, "i": 0, "j": 0,
                 "type": "CPU", "tasks": []},
                {"l": 0, "i": 0, "j": 1,
                 "type": "GPU-S", "tasks": []},
            ], "task_site": {}}
        with tempfile.TemporaryDirectory(
                prefix="ilp_protocol_share_smoke_") as raw:
            result = self.run_bom_case(
                Path(raw), self.sharing_params(2), reference,
                extra=["--noc-traffic", "deterministic-dispatch"])
        protocol = result["protocol_model"]
        self.assertFalse(protocol["diagnostic_only"])
        self.assertTrue(protocol["used_as_objective"])
        self.assertTrue(protocol["included_in_T"])
        self.assertTrue(protocol["co_resident_compute_nonoverlap"])
        schedule = protocol["compute_schedule"]
        first, second = schedule["e(0)"], schedule["e(1)"]
        self.assertEqual(first["resource_site"], second["resource_site"])
        self.assertTrue(
            first["finish"] <= second["start"] + 1e-8 or
            second["finish"] <= first["start"] + 1e-8)
        self.assertGreaterEqual(
            result["predicted_makespan_cycles"] + 1e-7,
            result["protocol_makespan_cycles"])
        self.assertEqual(
            result["objective_contract"]["legacy_task_schedule_role"],
            "hard-feasibility-only")

    def test_different_chiplets_can_overlap(self):
        reference = {
            "sites": [
                {"l": 0, "i": 0, "j": 0, "type": "CPU", "tasks": []},
                {"l": 0, "i": 0, "j": 1, "type": "GPU-S", "tasks": []},
                {"l": 0, "i": 0, "j": 2, "type": "GPU-S", "tasks": []},
            ], "task_site": {}}
        with tempfile.TemporaryDirectory(prefix="ilp_parallel_smoke_") as raw:
            result = self.run_bom_case(
                Path(raw), self.sharing_params(1, two_gpus=True), reference)
            first = result["task_schedule"]["e(0)"]
            second = result["task_schedule"]["e(1)"]
            self.assertLess(first["start"], second["finish"])
            self.assertLess(second["start"], first["finish"])
            self.assertLessEqual(result["predicted_makespan_cycles"], 20.001)

    def test_link_capacity_and_removed_c9(self):
        reference = {
            "sites": [
                {"l": 0, "i": 0, "j": 0, "type": "CPU", "tasks": []},
                {"l": 0, "i": 0, "j": 1, "type": "GPU-S", "tasks": []},
            ], "task_site": {}}
        with tempfile.TemporaryDirectory(prefix="ilp_link_smoke_") as raw:
            result = self.run_bom_case(
                Path(raw), self.sharing_params(1, edge_volume=800), reference,
                extra=["--p-col", "0"])
            self.assertIn("--p-col/C9 is retired and ignored",
                          self._last_bom_stderr)
            self.assertNotIn("p_col", result["budgets"])
            self.assertGreaterEqual(result["predicted_makespan_cycles"],
                                    811.0 - 1e-8)
            routing = result["routing"]
            self.assertLessEqual(routing["maximum_link_utilization"],
                                 1.0 + 1e-8)
            self.assertLessEqual(routing["maximum_flow_residual_KiB"], 1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
