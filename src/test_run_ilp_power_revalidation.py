#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import contextlib
import io
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ilp_power_revalidation as driver


@unittest.skip(
    "retired: the dedicated ILP-Power revalidation workflow was superseded "
    "by the single-objective ILP-Perf experiment")
class ILPPowerRevalidationDriverTest(unittest.TestCase):
    def _cases(self):
        return (driver._budget_cases(driver._resolve(
                    driver.DEFAULT_BUDGET_SOURCE)) +
                driver._scale_cases(driver._resolve(
                    driver.DEFAULT_SCALE_SOURCE), driver._resolve(
                    driver.DEFAULT_SCALE_PARAMS)))

    def test_canonical_plan_has_13_cases_without_local_density_budget(self):
        cases = self._cases()
        self.assertEqual(len(cases), 13)
        self.assertEqual(sum(case["suite"] == "budget" for case in cases), 8)
        self.assertEqual(sum(case["suite"] == "scaling" for case in cases), 5)
        self.assertTrue(all("p_col" not in case["budgets"]
                            for case in cases))

    def test_solver_command_contains_every_fixed_protocol_argument(self):
        case = self._cases()[0]
        command = driver._solver_command(
            case, Path("solution.json"), Path(driver.DEFAULT_GUROBI_PYTHON))
        joined = "\n".join(command)
        for fragment in (
                "--mode\nilp-power",
                "--placement-from\n",
                "--cpu-anchor\nfree",
                "--power-perf-relax\n0.20",
                "--noc-traffic\ndeterministic-dispatch",
                "--dispatch-tokens-per-source\n32",
                "--dispatch-hidden\n128",
                "--threads\n1",
                "--time-limit\n30.0",
                "--secondary-time\n30.0"):
            self.assertIn(fragment, joined)
        self.assertNotIn("--p-col", command)

    def test_dry_run_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary) / "must_not_exist"
            with contextlib.redirect_stdout(io.StringIO()):
                code = driver.main([
                    "--dry-run", "--case", "budget:area_970",
                    "--out-root", str(out_root)])
            self.assertEqual(code, 0)
            self.assertFalse(out_root.exists())

    def test_solution_contract_uses_optimization_mode(self):
        # ilp_layout's serialized field is intentionally not the CLI name.
        self.assertNotEqual("mode", "optimization_mode")

    def test_replay_jobs_two_runs_two_isolated_workers(self):
        state = {"active": 0, "maximum": 0}
        lock = threading.Lock()

        def fake_run(command, stdout_path, stderr_path):
            del command, stdout_path, stderr_path
            with lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            time.sleep(0.03)
            with lock:
                state["active"] -= 1
            return 0

        invocations = [{
            "name": "job%d" % index,
            "cases": ["case%d" % index],
            "command": ["fake"],
            "out_dir": "out%d" % index,
            "status": "planned",
            "results_json": None,
            "estimated_cost_rank": 4 - index,
            "stdout_log": "stdout%d" % index,
            "stderr_log": "stderr%d" % index,
        } for index in range(4)]
        with mock.patch.object(driver, "_run_command", side_effect=fake_run):
            completed = driver._execute_replay_invocations(invocations, 2)
        self.assertEqual(state["maximum"], 2)
        self.assertEqual([item["name"] for item in completed],
                         ["job0", "job1", "job2", "job3"])
        self.assertTrue(all(item["status"] == "complete"
                            for item in completed))

    def test_replay_jobs_rejects_more_than_two(self):
        with self.assertRaises(ValueError):
            driver._execute_replay_invocations([], 3)

    def test_jobs_two_submission_is_longest_first(self):
        names = ["budget", "scaling_e008_m06", "scaling_e016_m08",
                 "scaling_e032_m12", "scaling_e064_m12",
                 "scaling_e128_m16"]
        invocations = [{"name": name,
                        "estimated_cost_rank": driver.REPLAY_COST_RANK[name]}
                       for name in names]
        scheduled = driver._replay_submission_order(invocations, 2)
        self.assertEqual([item["name"] for item in scheduled], [
            "scaling_e128_m16", "scaling_e064_m12",
            "scaling_e032_m12", "scaling_e016_m08",
            "scaling_e008_m06", "budget"])
        # Serial mode remains backward-compatible.
        self.assertEqual(driver._replay_submission_order(invocations, 1),
                         invocations)

    def test_replay_commands_set_both_timeout_layers(self):
        cases = self._cases()
        args = driver._parse_args([])
        out_root = Path("out/test-timeout-plan")
        for suite, expected_inner, expected_outer in (
                ("budget", 1800, 2400),
                ("scaling", 21600, 25200)):
            case = next(item for item in cases if item["suite"] == suite)
            command = driver._replay_command(
                [case], out_root / suite, Path(sys.executable), args)
            inner = int(command[command.index("--moe-e2e-timeout") + 1])
            outer = int(float(command[command.index("--legosim-timeout") + 1]))
            self.assertEqual(inner, expected_inner)
            self.assertEqual(outer, expected_outer)
            self.assertGreaterEqual(outer, inner + 600)

    def test_outer_timeout_margin_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            out_root = Path(temporary) / "no-output"
            with self.assertRaises(ValueError):
                driver.main([
                    "--dry-run", "--out-root", str(out_root),
                    "--scale-legosim-timeout", "22000",
                    "--scale-e2e-timeout", "21600"])
            self.assertFalse(out_root.exists())


if __name__ == "__main__":
    unittest.main()
