#!/usr/bin/env python3
"""Regression tests for the independent LEGOSim protocol surrogate."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    from src import evaluate_legosim_protocol_surrogate as protocol
except ImportError:  # direct: python src/test_evaluate_legosim_protocol_surrogate.py
    import evaluate_legosim_protocol_surrogate as protocol


GROUP_PI = [
    0.28115461648760093,
    0.17307096768742117,
    0.13030477798289436,
    0.10653767749028356,
    0.09113101678025085,
    0.08021199972289504,
    0.07200731062220163,
    0.06558163322645252,
]

GPU_CYCLES = [4663, 4670, 23726, 20619, 18338, 17002, 16488, 16120]

# Stable logical-worker coordinates from the audited A1550 rerun.  Coordinates
# are (x=column, y=row), exactly as written to moe_coords.txt.
PERF_COORDS = [
    (2, 0), (1, 2), (3, 3), (0, 1),
    (0, 2), (3, 2), (1, 1), (4, 0),
]
POWER_COORDS = [
    (3, 3), (4, 1), (2, 3), (2, 2),
    (3, 1), (3, 2), (1, 2), (2, 1),
]


def _params():
    return {
        "meta": {
            "config": "mixtral",
            "experts": 8,
            "groups": 8,
            "topk": 2,
            "bytes_elem": 2,
            "group_pi": list(GROUP_PI),
        },
        "expert_tasks": ["e(%d)" % group for group in range(8)],
    }


def _solution(coords, router_xy):
    sites = [{
        "l": 0,
        "i": int(router_xy[1]),
        "j": int(router_xy[0]),
        "type": "CPU",
        "tasks": ["t_gate", "t_norm"],
    }]
    for group, (x, y) in enumerate(coords):
        tasks = ["e(%d)" % group]
        if group == 0:
            tasks = ["t_attn", "t_dispatch", "e(0)", "t_combine"]
        sites.append({
            "l": 0,
            "i": int(y),
            "j": int(x),
            "type": "GPU-L" if group < 2 else "GPU-S",
            "tasks": tasks,
        })
    return {"grid": {"L": 1, "X": 6, "Y": 6}, "sites": sites}


class ProtocolSurrogateTests(unittest.TestCase):
    def test_a1550_stable_id_regression_matches_protocol_ranking(self):
        perf = protocol.evaluate_protocol_surrogate(
            _params(), _solution(PERF_COORDS, (0, 0)))
        power = protocol.evaluate_protocol_surrogate(
            _params(), _solution(POWER_COORDS, (4, 5)))

        self.assertEqual(perf["protocol_cycles"], 100933)
        self.assertEqual(power["protocol_cycles"], 99677)
        self.assertEqual(perf["protocol_cycles"] - power["protocol_cycles"],
                         1256)
        self.assertLess(power["protocol_cycles"], perf["protocol_cycles"])

        self.assertEqual(perf["message_count"], 128)
        self.assertEqual(power["message_count"], 128)
        self.assertEqual(perf["phases"]["init"]["message_count"], 8)
        self.assertEqual(perf["phases"]["dispatch"]["message_count"], 56)
        self.assertEqual(perf["phases"]["combine"]["message_count"], 56)
        self.assertEqual(perf["phases"]["summary"]["message_count"], 8)
        self.assertEqual(
            perf["phases"]["dispatch"]["assignment_hops"], 1334)
        self.assertEqual(
            power["phases"]["dispatch"]["assignment_hops"], 945)
        self.assertEqual(
            perf["dispatch"]["column_sums"],
            [144, 89, 67, 54, 47, 41, 37, 33])
        self.assertEqual(
            [worker["worker_id"] for worker in perf["workers"]],
            list(range(8)))
        self.assertEqual(
            [worker["compute_cycles"] for worker in perf["workers"]],
            GPU_CYCLES)

    def test_site_serialization_order_does_not_change_protocol_ids(self):
        solution = _solution(PERF_COORDS, (0, 0))
        reference = protocol.evaluate_protocol_surrogate(_params(), solution)
        solution["sites"] = list(reversed(solution["sites"]))
        reordered = protocol.evaluate_protocol_surrogate(_params(), solution)
        self.assertEqual(reordered["protocol_cycles"],
                         reference["protocol_cycles"])
        self.assertEqual(reordered["dispatch"]["sha256"],
                         reference["dispatch"]["sha256"])
        self.assertEqual(
            [worker["group"] for worker in reordered["workers"]],
            list(range(8)))

    def test_existing_gpu_cycles_tsv_and_json_type_profile_are_supported(self):
        solution = _solution(PERF_COORDS, (0, 0))
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            tsv = root / "gpu_cycles.tsv"
            tsv.write_text("".join(
                "proc_r2_p1_t%d/gpgpusim.0.%d.log\t%d\n" %
                (group + 1, group, GPU_CYCLES[group])
                for group in range(8)), encoding="utf-8")
            loaded_tsv = protocol.load_gpu_profile(str(tsv))
            tsv_result = protocol.evaluate_protocol_surrogate(
                _params(), solution, gpu_profile=loaded_tsv)
            self.assertEqual(tsv_result["protocol_cycles"], 100933)

            typed = {
                "source": "unit-test-typed-profile",
                "group_type_cycles": {
                    str(group): {
                        "GPU-L" if group < 2 else "GPU-S": GPU_CYCLES[group]
                    }
                    for group in range(8)
                },
            }
            json_path = root / "profile.json"
            json_path.write_text(json.dumps(typed), encoding="utf-8")
            loaded_json = protocol.load_gpu_profile(str(json_path))
            json_result = protocol.evaluate_protocol_surrogate(
                _params(), solution, gpu_profile=loaded_json)
            self.assertEqual(json_result["protocol_cycles"], 100933)
            self.assertEqual(json_result["gpu_profile"]["source"],
                             "unit-test-typed-profile")

    def test_q_greater_than_one_fails_closed_for_current_cpp_abi(self):
        solution = _solution(PERF_COORDS, (0, 0))
        # Move e(1) onto e(0)'s site and remove its original expert task.
        solution["sites"][1]["tasks"].append("e(1)")
        solution["sites"][2]["tasks"] = []
        with self.assertRaisesRegex(ValueError, "q=1 protocol surrogate"):
            protocol.evaluate_protocol_surrogate(_params(), solution)

    def test_include_events_exposes_endpoint_max_plus_fields(self):
        result = protocol.evaluate_protocol_surrogate(
            _params(), _solution(POWER_COORDS, (4, 5)),
            include_events=True)
        self.assertEqual(len(result["events"]), 128)
        first = result["events"][0]
        self.assertEqual(first["phase"], "init")
        self.assertEqual(first["write_cycle"], 1000)
        self.assertEqual(first["read_cycle"], 1000)
        self.assertEqual(first["source_finish"],
                         first["write_cycle"] + first["source_delay"])
        self.assertEqual(
            first["destination_finish"],
            max(first["write_cycle"] + first["destination_delay"],
                first["read_cycle"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
