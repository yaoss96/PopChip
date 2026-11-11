#!/usr/bin/env python3
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_joint_bom_scaling_metrics as collector


def _solution():
    return {
        "sites": [
            {"type": "CPU", "l": 0, "i": 0, "j": 0,
             "tasks": ["dispatch", "gate"]},
            {"type": "GPU-L", "l": 0, "i": 1, "j": 1,
             "tasks": ["e0"]},
            {"type": "GPU-S", "l": 0, "i": 3, "j": 3,
             "tasks": ["e1"]},
        ],
        "task_site": {
            "dispatch": [0, 0, 0], "gate": [0, 0, 0],
            "e0": [0, 1, 1], "e1": [0, 3, 3],
        },
    }


class BundleContractTests(unittest.TestCase):
    def test_complete_bundle_move_preserves_signature(self):
        base = _solution()
        moved = copy.deepcopy(base)
        moved["sites"][1].update({"l": 0, "i": 2, "j": 2})
        moved["task_site"]["e0"] = [0, 2, 2]
        self.assertEqual(
            collector._bundle_multiset(base, "base"),
            collector._bundle_multiset(moved, "moved"))

    def test_task_site_coordinate_disagreement_is_rejected(self):
        value = _solution()
        value["task_site"]["e0"] = [0, 2, 2]
        with self.assertRaisesRegex(ValueError, "disagrees"):
            collector._bundle_multiset(value, "bad-coordinate")

    def test_duplicate_anchor_is_rejected(self):
        value = _solution()
        value["sites"][2].update({"l": 0, "i": 1, "j": 1})
        value["task_site"]["e1"] = [0, 1, 1]
        with self.assertRaisesRegex(ValueError, "multiple bundles"):
            collector._bundle_multiset(value, "duplicate-anchor")

    def test_type_or_bundle_change_changes_signature(self):
        base = _solution()
        changed = copy.deepcopy(base)
        changed["sites"][1]["type"] = "GPU-S"
        self.assertNotEqual(
            collector._bundle_multiset(base, "base"),
            collector._bundle_multiset(changed, "changed"))

    def test_malformed_and_duplicate_tasks_are_rejected(self):
        malformed = _solution()
        malformed["sites"][0]["tasks"] = "dispatch"
        with self.assertRaises(ValueError):
            collector._bundle_multiset(malformed, "malformed")
        duplicate = _solution()
        duplicate["sites"][2]["tasks"].append("e0")
        with self.assertRaisesRegex(ValueError, "multiple bundles"):
            collector._bundle_multiset(duplicate, "duplicate-task")


if __name__ == "__main__":
    unittest.main()
