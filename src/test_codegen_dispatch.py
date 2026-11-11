#!/usr/bin/env python3
"""Lightweight regression tests for formal large-E dispatch generation."""

import json
import re
import tempfile
import unittest
from pathlib import Path

try:
    from src import codegen
except ImportError:  # direct: python src/test_codegen_dispatch.py
    import codegen


SCALES = (8, 16, 32, 64, 128)
TOKENS_PER_SOURCE = 32
TOPK = 2
ASSIGNMENTS_PER_SOURCE = TOKENS_PER_SOURCE * TOPK


def _zipf_weights(experts, alpha=1.15):
    return [1.0 / ((group + 1) ** alpha) for group in range(experts)]


def _solution(groups_by_worker):
    experts = len(groups_by_worker)
    # The list position chooses a coordinate only.  Protocol worker IDs must
    # come from the logical e(g) identity, never from this coordinate order.
    sites = []
    for worker, group in enumerate(groups_by_worker):
        sites.append({
            "l": 0,
            "i": worker,
            "j": 0,
            "type": "GPU-S",
            "tasks": ["e(%d)" % group],
        })
    return {
        "grid": {"L": 1, "X": experts + 1, "Y": 1},
        "sites": sites,
    }


def _params(experts, weights=None):
    return {
        "expert_tasks": ["e(%d)" % group for group in range(experts)],
        "meta": {
            "experts": experts,
            "groups": experts,
            "topk": TOPK,
            "group_pi": list(weights or _zipf_weights(experts)),
        },
    }


def _sim_cfg():
    return {
        "tokens_per_source": TOKENS_PER_SOURCE,
        "hidden": 128,
        "intermediate": 64,
        "ffn_repeat": 1,
    }


def _read_physical_matrix(path, experts):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines)
                 if line.startswith("round ")) + 1
    return [
        [int(value) for value in lines[start + row].split()]
        for row in range(experts)
    ]


def _logical_matrix(physical, group_to_worker):
    experts = len(physical)
    owner = [int(group_to_worker[str(group)]) for group in range(experts)]
    return [
        [physical[owner[source]][owner[destination]]
         for destination in range(experts)]
        for source in range(experts)
    ]


class FormalDispatchTests(unittest.TestCase):
    def _emit(self, experts, groups_by_worker, root, suffix):
        out_dir = Path(root) / suffix
        out_dir.mkdir()
        meta = codegen.emit_dispatch_matrix(
            _solution(groups_by_worker), _params(experts), str(out_dir),
            _sim_cfg())
        physical = _read_physical_matrix(
            out_dir / "dispatch_matrix.txt", experts)
        logical = _logical_matrix(physical, meta["group_to_worker"])
        return out_dir, meta, physical, logical

    def test_e8_to_e128_exact_margins_determinism_and_layout_invariance(self):
        for experts in SCALES:
            with self.subTest(experts=experts), tempfile.TemporaryDirectory() as root:
                identity = list(range(experts))
                reverse = list(reversed(identity))
                rotated = identity[experts // 3:] + identity[:experts // 3]
                layouts = (identity, reverse, rotated)

                emitted = [
                    self._emit(experts, permutation, root, "layout%d" % index)
                    for index, permutation in enumerate(layouts)
                ]
                reference_logical = emitted[0][3]
                reference_protocol = (
                    emitted[0][0] / "dispatch_matrix.txt").read_bytes()
                expected_columns = codegen._largest_remainder(
                    _zipf_weights(experts),
                    experts * ASSIGNMENTS_PER_SOURCE)
                expected_owner = {
                    str(group): group for group in range(experts)
                }

                for out_dir, meta, physical, logical in emitted:
                    self.assertEqual(logical, reference_logical)
                    # Moving an expert changes only the worker's coordinate.
                    # Both protocol axes remain in logical group order, so the
                    # file consumed by dispatch and combine is byte-identical.
                    self.assertEqual(physical, reference_logical)
                    self.assertEqual(meta["group_to_worker"], expected_owner)
                    self.assertEqual(
                        meta["worker_id_semantics"],
                        "worker_id_equals_logical_group_id_q1")
                    self.assertEqual(
                        (out_dir / "dispatch_matrix.txt").read_bytes(),
                        reference_protocol)
                    self.assertEqual(
                        [entry["worker"] for entry in meta["workers"]],
                        list(range(experts)))
                    self.assertEqual(
                        [entry["tasks"] for entry in meta["workers"]],
                        [["e(%d)" % group] for group in range(experts)])
                    self.assertEqual(
                        [sum(row) for row in physical],
                        [ASSIGNMENTS_PER_SOURCE] * experts)
                    physical_columns = [
                        sum(physical[row][column] for row in range(experts))
                        for column in range(experts)
                    ]
                    logical_columns = [
                        sum(logical[row][column] for row in range(experts))
                        for column in range(experts)
                    ]
                    self.assertEqual(logical_columns, expected_columns)
                    self.assertTrue(all(value > 0 for value in physical_columns))
                    self.assertEqual(sum(map(sum, physical)),
                                     experts * ASSIGNMENTS_PER_SOURCE)
                    self.assertEqual(meta["topk"], TOPK)
                    self.assertEqual(meta["tokens_per_source"],
                                     TOKENS_PER_SOURCE)
                    self.assertEqual(meta["total_route_assignments"],
                                     experts * ASSIGNMENTS_PER_SOURCE)
                    self.assertEqual(meta["global_group_target_assignments"],
                                     expected_columns)
                    self.assertEqual(meta["global_group_actual_assignments"],
                                     expected_columns)
                    self.assertEqual(meta["global_group_error_assignments"],
                                     [0] * experts)
                    self.assertEqual(meta["global_group_max_abs_error"], 0)
                    self.assertEqual(meta["zero_worker_count"], 0)
                    self.assertEqual(meta["zero_workers"], [])
                    self.assertFalse(meta["per_source_rows_replicated"])
                    self.assertGreater(len({tuple(row) for row in logical}), 1)

                # Same solution and inputs must reproduce both files byte for byte.
                repeat_dir, repeat_meta, _, repeat_logical = self._emit(
                    experts, identity, root, "repeat")
                self.assertEqual(repeat_logical, reference_logical)
                self.assertEqual(repeat_meta, emitted[0][1])
                self.assertEqual(
                    (emitted[0][0] / "dispatch_matrix.txt").read_bytes(),
                    (repeat_dir / "dispatch_matrix.txt").read_bytes())
                self.assertEqual(
                    (emitted[0][0] / "dispatch_meta.json").read_bytes(),
                    (repeat_dir / "dispatch_meta.json").read_bytes())

    def test_bench_and_coord_protocol_ids_follow_groups_not_coordinates(self):
        experts = 16
        # e(0) is deliberately placed at the largest coordinate, e(15) at the
        # smallest, and the serialized site list is reversed once more.  This
        # catches both coordinate sorting and input-list-order regressions.
        sol = _solution(list(reversed(range(experts))))
        sol["sites"] = list(reversed(sol["sites"]))
        params = _params(experts)
        workers = codegen.gpu_worker_sites(sol, params)
        self.assertEqual(
            [[task for task in site["tasks"] if task.startswith("e(")]
             for site in workers],
            [["e(%d)" % group] for group in range(experts)])

        with tempfile.TemporaryDirectory() as root:
            codegen.emit_bench_yaml(
                sol, params, experts + 1, 1, root, sim_cfg=_sim_cfg())
            codegen.emit_coord_file(
                sol, params, experts + 1, 1, root)
            meta = codegen.emit_dispatch_matrix(
                sol, params, root, _sim_cfg())

            bench_pairs = []
            for line in (Path(root) / "bench.yml").read_text(
                    encoding="utf-8").splitlines():
                match = re.search(
                    r'args: \["(\d+)",.*tasks: e\((\d+)\)', line)
                if match:
                    bench_pairs.append(tuple(int(value)
                                             for value in match.groups()))
            self.assertEqual(
                bench_pairs, [(group, group) for group in range(experts)])

            coord_pairs = []
            for line in (Path(root) / "moe_coords.txt").read_text(
                    encoding="utf-8").splitlines():
                match = re.match(
                    r'gpu (\d+) .*tasks=e\((\d+)\)', line)
                if match:
                    coord_pairs.append(tuple(int(value)
                                             for value in match.groups()))
            self.assertEqual(
                coord_pairs, [(group, group) for group in range(experts)])
            self.assertEqual(
                meta["group_to_worker"],
                {str(group): group for group in range(experts)})

    def test_zero_token_expert_is_rejected(self):
        experts = 8
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "zero-token GPU workers"):
                codegen.emit_dispatch_matrix(
                    _solution(list(range(experts))),
                    _params(experts, weights=[1.0] + [0.0] * (experts - 1)),
                    root, _sim_cfg())

    def test_non_bijective_group_worker_mapping_is_rejected(self):
        experts = 8
        sol = _solution(list(range(experts)))
        sol["sites"][0]["tasks"].append("e(1)")
        sol["sites"][1]["tasks"] = []
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "exactly one logical"):
                codegen.emit_dispatch_matrix(
                    sol, _params(experts), root, _sim_cfg())

    def test_layout_report_accepts_uncertified_mip_gap(self):
        """A time-limited legal incumbent may have no finite MIP bound."""
        sol = _solution([0])
        sol.update({
            "predicted_makespan_cycles": 123.0,
            "mip_gap": None,
            "solver_wall_s": 30.0,
        })
        with tempfile.TemporaryDirectory() as root:
            report = codegen.emit_report(sol, _params(1), 2, 1, root)
            self.assertIn("MIP gap: uncertified", report)
            self.assertIn(
                "MIP gap: uncertified",
                (Path(root) / "layout_report.txt").read_text(
                    encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
