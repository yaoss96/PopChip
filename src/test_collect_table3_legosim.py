#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused fail-closed tests for collect_table3_legosim.py."""

from __future__ import print_function

import contextlib
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_table3_legosim as collector


class Table3CollectorTest(unittest.TestCase):
    def test_latex_rows_identify_metric_and_power_scope(self):
        rows = [{
            "experts": 8,
            "mesh": "6x6",
            "gpu_l_s": "2/6",
            "ilp_synchronized_cycles": 104859,
            "ilp_orion_noc_power_W": 0.541871,
            "floorplet_synchronized_cycles": 103950,
            "floorplet_orion_noc_power_W": 0.408341,
            "tap_synchronized_cycles": 105379,
            "tap_orion_noc_power_W": 0.462949,
        }]
        text = collector._latex_text(rows)
        self.assertIn("synchronized LEGOSim", text)
        self.assertIn("NoC only", text)
        self.assertIn("104{,}859", text)
        self.assertIn("0.541871", text)

    def test_missing_manifest_and_incomplete_suite_write_no_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results.json").write_text(json.dumps({
                "schema_version": 1,
                "status": "running",
                "finished_utc": None,
                "rounds": 2,
                "codegen_only": False,
                "params_path": str(root / "missing_params.json"),
                "params_sha256": "0" * 64,
                "layouts": [],
            }), encoding="utf-8")
            prefix = root / "must_not_exist"
            with contextlib.redirect_stderr(io.StringIO()):
                code = collector.main([
                    str(root), "--out-prefix", str(prefix)])
            self.assertEqual(code, 1)
            self.assertFalse(Path(str(prefix) + ".json").exists())
            self.assertFalse(Path(str(prefix) + ".csv").exists())
            self.assertFalse(Path(str(prefix) + ".tex").exists())

    def test_tampered_input_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "schema_version": 1,
                "expected_input_count": 15,
                "actual_input_count": 15,
                "all_input_audits_pass": False,
                "all_scales_exact_bom_match": True,
                "all_scales_exact_bundle_match": True,
                "input_policy": {},
                "global_audits": {},
                "scale_audits": [],
                "inputs": [],
            }
            (root / collector.FINAL_INPUT_MANIFEST).write_text(
                json.dumps(manifest), encoding="utf-8")
            errors = []
            bindings, _ = collector._validate_final_input_manifest(
                root, errors)
            self.assertEqual(bindings, {})
            self.assertTrue(any("all_input_audits_pass" in error
                                for error in errors))

    def test_ilp_copy_relabelled_as_proxy_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proxy = root / "canonical_floorplet.json"
            copied_ilp = root / "copied_ilp_labelled_floorplet.json"
            proxy.write_bytes(b'{"placement":"proxy"}\n')
            copied_ilp.write_bytes(b'{"placement":"ilp"}\n')
            canonical = {
                "solution_path": proxy.resolve(),
                "solution_sha256": hashlib.sha256(
                    proxy.read_bytes()).hexdigest(),
            }
            errors = []
            collector._validate_record_canonical_identity(
                copied_ilp.resolve(),
                hashlib.sha256(copied_ilp.read_bytes()).hexdigest(),
                canonical, errors)
            self.assertTrue(any("solution path" in error for error in errors))
            self.assertTrue(any("solution SHA-256" in error
                                for error in errors))

    def test_legacy_proxy_and_secondary300_paths_are_rejected(self):
        for raw in (
                "out/scaling_prior_proxy_experiment_20260731/solutions/x.json",
                "out/scaling_ilp_secondary300_explore_20260801/solutions/x.json"):
            errors = []
            collector._canonical_path(raw, "canonical solution", errors)
            self.assertTrue(any("forbidden legacy path" in error
                                for error in errors))
            nested_errors = []
            collector._reject_legacy_manifest_strings(
                {"hidden": [raw]}, "manifest", nested_errors)
            self.assertTrue(any("forbidden legacy input marker" in error
                                for error in nested_errors))

    def test_staged_manifest_detects_semantic_input_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codegen = root / "codegen"
            evidence = root / "evidence"
            codegen.mkdir()
            evidence.mkdir()
            names = [
                "coords.json", "dispatch_matrix.txt", "dispatch_meta.json",
                "moe_coords.txt", "params.input.json", "solution.input.json",
            ]
            records = []
            for name in names:
                path = codegen / name
                path.write_text("original-%s\n" % name, encoding="utf-8")
                records.append("%s  ./%s" % (
                    hashlib.sha256(path.read_bytes()).hexdigest(), name))
            catalog = "\n".join(records) + "\n"
            for name in ("source_input_manifest.sha256",
                         "staged_input_manifest.sha256"):
                (evidence / name).write_text(catalog, encoding="utf-8")
            (codegen / "moe_coords.txt").write_text(
                "tampered\n", encoding="utf-8")
            errors = []
            collector._bind_codegen_to_staged_manifests(
                codegen, evidence, errors)
            self.assertTrue(any("moe_coords.txt staged SHA-256" in error
                                for error in errors))

    def test_coordinate_and_mesh_topology_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coords = root / "moe_coords.txt"
            coords.write_text(
                "width 2\nrouter 0 0\nngpu 2\n"
                "gpu 0 1 0\ngpu 1 1 0\n", encoding="utf-8")
            errors = []
            collector._parse_moe_coords(coords, 2, 2, 2, errors)
            self.assertTrue(any("coordinates are not unique" in error
                                for error in errors))

            topology = root / "mesh_2x2.gv"
            topology.write_text(
                "graph mesh_2x2\n{\n0--1\n0--2\n2--3\n}\n",
                encoding="utf-8")
            errors = []
            collector._validate_topology(topology, 2, 2, errors)
            self.assertTrue(any("edge set" in error for error in errors))

    def test_dispatch_margin_tamper_is_rejected(self):
        matrix = [[32, 32], [32, 32]]
        metadata = {
            "ngpu": 2,
            "groups": 2,
            "topk": 2,
            "tokens_per_source": 32,
            "total_route_assignments": 128,
            "per_source_route_assignments": [64, 64],
            "group_to_worker": {"0": 0, "1": 1},
            "worker_token_assignments": [64, 64],
            "global_group_actual_assignments": [64, 64],
            "global_group_target_assignments": [65, 63],
            "global_group_error_assignments": [0, 0],
            "global_group_max_abs_error": 0,
            "zero_worker_count": 0,
            "zero_workers": [],
        }
        errors = []
        collector._validate_dispatch_semantics(
            matrix, metadata, {"meta": {"group_pi": [0.5, 0.5]}},
            2, errors)
        self.assertTrue(any("logical target totals" in error
                            for error in errors))

    def test_old_pass_evidence_recomputes_raw_metrics(self):
        results_path = (collector.PROJECT_ROOT /
                        "out/scaling_legosim_real_v2_20260801/e008/results.json")
        if not results_path.is_file():
            self.skipTest("old PASS evidence fixture is unavailable")
        manifest = json.loads(results_path.read_text(encoding="utf-8"))
        record = next(row for row in manifest["layouts"]
                      if row["method"] == "ilp")
        audited = collector.audit._audit_layout(
            results_path.resolve(), manifest, 8, record, 2)
        errors = []
        raw = collector._stream_raw_evidence(
            Path(audited["evidence_dir"]), 8, audited, errors)
        self.assertEqual(errors, [])
        self.assertEqual(raw["synchronized_cycles_from_interchiplet"], 104859)
        self.assertEqual(raw["synchronized_cycles_from_router"], 104859)
        self.assertEqual(raw["bench_cycles_from_raw_trace"], 104859)
        self.assertEqual(raw["gpu_cycle_rows_from_raw_logs"], 8)
        self.assertEqual(raw["gpu_cycle_max_from_raw_logs"], 22973)
        self.assertEqual(raw["popnet_round2_latest"]["finished"], 128)
        self.assertEqual(raw["popnet_round2_latest"]["power"], 0.541871)
        self.assertEqual(raw["link_payloads_accepted"], 0)

    def test_raw_tar_hash_path_and_link_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            tar_path = evidence / "run_evidence.tar.gz"
            manifest_payload = (
                ("0" * 64) + "  ./bench.txt\n").encode("utf-8")
            with tarfile.open(str(tar_path), "w:gz") as archive:
                for name, payload in (
                        ("./raw_output_manifest.sha256", manifest_payload),
                        ("./bench.txt", b"1 2 0 0 0 0\n"),
                        ("../escape", b"must-not-be-extracted")):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                link = tarfile.TarInfo("./evil-link")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../outside"
                archive.addfile(link)
            errors = []
            collector._stream_raw_evidence(
                evidence, 1, {
                    "legosim_cycles": 2,
                    "bench_cycles": 2,
                    "completion_delta_cycles": 0,
                    "popnet_expected_finished": 1,
                    "popnet_total_finished": 1,
                    "popnet_average_delay": 1.0,
                    "orion_noc_power_W": 1.0,
                    "gpu_cycle_max": 1,
                }, errors)
            self.assertTrue(any("unsafe archive path" in error
                                for error in errors))
            self.assertTrue(any("unsupported link evil-link" in error
                                for error in errors))
            self.assertTrue(any("raw tar member bench.txt SHA-256" in error
                                for error in errors))
            self.assertFalse((evidence.parent / "escape").exists())

    def test_expected_mesh_and_bom_are_fixed(self):
        self.assertEqual(collector.EXPECTED_SCALES[8], {
            "mesh_side": 6, "gpu_l": 2, "gpu_s": 6})
        self.assertEqual(collector.EXPECTED_SCALES[128], {
            "mesh_side": 16, "gpu_l": 32, "gpu_s": 96})
        self.assertEqual(collector._exact_bom(64), {
            "CPU": 1, "GPU-L": 16, "GPU-S": 48})


if __name__ == "__main__":
    unittest.main()
