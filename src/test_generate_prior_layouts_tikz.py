#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_prior_layouts_tikz as generator


CSV_FIELDS = [
    "schema_version", "suite", "experiment_id", "case_id",
    "sweep_dimension", "budget_value", "area_budget_mm2",
    "power_budget_W", "experts", "mesh", "method", "method_raw",
    "cpu_count", "gpu_l_count", "gpu_s_count",
    "shared_bom_peak_power_W", "synchronized_cycles",
    "orion_noc_power_W", "noc_active_cycles",
    "nominal_orion_dynamic_noc_energy_uJ", "legosim_validation",
    "audit_status", "evidence_tier", "solution_path", "solution_sha256",
    "evidence_dir", "source_results_json", "errors",
]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _sha(path):
    return generator._digest(path.read_bytes())


def _site(chip_type, row, column, tasks):
    return {"l": 0, "i": row, "j": column, "type": chip_type,
            "tasks": list(tasks)}


BUNDLES = {
    "cpu": ("CPU", ("t_gate", "t_norm")),
    "e0": ("GPU-L", ("e(0)", "t_attn", "t_combine")),
    "e1": ("GPU-L", ("e(1)",)),
    "e2": ("GPU-S", ("e(2)",)),
    "e3": ("GPU-S", ("e(3)",)),
    "e4": ("GPU-S", ("e(4)",)),
    "e5": ("GPU-S", ("e(5)", "t_dispatch")),
    "e6": ("GPU-S", ("e(6)",)),
    "e7": ("GPU-S", ("e(7)",)),
}

PLACEMENTS = {
    "ilp_perf": {
        "cpu": (0, 0), "e1": (0, 2), "e6": (1, 0),
        "e7": (2, 0), "e3": (2, 1), "e2": (2, 2),
        "e0": (2, 3), "e5": (3, 0), "e4": (3, 1),
    },
    "floorplet": {
        "e2": (0, 2), "e7": (1, 0), "e4": (1, 1),
        "e5": (1, 2), "e3": (1, 3), "e6": (1, 4),
        "cpu": (2, 1), "e0": (2, 2), "e1": (2, 4),
    },
    "tap": {
        "e3": (0, 3), "e7": (0, 4), "e4": (0, 5),
        "e5": (1, 3), "cpu": (1, 5), "e1": (2, 0),
        "e2": (2, 2), "e0": (2, 4), "e6": (3, 3),
    },
}


def _sites(method):
    output = []
    for name, (row, column) in PLACEMENTS[method].items():
        chip_type, tasks = BUNDLES[name]
        output.append(_site(chip_type, row, column, tasks))
    return output


def _task_site(sites):
    result = {}
    for site in sites:
        anchor = [site["l"], site["i"], site["j"]]
        for task in site["tasks"]:
            result[task] = anchor
    return result


def _directed_links():
    links = []
    for index, (src, dst) in enumerate(sorted(generator._expected_directed_links())):
        links.append({
            "id": index, "src": list(src), "dst": list(dst),
            "load_KiB": 1.0, "capacity_KiB": 1000.0,
            "utilization": 0.001,
            "phase_load_KiB": {"dispatch": 0.5, "combine": 0.5},
        })
    return links


def _ilp_solution():
    sites = _sites("ilp_perf")
    bundles = tuple(sorted(
        (site["type"], tuple(sorted(site["tasks"]))) for site in sites))
    value = {
        "optimization_mode": "ilp-perf",
        "mode_label": "ILP-Perf",
        "cpu_anchor_policy": "free",
        "primary_objective": "minimize joint MILP makespan T",
        "primary_has_incumbent": True,
        "primary_incumbent_cycles": 98229.0,
        "performance_cap_enforced": False,
        "performance_cap_cycles": None,
        "compute_selection_cap_cycles": None,
        "predicted_makespan_cycles": 98229.0,
        "performance_proxy_cycles": 98229.0,
        "tightened_performance_proxy_cycles": 98229.0,
        "protocol_makespan_cycles": 98229.0,
        "expert_slots": {"CPU": 1, "GPU-L": 1, "GPU-S": 1},
        "objective_contract": {
            "active_objective_count": 1,
            "active_objective": "min T",
            "joint_selection_placement_scheduling": True,
            "protocol_completion_in_T": True,
            "protocol_used_as_separate_objective": False,
            "legacy_task_schedule_role": "hard-feasibility-only",
            "energy_or_byte_hop_tie_break": False,
            "T_definition": (
                "fixed-order replay max-plus completion plus Mesh capacity epigraph"),
        },
        "protocol_model": {
            "included_in_T": True, "diagnostic_only": False,
            "used_as_objective": True, "used_as_separate_objective": False,
            "co_resident_compute_nonoverlap": True,
            "completion_cycles": 98229.0,
        },
        "noc_power_proxy": {
            "diagnostic_only": True, "used_as_objective": False,
        },
        "routing": {
            "kind": "phase-separated-source-aggregated-splittable-packet-flow",
            "capacity_horizon": "T", "directed_full_duplex": True,
            "directed_link_count": 120, "flow_scale_bytes": 1024.0,
            "link_loads": _directed_links(),
        },
        "grid": dict(generator.GRID),
        "budgets": {
            "a_max": 1550.0, "p_max": 1625.0, "a_max_layer": None,
            "nmin": {"CPU": 1}, "nmax": {"CPU": 1},
        },
        "sites": sites, "task_site": _task_site(sites),
        "total_area_mm2": 1550.0, "peak_power_W": 1205.0,
        "bundle_signature": generator._bundle_signature(bundles),
    }
    for stage in ("secondary", "tertiary", "quaternary"):
        value[stage + "_objective"] = None
        value[stage + "_objective_value"] = None
        value[stage + "_status"] = None
        value[stage + "_mip_gap"] = None
        value[stage + "_has_incumbent"] = False
        value[stage + "_wall_s"] = 0.0
        value[stage + "_gap_certificate"] = {
            "certified": False, "reason": "stage-not-run"}
    value["secondary_time_limit_s"] = None
    value["tertiary_time_limit_s"] = None
    return value


def _baseline_solution(method):
    sites = _sites(method)
    if method == "floorplet":
        name = "floorplet-inspired-exact-MILP"
    else:
        name = "tap-spread-SA"
    return {
        "method": name, "proxy_not_original_tool": True,
        "search_space": "exact BOM and immutable complete task bundles",
        "grid": dict(generator.GRID),
        "budgets": {
            "a_max": 1550.0, "p_max": 1625.0, "a_max_layer": None,
            "nmin": {"CPU": 1}, "nmax": {"CPU": 1},
        },
        "sites": sites, "task_site": _task_site(sites),
        "strict_evaluate": {
            "feasible": True, "reasons": [], "total_area_mm2": 1550.0,
            "peak_power_W": 1205.0,
        },
    }


def _write_csv(path, rows):
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS,
                                extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with open(str(path), "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _protocol():
    return json.loads(json.dumps(generator.MANIFEST_PROTOCOL))


def _fixture(root):
    canonical = (root / "canonical_final").resolve()
    paths = {
        "ilp_perf": canonical / "solutions/budget/area_1550/ilp_perf.json",
        "floorplet": canonical /
        "baselines/budget/area_1550/prior/floorplet.json",
        "tap": canonical / "baselines/budget/area_1550/prior/tap.json",
    }
    _write_json(paths["ilp_perf"], _ilp_solution())
    _write_json(paths["floorplet"], _baseline_solution("floorplet"))
    _write_json(paths["tap"], _baseline_solution("tap"))

    cycles = {"ilp_perf": 104401, "floorplet": 105910, "tap": 108476}
    replay_path = canonical / "replays/budget/results.json"
    replay = {
        "schema_version": 1, "status": "complete", "codegen_only": False,
        "rounds": 2, "layouts": [],
    }
    for method in generator.METHODS:
        key = method["key"]
        replay["layouts"].append({
            "case_id": generator.CASE_ID, "method": method["manifest"],
            "status": "ok", "solution_path": str(paths[key]),
            "solution_sha256": _sha(paths[key]),
            "legosim": {
                "status": "ok", "validation": "PASS",
                "synchronized_makespan_cycles": cycles[key],
            },
        })
    _write_json(replay_path, replay)

    case = {
        "key": generator.CASE_KEY, "suite": "budget",
        "case_id": generator.CASE_ID, "grid": dict(generator.GRID),
        "budgets": {"a_max": 1550.0, "p_max": 1625.0},
        "ilp_perf": {
            "status": "complete", "solution_path": str(paths["ilp_perf"]),
            "solution_sha256": _sha(paths["ilp_perf"]),
        },
        "prior": {
            "status": "complete",
            "floorplet_solution_path": str(paths["floorplet"]),
            "tap_solution_path": str(paths["tap"]),
            "solution_sha256": {
                "floorplet": _sha(paths["floorplet"]),
                "tap": _sha(paths["tap"]),
            },
        },
    }
    cases = [case]
    cases.extend({"suite": "budget", "case_id": case_id}
                 for case_id in generator.EXPECTED_BUDGET_CASES
                 if case_id != generator.CASE_ID)
    cases.extend({"suite": "scaling", "case_id": case_id}
                 for case_id in generator.EXPECTED_SCALING_CASES)
    manifest = {
        "schema_version": 5, "status": "complete",
        "out_root": str(canonical), "protocol": _protocol(),
        "selected_case_count": 13, "expected_ilp_solutions": 13,
        "expected_replay_rows": 47, "cases": cases,
        "replay_invocations": [{
            "name": "budget", "status": "complete",
            "cases": [generator.CASE_KEY],
            "results_json": str(replay_path),
            "results_sha256": _sha(replay_path),
            "expected_solutions": [{
                "case_id": generator.CASE_ID, "method": method["manifest"],
                "solution_path": str(paths[method["key"]]),
            } for method in generator.METHODS],
        }],
    }
    manifest_path = canonical / "manifest.json"
    _write_json(manifest_path, manifest)

    metrics_path = root / "unified.csv"
    method_manifest = {row["key"]: row["manifest"]
                       for row in generator.METHODS}
    method_manifest["matched_random"] = "matched-random"
    rows = []
    for suite, case_id, key in sorted(generator.EXPECTED_CSV_KEYS):
        is_target = (suite == "budget" and case_id == generator.CASE_ID and
                     key in paths)
        rows.append({
            "schema_version": 1, "suite": suite,
            "experiment_id": canonical.name, "case_id": case_id,
            "sweep_dimension": "area" if suite == "budget" else "experts",
            "budget_value": 1550 if case_id == generator.CASE_ID else 1,
            "area_budget_mm2": 1550 if case_id == generator.CASE_ID else 1,
            "power_budget_W": 1625 if case_id == generator.CASE_ID else 1,
            "experts": 8, "mesh": "6x6", "method": key,
            "method_raw": method_manifest.get(key, key), "cpu_count": 1,
            "gpu_l_count": 2, "gpu_s_count": 6,
            "shared_bom_peak_power_W": 1205,
            "synchronized_cycles": cycles[key] if is_target else 1,
            "orion_noc_power_W": 0.4, "noc_active_cycles": 8000,
            "nominal_orion_dynamic_noc_energy_uJ": 3.2,
            "legosim_validation": "PASS", "audit_status": "PASS",
            "evidence_tier": ("fresh_constraint_replay_v1"
                              if suite == "budget" else "strict_scaling_v3"),
            "solution_path": str(paths[key]) if is_target else "",
            "solution_sha256": _sha(paths[key]) if is_target else "",
            "evidence_dir": str(root / "evidence" / key),
            "source_results_json": str(replay_path), "errors": "",
        })
    _write_csv(metrics_path, rows)
    return canonical, metrics_path, paths


def _refresh_solution_binding(canonical, metrics_path, method):
    manifest_path = canonical / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    path = generator._canonical_solution_path(canonical, method)
    digest = _sha(path)
    if method == "ilp_perf":
        case["ilp_perf"]["solution_sha256"] = digest
    else:
        case["prior"]["solution_sha256"][method] = digest

    replay_path = canonical / "replays/budget/results.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    for row in replay["layouts"]:
        normalized = row["method"].replace("-", "_")
        if normalized == method:
            row["solution_sha256"] = digest
    _write_json(replay_path, replay)
    manifest["replay_invocations"][0]["results_sha256"] = _sha(replay_path)
    _write_json(manifest_path, manifest)

    rows = _read_csv(metrics_path)
    for row in rows:
        if row["method"] == method:
            row["solution_sha256"] = digest
    _write_csv(metrics_path, rows)


class PriorLayoutsTikzGeneratorTest(unittest.TestCase):
    def test_generates_exact_a1550_comparison_with_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical, metrics, paths = _fixture(root)
            output = root / "paper" / "prior-layouts-tikz.tex"
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                status = generator.main([
                    "--canonical-root", str(canonical),
                    "--metrics-csv", str(metrics),
                    "--output", str(output),
                ])
            self.assertEqual(status, 0)
            report = json.loads(captured.getvalue())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["synchronized_cycles"], {
                "ilp_perf": 104401, "floorplet": 105910, "tap": 108476})
            text = output.read_text(encoding="utf-8")
            self.assertEqual(text.count(r"\begin{minipage}{0.31\textwidth}"), 3)
            self.assertIn("Manifest SHA-256: %s" %
                          _sha(canonical / "manifest.json"), text)
            self.assertIn("Unified metrics CSV SHA-256: %s" % _sha(metrics), text)
            self.assertIn("Budget replay SHA-256: %s" %
                          _sha(canonical / "replays/budget/results.json"), text)
            for method, path in paths.items():
                self.assertIn("%s solution SHA-256: %s" %
                              (method, _sha(path)), text)
            self.assertIn("(a) ILP-Perf: 104{,}401 cycles", text)
            self.assertIn(
                r"(b) Floorplet proxy~\cite{floorplet}: 105{,}910 cycles", text)
            self.assertIn(
                r"(c) TAP-2.5D proxy~\cite{tap25d}: 108{,}476 cycles", text)
            self.assertFalse(any(output.parent.glob(output.name + ".tmp.*")))

    def test_rejects_non_perf_contract_q_and_mesh_capacity(self):
        mutations = (
            ("expert_slots", lambda value:
             value.__setitem__("expert_slots",
                               {"CPU": 1, "GPU-L": 2, "GPU-S": 1}), "q=1"),
            ("directed_capacity", lambda value:
             value["routing"].__setitem__("directed_full_duplex", False),
             "directed_full_duplex"),
            ("post_t", lambda value:
             value.__setitem__("secondary_objective", "min energy"),
             "post-T field"),
        )
        for name, mutation, message in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                canonical, metrics, paths = _fixture(root)
                value = json.loads(paths["ilp_perf"].read_text(encoding="utf-8"))
                mutation(value)
                _write_json(paths["ilp_perf"], value)
                _refresh_solution_binding(canonical, metrics, "ilp_perf")
                with self.assertRaisesRegex(ValueError, message):
                    generator._render(canonical, metrics)

    def test_rejects_c9_and_mismatched_immutable_bundles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical, metrics, _ = _fixture(root)
            manifest_path = canonical / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["protocol"]["local_power_density_constraint"] = True
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError,
                                        "local_power_density_constraint"):
                generator._render(canonical, metrics)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical, metrics, paths = _fixture(root)
            ilp = json.loads(paths["ilp_perf"].read_text(encoding="utf-8"))
            ilp["budgets"]["p_col"] = 500.0
            _write_json(paths["ilp_perf"], ilp)
            _refresh_solution_binding(canonical, metrics, "ilp_perf")
            with self.assertRaisesRegex(ValueError, "removed C9/p_col"):
                generator._render(canonical, metrics)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical, metrics, paths = _fixture(root)
            floorplet = json.loads(
                paths["floorplet"].read_text(encoding="utf-8"))
            e4 = next(site for site in floorplet["sites"]
                      if "e(4)" in site["tasks"])
            e5 = next(site for site in floorplet["sites"]
                      if "e(5)" in site["tasks"])
            e5["tasks"].remove("t_dispatch")
            e4["tasks"].append("t_dispatch")
            floorplet["task_site"]["t_dispatch"] = [e4["l"], e4["i"], e4["j"]]
            _write_json(paths["floorplet"], floorplet)
            _refresh_solution_binding(canonical, metrics, "floorplet")
            with self.assertRaisesRegex(ValueError,
                                        "task bundles differ from ILP-Perf"):
                generator._render(canonical, metrics)

    def test_csv_and_manifest_bindings_fail_without_replacing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical, metrics, _ = _fixture(root)
            rows = _read_csv(metrics)
            next(row for row in rows if row["method"] == "tap" and
                 row["case_id"] == generator.CASE_ID)[
                "synchronized_cycles"] = "108477"
            _write_csv(metrics, rows)
            output = root / "paper" / "prior-layouts-tikz.tex"
            output.parent.mkdir(parents=True)
            output.write_text("sentinel\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees with replay"):
                generator.main([
                    "--canonical-root", str(canonical),
                    "--metrics-csv", str(metrics), "--output", str(output),
                ])
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical, metrics, _ = _fixture(root)
            manifest_path = canonical / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cases"][0]["prior"]["solution_sha256"]["tap"] = "0" * 64
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError,
                                        "tap solution SHA-256 mismatch"):
                generator._render(canonical, metrics)

    def test_superseded_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical, metrics, _ = _fixture(root)
            (canonical / "SUPERSEDED.txt").write_text("invalid\n",
                                                       encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "marked SUPERSEDED"):
                generator._render(canonical, metrics)


if __name__ == "__main__":
    unittest.main()
