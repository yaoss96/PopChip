#!/usr/bin/env python3
"""Build and independently audit the canonical 15-input LEGOSim manifest.

This script is intentionally input-only: it never invokes code generation or a
simulator.  It admits only the 30 s + 30 s two-stage ILP solutions and the
proxies regenerated from those solutions.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
ILP_ROOT = ROOT / "out" / "scaling_ilp_secondary_20260801"
PROXY_ROOT = ROOT / "out" / "scaling_secondary_proxy_experiment_20260801"
RESULT_ROOT = ROOT / "out" / "scaling_legosim_final_twostage_20260801"
PAIRS: Tuple[Tuple[int, int], ...] = (
    (8, 6),
    (16, 8),
    (32, 12),
    (64, 12),
    (128, 16),
)
METHODS = ("ilp", "floorplet", "tap")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mapped_bundle_signature(solution: Dict[str, Any]) -> List[List[Any]]:
    site_types: Dict[Tuple[int, int, int], str] = {}
    for site in solution.get("sites", []):
        anchor = (int(site["l"]), int(site["i"]), int(site["j"]))
        if anchor in site_types:
            raise AssertionError("duplicate site anchor: %r" % (anchor,))
        site_types[anchor] = str(site["type"])

    tasks_at: Dict[Tuple[int, int, int], List[str]] = {
        anchor: [] for anchor in site_types
    }
    for task, raw_anchor in solution.get("task_site", {}).items():
        anchor = tuple(int(value) for value in raw_anchor)
        if anchor not in site_types:
            raise AssertionError("task mapped to undeclared site: %s -> %r" % (task, anchor))
        tasks_at[anchor].append(str(task))

    signature = [
        [kind, sorted(tasks_at[anchor])]
        for anchor, kind in site_types.items()
    ]
    return sorted(signature, key=lambda item: (item[0], item[1]))


def declared_bundle_signature(solution: Dict[str, Any]) -> List[List[Any]]:
    signature = [
        [str(site["type"]), sorted(str(task) for task in site.get("tasks", []))]
        for site in solution.get("sites", [])
    ]
    return sorted(signature, key=lambda item: (item[0], item[1]))


def solution_bom(solution: Dict[str, Any]) -> Dict[str, int]:
    counts = Counter(str(site["type"]) for site in solution.get("sites", []))
    return {kind: int(counts.get(kind, 0)) for kind in ("CPU", "GPU-L", "GPU-S")}


def expected_solution_path(experts: int, mesh: int, method: str) -> Path:
    pair = "e%03d_m%02d" % (experts, mesh)
    if method == "ilp":
        return ILP_ROOT / "solutions" / (pair + "_ilp_secondary.json")
    return PROXY_ROOT / "solutions" / (pair + "_" + method + ".json")


def snapshot_result(
    experts: int,
    mesh: int,
    method: str,
    solution_sha: str,
    params_sha: str,
) -> Dict[str, Any]:
    pair = "e%03d_m%02d" % (experts, mesh)
    result_dir = RESULT_ROOT / ("e%03d_%s" % (experts, method))
    result_json = result_dir / "results.json"
    codegen_dir = result_dir / "layouts" / pair / method / "codegen"
    staged_solution = codegen_dir / "solution.input.json"
    staged_params = codegen_dir / "params.input.json"

    staged_solution_sha = sha256(staged_solution) if staged_solution.is_file() else None
    staged_params_sha = sha256(staged_params) if staged_params.is_file() else None
    base: Dict[str, Any] = {
        "result_dir": rel(result_dir),
        "results_json_path": rel(result_json),
        "results_json_present": result_json.is_file(),
        "status": "not_started",
        "finished_utc": None,
        "completed_pass": False,
        "staged_solution_path": rel(staged_solution),
        "staged_solution_present": staged_solution.is_file(),
        "staged_solution_sha256": staged_solution_sha,
        "staged_solution_hash_match": staged_solution_sha == solution_sha,
        "staged_params_path": rel(staged_params),
        "staged_params_present": staged_params.is_file(),
        "staged_params_sha256": staged_params_sha,
        "staged_params_hash_match": staged_params_sha == params_sha,
        "legosim_validation": None,
        "synchronized_makespan_cycles": None,
        "popnet_total_power_W": None,
    }
    if not result_json.is_file():
        return base

    result = load_json(result_json)
    base["status"] = str(result.get("status", "unknown"))
    base["finished_utc"] = result.get("finished_utc")
    if str(result.get("params_sha256", "")).lower() != params_sha:
        base["result_params_hash_match"] = False
    else:
        base["result_params_hash_match"] = True

    layouts = [
        row
        for row in result.get("layouts", [])
        if row.get("method") == method and row.get("case_id") == pair
    ]
    base["matching_layout_rows"] = len(layouts)
    if len(layouts) != 1:
        return base

    row = layouts[0]
    row_solution_match = str(row.get("solution_sha256", "")).lower() == solution_sha
    legosim = row.get("legosim", {})
    validation = legosim.get("validation")
    base["result_solution_hash_match"] = row_solution_match
    base["legosim_validation"] = validation
    base["synchronized_makespan_cycles"] = legosim.get(
        "synchronized_makespan_cycles"
    )
    base["popnet_total_power_W"] = legosim.get("summary", {}).get(
        "popnet_total_power"
    )
    base["completed_pass"] = bool(
        result.get("status") == "complete"
        and row.get("status") == "ok"
        and validation == "PASS"
        and row_solution_match
        and base["result_params_hash_match"]
    )
    return base


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    secondary_summary_path = ILP_ROOT / "secondary_audit.json"
    proxy_summary_path = PROXY_ROOT / "metadata.json"
    secondary_summary = load_json(secondary_summary_path)
    proxy_summary = load_json(proxy_summary_path)

    assert secondary_summary.get("pass") is True
    assert int(secondary_summary.get("completed_pairs", -1)) == len(PAIRS)
    assert proxy_summary.get("pass") is True
    assert proxy_summary.get("status") == "complete"
    assert int(proxy_summary.get("completed_count", -1)) == len(PAIRS)

    records: List[Dict[str, Any]] = []
    scale_audits: List[Dict[str, Any]] = []

    for experts, mesh in PAIRS:
        pair = "e%03d_m%02d" % (experts, mesh)
        ilp_audit_path = ILP_ROOT / "audits" / (pair + "_audit.json")
        ilp_command_path = ILP_ROOT / "commands" / (pair + "_command.json")
        proxy_audit_path = PROXY_ROOT / "audits" / (pair + "_proxy_audit.json")
        ilp_audit = load_json(ilp_audit_path)
        ilp_command = load_json(ilp_command_path)
        proxy_audit = load_json(proxy_audit_path)

        assert ilp_audit.get("pass") is True
        assert ilp_audit.get("validation", {}).get("pass") is True
        assert proxy_audit.get("pass") is True
        solver_options = ilp_command.get("solver_options", {})
        assert float(solver_options.get("time_limit")) == 30.0
        assert float(solver_options.get("secondary_time")) == 30.0
        assert solver_options.get("mode") == "certified-latency"

        params_path = ROOT / ilp_audit["artifacts"]["params"]
        params_sha = sha256(params_path)
        assert params_sha == str(ilp_audit["artifacts"]["params_sha256"]).lower()
        assert params_sha == str(proxy_audit["provenance"]["params_sha256"]).lower()
        assert rel(params_path) == proxy_audit["provenance"]["params_path"]

        audit_proxy_by_method = {
            entry["method"]: entry for entry in proxy_audit.get("proxies", [])
        }
        assert set(audit_proxy_by_method) == {"floorplet", "tap"}

        method_solutions: Dict[str, Dict[str, Any]] = {}
        method_paths: Dict[str, Path] = {}
        method_hashes: Dict[str, str] = {}
        method_boms: Dict[str, Dict[str, int]] = {}
        method_bundle_hashes: Dict[str, str] = {}

        for method in METHODS:
            solution_path = expected_solution_path(experts, mesh, method)
            solution = load_json(solution_path)
            solution_sha = sha256(solution_path)

            if method == "ilp":
                assert rel(solution_path) == ilp_audit["artifacts"]["solution"]
                assert solution_sha == str(
                    ilp_audit["artifacts"]["solution_sha256"]
                ).lower()
                assert rel(solution_path) == proxy_audit["source_ilp"]["path"]
                assert solution_sha == str(proxy_audit["source_ilp"]["sha256"]).lower()
                audit_validation = proxy_audit["source_ilp"]["validation"]
                source_kind = "two_stage_ilp_30s_primary_30s_secondary"
            else:
                proxy_entry = audit_proxy_by_method[method]
                assert rel(solution_path) == proxy_entry["solution_path"]
                assert solution_sha == str(proxy_entry["solution_sha256"]).lower()
                audit_validation = proxy_entry["validation"]
                source_kind = "regenerated_proxy_from_two_stage_ilp"

            assert audit_validation.get("pass") is True
            assert audit_validation.get("exact_BOM_equal") is True
            assert audit_validation.get("immutable_bundle_multiset_equal") is True
            assert audit_validation.get("candidate_serialization_consistent") is True

            grid = solution.get("grid", {})
            assert int(grid.get("L")) == 1
            assert int(grid.get("X")) == mesh
            assert int(grid.get("Y")) == mesh
            declared = declared_bundle_signature(solution)
            mapped = mapped_bundle_signature(solution)
            assert declared == mapped
            bundle_hash = canonical_hash(mapped)
            assert bundle_hash == str(
                audit_validation["bundle_multiset_sha256"]
            ).lower()

            method_solutions[method] = solution
            method_paths[method] = solution_path
            method_hashes[method] = solution_sha
            method_boms[method] = solution_bom(solution)
            method_bundle_hashes[method] = bundle_hash

        bom_match = len({json.dumps(v, sort_keys=True) for v in method_boms.values()}) == 1
        bundle_match = len(set(method_bundle_hashes.values())) == 1
        assert bom_match
        assert bundle_match

        expected_bundle_hash = method_bundle_hashes["ilp"]
        expected_bom = method_boms["ilp"]
        assert expected_bundle_hash == str(
            proxy_audit["source_ilp"]["bundle_multiset_sha256"]
        ).lower()

        scale_audits.append(
            {
                "pair": pair,
                "experts": experts,
                "mesh": "%dx%d" % (mesh, mesh),
                "ilp_audit_path": rel(ilp_audit_path),
                "ilp_audit_sha256": sha256(ilp_audit_path),
                "ilp_audit_pass": True,
                "proxy_audit_path": rel(proxy_audit_path),
                "proxy_audit_sha256": sha256(proxy_audit_path),
                "proxy_audit_pass": True,
                "primary_time_limit_s": 30.0,
                "secondary_time_limit_s": 30.0,
                "exact_bom_match_across_methods": bom_match,
                "exact_bundle_match_across_methods": bundle_match,
                "bom": expected_bom,
                "bundle_signature_sha256": expected_bundle_hash,
            }
        )

        for method in METHODS:
            solution_sha = method_hashes[method]
            result = snapshot_result(
                experts, mesh, method, solution_sha, params_sha
            )
            records.append(
                {
                    "pair": pair,
                    "experts": experts,
                    "mesh": "%dx%d" % (mesh, mesh),
                    "mesh_rows": mesh,
                    "mesh_columns": mesh,
                    "method": method,
                    "source_kind": (
                        "two_stage_ilp_30s_primary_30s_secondary"
                        if method == "ilp"
                        else "regenerated_proxy_from_two_stage_ilp"
                    ),
                    "solution_path": rel(method_paths[method]),
                    "solution_sha256": solution_sha,
                    "params_path": rel(params_path),
                    "params_sha256": params_sha,
                    "gpu_l": expected_bom["GPU-L"],
                    "gpu_s": expected_bom["GPU-S"],
                    "gpu_l_s": "%d/%d" % (
                        expected_bom["GPU-L"], expected_bom["GPU-S"]
                    ),
                    "bom": method_boms[method],
                    "bundle_signature_sha256": method_bundle_hashes[method],
                    "exact_bom_match_across_methods": bom_match,
                    "exact_bundle_match_across_methods": bundle_match,
                    "ilp_audit_pass": True,
                    "proxy_audit_pass": True,
                    "result": result,
                }
            )

    assert len(records) == 15
    assert all(row["ilp_audit_pass"] for row in records)
    assert all(row["proxy_audit_pass"] for row in records)
    assert all(row["exact_bom_match_across_methods"] for row in records)
    assert all(row["exact_bundle_match_across_methods"] for row in records)

    generated_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    status_counts = Counter(row["result"]["status"] for row in records)
    complete_pass_count = sum(bool(row["result"]["completed_pass"]) for row in records)
    manifest = {
        "schema_version": 1,
        "generated_utc": generated_utc,
        "claim_boundary": (
            "Canonical inputs for two-round large-scale LEGOSim replay; this "
            "manifest builder performs no simulation and reports result state only "
            "as observed at generated_utc."
        ),
        "input_policy": {
            "ilp_source": "30 s primary plus 30 s communication-energy secondary stage",
            "proxy_source": "proxies regenerated from the canonical two-stage ILP bundles",
            "canonical_ilp_root": rel(ILP_ROOT),
            "canonical_proxy_root": rel(PROXY_ROOT),
            "result_root": rel(RESULT_ROOT),
        },
        "global_audits": {
            "two_stage_ilp_audit_path": rel(secondary_summary_path),
            "two_stage_ilp_audit_sha256": sha256(secondary_summary_path),
            "two_stage_ilp_audit_pass": True,
            "proxy_metadata_path": rel(proxy_summary_path),
            "proxy_metadata_sha256": sha256(proxy_summary_path),
            "all_proxy_audits_pass": True,
        },
        "expected_input_count": 15,
        "actual_input_count": len(records),
        "all_input_audits_pass": True,
        "all_scales_exact_bom_match": True,
        "all_scales_exact_bundle_match": True,
        "result_snapshot": {
            "status_counts": dict(sorted(status_counts.items())),
            "completed_pass_count": complete_pass_count,
            "remaining_count": len(records) - complete_pass_count,
        },
        "scale_audits": scale_audits,
        "inputs": records,
    }

    json_path = RESULT_ROOT / "final_input_manifest.json"
    csv_path = RESULT_ROOT / "final_input_manifest.csv"
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    fieldnames = [
        "generated_utc",
        "pair",
        "experts",
        "mesh",
        "method",
        "source_kind",
        "solution_path",
        "solution_sha256",
        "params_path",
        "params_sha256",
        "gpu_l",
        "gpu_s",
        "gpu_l_s",
        "cpu",
        "bundle_signature_sha256",
        "exact_bom_match_across_methods",
        "exact_bundle_match_across_methods",
        "ilp_audit_pass",
        "proxy_audit_pass",
        "result_dir",
        "result_status",
        "result_completed_pass",
        "staged_solution_hash_match",
        "staged_params_hash_match",
        "legosim_validation",
        "synchronized_makespan_cycles",
        "popnet_total_power_W",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            result = row["result"]
            writer.writerow(
                {
                    "generated_utc": generated_utc,
                    "pair": row["pair"],
                    "experts": row["experts"],
                    "mesh": row["mesh"],
                    "method": row["method"],
                    "source_kind": row["source_kind"],
                    "solution_path": row["solution_path"],
                    "solution_sha256": row["solution_sha256"],
                    "params_path": row["params_path"],
                    "params_sha256": row["params_sha256"],
                    "gpu_l": row["gpu_l"],
                    "gpu_s": row["gpu_s"],
                    "gpu_l_s": row["gpu_l_s"],
                    "cpu": row["bom"]["CPU"],
                    "bundle_signature_sha256": row["bundle_signature_sha256"],
                    "exact_bom_match_across_methods": row[
                        "exact_bom_match_across_methods"
                    ],
                    "exact_bundle_match_across_methods": row[
                        "exact_bundle_match_across_methods"
                    ],
                    "ilp_audit_pass": row["ilp_audit_pass"],
                    "proxy_audit_pass": row["proxy_audit_pass"],
                    "result_dir": result["result_dir"],
                    "result_status": result["status"],
                    "result_completed_pass": result["completed_pass"],
                    "staged_solution_hash_match": result[
                        "staged_solution_hash_match"
                    ],
                    "staged_params_hash_match": result["staged_params_hash_match"],
                    "legosim_validation": result["legosim_validation"],
                    "synchronized_makespan_cycles": result[
                        "synchronized_makespan_cycles"
                    ],
                    "popnet_total_power_W": result["popnet_total_power_W"],
                }
            )

    print("PASS: audited %d canonical inputs" % len(records))
    print("JSON: %s" % rel(json_path))
    print("CSV: %s" % rel(csv_path))
    print("result status counts: %s" % dict(sorted(status_counts.items())))
    print("completed PASS: %d/%d" % (complete_pass_count, len(records)))


if __name__ == "__main__":
    main()
