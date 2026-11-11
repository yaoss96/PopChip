#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap one canonical 13-case run from verified split Perf solves.

The split roots are useful for running the area, power, and scaling suites in
parallel.  Paper-data collection, however, intentionally accepts only one
canonical manifest and one fresh 60-layout replay root.  This utility imports
only the verified ILP-Perf solutions.  The canonical driver then recomputes
every ILP-Power solution against the canonical Perf path, regenerates all
baselines, and performs all replays.

No split manifest is copied or edited.  Every imported solution is checked
against its source manifest hash and the current solution contract before a
new canonical manifest is created.
"""

from __future__ import print_function

import argparse
import json
import shutil
import sys
from pathlib import Path

import run_full_constraint_revalidation as protocol


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Import verified split ILP-Perf solutions into one "
                    "canonical 13-case protocol root")
    parser.add_argument("--area-root", required=True)
    parser.add_argument("--power-root", required=True)
    parser.add_argument("--scaling-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--budget-source",
                        default=protocol.DEFAULT_BUDGET_SOURCE)
    parser.add_argument("--scale-source", default=protocol.DEFAULT_SCALE_SOURCE)
    parser.add_argument("--scale-params", default=protocol.DEFAULT_SCALE_PARAMS)
    parser.add_argument("--gurobi-python",
                        default=protocol.DEFAULT_GUROBI_PYTHON)
    parser.add_argument("--system-python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _case_partitions(cases):
    area = [case for case in cases
            if case["suite"] == "budget" and
            case["case_id"].startswith("area_")]
    power = [case for case in cases
             if case["suite"] == "budget" and
             case["case_id"].startswith("power_")]
    scaling = [case for case in cases if case["suite"] == "scaling"]
    if (len(area), len(power), len(scaling)) != (4, 4, 5):
        raise RuntimeError(
            "canonical case partition must be 4 area + 4 power + 5 scaling")
    return area, power, scaling


def _verified_imports(source_root, source_cases):
    source_root = Path(source_root).resolve()
    manifest = protocol._load_manifest(source_root, source_cases)
    if manifest.get("status") not in (
            "solve_complete", "baselines_complete", "replay_complete"):
        raise RuntimeError(
            "split manifest is not solve-complete: %s (%s)" %
            (source_root, manifest.get("status")))

    imports = []
    for case in source_cases:
        row = manifest["cases_by_key"][case["key"]]["ilp_perf"]
        if row.get("status") != "complete":
            raise RuntimeError(
                "split ILP-Perf is not complete for %s" % case["key"])
        source = protocol._solution_path(source_root, case, "ilp-perf")
        protocol._require_file(source, "split ILP-Perf solution")
        digest = protocol._sha256(source)
        if row.get("solution_sha256") != digest:
            raise RuntimeError(
                "split manifest hash mismatch for %s" % case["key"])
        protocol._validate_solution(source, "ilp-perf", case)
        imports.append({
            "case": case,
            "source_root": source_root,
            "source_manifest": source_root / "manifest.json",
            "source_manifest_sha256": protocol._sha256(
                source_root / "manifest.json"),
            "source": source,
            "sha256": digest,
        })
    return imports


def _summary(rows, out_root, dry_run):
    return {
        "schema_version": 1,
        "status": "dry-run" if dry_run else "bootstrapped",
        "policy": (
            "import only hash-verified ILP-Perf; recompute ILP-Power, "
            "baselines, and all LEGOSim replays in the canonical root"),
        "canonical_root": str(out_root),
        "imported_perf_count": len(rows),
        "imports": [{
            "case_key": row["case"]["key"],
            "source_root": str(row["source_root"]),
            "source_manifest": str(row["source_manifest"]),
            "source_manifest_sha256": row["source_manifest_sha256"],
            "source_solution": str(row["source"]),
            "solution_sha256": row["sha256"],
        } for row in rows],
        "next_commands": [
            "python src/run_full_constraint_revalidation.py --phase "
            "solve-only --out-root %s --resume" % out_root,
            "python src/run_full_constraint_revalidation.py --phase "
            "baselines-only --out-root %s --resume" % out_root,
            "python src/run_full_constraint_revalidation.py --phase "
            "replay-only --out-root %s --resume --replay-jobs 2" % out_root,
        ],
    }


def main(argv=None):
    args = _parse_args(argv)
    area_root = protocol._resolve(args.area_root)
    power_root = protocol._resolve(args.power_root)
    scaling_root = protocol._resolve(args.scaling_root)
    out_root = protocol._resolve(args.out_root)
    gurobi_python = protocol._resolve(args.gurobi_python)
    system_python = protocol._resolve(args.system_python)

    for path, label in ((gurobi_python, "Gurobi Python"),
                        (system_python, "system Python")):
        protocol._require_file(path, label)
    if out_root.exists():
        raise RuntimeError("refusing to overwrite canonical root: %s" %
                           out_root)

    cases = (protocol._budget_cases(protocol._resolve(args.budget_source)) +
             protocol._scale_cases(protocol._resolve(args.scale_source),
                                   protocol._resolve(args.scale_params)))
    if len(cases) != 13:
        raise RuntimeError("canonical protocol must contain 13 cases")
    area_cases, power_cases, scaling_cases = _case_partitions(cases)

    rows = []
    rows.extend(_verified_imports(area_root, area_cases))
    rows.extend(_verified_imports(power_root, power_cases))
    rows.extend(_verified_imports(scaling_root, scaling_cases))
    if len(rows) != 13:
        raise RuntimeError("expected 13 verified ILP-Perf imports")

    summary = _summary(rows, out_root, args.dry_run)
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    out_root.mkdir(parents=True)
    manifest = protocol._new_manifest(
        cases, out_root, gurobi_python, system_python)
    protocol._save_manifest(out_root, manifest)
    for row in rows:
        target = protocol._solution_path(
            out_root, row["case"], "ilp-perf")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(row["source"]), str(target))
        if protocol._sha256(target) != row["sha256"]:
            raise RuntimeError(
                "copied solution hash mismatch for %s" %
                row["case"]["key"])
        protocol._validate_solution(target, "ilp-perf", row["case"])

    protocol._atomic_json(out_root / "perf_import_provenance.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        sys.exit(2)
