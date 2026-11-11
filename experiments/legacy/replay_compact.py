#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay a compact ILP layout in LEGOSim through the SAME pipeline that
produced the frozen baselines (run_budget_mesh_experiment._codegen_layout +
WSL run_ilp_e2e.sh, rounds=2), so the new ILP cycle count is directly
comparable to the existing Floorplet/TAP numbers.  Codegen runs under system
Python 3.12; LEGOSim runs under WSL.  Usage:

    python replay_compact.py e064_m12 064 <solution.json> <out_dir_name>
"""
from __future__ import print_function
import os, sys, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PARAMS = ROOT + "/data/params/experts_%s.json"
SYS_PY = os.environ.get("PYTHON", sys.executable)

cid = sys.argv[1]
suf = sys.argv[2]
sol = os.path.abspath(sys.argv[3])
outname = sys.argv[4] if len(sys.argv) > 4 else ("compact_replay_" + cid)
out_dir = ROOT + "/out/" + outname

cmd = [
    SYS_PY, ROOT + "/src/replay_legosim_solutions.py",
    "--params", PARAMS % suf,
    "--layout", "%s:ilp-compact:%s" % (cid, sol),
    "--out-dir", out_dir,
    "--rounds", "2",
    "--moe-gpgpusim-max-parallel", "8",
]
print("REPLAY CMD:", " ".join(cmd), flush=True)
rc = subprocess.call(cmd, cwd=ROOT)
print("replay rc=", rc, "out_dir=", out_dir, flush=True)
sys.exit(rc)
