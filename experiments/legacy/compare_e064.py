#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e064: warm-from-Floorplet, primary-only vs primary+byte-hop-compaction.
Compare the coordinate-recomputed protocol makespan (tracks measured ~0.1%)
against Floorplet's measured 2082205 / TAP 2102613."""
from __future__ import print_function
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solve_compact_export as S

TLP = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
TLT = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
CID = sys.argv[1] if len(sys.argv) > 1 else "e064_m12"
OUT = os.path.join(S.WD, "compare_%s_results.json" % CID)

results = []
for tag, tb in [("primary_only", False), ("primary+compact", True)]:
    outp = os.path.join(S.OUTDIR, "%s_%s.json" % (CID, tag))
    r = S.run(CID, TLP, TLT, warm=True, tiebreak=tb, out=outp)
    r["tag"] = tag
    results.append(r)
    json.dump(results, open(OUT, "w"), indent=2)
print("DONE ->", OUT)
