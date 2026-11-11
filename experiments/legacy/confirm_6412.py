# -*- coding: utf-8 -*-
"""Confirm the PURE min-T (no byte-hop) warm-from-compact solve at a 45-s budget
returns a layout byte-identical to Floorplet for e064 and e128, locking the
'45-s run' claim and the tie without invoking any tie-break stage."""
from __future__ import print_function
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solve_compact_export as S
B = S.ROOT + "/out/joint_bom_scaling_fresh_20260802"
def norm(sites):
    d={}
    for s in sites: d[(s["i"],s["j"])]=(s["type"],tuple(sorted(s.get("tasks",[]))))
    return d
for cid in ["e064_m12","e128_m16"]:
    outp=os.path.join(S.OUTDIR,"%s_warm45_primary.json"%cid)
    r=S.run(cid, 45.0, 45.0, warm=True, tiebreak=False, out=outp)
    if r is None:
        print("%s: no incumbent at 45s"%cid); continue
    sol=json.load(open(outp))
    floor=json.load(open("%s/baselines/scaling/%s/prior/floorplet.json"%(B,cid)))
    same=(norm(sol["sites"])==norm(floor["sites"]))
    print("=== %s (45s pure min-T warm) proto=%.0f primary_inc=%s bound=%s ==="%(
        cid, sol["protocol_makespan_cycles"], r["primary_incumbent_cycles"], r["bound_cycles"]))
    print("    layout==Floorplet (byte-identical)? %s"%same)
