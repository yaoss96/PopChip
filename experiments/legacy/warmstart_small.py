# -*- coding: utf-8 -*-
"""Warm-start (from Floorplet) primary min-T solves for 8/16/32 experts,
then diff the resulting layout against the frozen ILP layout and Floorplet."""
from __future__ import print_function
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solve_compact_export as S

B = S.ROOT + "/out/joint_bom_scaling_fresh_20260802"
def norm(sites):
    d={}
    for s in sites: d[(s["i"],s["j"])]=(s["type"],tuple(sorted(s.get("tasks",[]))))
    return d
def ident(a,b):
    na,nb=norm(a),norm(b)
    return na==nb

TL = float(sys.argv[1]) if len(sys.argv)>1 else 90.0
for cid in ["e008_m06","e016_m08","e032_m12"]:
    outp=os.path.join(S.OUTDIR,"%s_warm_primary.json"%cid)
    r=S.run(cid,TL,TL,warm=True,tiebreak=False,out=outp)
    sol=json.load(open(outp))
    frozen=json.load(open("%s/solutions/scaling/%s/ilp_perf.json"%(B,cid)))
    floor=json.load(open("%s/baselines/scaling/%s/prior/floorplet.json"%(B,cid)))
    same_frozen=ident(sol["sites"],frozen["sites"])
    same_floor=ident(sol["sites"],floor["sites"])
    print("=== %s ==="%cid)
    print("  warm proto_makespan=%.0f primary_inc=%s bound=%s"%(
        sol["protocol_makespan_cycles"], r["primary_incumbent_cycles"], r["bound_cycles"]))
    print("  layout==frozen_ILP? %s   layout==Floorplet? %s"%(same_frozen,same_floor))
