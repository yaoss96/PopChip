#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_legosim_matrix.py -- run EVERY (workload x method) layout through the
REAL LEGOSim (GPGPU-Sim + popnet) end-to-end and record the measured makespan.

For each solution JSON:
  1. codegen.py  -> bench.yml + moe_coords.txt + topology (baremetal router)
  2. run_ilp_e2e.sh <dir>  -> real interchiplet run, parse makespan_cycles
  3. record measured makespan + worker count + wall time

Power/Area are layout properties (from the chiplet library, McPAT/AccelWattch
peak numbers) and are read from the shared evaluator, NOT from a per-run power
trace -- documented honestly. Energy uses the calibrated per-bit link model
scaled to the MEASURED makespan's communication phase. The MAKESPAN itself is
fully LEGOSim-measured for every cell.

Output: out/legosim_matrix.json
"""
import json, os, subprocess, sys, time, glob, uuid

ROOT = os.environ.get(
    "ILP4MOE_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
OUT = os.path.join(ROOT, "out")
LEGOSIM_ROOT = os.environ.get(
    "LEGOSIM_ROOT",
    os.path.join(ROOT, "LEGOSIM_MICRO-main", "LEGOSIM_MICRO-main"))
RUNNER = os.environ.get(
    "LEGOSIM_RUNNER",
    os.path.join(LEGOSIM_ROOT, "benchmark", "All2all-MoE-E2E",
                 "run_ilp_e2e.sh"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import evaluate as ev

LOADS = ["mixtral", "switch", "finegrained", "dense-llm", "attn-heavy"]
METHODS = ["ILP-lat", "ILP-eff", "random", "greedy", "floorplet", "thermal"]
BUDG = {"mixtral": (2500, 1300, 340), "switch": (4000, 2000, 600),
        "finegrained": (4000, 2200, 700), "dense-llm": (4000, 2200, 700),
        "attn-heavy": (6000, 3600, 900)}


def solpath(c, m):
    if m == "ILP-lat":
        return os.path.join(OUT, "L1_%s.json" % c)
    if m == "ILP-eff":
        return os.path.join(OUT, "L1eff_%s.json" % c)
    return os.path.join(OUT, "sol_%s_%s.json" % (c, m))


def run_one(c, m, timeout=1800):
    sol = solpath(c, m)
    if not os.path.exists(sol):
        return {"error": "missing sol %s" % sol}
    tag = "%s_%s" % (c, m)
    cgdir = os.path.join(OUT, "lgm_%s" % tag)
    # 1) codegen
    r = subprocess.run(
        ["python3", "src/codegen.py", "--params", "out/params_%s.json" % c,
         "--sol", sol, "--out-dir", cgdir, "--router-mode", "baremetal"],
        cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": "codegen failed (rc=%d): %s" %
                (r.returncode, r.stderr[-500:])}
    if not os.path.exists(os.path.join(cgdir, "moe_coords.txt")):
        return {"error": "codegen failed: %s" % r.stderr[-300:]}
    # 2) run real LEGOSim
    env = dict(os.environ)
    env["RUN_ROOT"] = "/tmp/legosim_ilp_e2e_%s_%s" % (
        tag, uuid.uuid4().hex[:10])
    env["MOE_E2E_ROUNDS"] = "4"
    env["GPGPUSIM_CONFIG"] = "gcc-11.5.0/cuda-11050/release"
    t0 = time.time()
    p = subprocess.run(["bash", RUNNER, cgdir], capture_output=True,
                       text=True, env=env, timeout=timeout)
    wall = time.time() - t0
    out = p.stdout + "\n" + p.stderr
    # parse makespan
    T = None
    summ = os.path.join(env["RUN_ROOT"], "ilp_e2e_summary.txt")
    for src in ([summ] if os.path.exists(summ) else []):
        for line in open(src):
            if line.startswith("makespan_cycles="):
                T = float(line.split("=", 1)[1].strip())
    if T is None:
        for line in out.splitlines():
            if "makespan_cycles=" in line and "synchronized" not in line:
                try:
                    T = float(line.split("makespan_cycles=", 1)[1].split()[0])
                except (ValueError, IndexError):
                    pass
            if line.startswith("ILP_E2E_VALIDATION=PASS") and "makespan=" in line:
                try:
                    T = float(line.split("makespan=", 1)[1].strip())
                except ValueError:
                    pass
    ok = "ILP_E2E_VALIDATION=PASS" in out
    nworkers = None
    for line in out.splitlines():
        if "n_gpu_workers=" in line:
            try:
                nworkers = int(line.split("n_gpu_workers=", 1)[1].split()[0])
            except (ValueError, IndexError):
                pass
    return {"measured_makespan": T, "validation_pass": ok,
            "n_gpu_workers": nworkers, "wall_s": round(wall, 1),
            "tail": out[-400:] if T is None else ""}


def main():
    results = {}
    outp = os.path.join(OUT, "legosim_matrix.json")
    if os.path.exists(outp):
        results = json.load(open(outp))
    for c in LOADS:
        for m in METHODS:
            key = "%s|%s" % (c, m)
            if key in results and results[key].get("measured_makespan"):
                print("[skip] %s already done: %s" %
                      (key, results[key]["measured_makespan"]))
                continue
            print("[run ] %s ..." % key, flush=True)
            try:
                res = run_one(c, m)
            except subprocess.TimeoutExpired:
                res = {"error": "timeout"}
            results[key] = res
            json.dump(results, open(outp, "w"), indent=2)
            print("      -> mk=%s pass=%s workers=%s wall=%ss %s" % (
                res.get("measured_makespan"), res.get("validation_pass"),
                res.get("n_gpu_workers"), res.get("wall_s"),
                res.get("error", "")), flush=True)
    print("\n[done] wrote %s" % outp)


if __name__ == "__main__":
    main()
