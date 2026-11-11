#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
closed_loop.py -- the Section 6 closed loop:  PROFILE -> SOLVE -> CODEGEN ->
SIMULATE -> RECALIBRATE, with damped tau_h feedback and C14 successive
linearization on T_hat.

Two SIMULATE backends:
  --sim model   (default) an analytic CONTENTION model that adds queueing delay
                the contention-free ILP ignores. This lets the whole loop run
                end-to-end without a built LEGOSim, and it is deterministic so
                convergence is reproducible. The model inflates per-hop latency
                on links carrying concentrated all-to-all traffic -- exactly the
                effect Section 4.5 says the ILP is blind to.
  --sim legosim invoke the real  `interchiplet bench.yml -t 5 -e 0.005`  in a
                given benchmark dir and parse the measured cycle count +
                delayInfo.txt. Used when LEGOSim is built (see --legosim-dir).

Recalibration (Section 6 step 5):
    gamma = T_measured / T_predicted            (global congestion multiplier)
    if |1 - 1/gamma| > tol:  tau_h <- rho*gamma*tau_h + (1-rho)*tau_h   (damped)
    re-solve with updated tau_h AND C14 using T_hat = T_measured.
Cap at --max-iter outer iterations (doc recommends <=4).

Everything is logged to <out>/loop_log.json for the paper's convergence table.
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev


# ----------------------------------------------------------------------------
# Contention model (stand-in for LEGOSim measurement)
# ----------------------------------------------------------------------------
def simulate_model(params, sol, contention_k=0.6):
    """Return (T_measured, per_node_load, gamma_hint).

    Contention-free critical path comes from evaluate(). We then ADD queueing
    that the affine ILP delay model omits (Section 4.5). Two effects:

      (1) HUB SERIALIZATION. A node that is the src (or dst) of F distinct
          all-to-all flows (dispatch fan-out / combine fan-in) cannot inject
          them in parallel -- its single D2D port serializes them, adding
          (F-1) * t0 fixed packetization latency plus queueing of the bytes.
          The ILP charges each flow independently, so it never sees this.

      (2) LINK CONTENTION. When several concentrated flows share the hottest
          port, the excess over the mean port load serializes at beta.

    This is deterministic and monotone in placement compactness, so the outer
    loop's damped tau_h feedback is reproducible."""
    st = ev.site_types_from_solution(sol)
    ts = sol["task_site"]
    grid = sol["grid"]
    base = ev.evaluate(params, ts, st, grid, strict=False)
    if not base.get("feasible") and "makespan_cycles" not in base:
        return None, {}, 1.0
    T0 = base["makespan_cycles"]

    ic = params["interconnect"]
    beta = ic["beta"]
    t0 = ic["t0"]

    node_of = {}
    for s in sol["sites"]:
        for t in s["tasks"]:
            node_of[t] = (s["l"], s["i"], s["j"])

    load = {}      # bytes through each node's port (in+out, inter-chiplet only)
    fanout = {}    # number of distinct inter-chiplet flows per node
    for e in params["edges"]:
        su, du = node_of.get(e["src"]), node_of.get(e["dst"])
        if su is None or du is None or su == du:
            continue
        load[su] = load.get(su, 0.0) + e["vol"]
        load[du] = load.get(du, 0.0) + e["vol"]
        fanout[su] = fanout.get(su, 0) + 1
        fanout[du] = fanout.get(du, 0) + 1
    if not load:
        return T0, {}, 1.0

    # (1) hub serialization: worst node's (F-1) fixed packetization latencies
    max_fan = max(fanout.values())
    hub_delay = contention_k * max(0, max_fan - 1) * t0
    # (2) link contention: excess of hottest port over mean, serialized
    mean = sum(load.values()) / len(load)
    peak = max(load.values())
    q_delay = contention_k * (peak - mean) / beta

    T_meas = T0 + hub_delay + q_delay
    gamma = T_meas / T0 if T0 > 0 else 1.0
    return T_meas, {str(k): v for k, v in load.items()}, gamma


def simulate_legosim(bench_dir, legosim_dir, timeout=7200, coord_file=None,
                     run_root=None):
    """Run the REAL interchiplet flow via run_ilp_e2e.sh (which sets up the
    CUDA-11.5 + GPGPU-Sim + popnet environment, stages the ILP coord file, and
    validates the run) and parse the measured makespan. Returns
    (T_measured, raw_stdout) or (None, err).

    bench_dir must be a codegen output dir containing bench.yml + moe_coords.txt
    (i.e. what codegen.py --router-mode baremetal produced). This is the Task-4
    entry point wired into the Section-6 closed loop."""
    runner = os.path.join(legosim_dir, "benchmark", "All2all-MoE-E2E",
                          "run_ilp_e2e.sh")
    if not os.path.exists(runner):
        return None, "run_ilp_e2e.sh not found at %s" % runner
    if not os.path.exists(os.path.join(bench_dir, "moe_coords.txt")):
        return None, "no moe_coords.txt in %s (run codegen first)" % bench_dir
    rr = run_root or ("/tmp/legosim_ilp_e2e_" +
                      os.path.basename(os.path.abspath(bench_dir)))
    env = dict(os.environ)
    env["RUN_ROOT"] = rr
    env.setdefault("MOE_E2E_ROUNDS", "4")
    cmd = ["bash", runner, os.path.abspath(bench_dir)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
    except Exception as ex:
        return None, "run_ilp_e2e.sh failed: %s" % ex
    out = p.stdout + "\n" + p.stderr
    # Prefer the persisted summary (authoritative); fall back to stdout scrape.
    T = None
    summ = os.path.join(rr, "ilp_e2e_summary.txt")
    if os.path.exists(summ):
        for line in open(summ):
            if line.startswith("makespan_cycles="):
                try:
                    T = float(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
    if T is None:
        for line in out.splitlines():
            if line.startswith("makespan_cycles="):
                try:
                    T = float(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
    if T is None or T <= 0:
        return None, "no makespan parsed. tail:\n" + out[-600:]
    return T, out


# ----------------------------------------------------------------------------
# ILP solve wrapper (uses the Gurobi interpreter as a subprocess so this driver
# runs under system python3 too; if gurobipy is importable we call in-process).
# ----------------------------------------------------------------------------
def _winpath(p):
    """Convert a WSL /mnt/c/... path to a Windows C:\\... path so the Windows
    Gurobi python.exe can open it. Leaves already-Windows paths untouched."""
    ap = os.path.abspath(p)
    if ap.startswith("/mnt/") and len(ap) > 6 and ap[6] == "/":
        drive = ap[5].upper()
        return drive + ":\\" + ap[7:].replace("/", "\\")
    return p


def solve_ilp(params_path, grid, budgets, tau_h, t_hat, out_sol,
              gurobi_python, time_limit, mipgap, seed):
    is_win_py = gurobi_python.lower().endswith(".exe")
    conv = _winpath if is_win_py else (lambda x: x)
    ilp_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "ilp_layout.py")
    args = [gurobi_python, conv(ilp_script),
            "--params", conv(params_path),
            "--L", str(grid["L"]), "--X", str(grid["X"]), "--Y", str(grid["Y"]),
            "--a-max", str(budgets["a_max"]), "--p-max", str(budgets["p_max"]),
            "--p-col", str(budgets["p_col"]),
            "--time-limit", str(time_limit), "--mipgap", str(mipgap),
            "--seed", str(seed), "--quiet", "-o", conv(out_sol)]
    if budgets.get("e_bar") is not None:
        args += ["--e-bar", str(budgets["e_bar"])]
    if t_hat is not None:
        args += ["--t-hat", str(t_hat)]
    p = subprocess.run(args, capture_output=True, text=True)
    if not os.path.exists(out_sol):
        raise RuntimeError("ILP solve produced no solution:\n%s\n%s"
                           % (p.stdout, p.stderr))
    return json.load(open(out_sol))


def compute_only_makespan(params, sol):
    """The compute-only critical path C: makespan with the interconnect zeroed.

    C is the sum of measured GPGPU-Sim compute durations along the DAG's
    critical chain; it is ALREADY correct (the cycles are real, Section 5) and
    does NOT move under recalibration. Only the communication residual
    (T - C) is what the affine delay model under-predicts, so the closed loop
    must fit gamma to that residual -- not to the global makespan, which is
    dominated by C and therefore barely responds to tau_h/t0."""
    st = ev.site_types_from_solution(sol)
    ts = sol["task_site"]
    p0 = json.loads(json.dumps(params))
    p0["interconnect"]["t0"] = 0.0
    p0["interconnect"]["tau_h"] = 0.0
    p0["interconnect"]["tau_v"] = 0.0
    p0["interconnect"]["beta"] = 1e18
    r = ev.evaluate(p0, ts, st, sol["grid"], strict=False)
    return r.get("makespan_cycles", 0.0)


def patch_interconnect(params_path, tau_h, t0, work_path):
    """Write a copy of params with interconnect.tau_h AND t0 overridden.

    Section 6 recalibrates the LATENCY model, which has two calibrated pieces
    (Section 5 table): tau_h -- the per-hop horizontal latency (the congestion-
    sensitive term) -- and t0 -- the FIXED D2D serialization + packetization
    latency charged once per inter-chiplet edge. On a compact ILP layout the
    hop counts are tiny, so measured/predicted discrepancy is dominated by the
    fixed per-message overhead (router sync stamps + popnet packetization), NOT
    by per-hop congestion. We therefore feed the damped gamma into BOTH terms so
    the predicted critical-path communication latency can actually track the
    measured makespan; scaling tau_h alone has almost no leverage here."""
    params = json.load(open(params_path))
    params["interconnect"]["tau_h"] = tau_h
    params["interconnect"]["t0"] = t0
    with open(work_path, "w") as fh:
        json.dump(params, fh, indent=2)
    return work_path


def main():
    ap = argparse.ArgumentParser(description="Section-6 closed loop")
    ap.add_argument("--params", required=True)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--X", type=int, default=4)
    ap.add_argument("--Y", type=int, default=4)
    ap.add_argument("--a-max", type=float, default=2500.0)
    ap.add_argument("--p-max", type=float, default=1300.0)
    ap.add_argument("--p-col", type=float, default=340.0)
    ap.add_argument("--e-bar", type=float, default=None)
    ap.add_argument("--rho", type=float, default=0.7, help="damping factor "
                    "(0.7 converges the mixtral smoke loop in <=4 iters)")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="stop when |T_meas-T_pred|/T_meas < tol")
    ap.add_argument("--max-iter", type=int, default=4)
    ap.add_argument("--sim", choices=["model", "legosim"], default="model")
    ap.add_argument("--router-mode", choices=["baremetal", "sniper"],
                    default="baremetal",
                    help="codegen router mode for legosim sim (baremetal: "
                         "router off makespan critical path, no Sniper)")
    ap.add_argument("--legosim-dir", default="/mnt/d/legosim/LEGOSIM_MICRO")
    ap.add_argument("--bench-dir", default=None,
                    help="benchmark dir to run interchiplet in (legosim mode)")
    ap.add_argument("--gurobi-python",
                    default=os.environ.get("GUROBI_PYTHON", sys.executable))
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--mipgap", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--contention-k", type=float, default=0.6,
                    help="contention model strength (model sim only)")
    ap.add_argument("--out", default="out/loop")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    grid = {"L": a.L, "X": a.X, "Y": a.Y}
    budgets = {"a_max": a.a_max, "p_max": a.p_max, "p_col": a.p_col,
               "e_bar": a.e_bar}

    base_params = json.load(open(a.params))
    tau_h = base_params["interconnect"]["tau_h"]
    t0 = base_params["interconnect"]["t0"]
    t_hat = None
    log = {"iterations": [], "config": {"grid": grid, "budgets": budgets,
                                        "rho": a.rho, "tol": a.tol,
                                        "sim": a.sim}}

    print("=" * 64)
    print("CLOSED LOOP  (sim=%s, rho=%.2f, tol=%.0f%%, max_iter=%d)"
          % (a.sim, a.rho, a.tol * 100, a.max_iter))
    print("=" * 64)

    converged = False
    for it in range(a.max_iter):
        # ---- profile step: patch tau_h + t0 into a working params ----
        work_params = os.path.join(a.out, "params_iter%d.json" % it)
        patch_interconnect(a.params, tau_h, t0, work_params)

        # ---- SOLVE ----
        out_sol = os.path.join(a.out, "sol_iter%d.json" % it)
        sol = solve_ilp(work_params, grid, budgets, tau_h, t_hat, out_sol,
                        a.gurobi_python, a.time_limit, a.mipgap, a.seed)
        T_pred = sol["predicted_makespan_cycles"]

        # ---- CODEGEN ----
        cg_dir = os.path.join(a.out, "legosim_iter%d" % it)
        subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(__file__), "codegen.py"),
                        "--params", work_params, "--sol", out_sol,
                        "--router-mode", a.router_mode,
                        "--out-dir", cg_dir], capture_output=True, text=True)

        # ---- SIMULATE ----
        if a.sim == "model":
            T_meas, load, gamma = simulate_model(base_params, sol,
                                                 a.contention_k)
        else:
            bench_dir = a.bench_dir or cg_dir
            T_meas, raw = simulate_legosim(bench_dir, a.legosim_dir)
            if T_meas is None:
                print("[loop] LEGOSim sim failed (%s); falling back to model"
                      % raw[:200])
                T_meas, load, gamma = simulate_model(base_params, sol,
                                                     a.contention_k)
            else:
                gamma = T_meas / T_pred if T_pred > 0 else 1.0

        # Communication-residual congestion multiplier (Section 4.5/6). The
        # measured makespan is C (real compute critical path, fixed) + a comm
        # term the affine ILP under-predicts. Fitting gamma to the RESIDUAL
        # (T-C) -- not the global T -- is what gives the recalibration real
        # leverage: t0/tau_h only affect the comm part, so a global gamma
        # (~1.38) would barely move T_pred while the residual gamma (~4.7)
        # actually closes the gap in <=4 damped steps.
        C = compute_only_makespan(base_params, sol)
        comm_pred = max(T_pred - C, 1e-9)
        comm_meas = max(T_meas - C, 0.0)
        gamma_comm = comm_meas / comm_pred if comm_pred > 0 else 1.0

        err = abs(T_meas - T_pred) / T_meas if T_meas else 0.0
        rec = {"iter": it, "tau_h": tau_h, "t0": t0, "T_pred": T_pred,
               "T_meas": T_meas, "compute_path_C": C, "gamma": gamma,
               "gamma_comm": gamma_comm, "rel_err": err,
               "mip_gap": sol.get("mip_gap"), "solver_wall_s":
               sol.get("solver_wall_s"), "sites": len(sol["sites"])}
        log["iterations"].append(rec)
        print("[iter %d] tau_h=%.3f t0=%.0f  T_pred=%.3e  T_meas=%.3e  "
              "C=%.3e  gamma_comm=%.3f  err=%.1f%%  gap=%.3f%%  wall=%.1fs"
              % (it, tau_h, t0, T_pred, T_meas, C, gamma_comm, err * 100,
                 (sol.get("mip_gap") or 0) * 100, sol.get("solver_wall_s") or 0))

        if err < a.tol:
            print("[loop] CONVERGED at iter %d (err %.1f%% < tol %.0f%%)"
                  % (it, err * 100, a.tol * 100))
            converged = True
            log["final_solution"] = out_sol
            break

        # ---- RECALIBRATE (damped) + set T_hat for C14 next iter ----
        # Damped fixed-point on BOTH latency terms (Section 6), driven by the
        # communication-residual multiplier gamma_comm. rho relaxes toward the
        # fixed point so the iteration cannot oscillate (doc Section 6).
        tau_h_new = a.rho * gamma_comm * tau_h + (1 - a.rho) * tau_h
        t0_new = a.rho * gamma_comm * t0 + (1 - a.rho) * t0
        print("[iter %d] recalibrate tau_h: %.3f -> %.3f   t0: %.1f -> %.1f "
              "(damped, rho=%.2f)"
              % (it, tau_h, tau_h_new, t0, t0_new, a.rho))
        tau_h = tau_h_new
        t0 = t0_new
        t_hat = T_meas    # successive linearization point for C14

    if not converged:
        print("[loop] stopped at max_iter=%d (best error %.1f%%)"
              % (a.max_iter, log["iterations"][-1]["rel_err"] * 100))
        log["final_solution"] = os.path.join(
            a.out, "sol_iter%d.json" % (len(log["iterations"]) - 1))

    log["converged"] = converged
    with open(os.path.join(a.out, "loop_log.json"), "w") as fh:
        json.dump(log, fh, indent=2)
    print("[loop] wrote %s/loop_log.json  (final: %s)"
          % (a.out, log["final_solution"]))


if __name__ == "__main__":
    main()
