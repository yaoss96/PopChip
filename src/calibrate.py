#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate.py -- Section-5 cost-model calibration.

Turns MEASURED isolated-kernel cycle counts (from profile_kernels.sh) into a
params.json whose per-(task,type) compute costs are the real GPGPU-Sim numbers
instead of the analytic placeholders in workload.py.

Doc Section 5 contract:
    c_{v,GPU} = gpu_tot_sim_cycle from a single-chiplet run of the isolated
                kernel, per (task, chiplet-type).

Empirically (see profile CSV) the smoke-scale expert MLP kernel is essentially
token-INDEPENDENT: GPU-L ~6.4k cyc, GPU-S ~25k cyc, flat across 1..32 tokens.
So a single measured constant per chiplet type is a faithful c_{v,GPU}; we use
the MEDIAN across the token sweep to be robust to the +1 startup jitter.

Energy eta is scaled from the placeholder by the SAME ratio (measured/analytic)
so the epsilon-constraint stays consistent (we have no AccelWattch log in this
baremetal build; documented honestly -- energy is analytic * measured-cycle
ratio, latency is fully measured).

CPU control tasks (gate/norm) keep their analytic cost: they run bare-metal off
the makespan critical path (no Sniper on this host), so their exact cycle count
does not enter the measured makespan. This is stated in the design doc.

Usage:
  python3 calibrate.py --params params_smoke.json --profile profile_smoke.csv \
      -o params_measured.json
"""

import argparse
import csv
import json
import statistics


def load_profile(path):
    """Return {chiplet_type: median_cycles} from the profile CSV."""
    per_type = {}
    with open(path) as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            c = row.get("cycles")
            if c is None or c == "NA":
                continue
            per_type.setdefault(row["type"], []).append(float(c))
    return {t: statistics.median(v) for t, v in per_type.items() if v}


def main():
    ap = argparse.ArgumentParser(description="Calibrate params from profiling")
    ap.add_argument("--params", required=True,
                    help="base params.json (smoke-scale workload)")
    ap.add_argument("--profile", required=True,
                    help="profile CSV from profile_kernels.sh")
    ap.add_argument("-o", "--out", default="params_measured.json")
    ap.add_argument("--gpu-types", default="GPU-L,GPU-S",
                    help="comma list of library types the profiler covered")
    a = ap.parse_args()

    params = json.load(open(a.params))
    meas = load_profile(a.profile)
    gpu_types = [t.strip() for t in a.gpu_types.split(",") if t.strip()]

    missing = [t for t in gpu_types if t not in meas]
    if missing:
        raise SystemExit("profile CSV missing measured cycles for: %s"
                         % missing)

    tasks = params["tasks"]
    n_cal = 0
    cal_log = {}
    for v, spec in tasks.items():
        cyc = spec["cyc"]
        eta = spec["eta"]
        for k in list(cyc.keys()):
            if k in meas:
                old = cyc[k]
                new = meas[k]
                # scale energy by the same measured/analytic latency ratio so
                # the two stay physically consistent (no separate power log).
                if old > 0:
                    eta[k] = eta[k] * (new / old)
                cyc[k] = new
                n_cal += 1
                cal_log.setdefault(v, {})[k] = {"analytic": old, "measured": new}
            # CPU (and any type not profiled) is left as-is (off critical path).

    params.setdefault("meta", {})["calibration"] = {
        "source": "GPGPU-Sim isolated-kernel profiling (Section 5)",
        "measured_cycles_by_type": meas,
        "n_task_type_pairs_calibrated": n_cal,
        "note": "compute cycles = measured gpu_tot_sim_cycle (token-independent "
                "at smoke scale); CPU control tasks left analytic (off "
                "makespan critical path, baremetal router).",
    }
    with open(a.out, "w") as fh:
        json.dump(params, fh, indent=2)

    print("[calibrate] measured cycles by type: %s"
          % {k: round(v, 1) for k, v in meas.items()})
    print("[calibrate] calibrated %d (task,type) compute costs" % n_cal)
    for v, d in cal_log.items():
        for k, r in d.items():
            print("    %-12s %-6s  %.1f -> %.1f cyc"
                  % (v, k, r["analytic"], r["measured"]))
    print("[calibrate] wrote %s" % a.out)


if __name__ == "__main__":
    main()
