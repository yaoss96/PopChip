# -*- coding: utf-8 -*-
"""Build popularity-BLIND selection variants of an ILP solution.

Joint ILP-Perf assigns the fast GPU-L dies to the hottest experts (popularity
aware).  These variants keep the SAME bill of materials and the SAME placement
(coordinates), but randomly permute which expert runs on which expert-bearing
die, so the fast dies no longer land on the hot experts.  Comparing the measured
LEGOSim cycles of ILP-Perf vs these variants isolates the value of joint,
popularity-aware SELECTION (compute assignment), independent of placement.

Non-expert control tasks (t_attn/t_combine/t_dispatch/t_gate/t_norm) stay on
their original sites; only the e(g) expert labels are permuted across the sites
that currently host an expert.
"""
from __future__ import print_function
import argparse, copy, json, os, random, re

EXPERT = re.compile(r"^e\((\d+)\)$")


def _is_expert(task):
    return bool(EXPERT.match(task.strip()))


def load(path):
    with open(path) as fh:
        return json.load(fh)


def expert_bearing_sites(sol):
    """Return list of (site_index, expert_task) for every site with one e(g)."""
    out = []
    for idx, s in enumerate(sol["sites"]):
        experts = [t for t in (s.get("tasks") or []) if _is_expert(t)]
        if len(experts) > 1:
            raise ValueError("site hosts >1 expert; q>1 not handled here")
        if experts:
            out.append((idx, experts[0]))
    return out


def make_variant(sol, seed):
    v = copy.deepcopy(sol)
    pairs = expert_bearing_sites(v)
    site_indices = [p[0] for p in pairs]
    experts = [p[1] for p in pairs]
    perm = list(experts)
    rng = random.Random(seed)
    # derangement-ish shuffle: reject identity so the assignment truly changes
    for _ in range(200):
        rng.shuffle(perm)
        if perm != experts:
            break
    # rewrite site task lists and task_site
    for (idx, old_expert), new_expert in zip(pairs, perm):
        tasks = v["sites"][idx].get("tasks") or []
        v["sites"][idx]["tasks"] = [new_expert if _is_expert(t) else t
                                    for t in tasks]
    # task_site: expert -> its new site coordinate
    site_coord = {}
    for idx in site_indices:
        s = v["sites"][idx]
        site_coord[idx] = [s["l"], s["i"], s["j"]]
    for (idx, _old), new_expert in zip(pairs, perm):
        v["task_site"][new_expert] = list(site_coord[idx])
    v["_selection_variant"] = {"kind": "popularity_blind_random_assignment",
                               "seed": seed}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True, help="e.g. e008_m06")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    sol = load(a.solution)
    os.makedirs(a.out_dir, exist_ok=True)
    pairs = expert_bearing_sites(sol)
    gpu_l = [e for (i, e) in pairs if sol["sites"][i]["type"] == "GPU-L"]
    print("%s: %d expert-bearing sites; ILP puts GPU-L on experts %s"
          % (a.tag, len(pairs), sorted(gpu_l, key=lambda x: int(EXPERT.match(x).group(1)))))
    for seed in a.seeds:
        v = make_variant(sol, seed)
        vp = [(i, e) for (i, e) in expert_bearing_sites(v)]
        gl = [e for (i, e) in vp if v["sites"][i]["type"] == "GPU-L"]
        out = os.path.join(a.out_dir, "%s_blind_seed%d.json" % (a.tag, seed))
        json.dump(v, open(out, "w"), indent=1)
        print("  seed%d -> GPU-L now on experts %s  (%s)"
              % (seed, sorted(gl, key=lambda x: int(EXPERT.match(x).group(1))),
                 os.path.basename(out)))


if __name__ == "__main__":
    main()
