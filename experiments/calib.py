# -*- coding: utf-8 -*-
"""Analytical T (exact protocol evaluator) vs. measured LEGOSim cycles for every
layout in results/legosim/measured.csv."""
from __future__ import print_function
import os
import ispa_common as C
import drift_lib as D


def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i])
    out = [0] * len(v)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
            j += 1
        for k in range(i, j + 1):
            out[s[k]] = (i + j) / 2.0
        i = j + 1
    return out


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = sum((x - ma) ** 2 for x in a) ** .5
    sb = sum((y - mb) ** 2 for y in b) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def main():
    rows = []
    for r in C.read_measured():
        T = D.FastEval(C.read_json(r["params"])).T(C.read_json(r["layout"]))
        rows.append({"case": r["case"], "method": r["method"], "measured": r["measured_cycles"],
                     "analytic": T, "ratio": r["measured_cycles"] / T})
    for r in rows:
        print("%-9s %-16s meas=%9d  T=%11.1f  meas/T=%.4f" % (
            r["case"], r["method"], r["measured"], r["analytic"], r["ratio"]))
    C.write_json(os.path.join(C.RESULTS, "calibration.json"), rows)
    for cid in sorted(set(r["case"] for r in rows)):
        rr = [r for r in rows if r["case"] == cid]
        a = [r["analytic"] for r in rr]
        m = [r["measured"] for r in rr]
        rat = [r["ratio"] for r in rr]
        print(cid, "n=%d" % len(rr), "spearman=%.3f" % pearson(rank(a), rank(m)),
              "ratio %.4f..%.4f" % (min(rat), max(rat)))
    a = [r["analytic"] for r in rows]
    m = [r["measured"] for r in rows]
    print("ALL n=%d pearson=%.4f spearman=%.4f" % (len(rows), pearson(a, m), pearson(rank(a), rank(m))))


if __name__ == "__main__":
    main()
