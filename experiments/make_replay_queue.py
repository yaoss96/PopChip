# -*- coding: utf-8 -*-
"""Write a replay queue for every layout in results/legosim/measured.csv.

Usage: python make_replay_queue.py [CASE_PREFIX] > queue.txt
       LEGOSIM_ROOT=/path/to/LEGOSIM_MICRO sh replay_queue.sh queue.txt
The queue reproduces the measured cycles (repeated replays agree within a few
cycles); outputs go to out/replay/<case>_<method>.
"""
from __future__ import print_function
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
prefix = sys.argv[1] if len(sys.argv) > 1 else ""
with io.open(os.path.join(ROOT, "results", "legosim", "measured.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if not r["case"].startswith(prefix):
            continue
        out = "out/replay/%s_%s" % (r["case"], r["method"])
        print("%s|%s:%s:%s|%s" % (r["params"], r["case"], r["method"], r["layout"], out))
