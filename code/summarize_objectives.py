"""The retrieval objectives on the test block, each arm against the plain arm of the same run.

Records written by p3_mixed_objective.py, p3_flattened_weights.py and p3_weight_family.py carry, for every arm, the
floored objective J_u and (p3_weight_family.py) the cut objective K_u on the test block, at one or more floors. For
each arm and floor the script reports the ratio to the plain arm of the same run: median, range, and the number of
splits on which the arm is lower. With --splits it keeps only those seeds.

usage: python code/summarize_objectives.py <dir or records> [...] [--splits 101 102 ...] [--out FILE]
"""
import glob
import json
import os
import statistics as st
import sys

args = sys.argv[1:]
out_path, keep = None, None
if "--out" in args:
    i = args.index("--out")
    out_path = args[i + 1]
    del args[i:i + 2]
if "--splits" in args:
    i = args.index("--splits")
    j = i + 1
    while j < len(args) and not args[j].startswith("--"):
        j += 1
    keep = {int(s) for s in args[i + 1:j]}
    del args[i:j]
files = []
for a in args:
    files += sorted(glob.glob(os.path.join(a, "*.json"))) if os.path.isdir(a) else [a]
recs = []
for f in files:
    r = json.load(open(f, encoding="utf-8"))
    if "arms" in r and "plain" in r["arms"] and (keep is None or r["seed"] in keep):
        recs.append(r)
if not recs:
    sys.exit("no records with a plain arm")
table = {}
for r in recs:
    pl = r["arms"]["plain"]
    for name, a in r["arms"].items():
        if name == "plain":
            continue
        for obj in ("J_test", "K_test"):
            for fl, v in a.get(obj, {}).items():
                base = pl.get(obj, {}).get(fl)
                if base:
                    table.setdefault((name, obj, fl), []).append((r["seed"], v / base))
rep = {"records": len(recs), "seeds": sorted(r["seed"] for r in recs), "ratios": {}}
print(f"{len(recs)} records, seeds {rep['seeds']}")
for (name, obj, fl), vals in sorted(table.items()):
    xs = [x for _, x in vals]
    row = {"n": len(xs), "median": st.median(xs), "min": min(xs), "max": max(xs), "lower": sum(x < 1 for x in xs),
           "by_seed": {s: x for s, x in vals}}
    rep["ratios"][f"{name} {obj} {fl}"] = row
    print(f"{name:22s} {obj} {fl:7s} median {row['median']:7.3f}  range [{row['min']:.3f}, {row['max']:.3f}]  "
          f"lower {row['lower']}/{row['n']}")
if out_path:
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)
    print("wrote", out_path)
