"""Summarize the kernel-selection lanes (p3_krr_select.py): each criterion against the forward choice of the same family.

For every family (krr, ard) and criterion (J_u floored, K_u cut) the script takes the paired difference from the
forward arm of the same run in the radiance error and in the 95th percentiles of the retrieval error (all bands, and on
the physical domain above 1e-12, 1e-3 and 1e-2 T_train), and reports the mean, the sample standard deviation over splits
and the number of splits on which the criterion's arm is lower. With --p2 <dir of the second paper's run records
(tq_s<seed>_raw_w512.json)>, it also checks that the forward arm chose the second paper's hyperparameters on the same
split: the same nu, scale, nugget and input weights, and the same subsample validation error to a relative 1e-6.

usage: python code/summarize_p3ks.py <record or directory> [...] [--p2 DIR] [--out results/p3ks_summary.json]
"""
import glob
import json
import os
import statistics as st
import sys

args = sys.argv[1:]
p2_dir = None
out_path = "results/p3ks_summary.json"
for flag in ("--p2", "--out"):
    if flag in args:
        i = args.index(flag)
        if flag == "--p2":
            p2_dir = args[i + 1]
        else:
            out_path = args[i + 1]
        del args[i:i + 2]

files = []
for a in args:
    if os.path.isdir(a):
        files += [p for p in glob.glob(os.path.join(a, "**", "*.json"), recursive=True)
                  if not p.endswith((".env.json", ".kaggle_status.json", ".record.json"))]
    else:
        files.append(a)
runs = {}
for p in sorted(set(files)):
    try:
        r = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        continue
    if isinstance(r, dict) and r.get("kind") == "p3_krr_select":
        if r["seed"] in runs:
            sys.exit(f"two records for seed {r['seed']}: {runs[r['seed']]['_path']} and {p}")
        r["_path"] = p
        runs[r["seed"]] = r
if not runs:
    sys.exit("no p3_krr_select records found")

KEYS = ["radiance", "allband_p95", "p95@1e-12", "p95@0.001", "p95@0.01"]
seeds = sorted(runs)
report = {"splits": seeds, "families": {}, "reproduction": {}}
print(f"{len(seeds)} runs: splits {seeds}")
for fam in ("krr", "ard"):
    arms = sorted({a for r in runs.values() for a in r["arms"] if a.startswith(fam + "_") and a != fam + "_forward"})
    if not arms:
        continue
    print(f"\n[{fam}]  difference from {fam}_forward of the same run (mean +- sd over splits, splits lower)")
    print(f"{'arm':>12s} {'n':>3s}" + "".join(f"{k:>24s}" for k in KEYS))
    fam_rep = {}
    base = {s: runs[s]["arms"].get(fam + "_forward") for s in seeds}
    for arm in arms:
        row, rep = f"{arm:>12s}", {}
        have = [s for s in seeds if arm in runs[s]["arms"] and base[s] is not None]
        row += f" {len(have):3d}"
        for k in KEYS:
            d = [runs[s]["arms"][arm][k] - base[s][k] for s in have]
            if not d:
                row += f"{'-':>24s}"
                continue
            m, sd = st.mean(d), (st.stdev(d) if len(d) > 1 else 0.0)
            lower = sum(x < 0 for x in d)
            rep[k] = {"mean": m, "sd": sd, "lower": lower, "n": len(d), "values": dict(zip(map(str, have), d))}
            row += f"{m:+11.4f}+-{sd:8.4f} {lower:2d}/{len(d):<2d}"
        fam_rep[arm] = rep
        print(row)
    report["families"][fam] = fam_rep
    fwd = [base[s] for s in seeds if base[s] is not None]
    print(f"   {fam}_forward itself: radiance {st.mean(f['radiance'] for f in fwd):.4f}%, "
          f"p95@1e-3 {st.mean(f['p95@0.001'] for f in fwd):.3f}, p95@1e-2 {st.mean(f['p95@0.01'] for f in fwd):.3f} "
          f"(means over {len(fwd)} splits)")

if p2_dir:
    print("\nforward arm against the second paper's tuned hyperparameters (same split):")
    for s in seeds:
        p = os.path.join(p2_dir, f"tq_s{s}_raw_w512.json")
        if not os.path.exists(p):
            print(f"  s{s}: no record {os.path.basename(p)}")
            continue
        hyper = json.load(open(p, encoding="utf-8"))["hyper"]
        res = {}
        for fam in ("krr", "ard"):
            arm = runs[s]["arms"].get(fam + "_forward")
            if arm is None or fam not in hyper:
                continue
            for c, h2 in hyper[fam].items():
                h3 = arm["hp"][c]
                same = (h2["nu"] == h3["nu"] and h2["scale"] == h3["scale"] and h2["nugget"] == h3["nugget"]
                        and abs(h2["val_sub"] - h3["val_sub"]) <= 1e-6 * abs(h2["val_sub"])
                        and (fam != "ard" or [round(x, 4) for x in h2["w"]] == h3["w"]))
                res[f"{fam}_{c}"] = same
                if not same:
                    print(f"  s{s} {fam} {c}: DIFFERS  p2 {h2}  here {h3}")
        report["reproduction"][str(s)] = res
        print(f"  s{s}: {sum(res.values())}/{len(res)} component choices identical")

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)
print("\nwrote", out_path)
