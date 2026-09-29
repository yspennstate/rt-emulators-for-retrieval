"""Summarize the retrieval-weighted training lanes: each weighted or joint arm against the plain arm of the same run.

Every lane record (pilot_s<seed><tag>.json or <tag>.json, written by retrieval_weighted_pilot.py) holds a plain arm
and one or more arms trained on the transmission-weighted objective (weighted_u<u>) or on the joint first-order
retrieval loss (joint_u<u>), all fitted in one session on one split. For every non-plain arm the script takes the
paired difference from that run's plain arm in the radiance error and in the 95th percentiles of the retrieval error
(all bands, and on the physical domain at the three floors), and reports per arm the mean and sample standard
deviation over splits and the number of splits on which the arm is lower. It also reports how far the plain arms of
two lanes on the same split agree, which bounds the run-to-run noise of the comparison.

With --profile <EMIT array dir>, it also computes for each arm the ratio of the conditional mean square of e_R
(|e_a| + R|e_t| + R^2 t|e_s|) to the plain arm's, in the transmission quantile bins of the second paper's profile
figure, from the saved predictions pred_s<seed>_<arm><tag>.npz beside each record.

usage: python code/summarize_p3rw.py <record or directory> [...] [--profile <dir>] [--out results/p3rw_summary.json]
"""
import glob
import json
import os
import statistics as st
import sys

import numpy as np

args = sys.argv[1:]
prof_dir = out_path = None
if "--profile" in args:
    i = args.index("--profile")
    prof_dir = args[i + 1]
    del args[i:i + 2]
out_path = "results/p3rw_summary.json"
if "--out" in args:
    i = args.index("--out")
    out_path = args[i + 1]
    del args[i:i + 2]

files = []
for a in args:
    if os.path.isdir(a):
        files += [p for p in glob.glob(os.path.join(a, "**", "*.json"), recursive=True)
                  if not p.endswith((".env.json", ".kaggle_status.json"))]
    else:
        files.append(a)
runs, seen = [], set()
for p in sorted(set(files)):
    try:
        r = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        continue
    if isinstance(r, dict) and "arms" in r and "plain" in r.get("arms", {}):
        key = json.dumps([r["seed"], r["arms"]], sort_keys=True)   # a lane writes pilot_s<seed><tag>.json and <tag>.json
        if key in seen:
            continue
        seen.add(key)
        r["_path"] = p
        runs.append(r)
if not runs:
    sys.exit("no lane records with a plain arm found")

KEYS = ["radiance", "allband_p95", "p95@1e-12", "p95@0.001", "p95@0.01"]


def config(r):
    """Runs are compared only within one configuration: the data sizes and the training setup. T_train, the median
    training transmission, differs from split to split and is a property of the split, not of the configuration."""
    return (f"n_test={r['n_test']} n_train={r['n_train']} widths={'x'.join(map(str, r['widths']))} "
            f"epochs={r['epochs']} members={r.get('members', 1)}")   # p3_retrieval_stopping.py trains one member


def ms(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else float("nan"))


summary = {"runs": len(runs), "configs": {}}
for cfg in sorted({config(r) for r in runs}):
    group = [r for r in runs if config(r) == cfg]
    diffs, plains = {}, {}
    for r in group:
        base = r["arms"]["plain"]
        plains.setdefault(r["seed"], []).append(base)
        for name, arm in r["arms"].items():
            if name == "plain":
                continue
            d = diffs.setdefault(name, {k: [] for k in KEYS} | {"seeds": []})
            d["seeds"].append(r["seed"])
            for k in KEYS:
                d[k].append(arm[k] - base[k])
    out = {"runs": len(group), "seeds": sorted({r["seed"] for r in group}), "arms": {}}
    print(f"\n[{cfg}] {len(group)} lane records, splits {out['seeds']}")
    print(f"{'arm':>16} {'n':>3} {'d radiance [pp]':>18} {'lower':>6} "
          + " ".join(f"{'d ' + k:>16} {'lower':>6}" for k in KEYS[1:]))
    for name in sorted(diffs):
        d = diffs[name]
        n = len(d["seeds"])
        row = {"splits": d["seeds"]}
        line = f"{name:>16} {n:>3}"
        for k in KEYS:
            m, s = ms(d[k])
            lower = sum(x < 0 for x in d[k])
            row[k] = {"mean": m, "sd": s, "lower": lower, "n": n, "values": d[k]}
            line += (f" {m:+9.4f}±{s:<7.4f} {lower:>3}/{n:<2}" if k == "radiance"
                     else f" {m:+8.2f}±{s:<6.2f} {lower:>3}/{n:<2}")
        out["arms"][name] = row
        print(line)
    agree = {s: {k: max(p[k] for p in ps) - min(p[k] for p in ps) for k in KEYS}
             for s, ps in plains.items() if len(ps) > 1}
    if agree:
        out["plain_arm_spread_same_split"] = agree
        worst = {k: max(v[k] for v in agree.values()) for k in KEYS}
        print("largest spread between the plain arms of two lanes on one split:",
              ", ".join(f"{k} {v:.4f}" for k, v in worst.items()))
    summary["configs"][cfg] = out

if prof_dir:
    C = ("Y1", "Y2", "Y3", "Y4")
    Y = {c: np.load(os.path.join(prof_dir, c + ".npy"), mmap_mode="r") for c in C}
    R, RHO, QMIN = 0.9, 0.7, 0.3
    QS = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    prof = {}
    nb = Y["Y1"].shape[1]
    for r in runs:
        folder, seed = os.path.dirname(r["_path"]), r["seed"]
        tag = os.path.basename(r["_path"])[:-5]
        if tag.startswith("pilot_s"):                  # pilot_s<seed><tag>.json: the predictions carry <tag>
            tag = tag[len(f"pilot_s{seed}"):]
        cand = glob.glob(os.path.join(folder, f"pred_s{seed}_plain{tag}.npz"))
        if not cand:
            continue
        pl = np.load(cand[0])
        if pl["Y1"].shape[1] != nb or r["n_test"] != len(pl["idx_te"]):
            print(f"profile: {os.path.basename(r['_path'])} is not on these arrays ({pl['Y1'].shape[1]} bands), skipped")
            continue
        idx = pl["idx_te"]
        a, t, s = (np.asarray(Y["Y1"][idx]), np.asarray(Y["Y2"][idx]) + np.asarray(Y["Y3"][idx]),
                   np.asarray(Y["Y4"][idx]))
        T = float(r["T_train"])                      # the run's own median positive training transmission
        dom = (t >= 1e-12 * T) & (s >= 0) & (s < 1) & (1 - RHO * s >= QMIN)
        edges = np.quantile(t[dom], QS)

        def eR2(P):
            e = np.abs(P["Y1"] - a) + R * np.abs(P["Y2"] + P["Y3"] - t) + R * R * t * np.abs(P["Y4"] - s)
            return e * e

        base = eR2(pl)
        for name in r["arms"]:
            if name == "plain":
                continue
            f = os.path.join(folder, f"pred_s{seed}_{name}{tag}.npz")
            if not os.path.exists(f):
                continue
            q = eR2(np.load(f))
            ratios = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = dom & (t >= lo) & (t <= hi)
                ratios.append(float(q[m].mean() / base[m].mean()) if m.any() and base[m].mean() > 0 else None)
            prof.setdefault((config(r), name), []).append(ratios)
    if prof:
        summary["profile_ratio_bins_percent"] = ["0-1", "1-5", "5-10", "10-25", "25-50", "50-100"]
        for (cfg, name), v in sorted(prof.items()):
            med = [float(np.nanmedian([row[j] if row[j] is not None else np.nan for row in v])) for j in range(6)]
            summary["configs"][cfg].setdefault("profile_ratio_median", {})[name] = med
            print(f"profile [{cfg}] {name:>16} over {len(v)} runs: " + " ".join(f"{x:5.2f}" for x in med))
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
json.dump(summary, open(out_path, "w", encoding="utf-8", newline="\n"), indent=1)
print("wrote", out_path)
