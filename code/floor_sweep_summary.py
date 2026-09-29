"""The training floor chosen on the validation block, from the floor-sweep lanes (p3fs on EMIT, p3fsl on libRadtran).

The rule was fixed before the runs (the queue comment of 24 Sep 2026): on each split, among the networks trained on
the floored weights w1@u (u = 1000, 1, 0.3, 0.1, 0.03, 0.01), take the one with the lowest cut objective K at 1e-2 on
the VALIDATION block, and score it on the test block against the plain network of the same run. The script reports
the floor chosen on each split; the chosen network's paired differences from the plain one (mean, sample standard
deviation, splits lower) in the radiance error, the four tails, the tail with failures counted (p95inf, None when more
than five percent fail) and the share above five points with failures counted; the same for every fixed floor, so
that the choice can be read against the whole sweep; the test-best floor on each split for comparison (not a
procedure: it looks at the test block); and the two controls, phys-w1 (the band variance alone) against plain and
against w1@1000, which differ from it only in the albedo weights.

usage: python code/floor_sweep_summary.py <dir or records> [...] [--rule-key K --rule-floor u0.01] [--out FILE]
"""
import glob
import json
import os
import statistics as st
import sys

args = sys.argv[1:]
out_path, rule_key, rule_floor = None, "K", "u0.01"
for flag in ("--out", "--rule-key", "--rule-floor"):
    if flag in args:
        i = args.index(flag)
        val = args[i + 1]
        del args[i:i + 2]
        if flag == "--out":
            out_path = val
        elif flag == "--rule-key":
            rule_key = val
        else:
            rule_floor = val
files = []
for a in args:
    files += sorted(glob.glob(os.path.join(a, "*.json"))) if os.path.isdir(a) else [a]
recs = {}
for f in files:
    try:
        r = json.load(open(f, encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        continue
    if isinstance(r, dict) and "plain" in r.get("arms", {}) and "seed" in r:
        recs.setdefault(r["seed"], {}).update(r["arms"])
if not recs:
    sys.exit("no records")
S = sorted(recs)
KEYS = ["radiance", "allband_p95", "p95@1e-12", "p95@0.001", "p95@0.01", "p95inf@0.001", "over5_pct@0.001",
        "failed_pct@0.001"]


def floored(a):
    return a.get("weight") == "floored" and a.get("family") == "w" and a.get("stopping") != "plain validation loss"


def diffs(pairs):
    """pairs: list of (arm scores, plain scores); returns {key: mean, sd, lower, n} over the pairs where both exist."""
    row = {}
    for k in KEYS:
        d = [x[k] - y[k] for x, y in pairs if x.get(k) is not None and y.get(k) is not None]
        if d:
            row[k] = {"mean": st.mean(d), "sd": st.stdev(d) if len(d) > 1 else 0.0, "lower": sum(v < 0 for v in d),
                      "n": len(d)}
    return row


def show(label, row):
    print(f"  {label:26s} " + " ".join(f"{k.replace('p95@', '').replace('_pct@', '')} {v['mean']:+.3f}({v['lower']}/{v['n']})"
                                       for k, v in row.items()))


rep = {"splits": S, "rule": f"lowest validation {rule_key} at {rule_floor} among the floored arms", "chosen": {},
       "oracle_on_test": {}}
chosen_pairs, oracle_pairs = [], []
for s in S:
    A = recs[s]
    cands = {n: a for n, a in A.items() if floored(a) and "val" in a}
    if not cands:
        continue
    pick = min(cands, key=lambda n: cands[n]["val"][rule_key][rule_floor])
    best = min(cands, key=lambda n: cands[n]["p95@0.001"])
    rep["chosen"][s] = {"arm": pick, "u": cands[pick]["u"],
                        "val_rule": {n: a["val"][rule_key][rule_floor] for n, a in sorted(cands.items(), key=lambda x: -x[1]["u"])}}
    rep["oracle_on_test"][s] = {"arm": best, "u": cands[best]["u"]}
    chosen_pairs.append((cands[pick], A["plain"]))
    oracle_pairs.append((cands[best], A["plain"]))
print(f"{len(S)} splits {S}; rule: {rep['rule']}")
print("floor chosen per split: " + ", ".join(f"{s}:{v['u']:g}" for s, v in rep["chosen"].items()))
print("test-best floor (for comparison only): " + ", ".join(f"{s}:{v['u']:g}" for s, v in rep["oracle_on_test"].items()))
pl = {k: st.mean(recs[s]["plain"][k] for s in S if recs[s]["plain"].get(k) is not None) for k in KEYS
      if any(recs[s]["plain"].get(k) is not None for s in S)}
print("plain means: " + ", ".join(f"{k} {v:.3f}" for k, v in pl.items()))
rep["plain_mean"] = pl
rep["chosen_vs_plain"] = diffs(chosen_pairs)
rep["oracle_vs_plain"] = diffs(oracle_pairs)
show("chosen on validation", rep["chosen_vs_plain"])
show("test-best (not a rule)", rep["oracle_vs_plain"])
rep["fixed_floor_vs_plain"] = {}
arm_names = sorted({n for s in S for n, a in recs[s].items() if n != "plain"},
                   key=lambda n: (-(recs[S[0]].get(n, {}).get("u") or 0), n))
for n in arm_names:
    pairs = [(recs[s][n], recs[s]["plain"]) for s in S if n in recs[s]]
    rep["fixed_floor_vs_plain"][n] = diffs(pairs)
    show(n, rep["fixed_floor_vs_plain"][n])
if all("phys-w1" in recs[s] and "w1_u1000" in recs[s] for s in S):
    rep["phys_vs_w1000"] = diffs([(recs[s]["phys-w1"], recs[s]["w1_u1000"]) for s in S])
    show("phys-w1 minus w1@1000", rep["phys_vs_w1000"])
if out_path:
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1)
    print("wrote", out_path)
