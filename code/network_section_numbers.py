"""Every number that Section 5.3 (Networks fitted on the retrieval error) quotes for the clean table, from the records.

Reads the clean-table records (p3_weight_family.py with --drop-rows) and the pilot records on the table as provided,
and prints, in the order the text uses them: the plain network's scores; for every arm the mean difference from the
plain network and the number of splits on which it is lower; the relative drop of the tails at u=0.1; the failure rates;
the ratios of the floored and cut objectives on the test blocks to the plain network's (median, number lower); the
lowest J at each floor across arms; and the plain network with and without the failed rows, paired by split.

usage: python code/network_section_numbers.py [CLEAN_DIR ...] [--pilot DIR]
"""
import glob
import json
import os
import statistics as st
import sys

args = sys.argv[1:]
pilot = os.environ.get("P3_PILOT", "results/pilot")
if "--pilot" in args:
    i = args.index("--pilot")
    pilot = args[i + 1]
    del args[i:i + 2]
dirs = args or ["results/dgx/p3cl"]
recs = {}
for d in dirs:
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        if "arms" in r and "plain" in r["arms"] and r.get("dropped_rows"):
            recs.setdefault(r["seed"], {}).update(r["arms"])
S = sorted(recs)
print(f"clean splits: {len(S)} {S}")
KEYS = ["radiance", "allband_p95", "p95@1e-12", "p95@0.001", "p95@0.01"]
pl = {k: st.mean(recs[s]["plain"][k] for s in S) for k in KEYS}
print("plain means: " + ", ".join(f"{k} {v:.3f}" for k, v in pl.items()))
arms = sorted({a for s in S for a in recs[s] if a != "plain"})
for a in arms:
    ss = [s for s in S if a in recs[s]]
    cells = []
    for k in KEYS + ["failed_pct@0.001", "failed_pct@1e-12"]:
        d = [recs[s][a][k] - recs[s]["plain"][k] for s in ss]
        cells.append(f"{k.replace('p95@', '').replace('failed_pct@', 'fail')} {st.mean(d):+.3f}({sum(x < 0 for x in d)})")
    print(f"  {a:14s} n={len(ss)} " + " ".join(cells))
if all("w1_u0.1" in recs[s] for s in S):
    for k in KEYS[2:]:
        d = st.mean(recs[s]["w1_u0.1"][k] - recs[s]["plain"][k] for s in S)
        print(f"u=0.1 relative drop {k}: {-d / pl[k]:.3f}")
for a in ("plain", "w1_u0.1", "cut-w1_u0.01"):
    if all(a in recs[s] for s in S):
        f3 = [recs[s][a]["failed_pct@0.001"] for s in S]
        f12 = [recs[s][a]["failed_pct@1e-12"] for s in S]
        print(f"failures {a}: above 1e-3 mean {st.mean(f3):.3f} [{min(f3):.3f}, {max(f3):.3f}]; above 1e-12 "
              f"[{min(f12):.2f}, {max(f12):.2f}], over 5 pct on {sum(x > 5 for x in f12)}/{len(S)}")
print("objective ratios on the test blocks (arm / plain): median, lower on")
best = {}
for a in arms:
    for obj in ("J_test", "K_test"):
        for fl in sorted(recs[S[0]].get(a, {}).get(obj, {})):
            xs = [recs[s][a][obj][fl] / recs[s]["plain"][obj][fl] for s in S if a in recs[s]]
            print(f"  {a:14s} {obj} {fl:7s} {st.median(xs):8.3f} lower {sum(x < 1 for x in xs)}/{len(xs)}")
            if obj == "J_test":
                best.setdefault(fl, []).append((st.median(xs), a))
for fl, v in sorted(best.items()):
    print(f"lowest J_test {fl}: " + ", ".join(f"{a} {m:.3f}" for m, a in sorted(v)[:3]))
for fl in ("u0.01", "u0.001"):   # on how many splits each arm has the lowest J_test at this floor
    wins = {}
    for s in S:
        vals = {a: recs[s][a]["J_test"][fl] for a in arms if fl in recs[s][a].get("J_test", {})}
        a = min(vals, key=vals.get)
        wins[a] = wins.get(a, 0) + 1
    print(f"lowest J_test {fl} per split: {wins}")
prov = {}
for f in glob.glob(os.path.join(pilot, "**", "*.json"), recursive=True):
    try:
        r = json.load(open(f, encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        continue
    if isinstance(r, dict) and "arms" in r and "plain" in r["arms"] and "seed" in r and not r.get("dropped_rows"):
        prov.setdefault(r["seed"], r["arms"]["plain"])
P = [s for s in S if s in prov]
print(f"plain with and without the failed rows, paired on {len(P)} splits:")
for k in KEYS:
    d = [recs[s]["plain"][k] - prov[s][k] for s in P]
    print(f"  {k}: {st.mean(prov[s][k] for s in P):.3f} -> {st.mean(recs[s]['plain'][k] for s in P):.3f} "
          f"({st.mean(d):+.3f}, lower on {sum(x < 0 for x in d)})")
for c in ("Y1", "Y2", "Y3", "Y4"):
    print(f"  component {c}: {st.mean(prov[s]['components'][c] for s in P):.3f} -> "
          f"{st.mean(recs[s]['plain']['components'][c] for s in P):.3f}")
