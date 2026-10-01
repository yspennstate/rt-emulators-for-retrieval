"""Every number that Section 5.7 (Kernel parameters chosen on the retrieval error) quotes, from the lane records.

Reads p3_krr_select.py records (the Kaggle lanes p3ks on the table as provided, p3kc without the two failed
evaluations) and prints, over the splits whose validation block holds no failed evaluation (split 105 is left out of
the lanes on the table as provided; a record with dropped_rows keeps every split): the mean scores of the forward
choice and of each retrieval choice, the per-split differences and the number of splits on which the choice is lower,
and the median share of the input weight of each input under the forward and the cut choice at u=1e-2 (input-scaled
kernel).

With --networks DIR (the clean-table network records of Section 5.3, p3_weight_family.py with --drop-rows), it also
pairs the input-scaled kernel chosen on the cut criterion at u=1e-2 with the plain network and the network floored at
u=1e-1, split by split: mean radiance error and tail above 1e-3 T_train, splits on which the kernel is lower, and the
median ratio of the tails.

usage: python code/kernel_section_numbers.py <lane outputs dir> [--keep-105] [--networks DIR]
"""
import glob
import json
import os
import statistics as st
import sys

args = sys.argv[1:]
keep105 = "--keep-105" in args
args = [a for a in args if a != "--keep-105"]
netdir = None
if "--networks" in args:
    i = args.index("--networks")
    netdir = args[i + 1]
    del args[i:i + 2]
recs = []
for f in sorted(glob.glob(os.path.join(args[0], "**", "*.json"), recursive=True)):
    try:
        r = json.load(open(f, encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        continue
    if isinstance(r, dict) and "arms" in r and "ard_forward" in r["arms"] and "seed" in r:
        if r["seed"] == 105 and not r.get("dropped_rows") and not keep105:
            continue
        if all(x["seed"] != r["seed"] for x in recs):
            recs.append(r)
print(f"{len(recs)} splits: {sorted(r['seed'] for r in recs)}; dropped rows: {recs[0].get('dropped_rows')}")
KEYS = ["radiance", "allband_p95", "p95@1e-12", "p95@0.001", "p95@0.01", "failed_pct@0.001"]
for fam in ("ard", "krr"):
    fwd = f"{fam}_forward"
    print(f"[{fam}] {fwd}: " + " ".join(f"{k} {st.mean(r['arms'][fwd][k] for r in recs):.3f}" for k in KEYS))
    for a in sorted(k for k in recs[0]["arms"] if k.startswith(fam + "_") and k != fwd):
        cells = []
        for k in KEYS:
            d = [r["arms"][a][k] - r["arms"][fwd][k] for r in recs]
            cells.append(f"{k.replace('p95@', '').replace('_pct@', '')} {st.mean(r['arms'][a][k] for r in recs):.3f} "
                         f"({st.mean(d):+.3f}, {sum(x < 0 for x in d)} lower, [{min(d):+.2f}, {max(d):+.2f}])")
        print(f"   {a:11s} " + " | ".join(cells))
names = ["aerosol", "elevation", "water vapour", "azimuth", "solar zenith", "view zenith"]
for a in ("ard_forward", "ard_K0.01"):
    for c in ("Y1", "Y2", "Y3", "Y4"):
        W = []
        for r in recs:
            w = r["arms"].get(a, {}).get("hp", {}).get(c, {}).get("w")
            if w:
                s = sum(float(x) for x in w)
                W.append([float(x) / s for x in w])
        if W:
            print(f"input-weight share {a} {c}: " + ", ".join(f"{n} {st.median(col):.2f}" for n, col in zip(names, zip(*W))))

if netdir:
    net = {}
    for f in sorted(glob.glob(os.path.join(netdir, "*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        if "arms" in r and "plain" in r["arms"] and r.get("dropped_rows"):
            net[r["seed"]] = r["arms"]
    ker = {r["seed"]: r["arms"]["ard_K0.01"] for r in recs}
    S = sorted(set(net) & set(ker))
    print(f"kernel ard_K0.01 against the networks, {len(S)} splits {S}")
    for arm in ("plain", "w1_u0.1"):
        for k in ("radiance", "p95@0.001"):
            kv = [ker[s][k] for s in S]
            nv = [net[s][arm][k] for s in S]
            print(f"   {arm:8s} {k:10s} kernel {st.mean(kv):.3f} network {st.mean(nv):.3f} "
                  f"kernel lower on {sum(a < b for a, b in zip(kv, nv))}, median ratio "
                  f"{st.median(a / b for a, b in zip(kv, nv)):.2f}, ratios {min(a / b for a, b in zip(kv, nv)):.2f}"
                  f"..{max(a / b for a, b in zip(kv, nv)):.2f}")
