"""Every number that the paragraph on the EMIT table at four reflectances quotes, from results/dgx/p3pr_rescore and
results/dgx/members_rescore: per reflectance, the mean scores of the plain, floored and flattened networks with failed
inversions counted, the number of splits on which each weighted network is below the plain one, and the emulators of
the second paper on the same scores.

usage: python code/emit_rho_numbers.py
"""
import glob
import json
import math
import statistics as st

RHOS = ["0.1", "0.4", "0.7", "0.9"]
K = ["radiance", "allband_p95", "p95@1e-12", "p95@0.001", "p95@0.01"]
emit = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob("results/dgx/p3pr_rescore/p3pr_s*.json"))]
print(f"{len(emit)} splits")


def mean(xs):
    return float("inf") if any(math.isinf(x) for x in xs) else st.mean(xs)


for rho in RHOS:
    print(f"== rho {rho}")
    for arm in ("plain", "w1_u0.1", "flat0.5_u0.01"):
        ms = {k: mean([r["arms"][arm][rho][k] for r in emit]) for k in K + ["failed_pct@1e-12", "failed_pct@0.001",
                                                                          "p95x@1e-12", "p95x@0.001"]}
        low = {k: sum(1 for r in emit if r["arms"][arm][rho][k] < r["arms"]["plain"][rho][k]) for k in K}
        print(f"  {arm:14s} " + "  ".join(f"{k} {ms[k]:.4g}" + ("" if arm == "plain" else f" ({low[k]})") for k in K)
              + f"  fail12 {ms['failed_pct@1e-12']:.2f} fail3 {ms['failed_pct@0.001']:.3f}  excl12 {ms['p95x@1e-12']:.3g}"
              f" excl3 {ms['p95x@0.001']:.3g}")
    f_vs_fl = {k: sum(1 for r in emit if r["arms"]["flat0.5_u0.01"][rho][k] < r["arms"]["w1_u0.1"][rho][k]) for k in K}
    print("  flattened below floored on splits: " + "  ".join(f"{k} {v}" for k, v in f_vs_fl.items()))

# ---- the comparisons the paragraph quotes, from the same records
print("== failures counted against left out (mean over splits of the inclusive minus the exclusive tail)")
for arm in ("plain", "w1_u0.1", "flat0.5_u0.01"):
    for f in ("1e-12", "0.001"):
        d = [mean([r["arms"][arm][rho][f"p95@{f}"] - r["arms"][arm][rho][f"p95x@{f}"] for r in emit]) for rho in RHOS]
        print(f"  {arm:14s} floor {f:6s} " + "  ".join(f"rho {rho}: {x:+.3f}" for rho, x in zip(RHOS, d)))
print("== flattened over plain (ratio of the means; range of the split-by-split ratios)")
for f in ("1e-12", "0.001"):
    for rho in RHOS:
        a = [r["arms"]["flat0.5_u0.01"][rho][f"p95@{f}"] for r in emit]
        b = [r["arms"]["plain"][rho][f"p95@{f}"] for r in emit]
        rs = [x / y for x, y in zip(a, b)]
        print(f"  floor {f:6s} rho {rho}: {st.mean(a) / st.mean(b):.3f}  splits {min(rs):.3f}-{max(rs):.3f}")
mem = {}
for f in sorted(glob.glob("results/dgx/members_rescore/members_s*_w*.json")):
    r = json.load(open(f, encoding="utf-8"))
    mem[(r["seed"], r["width"])] = r
seeds = sorted({s for s, _ in mem})
print(f"== members: seeds {seeds}, widths {sorted({w for _, w in mem})}")
for r in emit:
    s = r["seed"]
    test_drop = sorted(k for k, v in r["dropped_rows"].items() if v == "test")
    for w in (512, 2000):
        m = mem.get((s, w))
        if m is None:
            print(f"  split {s} width {w}: no member record")
            continue
        if sorted(str(x) for x in m["dropped_from_test"]) != test_drop:
            print(f"  split {s} width {w}: dropped rows differ, members {m['dropped_from_test']} reruns {test_drop}")
print("  dropped-row check done")


def fam(w, name, rho, k):
    return [mem[(s, w)]["families"][name][rho][k] for s in seeds]


for w in (512, 2000):
    for rho in RHOS:
        rad_stack = sum(a < b for a, b in zip(fam(w, "stack", rho, "radiance"), fam(w, "dkr", rho, "radiance")))
        t3 = sum(a < b for a, b in zip(fam(w, "dkr", rho, "p95@0.001"), fam(w, "stack", rho, "p95@0.001")))
        t12 = sum(a < b for a, b in zip(fam(w, "dkr", rho, "p95@1e-12"), fam(w, "stack", rho, "p95@1e-12")))
        flat3 = [r["arms"]["flat0.5_u0.01"][rho]["p95@0.001"] for r in sorted(emit, key=lambda r: r["seed"])]
        dk_vs_flat = sum(a < b for a, b in zip(fam(w, "dkr", rho, "p95@0.001"), flat3))
        print(f"  width {w} rho {rho}: stack radiance below kernel-on-features {rad_stack}/10; kernel-on-features tail "
              f"below stack 1e-3 {t3}/10, 1e-12 {t12}/10; below the flattened network 1e-3 {dk_vs_flat}/10")
