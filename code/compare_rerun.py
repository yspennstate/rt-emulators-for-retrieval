"""Check that the p3pr runs (the principal EMIT networks run again with their predictions saved) reproduce the networks
of the paper's Table 1: plain and floored at u = 0.1 against results/dgx/p3cl, flattened (kappa 1/2, u = 0.01) against
results/dgx/p3fc, split by split, in every score of the records at rho = 0.7.

usage: python code/compare_rerun.py [results/dgx/p3pr]
"""
import glob
import json
import os
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "results/dgx/p3pr"
KEYS = ["radiance", "allband_p95", "p95@1e-12", "p95@0.001", "p95@0.01", "failed_pct@0.001"]
worst = 0.0
n = 0
for f in sorted(glob.glob(os.path.join(src, "p3pr_s*.json"))):
    rr = json.load(open(f, encoding="utf-8"))
    s = rr["seed"]
    ref = {"plain": ("p3cl", "plain"), "w1_u0.1": ("p3cl", "w1_u0.1"), "flat0.5_u0.01": ("p3fc", "flat0.5_u0.01")}
    for arm, (fam, refarm) in ref.items():
        r0 = json.load(open(os.path.join("results/dgx", fam, f"{fam}_s{s}.json"), encoding="utf-8"))["arms"][refarm]
        r1 = rr["arms"][arm]
        d = max(abs(r1[k] - r0[k]) for k in KEYS)
        worst = max(worst, d)
        n += 1
        if d > 1e-4:
            print(f"s{s} {arm}: largest difference {d:.3g} " + ", ".join(f"{k} {r0[k]:.4f}->{r1[k]:.4f}" for k in KEYS))
print(f"{n} split-arm pairs compared; largest absolute difference in any score: {worst:.3g}")
