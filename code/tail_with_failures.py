"""The retrieval tail with failed inversions counted as errors, from saved test-block predictions.

The scores of the drivers (the second paper's convention) leave a failed inversion - a predicted denominator that is not
positive, or a non-finite value - out of the 95th percentile and report the failure rate beside it. An arm that fails
more often can then show a smaller percentile. This script recomputes, for every arm of a preds file written by
p3_weight_family.py (--save-preds), on the physical domain above f T_train (f = 1e-12, 1e-3, 1e-2; 0 <= s < 1,
1 - rho s >= 0.3), the 95th percentile of |rho_hat - rho| in points with each failure counted as an infinite error,
beside the drivers' percentile and failure rate.

usage: python code/tail_with_failures.py --data <EMIT arrays> --preds <preds.npz> [--drop-rows 4011 7439]
       [--seed 101] [--out FILE]
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--preds", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--drop-rows", type=int, nargs="*", default=[])
ap.add_argument("--out", default="")
args = ap.parse_args()
C = ["Y1", "Y2", "Y3", "Y4"]
RHO = 0.7
Y = {c: np.load(f"{args.data}/{c}.npy") for c in C}
T = Y["Y2"] + Y["Y3"]
n_all = T.shape[0]
perm = np.random.RandomState(args.seed).permutation(n_all)
n_te = int(round(0.1 * n_all))
tr_full = perm[n_te:]
vperm = np.random.RandomState(args.seed + 10000).permutation(len(tr_full))
n_val = int(round(0.1 * len(tr_full)))
idx_tr = tr_full[vperm[n_val:]]
idx_tr = idx_tr[~np.isin(idx_tr, args.drop_rows)]
ttr = T[idx_tr]
T_train = float(np.median(ttr[ttr > 0]))
z = np.load(args.preds)
idx = z["idx_te"]
a, t, s = Y["Y1"][idx], T[idx], Y["Y4"][idx]
L = a + RHO * t / (1 - RHO * s)
arms = sorted({k.split("|")[0] for k in z.files if "|" in k})
rep = {"seed": args.seed, "T_train": T_train, "arms": {}}
for arm in arms:
    P = {c: z[f"{arm}|{c}"].astype(np.float64) for c in C}
    ah, th, sh = P["Y1"], P["Y2"] + P["Y3"], P["Y4"]
    u_ = L - ah
    den = th + sh * u_
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = u_ / den
    ok = np.isfinite(rho) & (den > 0)
    err = np.where(ok, np.abs(rho - RHO), np.inf)
    row = {}
    for f in (1e-12, 1e-3, 1e-2):
        dom = (t >= f * T_train) & (s >= 0) & (s < 1) & (1 - RHO * s >= 0.3)
        e = err[dom]
        row[f"{f:g}"] = {"p95_failures_infinite": 100 * float(np.quantile(e, 0.95)) if np.isfinite(np.quantile(e, 0.95)) else None,
                         "p95_survivors": 100 * float(np.quantile(e[np.isfinite(e)], 0.95)),
                         "failed_pct": 100 * float((~np.isfinite(e)).mean()),
                         "above_5_points_pct": 100 * float((e > 0.05).mean())}
    rep["arms"][arm] = row
    print(f"{arm:16s} " + " | ".join(
        f"@{f}: surv {v['p95_survivors']:6.2f} inf-fail {('%6.2f' % v['p95_failures_infinite']) if v['p95_failures_infinite'] is not None else '   inf'} "
        f"fail {v['failed_pct']:5.2f}% >5pt {v['above_5_points_pct']:5.2f}%" for f, v in row.items()))
if args.out:
    json.dump(rep, open(args.out, "w", encoding="utf-8"), indent=1)
    print("wrote", args.out)
