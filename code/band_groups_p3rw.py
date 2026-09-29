"""Where the weighted networks moved their accuracy: the pilot's arms against the plain arm of the same lane, by band.

The bands of each split are grouped by their median training transmission relative to T_train: opaque (below 1e-3),
absorbing (1e-3 to 1e-1) and window (above 1e-1). For every arm of every lane (pred_s<seed>_<arm><tag>.npz beside the
lane record) the script reports, per group, the median over the group's bands of the ratio of the arm's mean squared
error to the plain arm's, for the path radiance Y1 and the transmission Y2 + Y3, and the 95th percentile of the
retrieval error at rho = 0.7 over the group's entries on the physical domain above 1e-3 T_train (the pilot's domain:
0 <= s < 1, 1 - rho s >= 0.3, a failed inversion counted as not finite and left out, as in the pilot's score).

usage: EMIT_DATA=<dir> python code/band_groups_p3rw.py <lane output dirs ...> [--out results/p3rw_band_groups.json]
"""
import glob
import json
import os
import statistics as st
import sys

import numpy as np

args = sys.argv[1:]
out_path = "results/p3rw_band_groups.json"
if "--out" in args:
    i = args.index("--out")
    out_path = args[i + 1]
    del args[i:i + 2]
D = os.environ["EMIT_DATA"]
Ys = {c: np.load(os.path.join(D, c + ".npy")) for c in ("Y1", "Y2", "Y3", "Y4")}
T = Ys["Y2"] + Ys["Y3"]
RHO, DOM = 0.7, 1e-3


def split(seed):
    n_all = T.shape[0]
    perm = np.random.RandomState(seed).permutation(n_all)
    n_te = int(round(0.1 * n_all))
    idx_te, tr_full = perm[:n_te], perm[n_te:]
    vperm = np.random.RandomState(seed + 10000).permutation(len(tr_full))
    n_val = int(round(0.1 * len(tr_full)))
    return idx_te, tr_full[vperm[n_val:]]


def retrieval_err(P, rows):
    a, t, s = Ys["Y1"][rows], T[rows], Ys["Y4"][rows]
    ah, th, sh = P["Y1"], P["Y2"] + P["Y3"], P["Y4"]
    L = a + RHO * t / (1 - RHO * s)
    u = L - ah
    den = th + sh * u
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = u / den
    ok = np.isfinite(rho) & (den > 0)
    return np.abs(rho - RHO), ok


report = {}
for d in args:
    for f in sorted(glob.glob(os.path.join(d, "**", "pred_s*_plain*.npz"), recursive=True)):
        base = os.path.basename(f)
        seed = int(base.split("_")[1][1:])
        tag = base.split("_plain", 1)[1][:-4]
        idx_te, idx_tr = split(seed)
        zp = np.load(f)
        if not np.array_equal(zp["idx_te"], idx_te):
            sys.exit(f"{f}: test rows differ from the split")
        ttr = T[idx_tr]
        T_train = float(np.median(ttr[ttr > 0]))
        med = np.median(np.clip(ttr, 0, None), axis=0) / T_train
        groups = {"opaque": med < 1e-3, "absorbing": (med >= 1e-3) & (med < 1e-1), "window": med >= 1e-1}
        P0 = {c: zp[c] for c in ("Y1", "Y2", "Y3", "Y4")}
        err0, ok0 = retrieval_err(P0, idx_te)
        t_te, s_te = T[idx_te], Ys["Y4"][idx_te]
        dom = (t_te >= DOM * T_train) & (s_te >= 0) & (s_te < 1) & (1 - RHO * s_te >= 0.3)
        for g in sorted(glob.glob(os.path.join(os.path.dirname(f), f"pred_s{seed}_*{tag}.npz"))):
            arm = os.path.basename(g)[len(f"pred_s{seed}_"):-len(tag + ".npz")]
            if arm == "plain":
                continue
            za = np.load(g)
            P = {c: za[c] for c in ("Y1", "Y2", "Y3", "Y4")}
            err, ok = retrieval_err(P, idx_te)
            row = {"n_bands": {k: int(v.sum()) for k, v in groups.items()}}
            for name, gm in groups.items():
                mse = lambda A, B: ((A - B) ** 2).mean(0)  # noqa: E731
                r_a = mse(P["Y1"], Ys["Y1"][idx_te]) / mse(P0["Y1"], Ys["Y1"][idx_te])
                r_t = mse(P["Y2"] + P["Y3"], t_te) / mse(P0["Y2"] + P0["Y3"], t_te)
                m0, m1 = dom & ok0 & gm[None, :], dom & ok & gm[None, :]
                row[name] = {"mse_ratio_a": float(np.median(r_a[gm])) if gm.any() else None,
                             "mse_ratio_t": float(np.median(r_t[gm])) if gm.any() else None,
                             "p95_plain": 100 * float(np.quantile(err0[m0], 0.95)) if m0.any() else None,
                             "p95_arm": 100 * float(np.quantile(err[m1], 0.95)) if m1.any() else None}
            report.setdefault(arm, {})[str(seed) + tag] = row

print("median over lanes of the per-group numbers (arm / plain mse ratio for a and t; p95 on the domain, plain -> arm)")
summary = {}
for arm in sorted(report):
    lanes = report[arm]
    line = f"{arm:>16s} ({len(lanes):2d} lanes)"
    summary[arm] = {}
    for g in ("opaque", "absorbing", "window"):
        vals = [v[g] for v in lanes.values() if v[g]["mse_ratio_a"] is not None]
        if not vals:
            continue
        s_ = {k: st.median([v[k] for v in vals if v[k] is not None]) for k in ("mse_ratio_a", "mse_ratio_t", "p95_plain", "p95_arm")
              if any(v[k] is not None for v in vals)}
        summary[arm][g] = s_
        line += (f"  {g}: a x{s_['mse_ratio_a']:.2f} t x{s_['mse_ratio_t']:.2f}"
                 + (f" p95 {s_['p95_plain']:.2f}->{s_['p95_arm']:.2f}" if 'p95_arm' in s_ else ""))
    print(line)
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
json.dump({"per_lane": report, "median": summary}, open(out_path, "w", encoding="utf-8"), indent=1)
print("wrote", out_path)
