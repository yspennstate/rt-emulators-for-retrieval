"""Where the plain network changes when the two failed evaluations of the EMIT table leave its training block.

Compares the test-block predictions of the plain network fitted on the table as provided (pred_s<seed>_plain<tag>.npz
of the pilot) with those of the plain arm fitted without rows 4011 and 7439 (the preds file of p3_weight_family.py
with --drop-rows), on the test states the two runs share. Reports, per component, the mean relative l2 error across
bands and its change, and per band group (opaque, absorbing, window: median training transmission below 1e-3, between
1e-3 and 1e-1, above 1e-1 T_train), the mean squared error of each component in units of the clean training standard
deviation, for both fits. Also the training standard deviation of each component with and without the two rows.

usage: python code/failed_rows_effect.py --data <EMIT arrays> --seed 101 --asis <pilot plain preds .npz>
       --clean <p3_weight_family preds .npz> [--arm plain] [--out results/failed_rows_effect_s101.json]
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--asis", required=True)
ap.add_argument("--clean", required=True)
ap.add_argument("--arm", default="plain")
ap.add_argument("--out", default="")
args = ap.parse_args()
C = ["Y1", "Y2", "Y3", "Y4"]
BAD = [4011, 7439]
Y = {c: np.load(f"{args.data}/{c}.npy") for c in C}
T = Y["Y2"] + Y["Y3"]
n_all = T.shape[0]
perm = np.random.RandomState(args.seed).permutation(n_all)
n_te = int(round(0.1 * n_all))
tr_full = perm[n_te:]
vperm = np.random.RandomState(args.seed + 10000).permutation(len(tr_full))
n_val = int(round(0.1 * len(tr_full)))
idx_tr = tr_full[vperm[n_val:]]
tr_clean = idx_tr[~np.isin(idx_tr, BAD)]
ttr = T[tr_clean]
T_train = float(np.median(ttr[ttr > 0]))
med = np.median(ttr, axis=0) / T_train
groups = {"opaque": med < 1e-3, "absorbing": (med >= 1e-3) & (med < 1e-1), "window": med >= 1e-1}

A = np.load(args.asis)
B = np.load(args.clean)
ia, ib = A["idx_te"], B["idx_te"]
common = np.intersect1d(ia, ib)
common = common[~np.isin(common, BAD)]
rb = {int(i): k for k, i in enumerate(ib)}
sel_b = np.array([rb[int(i)] for i in common])
pb = {c: B[f"{args.arm}|{c}"][sel_b].astype(np.float64) for c in C}
ra = {int(i): k for k, i in enumerate(ia)}
sel_a = np.array([ra[int(i)] for i in common])
pa = {c: A[c][sel_a] for c in C}
truth = {c: Y[c][common] for c in C}
sd_clean = {c: np.sqrt(Y[c][tr_clean].var(0)) for c in C}
sd_asis = {c: np.sqrt(Y[c][idx_tr].var(0)) for c in C}
rep = {"seed": args.seed, "test_states": int(len(common)), "bands_per_group": {g: int(m.sum()) for g, m in groups.items()},
       "components": {}}
for c in C:
    rel = lambda P: float(np.mean(np.linalg.norm(P - truth[c], axis=1) / np.linalg.norm(truth[c], axis=1)))  # noqa: E731
    s = np.where(sd_clean[c] > 0, sd_clean[c], 1.0)
    row = {"rel_err_asis_pct": 100 * rel(pa[c]), "rel_err_clean_pct": 100 * rel(pb[c]),
           "sd_ratio_asis_over_clean": {"median": float(np.median(sd_asis[c] / s)), "max": float(np.max(sd_asis[c] / s))},
           "std_mse": {}}
    for g, m in groups.items():
        ea = float(np.mean(((pa[c] - truth[c]) / s)[:, m] ** 2))
        eb = float(np.mean(((pb[c] - truth[c]) / s)[:, m] ** 2))
        row["std_mse"][g] = {"asis": ea, "clean": eb, "ratio_clean_over_asis": eb / ea if ea > 0 else None}
    rep["components"][c] = row
    print(f"{c}: rel err {row['rel_err_asis_pct']:.3f}% -> {row['rel_err_clean_pct']:.3f}%  sd ratio median "
          f"{row['sd_ratio_asis_over_clean']['median']:.2f} max {row['sd_ratio_asis_over_clean']['max']:.2f}  "
          + "  ".join(f"{g} {v['ratio_clean_over_asis']:.2f}" for g, v in row["std_mse"].items()))
if args.out:
    json.dump(rep, open(args.out, "w", encoding="utf-8"), indent=1)
    print("wrote", args.out)
