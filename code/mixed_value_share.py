"""How much of the mixed loss the retrieval term carries, at the plain network.

p3_mixed_objective.py trains each component network on the loss
    mean over entries of  ((1 - alpha) + alpha wbar) e~^2,
with e~ the standardized residual and wbar the floored retrieval weight of the component (times the band's variance,
so that wbar e~^2 is the physical weighted error) normalized to mean one on the training block. alpha is the share of
the WEIGHT. The share of the loss VALUE is
    S(alpha) = alpha r / ((1 - alpha) + alpha r),   r = mean(wbar e~^2) / mean(e~^2),
and at a minimizer of the plain loss it is the retrieval term alone that sets the gradient. For the plain arm of each
pilot lane (pred_s<seed>_plain<tag>.npz, the test block) the script reports r per component, S at the trained alphas,
and, on the training block, the effective fraction rho = mean(wbar)^2 / mean(wbar^2) of each component's weight with
the effective fraction of the mixture, 1 / (1 - alpha^2 + alpha^2 / rho).

usage: python code/mixed_value_share.py <dir with pilot lanes> --data <arrays dir> [--u 0.01]
       [--alpha 0.05 0.2 0.5] [--out results/mixed_value_share.json]
"""
import glob
import json
import os
import re
import statistics as st
import sys

import numpy as np

args = sys.argv[1:]
opts = {"--out": "results/mixed_value_share.json", "--data": os.environ.get("EMIT_DATA", ""), "--u": "0.01"}
alphas = [0.05, 0.2, 0.5]
if "--alpha" in args:
    i = args.index("--alpha")
    j = i + 1
    while j < len(args) and not args[j].startswith("--"):
        j += 1
    alphas = [float(a) for a in args[i + 1:j]]
    del args[i:j]
for k in list(opts):
    if k in args:
        i = args.index(k)
        opts[k] = args[i + 1]
        del args[i:i + 2]
ROOT = args[0]
U = float(opts["--u"])
C = ["Y1", "Y2", "Y3", "Y4"]
R = 0.9
Ys = {c: np.load(os.path.join(opts["--data"], c + ".npy")) for c in C}
T = Ys["Y2"] + Ys["Y3"]


def split(seed):
    """p3_mixed_objective.py's split."""
    n_all = T.shape[0]
    perm = np.random.RandomState(seed).permutation(n_all)
    n_te = int(round(0.1 * n_all))
    idx_te, tr_full = perm[:n_te], perm[n_te:]
    vperm = np.random.RandomState(seed + 10000).permutation(len(tr_full))
    n_val = int(round(0.1 * len(tr_full)))
    return idx_te, tr_full[vperm[n_val:]]


def raw_weights(rows, u, T_train, std):
    """p3_mixed_objective.py's weights()."""
    tau = u * T_train
    t = T[rows]
    den = np.maximum(t, tau) ** 2
    w = {"Y1": 1.0 / den, "Y2": R ** 2 / den, "Y3": R ** 2 / den,
         "Y4": R ** 4 * np.clip(t, 0, None) ** 2 / den}
    return {c: w[c] * std[c][None, :] ** 2 for c in C}


PLAIN = re.compile(r"pred_s(\d+)_plain(.*)\.npz$")
lanes = []
for f in sorted(glob.glob(os.path.join(ROOT, "**", "pred_s*_plain*.npz"), recursive=True)):
    m = PLAIN.search(os.path.basename(f))
    seed, tag = int(m.group(1)), m.group(2)
    z = np.load(f)
    idx_te, idx_tr = split(seed)
    if not np.array_equal(z["idx_te"], idx_te):
        sys.exit(f"{f}: test rows differ from the split of seed {seed}")
    ttr = T[idx_tr]
    T_train = float(np.median(ttr[ttr > 0]))
    std = {}
    for c in C:
        s = np.sqrt(Ys[c][idx_tr].var(0))
        s[s == 0] = 1.0
        std[c] = s
    wtr = raw_weights(idx_tr, U, T_train, std)
    wte = raw_weights(idx_te, U, T_train, std)
    row = {"lane": tag, "seed": seed, "r": {}, "S": {}, "rho": {}, "neff_mix": {}}
    for c in C:
        m_c = wtr[c].mean()
        wbar_tr, wbar_te = wtr[c] / m_c, wte[c] / m_c
        e2 = ((z[c] - Ys[c][idx_te]) / std[c][None, :]) ** 2
        r = float((wbar_te * e2).mean() / e2.mean())
        rho = float(wbar_tr.mean() ** 2 / (wbar_tr ** 2).mean())
        row["r"][c] = r
        row["rho"][c] = rho
        row["S"][c] = {f"{a:g}": a * r / ((1 - a) + a * r) for a in alphas}
        row["neff_mix"][c] = {f"{a:g}": 1.0 / (1 - a * a + a * a / rho) for a in alphas}
    lanes.append(row)
    print(f"{tag:14s} r " + " ".join(f"{c} {row['r'][c]:8.2f}" for c in C)
          + "   S(0.05) " + " ".join(f"{row['S'][c]['0.05']:.2f}" for c in C)
          + "   rho " + " ".join(f"{row['rho'][c]:.3f}" for c in C))

summary = {}
for c in C:
    summary[c] = {"r": {"median": st.median(l["r"][c] for l in lanes), "min": min(l["r"][c] for l in lanes),
                        "max": max(l["r"][c] for l in lanes)},
                  "rho": st.median(l["rho"][c] for l in lanes),
                  "S": {f"{a:g}": st.median(l["S"][c][f"{a:g}"] for l in lanes) for a in alphas},
                  "neff_mix": {f"{a:g}": st.median(l["neff_mix"][c][f"{a:g}"] for l in lanes) for a in alphas}}
print(json.dumps(summary, indent=1))
os.makedirs(os.path.dirname(opts["--out"]) or ".", exist_ok=True)
json.dump({"data": opts["--data"], "u": U, "alphas": alphas, "lanes": lanes, "summary": summary},
          open(opts["--out"], "w", encoding="utf-8"), indent=1)
print("wrote", opts["--out"])
