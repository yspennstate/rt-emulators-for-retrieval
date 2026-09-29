"""The forward-inverse reversal of the stack and the feature kernel, read through the change of measure.

For each saved run of the training-target campaign (raw arm) and each floor tau0, on the physical domain
t >= tau0 T_train, 0 <= s < 1, q = 1 - rho s >= 0.3 at rho = 0.7, the script computes for the convex stack (f) and the
kernel on features (h):
  d = v_rho' e with v_rho = (q^2/t, rho q/t, rho^2), the first-order retrieval error (up to sign),
  r = (t/q^2) d, the first-order forward (radiance) error at rho,
and checks the identity E[r_f^2] - E[r_h^2] = E[g] E[Delta] + Cov(g, Delta) with g = t^2/q^4 and Delta = d_f^2 - d_h^2.
The reversal (forward order and retrieval order disagree) happens exactly when E[Delta] > 0 and
Cov(g, Delta) < -E[g] E[Delta]. It also reports the exact constrained retrieval error of both, to show whether the
first-order order is the realized one.

usage: python code/check_change_of_measure.py <EMIT data dir with Y1-Y4.npy> <run npz> [<run npz> ...]
"""
import json
import sys

import numpy as np

RHO, R, QMIN = 0.7, 0.9, 0.3
D = sys.argv[1]
Y = {c: np.load(f"{D}/{c}.npy", mmap_mode="r") for c in ("Y1", "Y2", "Y3", "Y4")}
out = {}
for path in sys.argv[2:]:
    z = np.load(path)
    te, tr = z["idx_te"], z["idx_tr"]
    ttr = np.asarray(Y["Y2"][tr]) + np.asarray(Y["Y3"][tr])
    T = float(np.median(ttr[ttr > 0]))
    a, t, s = (np.asarray(Y["Y1"][te]), np.asarray(Y["Y2"][te]) + np.asarray(Y["Y3"][te]), np.asarray(Y["Y4"][te]))
    S = float(np.max(np.asarray(Y["Y4"][tr])[np.asarray(Y["Y4"][tr]) < 1]))
    q = 1 - RHO * s
    L = a + RHO * t / q
    fam = {}
    for k in ("stack", "dkr"):
        ea, et, es = z[f"{k}_Y1"] - a, (z[f"{k}_Y2"] + z[f"{k}_Y3"]) - t, z[f"{k}_Y4"] - s
        d = q * q / t * ea + RHO * q / t * et + RHO * RHO * es
        th, sh = np.maximum(z[f"{k}_Y2"] + z[f"{k}_Y3"], 0), np.clip(z[f"{k}_Y4"], 0, S)
        den = th + sh * (L - z[f"{k}_Y1"])
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = np.clip((L - z[f"{k}_Y1"]) / den, 0, R)
        rb = np.where(den > 0, rb, 0.0)                 # a failed inversion returns a value in [0, R]
        fam[k] = (d, rb)
    row = {}
    for tau0 in (1e-12, 1e-3):
        m = (t >= tau0 * T) & (s >= 0) & (s < 1) & (q >= QMIN)
        g = (t * t / q ** 4)[m]
        df, dh = fam["stack"][0][m], fam["dkr"][0][m]
        Delta = df * df - dh * dh
        fw = [float(np.mean(g * df * df)), float(np.mean(g * dh * dh))]
        rt = [float(np.mean(df * df)), float(np.mean(dh * dh))]
        cov = float(np.mean(g * Delta) - np.mean(g) * np.mean(Delta))
        ex = [float(np.mean((fam[k][1][m] - RHO) ** 2)) for k in ("stack", "dkr")]
        row[f"{tau0:g}"] = dict(entries=int(m.sum()), forward_msq=fw, retrieval_msq_first_order=rt,
                                E_g=float(np.mean(g)), E_Delta=float(np.mean(Delta)), cov_g_Delta=cov,
                                identity_gap=float((fw[0] - fw[1]) - (np.mean(g) * np.mean(Delta) + cov)),
                                reversal=bool(fw[0] < fw[1] and rt[0] > rt[1]),
                                exact_constrained_msq=ex)
    out[path.replace("\\", "/").split("/")[-1][:-4]] = row
    print(path.split("/")[-1], json.dumps(row))
json.dump(out, open("results/change_of_measure_check.json", "w", encoding="utf-8", newline="\n"), indent=1)
