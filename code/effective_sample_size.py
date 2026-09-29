"""Effective sample size of retrieval-weighted training truncated at a floor.

With entry weights w = t^-2 1{t > tau} (the first-order retrieval weight of the path radiance, cut below the floor)
the effective sample size of a weighted fit is n_eff = (sum w)^2 / sum w^2. Under P(t <= u) ~ A u^beta with beta < 2
it is of order n P(t <= tau): the weight concentrates on the entries just above the floor. The script computes n_eff
over the entries (state, band) of the EMIT table, for the weights 1/max(t, tau)^2 of the retrieval-weighted lanes and
for the cut weights, at floors u T_train, and prints n_eff / n beside P(t <= tau).

usage: python code/effective_sample_size.py <EMIT array dir> [out.json]
"""
import json
import sys

import numpy as np

D = sys.argv[1]
t = (np.load(f"{D}/Y2.npy") + np.load(f"{D}/Y3.npy")).ravel()
s = np.load(f"{D}/Y4.npy").ravel()
dom = (t > 0) & (s >= 0) & (s < 1)
t = t[dom]
T = float(np.median(t))
out = {"entries": int(t.size), "T_median": T, "floors": {}}
for u in (1e-1, 1e-2, 1e-3, 1e-6, 1e-12):
    tau = u * T
    for kind, w in (("floored", 1.0 / np.maximum(t, tau) ** 2), ("cut", np.where(t > tau, t ** -2.0, 0.0))):
        w = w / w.max()                     # scale-free; avoids overflow in the squares
        neff = float(w.sum() ** 2 / (w ** 2).sum())
        out["floors"].setdefault(f"{u:g}", {})[kind] = neff / t.size
    out["floors"][f"{u:g}"]["P(t<=tau)"] = float(np.mean(t <= tau))
    f = out["floors"][f"{u:g}"]
    print(f"u={u:g}: n_eff/n floored {f['floored']:.4f}, cut {f['cut']:.4f}; P(t<=tau) {f['P(t<=tau)']:.4f}")
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w", encoding="utf-8", newline="\n"), indent=1)
