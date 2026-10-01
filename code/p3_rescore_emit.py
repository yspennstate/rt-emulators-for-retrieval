"""Rescore the EMIT networks of p3_weight_family.py from their saved test predictions: the retrieval at several surface
reflectances, with failed inversions counted as infinite errors.

For every record <records>/<tag>.json with predictions <preds>/<tag>_preds.npz (written with --save-preds), the test
block is rebuilt with the driver's split code (seed, dropped rows) and checked against the record and the saved indices,
and at rho = 0.7 the radiance error and the failure-excluding tails are checked against the record. At each rho the
scores are those of p3_rescore_pkan.py on the EMIT table:

  radiance        mean over test states of the relative l2 error of the radiance over the 285 bands, %
  allband_p95     95th percentile of |rho_hat - rho| over all entries, points; a failed inversion counts as infinite
  p95@f           the same over the physical domain above f T_train (t >= f T_train, 0 <= s < 1, 1 - rho s >= 0.3)
  p95x@f          the failure-excluding tail of the paper's first version (failed inversions left out)
  failed_pct@f    share of the physical domain where the inversion fails (non-finite, or a non-positive denominator)

usage: EMIT_DATA=<dir> python code/p3_rescore_emit.py --records <dir> --preds <dir> --out results/dgx/p3pr_rescore
"""
import argparse
import glob
import json
import os
import pathlib

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--records", required=True)
p.add_argument("--preds", required=True)
p.add_argument("--out", default="results/dgx/p3pr_rescore")
p.add_argument("--rhos", nargs="+", type=float, default=[0.1, 0.4, 0.7, 0.9])
p.add_argument("--data", default=os.environ.get("EMIT_DATA", os.path.expanduser("~/p2/data/emit")))
args = p.parse_args()

FLOORS = (1e-12, 1e-3, 1e-2)
C = ["Y1", "Y2", "Y3", "Y4"]
OUT = pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
DATA = pathlib.Path(args.data)
X = np.load(DATA / "X.npy")
Ys = {c: np.load(DATA / (c + ".npy")) for c in C}
n_all = X.shape[0]
T_all = Ys["Y2"] + Ys["Y3"]


def blocks(seed, drop):
    """The driver's split, line for line (p3_weight_family.py, data section)."""
    perm = np.random.RandomState(seed).permutation(n_all)
    n_te = int(round(0.1 * n_all))
    idx_te, tr_full = perm[:n_te], perm[n_te:]
    vperm = np.random.RandomState(seed + 10000).permutation(len(tr_full))
    n_val = int(round(0.1 * len(tr_full)))
    idx_val, idx_tr = tr_full[vperm[:n_val]], tr_full[vperm[n_val:]]
    DROP = np.array(sorted(set(drop)), dtype=int)
    if DROP.size:
        idx_te, idx_val, idx_tr = (b[~np.isin(b, DROP)] for b in (idx_te, idx_val, idx_tr))
    return idx_te, idx_val, idx_tr


def quantile_with_inf(err, q=0.95):
    e = np.sort(err)
    if e.size == 0:
        return float("nan")
    pos = q * (e.size - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    if not np.isfinite(e[hi]):
        return float("inf") if (hi != lo or not np.isfinite(e[lo])) else float(e[lo])
    return float(e[lo] + (pos - lo) * (e[hi] - e[lo]))


def scores(Tt, P, T_train, rho):
    a, t, s = Tt["Y1"], Tt["Y2"] + Tt["Y3"], Tt["Y4"]
    ah, th, sh = P["Y1"], P["Y2"] + P["Y3"], P["Y4"]
    L = a + rho * t / (1 - rho * s)
    Lh = ah + rho * th / (1 - rho * sh)
    out = {"radiance": 100 * float(np.mean(np.linalg.norm(L - Lh, axis=1) / np.linalg.norm(L, axis=1)))}
    u_ = L - ah
    den = th + sh * u_
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = u_ / den
    ok = np.isfinite(rr) & (den > 0)
    err = np.where(ok, np.abs(rr - rho), np.inf)
    out["allband_p95"] = 100 * quantile_with_inf(err.ravel())
    out["failed_pct_all"] = 100 * float((~ok).mean())
    for f in FLOORS:
        dom = (t >= f * T_train) & (s >= 0) & (s < 1) & (1 - rho * s >= 0.3)
        out[f"p95@{f:g}"] = 100 * quantile_with_inf(err[dom])
        good = dom & ok
        out[f"p95x@{f:g}"] = 100 * float(np.quantile(np.abs(rr[good] - rho), 0.95)) if good.any() else float("nan")
        out[f"failed_pct@{f:g}"] = 100 * float((dom & ~ok).sum() / max(dom.sum(), 1))
        out[f"coverage@{f:g}"] = 100 * float(dom.mean())
    return out


n = 0
for rp in sorted(glob.glob(os.path.join(args.records, "*.json"))):
    rec = json.load(open(rp))
    tag = rec.get("tag")
    zp = os.path.join(args.preds, f"{tag}_preds.npz")
    if not tag or not os.path.isfile(zp):
        continue
    z = np.load(zp)
    drop = [int(k) for k in (rec.get("dropped_rows") or {})]
    idx_te, idx_val, idx_tr = blocks(rec["seed"], drop)
    assert np.array_equal(idx_te, z["idx_te"]), tag
    assert len(idx_te) == rec["n_test"] and len(idx_tr) == rec["n_train"], tag
    t_tr = T_all[idx_tr]
    T_train = float(np.median(t_tr[t_tr > 0]))
    assert abs(T_train - rec["T_train"]) <= 1e-12 * max(1.0, abs(T_train)), (tag, T_train, rec["T_train"])
    Tt = {c: Ys[c][idx_te].astype(np.float64) for c in C}
    names = sorted({k.split("|")[0] for k in z.keys() if "|" in k})
    out = {"tag": tag, "seed": rec["seed"], "T_train": T_train, "dropped_rows": rec.get("dropped_rows"), "rhos": args.rhos,
           "arms": {}, "check": {}}
    for arm in names:
        P = {c: z[f"{arm}|{c}"].astype(np.float64) for c in C}
        out["arms"][arm] = {f"{r:g}": scores(Tt, P, T_train, r) for r in args.rhos}
        ref = rec["arms"].get(arm)
        if ref is not None:
            mine = scores(Tt, P, T_train, 0.7)
            out["check"][arm] = max(abs(mine["radiance"] - ref["radiance"]) / max(ref["radiance"], 1e-12),
                                    *[abs(mine[f"p95x@{f:g}"] - ref[f"p95@{f:g}"]) / max(ref[f"p95@{f:g}"], 1e-12)
                                      for f in FLOORS])
    json.dump(out, open(OUT / f"{tag}.json", "w"), indent=1)
    n += 1
    worst = max(out["check"].values()) if out["check"] else float("nan")
    print(f"{tag}: {len(names)} arms rescored; largest relative deviation from the record at rho 0.7: {worst:.2e}")
print(f"{n} records rescored into {OUT}")
