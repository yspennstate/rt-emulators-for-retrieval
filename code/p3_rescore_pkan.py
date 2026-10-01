"""Rescore the benchmark networks of p3_pkan2.py from their saved test predictions: the retrieval at several surface
reflectances, with failed inversions counted as infinite errors.

For every record results/pkan2/<tag>.json whose run saved its predictions (<out>/<tag>__<arm>__te.npy, written with
--save-preds), the test block is rebuilt with the driver's own split code (the release's split, or the
out-of-distribution split of the record's seed) and checked against the record (rows, T_train, and at rho = 0.7 the
radiance error and the failure-excluding tails it reports). At each rho the scores are

  radiance        mean over test states of the relative l2 error of the top-of-atmosphere reflectance over bands, %
  allband_p95     95th percentile of |rho_hat - rho| over all rows, points; a failed inversion counts as infinite
  p95@f           the same over the physical domain above f T_train (t >= f T_train, 0 <= s < 1, 1 - rho s >= 0.3)
  p95x@f          the failure-excluding tail of the paper's first version (failed inversions left out)
  failed_pct@f    share of the physical domain where the inversion fails (non-finite, or a non-positive denominator)

A tail is infinite when more than 5% of its rows fail.

usage: DATA_NEW=<dir> python code/p3_rescore_pkan.py --records results/pkan2 --preds <dir of *__te.npy> --out results/pkan2_rescore
"""
import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--records", default="results/pkan2")
p.add_argument("--preds", required=True)
p.add_argument("--out", default="results/pkan2_rescore")
p.add_argument("--rhos", nargs="+", type=float, default=[0.1, 0.4, 0.7, 0.9])
args = p.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_data  # noqa: E402

FLOORS = (1e-12, 1e-3, 1e-2)
OUT = pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

root = bench_data.DN / "pkanrtm"
z = np.load(root / "paired_arrays.npz", allow_pickle=True)
X, Yh, sid, band = z["X"], z["Yh"], z["sid"], z["band"]
aero, prof, rel = bench_data._pkanrtm_cats(root, sid, band)


def masks_for(split, seed):
    """The driver's split, line for line (p3_pkan2.py, data section)."""
    if split == "official":
        return {"tr": rel == "train", "va": rel == "val", "te": rel == "test"}
    states, first = np.unique(sid, return_index=True)
    aod_s, cwv_s = X[first, 4], X[first, 5]
    qa, qc = np.quantile(aod_s, 0.85), np.quantile(cwv_s, 0.85)
    te_states = set(states[(aod_s > qa) | (cwv_s > qc)])
    rest = np.array(sorted(set(states) - te_states))
    perm = np.random.default_rng(seed).permutation(len(rest))
    va_states = set(rest[perm[:int(round(0.15 * len(rest)))]])
    is_te = np.isin(sid, list(te_states))
    is_va = np.isin(sid, list(va_states))
    return {"tr": ~(is_te | is_va), "va": is_va, "te": is_te}


def quantile_with_inf(err, q=0.95):
    """numpy's linear quantile, with +inf for failed rows: infinite when the quantile reaches them."""
    e = np.sort(err)
    if e.size == 0:
        return float("nan")
    pos = q * (e.size - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    if not np.isfinite(e[hi]):
        return float("inf") if (hi != lo or not np.isfinite(e[lo])) else float(e[lo])
    return float(e[lo] + (pos - lo) * (e[hi] - e[lo]))


def scores(yt, yp, s_ids, T_train, rho):
    a, t, s = yt[:, 0], yt[:, 1], yt[:, 2]
    ah, th, sh = yp[:, 0], yp[:, 1], yp[:, 2]
    L = a + rho * t / (1 - rho * s)
    Lh = ah + rho * th / (1 - rho * sh)
    order = np.argsort(s_ids, kind="stable")
    ss = s_ids[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    num = np.add.reduceat(((L - Lh) ** 2)[order], starts)
    den_s = np.add.reduceat((L ** 2)[order], starts)
    out = {"radiance": 100.0 * float(np.mean(np.sqrt(num) / np.sqrt(np.maximum(den_s, 1e-300))))}
    u_ = L - ah
    den = th + sh * u_
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = u_ / den
    ok = np.isfinite(rr) & (den > 0)
    err = np.where(ok, np.abs(rr - rho), np.inf)
    out["allband_p95"] = 100 * quantile_with_inf(err)
    out["failed_pct_all"] = 100 * float((~ok).mean())
    for f in FLOORS:
        dom = (t >= f * T_train) & (s >= 0) & (s < 1) & (1 - rho * s >= 0.3)
        out[f"p95@{f:g}"] = 100 * quantile_with_inf(err[dom])
        good = dom & ok
        out[f"p95x@{f:g}"] = 100 * float(np.quantile(np.abs(rr[good] - rho), 0.95)) if good.any() else float("nan")
        out[f"failed_pct@{f:g}"] = 100 * float((dom & ~ok).sum() / max(dom.sum(), 1))
        out[f"coverage@{f:g}"] = 100 * float(dom.mean())
    return out


summary = []
for rp in sorted(glob.glob(os.path.join(args.records, "*.json"))):
    rec = json.load(open(rp))
    tag = rec["tag"]
    files = sorted(glob.glob(os.path.join(args.preds, f"{tag}__*__te.npy")))
    if not files:
        print(f"{tag}: no saved predictions, skipped")
        continue
    m = masks_for(rec["split"], rec["seed"])
    rows = {k: int(v.sum()) for k, v in m.items()}
    assert rows == rec["rows"], (tag, rows, rec["rows"])
    yt = Yh[m["te"]].astype(np.float64)
    s_ids = sid[m["te"]]
    t_tr = np.clip(Yh[m["tr"]][:, 1], 0, None)
    T_train = float(np.median(t_tr[t_tr > 0]))
    assert abs(T_train - rec["T_train"]) <= 1e-12 * max(1.0, abs(T_train)), (tag, T_train, rec["T_train"])
    out = {"tag": tag, "split": rec["split"], "seed": rec["seed"], "T_train": T_train, "rhos": args.rhos, "arms": {}, "check": {}}
    for fp in files:
        arm = os.path.basename(fp)[len(tag) + 2:-len("__te.npy")]
        arm_rec = arm.replace("mix0.3_", "mix0.3:").replace("mix0.1_", "mix0.1:").replace("mix0.5_", "mix0.5:")
        yp = np.load(fp).astype(np.float64)
        assert yp.shape == yt.shape, (tag, arm, yp.shape, yt.shape)
        out["arms"][arm_rec] = {f"{r:g}": scores(yt, yp, s_ids, T_train, r) for r in args.rhos}
        ref = rec["arms"].get(arm_rec, {}).get("retrieval")
        if ref is not None:                       # the record's rho = 0.7 scores from the float64 predictions
            mine = scores(yt, yp, s_ids, T_train, 0.7)
            dev = max(abs(mine["radiance"] - ref["radiance"]) / max(ref["radiance"], 1e-12),
                      *[abs(mine[f"p95x@{f:g}"] - ref[f"p95@{f:g}"]) / max(ref[f"p95@{f:g}"], 1e-12) for f in FLOORS])
            out["check"][arm_rec] = dev
    worst = max(out["check"].values()) if out["check"] else float("nan")
    json.dump(out, open(OUT / f"{tag}.json", "w"), indent=1)
    summary.append((tag, len(out["arms"]), worst))
    print(f"{tag}: {len(out['arms'])} arms rescored; largest relative deviation from the record at rho 0.7: {worst:.2e}")
print(f"{len(summary)} records rescored into {OUT}")
