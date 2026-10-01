"""Score the emulators of the second paper on the EMIT table without its two failed evaluations, from their saved test
predictions, with the scores of p3_rescore_emit.py: the retrieval at several surface reflectances, every failed
inversion counted as an infinite error.

The predictions are those of the second paper's training-target lanes with the drop-failed policy
(code/p2_members_dropfailed/): tq_s<seed>_dropfailed_w<width>.npz in --preds, one array per family and component
(ridge3: cubic ridge; krr: isotropic Matern kernel ridge; ard: input-scaled Matern kernel ridge; dnn: the network;
dnn_corr: the network with a kernel correction of its residuals; dkr: kernel ridge on the network's last hidden
features; stack: the second paper's convex stack of the six). The two failed evaluations are left out of the test
block when one falls in it (split 104), as in the paper's other scores. At rho = 0.7 the radiance error is checked
against the lane's own record (<records>/tq_s<seed>_dropfailed_w<width>.json) where it exists.

usage: EMIT_DATA=<dir> python code/p3_rescore_members.py --preds <dir> --records <dir> --out results/dgx/members_rescore
"""
import argparse
import glob
import json
import os
import pathlib
import re

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--preds", required=True)
p.add_argument("--records", default="")
p.add_argument("--out", default="results/dgx/members_rescore")
p.add_argument("--rhos", nargs="+", type=float, default=[0.1, 0.4, 0.7, 0.9])
p.add_argument("--data", default=os.environ.get("EMIT_DATA", os.path.expanduser("~/p2/data/emit")))
args = p.parse_args()

FLOORS = (1e-12, 1e-3, 1e-2)
FAILED = (4011, 7439)
C = ["Y1", "Y2", "Y3", "Y4"]
OUT = pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
DATA = pathlib.Path(args.data)
Ys = {c: np.load(DATA / (c + ".npy")) for c in C}
T_all = Ys["Y2"] + Ys["Y3"]


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
    """The scores of p3_rescore_emit.py."""
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
    return out


n = 0
for fp in sorted(glob.glob(os.path.join(args.preds, "tq_s*_dropfailed_w*.npz"))):
    m = re.search(r"tq_s(\d+)_dropfailed_w(\d+)\.npz$", fp)
    seed, width = int(m.group(1)), int(m.group(2))
    z = np.load(fp, allow_pickle=True)
    idx_te, idx_tr = z["idx_te"], z["idx_tr"]
    keep = ~np.isin(idx_te, FAILED)
    assert not np.isin(idx_tr, FAILED).any(), fp
    t_tr = T_all[idx_tr]
    T_train = float(np.median(t_tr[t_tr > 0]))
    Tt = {c: Ys[c][idx_te[keep]].astype(np.float64) for c in C}
    fams = sorted({re.sub(r"_Y[1-4]$", "", k) for k in z.keys() if re.search(r"_Y[1-4]$", k)})
    out = {"seed": seed, "width": width, "T_train": T_train, "n_test": int(keep.sum()),
           "dropped_from_test": [int(i) for i in idx_te[~keep]], "rhos": args.rhos, "families": {}, "check": {}}
    rec = None
    rp = os.path.join(args.records, f"tq_s{seed}_dropfailed_w{width}.json") if args.records else ""
    if rp and os.path.isfile(rp):
        rec = json.load(open(rp, encoding="utf-8"))
    for fam in fams:
        P = {c: z[f"{fam}_{c}"][keep].astype(np.float64) for c in C}
        out["families"][fam] = {f"{r:g}": scores(Tt, P, T_train, r) for r in args.rhos}
        if rec is not None and not keep.all():
            pass                                   # the lane scored the test block with the failed state in it
        elif rec is not None:
            ref = (rec.get("families") or {}).get(fam, {}).get("rel_l2_radiance")
            if ref is not None:
                mine = scores(Tt, P, T_train, 0.7)["radiance"] / 100
                out["check"][fam] = abs(mine - ref) / max(ref, 1e-12)
    json.dump(out, open(OUT / f"members_s{seed}_w{width}.json", "w"), indent=1)
    n += 1
    worst = max(out["check"].values()) if out["check"] else float("nan")
    print(f"s{seed} w{width}: {len(fams)} families; T_train {T_train:.4g}; radiance vs the lane's record, largest "
          f"relative difference {worst:.2e}")
print(f"{n} prediction files scored into {OUT}")
