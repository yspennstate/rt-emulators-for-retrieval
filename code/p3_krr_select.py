"""Kernel hyperparameters chosen for the retrieval.

The second paper tunes each kernel ridge regression on the relative error of its component over the validation block
(emit_campaign.py at commit 1f5d4df; its split, PCA and kernel code is copied here unchanged, training policy raw).
This script runs the same tuning grid, and the same coordinate search over the input length scales, on two kinds of
criterion: the forward one, which reproduces the second paper's choice, and the retrieval-weighted squared error of
the pilot (retrieval_weighted_pilot.py) on the same validation block,

    J_u(c) = [ sum w_c (Yhat_c - Y_c)^2 / sum w_c Y_c^2 ]^(1/2),
    w = 1, R^2, R^2, R^4 t^2   divided by  max(t, u T_train)^2     for c = Y1, Y2, Y3, Y4,

over every validation entry, in physical units, with t = Y2 + Y3 the true transmission of the entry and T_train the
median positive training transmission; and the same with the weights cut at the floor instead of floored,

    K_u(c):  w = 1, R^2, R^2, R^4 t^2  divided by  t^2,  on the entries with t >= u T_train, zero below,

the first-order bound of the mean squared retrieval error on the physical domain above u T_train. The floored weights
put most of their mass on entries below the floor (a fraction (2 - beta)/2 as the floor goes to zero), so J_u tunes for
entries the physical domain excludes; K_u does not. (The ratio and the square root leave the choice among candidates
unchanged; they put both criteria on the scale of the relative error, so that the coordinate search's fixed
improvement threshold means the same for every criterion.) The fit is the plain squared error in PCA space on every
training row in every arm; only the choice among the candidates sees the criterion.

Families: krr (isotropic Matern; the grid is scored on every criterion in one pass) and ard (per-input length
scales; one coordinate search per criterion). Test scores are those of the pilot: mean relative radiance error at
rho = 0.7, the all-band 95th percentile of the retrieval error, and its 95th percentiles on the physical domain above
1e-12, 1e-3 and 1e-2 T_train.

usage: EMIT_DATA=<dir> python p3_krr_select.py --seed 101 [--u 1e-1 1e-2 1e-3] [--u-cut 1e-1 1e-2 1e-3]
       [--u-ard 1e-2] [--u-cut-ard 1e-2 1e-3] [--threads 4] [--families krr,ard] [--out results] [--tag TAG]
       [--record FILE] [--ntrain N] [--tune-sub 6000]
       [--check-tune Y1]   (tune the isotropic kernel of one component on every criterion, print, and stop)
"""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import time

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--u", type=float, nargs="*", default=[1e-1, 1e-2, 1e-3], help="floors of J_u (floored) for krr")
p.add_argument("--u-cut", type=float, nargs="*", default=[1e-1, 1e-2, 1e-3], help="floors of K_u (cut) for krr")
p.add_argument("--u-ard", type=float, nargs="*", default=[1e-2], help="floors of J_u (floored) for ard")
p.add_argument("--u-cut-ard", type=float, nargs="*", default=[1e-2, 1e-3], help="floors of K_u (cut) for ard")
p.add_argument("--families", default="krr,ard")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--pca_rank", type=int, default=64)
p.add_argument("--tune-sub", type=int, default=6000)
p.add_argument("--ntrain", type=int, default=0)
p.add_argument("--data", default="")
p.add_argument("--out", default="results")
p.add_argument("--tag", default="")
p.add_argument("--record", default="")
p.add_argument("--check-tune", default="")
p.add_argument("--no-preds", action="store_true")
p.add_argument("--drop-rows", type=int, nargs="*", default=[])
args = p.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, str(args.threads))
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402

DATA = pathlib.Path(args.data or os.environ.get("EMIT_DATA", os.path.expanduser("~/p2/data/emit")))
OUT = pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
COMPONENTS = C = ["Y1", "Y2", "Y3", "Y4"]
RHO, R = 0.7, 0.9
FLOORS = (1e-12, 1e-3, 1e-2)
PCA_RANK = args.pca_rank
TUNE_SUB = args.tune_sub
FAMS = set(args.families.split(","))
TAG = args.tag or f"p3ks_s{args.seed}"
t0 = time.time()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def log(msg):
    print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)


# ---- data and the fresh split (emit_campaign.py at 1f5d4df, unchanged; policy raw keeps the training rows) ----
X = np.load(DATA / "X.npy")
Ys = {c: np.load(DATA / (c + ".npy")) for c in COMPONENTS}
n_all = X.shape[0]
rng_split = np.random.RandomState(args.seed)
perm = rng_split.permutation(n_all)
n_te = int(round(0.1 * n_all))
idx_te = perm[:n_te]
tr_full = perm[n_te:]
rng_val = np.random.RandomState(args.seed + 10000)
vperm = rng_val.permutation(len(tr_full))
n_val = int(round(0.1 * len(tr_full)))
idx_val = tr_full[vperm[:n_val]]
idx_tr = tr_full[vperm[n_val:]]
# rows left out of every block after the split (the EMIT table's failed evaluations are 4011 and 7439)
DROP = np.array(sorted(set(args.drop_rows)), dtype=int)
dropped_from = {int(i): ("test" if i in idx_te else "validation" if i in idx_val else "training") for i in DROP}
if DROP.size:
    idx_te, idx_val, idx_tr = (b[~np.isin(b, DROP)] for b in (idx_te, idx_val, idx_tr))
if args.ntrain and args.ntrain < len(idx_tr):
    idx_tr = idx_tr[:args.ntrain]
if not np.isfinite(X).all() or not all(np.isfinite(Y).all() for Y in Ys.values()):
    raise ValueError("Nonfinite inputs or targets require an explicit missing-data policy")
n = len(idx_tr)
log(f"seed {args.seed}: train={n} val={len(idx_val)} test={len(idx_te)} rank={PCA_RANK} tune_sub={TUNE_SUB}")


class Standardizer:
    def __init__(self, A):
        self.mean = A.mean(axis=0)
        self.std = np.sqrt(A.var(axis=0))
        self.std[self.std == 0] = 1.0

    def fwd(self, A):
        return (A - self.mean) / self.std

    def inv(self, A):
        return A * self.std + self.mean


class PCAReducer:
    def __init__(self, Y_tr_std, rank):
        U, S, Vt = np.linalg.svd(Y_tr_std - Y_tr_std.mean(axis=0), full_matrices=False)
        self.center = Y_tr_std.mean(axis=0)
        self.Vt = Vt[:rank]
        self.evr = float((S[:rank] ** 2).sum() / (S ** 2).sum())

    def fwd(self, Y_std):
        return (Y_std - self.center) @ self.Vt.T

    def inv(self, Z):
        return Z @ self.Vt + self.center


xstd = Standardizer(X[idx_tr])
Xs = xstd.fwd(X)
Xtr, Xva, Xte = Xs[idx_tr], Xs[idx_val], Xs[idx_te]
ystd, pca, Ztr, Zva = {}, {}, {}, {}
for c in COMPONENTS:
    ystd[c] = Standardizer(Ys[c][idx_tr])
    pca[c] = PCAReducer(ystd[c].fwd(Ys[c][idx_tr]), PCA_RANK)
    Ztr[c] = pca[c].fwd(ystd[c].fwd(Ys[c][idx_tr]))
    Zva[c] = pca[c].fwd(ystd[c].fwd(Ys[c][idx_val]))
Y_true_va = {c: Ys[c][idx_val] for c in COMPONENTS}
Y_true_te = {c: Ys[c][idx_te] for c in COMPONENTS}


def phys(c, Z):
    return ystd[c].inv(pca[c].inv(Z))


def rel_l2(Y_true, Y_pred):
    num = np.linalg.norm(Y_true - Y_pred, axis=1)
    den = np.linalg.norm(Y_true, axis=1)
    return float(np.mean(num / den))


def val_err(c, Z):
    return rel_l2(Y_true_va[c], phys(c, Z))


# ---- the retrieval-weighted criterion on the validation block ----
T = Ys["Y2"] + Ys["Y3"]
t_tr = T[idx_tr]
T_train = float(np.median(t_tr[t_tr > 0]))
t_va = np.clip(T[idx_val], 0, None)


def jweights(kind, u):
    """kind 'J': weights floored at u T_train; kind 'K': weights cut at u T_train (zero below)."""
    tau = u * T_train
    den = np.maximum(t_va, tau) ** 2
    keep = np.ones_like(t_va) if kind == "J" else (t_va >= tau).astype(float)
    return {"Y1": keep / den, "Y2": R ** 2 * keep / den, "Y3": R ** 2 * keep / den,
            "Y4": R ** 4 * t_va ** 2 * keep / den}


CRITS = sorted({("J", u) for u in args.u + args.u_ard} | {("K", u) for u in args.u_cut + args.u_cut_ard})
JW = {k: jweights(*k) for k in CRITS}
JDEN = {(k, c): float((JW[k][c] * Y_true_va[c] ** 2).sum()) for k in JW for c in C}


def j_err(c, Z, k):
    e = phys(c, Z) - Y_true_va[c]
    return float(np.sqrt((JW[k][c] * e * e).sum() / JDEN[(k, c)]))


def crit_name(k):
    return "forward" if k is None else f"{k[0]}{k[1]:g}"


def crit_fn(c, k):
    if k is None:
        return lambda Zv: val_err(c, Zv)
    return lambda Zv: j_err(c, Zv, k)


def kind_of(k):
    return None if k is None else {"J": "floored", "K": "cut"}[k[0]]


# ---- kernels (emit_campaign.py at 1f5d4df, unchanged) ----

def sqd(A, B):
    D2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(D2, 0.0, out=D2)
    return D2


def matern(D2, ls, nu):
    r = np.sqrt(D2) / ls
    if nu == 1.5:
        a = np.sqrt(3.0) * r
        return (1.0 + a) * np.exp(-a)
    a = np.sqrt(5.0) * r
    return (1.0 + a + a * a / 3.0) * np.exp(-a)


def solve_krr(K, Y, nug):
    Kr = K.copy(); Kr.flat[::len(K) + 1] += nug * len(K)
    c = cho_factor(Kr, lower=True, check_finite=False, overwrite_a=True)
    return cho_solve(c, Y, check_finite=False)


class KRR:
    """Exact Matern KRR on inputs F (rows), targets in PCA space, tuned on validation by err_fn
    over a subsample, refit on every row. `w` is a diagonal metric applied after standardization."""

    def __init__(self, Ftr, Fva, Fte, seed):
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
        self.Ftr, self.Fva, self.Fte = (Ftr - mu) / sd, (Fva - mu) / sd, (Fte - mu) / sd
        rng = np.random.RandomState(seed + 777)
        self.sub = rng.permutation(len(self.Ftr))[:min(TUNE_SUB, len(self.Ftr))]

    def tune(self, Ytr_sub_fn, err_fn, w=None, nus=(1.5, 2.5), scales=(0.5, 1.0, 2.0, 4.0),
             nugs=(1e-8, 1e-6, 1e-4, 1e-2)):
        Ftr, Fva = self.Ftr, self.Fva
        if w is not None:
            Ftr, Fva = Ftr * w, Fva * w
        S_ = Ftr[self.sub]
        D2s, D2vs = sqd(S_, S_), sqd(Fva, S_)
        med = float(np.sqrt(np.median(D2s[np.triu_indices(len(S_), 1)])))
        Ysub = Ytr_sub_fn(self.sub)
        best = (np.inf, None)
        for nu in nus:
            for sc in scales:
                Ks, Kvs = matern(D2s, sc * med, nu), matern(D2vs, sc * med, nu)
                for nug in nugs:
                    try:
                        alpha = solve_krr(Ks, Ysub, nug)
                    except np.linalg.LinAlgError:
                        continue
                    e = err_fn(Kvs @ alpha)
                    if e < best[0]:
                        best = (e, dict(nu=nu, scale=sc, nugget=nug, med=med, val_sub=e))
        return best[1]

    def fit_predict(self, Ytr, hp, w=None):
        Ftr, Fva, Fte = self.Ftr, self.Fva, self.Fte
        if w is not None:
            Ftr, Fva, Fte = Ftr * w, Fva * w, Fte * w
        ls = hp["scale"] * hp["med"]
        K = matern(sqd(Ftr, Ftr), ls, hp["nu"])
        alpha = solve_krr(K, Ytr, hp["nugget"])
        del K
        outs = []
        for F_ in (Fva, Fte):   # the second paper also predicts the training rows; nothing here reads them
            pred = np.empty((len(F_), Ytr.shape[1]))
            for k in range(0, len(F_), 4000):
                pred[k:k + 4000] = matern(sqd(F_[k:k + 4000], Ftr), ls, hp["nu"]) @ alpha
            outs.append(pred)
        return outs


def ard_search(krr, Ytr_sub_fn, err_fn, hp0, d):
    """Coordinate search over per-input multipliers, two sweeps, on the tuning subsample."""
    w = np.ones(d)
    base = hp0["val_sub"]
    hp = dict(hp0)
    for sweep in range(2):
        for j in range(d):
            best_m, best_e = 1.0, base
            for m in (0.25, 0.5, 2.0, 4.0):
                wt = w.copy(); wt[j] *= m
                cand = krr.tune(Ytr_sub_fn, err_fn, w=wt, nus=(hp["nu"],), scales=(hp["scale"],))
                if cand is not None and cand["val_sub"] < best_e - 1e-7:
                    best_e, best_m = cand["val_sub"], m
            w[j] *= best_m
            base = best_e
    hp = krr.tune(Ytr_sub_fn, err_fn, w=w)   # re-tune scale, nugget and nu at the found metric
    return w, hp


def tune_many(krr, Ytr_sub_fn, err_fns, nus=(1.5, 2.5), scales=(0.5, 1.0, 2.0, 4.0), nugs=(1e-8, 1e-6, 1e-4, 1e-2)):
    """KRR.tune for several criteria at once: the same loop order and the same strict improvement rule per criterion,
    so each result is what KRR.tune returns for that criterion alone; each candidate is fitted once."""
    S_ = krr.Ftr[krr.sub]
    D2s, D2vs = sqd(S_, S_), sqd(krr.Fva, S_)
    med = float(np.sqrt(np.median(D2s[np.triu_indices(len(S_), 1)])))
    Ysub = Ytr_sub_fn(krr.sub)
    best = {k: (np.inf, None) for k in err_fns}
    grid = []
    for nu in nus:
        for sc in scales:
            Ks, Kvs = matern(D2s, sc * med, nu), matern(D2vs, sc * med, nu)
            for nug in nugs:
                try:
                    alpha = solve_krr(Ks, Ysub, nug)
                except np.linalg.LinAlgError:
                    continue
                Zv = Kvs @ alpha
                row = dict(nu=nu, scale=sc, nugget=nug)
                for k, fn in err_fns.items():
                    e = fn(Zv)
                    row[k] = e
                    if e < best[k][0]:
                        best[k] = (e, dict(nu=nu, scale=sc, nugget=nug, med=med, val_sub=e))
                grid.append(row)
    return {k: v[1] for k, v in best.items()}, grid


def score(P):
    """Forward and retrieval metrics of predicted test components P (physical units); the pilot's score()."""
    Tt = {c: Ys[c][idx_te] for c in C}
    a, t, s = Tt["Y1"], Tt["Y2"] + Tt["Y3"], Tt["Y4"]
    ah, th, sh = P["Y1"], P["Y2"] + P["Y3"], P["Y4"]
    L = a + RHO * t / (1 - RHO * s)
    Lh = ah + RHO * th / (1 - RHO * sh)
    rel = lambda A, B: float(np.mean(np.linalg.norm(A - B, axis=1) / np.linalg.norm(A, axis=1)))  # noqa: E731
    out = {"radiance": 100 * rel(L, Lh), "components": {c: 100 * rel(Tt[c], P[c]) for c in C}}
    u_ = L - ah
    den = th + sh * u_
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = u_ / den
    err = np.abs(np.where(np.isfinite(rho), rho, 0.0) - RHO)
    out["allband_p95"] = 100 * float(np.quantile(err, 0.95))
    ok = np.isfinite(rho) & (den > 0)
    for f in FLOORS:
        dom = (t >= f * T_train) & (s >= 0) & (s < 1) & (1 - RHO * s >= 0.3)
        good = dom & ok
        out[f"p95@{f:g}"] = 100 * float(np.quantile(np.abs(rho[good] - RHO), 0.95))
        out[f"failed_pct@{f:g}"] = 100 * float((dom & ~ok).sum() / max(dom.sum(), 1))
        out[f"coverage@{f:g}"] = 100 * float(dom.mean())
    return out


krr_x = KRR(Xtr, Xva, Xte, args.seed)
UK = ([None] + [("J", u) for u in sorted(args.u, reverse=True)]          # forward first, then floored, then cut
      + [("K", u) for u in sorted(args.u_cut, reverse=True)])
UA = ([None] + [("J", u) for u in sorted(args.u_ard, reverse=True)]
      + [("K", u) for u in sorted(args.u_cut_ard, reverse=True)])

if args.check_tune:
    c = args.check_tune
    hps, grid = tune_many(krr_x, lambda s_: Ztr[c][s_], {crit_name(u): crit_fn(c, u) for u in UK})
    for k, hp in hps.items():
        log(f"{c} {k}: {hp}")
    raise SystemExit(0)

rec = {"tag": TAG, "kind": "p3_krr_select", "seed": args.seed, "n_train": int(n), "n_val": int(len(idx_val)),
       "n_test": int(len(idx_te)), "T_train": T_train, "pca_rank": PCA_RANK, "tune_sub": TUNE_SUB,
       "pca_evr": {c: pca[c].evr for c in C}, "criteria_krr": [crit_name(k) for k in UK],
       "criteria_ard": [crit_name(k) for k in UA], "R": R, "rho": RHO,
       "dropped_rows": dropped_from, "driver_sha256": sha256(pathlib.Path(__file__)),
       "data_sha256": {k: sha256(DATA / (k + ".npy")) for k in ["X"] + C},
       "env": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
               "threads": args.threads, "cpus": os.cpu_count()},
       "arms": {}, "grids": {}}
fits = {}      # (family, component, hp key) -> (val prediction, test prediction) in PCA space


def fitted(fam, c, hp, w=None):
    key = (fam, c, hp["nu"], hp["scale"], hp["nugget"], None if w is None else tuple(np.round(w, 6)))
    if key not in fits:
        pva, pte = krr_x.fit_predict(Ztr[c], hp, w=w)
        fits[key] = (pva, pte)
        log(f"  fit {fam} {c} nu={hp['nu']} scale={hp['scale']} nugget={hp['nugget']:g}"
            + ("" if w is None else f" w={[round(float(x), 4) for x in w]}"))
    return fits[key]


def val_all(c, pva):
    return {"forward": val_err(c, pva), **{crit_name(k): j_err(c, pva, k) for k in JW}}


arm_preds = {}
if "krr" in FAMS:
    hp_all = {crit_name(u): {} for u in UK}
    for c in C:
        hps, grid = tune_many(krr_x, lambda s_: Ztr[c][s_], {crit_name(u): crit_fn(c, u) for u in UK})
        rec["grids"][f"krr_{c}"] = grid
        for k, hp in hps.items():
            hp_all[k][c] = hp
        log(f"krr {c} tuned: " + "; ".join(f"{k} nu={h['nu']} sc={h['scale']} nug={h['nugget']:g}"
                                            for k, h in hps.items()))
    for u in UK:
        k = crit_name(u)
        P, val = {}, {}
        for c in C:
            pva, pte = fitted("krr", c, hp_all[k][c])
            P[c], val[c] = phys(c, pte), val_all(c, pva)
        arm = f"krr_{k}"
        rec["arms"][arm] = {"family": "krr", "criterion": k, "weights": kind_of(u), "floor": None if u is None else u[1],
                            "hp": hp_all[k], "val": val, **score(P)}
        arm_preds[arm] = P
        m = rec["arms"][arm]
        log(f"== {arm}: rad {m['radiance']:.4f}%  allband {m['allband_p95']:.2f}  p95@1e-12 {m['p95@1e-12']:.2f}  "
            f"p95@1e-3 {m['p95@0.001']:.2f}  p95@1e-2 {m['p95@0.01']:.2f}")

if "ard" in FAMS:
    d = Xtr.shape[1]
    hp_all, w_all = {crit_name(u): {} for u in UA}, {crit_name(u): {} for u in UA}
    for c in C:
        hp0s, _ = tune_many(krr_x, lambda s_: Ztr[c][s_], {crit_name(u): crit_fn(c, u) for u in UA})
        for u in UA:
            k = crit_name(u)
            w_ard, hp = ard_search(krr_x, lambda s_: Ztr[c][s_], crit_fn(c, u), hp0s[k], d)
            hp = dict(hp); hp["w"] = [round(float(x), 4) for x in w_ard]
            hp_all[k][c], w_all[k][c] = hp, w_ard
            log(f"ard {c} {k}: w={hp['w']} nu={hp['nu']} sc={hp['scale']} nug={hp['nugget']:g}")
    for u in UA:
        k = crit_name(u)
        P, val = {}, {}
        for c in C:
            pva, pte = fitted("ard", c, hp_all[k][c], w=w_all[k][c])
            P[c], val[c] = phys(c, pte), val_all(c, pva)
        arm = f"ard_{k}"
        rec["arms"][arm] = {"family": "ard", "criterion": k, "weights": kind_of(u), "floor": None if u is None else u[1],
                            "hp": hp_all[k], "val": val, **score(P)}
        arm_preds[arm] = P
        m = rec["arms"][arm]
        log(f"== {arm}: rad {m['radiance']:.4f}%  allband {m['allband_p95']:.2f}  p95@1e-12 {m['p95@1e-12']:.2f}  "
            f"p95@1e-3 {m['p95@0.001']:.2f}  p95@1e-2 {m['p95@0.01']:.2f}")

rec["minutes"] = round((time.time() - t0) / 60, 1)
tmp = OUT / (TAG + ".json.tmp")
tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
os.replace(tmp, OUT / (TAG + ".json"))
if args.record:
    pathlib.Path(args.record).write_text(json.dumps(rec, indent=1), encoding="utf-8")
if not args.no_preds:
    np.savez_compressed(OUT / (TAG + "_preds.npz"), idx_te=idx_te, idx_tr=idx_tr, idx_val=idx_val,
                        **{f"{arm}_{c}": P[c].astype(np.float64) for arm, P in arm_preds.items() for c in C})
log(f"done {TAG} in {rec['minutes']} min")
