"""Kernel ridge regression fitted on the retrieval-weighted error.

Section 7.3 fits every kernel on the plain squared error and moves only its parameters. Here the fit itself is
weighted. For each component c and band b,

    minimize over f   sum_i w_c(i, b) (f(x_i) - y_c(i, b))^2 + lambda ||f||^2,
    w = 1, R^2, R^2, R^4 t^2  divided by  max(t, u T_train)^2               (floored, the weights of J_u)
    w = 1, R^2, R^2, R^4 t^2  divided by  t^2 where t >= u T_train, 0 below  (cut, the weights of K_u)

with t = Y2 + Y3 the true transmission of the entry: the truncated reweighting of Ma, Pathak and Wainwright, with the
ratio the retrieval supplies. The weights change from band to band, so every band is its own solve,
alpha_b = (K + lambda W_b^{-1})^{-1} y_b over the rows with w > 0; the plain arm (w = 1) is one solve for all bands.
To keep 285 solves per component affordable, every arm is fitted on the same training rows, the tuning subsample of
p3_krr_select.py (6,000 rows), in band space (targets standardized per band, no PCA), with the isotropic Matern kernel
whose smoothness, length scale and nugget the forward criterion chooses on the validation block, as in the second
paper. The weights are scaled to mean one over each band's fitted rows, so that the nugget means the same in every arm
and the plain arm is the weighted solve with w = 1. Only the weights differ between the arms. The normalized weights of
Y1, Y2 and Y3 coincide band by band, so components that also share their kernel parameters share every factorization.

Test scores are those of the pilot, plus the floored and cut objectives of every arm on the validation and test blocks
(sqrt(sum w e^2 / sum w Y^2) per component: J_u and K_u of p3_krr_select.py).

usage: EMIT_DATA=<dir> python p3_krr_weighted.py --seed 101 [--u 1e-1 1e-2 1e-3] [--u-cut 1e-2] [--nfit 6000]
       [--tune-sub 6000] [--threads 4] [--drop-rows 4011 7439] [--out results] [--tag TAG] [--record FILE]
       [--flat 0.5@1e-2 0.25@1e-2] [--flat-only]
--flat adds arms fitted on the flattened floored weights (w_J)^kappa of Shimodaira (2000), the weights of the network
arms flat<kappa>_u<u> of p3_weight_family.py; --flat-only fits the plain arm and those alone.
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
p.add_argument("--u", type=float, nargs="*", default=[1e-1, 1e-2, 1e-3], help="floors of the floored weights")
p.add_argument("--u-cut", type=float, nargs="*", default=[1e-2], help="floors of the cut weights")
p.add_argument("--nfit", type=int, default=6000, help="fitted rows: the first nfit of the tuning subsample")
p.add_argument("--tune-sub", type=int, default=6000)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--pca_rank", type=int, default=64)
p.add_argument("--data", default="")
p.add_argument("--out", default="results")
p.add_argument("--tag", default="")
p.add_argument("--record", default="")
p.add_argument("--drop-rows", type=int, nargs="*", default=[])
p.add_argument("--flat", nargs="*", default=[],
               help="flattened floored weights kappa@u, e.g. 0.5@1e-2: the weights of J_u raised to the power kappa "
                    "(Shimodaira's flattening), renormalized per band like every other arm")
p.add_argument("--flat-only", action="store_true",
               help="fit only the plain and the flattened arms; J_u and K_u are still scored at --u and --u-cut")
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
TAG = args.tag or f"p3kw_s{args.seed}"
t0 = time.time()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def log(msg):
    print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)


# ---- data and the fresh split (p3_krr_select.py, unchanged) ----
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
DROP = np.array(sorted(set(args.drop_rows)), dtype=int)
dropped_from = {int(i): ("test" if i in idx_te else "validation" if i in idx_val else "training") for i in DROP}
if DROP.size:
    idx_te, idx_val, idx_tr = (b[~np.isin(b, DROP)] for b in (idx_te, idx_val, idx_tr))
if not np.isfinite(X).all() or not all(np.isfinite(Y).all() for Y in Ys.values()):
    raise ValueError("Nonfinite inputs or targets require an explicit missing-data policy")
n = len(idx_tr)
log(f"seed {args.seed}: train={n} val={len(idx_val)} test={len(idx_te)} nfit={args.nfit} tune_sub={TUNE_SUB}")


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
ystd, pca, Ztr = {}, {}, {}
for c in COMPONENTS:
    ystd[c] = Standardizer(Ys[c][idx_tr])
    pca[c] = PCAReducer(ystd[c].fwd(Ys[c][idx_tr]), PCA_RANK)
    Ztr[c] = pca[c].fwd(ystd[c].fwd(Ys[c][idx_tr]))
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


# ---- the entry weights of J_u (floored) and K_u (cut) ----
T = Ys["Y2"] + Ys["Y3"]
t_tr = T[idx_tr]
T_train = float(np.median(t_tr[t_tr > 0]))
t_va = np.clip(T[idx_val], 0, None)
t_te = np.clip(T[idx_te], 0, None)


def entry_weights(c, t, kind, u):
    """The weights of J_u (kind 'J', floored at u T_train) or K_u (kind 'K', cut there) for component c at the true
    transmissions t; the same expressions as jweights() in p3_krr_select.py."""
    tau = u * T_train
    num = R ** 4 * t ** 2 if c == "Y4" else (1.0 if c == "Y1" else R ** 2)
    if kind.startswith("F"):   # flattened: the floored weights of J_u to the power kappa, kind 'F<kappa>'
        return (num / np.maximum(t, tau) ** 2) ** float(kind[1:])
    if kind == "J":
        return num / np.maximum(t, tau) ** 2
    keep = t >= tau
    return np.where(keep, num / np.where(keep, t, 1.0) ** 2, 0.0)


def objective(c, Yp, Yt, t, kind, u):
    w = entry_weights(c, t, kind, u)
    e = Yp - Yt
    return float(np.sqrt((w * e * e).sum() / (w * Yt * Yt).sum()))


# ---- kernels (p3_krr_select.py, unchanged) ----

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
    """Inputs standardized on the training rows and the tuning subsample, as in p3_krr_select.py."""

    def __init__(self, Ftr, Fva, Fte, seed):
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
        self.Ftr, self.Fva, self.Fte = (Ftr - mu) / sd, (Fva - mu) / sd, (Fte - mu) / sd
        rng = np.random.RandomState(seed + 777)
        self.sub = rng.permutation(len(self.Ftr))[:min(TUNE_SUB, len(self.Ftr))]


def tune_forward(krr, Ytr_sub_fn, err_fn, nus=(1.5, 2.5), scales=(0.5, 1.0, 2.0, 4.0),
                 nugs=(1e-8, 1e-6, 1e-4, 1e-2)):
    """tune_many() of p3_krr_select.py with the forward criterion alone: the second paper's choice."""
    S_ = krr.Ftr[krr.sub]
    D2s, D2vs = sqd(S_, S_), sqd(krr.Fva, S_)
    med = float(np.sqrt(np.median(D2s[np.triu_indices(len(S_), 1)])))
    Ysub = Ytr_sub_fn(krr.sub)
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


# ---- the forward choice of the kernel parameters, per component ----
krr_x = KRR(Xtr, Xva, Xte, args.seed)
hp = {}
for c in C:
    hp[c] = tune_forward(krr_x, lambda s_, c=c: Ztr[c][s_], lambda Zv, c=c: val_err(c, Zv))
    log(f"{c} forward choice: nu={hp[c]['nu']} scale={hp[c]['scale']} nugget={hp[c]['nugget']:g} "
        f"val={hp[c]['val_sub']:.5f}")

fit = krr_x.sub[:min(args.nfit, len(krr_x.sub))]      # the fitted rows, shared by every arm
m_all = len(fit)
Ffit = krr_x.Ftr[fit]
Yfit = {c: ystd[c].fwd(Ys[c][idx_tr])[fit] for c in C}   # standardized per band on the training rows
t_fit = np.clip(t_tr[fit], 0, None)
kcache = {}


def kernels(h):
    key = (h["nu"], h["scale"], h["med"])
    if key not in kcache:
        ls = h["scale"] * h["med"]
        kcache.clear()   # one kernel set in memory at a time; the groups below are ordered by kernel
        kcache[key] = (matern(sqd(Ffit, Ffit), ls, h["nu"]), matern(sqd(krr_x.Fva, Ffit), ls, h["nu"]),
                       matern(sqd(krr_x.Fte, Ffit), ls, h["nu"]))
    return kcache[key]


def weight_class(c):
    return "albedo" if c == "Y4" else "flux"


def fit_arm(kind, u):
    """Standardized validation and test predictions of every component, fitted with the arm's weights."""
    preds = {}
    groups = {}
    for c in C:
        key = (hp[c]["nu"], hp[c]["scale"], hp[c]["med"], hp[c]["nugget"],
               "plain" if kind is None else weight_class(c))
        groups.setdefault(key, []).append(c)
    for key, comps in sorted(groups.items(), key=lambda kv: kv[0][:3]):
        K, Kva, Kte = kernels(hp[comps[0]])
        nug = hp[comps[0]]["nugget"]
        Yall = [Yfit[c] for c in comps]
        B = Yall[0].shape[1]
        if kind is None:
            A = solve_krr(K, np.hstack(Yall), nug)
            PV, PT = Kva @ A, Kte @ A
            for j, c in enumerate(comps):
                preds[c] = (PV[:, j * B:(j + 1) * B], PT[:, j * B:(j + 1) * B])
            continue
        W = entry_weights(comps[0], t_fit, kind, u)
        PV = [np.zeros((Kva.shape[0], B)) for _ in comps]
        PT = [np.zeros((Kte.shape[0], B)) for _ in comps]
        empty = 0
        for b in range(B):
            keep = W[:, b] > 0
            m = int(keep.sum())
            if m == 0:   # no fitted row carries weight: the prediction is the training mean
                empty += 1
                continue
            wk = W[keep, b] / W[keep, b].mean()
            full = m == m_all
            Kb = K.copy() if full else K[np.ix_(keep, keep)]
            Kb.flat[::m + 1] += nug * m / wk
            cf = cho_factor(Kb, lower=True, check_finite=False, overwrite_a=True)
            A = cho_solve(cf, np.column_stack([Y[keep, b] for Y in Yall]), check_finite=False)
            pv = (Kva if full else Kva[:, keep]) @ A
            pt = (Kte if full else Kte[:, keep]) @ A
            for j in range(len(comps)):
                PV[j][:, b], PT[j][:, b] = pv[:, j], pt[:, j]
        for j, c in enumerate(comps):
            preds[c] = (PV[j], PT[j])
        log(f"  {name(kind, u)} {'+'.join(comps)}: {B} band solves"
            + (f", {empty} bands with no weighted row" if empty else ""))
    return {c: (ystd[c].inv(preds[c][0]), ystd[c].inv(preds[c][1])) for c in C}


def name(kind, u):
    if kind is not None and kind.startswith("F"):
        return f"flat{float(kind[1:]):g}_u{u:g}"
    return "plain" if kind is None else f"{'floored' if kind == 'J' else 'cut'}_u{u:g}"


OBJS = [("J", u) for u in sorted(args.u, reverse=True)] + [("K", u) for u in sorted(args.u_cut, reverse=True)]
FLATS = [("F%g" % float(k), float(v)) for k, v in (s.split("@") for s in args.flat)]
ARMS = [(None, None)] + ([] if args.flat_only else OBJS) + FLATS
rec = {"tag": TAG, "kind": "p3_krr_weighted", "seed": args.seed, "n_train": int(n), "n_fit": int(m_all),
       "n_val": int(len(idx_val)), "n_test": int(len(idx_te)), "T_train": T_train, "tune_sub": TUNE_SUB,
       "pca_rank_tuning": PCA_RANK, "R": R, "rho": RHO, "hp": hp, "dropped_rows": dropped_from,
       "driver_sha256": sha256(pathlib.Path(__file__)),
       "data_sha256": {k: sha256(DATA / (k + ".npy")) for k in ["X"] + C},
       "env": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
               "threads": args.threads, "cpus": os.cpu_count()},
       "arms": {}}
for kind, u in ARMS:
    ta = time.time()
    P = fit_arm(kind, u)
    arm = name(kind, u)
    val_obj = {f"{k}{v:g}": {c: objective(c, P[c][0], Y_true_va[c], t_va, k, v) for c in C} for k, v in OBJS}
    test_obj = {f"{k}{v:g}": {c: objective(c, P[c][1], Y_true_te[c], t_te, k, v) for c in C} for k, v in OBJS}
    rec["arms"][arm] = {"weights": None if kind is None else {"J": "floored", "K": "cut"}.get(kind, "flattened"),
                        "kappa": float(kind[1:]) if kind is not None and kind.startswith("F") else None, "floor": u,
                        "val_forward": {c: rel_l2(Y_true_va[c], P[c][0]) for c in C},
                        "val_obj": val_obj, "test_obj": test_obj, "minutes": round((time.time() - ta) / 60, 2),
                        **score({c: P[c][1] for c in C})}
    m = rec["arms"][arm]
    log(f"== {arm}: rad {m['radiance']:.4f}%  allband {m['allband_p95']:.2f}  p95@1e-12 {m['p95@1e-12']:.2f}  "
        f"p95@1e-3 {m['p95@0.001']:.2f}  p95@1e-2 {m['p95@0.01']:.2f}  ({m['minutes']} min)")

rec["minutes"] = round((time.time() - t0) / 60, 1)
tmp = OUT / (TAG + ".json.tmp")
tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
os.replace(tmp, OUT / (TAG + ".json"))
if args.record:
    pathlib.Path(args.record).write_text(json.dumps(rec, indent=1), encoding="utf-8")
log(f"done {TAG} in {rec['minutes']} min")
