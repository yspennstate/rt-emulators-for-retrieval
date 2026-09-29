"""The network trained mostly on the plain loss, with a small share of the retrieval weight.

The pilot (retrieval_weighted_pilot.py) trains the network on the plain loss and on the floored retrieval weights alone;
both extremes of the family of mixed weights w = (1 - alpha) + alpha wbar_u, where wbar_u is the pilot's floored weight
normalized to mean one on the training block of each component. The stack refit chose alpha near the plain end of the
same family (the frontier), and the notes compute the price of the mixture before training: the effective fraction is
1 / (1 - alpha^2 + alpha^2 / rho_u), with rho_u the effective fraction of wbar_u. This script trains the pilot's network
(same split, seeds, standardization, architecture and schedule; early stopping on the weighted validation loss, as in
the pilot) on the plain loss and on the mixed weights for each alpha in --alpha at each floor in --u, and scores every
arm with the pilot's score() on the test block, and with the pilot's objective J_u on the test block.

usage: EMIT_DATA=<dir> python p3_mixed_objective.py --seed 101 [--alpha 0.05 0.2 0.5] [--u 1e-2] [--epochs 150]
       [--threads 4] [--data DIR] [--out results] [--tag TAG] [--record FILE]
"""
import argparse
import hashlib
import json
import os
import pathlib
import time

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.2, 0.5])
p.add_argument("--u", type=float, nargs="+", default=[1e-2])
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--widths", default="512,512,512")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--data", default="")
p.add_argument("--out", default="results")
p.add_argument("--tag", default="")
p.add_argument("--record", default="")
args = p.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, str(args.threads))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(args.threads)
DATA = pathlib.Path(args.data or os.environ.get("EMIT_DATA", os.path.expanduser("~/p2/data/emit")))
OUT = pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
C = ["Y1", "Y2", "Y3", "Y4"]
RHO, R = 0.7, 0.9
FLOORS = (1e-12, 1e-3, 1e-2)
WIDTHS = tuple(int(w) for w in args.widths.split(","))
TAG = args.tag or f"p3mx_s{args.seed}"
t0 = time.time()


def log(msg):
    print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)


# ---- data, split, standardization and weights: retrieval_weighted_pilot.py, unchanged ----
X = np.load(DATA / "X.npy")
Ys = {c: np.load(DATA / (c + ".npy")) for c in C}
n_all = X.shape[0]
perm = np.random.RandomState(args.seed).permutation(n_all)
n_te = int(round(0.1 * n_all))
idx_te, tr_full = perm[:n_te], perm[n_te:]
vperm = np.random.RandomState(args.seed + 10000).permutation(len(tr_full))
n_val = int(round(0.1 * len(tr_full)))
idx_val, idx_tr = tr_full[vperm[:n_val]], tr_full[vperm[n_val:]]


class Std:
    def __init__(self, A):
        self.mean, self.std = A.mean(0), np.sqrt(A.var(0))
        self.std[self.std == 0] = 1.0

    def fwd(self, A):
        return (A - self.mean) / self.std

    def inv(self, A):
        return A * self.std + self.mean


xs = Std(X[idx_tr])
Xtr, Xva, Xte = (xs.fwd(X[i]) for i in (idx_tr, idx_val, idx_te))
ystd = {c: Std(Ys[c][idx_tr]) for c in C}
T = Ys["Y2"] + Ys["Y3"]
t_tr = T[idx_tr]
T_train = float(np.median(t_tr[t_tr > 0]))


def weights(rows, u):
    tau = u * T_train
    t = T[rows]
    den = np.maximum(t, tau) ** 2
    w = {"Y1": 1.0 / den, "Y2": R ** 2 / den, "Y3": R ** 2 / den,
         "Y4": R ** 4 * np.clip(t, 0, None) ** 2 / den}
    return {c: w[c] * ystd[c].std[None, :] ** 2 for c in C}


def build(d_in, d_out):
    layers, prev = [], d_in
    for wdt in WIDTHS:
        layers += [nn.Linear(prev, wdt), nn.ELU()]
        prev = wdt
    layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers)


def train(c, seed, wtr, wva, patience=25):
    """The pilot's train()."""
    torch.manual_seed(seed)
    ytr = torch.tensor(ystd[c].fwd(Ys[c][idx_tr]), dtype=torch.float32)
    yva = torch.tensor(ystd[c].fwd(Ys[c][idx_val]), dtype=torch.float32)
    xtr, xva, xte = (torch.tensor(a, dtype=torch.float32) for a in (Xtr, Xva, Xte))
    Wtr = None if wtr is None else torch.tensor(wtr, dtype=torch.float32)
    Wva = None if wva is None else torch.tensor(wva, dtype=torch.float32)

    def loss(pred, y, w):
        e2 = (pred - y) ** 2
        return e2.mean() if w is None else (w * e2).mean()

    model = build(xtr.shape[1], ytr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    best, best_val, bad, ep = None, np.inf, 0, 0
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 1024):
            b = order[i:i + 1024]
            opt.zero_grad()
            loss(model(xtr[b]), ytr[b], None if Wtr is None else Wtr[b]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = loss(model(xva), yva, Wva).item()
        if vl < best_val - 1e-9:
            best_val, bad, best = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        return ystd[c].inv(model(xte).numpy().astype(np.float64)), ep + 1


def score(P):
    """The pilot's score()."""
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


def J_test(P, u):
    t = np.clip(T[idx_te], 0, None)
    e = {c: P[c] - Ys[c][idx_te] for c in C}
    L = (e["Y1"] ** 2 + R ** 2 * (e["Y2"] ** 2 + e["Y3"] ** 2) + R ** 4 * t ** 2 * e["Y4"] ** 2) / np.maximum(t, u * T_train) ** 2
    return float(L.mean())


arms = [("plain", None, None)] + [(f"mix{a:g}_u{u:g}", a, u) for u in args.u for a in args.alpha]
rec = {"tag": TAG, "kind": "p3_mixed_objective", "seed": args.seed, "n_train": int(len(idx_tr)),
       "n_val": int(len(idx_val)), "n_test": int(len(idx_te)), "T_train": T_train, "widths": list(WIDTHS),
       "epochs": args.epochs, "members": 1, "alpha": args.alpha, "u": args.u,
       "driver_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), "arms": {}}
for name, a, u in arms:
    if a is None:
        wtr, wva = {c: None for c in C}, {c: None for c in C}
    else:
        wt, wv = weights(idx_tr, u), weights(idx_val, u)
        wtr, wva = {}, {}
        for c in C:
            m = wt[c].mean()          # the pilot's normalization of the weights to mean one on the training block
            wtr[c] = (1 - a) + a * wt[c] / m
            wva[c] = (1 - a) + a * wv[c] / m
    P, eps = {}, {}
    for c in C:
        P[c], eps[c] = train(c, 1000 * args.seed, wtr[c], wva[c])
    rec["arms"][name] = {"alpha": a, "u": u, "epochs_run": {c: [eps[c]] for c in C}, **score(P),
                         "J_test": {f"u{v:g}": J_test(P, v) for v in args.u}}
    m = rec["arms"][name]
    log(f"== {name}: rad {m['radiance']:.4f}%  allband {m['allband_p95']:.2f}  p95@1e-12 {m['p95@1e-12']:.2f}  "
        f"p95@1e-3 {m['p95@0.001']:.2f}  p95@1e-2 {m['p95@0.01']:.2f}  J_test {m['J_test']}")
rec["minutes"] = round((time.time() - t0) / 60, 1)
tmp = OUT / (TAG + ".json.tmp")
tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
os.replace(tmp, OUT / (TAG + ".json"))
if args.record:
    pathlib.Path(args.record).write_text(json.dumps(rec, indent=1), encoding="utf-8")
log(f"done {TAG} in {rec['minutes']} min")
