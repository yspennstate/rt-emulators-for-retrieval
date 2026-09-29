"""The network of the second paper trained on the plain objective, stopped on the retrieval criterion.

The pilot (retrieval_weighted_pilot.py) trains the per-component network on the plain squared error and keeps the epoch
with the lowest plain validation loss (patience 25). This script runs the same training, with the same split, seeds,
standardization, architecture and schedule, and keeps in addition the epoch that is best on a retrieval criterion of
the validation block:

    J_u(c)  floored   w = 1, R^2, R^2, R^4 t^2  over  max(t, u T_train)^2
    K_u(c)  cut       the same with t^2 in the denominator, on the entries with t >= u T_train, zero below

as the weighted relative RMS [sum w (Yhat - Y)^2 / sum w Y^2]^(1/2) in physical units (p3_krr_select.py). Every arm is
read from the one training trajectory of each component: the plain arm freezes its state exactly when the pilot's
patience rule stops it, so it is the pilot's plain arm; each retrieval criterion keeps its own best state and its own
patience, and training continues while any criterion is still improving (at most --epochs epochs). Choosing the epoch
is a choice among at most --epochs candidates on the retrieval criterion; nothing in the fit sees the weights.

Scores are the pilot's: radiance error at rho = 0.7, all-band 95th percentile of the retrieval error, and its 95th
percentiles on the physical domain above 1e-12, 1e-3 and 1e-2 T_train.

usage: EMIT_DATA=<dir> python p3_retrieval_stopping.py --seed 101 [--floored 1e-2] [--cut 1e-1 1e-2 1e-3]
       [--epochs 150] [--widths 512,512,512] [--threads 4] [--out results] [--tag TAG] [--record FILE] [--ntrain N]
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
p.add_argument("--floored", type=float, nargs="*", default=[1e-2])
p.add_argument("--cut", type=float, nargs="*", default=[1e-1, 1e-2, 1e-3])
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--widths", default="512,512,512")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--ntrain", type=int, default=0)
p.add_argument("--data", default="")
p.add_argument("--out", default="results")
p.add_argument("--tag", default="")
p.add_argument("--record", default="")
p.add_argument("--drop-rows", type=int, nargs="*", default=[])
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
TAG = args.tag or f"p3es_s{args.seed}"
t0 = time.time()


def log(msg):
    print(f"[{(time.time() - t0) / 60:6.1f} min] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---- data, split and standardization: retrieval_weighted_pilot.py, unchanged ----
X = np.load(DATA / "X.npy")
Ys = {c: np.load(DATA / (c + ".npy")) for c in C}
n_all = X.shape[0]
perm = np.random.RandomState(args.seed).permutation(n_all)
n_te = int(round(0.1 * n_all))
idx_te, tr_full = perm[:n_te], perm[n_te:]
vperm = np.random.RandomState(args.seed + 10000).permutation(len(tr_full))
n_val = int(round(0.1 * len(tr_full)))
idx_val, idx_tr = tr_full[vperm[:n_val]], tr_full[vperm[n_val:]]
# rows left out of every block after the split (the EMIT table's failed evaluations are 4011 and 7439)
DROP = np.array(sorted(set(args.drop_rows)), dtype=int)
dropped_from = {int(i): ("test" if i in idx_te else "validation" if i in idx_val else "training") for i in DROP}
if DROP.size:
    idx_te, idx_val, idx_tr = (b[~np.isin(b, DROP)] for b in (idx_te, idx_val, idx_tr))
if args.ntrain and args.ntrain < len(idx_tr):
    idx_tr = idx_tr[:args.ntrain]


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
log(f"seed {args.seed}: train={len(idx_tr)} val={len(idx_val)} test={len(idx_te)} T_train={T_train:.4g}")

# ---- the retrieval criteria on the validation block (p3_krr_select.py) ----
t_va = np.clip(T[idx_val], 0, None)
Y_va = {c: Ys[c][idx_val] for c in C}


def jweights(kind, u):
    tau = u * T_train
    den = np.maximum(t_va, tau) ** 2
    keep = np.ones_like(t_va) if kind == "J" else (t_va >= tau).astype(float)
    return {"Y1": keep / den, "Y2": R ** 2 * keep / den, "Y3": R ** 2 * keep / den,
            "Y4": R ** 4 * t_va ** 2 * keep / den}


CRITS = [("J", u) for u in args.floored] + [("K", u) for u in args.cut]
NAME = {k: f"{k[0]}{k[1]:g}" for k in CRITS}
JW = {k: jweights(*k) for k in CRITS}
JDEN = {(k, c): float((JW[k][c] * Y_va[c] ** 2).sum()) for k in CRITS for c in C}


def crit_value(k, c, P_va):
    e = P_va - Y_va[c]
    return float(np.sqrt((JW[k][c] * e * e).sum() / JDEN[(k, c)]))


# ---- the network and its training: retrieval_weighted_pilot.py, plain arm, with the extra trackers ----
def build(d_in, d_out):
    layers, prev = [], d_in
    for wdt in WIDTHS:
        layers += [nn.Linear(prev, wdt), nn.ELU()]
        prev = wdt
    layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers)


def train_tracked(c, seed, patience=25):
    torch.manual_seed(seed)
    ytr = torch.tensor(ystd[c].fwd(Ys[c][idx_tr]), dtype=torch.float32)
    yva = torch.tensor(ystd[c].fwd(Ys[c][idx_val]), dtype=torch.float32)
    xtr, xva, xte = (torch.tensor(a, dtype=torch.float32) for a in (Xtr, Xva, Xte))

    def loss(pred, y):
        return ((pred - y) ** 2).mean()

    model = build(xtr.shape[1], ytr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    # plain tracker: the pilot's rule exactly (absolute improvement 1e-9 on the standardized loss, patience 25)
    plain = {"best": np.inf, "bad": 0, "state": None, "epoch": 0, "done": False}
    trk = {k: {"best": np.inf, "bad": 0, "state": None, "epoch": 0, "done": False} for k in CRITS}
    ep = 0
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 1024):
            b = order[i:i + 1024]
            opt.zero_grad()
            loss(model(xtr[b]), ytr[b]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            zva = model(xva)
            vl = loss(zva, yva).item()
        if not plain["done"]:
            if vl < plain["best"] - 1e-9:
                plain.update(best=vl, bad=0, epoch=ep + 1, state={k: v.clone() for k, v in model.state_dict().items()})
            else:
                plain["bad"] += 1
                if plain["bad"] >= patience:
                    plain["done"] = True
        P_va = ystd[c].inv(zva.numpy().astype(np.float64))
        for k, tr_ in trk.items():
            if tr_["done"]:
                continue
            v = crit_value(k, c, P_va)
            if v < tr_["best"] * (1 - 1e-6):
                tr_.update(best=v, bad=0, epoch=ep + 1, state={a: b.clone() for a, b in model.state_dict().items()})
            else:
                tr_["bad"] += 1
                if tr_["bad"] >= patience:
                    tr_["done"] = True
        if plain["done"] and all(t_["done"] for t_ in trk.values()):
            break
    out = {}
    for name, tr_ in [("plain", plain)] + [(NAME[k], trk[k]) for k in CRITS]:
        model.load_state_dict(tr_["state"])
        model.eval()
        with torch.no_grad():
            pva = ystd[c].inv(model(xva).numpy().astype(np.float64))
            pte = ystd[c].inv(model(xte).numpy().astype(np.float64))
        out[name] = {"test": pte, "epoch": tr_["epoch"], "val": {NAME[k]: crit_value(k, c, pva) for k in CRITS}}
    return out, ep + 1


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


arms = ["plain"] + [NAME[k] for k in CRITS]
per_c, epochs_run = {}, {}
for c in C:
    per_c[c], epochs_run[c] = train_tracked(c, 1000 * args.seed)   # the pilot's plain-arm seed (members = 1)
    log(f"{c}: ran {epochs_run[c]} epochs; chosen " + ", ".join(f"{a} {per_c[c][a]['epoch']}" for a in arms))
rec = {"tag": TAG, "kind": "p3_retrieval_stopping", "seed": args.seed, "n_train": int(len(idx_tr)),
       "n_val": int(len(idx_val)), "n_test": int(len(idx_te)), "T_train": T_train, "widths": list(WIDTHS),
       "epochs": args.epochs, "criteria": [NAME[k] for k in CRITS], "epochs_run": epochs_run,
       "dropped_rows": dropped_from, "driver_sha256": sha256(pathlib.Path(__file__)),
       "data_sha256": {k: sha256(DATA / (k + ".npy")) for k in ["X"] + C},
       "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
               "threads": args.threads},
       "arms": {}}
for a in arms:
    P = {c: per_c[c][a]["test"] for c in C}
    rec["arms"][a] = {"epoch": {c: per_c[c][a]["epoch"] for c in C}, "val": {c: per_c[c][a]["val"] for c in C},
                      **score(P)}
    m = rec["arms"][a]
    log(f"== {a}: rad {m['radiance']:.4f}%  allband {m['allband_p95']:.2f}  p95@1e-12 {m['p95@1e-12']:.2f}  "
        f"p95@1e-3 {m['p95@0.001']:.2f}  p95@1e-2 {m['p95@0.01']:.2f}")
rec["minutes"] = round((time.time() - t0) / 60, 1)
tmp = OUT / (TAG + ".json.tmp")
tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
os.replace(tmp, OUT / (TAG + ".json"))
if args.record:
    pathlib.Path(args.record).write_text(json.dumps(rec, indent=1), encoding="utf-8")
np.savez_compressed(OUT / (TAG + "_preds.npz"), idx_te=idx_te,
                    **{f"{a}_{c}": per_c[c][a]["test"] for a in arms for c in C})
log(f"done {TAG} in {rec['minutes']} min")
