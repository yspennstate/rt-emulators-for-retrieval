"""Does the network trained on the floored objective fail to fit it, or fit it and fail to generalize?

The pilot (retrieval_weighted_pilot.py) trains the per-component network plainly and on the floored objective. On the
test block the plain network has the lower value of the floored objective itself (p3rw_objective_on_test.py). This script
repeats the pilot's plain and weighted arms exactly (split, seeds, standardization, weights, normalization, loss, early
stopping) and reports the pilot's objective
    J_u = mean of (e_a^2 + R^2 (e_2^2 + e_3^2) + R^4 t^2 e_s^2) / max(t, u T_train)^2
on the training, validation and test blocks for both arms, over all entries and over those below and above the floor.
A weighted arm with the lower training value and the higher test value fits and fails to generalize; one with the higher
training value fails to fit.

usage: EMIT_DATA=<dir> python p3_weighted_diagnostic.py --seed 101 [--u 1e-3 1e-1] [--epochs 150] [--threads 4]
       [--out results] [--tag TAG] [--record FILE]
"""
import argparse
import hashlib
import json
import os
import pathlib
import time

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--u", type=float, nargs="+", default=[1e-3, 1e-1])
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
R = 0.9
WIDTHS = tuple(int(w) for w in args.widths.split(","))
TAG = args.tag or f"p3wd_s{args.seed}"
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
    if u is None:
        return {c: None for c in C}
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
    """The pilot's train(), returning the predictions on all three blocks."""
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
        return {blk: ystd[c].inv(model(x).numpy().astype(np.float64)) for blk, x in (("tr", xtr), ("va", xva), ("te", xte))}, ep + 1


ROWS = {"tr": idx_tr, "va": idx_val, "te": idx_te}


def J(P, blk, u):
    rows = ROWS[blk]
    t = np.clip(T[rows], 0, None)
    tau = u * T_train
    e = {c: P[c][blk] - Ys[c][rows] for c in C}
    L = (e["Y1"] ** 2 + R ** 2 * (e["Y2"] ** 2 + e["Y3"] ** 2) + R ** 4 * t ** 2 * e["Y4"] ** 2) / np.maximum(t, tau) ** 2
    return {"all": float(L.mean()), "below": float(L[t < tau].mean()), "above": float(L[t >= tau].mean())}


arms = [("plain", None)] + [(f"weighted_u{u:g}", u) for u in args.u]
preds, epochs = {}, {}
for name, u in arms:
    wtr, wva = weights(idx_tr, u), weights(idx_val, u)
    if u is not None:   # the pilot's normalization: each component's weights to mean one on the training block
        for c in C:
            m = wtr[c].mean()
            wtr[c], wva[c] = wtr[c] / m, wva[c] / m
    preds[name], epochs[name] = {}, {}
    for c in C:
        preds[name][c], epochs[name][c] = train(c, 1000 * args.seed, wtr[c], wva[c])
    log(f"{name}: epochs {epochs[name]}")
rec = {"tag": TAG, "kind": "p3_weighted_diagnostic", "seed": args.seed, "T_train": T_train, "u": args.u,
       "epochs": epochs, "driver_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), "J": {}}
for u in args.u:
    for name, _ in arms:
        rec["J"][f"{name}@u{u:g}"] = {blk: J(preds[name], blk, u) for blk in ("tr", "va", "te")}
    w = f"weighted_u{u:g}"
    for blk in ("tr", "va", "te"):
        a, b = rec["J"][f"{w}@u{u:g}"][blk], rec["J"][f"plain@u{u:g}"][blk]
        log(f"J_u{u:g} {blk}: weighted/plain  all {a['all'] / b['all']:.3f}  below {a['below'] / b['below']:.3f}  "
            f"above {a['above'] / b['above']:.3f}")
rec["minutes"] = round((time.time() - t0) / 60, 1)
tmp = OUT / (TAG + ".json.tmp")
tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
os.replace(tmp, OUT / (TAG + ".json"))
if args.record:
    pathlib.Path(args.record).write_text(json.dumps(rec, indent=1), encoding="utf-8")
log(f"done {TAG} in {rec['minutes']} min")
