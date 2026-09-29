"""Training the emulator for the retrieval: a network fitted on the transmission-weighted objective J_tau.

The retrieval error at an entry is at most e_R/t with e_R = |e_a| + R|e_t| + R^2 t|e_s| (the transfer theorem of the
radiative-transfer paper), and the mean squared retrieval error is at most R^2 F_t(tau) + 3 J_tau with
    J_tau = E[(e_a^2 + R^2 e_t^2 + R^4 t^2 e_s^2) / max(t, tau)^2].
This script trains the per-component network of the EMIT pipeline on that objective instead of the plain squared error,
and compares it with the plain network fitted in the same run on the same split and seed.

Per component and band the physical squared error is weighted by
    Y1 (path radiance a):      1 / max(t, tau)^2
    Y2, Y3 (the two fluxes):   R^2 / max(t, tau)^2          (the flux error of the retrieval is e2 + e3)
    Y4 (spherical albedo s):   R^4 t^2 / max(t, tau)^2
where t = Y2 + Y3 is the true transmission of the training entry, tau = u T_train and T_train is the median positive
training transmission. The network works in standardized units, so the weight of band b also carries sigma_b^2; the
weights of each component are normalised to mean one. Validation loss for early stopping uses the same weighting.

Split, standardization and scoring follow emit_campaign.py and conditioned_reflectance.py of the paper's repository:
RandomState(seed) permutation with 10 percent test, a 10 percent validation carve with RandomState(seed + 10000);
retrieval at rho = 0.7; physical domain t >= tau0 T_train, 0 <= s < 1, 1 - rho s >= 0.3; an inversion fails when its
predicted denominator is not positive or its value is not finite.

usage: EMIT_DATA=<dir with X.npy, Y1-Y4.npy> python retrieval_weighted_pilot.py --seed 101 --u 1e-3 [--members 1]
       [--epochs 150] [--widths 512,512,512] [--threads 4] [--out results] [--ntrain 0]
"""
import argparse
import hashlib
import json
import os
import pathlib
import time

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--u", type=float, nargs="+", default=[1e-3])
p.add_argument("--members", type=int, default=1)
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--widths", default="512,512,512")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--ntrain", type=int, default=0)
p.add_argument("--out", default="results")
p.add_argument("--tag", default="")
p.add_argument("--record", default="", help="also write the record here (a runner's completion file)")
p.add_argument("--joint", type=float, nargs="*", default=[],
               help="u values for the joint arm: one network for all four components, trained on the linearized "
                    "retrieval error at rho in {0.1, 0.4, 0.7} with t floored at u T_train")
p.add_argument("--skip-separate", action="store_true", help="run only the plain and joint arms")
p.add_argument("--data", default="", help="array directory (X.npy, Y1-Y4.npy); overrides EMIT_DATA")
p.add_argument("--drop-rows", type=int, nargs="*", default=[],
               help="rows left out of every block after the split (the EMIT table's failed evaluations are 4011 "
                    "and 7439), so that the splits are otherwise unchanged")
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
t0 = time.time()

X = np.load(DATA / "X.npy")
Ys = {c: np.load(DATA / (c + ".npy")) for c in C}
n_all = X.shape[0]
perm = np.random.RandomState(args.seed).permutation(n_all)
n_te = int(round(0.1 * n_all))
idx_te, tr_full = perm[:n_te], perm[n_te:]
vperm = np.random.RandomState(args.seed + 10000).permutation(len(tr_full))
n_val = int(round(0.1 * len(tr_full)))
idx_val, idx_tr = tr_full[vperm[:n_val]], tr_full[vperm[n_val:]]
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


def weights(rows, u):
    """Per-entry weights of the four components on the given rows, in standardized units (not yet normalised)."""
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


NB = Ys["Y1"].shape[1]
JOINT_RHOS = (0.1, 0.4, 0.7)
JOINT_FWD = 1e-2   # weight of the standardized squared error, which keeps the split of t into Y2 and Y3 identified


def train_joint(u, seed, patience=25):
    """One network for all four components, fitted on the linearized retrieval error.

    At a band with true (a, t, s), predicted (a^, t^, s^) and q = 1 - rho s, the first-order retrieval error is
    -(q/t)(q e_a + rho e_t) - rho^2 e_s, which is minus the forward error at rho divided by the slope t/q^2 of the
    radiance in the reflectance. The loss averages its square over rho in JOINT_RHOS and over the entries with
    q >= 0.3, with t floored at tau = u T_train, plus JOINT_FWD times the standardized squared error of all outputs."""
    torch.manual_seed(seed)
    tau = u * T_train
    Z = {k: np.concatenate([ystd[c].fwd(Ys[c][rows]) for c in C], 1) for k, rows in (("tr", idx_tr), ("va", idx_val))}
    sig = torch.tensor(np.concatenate([ystd[c].std for c in C]), dtype=torch.float32)
    mu = torch.tensor(np.concatenate([ystd[c].mean for c in C]), dtype=torch.float32)
    truth = {k: tuple(torch.tensor(A, dtype=torch.float32) for A in
                      (Ys["Y1"][rows], Ys["Y2"][rows] + Ys["Y3"][rows], Ys["Y4"][rows]))
             for k, rows in (("tr", idx_tr), ("va", idx_val))}
    ztr, zva = (torch.tensor(Z[k], dtype=torch.float32) for k in ("tr", "va"))
    xtr, xva, xte = (torch.tensor(a_, dtype=torch.float32) for a_ in (Xtr, Xva, Xte))

    def loss(zh, z, tr_):
        P = zh * sig + mu
        ah, th, sh = P[:, :NB], P[:, NB:2 * NB] + P[:, 2 * NB:3 * NB], P[:, 3 * NB:]
        a, t, s = tr_
        tt = torch.clamp(t, min=tau)
        tot, cnt = 0.0, 0.0
        for rho in JOINT_RHOS:
            q = 1 - rho * s
            m = (q >= 0.3).float()
            r = (q / tt) * (q * (ah - a) + rho * (th - t)) + rho ** 2 * (sh - s)
            tot = tot + (m * r ** 2).sum()
            cnt = cnt + m.sum()
        return tot / cnt + JOINT_FWD * ((zh - z) ** 2).mean()

    model = build(xtr.shape[1], 4 * NB)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    best, best_val, bad, ep = None, np.inf, 0, 0
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 1024):
            b = order[i:i + 1024]
            opt.zero_grad()
            loss(model(xtr[b]), ztr[b], tuple(T_[b] for T_ in truth["tr"])).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = loss(model(xva), zva, truth["va"]).item()
        if vl < best_val - 1e-12:
            best_val, bad, best = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        Zte = model(xte).numpy().astype(np.float64)
    return {c: ystd[c].inv(Zte[:, j * NB:(j + 1) * NB]) for j, c in enumerate(C)}, ep + 1


def score(P):
    """Forward and retrieval metrics of predicted test components P (physical units)."""
    Tt = {c: Ys[c][idx_te] for c in C}
    a, t, s = Tt["Y1"], Tt["Y2"] + Tt["Y3"], Tt["Y4"]
    ah, th, sh = P["Y1"], P["Y2"] + P["Y3"], P["Y4"]
    L = a + RHO * t / (1 - RHO * s)
    Lh = ah + RHO * th / (1 - RHO * sh)
    rel = lambda A, B: float(np.mean(np.linalg.norm(A - B, axis=1) / np.linalg.norm(A, axis=1)))  # noqa: E731
    out = {"radiance": 100 * rel(L, Lh), "components": {c: 100 * rel(Tt[c], P[c]) for c in C}}
    u = L - ah
    den = th + sh * u
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = u / den
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


arms = [("plain", None)] + ([] if args.skip_separate else [(f"weighted_u{u:g}", u) for u in args.u])
rec = {"seed": args.seed, "n_train": int(len(idx_tr)), "n_val": int(len(idx_val)), "n_test": int(len(idx_te)),
       "T_train": T_train, "widths": list(WIDTHS), "epochs": args.epochs, "members": args.members, "arms": {},
       "joint": {"rhos": list(JOINT_RHOS), "forward_weight": JOINT_FWD}, "dropped_rows": dropped_from,
       "driver_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
for u in args.joint:
    name = f"joint_u{u:g}"
    preds, eps = [], []
    for k in range(args.members):
        pr, ep = train_joint(u, 1000 * args.seed + 500 + k)
        preds.append(pr)
        eps.append(ep)
    P = {c: np.mean([pr[c] for pr in preds], axis=0) for c in C}
    print(f"{name} done [{(time.time() - t0) / 60:.1f} min]", flush=True)
    rec["arms"][name] = {"u": u, "epochs_run": eps, **score(P)}
    np.savez_compressed(OUT / f"pred_s{args.seed}_{name}{args.tag}.npz", idx_te=idx_te, **{c: P[c] for c in C})
    m = rec["arms"][name]
    print(f"== {name}: rad {m['radiance']:.4f}%  allband p95 {m['allband_p95']:.2f}  "
          f"p95@1e-12 {m['p95@1e-12']:.2f} (failed {m['failed_pct@1e-12']:.2f}%)  "
          f"p95@1e-3 {m['p95@0.001']:.2f}  p95@1e-2 {m['p95@0.01']:.2f}", flush=True)
for name, u in arms:
    wtr, wva = weights(idx_tr, u), weights(idx_val, u)
    if u is not None:   # normalise each component's weights to mean one on the training block
        for c in C:
            m = wtr[c].mean()
            wtr[c], wva[c] = wtr[c] / m, wva[c] / m
    P, eps = {}, {}
    for c in C:
        preds = []
        for k in range(args.members):
            pr, ep = train(c, 1000 * args.seed + k, wtr[c], wva[c])
            preds.append(pr)
            eps.setdefault(c, []).append(ep)
        P[c] = np.mean(preds, axis=0)
        print(f"{name} {c} done [{(time.time() - t0) / 60:.1f} min]", flush=True)
    rec["arms"][name] = {"u": u, "epochs_run": eps, **score(P)}
    np.savez_compressed(OUT / f"pred_s{args.seed}_{name}{args.tag}.npz", idx_te=idx_te, **{c: P[c] for c in C})
    m = rec["arms"][name]
    print(f"== {name}: rad {m['radiance']:.4f}%  allband p95 {m['allband_p95']:.2f}  "
          f"p95@1e-12 {m['p95@1e-12']:.2f} (failed {m['failed_pct@1e-12']:.2f}%)  "
          f"p95@1e-3 {m['p95@0.001']:.2f}  p95@1e-2 {m['p95@0.01']:.2f}", flush=True)
rec["minutes"] = round((time.time() - t0) / 60, 1)
(OUT / f"pilot_s{args.seed}{args.tag}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
if args.record:
    pathlib.Path(args.record).write_text(json.dumps(rec, indent=1), encoding="utf-8")
print("done", rec["minutes"], "min")
