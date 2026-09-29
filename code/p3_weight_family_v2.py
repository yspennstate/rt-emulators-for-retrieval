"""The network on the floored or the cut retrieval weight, pure, flattened or mixed, with either stopping rule.

Version 2 of p3_weight_family.py: every arm is also scored on the validation block (record key 'val': the pilot's
scores there and J_u, K_u at the report floors), so that a floor or a weight can be chosen on validation alone, and the
scores add the tail with every failed inversion counted as an infinite error (p95inf@f, None when failures exceed five
percent) and the share of the domain above five points with failures included (over5_pct@f). The training, the arms
and every score of version 1 are unchanged. One weight is added: 'phys-w1', the band variance sigma_b^2 alone, which
trains on the physical squared error instead of the standardized one; it separates the part of a retrieval weight
(in training coordinates w sigma_b^2) that only undoes the standardization from the transmission factor w.

p3_flattened_weights.py with the choice of weight added. In the standardized coordinates the networks are trained in,
the weight of an entry carries the variance of its band, and there the order of the two weights of the notes is the
reverse of their order in physical units. On the EMIT training block of split 101 at u = 1e-2 the floored weight
1 / max(t, tau)^2 has entry effective fraction 4e-4 for the path radiance (75 effective states; two nearly opaque
states hold 16 percent of it), the cut weight t^-2 1{t >= tau} 4.7e-3 (about 6,000 effective states): the floor gives
its largest value to the opaque entries of the high-variance bands, which the cut removes. Every weighted network of
the pilot, of p3_mixed_objective.py and of p3_flattened_weights.py used the floored weight.

That concentration turned out to come from two failed evaluations in the EMIT table, rows 4011 and 7439: from band
88 (about 1030 nm) on, their direct and diffuse terms are exactly zero in 197 bands and their spherical albedo is 2.0
in 107, while their path radiance matches their neighbours; they are the only rows with an albedo of one or more.
Floored, their zero transmission gives them the largest weight of every high-variance band. Without them the floored
weight keeps about 7,100 effective states for the path radiance at u = 1e-2 and 8,900 at 1e-3
(results/training_coordinates.json holds the numbers with them). --drop-rows removes rows from every block after the
split, so that the splits are otherwise those of the pilot and of the earlier drivers.

Arms (--arms): 'plain'; '<fam><value>' with fam 'w' (the pure weight; value 1), 'flat' (the weight to the power
value, renormalized to mean one on the training block) or 'mix' ((1 - value) + value times the weight); the weight is
the floored one unless the arm starts with 'cut-'; '@<u>' after the value sets the arm's floor (default --u); any
weighted arm may end with ':sp' to stop on the plain validation loss instead of the validation loss under its training
weights. Examples: cut-w1, cut-w1:sp, cut-mix0.2, w1@1e-3, w1@1e-2:sp. Everything else is p3_flattened_weights.py
unchanged: the split, seeds, standardization, architecture, schedule and score(); every arm records J_u (the floored
objective) and K_u (the cut objective: the entries at or above the floor, weighted by t^-2, averaged over all entries)
on the test block at each floor of --report-floors. --save-preds writes the test-block predictions of every arm
(float32).

usage: EMIT_DATA=<dir> python p3_weight_family.py --seed 101 [--arms plain w1@1e-1 w1@1e-2 w1@1e-3 ...] [--u 1e-2]
       [--drop-rows 4011 7439] [--report-floors 1e-1 1e-2 1e-3] [--epochs 150] [--threads 4] [--data DIR]
       [--out results] [--tag TAG] [--record FILE] [--save-preds]
"""
import argparse
import hashlib
import json
import os
import pathlib
import time

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--arms", nargs="+", default=["plain", "cut-w1", "cut-w1:sp", "cut-mix0.2", "cut-flat0.5"])
p.add_argument("--u", type=float, default=1e-2)
p.add_argument("--drop-rows", type=int, nargs="*", default=[])
p.add_argument("--report-floors", type=float, nargs="+", default=[1e-1, 1e-2, 1e-3])
p.add_argument("--save-preds", action="store_true")
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
TAG = args.tag or f"p3fs_s{args.seed}"
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
# rows left out of every block after the split, so that the splits are otherwise those of the pilot
DROP = np.array(sorted(set(args.drop_rows)), dtype=int)
dropped_from = {int(i): ("test" if i in idx_te else "validation" if i in idx_val else "training") for i in DROP}
if DROP.size:
    idx_te, idx_val, idx_tr = (b[~np.isin(b, DROP)] for b in (idx_te, idx_val, idx_tr))


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


def phys_weights(rows, u):
    """The physical squared error in the standardized coordinates of training: the weight of band b is sigma_b^2 for
    every component and entry, with no transmission factor (u is ignored)."""
    ones = np.ones((len(rows), 1))
    return {c: ones * ystd[c].std[None, :] ** 2 for c in C}


def cut_weights(rows, u):
    """weights() with 1 / max(t, tau)^2 replaced by t^-2 1{t >= tau}."""
    tau = u * T_train
    t = T[rows]
    keep = t >= tau
    inv = np.where(keep, 1.0 / np.where(keep, t, 1.0) ** 2, 0.0)
    w = {"Y1": inv, "Y2": R ** 2 * inv, "Y3": R ** 2 * inv, "Y4": R ** 4 * keep.astype(np.float64)}
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
        return (ystd[c].inv(model(xte).numpy().astype(np.float64)),
                ystd[c].inv(model(xva).numpy().astype(np.float64)), ep + 1)


def score(P, rows=None):
    """The pilot's score(), on the test block unless another block is given, plus the tail with every failed
    inversion counted as an infinite error (p95inf) and the share of the domain above 5 points, failures included."""
    rows = idx_te if rows is None else rows
    Tt = {c: Ys[c][rows] for c in C}
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
        e_inf = np.where(ok, np.abs(np.where(ok, rho, RHO) - RHO), np.inf)[dom]
        q = float(np.quantile(e_inf, 0.95))
        out[f"p95inf@{f:g}"] = 100 * q if np.isfinite(q) else None
        out[f"over5_pct@{f:g}"] = 100 * float((e_inf > 0.05).mean())
    return out


def J_test(P, u, rows=None):
    rows = idx_te if rows is None else rows
    t = np.clip(T[rows], 0, None)
    e = {c: P[c] - Ys[c][rows] for c in C}
    L = (e["Y1"] ** 2 + R ** 2 * (e["Y2"] ** 2 + e["Y3"] ** 2) + R ** 4 * t ** 2 * e["Y4"] ** 2) / np.maximum(t, u * T_train) ** 2
    return float(L.mean())


def K_test(P, u, rows=None):
    """The cut objective on the test block: J_u with 1 / max(t, tau)^2 replaced by t^-2 1{t >= tau}."""
    rows = idx_te if rows is None else rows
    t = np.clip(T[rows], 0, None)
    keep = t >= u * T_train
    e = {c: P[c] - Ys[c][rows] for c in C}
    num = e["Y1"] ** 2 + R ** 2 * (e["Y2"] ** 2 + e["Y3"] ** 2) + R ** 4 * t ** 2 * e["Y4"] ** 2
    return float(np.where(keep, num / np.where(keep, t, 1.0) ** 2, 0.0).mean())


def parse_arm(spec):
    """'plain' -> (name, None, None, None, None, False); '[cut-]<fam><value>[@<u>][:sp]' -> (name, kind, family,
    value, u, sp), with u the floor of the arm (--u when no '@' is given)."""
    base, _, stop = spec.partition(":")
    if stop not in ("", "sp"):
        raise SystemExit(f"unknown stopping rule in arm {spec!r}")
    if base == "plain":
        if stop:
            raise SystemExit("the plain arm already stops on the plain validation loss")
        return "plain", None, None, None, None, False
    base, _, ustr = base.partition("@")
    u = float(ustr) if ustr else args.u
    kind = "cut" if base.startswith("cut-") else "phys" if base.startswith("phys-") else "floored"
    base = base[4:] if kind == "cut" else base[5:] if kind == "phys" else base
    for fam in ("flat", "mix", "w"):
        if base.startswith(fam):
            val = float(base[len(fam):])
            if not 0 < val <= 1 or (fam == "w" and val != 1):
                raise SystemExit(f"arm {spec!r}: the parameter must lie in (0, 1], and be 1 for the pure weight")
            if kind == "phys":
                name = f"phys-{fam}{val:g}" + ("_sp" if stop else "")
            else:
                name = ("cut-" if kind == "cut" else "") + f"{fam}{val:g}_u{u:g}" + ("_sp" if stop else "")
            return name, kind, fam, val, u, bool(stop)
    raise SystemExit(f"unknown arm {spec!r}")


def kish(w):
    w = np.asarray(w, dtype=np.float64).ravel()
    return float(w.mean() ** 2 / (w ** 2).mean())


arms = [parse_arm(s) for s in args.arms]
if arms[0][0] != "plain":
    raise SystemExit("the first arm must be 'plain', the reference of every paired difference")
if len({a[0] for a in arms}) != len(arms):
    raise SystemExit("two arms share a name")
rec = {"tag": TAG, "kind": "p3_weight_family_v2", "seed": args.seed, "n_train": int(len(idx_tr)),
       "n_val": int(len(idx_val)), "n_test": int(len(idx_te)), "T_train": T_train, "widths": list(WIDTHS),
       "epochs": args.epochs, "members": 1, "arm_specs": args.arms, "u": args.u,
       "dropped_rows": dropped_from, "report_floors": args.report_floors,
       "driver_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), "arms": {}}
W = {}


def weights_for(kind, u):
    if (kind, u) not in W:
        f = {"floored": weights, "cut": cut_weights, "phys": phys_weights}[kind]
        W[(kind, u)] = (f(idx_tr, u), f(idx_val, u))
    return W[(kind, u)]


preds = {"idx_te": idx_te}
for name, kind, fam, val, u, sp in arms:
    if fam is None:
        wtr, wva = {c: None for c in C}, {c: None for c in C}
    else:
        wt_all, wv_all = weights_for(kind, u)
        wtr, wva = {}, {}
        for c in C:
            m = wt_all[c].mean()      # the pilot's normalization of the weights to mean one on the training block
            bt, bv = wt_all[c] / m, wv_all[c] / m
            if fam == "mix":
                wtr[c], wva[c] = (1 - val) + val * bt, (1 - val) + val * bv
            elif fam == "w":
                wtr[c], wva[c] = bt, bv
            else:
                ft, fv = bt ** val, bv ** val
                mf = ft.mean()        # the flattened weights renormalized to mean one on the training block
                wtr[c], wva[c] = ft / mf, fv / mf
    P, Pva, eps = {}, {}, {}
    for c in C:
        P[c], Pva[c], eps[c] = train(c, 1000 * args.seed, wtr[c], None if sp else wva[c])
    rec["arms"][name] = {"weight": kind, "family": fam, "value": val, "u": u,
                         "stopping": "plain validation loss" if (sp or fam is None) else "weighted validation loss",
                         "rho_train": {c: (1.0 if wtr[c] is None else kish(wtr[c])) for c in C},
                         "epochs_run": {c: [eps[c]] for c in C}, **score(P),
                         "J_test": {f"u{v:g}": J_test(P, v) for v in args.report_floors},
                         "K_test": {f"u{v:g}": K_test(P, v) for v in args.report_floors},
                         "val": {**{k: v for k, v in score(Pva, idx_val).items() if k != "components"},
                                 "J": {f"u{v:g}": J_test(Pva, v, idx_val) for v in args.report_floors},
                                 "K": {f"u{v:g}": K_test(Pva, v, idx_val) for v in args.report_floors}}}
    if args.save_preds:
        for c in C:
            preds[f"{name}|{c}"] = P[c].astype(np.float32)
    m = rec["arms"][name]
    log(f"== {name}: rad {m['radiance']:.4f}%  allband {m['allband_p95']:.2f}  p95@1e-12 {m['p95@1e-12']:.2f}  "
        f"p95@1e-3 {m['p95@0.001']:.2f}  p95@1e-2 {m['p95@0.01']:.2f}  J_test {m['J_test']}  "
        f"rho {', '.join(f'{c} {v:.3g}' for c, v in m['rho_train'].items())}")
if args.save_preds:
    np.savez_compressed(OUT / (TAG + "_preds.npz"), **preds)
rec["minutes"] = round((time.time() - t0) / 60, 1)
tmp = OUT / (TAG + ".json.tmp")
tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
os.replace(tmp, OUT / (TAG + ".json"))
if args.record:
    pathlib.Path(args.record).write_text(json.dumps(rec, indent=1), encoding="utf-8")
log(f"done {TAG} in {rec['minutes']} min")
