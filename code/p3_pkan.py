"""Retrieval-weighted networks on the pKANrtm corpus, scored against its published results.

The corpus of Mazid and Rishe (Remote Sens. 18:1826, 2026): paired 6S / libRadtran Sentinel-2 correction coefficients
(path reflectance, total transmittance, spherical albedo) for 50,000 states and 13 bands, 609,722 rows after the
release's quality control. The split is the release's own state-level assignment (35,000 / 7,500 / 7,500 states), the
inputs are theirs (eight state and band variables, the aerosol model and atmosphere profile one-hot, and with --lowfi
the 6S coefficients of the same row), and the forward scores are computed with the formulas of their
compute_error_metrics (surrogate_pipeline/train_utils.py: RMSE, MAE, R^2 averaged over the targets, MAPE and SMAPE with
eps 1e-6) on the three libRadtran coefficients of every test row, so the numbers stand beside their Tables 3 and 4.

Each arm is one network with three outputs trained on the sum of the three per-coefficient weighted squared errors,
the weights of the paper's J_u per row at the true transmission t:
    w_a = 1 / max(t, tau)^2,  w_t = R^2 / max(t, tau)^2,  w_s = R^4 t^2 / max(t, tau)^2,  tau = u T_train, R = 0.9
each normalized to mean one on the training rows. Arms: 'plain' (w = 1), 'floored@<u>', 'flat<kappa>@<u>' (the floored
weights to the power kappa, renormalized). With --residual the network predicts libRadtran minus 6S (the residual form
of their best models); the scores are always on the libRadtran coefficients.

Retrieval scores follow the paper's score(): L = a + rho t / (1 - rho s) at rho = 0.7, rho_hat inverted with the
predicted coefficients; the 95th percentile of |rho_hat - rho| (points) over all test rows and on the physical domain
above 1e-12, 1e-3 and 1e-2 T_train, the share of failed inversions there, and the radiance error, the mean over test
states of the relative L2 error of L over the state's bands.

usage: python p3_pkan.py --seed 0 [--arms plain floored@0.1 flat0.5@0.01] [--lowfi 1] [--residual 1] [--epochs 80]
       [--width 384] [--depth 4] [--threads 4] [--out DIR] [--tag TAG] [--record FILE] [--smoke]"""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--arms", nargs="+", default=["plain", "floored@0.1", "flat0.5@0.01"])
p.add_argument("--lowfi", type=int, default=1)
p.add_argument("--residual", type=int, default=1)
p.add_argument("--epochs", type=int, default=80)
p.add_argument("--width", type=int, default=384)
p.add_argument("--depth", type=int, default=4)
p.add_argument("--bs", type=int, default=1024)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--out", default="results")
p.add_argument("--tag", default="")
p.add_argument("--record", default="")
p.add_argument("--smoke", action="store_true", help="2,000 train states, 3 epochs")
args = p.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, str(args.threads))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(args.threads)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_data  # noqa: E402

t0 = time.time()
TAG = args.tag or f"p3pk_s{args.seed}"
OUT = pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
RHO, R = 0.7, 0.9
FLOORS = (1e-12, 1e-3, 1e-2)
EPS = 1e-6                       # PERCENT_METRIC_EPS of the pKANrtm release


def log(msg):
    print(f"[{(time.time() - t0) / 60:7.1f} min] {msg}", flush=True)


# ---------------------------------------------------------------- data: the release's rows, split and inputs
root = bench_data.DN / "pkanrtm"
z = np.load(root / "paired_arrays.npz", allow_pickle=True)
X, Yh, Yl, sid, band = z["X"], z["Yh"], z["Yl"], z["sid"], z["band"]
aero, prof, rel = bench_data._pkanrtm_cats(root, sid, band)
masks = {"tr": rel == "train", "va": rel == "val", "te": rel == "test"}
if args.smoke:
    keep_states = set(np.unique(sid[masks["tr"]])[:2000])
    masks["tr"] = masks["tr"] & np.array([s in keep_states for s in sid])
    masks["va"] = masks["va"] & (np.cumsum(masks["va"]) <= 5000)
    masks["te"] = masks["te"] & (np.cumsum(masks["te"]) <= 5000)
onehot = np.zeros((len(X), len(bench_data.AERO_LEVELS) + len(bench_data.PROF_LEVELS)))
onehot[np.arange(len(X)), aero] = 1.0
onehot[np.arange(len(X)), len(bench_data.AERO_LEVELS) + prof] = 1.0
Xin = np.concatenate([X, onehot] + ([Yl] if args.lowfi else []), 1)
if args.residual and not args.lowfi:
    sys.exit("--residual needs --lowfi 1 (the 6S coefficients are the base the residual is added to)")
Xs = {k: Xin[m] for k, m in masks.items()}
Yhs = {k: Yh[m] for k, m in masks.items()}
Yls = {k: Yl[m] for k, m in masks.items()}
sids = {k: sid[m] for k, m in masks.items()}
mu, sd = Xs["tr"].mean(0), Xs["tr"].std(0) + 1e-8
Xn = {k: ((v - mu) / sd).astype(np.float32) for k, v in Xs.items()}
base = {k: (Yls[k] if args.residual else np.zeros_like(Yhs[k])) for k in Xs}
Ttr = Yhs["tr"] - base["tr"]
ym, ysd = Ttr.mean(0), Ttr.std(0) + 1e-12
Zt = {k: ((Yhs[k] - base[k] - ym) / ysd).astype(np.float32) for k in Xs}
t_true = {k: np.clip(Yhs[k][:, 1], 0, None) for k in Xs}
T_train = float(np.median(t_true["tr"][t_true["tr"] > 0]))
log(f"rows tr {masks['tr'].sum()} va {masks['va'].sum()} te {masks['te'].sum()}; states "
    + ", ".join(f"{k} {len(np.unique(v))}" for k, v in sids.items()) + f"; inputs {Xin.shape[1]}; T_train {T_train:.4g}")


def weights(kind, u, kappa, split):
    """Per-row weights of the three coefficients, (n, 3), mean one per coefficient on the training rows."""
    t = t_true[split]
    if kind is None:
        return np.ones((len(t), 3), np.float32)
    tau = u * T_train
    m2 = np.maximum(t, tau) ** 2
    W = np.stack([1.0 / m2, R ** 2 / m2, R ** 4 * t ** 2 / m2], 1)
    if kind == "flat":
        W = W ** kappa
    ttr = t_true["tr"]
    m2tr = np.maximum(ttr, tau) ** 2
    Wtr = np.stack([1.0 / m2tr, R ** 2 / m2tr, R ** 4 * ttr ** 2 / m2tr], 1)
    if kind == "flat":
        Wtr = Wtr ** kappa
    return (W / Wtr.mean(0)).astype(np.float32)


def parse(spec):
    if spec == "plain":
        return spec, None, None, None
    fam, u = spec.split("@")
    if fam == "floored":
        return spec, "floored", float(u), 1.0
    if fam.startswith("flat"):
        return spec, "flat", float(u), float(fam[4:])
    raise SystemExit(f"unknown arm {spec}")


# ---------------------------------------------------------------- network
def make_net():
    layers, d = [], Xin.shape[1]
    for _ in range(args.depth):
        layers += [torch.nn.Linear(d, args.width), torch.nn.SiLU()]
        d = args.width
    layers.append(torch.nn.Linear(d, 3))
    return torch.nn.Sequential(*layers)


def predict(net, split):
    with torch.no_grad():
        out = np.concatenate([net(torch.from_numpy(Xn[split][i:i + 65536])).numpy()
                              for i in range(0, len(Xn[split]), 65536)])
    return out.astype(np.float64) * ysd + ym + base[split]


def train(kind, u, kappa, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    epochs = 3 if args.smoke else args.epochs
    steps = epochs * int(np.ceil(len(Xn["tr"]) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    Wtr = torch.from_numpy(weights(kind, u, kappa, "tr"))
    Wva = weights(kind, u, kappa, "va")
    Xtr, Ztr = torch.from_numpy(Xn["tr"]), torch.from_numpy(Zt["tr"])
    best, best_state, best_ep = np.inf, None, -1
    for ep in range(epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(Xtr)))
        for i in range(0, len(perm), args.bs):
            b = perm[i:i + args.bs]
            loss = (Wtr[b] * (net(Xtr[b]) - Ztr[b]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            pv = np.concatenate([net(torch.from_numpy(Xn["va"][i:i + 65536])).numpy() for i in range(0, len(Xn["va"]), 65536)])
        vl = float((Wva * (pv - Zt["va"]) ** 2).mean())        # stopping on the arm's own weighted validation loss
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"   epoch {ep}: train {loss.item():.4g} val {vl:.4g} (best {best:.4g} @ {best_ep})")
    net.load_state_dict(best_state)
    return net, best_ep


# ---------------------------------------------------------------- scores
def pkan_metrics(yt, yp):
    """compute_error_metrics of the pKANrtm release, overall and per target."""
    def one(a, b):
        ae = np.abs(b - a)
        out = {"rmse": float(np.sqrt(np.mean((b - a) ** 2))), "mae": float(ae.mean()),
               "mape": float(np.mean(ae / np.maximum(np.abs(a), EPS)) * 100.0),
               "smape": float(np.mean(2.0 * ae / np.maximum(np.abs(b) + np.abs(a), EPS)) * 100.0)}
        if a.ndim == 1:
            out["r2"] = float(1.0 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())
        else:
            out["r2"] = float(np.mean([1.0 - ((a[:, j] - b[:, j]) ** 2).sum() / ((a[:, j] - a[:, j].mean()) ** 2).sum()
                                       for j in range(a.shape[1])]))
        return out
    res = {k: one(yt[:, j], yp[:, j]) for j, k in enumerate(("rho_path", "T_total", "spher_alb"))}
    res["overall"] = one(yt, yp)
    return res


def retrieval_scores(yt, yp, s_ids):
    a, t, s = yt[:, 0], yt[:, 1], yt[:, 2]
    ah, th, sh = yp[:, 0], yp[:, 1], yp[:, 2]
    L = a + RHO * t / (1 - RHO * s)
    Lh = ah + RHO * th / (1 - RHO * sh)
    # radiance and component errors: mean over test states of the relative L2 error over the state's bands
    order = np.argsort(s_ids, kind="stable")
    ss = s_ids[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])

    def per_state(A, B):
        num = np.add.reduceat(((A - B) ** 2)[order], starts)
        den = np.add.reduceat((A ** 2)[order], starts)
        return 100.0 * float(np.mean(np.sqrt(num) / np.sqrt(np.maximum(den, 1e-300))))
    out = {"radiance": per_state(L, Lh), "components": {k: per_state(yt[:, j], yp[:, j])
                                                        for j, k in enumerate(("rho_path", "T_total", "spher_alb"))}}
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
        out[f"p95@{f:g}"] = 100 * float(np.quantile(np.abs(rho[good] - RHO), 0.95)) if good.any() else float("nan")
        out[f"failed_pct@{f:g}"] = 100 * float((dom & ~ok).sum() / max(dom.sum(), 1))
        out[f"coverage@{f:g}"] = 100 * float(dom.mean())
    for u in (1e-1, 1e-2, 1e-3):
        m2 = np.maximum(np.clip(t, 0, None), u * T_train) ** 2
        e = yp - yt
        out[f"J@{u:g}"] = float(((e[:, 0] ** 2 + R ** 2 * e[:, 1] ** 2 + R ** 4 * t ** 2 * e[:, 2] ** 2) / m2).mean())
    return out


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


rec = {"tag": TAG, "kind": "p3_pkan", "seed": args.seed, "protocol": "pKANrtm release split (state level), their inputs",
       "lowfi": args.lowfi, "residual": args.residual, "epochs": args.epochs, "width": args.width, "depth": args.depth,
       "bs": args.bs, "lr": args.lr, "smoke": args.smoke, "T_train": T_train,
       "rows": {k: int(v.sum()) for k, v in masks.items()}, "states": {k: int(len(np.unique(v))) for k, v in sids.items()},
       "driver_sha256": sha256(__file__), "data_sha256": sha256(root / "paired_arrays.npz"),
       "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "threads": args.threads},
       "published": {"pKANrtm_standard_split": {"rmse": 0.00619, "mae": 0.00186, "r2": 0.99472, "smape": 3.91249},
                     "sRTMNet_style_standard_split": {"rmse": 0.00691, "mae": 0.00211, "r2": 0.99418, "smape": 4.28170},
                     "source": "Mazid and Rishe, Remote Sens. 18:1826 (2026), Table 3"},
       "arms": {}}
rec["six_s_baseline"] = {"pkan_metrics": pkan_metrics(Yhs["te"], Yls["te"]),
                         "retrieval": retrieval_scores(Yhs["te"], Yls["te"], sids["te"])}
log(f"6S as the emulator: overall RMSE {rec['six_s_baseline']['pkan_metrics']['overall']['rmse']:.5f}")
for spec in args.arms:
    name, kind, u, kappa = parse(spec)
    ta = time.time()
    log(f"== arm {name}")
    net, best_ep = train(kind, u, kappa, 1000 + args.seed)
    yp = predict(net, "te")
    rec["arms"][name] = {"kind": kind, "u": u, "kappa": kappa, "best_epoch": best_ep,
                         "pkan_metrics": pkan_metrics(Yhs["te"], yp), "retrieval": retrieval_scores(Yhs["te"], yp, sids["te"]),
                         "minutes": round((time.time() - ta) / 60, 1)}
    m = rec["arms"][name]
    log(f"== {name}: RMSE {m['pkan_metrics']['overall']['rmse']:.5f} MAE {m['pkan_metrics']['overall']['mae']:.5f} "
        f"R2 {m['pkan_metrics']['overall']['r2']:.5f} SMAPE {m['pkan_metrics']['overall']['smape']:.3f} | radiance "
        f"{m['retrieval']['radiance']:.4f}% p95 all {m['retrieval']['allband_p95']:.2f} phys "
        + " ".join(f"{m['retrieval'][f'p95@{f:g}']:.2f}" for f in FLOORS) + f" ({m['minutes']} min)")
rec["minutes"] = round((time.time() - t0) / 60, 1)
tmp = OUT / (TAG + ".json.tmp")
tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
os.replace(tmp, OUT / (TAG + ".json"))
if args.record:
    pathlib.Path(args.record).write_text(json.dumps(rec, indent=1), encoding="utf-8")
log(f"done {TAG} in {rec['minutes']} min")
