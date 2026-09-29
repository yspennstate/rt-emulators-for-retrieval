"""Retrieval-weighted networks on the pKANrtm corpus, second wave: GPU, more arms, the out-of-distribution split, a
weighted kernel correction, and retrieval scores over a range of surface reflectances.

Same data, split, inputs, weights and scores as p3_pkan.py; what is new:
  --device auto|cpu|cuda      the network and the kernel correction run on the GPU it is given
  --split official|ood        official: the release's 35,000/7,500/7,500 state split. ood: the paper-2 protocol - test
                              states are those above the 0.85 quantile of aerosol optical depth OR of water-vapour column
                              (over states); the rest are split 85/15 train/validation by --seed
  arms                        plain, floored@<u>, flat<k>@<u> as before, plus
                                flatA<k>@<u>        flattened weights on path reflectance and transmittance, weight one on
                                                    the spherical albedo (its retrieval weight vanishes with t, so the
                                                    plain fit of it is kept)
                                mix<l>:<arm>        (1 - l) x the arm's weights + l x one, both mean one per coefficient
  --kc 1                      after each arm, kernel ridge on the network's residuals with the SAME per-row weights
                              (Nystrom, --kc-m landmarks, Gaussian kernel on the standardized inputs, bandwidth over
                              {0.5, 1, 2} x the landmarks' median distance and the ridge over nine decades, both chosen
                              on the arm's own weighted validation loss); scored as a separate entry '<arm>+kc'
  --rhos                      the retrieval scores are also given at these surface reflectances (0.7 stays the headline)
  --save-preds                the test predictions of every arm, float32, for later analysis

usage: python p3_pkan2.py --seed 0 --arms plain flat0.5@0.01 flatA0.5@0.01 --epochs 300 --width 512 --depth 5
       [--split official|ood] [--kc 1] [--device auto] [--out DIR] [--tag TAG] [--record FILE] [--smoke]"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0)
p.add_argument("--arms", nargs="+", default=["plain", "floored@0.1", "flat0.5@0.01"])
p.add_argument("--split", choices=["official", "ood"], default="official")
p.add_argument("--lowfi", type=int, default=1)
p.add_argument("--residual", type=int, default=1)
p.add_argument("--epochs", type=int, default=80)
p.add_argument("--width", type=int, default=384)
p.add_argument("--depth", type=int, default=4)
p.add_argument("--bs", type=int, default=1024)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--device", default="auto")
p.add_argument("--kc", type=int, default=0)
p.add_argument("--kc-m", type=int, default=4096)
p.add_argument("--rhos", nargs="+", type=float, default=[0.05, 0.1, 0.3, 0.5, 0.7])
p.add_argument("--save-preds", action="store_true")
p.add_argument("--out", default="results")
p.add_argument("--tag", default="")
p.add_argument("--record", default="")
p.add_argument("--smoke", action="store_true", help="2,000 train states, 3 epochs, 256 landmarks")
args = p.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, str(args.threads))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(args.threads)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_data  # noqa: E402

t0 = time.time()
TAG = args.tag or f"p3pk2_s{args.seed}"
OUT = pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
RHO, R = 0.7, 0.9
FLOORS = (1e-12, 1e-3, 1e-2)
EPS = 1e-6                       # PERCENT_METRIC_EPS of the pKANrtm release
if args.device == "auto":
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEV = torch.device(args.device)
if args.smoke:
    args.kc_m = min(args.kc_m, 256)


def log(msg):
    print(f"[{(time.time() - t0) / 60:7.1f} min] {msg}", flush=True)


# ---------------------------------------------------------------- data: the release's rows and inputs
root = bench_data.DN / "pkanrtm"
z = np.load(root / "paired_arrays.npz", allow_pickle=True)
X, Yh, Yl, sid, band = z["X"], z["Yh"], z["Yl"], z["sid"], z["band"]
aero, prof, rel = bench_data._pkanrtm_cats(root, sid, band)
if args.split == "official":
    masks = {"tr": rel == "train", "va": rel == "val", "te": rel == "test"}
else:
    # paper-2 protocol: OOD test states exceed the 0.85 quantile of aerosol (aod550, column 4) or water vapour
    # (cwv_cm, column 5), taken over states; the remaining states are split 85/15 by the seed
    states, first = np.unique(sid, return_index=True)
    aod_s, cwv_s = X[first, 4], X[first, 5]
    qa, qc = np.quantile(aod_s, 0.85), np.quantile(cwv_s, 0.85)
    te_states = set(states[(aod_s > qa) | (cwv_s > qc)])
    rest = np.array(sorted(set(states) - te_states))
    rng_split = np.random.default_rng(args.seed)
    perm = rng_split.permutation(len(rest))
    va_states = set(rest[perm[:int(round(0.15 * len(rest)))]])
    is_te = np.isin(sid, list(te_states))
    is_va = np.isin(sid, list(va_states))
    masks = {"tr": ~(is_te | is_va), "va": is_va, "te": is_te}
if args.smoke:
    keep_states = set(np.unique(sid[masks["tr"]])[:2000])
    masks["tr"] = masks["tr"] & np.isin(sid, list(keep_states))
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
log(f"device {DEV}; split {args.split}; rows tr {masks['tr'].sum()} va {masks['va'].sum()} te {masks['te'].sum()}; states "
    + ", ".join(f"{k} {len(np.unique(v))}" for k, v in sids.items()) + f"; inputs {Xin.shape[1]}; T_train {T_train:.4g}")
XT = {k: torch.from_numpy(v).to(DEV) for k, v in Xn.items()}
ZT = {k: torch.from_numpy(v).to(DEV) for k, v in Zt.items()}


def raw_weights(kind, u, kappa, t):
    if kind is None:
        return np.ones((len(t), 3), np.float64)
    tau = u * T_train
    m2 = np.maximum(t, tau) ** 2
    W = np.stack([1.0 / m2, R ** 2 / m2, R ** 4 * t ** 2 / m2], 1)
    if kind in ("flat", "flatA"):
        W = W ** kappa
    if kind == "flatA":
        W[:, 2] = 1.0
    return W


def weights(spec, split):
    """Per-row weights of the three coefficients, (n, 3), mean one per coefficient on the training rows."""
    kind, u, kappa, lam = spec
    W = raw_weights(kind, u, kappa, t_true[split])
    Wtr = raw_weights(kind, u, kappa, t_true["tr"])
    W = W / Wtr.mean(0)
    if lam:
        W = (1.0 - lam) * W + lam
    return W.astype(np.float32)


def parse(spec):
    """'plain' | 'floored@u' | 'flat<k>@u' | 'flatA<k>@u' | 'mix<l>:<one of those>' -> (kind, u, kappa, lam)."""
    lam = 0.0
    s = spec
    if s.startswith("mix"):
        head, s = s.split(":", 1)
        lam = float(head[3:])
    if s == "plain":
        return None, None, None, lam
    fam, u = s.split("@")
    if fam == "floored":
        return "floored", float(u), 1.0, lam
    if fam.startswith("flatA"):
        return "flatA", float(u), float(fam[5:]), lam
    if fam.startswith("flat"):
        return "flat", float(u), float(fam[4:]), lam
    raise SystemExit(f"unknown arm {spec}")


# ---------------------------------------------------------------- network
def make_net():
    layers, d = [], Xin.shape[1]
    for _ in range(args.depth):
        layers += [torch.nn.Linear(d, args.width), torch.nn.SiLU()]
        d = args.width
    layers.append(torch.nn.Linear(d, 3))
    return torch.nn.Sequential(*layers).to(DEV)


def forward_all(net, split):
    with torch.no_grad():
        return torch.cat([net(XT[split][i:i + 65536]) for i in range(0, len(XT[split]), 65536)])


def to_phys(Zp, split):
    return Zp.detach().cpu().numpy().astype(np.float64) * ysd + ym + base[split]


def train(spec, seed):
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    epochs = 3 if args.smoke else args.epochs
    n = len(XT["tr"])
    steps = epochs * int(math.ceil(n / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    Wtr = torch.from_numpy(weights(spec, "tr")).to(DEV)
    Wva = torch.from_numpy(weights(spec, "va")).to(DEV)
    best, best_state, best_ep = math.inf, None, -1
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=gen).to(DEV)
        for i in range(0, n, args.bs):
            b = perm[i:i + args.bs]
            loss = (Wtr[b] * (net(XT["tr"][b]) - ZT["tr"][b]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        vl = float((Wva * (forward_all(net, "va") - ZT["va"]) ** 2).mean())   # the arm's own weighted validation loss
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if ep % 25 == 0 or ep == epochs - 1:
            log(f"   epoch {ep}: train {loss.item():.4g} val {vl:.4g} (best {best:.4g} @ {best_ep})")
    if best_state is None:                           # every validation loss was NaN: keep the last state, say so
        log("   no finite validation loss; the last epoch's network is scored")
        return net, -1, float("nan")
    net.load_state_dict(best_state)
    return net, best_ep, best


# ---------------------------------------------------------------- weighted kernel correction of the residuals
def gauss(A, L, h2):
    return torch.exp(-torch.cdist(A, L).pow(2) / (2.0 * h2))


def kernel_correct(net, spec, seed):
    """Nystrom kernel ridge on the residuals, per coefficient, with the arm's row weights; returns the corrected
    standardized test prediction and a record. The ridge and bandwidth are chosen on the weighted validation loss;
    if no choice beats the uncorrected network there, the correction is zero and the record says so."""
    m = min(args.kc_m, len(XT["tr"]))
    g = torch.Generator(device="cpu").manual_seed(7000 + seed)
    idx = torch.randperm(len(XT["tr"]), generator=g)[:m].to(DEV)
    L = XT["tr"][idx]
    with torch.no_grad():
        Rtr = ZT["tr"] - forward_all(net, "tr")
        Rva = ZT["va"] - forward_all(net, "va")
        Wtr = torch.from_numpy(weights(spec, "tr")).to(DEV)
        Wva = torch.from_numpy(weights(spec, "va")).to(DEV)
        v0 = float((Wva * Rva ** 2).mean())
        D2 = torch.cdist(L, L).pow(2)
        med = float(torch.sqrt(D2[torch.triu(torch.ones_like(D2, dtype=torch.bool), 1)].median()))
        best = dict(val=v0, c=None, lam=None)
        best_alpha = None
        eye = torch.eye(m, dtype=torch.float64, device=DEV)
        for c in (0.5, 1.0, 2.0):
            h2 = (c * med) ** 2
            Kmm = torch.exp(-D2.double() / (2.0 * h2))
            A = torch.zeros(3, m, m, dtype=torch.float64, device=DEV)
            b = torch.zeros(3, m, dtype=torch.float64, device=DEV)
            for i in range(0, len(XT["tr"]), 16384):
                K = gauss(XT["tr"][i:i + 16384], L, h2)
                for j in range(3):
                    Kw = K * Wtr[i:i + 16384, j:j + 1]
                    A[j] += (Kw.T @ K).double()
                    b[j] += (Kw.T @ Rtr[i:i + 16384, j:j + 1]).double().squeeze(1)
            Kva = torch.cat([gauss(XT["va"][i:i + 16384], L, h2) for i in range(0, len(XT["va"]), 16384)])
            for lam in np.logspace(-8, 0, 9):
                alpha = torch.zeros(3, m, dtype=torch.float64, device=DEV)
                ok = True
                for j in range(3):
                    scale = float(torch.diagonal(A[j]).mean()) + 1e-30
                    M = A[j] + lam * scale * Kmm + 1e-10 * scale * eye
                    try:
                        Lc = torch.linalg.cholesky(M)
                        alpha[j] = torch.cholesky_solve(b[j].unsqueeze(1), Lc).squeeze(1)
                    except RuntimeError:
                        ok = False
                        break
                if not ok or not torch.isfinite(alpha).all():
                    continue
                corr_va = torch.cat([(Kva[i:i + 16384].double() @ alpha.T).float()      # chunked: no fp64 copy of Kva
                                     for i in range(0, len(Kva), 16384)])
                v = float((Wva * (Rva - corr_va) ** 2).mean())
                if v < best["val"]:
                    best, best_alpha = dict(val=v, c=c, lam=float(lam)), (alpha, h2)
            del A, b, Kva
        rec = {"m": m, "median_distance": med, "val_uncorrected": v0, "val_corrected": best["val"],
               "bandwidth_mult": best["c"], "ridge_rel": best["lam"], "applied": best_alpha is not None}
        Zte = forward_all(net, "te")
        if best_alpha is not None:
            alpha, h2 = best_alpha
            corr = torch.cat([(gauss(XT["te"][i:i + 16384], L, h2).double() @ alpha.T).float()
                              for i in range(0, len(XT["te"]), 16384)])
            Zte = Zte + corr
    return Zte, rec


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


def retrieval_scores(yt, yp, s_ids, rho=RHO, full=True):
    a, t, s = yt[:, 0], yt[:, 1], yt[:, 2]
    ah, th, sh = yp[:, 0], yp[:, 1], yp[:, 2]
    L = a + rho * t / (1 - rho * s)
    Lh = ah + rho * th / (1 - rho * sh)
    order = np.argsort(s_ids, kind="stable")
    ss = s_ids[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])

    def per_state(A, B):
        num = np.add.reduceat(((A - B) ** 2)[order], starts)
        den = np.add.reduceat((A ** 2)[order], starts)
        return 100.0 * float(np.mean(np.sqrt(num) / np.sqrt(np.maximum(den, 1e-300))))
    out = {"radiance": per_state(L, Lh)}
    if full:
        out["components"] = {k: per_state(yt[:, j], yp[:, j]) for j, k in enumerate(("rho_path", "T_total", "spher_alb"))}
    u_ = L - ah
    den = th + sh * u_
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = u_ / den
    err = np.abs(np.where(np.isfinite(rr), rr, 0.0) - rho)
    out["allband_p95"] = 100 * float(np.quantile(err, 0.95))
    ok = np.isfinite(rr) & (den > 0)
    for f in FLOORS:
        dom = (t >= f * T_train) & (s >= 0) & (s < 1) & (1 - rho * s >= 0.3)
        good = dom & ok
        out[f"p95@{f:g}"] = 100 * float(np.quantile(np.abs(rr[good] - rho), 0.95)) if good.any() else float("nan")
        out[f"failed_pct@{f:g}"] = 100 * float((dom & ~ok).sum() / max(dom.sum(), 1))
        if full:
            out[f"coverage@{f:g}"] = 100 * float(dom.mean())
    if full:
        for u in (1e-1, 1e-2, 1e-3):
            m2 = np.maximum(np.clip(t, 0, None), u * T_train) ** 2
            e = yp - yt
            out[f"J@{u:g}"] = float(((e[:, 0] ** 2 + R ** 2 * e[:, 1] ** 2 + R ** 4 * t ** 2 * e[:, 2] ** 2) / m2).mean())
    return out


def score(yp):
    return {"pkan_metrics": pkan_metrics(Yhs["te"], yp), "retrieval": retrieval_scores(Yhs["te"], yp, sids["te"]),
            "retrieval_by_rho": {f"{r:g}": retrieval_scores(Yhs["te"], yp, sids["te"], rho=r, full=False) for r in args.rhos}}


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def line(name, m):
    return (f"== {name}: RMSE {m['pkan_metrics']['overall']['rmse']:.5f} MAE {m['pkan_metrics']['overall']['mae']:.5f} "
            f"R2 {m['pkan_metrics']['overall']['r2']:.5f} SMAPE {m['pkan_metrics']['overall']['smape']:.3f} | radiance "
            f"{m['retrieval']['radiance']:.4f}% p95 all {m['retrieval']['allband_p95']:.2f} phys "
            + " ".join(f"{m['retrieval'][f'p95@{f:g}']:.2f}" for f in FLOORS))


rec = {"tag": TAG, "kind": "p3_pkan2", "seed": args.seed, "split": args.split,
       "protocol": ("pKANrtm release split (state level), their inputs" if args.split == "official" else
                    "paper-2 OOD split: test states above the 0.85 quantile of aod550 or cwv (over states), rest 85/15"),
       "lowfi": args.lowfi, "residual": args.residual, "epochs": args.epochs, "width": args.width, "depth": args.depth,
       "bs": args.bs, "lr": args.lr, "smoke": args.smoke, "T_train": T_train, "kc": args.kc, "kc_m": args.kc_m,
       "device": str(DEV), "gpu": (torch.cuda.get_device_name(0) if DEV.type == "cuda" else None),
       "rows": {k: int(v.sum()) for k, v in masks.items()}, "states": {k: int(len(np.unique(v))) for k, v in sids.items()},
       "driver_sha256": sha256(__file__), "data_sha256": sha256(root / "paired_arrays.npz"),
       "env": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "threads": args.threads},
       "published": {"pKANrtm_standard_split": {"rmse": 0.00619, "mae": 0.00186, "r2": 0.99472, "smape": 3.91249},
                     "sRTMNet_style_standard_split": {"rmse": 0.00691, "mae": 0.00211, "r2": 0.99418, "smape": 4.28170},
                     "pKANrtm_OOD_rmse": 0.01022,
                     "source": "Mazid and Rishe, Remote Sens. 18:1826 (2026), Tables 3 and 4"},
       "arms": {}}
rec["six_s_baseline"] = score(Yls["te"])
log(f"6S as the emulator: overall RMSE {rec['six_s_baseline']['pkan_metrics']['overall']['rmse']:.5f}")
for spec_name in args.arms:
    spec = parse(spec_name)
    ta = time.time()
    log(f"== arm {spec_name}")
    net, best_ep, best_val = train(spec, 1000 + args.seed)
    Zte = forward_all(net, "te")
    yp = to_phys(Zte, "te")
    rec["arms"][spec_name] = dict(kind=spec[0], u=spec[1], kappa=spec[2], mix=spec[3], best_epoch=best_ep,
                                  best_val=best_val, minutes=None, **score(yp))
    if args.save_preds:
        np.save(OUT / f"{TAG}__{spec_name.replace(':', '_')}__te.npy", yp.astype(np.float32))
    rec["arms"][spec_name]["minutes"] = round((time.time() - ta) / 60, 1)
    log(line(spec_name, rec["arms"][spec_name]) + f" ({rec['arms'][spec_name]['minutes']} min)")
    if args.kc:
        tk = time.time()
        Zc, krec = kernel_correct(net, spec, args.seed)
        ypc = to_phys(Zc, "te")
        name = spec_name + "+kc"
        rec["arms"][name] = dict(kind=spec[0], u=spec[1], kappa=spec[2], mix=spec[3], kernel=krec, **score(ypc),
                                 minutes=round((time.time() - tk) / 60, 1))
        if args.save_preds:
            np.save(OUT / f"{TAG}__{name.replace(':', '_')}__te.npy", ypc.astype(np.float32))
        log(line(name, rec["arms"][name]) + f" | kernel {krec}")
    del net
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
rec["minutes"] = round((time.time() - t0) / 60, 1)
if DEV.type == "cuda":
    rec["gpu_max_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
tmp = OUT / (TAG + ".json.tmp")
tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
os.replace(tmp, OUT / (TAG + ".json"))
if args.record:
    pathlib.Path(args.record).write_text(json.dumps(rec, indent=1), encoding="utf-8")
log(f"done {TAG} in {rec['minutes']} min")
