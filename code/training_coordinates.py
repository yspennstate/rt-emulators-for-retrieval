"""The retrieval weights in the coordinates the networks are trained in.

The networks are fitted to outputs standardized band by band, so against their plain loss (the mean standardized
squared error) the weight of entry (i, b) of component c is w_ib sigma_cb^2, with sigma_cb the training standard
deviation of the band: the retrieval-weighted physical squared error is the standardized squared error weighted by
w sigma^2. For the ten EMIT splits and the floors u in {1e-1, 1e-2, 1e-3} (tau = u T_train) the script reports, on
the training block, for the floored weight 1 / max(t, tau)^2 and the cut weight t^-2 1{t >= tau}:
  - the physical entry effective fraction E[w]^2 / E[w^2] (the quantity of the notes' Proposition on the effective
    sample size), and the effective number of states m_eff = (sum omega)^2 / sum omega^2, omega_i = sum_b w_ib;
  - the same two numbers for w sigma_c^2, per component (the path radiance Y1, the transmission terms Y2 and Y3, the
    albedo term Y4 = t^2 w sigma^2), and the share of the heaviest two states in it.
It also lists the nearly opaque states of the table (t below 1e-2 T_train in more than 200 of the 285 bands) and the
block each split puts them in.

usage: python code/training_coordinates.py --data <EMIT arrays dir> [--out results/training_coordinates.json]
"""
import argparse
import json
import statistics as st

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="results/training_coordinates.json")
ap.add_argument("--drop-rows", type=int, nargs="*", default=[],
                help="rows left out of every block after the split (the failed evaluations 4011 and 7439)")
args = ap.parse_args()
C = ["Y1", "Y2", "Y3", "Y4"]
Y = {c: np.load(f"{args.data}/{c}.npy") for c in C}
T = Y["Y2"] + Y["Y3"]
n_all = T.shape[0]
FLOORS = (1e-1, 1e-2, 1e-3)


def split(seed):
    perm = np.random.RandomState(seed).permutation(n_all)
    n_te = int(round(0.1 * n_all))
    idx_te, tr_full = perm[:n_te], perm[n_te:]
    vperm = np.random.RandomState(seed + 10000).permutation(len(tr_full))
    n_val = int(round(0.1 * len(tr_full)))
    blocks = idx_te, tr_full[vperm[:n_val]], tr_full[vperm[n_val:]]
    return tuple(b[~np.isin(b, args.drop_rows)] for b in blocks)


def entry_fraction(w):
    return float(w.mean() ** 2 / (w * w).mean())


def states(w):
    om = w.sum(1)
    top2 = np.sort(om)[-2:].sum() / om.sum()
    return float(om.sum() ** 2 / (om * om).sum()), float(top2)


med = float(np.median(T[T > 0]))
opaque = np.where((T < 1e-2 * med).sum(1) > 200)[0]
rep = {"data": args.data, "n_states": int(n_all), "bands": int(T.shape[1]), "dropped_rows": sorted(args.drop_rows),
       "nearly_opaque_states": [int(i) for i in opaque], "splits": {}}
print(f"{len(opaque)} nearly opaque states in the table: {opaque.tolist()}")
for seed in range(101, 111):
    te, va, tr = split(seed)
    where = {int(i): ("test" if i in set(te) else "validation" if i in set(va) else "training") for i in opaque}
    tt = T[tr]
    T_train = float(np.median(tt[tt > 0]))
    var = {c: Y[c][tr].var(0) for c in C}
    row = {"T_train": T_train, "m_train": int(len(tr)), "opaque_block": where, "floors": {}}
    for u in FLOORS:
        tau = u * T_train
        keep = tt >= tau
        W = {"floored": 1.0 / np.maximum(tt, tau) ** 2,
             "cut": np.where(keep, 1.0 / np.where(keep, tt, 1.0) ** 2, 0.0)}
        cell = {}
        for kind, w in W.items():
            m_eff, top2 = states(w)
            d = {"physical": {"entry_fraction": entry_fraction(w), "m_eff": m_eff, "top2_share": top2}}
            for c in C:
                wc = w * var[c][None, :] if c != "Y4" else np.clip(tt, 0, None) ** 2 * w * var[c][None, :]
                m_eff, top2 = states(wc)
                d[c] = {"entry_fraction": entry_fraction(wc), "m_eff": m_eff, "top2_share": top2}
            cell[kind] = d
        row["floors"][f"{u:g}"] = cell
    rep["splits"][str(seed)] = row
    f = row["floors"]["0.01"]["floored"]
    print(f"split {seed}: opaque in {sorted(set(where.values()))}; u=1e-2 floored Y1 {f['Y1']['entry_fraction']:.1e} "
          f"m_eff {f['Y1']['m_eff']:.0f} top2 {f['Y1']['top2_share']:.2f}; cut Y1 "
          f"{row['floors']['0.01']['cut']['Y1']['entry_fraction']:.1e} m_eff {row['floors']['0.01']['cut']['Y1']['m_eff']:.0f}")

summary = {}
for u in FLOORS:
    for kind in ("floored", "cut"):
        for c in ["physical"] + C:
            vals = [rep["splits"][s]["floors"][f"{u:g}"][kind][c] for s in rep["splits"]]
            summary[f"u={u:g} {kind} {c}"] = {
                k: {"median": st.median(v[k] for v in vals), "min": min(v[k] for v in vals),
                    "max": max(v[k] for v in vals)} for k in ("entry_fraction", "m_eff", "top2_share")}
rep["summary"] = summary
for k, v in summary.items():
    if "physical" in k or "Y1" in k or "Y2" in k:
        print(f"{k:24s} frac {v['entry_fraction']['median']:.2e} [{v['entry_fraction']['min']:.1e}, "
              f"{v['entry_fraction']['max']:.1e}]  m_eff {v['m_eff']['median']:.0f} [{v['m_eff']['min']:.0f}, "
              f"{v['m_eff']['max']:.0f}]  top2 {v['top2_share']['median']:.3f}")
json.dump(rep, open(args.out, "w", encoding="utf-8"), indent=1)
print("wrote", args.out)
