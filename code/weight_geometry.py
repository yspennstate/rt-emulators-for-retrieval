"""Where the retrieval weights sit on the EMIT table: the numbers quoted in the notes' paragraphs on the floored and cut
weights and on choosing among candidates, for one split.

For the floored weights w = 1 / max(t, u T_train)^2 and the cut weights v = t^-2 1{t >= u T_train} (the path-radiance
component's weights; the transmission components differ by the constant R^2), on the training and validation blocks:
  - the entry effective fraction E[w]^2 / E[w^2];
  - its exact factorization into the state part (sum omega)^2 / (m sum omega^2), omega_i = sum_b w_ib, and the band part;
  - the number of bands carrying 90% of the weight, and the heaviest bands;
  - the share of the weight on entries below the floor;
  - on the training block, the number of bands per state below 1e-3 T_train and the bands more than half below it;
  - on the validation block, the effective number of states and the size of the selection bound of the Proposition
    'choosing on a weighted criterion' for K = 32 candidates and delta = 0.05, against the unweighted bound.

usage: python code/weight_geometry.py --data <EMIT arrays dir> [--seed 101] [--out results/weight_geometry_s101.json]
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--out", default="")
ap.add_argument("--drop-rows", type=int, nargs="*", default=[],
                help="rows left out of both blocks after the split (the EMIT table's failed evaluations are 4011 "
                     "and 7439)")
args = ap.parse_args()
T = np.load(os.path.join(args.data, "Y2.npy")) + np.load(os.path.join(args.data, "Y3.npy"))
n_all = T.shape[0]
perm = np.random.RandomState(args.seed).permutation(n_all)
n_te = int(round(0.1 * n_all))
tr_full = perm[n_te:]
vperm = np.random.RandomState(args.seed + 10000).permutation(len(tr_full))
n_val = int(round(0.1 * len(tr_full)))
blocks = {"validation": tr_full[vperm[:n_val]], "training": tr_full[vperm[n_val:]]}
blocks = {k: b[~np.isin(b, args.drop_rows)] for k, b in blocks.items()}
ttr = T[blocks["training"]]
T_train = float(np.median(ttr[ttr > 0]))
K, DELTA = 32, 0.05
x = float(np.log(2 * K / DELTA))
rep = {"seed": args.seed, "T_train": T_train, "K": K, "delta": DELTA, "dropped_rows": sorted(args.drop_rows),
       "blocks": {}}
for name, idx in blocks.items():
    t = np.clip(T[idx], 0, None)
    m, B = t.shape
    rows = {}
    for kind in ("floored", "cut"):
        for u in (1e-1, 1e-2, 1e-3):
            tau = u * T_train
            w = 1.0 / np.maximum(t, tau) ** 2
            if kind == "cut":
                w = np.where(t >= tau, w, 0.0)
            ent = float(w.mean() ** 2 / (w ** 2).mean())
            om = w.sum(1)
            states = float(om.mean() ** 2 / (om ** 2).mean())
            pi2 = ((w / np.where(om > 0, om, 1)[:, None]) ** 2).sum(1)
            band_factor = float((om ** 2).sum() / (B * (om ** 2 * pi2).sum()))
            band_tot = w.sum(0)
            order = np.argsort(band_tot)[::-1]
            k90 = int(np.searchsorted(np.cumsum(band_tot[order]) / band_tot.sum(), 0.9) + 1)
            r = {"entry_fraction": ent, "state_fraction": states, "band_factor": band_factor,
                 "identity_check": abs(ent - states * band_factor), "bands_for_90pct": k90,
                 "heaviest_bands": sorted(order[:8].tolist()),
                 "weight_share_below_floor": float(w[t < tau].sum() / w.sum())}
            if name == "validation":
                meff = m * states
                r["m_eff"] = meff
                r["max_over_mean"] = float(om.max() / om.mean())
                r["bound_over_L_Eomega"] = 2 * (np.sqrt(2 * x / meff) + 2 * r["max_over_mean"] * x / (3 * m))
            rows[f"{kind} u={u:g}"] = r
    blk = {"m": m, "B": B, "weights": rows}
    if name == "training":
        k = (t < 1e-3 * T_train).sum(1)
        blk["bands_below_1e-3_per_state"] = {"median": float(np.median(k)), "share_of_states_with_none": float((k == 0).mean())}
        blk["bands_more_than_half_below_1e-3"] = np.where((t < 1e-3 * T_train).mean(0) > 0.5)[0].tolist()
    if name == "validation":
        blk["bound_unweighted_over_L_Eomega"] = 2 * (np.sqrt(2 * x / m) + 2 * x / (3 * m))
    rep["blocks"][name] = blk

for name, blk in rep["blocks"].items():
    print(f"== {name} block: m = {blk['m']}")
    for k, r in blk["weights"].items():
        extra = (f"  m_eff {r['m_eff']:.0f}  max/mean {r['max_over_mean']:.1f}  bound {r['bound_over_L_Eomega']:.3f}"
                 if "m_eff" in r else "")
        print(f"  {k:14s} entry {r['entry_fraction']:.4f} = states {r['state_fraction']:.3f} x bands {r['band_factor']:.4f}"
              f"  (identity gap {r['identity_check']:.1e})  90% in {r['bands_for_90pct']} bands  below-floor share "
              f"{r['weight_share_below_floor']:.3f}{extra}")
tb, vb = rep["blocks"]["training"], rep["blocks"]["validation"]
print("training: bands below 1e-3 T per state", tb["bands_below_1e-3_per_state"], "; bands more than half below:",
      tb["bands_more_than_half_below_1e-3"])
print(f"validation: unweighted bound {vb['bound_unweighted_over_L_Eomega']:.3f}")
if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rep, open(args.out, "w", encoding="utf-8"), indent=1)
    print("wrote", args.out)
