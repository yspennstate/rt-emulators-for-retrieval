"""The floored objective on the test block: where its value sits, and whether the network trained on it lowers it.

For the plain arm of one lane (--shares-tag) the script reports the share of the floored objective
    sum over test entries of  w e^2,   w = 1 / max(t, u T_train)^2
that comes from the entries below the floor and from the opaque, absorbing and window bands (median training
transmission below 1e-3, between 1e-3 and 1e-1, above 1e-1 T_train), separately for the path radiance (e = e_a) and the
transmission (e = e_2 + e_3), next to the share of the weight itself. For every lane (a directory holding
pred_s<seed>_plain<tag>.npz and the arms of the same run, pred_s<seed>_<arm><tag>.npz, written by the pilot) and every
weighted or joint arm with floor u, it then reports the ratio of the arm's value of the pilot's objective J_u on the test
block,
    J_u = mean of (e_a^2 + R^2 (e_2^2 + e_3^2) + R^4 t^2 e_s^2) / max(t, u T_train)^2,
to the plain arm's value in the same lane, over the entries below the floor, above it, and all of them. For a weighted
arm J_u is its own training objective. For a joint arm it is not: the joint loss is the linearized retrieval error, in
which the component errors can cancel, while J_u is the separable bound that allows no cancellation; its ratio says how
the joint arm does on that bound, not whether it lowered its own loss.

usage: python code/p3rw_objective_on_test.py <dir with lanes> --data <arrays dir> [--shares-tag p3rwB_s101]
       [--out results/p3rw_objective_on_test.json]
"""
import glob
import json
import os
import re
import statistics as st
import sys

import numpy as np

args = sys.argv[1:]
opts = {"--out": "results/p3rw_objective_on_test.json", "--data": os.environ.get("EMIT_DATA", ""), "--shares-tag": ""}
for k in list(opts):
    if k in args:
        i = args.index(k)
        opts[k] = args[i + 1]
        del args[i:i + 2]
ROOT = args[0]
D = opts["--data"]
Ys = {c: np.load(os.path.join(D, c + ".npy")) for c in ("Y1", "Y2", "Y3", "Y4")}
T = Ys["Y2"] + Ys["Y3"]
R = 0.9


def split(seed):
    n_all = T.shape[0]
    perm = np.random.RandomState(seed).permutation(n_all)
    n_te = int(round(0.1 * n_all))
    idx_te, tr_full = perm[:n_te], perm[n_te:]
    vperm = np.random.RandomState(seed + 10000).permutation(len(tr_full))
    n_val = int(round(0.1 * len(tr_full)))
    idx_tr = tr_full[vperm[n_val:]]
    ttr = T[idx_tr]
    return idx_te, idx_tr, float(np.median(ttr[ttr > 0]))


PLAIN = re.compile(r"pred_s(\d+)_plain(.*)\.npz$")
lanes = []
for f in sorted(glob.glob(os.path.join(ROOT, "**", "pred_s*_plain*.npz"), recursive=True)):
    m = PLAIN.search(os.path.basename(f))
    seed, tag = int(m.group(1)), m.group(2)
    arms = {}
    for g in sorted(glob.glob(os.path.join(os.path.dirname(f), f"pred_s{seed}_*{tag}.npz"))):
        arm = os.path.basename(g)[len(f"pred_s{seed}_"):-len(tag + ".npz")]
        if arm.startswith(("weighted_u", "joint_u")):
            arms[arm] = g
    lanes.append((seed, tag, f, arms))
if not lanes:
    sys.exit("no lanes found")

rep = {"data": D, "shares": {}, "ratios": []}
# where the value sits, at the plain arm's residuals of one lane
sh = [ln for ln in lanes if ln[1] == opts["--shares-tag"]] or lanes[:1]
seed, tag, fplain, _ = sh[0]
rep["shares_lane"] = tag
idx_te, idx_tr, T_train = split(seed)
z0 = np.load(fplain)
if not np.array_equal(z0["idx_te"], idx_te):
    sys.exit(f"{fplain}: test rows differ from the split")
med = np.median(np.clip(T[idx_tr], 0, None), axis=0) / T_train
groups = {"opaque": med < 1e-3, "absorbing": (med >= 1e-3) & (med < 1e-1), "window": med >= 1e-1}
t = np.clip(T[idx_te], 0, None)
for u in (1e-1, 1e-2, 1e-3):
    tau = u * T_train
    w = 1.0 / np.maximum(t, tau) ** 2
    for name, e2 in (("a", (z0["Y1"] - Ys["Y1"][idx_te]) ** 2), ("t", (z0["Y2"] + z0["Y3"] - t) ** 2)):
        L = w * e2
        rep["shares"][f"u={u:g} {name}"] = {
            "weight_below": float(w[t < tau].sum() / w.sum()), "value_below": float(L[t < tau].sum() / L.sum()),
            "weight_by_group": {g: float(w[:, m].sum() / w.sum()) for g, m in groups.items()},
            "value_by_group": {g: float(L[:, m].sum() / L.sum()) if m.any() else None for g, m in groups.items()}}
rep["bands_per_group"] = {g: int(m.sum()) for g, m in groups.items()}

# does the arm trained on J_u lower J_u on the test block?
for seed, tag, fplain, arms in lanes:
    idx_te, idx_tr, T_train = split(seed)
    t = np.clip(T[idx_te], 0, None)
    z0 = np.load(fplain)
    if not np.array_equal(z0["idx_te"], idx_te):
        sys.exit(f"{fplain}: test rows differ from the split")
    for arm, g in arms.items():
        za = np.load(g)
        tau = float(arm.split("_u")[1]) * T_train
        den = np.maximum(t, tau) ** 2

        def J(z, mask):
            e = {c: z[c] - Ys[c][idx_te] for c in ("Y1", "Y2", "Y3", "Y4")}
            L = (e["Y1"] ** 2 + R ** 2 * (e["Y2"] ** 2 + e["Y3"] ** 2) + R ** 4 * t ** 2 * e["Y4"] ** 2) / den
            return float(L[mask].mean())
        below, allm = t < tau, np.ones_like(t, dtype=bool)
        row = {"seed": seed, "lane": tag, "arm": arm, "all": J(za, allm) / J(z0, allm)}
        if below.any():
            row["below"] = J(za, below) / J(z0, below)
        if (~below).any():
            row["above"] = J(za, ~below) / J(z0, ~below)
        rep["ratios"].append(row)

for k, v in rep["shares"].items():
    print(f"{k}: weight below {v['weight_below']:.3f}, value below {v['value_below']:.3f};  value by group "
          + ", ".join(f"{g} {x:.3f}" for g, x in v["value_by_group"].items() if x is not None))
rep["ratio_summary"] = {}
for arm in sorted({x["arm"] for x in rep["ratios"]}):
    r = [x for x in rep["ratios"] if x["arm"] == arm]
    s = {k: {"median": st.median(x[k] for x in r if k in x), "min": min(x[k] for x in r if k in x),
             "max": max(x[k] for x in r if k in x)} for k in ("below", "above", "all") if any(k in x for x in r)}
    s["arm_lower_on"] = sum(x["all"] < 1 for x in r)
    s["n"] = len(r)
    rep["ratio_summary"][arm] = s
    print(f"{arm}: J_arm/J_plain on test, median [min, max]: "
          + ("below %.3f [%.3f, %.3f], " % (s["below"]["median"], s["below"]["min"], s["below"]["max"]) if "below" in s else "")
          + ("above %.2f, " % s["above"]["median"] if "above" in s else "")
          + f"all {s['all']['median']:.2f}; arm lower on {s['arm_lower_on']}/{s['n']}")
os.makedirs(os.path.dirname(opts["--out"]) or ".", exist_ok=True)
json.dump(rep, open(opts["--out"], "w", encoding="utf-8"), indent=1)
print("wrote", opts["--out"])
