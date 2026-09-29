"""The trade-off between forward and retrieval error across stacking objectives.

At rho = 0.7 the first-order retrieval error of an entry is d = v^T e and its forward error is r = (t/q^2) d, so the
squared forward error is the squared retrieval error weighted by g = t^2/q^4 (the note, Proposition on the change of
measure). The stacks here minimize
    sum over fitting entries of  ((1 - alpha) + alpha g / mean(g)) d^2,
alpha = 0 being the first-order retrieval error and alpha = 1 the squared forward error at rho = 0.7, with t replaced by
max(t, tau_fit T_train) in v and g. The halves, members, fitting mask and scoring are those of
retrieval_stack_crossfit.py.

usage: EMIT_DATA=<dir> P2_REPO=<paper-2 repository> python stack_frontier_crossfit.py <preds.npz> [...] --out FILE
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_stack_crossfit import COMPONENTS, FLOORS, MEMBERS, RHO, radiance_rel_l2, simplex_qp  # noqa: E402

ALPHAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--tau-fit", type=float, default=1e-3)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(os.environ["P2_REPO"], "code"))
    from conditioned_reflectance import evaluate, training_flux_scale
    D = os.environ["EMIT_DATA"]
    Y = {c: np.load(os.path.join(D, c + ".npy")) for c in COMPONENTS}
    report = {"rho": RHO, "tau_fit": args.tau_fit, "alphas": ALPHAS, "members": MEMBERS, "runs": {}}
    for path in args.preds:
        z = np.load(path, allow_pickle=False)
        te, tr = z["idx_te"], z["idx_tr"]
        T = {c: Y[c][te] for c in COMPONENTS}
        scale = training_flux_scale({c: Y[c][tr] for c in COMPONENTS})
        P = {m: {c: z[f"{m}_{c}"] for c in COMPONENTS} for m in MEMBERS}
        t, s = T["Y2"] + T["Y3"], T["Y4"]
        q = 1 - RHO * s
        tf = np.maximum(t, args.tau_fit * scale)
        v = (q * q / tf, RHO * q / tf, np.full_like(t, RHO * RHO))
        g = tf * tf / q ** 4
        fit_mask = (t > 0) & (s >= 0) & (s < 1) & (t >= args.tau_fit * scale)
        perm = np.random.RandomState(args.split_seed).permutation(len(te))
        halves = (np.sort(perm[: len(te) // 2]), np.sort(perm[len(te) // 2:]))
        out = {"halves": []}
        for h in (0, 1):
            fit, ev = halves[h], halves[1 - h]
            mk = fit_mask[fit].ravel()
            gf = g[fit].ravel()[mk]
            cols = []
            for c, vc in (("Y1", v[0]), ("Y2", v[1]), ("Y3", v[1]), ("Y4", v[2])):
                for m in MEMBERS:
                    cols.append(((P[m][c][fit] - T[c][fit]) * vc[fit]).ravel()[mk])
            A = np.stack(cols, 1)
            k = len(MEMBERS)
            truth = {c: T[c][ev] for c in COMPONENTS}
            truth_fit = {c: T[c][fit] for c in COMPONENTS}
            # the best single member's tail on the fitting half, the bound of the selection rule below
            member_fit_p95 = {m: 100 * evaluate(truth_fit, {c: P[m][c][fit] for c in COMPONENTS}, flux_scale=scale,
                                                threshold=1e-3, rho=RHO)["p95_absolute_error"] for m in MEMBERS}
            rows = []
            for alpha in ALPHAS:
                wgt = (1 - alpha) + alpha * gf / gf.mean()
                w, ok = simplex_qp(A.T @ (A * wgt[:, None]), [np.arange(i * k, (i + 1) * k) for i in range(4)])
                pr = {c: sum(w[i * k + j] * P[m][c][ev] for j, m in enumerate(MEMBERS)) for i, c in enumerate(COMPONENTS)}
                pf = {c: sum(w[i * k + j] * P[m][c][fit] for j, m in enumerate(MEMBERS)) for i, c in enumerate(COMPONENTS)}
                row = {"alpha": alpha, "qp_converged": ok, "radiance_pct": 100 * radiance_rel_l2(truth, pr),
                       "fit_radiance_pct": 100 * radiance_rel_l2(truth_fit, pf),
                       "fit_p95@0.001": 100 * evaluate(truth_fit, pf, flux_scale=scale, threshold=1e-3,
                                                       rho=RHO)["p95_absolute_error"],
                       "weights": {c: [round(float(x), 4) for x in w[i * k:(i + 1) * k]] for i, c in enumerate(COMPONENTS)}}
                for fl in FLOORS:
                    r = evaluate(truth, pr, flux_scale=scale, threshold=fl, rho=RHO)
                    row[f"p95@{fl:g}"] = None if r["p95_absolute_error"] is None else 100 * r["p95_absolute_error"]
                rows.append(row)
                print(os.path.basename(path)[:-4], "half", h, "alpha", alpha,
                      round(row["radiance_pct"], 4), round(row["p95@0.001"], 3), flush=True)
            # selection on the fitting half only: the lowest radiance error among the objectives whose tail is no
            # larger than that of the best single member, both measured on the fitting half
            bound = min(member_fit_p95.values())
            ok_rows = [r for r in rows if r["fit_p95@0.001"] <= bound] or [min(rows, key=lambda r: r["fit_p95@0.001"])]
            chosen = min(ok_rows, key=lambda r: r["fit_radiance_pct"])
            out["halves"].append({"rows": rows, "member_fit_p95@0.001": member_fit_p95, "selected_alpha": chosen["alpha"]})
        report["runs"][os.path.basename(path)] = out
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
