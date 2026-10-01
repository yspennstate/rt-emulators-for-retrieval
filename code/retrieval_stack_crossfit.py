"""Stacking for the retrieval: convex weights chosen on the first-order retrieval error instead of the component error.

For one run of the EMIT pipeline (saved float64 test predictions of every family, results/target_quality/preds), the
test states are split in two halves by a fixed draw. On each half two stacks are fitted and then scored on the other
half, so that no state takes part in both the choice of the weights and their evaluation; the members were trained
without any test state.

  component stack   per component Y1..Y4, convex weights minimizing the mean over states of the relative L2 error of
                    that component across bands (the objective of the recorded stack, fitted here on the fitting half);
  retrieval stack   convex weights for the four components jointly minimizing sum (v^T e)^2 over the fitting entries,
                    e = (e_a, e_t, e_s) = (e_Y1, e_Y2 + e_Y3, e_Y4) and v = (q^2/t, rho q/t, rho^2) at rho = 0.7 with
                    t replaced by max(t, tau_fit T_train): the first-order retrieval error of the paper (Corollary 6.2).
                    The fitting entries are those of the physical domain with t >= tau_fit T_train.

Both stacks, their members and the recorded stack are scored on the evaluation half with the scorer of the main
pipeline (conditioned_reflectance.evaluate: physical domain, floors 1e-12, 1e-3, 1e-2 of the median positive training
flux) and by the mean relative L2 radiance error at rho = 0.7.

usage: EMIT_DATA=<dir> P2_REPO=<paper-2 repository> python retrieval_stack_crossfit.py <preds.npz> [...] --out FILE
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

RHO = 0.7
FLOORS = (1e-12, 1e-3, 1e-2)
COMPONENTS = ("Y1", "Y2", "Y3", "Y4")
MEMBERS = ("ridge3", "krr", "ard", "dnn", "dnn_corr", "dkr")


def simplex_qp(G, blocks):
    """Minimize w^T G w over w with each block of coordinates on the probability simplex."""
    n = G.shape[0]
    Gs = G / np.trace(G) * n
    w0 = np.concatenate([np.full(len(b), 1.0 / len(b)) for b in blocks])
    cons = [{"type": "eq", "fun": (lambda w, b=b: np.sum(w[b]) - 1.0), "jac": (lambda w, b=b: np.eye(n)[b].sum(0))}
            for b in blocks]
    res = minimize(lambda w: w @ Gs @ w, w0, jac=lambda w: 2 * Gs @ w, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n, constraints=cons, options={"maxiter": 500, "ftol": 1e-14})
    w = np.clip(res.x, 0, None)
    for b in blocks:
        w[b] /= w[b].sum()
    return w, bool(res.success)


def radiance_rel_l2(truth, pred):
    def rad(v):
        return v["Y1"] + RHO * (v["Y2"] + v["Y3"]) / (1 - RHO * v["Y4"])
    L, Lh = rad(truth), rad(pred)
    return float(np.mean(np.linalg.norm(Lh - L, axis=1) / np.linalg.norm(L, axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--tau-fit", type=float, default=1e-3)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--fit-floored", action="store_true",
                    help="fit the retrieval stack on every entry with 0 <= s < 1, t floored at tau_fit T_train in v (the "
                         "floored criterion), instead of on the entries with t >= tau_fit T_train (the cut criterion)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(os.environ["P2_REPO"], "code"))
    from conditioned_reflectance import evaluate, training_flux_scale
    D = os.environ["EMIT_DATA"]
    Y = {c: np.load(os.path.join(D, c + ".npy")) for c in COMPONENTS}
    report = {"rho": RHO, "tau_fit": args.tau_fit, "members": MEMBERS, "runs": {}}
    if args.fit_floored:
        report["fit"] = "floored: every entry with 0 <= s < 1, t floored at tau_fit T_train"
    for path in args.preds:
        z = np.load(path, allow_pickle=False)
        te, tr = z["idx_te"], z["idx_tr"]
        T = {c: Y[c][te] for c in COMPONENTS}
        scale = training_flux_scale({c: Y[c][tr] for c in COMPONENTS})
        P = {m: {c: z[f"{m}_{c}"] for c in COMPONENTS} for m in MEMBERS}
        rec = {c: z[f"stack_{c}"] for c in COMPONENTS}
        t, s = T["Y2"] + T["Y3"], T["Y4"]
        q = 1 - RHO * s
        tf = np.maximum(t, args.tau_fit * scale)
        v = (q * q / tf, RHO * q / tf, np.full_like(t, RHO * RHO))
        fit_mask = (t > 0) & (s >= 0) & (s < 1) & (t >= args.tau_fit * scale)
        if args.fit_floored:
            fit_mask = (s >= 0) & (s < 1)
        perm = np.random.RandomState(args.split_seed).permutation(len(te))
        halves = (np.sort(perm[: len(te) // 2]), np.sort(perm[len(te) // 2:]))
        out = {"flux_scale": scale, "halves": []}
        for h in (0, 1):
            fit, ev = halves[h], halves[1 - h]
            # component stack, one simplex per component, on the recorded stack's objective: the mean over states of
            # the relative L2 error of the component across bands
            wc = {}
            for c in COMPONENTS:
                Pk = [P[m][c][fit] for m in MEMBERS]
                Tn = np.linalg.norm(T[c][fit], axis=1)

                def obj(w, Pk=Pk, Tc=T[c][fit], Tn=Tn):
                    return float(np.mean(np.linalg.norm(sum(wi * Pi for wi, Pi in zip(w, Pk)) - Tc, axis=1) / Tn))
                k0 = len(MEMBERS)
                res = minimize(obj, np.ones(k0) / k0, bounds=[(0, 1)] * k0, method="SLSQP",
                               constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                               options=dict(maxiter=300, ftol=1e-12))
                w = np.maximum(res.x, 0)
                wc[c] = w / w.sum()
            # retrieval stack, four simplices coupled through v
            mk = fit_mask[fit].ravel()
            cols = []
            for c, vc in (("Y1", v[0]), ("Y2", v[1]), ("Y3", v[1]), ("Y4", v[2])):
                for m in MEMBERS:
                    cols.append(((P[m][c][fit] - T[c][fit]) * vc[fit]).ravel()[mk])
            A = np.stack(cols, 1)
            k = len(MEMBERS)
            wr_all, ok = simplex_qp(A.T @ A, [np.arange(i * k, (i + 1) * k) for i in range(4)])
            wr = {c: wr_all[i * k:(i + 1) * k] for i, c in enumerate(COMPONENTS)}
            combos = {"component_stack": wc, "retrieval_stack": wr}
            preds = {name: {c: sum(w[c][j] * P[m][c][ev] for j, m in enumerate(MEMBERS)) for c in COMPONENTS}
                     for name, w in combos.items()}
            preds["recorded_stack"] = {c: rec[c][ev] for c in COMPONENTS}
            for m in MEMBERS:
                preds[m] = {c: P[m][c][ev] for c in COMPONENTS}
            truth = {c: T[c][ev] for c in COMPONENTS}
            scores = {}
            for name, pr in preds.items():
                row = {"radiance_pct": 100 * radiance_rel_l2(truth, pr)}
                for fl in FLOORS:
                    r = evaluate(truth, pr, flux_scale=scale, threshold=fl, rho=RHO)
                    row[f"p95@{fl:g}"] = None if r["p95_absolute_error"] is None else 100 * r["p95_absolute_error"]
                    row[f"failed_pct@{fl:g}"] = 100 * r["failed_inversions"] / max(r["retained_entries"], 1)
                scores[name] = row
            out["halves"].append({"fit_states": int(len(fit)), "eval_states": int(len(ev)), "qp_converged": ok,
                                  "weights": {n: {c: [round(float(x), 6) for x in w[c]] for c in COMPONENTS}
                                              for n, w in combos.items()},
                                  "scores": scores})
            print(os.path.basename(path), "half", h, {n: (round(sc["radiance_pct"], 4), round(sc["p95@0.001"], 3))
                                                      for n, sc in scores.items()}, flush=True)
        report["runs"][os.path.basename(path)] = out
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
