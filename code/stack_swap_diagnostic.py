"""Which component carries the stack's retrieval tail: the component stack of retrieval_stack_crossfit.py with the
weights of one component replaced by those of the retrieval stack, scored on the same evaluation half.

usage: EMIT_DATA=<dir> P2_REPO=<paper-2 repository> python stack_swap_diagnostic.py <crossfit.json> <preds.npz> ...
"""
import json
import os
import sys

import numpy as np

RHO = 0.7
COMPONENTS = ("Y1", "Y2", "Y3", "Y4")


def main():
    rep = json.load(open(sys.argv[1], encoding="utf-8"))
    sys.path.insert(0, os.path.join(os.environ["P2_REPO"], "code"))
    from conditioned_reflectance import evaluate
    D = os.environ["EMIT_DATA"]
    Y = {c: np.load(os.path.join(D, c + ".npy")) for c in COMPONENTS}
    members = rep["members"]
    for path in sys.argv[2:]:
        run = rep["runs"][os.path.basename(path)]
        z = np.load(path, allow_pickle=False)
        te = z["idx_te"]
        perm = np.random.RandomState(0).permutation(len(te))
        halves = (np.sort(perm[: len(te) // 2]), np.sort(perm[len(te) // 2:]))
        for h, half in enumerate(run["halves"]):
            ev = halves[1 - h]
            truth = {c: Y[c][te[ev]] for c in COMPONENTS}
            P = {m: {c: z[f"{m}_{c}"][ev] for c in COMPONENTS} for m in members}
            wc, wr = half["weights"]["component_stack"], half["weights"]["retrieval_stack"]
            out = {}
            for label, swap in (("component", ()), ("swap Y1", ("Y1",)), ("swap Y2+Y3", ("Y2", "Y3")),
                                ("swap Y4", ("Y4",)), ("swap all", COMPONENTS)):
                w = {c: (wr[c] if c in swap else wc[c]) for c in COMPONENTS}
                pr = {c: sum(w[c][j] * P[m][c] for j, m in enumerate(members)) for c in COMPONENTS}
                r = evaluate(truth, pr, flux_scale=run["flux_scale"], threshold=1e-3, rho=RHO)
                out[label] = round(100 * r["p95_absolute_error"], 3)
            print(os.path.basename(path)[:-4], "half", h, out, flush=True)


if __name__ == "__main__":
    main()
