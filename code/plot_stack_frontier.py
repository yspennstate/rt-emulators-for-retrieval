"""The forward-retrieval frontier of the stacks, one panel per run, from stack_frontier_crossfit.py and
retrieval_stack_crossfit.py. Radiance relative L2 error against the 95th percentile of the retrieval error on the
physical domain at the floor 1e-3, both on the held-out half.

usage: python plot_stack_frontier.py results/stack_frontier_raw.json results/retrieval_stack_crossfit_raw.json OUT.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = json.load(open(sys.argv[1], encoding="utf-8"))
C = json.load(open(sys.argv[2], encoding="utf-8"))
runs = list(F["runs"])
fig, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 3.3), sharey=False)
for ax, run in zip(axes, runs):
    for h, half in enumerate(F["runs"][run]["halves"]):
        rows = half["rows"]
        col = ("C0", "C1")[h]
        ax.plot([r["radiance_pct"] for r in rows], [r["p95@0.001"] for r in rows], "-o", ms=2.5, lw=1, color=col,
                label=f"stacks, half {h + 1}")
        for r in rows:
            if r["alpha"] in (0.0, 0.95, 1.0):
                ax.annotate(f"{r['alpha']:g}", (r["radiance_pct"], r["p95@0.001"]), fontsize=6, color=col,
                            xytext=(3, 2), textcoords="offset points")
        sc = C["runs"][run]["halves"][h]["scores"]
        ax.plot(sc["component_stack"]["radiance_pct"], sc["component_stack"]["p95@0.001"], "s", color=col, ms=5,
                mfc="none", label="component stack" if h == 0 else None)
        ax.plot(sc["dkr"]["radiance_pct"], sc["dkr"]["p95@0.001"], "^", color=col, ms=5,
                label="kernel on features" if h == 0 else None)
    ax.set_title(run.replace("tq_", "").replace("_raw", "").replace(".npz", ""), fontsize=9)
    ax.set_xlabel("radiance rel. $L^2$ [%]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
axes[0].set_ylabel(r"$p_{95}$ of $|\hat\rho-\rho|$ [pp], $\tau_0=10^{-3}$", fontsize=8)
axes[0].legend(fontsize=6, frameon=False)
fig.tight_layout()
fig.savefig(sys.argv[3], dpi=160)
