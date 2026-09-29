"""Figures of the paper, from the committed results.

fig_weight: (a) the effective fraction n_eff/n of the floored and cut retrieval weights on the EMIT table against the
limits of the Proposition on the effective sample size, (1.10 and 0.13 times P(t <= tau) at beta = 0.12);
(b) the share of the floored weight that lies below the floor, and the share of the floored objective's value there at
the plain network's test residuals (split 101), for the path radiance and the transmission, with the limit (2 - beta)/2.
fig_frontier: the stacks of the family alpha F + (1 - alpha) G on one held-out half of two EMIT runs, radiance error
against the tail above 1e-3 T_train, with the component stack, the kernel on features and the stack chosen on the
fitting half.

usage: python code/plot_paper_figures.py [--out figures]
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "figures"
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#dcdcd8"
BETA = 0.12
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.linewidth": 0.8, "legend.frameon": False})


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---- fig_weight ----
ess = json.load(open("results/effective_sample_size.json", encoding="utf-8"))["floors"]
oot = json.load(open("results/p3rw_objective_on_test.json", encoding="utf-8"))["shares"]
us = sorted((float(u) for u in ess), reverse=True)
P = [ess[f"{u:g}" if f"{u:g}" in ess else str(u)]["P(t<=tau)"] for u in us]
fl = [ess[f"{u:g}" if f"{u:g}" in ess else str(u)]["floored"] for u in us]
ct = [ess[f"{u:g}" if f"{u:g}" in ess else str(u)]["cut"] for u in us]
lim_fl = (4 - BETA) / (2 - BETA) ** 2
lim_ct = BETA * (4 - BETA) / (2 - BETA) ** 2

fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
a.loglog(us, fl, "-o", color=BLUE, lw=2, ms=5, label="floored weights")
a.loglog(us, [lim_fl * p for p in P], "--", color=BLUE, lw=1.2, label=r"$\frac{4-\beta}{(2-\beta)^2}\,P(t\leq\tau)$")
a.loglog(us, ct, "-s", color=ORANGE, lw=2, ms=5, label="cut weights")
a.loglog(us, [lim_ct * p for p in P], "--", color=ORANGE, lw=1.2,
         label=r"$\frac{\beta(4-\beta)}{(2-\beta)^2}\,P(t\leq\tau)$")
a.set_xlabel(r"floor $u$ ($\tau=uT$)")
a.set_ylabel(r"effective fraction $n_{\rm eff}/n$")
a.invert_xaxis()
a.legend(fontsize=7, loc="lower left")
a.set_title("(a) effective sample size", fontsize=9, loc="left")
style(a)

u3 = [1e-1, 1e-2, 1e-3]
wb = [oot[f"u={u:g} a"]["weight_below"] for u in u3]
va = [oot[f"u={u:g} a"]["value_below"] for u in u3]
vt = [oot[f"u={u:g} t"]["value_below"] for u in u3]
b.semilogx(u3, wb, "-o", color=BLUE, lw=2, ms=5, label="weight")
b.semilogx(u3, va, "-s", color=ORANGE, lw=2, ms=5, label="value, path radiance")
b.semilogx(u3, vt, "-^", color=AQUA, lw=2, ms=6, label="value, transmission")
b.axhline((2 - BETA) / 2, color=MUTED, lw=1, ls=":")
b.text(3.2e-2, (2 - BETA) / 2 + 0.012, r"limit $(2-\beta)/2$ of the weight share", color=MUTED, fontsize=7,
       ha="center", va="bottom")
b.set_xlabel(r"floor $u$")
b.set_ylabel("share below the floor")
b.set_ylim(0, 1.05)
b.invert_xaxis()
b.legend(fontsize=7, loc="lower right")
b.set_title("(b) where the weight and the value sit", fontsize=9, loc="left")
style(b)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_weight.{ext}"), dpi=200)
plt.close(fig)

# ---- fig_frontier ----
F = json.load(open("results/stack_frontier_raw.json", encoding="utf-8"))
Cx = json.load(open("results/retrieval_stack_crossfit_raw.json", encoding="utf-8"))
runs = [("tq_s106_raw_w512.npz", "split 106, width 512"), ("tq_s107_raw_w2000.npz", "split 107, width 2000")]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
for ax, (run, title) in zip(axes, runs):
    half = F["runs"][run]["halves"][0]
    rows = half["rows"]
    ax.plot([r["radiance_pct"] for r in rows], [r["p95@0.001"] for r in rows], "-o", color=BLUE, lw=1.5, ms=3.5,
            label=r"stacks of the family, $\alpha$ from 0 to 1")
    for r in rows:
        if r["alpha"] in (0.0, 1.0):
            ax.annotate(rf"$\alpha={r['alpha']:g}$", (r["radiance_pct"], r["p95@0.001"]), fontsize=7, color=MUTED,
                        xytext=(4, 2), textcoords="offset points")
    chosen = min(rows, key=lambda r: abs(r["alpha"] - half["selected_alpha"]))
    ax.plot(chosen["radiance_pct"], chosen["p95@0.001"], "o", color=ORANGE, ms=8, mec="white", mew=1.2,
            label=r"chosen on the fitting half")
    ax.annotate(rf"$\alpha={half['selected_alpha']:g}$", (chosen["radiance_pct"], chosen["p95@0.001"]), fontsize=7,
                color=MUTED, xytext=(-6, -12), textcoords="offset points", ha="right")
    sc = Cx["runs"][run]["halves"][0]["scores"]
    ax.plot(sc["component_stack"]["radiance_pct"], sc["component_stack"]["p95@0.001"], "s", color=INK, ms=7,
            mfc="white", mew=1.3, label="component stack")
    ax.plot(sc["dkr"]["radiance_pct"], sc["dkr"]["p95@0.001"], "^", color=INK, ms=7, label="kernel on features")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("radiance error [%]")
    style(ax)
axes[0].set_ylabel(r"tail of $|\hat\rho-\rho|$ above $10^{-3}T$ [points]")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_frontier.{ext}"), dpi=200)
plt.close(fig)
print("wrote", os.path.join(OUT, "fig_weight.pdf"), os.path.join(OUT, "fig_frontier.pdf"))
