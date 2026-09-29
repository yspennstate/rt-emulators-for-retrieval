"""The networks table of the paper, from the run records: each arm against the plain network of the same run.

Two blocks: the EMIT table as provided (the pilot's records, retrieval_weighted_pilot.py, arms weighted_u<u> and
joint_u<u>) and the table without its two failed evaluations (p3_weight_family.py / _v2.py records, arms w1_u<u>,
mix<a>_u<u>, phys-w1, and the pilot run with --drop-rows for the joint arms). For every arm: the mean and sample
standard deviation over splits of the difference from the plain arm in the radiance error (percentage points) and in
the 95th percentiles of the retrieval error (points; all bands and the physical domain above 1e-12, 1e-3 and 1e-2
T_train), the number of splits on which the arm is lower, and the mean difference in the failure rate at 1e-12 and 1e-3.
Writes a JSON summary and the LaTeX rows.

usage: python code/make_network_table.py --block "as provided" DIR [DIR ...] --block "without the failed evaluations"
       DIR [...] [--arms ARM ...] [--out-json FILE] [--out-tex FILE] [--into PAPER.tex]

--into replaces the lines between "% network-table-rows: begin" and "% network-table-rows: end" in the paper.
"""
import glob
import json
import os
import statistics as st
import sys

args = sys.argv[1:]
blocks, arm_order, out_json, out_tex, into = [], None, None, None, None
i = 0
while i < len(args):
    a = args[i]
    if a == "--block":
        name, j = args[i + 1], i + 2
        dirs = []
        while j < len(args) and not args[j].startswith("--"):
            dirs.append(args[j])
            j += 1
        blocks.append((name, dirs))
        i = j
    elif a == "--arms":
        j = i + 1
        arm_order = []
        while j < len(args) and not args[j].startswith("--"):
            arm_order.append(args[j])
            j += 1
        i = j
    elif a == "--out-json":
        out_json, i = args[i + 1], i + 2
    elif a == "--out-tex":
        out_tex, i = args[i + 1], i + 2
    elif a == "--into":
        into, i = args[i + 1], i + 2
    else:
        sys.exit(f"unknown argument {a}")

METRICS = [("radiance", "radiance [pp]"), ("allband_p95", "all bands"), ("p95@1e-12", "phys. 1e-12"),
           ("p95@0.001", "phys. 1e-3"), ("p95@0.01", "phys. 1e-2")]
LABEL = {}
for u, tex in (("1000", "u\\to\\infty"), ("1", "u=1"), ("0.3", "u=0.3"), ("0.1", "u=10^{-1}"), ("0.03", "u=0.03"),
               ("0.01", "u=10^{-2}"), ("0.001", "u=10^{-3}")):
    LABEL[f"weighted_u{u}"] = LABEL[f"w1_u{u}"] = f"floored, ${tex}$"
    LABEL[f"joint_u{u}"] = f"joint, ${tex}$"
    LABEL[f"w1_u{u}_sp"] = f"floored, ${tex}$, plain stopping"
    LABEL[f"cut-w1_u{u}"] = f"cut, ${tex}$"
    LABEL[f"mix0.2_u{u}"] = f"mixed, $\\alpha=0.2$, ${tex}$"
    for k in ("0.25", "0.5"):   # the flattened weights of p3fc, bar w_tau^kappa renormalized to mean one
        LABEL[f"flat{k}_u{u}"] = f"flattened, $\\kappa={k}$, ${tex}$"
        LABEL[f"flat{k}_u{u}_sp"] = f"flattened, $\\kappa={k}$, ${tex}$, plain stopping"
LABEL["phys-w1"] = "physical squared error"


def records(dirs):
    out = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            try:
                r = json.load(open(f, encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(r, dict) and "arms" in r and "plain" in r.get("arms", {}) and "seed" in r:
                out.setdefault(r["seed"], {}).update({k: v for k, v in r["arms"].items() if k != "plain"})
                out[r["seed"]].setdefault("plain", r["arms"]["plain"])
    return out


def fmt(m, s, digits):
    return f"${m:+.{digits}f}\\pm{s:.{digits}f}$"


rep = {"blocks": {}}
tex = []
for name, dirs in blocks:
    recs = records(dirs)
    tex.append("\\multicolumn{7}{@{}l}{\\emph{" + name + "}}\\\\")
    arms = arm_order or sorted({k for r in recs.values() for k in r if k != "plain"})
    rows = {}
    for arm in arms:
        seeds = [s for s in sorted(recs) if arm in recs[s]]
        if not seeds:
            continue
        row = {"splits": seeds}
        for key, _ in METRICS:
            d = [recs[s][arm][key] - recs[s]["plain"][key] for s in seeds]
            row[key] = {"mean": st.mean(d), "sd": st.stdev(d) if len(d) > 1 else 0.0, "lower": sum(x < 0 for x in d)}
        for f in ("1e-12", "0.001"):
            k = f"failed_pct@{f}"
            d = [recs[s][arm][k] - recs[s]["plain"][k] for s in seeds]
            row[k] = {"mean": st.mean(d)}
        rows[arm] = row
        cells = []
        for key, _ in METRICS:
            v = row[key]
            cells.append(fmt(v["mean"], v["sd"], 2 if key == "radiance" else 1) + f" ({v['lower']})")
        cells.append(f"${row['failed_pct@0.001']['mean']:+.2f}$")
        tex.append(f"{LABEL.get(arm, arm)} & " + " & ".join(cells) + "\\\\")
    plain = {key: st.mean(recs[s]["plain"][key] for s in recs) for key, _ in METRICS}
    rep["blocks"][name] = {"splits": sorted(recs), "plain_mean": plain, "arms": rows}
    print(f"== {name}: {len(recs)} splits; plain means " + ", ".join(f"{k} {v:.3f}" for k, v in plain.items()))
    for arm, row in rows.items():
        print(f"   {arm:22s} n={len(row['splits']):2d} " + " ".join(
            f"{key.split('@')[-1]}:{row[key]['mean']:+.2f}({row[key]['lower']})" for key, _ in METRICS)
              + f"  dfail@1e-3 {row['failed_pct@0.001']['mean']:+.3f}")
    tex.append("\\midrule")
if out_json:
    json.dump(rep, open(out_json, "w", encoding="utf-8"), indent=1)
    print("wrote", out_json)
if out_tex:
    open(out_tex, "w", encoding="utf-8").write("\n".join(tex[:-1]) + "\n")
    print("wrote", out_tex)
if into:
    # the rows go straight into the paper between the two marker lines (a \multicolumn cannot follow \input in a cell)
    BEGIN, END = "% network-table-rows: begin", "% network-table-rows: end"
    lines = open(into, encoding="utf-8").read().split("\n")
    a, b = lines.index(BEGIN), lines.index(END)
    lines[a + 1:b] = tex[:-1]
    open(into, "w", encoding="utf-8").write("\n".join(lines))
    print("rows written into", into)
