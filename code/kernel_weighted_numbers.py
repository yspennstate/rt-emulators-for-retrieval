"""Numbers for the kernels fitted on the weighted error (p3_krr_weighted.py records, lanes p3kw_s101..s110).

Every weighted arm is paired with the plain fit of its own run (same split, same 6,000 fitted rows, same kernel
parameters): for each score, the median of the paired differences over the splits, their range, and on how many
splits the arm is lower; for each arm, its own objective on the test block against the plain fit's (median ratio and
the number of splits on which the arm attains it better).

usage: python code/kernel_weighted_numbers.py results/dgx/p3kw [--out notes/kernel_weighted_numbers_10splits.md]
"""
import glob
import json
import os
import statistics as st
import sys

d = sys.argv[1]
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
recs = {}
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    r = json.load(open(f, encoding="utf-8"))
    if r.get("kind") == "p3_krr_weighted":
        recs[r["seed"]] = r
if not recs:
    sys.exit(f"no p3_krr_weighted records in {d}")
S = sorted(recs)
SCORES = [("radiance", "radiance error (pp)", 3), ("allband_p95", "all-band tail", 2), ("p95@1e-12", "tail, physical 1e-12", 2),
          ("p95@0.001", "tail, physical 1e-3", 2), ("p95@0.01", "tail, physical 1e-2", 2),
          ("failed_pct@0.001", "failed inversions above 1e-3 (%)", 3)]
arms = [a for a in recs[S[0]]["arms"] if a != "plain"]
lines = [f"# Kernel ridge regression fitted on the weighted error: {len(S)} splits ({', '.join(map(str, S))})", ""]
first = recs[S[0]]
lines.append(f"Fitted rows {first['n_fit']} of {first['n_train']}; validation {first['n_val']}, test {first['n_test']}; "
             f"dropped rows {sorted(first['dropped_rows'])}.")
shas = {json.dumps(r["data_sha256"], sort_keys=True) for r in recs.values()}
drivers = {r["driver_sha256"][:16] for r in recs.values()}
lines.append(f"Data fingerprints identical across runs: {len(shas) == 1}; driver sha256 {sorted(drivers)}.")
mins = [r["minutes"] for r in recs.values()]
lines.append(f"Minutes per run: median {st.median(mins):.1f}, range {min(mins):.1f}-{max(mins):.1f}.")
hp = {c: sorted({(recs[s]['hp'][c]['nu'], recs[s]['hp'][c]['scale'], recs[s]['hp'][c]['nugget']) for s in S}) for c in ("Y1", "Y2", "Y3", "Y4")}
lines.append("Kernel parameters chosen on the forward criterion (nu, scale, nugget) per component: "
             + "; ".join(f"{c} {v}" for c, v in hp.items()) + ".")
P = {s: recs[s]["arms"]["plain"] for s in S}
lines += ["", "## The plain fit", ""]
for k, name, dg in SCORES:
    v = [P[s][k] for s in S]
    lines.append(f"- {name}: median {st.median(v):.{dg}f}, range {min(v):.{dg}f} to {max(v):.{dg}f}")
for arm in arms:
    A = {s: recs[s]["arms"][arm] for s in S}
    kind, u = A[S[0]]["weights"], A[S[0]]["floor"]
    own = None if kind == "flattened" else ("J" if kind == "floored" else "K") + f"{u:g}"
    kap = f", kappa = {A[S[0]]['kappa']:g}" if kind == "flattened" else ""
    lines += ["", f"## {arm} (weights {kind}{kap}, u = {u:g}) against the plain fit", ""]
    for k, name, dg in SCORES:
        diff = [A[s][k] - P[s][k] for s in S]
        rel = [A[s][k] / P[s][k] for s in S if P[s][k] > 0]
        lines.append(f"- {name}: median change {st.median(diff):+.{dg}f} (range {min(diff):+.{dg}f} to {max(diff):+.{dg}f}), "
                     f"median ratio {st.median(rel):.3f}, lower on {sum(x < 0 for x in diff)} of {len(S)}")
    for ob in A[S[0]]["test_obj"]:
        ra = [sum(A[s]["test_obj"][ob].values()) / sum(P[s]["test_obj"][ob].values()) for s in S]
        per = {c: st.median([A[s]["test_obj"][ob][c] / P[s]["test_obj"][ob][c] for s in S]) for c in ("Y1", "Y2", "Y3", "Y4")}
        tag = " (its own objective)" if ob == own else ""
        lines.append(f"- test objective {ob}{tag}: summed over components, median ratio {st.median(ra):.3f}, "
                     f"lower on {sum(x < 1 for x in ra)} of {len(S)}; per component median ratios "
                     + ", ".join(f"{c} {v:.3f}" for c, v in per.items()))
text = "\n".join(lines) + "\n"
print(text)
if out:
    open(out, "w", encoding="utf-8").write(text)
