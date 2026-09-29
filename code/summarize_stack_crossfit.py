"""Summarize the cross-fitted stacks over every run and held-out half.

For each run (one split at one width) and each half, the crossfit reports give the held-out scores of the component
stack (fitted on the recorded stack's objective), the retrieval stack (fitted on the first-order retrieval error), the
recorded stack and the six members; the frontier reports give, for the same halves, the stack whose objective was
chosen on the fitting half alone. Runs that appear in more than one report must agree, and are counted once.

usage: python code/summarize_stack_crossfit.py --crossfit <reports...> --frontier <reports...> [--out FILE]
"""
import argparse
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("--crossfit", nargs="+", required=True)
ap.add_argument("--frontier", nargs="+", required=True)
ap.add_argument("--out", default="")
args = ap.parse_args()
TAIL = "p95@0.001"


def agree(a, b, rel=1e-4):
    """Equal up to the solver's tolerance: across machines the scores agree to about 1e-8 and the QP weights to about
    1e-5 relative (split 109 at width 2000: one weight differs in its fifth digit), so 1e-4 separates rounding from
    a real difference."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(agree(a[k], b[k], rel) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(agree(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= rel * max(abs(a), abs(b), 1e-3)
    return a == b


def collect(paths):
    runs = {}
    for p in paths:
        for name, r in json.load(open(p, encoding="utf-8"))["runs"].items():
            if name in runs:
                if not agree(runs[name], r):
                    raise SystemExit(f"{name} differs between reports")
                continue
            runs[name] = r
    return runs


MEMBERS = ["ridge3", "krr", "ard", "dnn", "dnn_corr", "dkr"]     # the order the stacking scripts write weights in
members_of = {}
for p in args.crossfit:
    members_of.update({n: json.load(open(p, encoding="utf-8"))["members"] for n in json.load(open(p, encoding="utf-8"))["runs"]})
cross, front = collect(args.crossfit), collect(args.frontier)
for n in cross:
    cross[n] = dict(cross[n], members=members_of[n])
halves = []
for name in sorted(cross):
    if name not in front:
        continue
    m = re.match(r"(?:tq|lrtc)_s(\d+)(?:_raw)?_(w\d+)\.npz", name)
    for i, (hc, hf) in enumerate(zip(cross[name]["halves"], front[name]["halves"])):
        sc = hc["scores"]
        row = [r for r in hf["rows"] if r["alpha"] == hf["selected_alpha"]][0]
        mem = cross[name]["members"] if "members" in cross[name] else MEMBERS
        wts = hc["weights"]
        halves.append({"split": int(m.group(1)), "width": m.group(2), "half": i, "alpha": hf["selected_alpha"],
                       **{f"retrieval_w_dkr_{c}": wts["retrieval_stack"][c][mem.index("dkr")] for c in ("Y1", "Y2", "Y3")},
                       **{f"component_w_ard_{c}": wts["component_stack"][c][mem.index("ard")] for c in ("Y2", "Y3")},
                       **{f"{k}_{q}": sc[k][q] for k in ("component_stack", "retrieval_stack", "recorded_stack", "dkr")
                          for q in ("radiance_pct", TAIL)},
                       "chosen_radiance_pct": row["radiance_pct"], f"chosen_{TAIL}": row[TAIL]})
n = len(halves)


def rng(key):
    v = [h[key] for h in halves]
    return [round(min(v), 4), round(max(v), 4)]


def count(pred):
    return sum(1 for h in halves if pred(h))


summary = {
    "runs": sorted({(h["split"], h["width"]) for h in halves}), "halves": n,
    "range": {k: rng(k) for k in halves[0] if k.endswith(("radiance_pct", TAIL))},
    "component_tail_above_dkr": count(lambda h: h[f"component_stack_{TAIL}"] > h[f"dkr_{TAIL}"]),
    "component_radiance_below_dkr": count(lambda h: h["component_stack_radiance_pct"] < h["dkr_radiance_pct"]),
    "retrieval_tail_below_component": count(lambda h: h[f"retrieval_stack_{TAIL}"] < h[f"component_stack_{TAIL}"]),
    "retrieval_radiance_above_component": count(lambda h: h["retrieval_stack_radiance_pct"] > h["component_stack_radiance_pct"]),
    "chosen_radiance_below_dkr": count(lambda h: h["chosen_radiance_pct"] < h["dkr_radiance_pct"]),
    "chosen_tail_below_component": count(lambda h: h[f"chosen_{TAIL}"] < h[f"component_stack_{TAIL}"]),
    "chosen_tail_below_dkr": count(lambda h: h[f"chosen_{TAIL}"] < h[f"dkr_{TAIL}"]),
    "chosen_over_component_tail": [round(min(h[f"chosen_{TAIL}"] / h[f"component_stack_{TAIL}"] for h in halves), 3),
                                   round(max(h[f"chosen_{TAIL}"] / h[f"component_stack_{TAIL}"] for h in halves), 3)],
    "chosen_minus_dkr_tail": [round(min(h[f"chosen_{TAIL}"] - h[f"dkr_{TAIL}"] for h in halves), 3),
                              round(max(h[f"chosen_{TAIL}"] - h[f"dkr_{TAIL}"] for h in halves), 3)],
    "selected_alpha": sorted({h["alpha"] for h in halves}),
    "weights": {k: rng(k) for k in halves[0] if k.startswith(("retrieval_w_", "component_w_"))},
}
for k, v in summary.items():
    print(f"{k:38s} {v}")
if args.out:
    json.dump({"summary": summary, "halves": halves}, open(args.out, "w", encoding="utf-8"), indent=1)
    print("wrote", args.out)
