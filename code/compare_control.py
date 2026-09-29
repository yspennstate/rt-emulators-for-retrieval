"""Does the clean-table driver reproduce the pilot on the table as provided?

p3cd runs p3_weight_family.py (the driver of the clean-table lanes p3cl) WITHOUT --drop-rows on splits 101 and 105.
If its plain and floored arms equal the pilot's (retrieval_weighted_pilot.py, the Kaggle lanes p3rwA/p3rwB) on the same
splits, the upper and lower blocks of Table 1 differ only in the two failed evaluations; if they do not, the driver is
a second difference and the lower block cannot be read as the effect of the two states alone.

usage: python code/compare_control.py [results/dgx/p3cd] [--pilot DIR]
"""
import glob
import json
import os
import sys

args = sys.argv[1:]
pilot = os.environ.get("P3_PILOT", "results/pilot")
if "--pilot" in args:
    i = args.index("--pilot")
    pilot = args[i + 1]
    del args[i:i + 2]
ctrl_dir = args[0] if args else "results/dgx/p3cd"
prov = {}
for f in glob.glob(os.path.join(pilot, "**", "*.json"), recursive=True):
    try:
        r = json.load(open(f, encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        continue
    if isinstance(r, dict) and "arms" in r and "seed" in r and not r.get("dropped_rows"):
        prov.setdefault(r["seed"], {}).update(r["arms"])
PAIRS = [("plain", "plain"), ("w1_u0.1", "weighted_u0.1"), ("w1_u0.01", "weighted_u0.01"),
         ("w1_u0.001", "weighted_u0.001")]
KEYS = ["radiance", "allband_p95", "p95@1e-12", "p95@0.001", "p95@0.01"]
worst = 0.0
for f in sorted(glob.glob(os.path.join(ctrl_dir, "*.json"))):
    c = json.load(open(f, encoding="utf-8"))
    s = c["seed"]
    print(f"split {s}: control dropped_rows={c.get('dropped_rows')}, driver {str(c.get('driver_sha256'))[:12]}")
    for ca, pa in PAIRS:
        if ca not in c["arms"] or pa not in prov.get(s, {}):
            print(f"   {ca}: missing ({ca in c['arms']}, {pa in prov.get(s, {})})")
            continue
        x, y = c["arms"][ca], prov[s][pa]
        d = {k: x[k] - y[k] for k in KEYS}
        rel = max(abs(d[k]) / max(abs(y[k]), 1e-12) for k in KEYS)
        worst = max(worst, rel)
        print(f"   {ca:10s} vs {pa:15s} " + " ".join(f"{k} {x[k]:.4f}/{y[k]:.4f}" for k in KEYS)
              + f"  max rel diff {rel:.2e}")
print(f"largest relative difference over all arms and scores: {worst:.2e}")
