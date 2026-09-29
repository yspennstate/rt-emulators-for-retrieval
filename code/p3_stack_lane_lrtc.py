"""One run: stacking for the retrieval on the libRadtran runs of the second paper (sixteen inputs).

Same reports as p3_stack_lane.py, for the prediction files ~/p23/results_lrtc/preds/lrtc_s<seed>_w512.npz, with the
thirteen-band libRadtran arrays in ~/p23/data_lrt13cats; reports go to ~/p23/results_p3stack with the suffix lrtc, and
the runner's completion record ~/p23/results/<tag>.json lists the files used and any that were missing.

usage (runner line): 4|2|0|<tag>|p3_stack_lane_lrtc.py --tag <tag> --seeds 103 104 105 106 107 108 109 110
"""
import argparse
import json
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--seeds", type=int, nargs="+", required=True)
args = ap.parse_args()

HOME = os.path.expanduser("~")
PREDS = os.path.join(HOME, "p23", "results_lrtc", "preds")
OUT = os.path.join(HOME, "p23", "results_p3stack")
RECORD = os.path.join(HOME, "p23", "results", f"{args.tag}.json")
HERE = os.path.dirname(os.path.abspath(__file__))

os.makedirs(OUT, exist_ok=True)
wanted = [os.path.join(PREDS, f"lrtc_s{s}_w512.npz") for s in args.seeds]
files = [f for f in wanted if os.path.exists(f)]
missing = [os.path.basename(f) for f in wanted if not os.path.exists(f)]
if not files:
    sys.exit("no prediction files")
env = dict(os.environ, P2_REPO=os.path.join(HOME, "p23", "tq_repo_1f5d4df"),
           EMIT_DATA=os.path.join(HOME, "p23", "data_lrt13cats"),
           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
outputs = []
for script, out, extra in (("retrieval_stack_crossfit.py", "crossfit_lrtc.json", []),
                           ("stack_frontier_crossfit.py", "frontier_lrtc.json", [])):
    cmd = [sys.executable, "-u", os.path.join(HERE, script)] + files + extra + ["--out", os.path.join(OUT, out)]
    print("running", script, " ".join(extra), flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc:
        sys.exit(rc)
    outputs.append(out)
with open(RECORD, "w", encoding="utf-8") as f:
    json.dump({"tag": args.tag, "files": [os.path.basename(x) for x in files], "missing": missing, "outputs": outputs}, f,
              indent=1)
print("done", len(files), "files;", "missing:", missing, flush=True)
