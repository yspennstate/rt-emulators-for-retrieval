"""One run: stacking for the retrieval on the raw-arm predictions of the given splits, both widths.

The same three reports as p3_stack_lane.py (two cross-fit split seeds and the objective frontier), for prediction
files named tq_s<seed>_raw_<width>.npz in --preds, written to ~/p23/results_p3stack with the suffix of the seeds; the
runner's completion record ~/p23/results/<tag>.json lists the files used and any that were missing.

usage (runner line): 4|2|0|<tag>|p3_stack_lane_seeds.py --tag <tag> --seeds 109 110 --preds <dir>
"""
import argparse
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--preds", required=True)
ap.add_argument("--arm", default="raw", help="training arm of the member files tq_s<seed>_<arm>_<width>.npz "
                                             "(raw, or dropfailed for the members refitted without the failed states)")
ap.add_argument("--wait-hours", type=float, default=0.0,
                help="wait up to this long, checking every five minutes, for every member file to exist (the runner "
                     "can start this line while the member lanes queued before it are still running)")
args = ap.parse_args()

HOME = os.path.expanduser("~")
OUT = os.path.join(HOME, "p23", "results_p3stack")
RECORD = os.path.join(HOME, "p23", "results", f"{args.tag}.json")
HERE = os.path.dirname(os.path.abspath(__file__))
suffix = "s" + "_s".join(str(s) for s in args.seeds)

os.makedirs(OUT, exist_ok=True)
wanted = [os.path.join(args.preds, f"tq_s{s}_{args.arm}_{c}.npz") for s in args.seeds for c in ("w512", "w2000")]
deadline = time.time() + 3600 * args.wait_hours
while not all(os.path.exists(f) for f in wanted) and time.time() < deadline:
    time.sleep(300)
files = [f for f in wanted if os.path.exists(f)]
missing = [os.path.basename(f) for f in wanted if not os.path.exists(f)]
if not files:
    sys.exit("no prediction files")
env = dict(os.environ, P2_REPO=os.path.join(HOME, "p23", "tq_repo_1f5d4df"), EMIT_DATA=os.path.join(HOME, "p2", "data", "emit"),
           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
outputs = []
for script, out, extra in (("retrieval_stack_crossfit.py", f"crossfit_{args.arm}_{suffix}.json", []),
                           ("retrieval_stack_crossfit.py", f"crossfit_{args.arm}_split1_{suffix}.json", ["--split-seed", "1"]),
                           ("stack_frontier_crossfit.py", f"frontier_{args.arm}_{suffix}.json", [])):
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
