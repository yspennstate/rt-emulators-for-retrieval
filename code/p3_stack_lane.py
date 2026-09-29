"""One run: stacking for the retrieval on the raw-arm predictions of splits 101-105 at both widths.

Runs retrieval_stack_crossfit.py at two split seeds and stack_frontier_crossfit.py on every prediction file that exists
under ~/p23/results_tq/preds, with the scorer of the training-target experiment (~/p23/tq_repo_1f5d4df/code) and the
EMIT arrays in ~/p2/data/emit, writes the three reports to ~/p23/results_p3stack and, last, the runner's completion
record ~/p23/results/p3stack_dgx.json listing the files used and any that were missing.

usage (runner line): 4|2|0|p3stack_dgx|p3_stack_lane.py
"""
import json
import os
import subprocess
import sys

HOME = os.path.expanduser("~")
PREDS = os.path.join(HOME, "p23", "results_tq", "preds")
OUT = os.path.join(HOME, "p23", "results_p3stack")
RECORD = os.path.join(HOME, "p23", "results", "p3stack_dgx.json")
HERE = os.path.dirname(os.path.abspath(__file__))

os.makedirs(OUT, exist_ok=True)
wanted = [os.path.join(PREDS, f"tq_s{s}_raw_{c}.npz") for s in range(101, 106) for c in ("w512", "w2000")]
files = [f for f in wanted if os.path.exists(f)]
missing = [os.path.basename(f) for f in wanted if not os.path.exists(f)]
if not files:
    sys.exit("no prediction files")
env = dict(os.environ, P2_REPO=os.path.join(HOME, "p23", "tq_repo_1f5d4df"), EMIT_DATA=os.path.join(HOME, "p2", "data", "emit"),
           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
for script, out, extra in (("retrieval_stack_crossfit.py", "crossfit_raw.json", []),
                           ("retrieval_stack_crossfit.py", "crossfit_raw_split1.json", ["--split-seed", "1"]),
                           ("stack_frontier_crossfit.py", "frontier_raw.json", [])):
    cmd = [sys.executable, "-u", os.path.join(HERE, script)] + files + extra + ["--out", os.path.join(OUT, out)]
    print("running", script, " ".join(extra), flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc:
        sys.exit(rc)
with open(RECORD, "w", encoding="utf-8") as f:
    json.dump({"tag": "p3stack_dgx", "files": [os.path.basename(x) for x in files], "missing": missing,
               "outputs": sorted(os.listdir(OUT))}, f, indent=1)
print("done", len(files), "files;", "missing:", missing, flush=True)
