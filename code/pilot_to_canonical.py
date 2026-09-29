"""Put the pilot's predictions in the format of the paper's canonical scorer (conditioned_reflectance.py).

Reads <out>/pred_s<seed>_<arm><tag>.npz (idx_te and Y1-Y4 of each arm) and the pilot record
<out>/pilot_s<seed><tag>.json, and writes <out>/canon_s<seed><tag>.npz (idx_te and <arm>_Y1 ... <arm>_Y4) with
<out>/canon_s<seed><tag>.json (seed, ntrain, data digests, families), which the scorer reads with --predictions and
--record.

usage: EMIT_DATA=<dir> python pilot_to_canonical.py <out dir> <seed> [tag]
"""
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

out, seed = sys.argv[1], int(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else ""
data = os.environ["EMIT_DATA"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


rec = json.load(open(os.path.join(out, f"pilot_s{seed}{tag}.json"), encoding="utf-8"))
arrays, idx_te = {}, None
for p in sorted(glob.glob(os.path.join(out, f"pred_s{seed}_*{tag}.npz"))):
    m = re.match(rf"pred_s{seed}_(.+){re.escape(tag)}\.npz$", os.path.basename(p))
    if not m:
        continue
    arm = m.group(1)
    z = np.load(p, allow_pickle=False)
    if idx_te is None:
        idx_te = z["idx_te"]
    elif not np.array_equal(idx_te, z["idx_te"]):
        raise SystemExit(f"{p}: test indices differ between arms")
    for c in ("Y1", "Y2", "Y3", "Y4"):
        arrays[f"{arm}_{c}"] = np.asarray(z[c], dtype=np.float64)
fams = sorted({k.rsplit("_", 1)[0] for k in arrays})
np.savez(os.path.join(out, f"canon_s{seed}{tag}.npz"), idx_te=idx_te, **arrays)
canon = {"seed": seed, "ntrain": rec["n_train"], "families": {f: {} for f in fams},
         "data_sha": {c: sha256(os.path.join(data, c + ".npy")) for c in ("X", "Y1", "Y2", "Y3", "Y4")}}
json.dump(canon, open(os.path.join(out, f"canon_s{seed}{tag}.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({"families": fams, "ntrain": rec["n_train"], "test": int(len(idx_te))}))
