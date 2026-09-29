"""Where the two failed evaluations of the EMIT table fail, band by band.

For states 4011 and 7439: the transmission t = Y2 + Y3 over the median positive transmission T, against the mean of the
five nearest states in the standardized inputs; the spherical albedo; how many bands have t = 0, Y4 >= 1 and Y4 == 2;
the bands past the onset whose transmission is still above the floors 1e-1, 1e-2 and 1e-3 T (the entries the cut
weights count); and the path radiance's relative difference from the neighbours below 800 nm and beyond 1030 nm,
against the same quantity for 400 random states at a similar neighbour distance.

usage: python code/failed_rows_bands.py <dir with X.npy, Y1.npy .. Y4.npy> [--out results/failed_rows_bands.json]
"""
import json
import os
import sys

import numpy as np

args = sys.argv[1:]
out = "results/failed_rows_bands.json"
if "--out" in args:
    i = args.index("--out")
    out = args[i + 1]
    del args[i:i + 2]
d = args[0]
X = np.load(os.path.join(d, "X.npy"))
Y = {c: np.load(os.path.join(d, c + ".npy")) for c in ("Y1", "Y2", "Y3", "Y4")}
t = Y["Y2"] + Y["Y3"]
T = float(np.median(t[t > 0]))
wl = np.linspace(381, 2493, t.shape[1])        # the EMIT grid, 381-2493 nm in 285 bands
Z = (X - X.mean(0)) / X.std(0)


def neighbours(r, k=5):
    dist = np.sqrt(((Z - Z[r]) ** 2).sum(1))
    nb = np.argsort(dist)[1:k + 1]
    return nb, float(dist[nb].mean())


def y1_reldiff(r):
    nb, dm = neighbours(r)
    m = Y["Y1"][nb].mean(0)
    rel = np.abs(Y["Y1"][r] - m) / np.maximum(np.abs(m), 1e-12)
    return dm, float(np.median(rel[:58])), float(np.median(rel[88:]))


rng = np.random.default_rng(0)
ref = np.array([y1_reldiff(r) for r in rng.choice(t.shape[0], 400, replace=False)])
rep = {"T": T, "states": {}}
for r in (4011, 7439):
    nb, dm = neighbours(r)
    tn = t[nb].mean(0)
    onset = int(np.argmax((Y["Y4"][r] >= 1)))
    row = {"zero_t_bands": int((t[r] == 0).sum()), "first_zero_t_band": int(np.argmax(t[r] == 0)),
           "albedo_ge1_bands": int((Y["Y4"][r] >= 1).sum()), "albedo_eq2_bands": int((Y["Y4"][r] == 2.0).sum()),
           "onset_band": onset, "onset_nm": float(wl[onset]),
           "bands": [{"band": b, "nm": round(float(wl[b])), "t_over_T": float(t[r, b] / T),
                      "neighbours_t_over_T": float(tn[b] / T), "Y4": float(Y["Y4"][r, b])} for b in range(54, 72)],
           "above_floor_after_onset": {f"{u:g}": [b for b in range(onset, t.shape[1]) if t[r, b] >= u * T]
                                       for u in (1e-1, 1e-2, 1e-3)}}
    dm, below, beyond = y1_reldiff(r)
    near = ref[np.abs(ref[:, 0] - dm) < 0.1]
    row["path_radiance"] = {"neighbour_distance": dm, "reldiff_below_800nm": below, "reldiff_beyond_1030nm": beyond,
                            "reference_states_at_similar_distance": int(len(near)),
                            "reference_median_beyond_1030nm": float(np.median(near[:, 2])) if len(near) else None,
                            "reference_q90_beyond_1030nm": float(np.quantile(near[:, 2], 0.9)) if len(near) else None}
    rep["states"][str(r)] = row
    print(f"state {r}: onset band {onset} ({wl[onset]:.0f} nm); t=0 in {row['zero_t_bands']} bands from band "
          f"{row['first_zero_t_band']}; albedo >= 1 in {row['albedo_ge1_bands']}, == 2 in {row['albedo_eq2_bands']}; "
          f"above the floors after the onset: {row['above_floor_after_onset']}")
    print(f"   path radiance vs neighbours: below 800 nm {below:.3f}, beyond 1030 nm {beyond:.3f} "
          f"(reference at similar distance: median {row['path_radiance']['reference_median_beyond_1030nm']}, "
          f"90th pct {row['path_radiance']['reference_q90_beyond_1030nm']}, n={len(near)})")
bad = np.zeros(t.shape[0], bool)
bad[[4011, 7439]] = True
rep["only_failed_below_floor"] = {}
for u in (1e-2, 1e-3):
    below = t < u * T
    only = np.where((below[~bad].sum(0) == 0) & (below[bad].sum(0) > 0))[0]
    rep["only_failed_below_floor"][f"{u:g}"] = {"bands": int(len(only)), "from_nm": float(wl[only.min()]),
                                                "to_nm": float(wl[only.max()])}
    print(f"u={u:g}: the failed states are the only ones below the floor in {len(only)} bands, "
          f"{wl[only.min()]:.0f}-{wl[only.max()]:.0f} nm")
sd = Y["Y1"][~bad].std(0)   # the band standard deviation of the path radiance over the ordinary states
rep["path_radiance_band_sd_median"] = {}
for lo, hi in ((381, 700), (700, 1000), (1000, 1300), (1300, 2500)):
    m = (wl >= lo) & (wl < hi)
    rep["path_radiance_band_sd_median"][f"{lo}-{hi}nm"] = float(np.median(sd[m]))
    print(f"path radiance, bands {lo}-{hi} nm: median standard deviation {np.median(sd[m]):.3g}")
json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
print("wrote", out)
