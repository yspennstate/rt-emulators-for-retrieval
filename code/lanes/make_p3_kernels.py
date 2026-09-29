"""Write the Kaggle kernel directories of the EMIT pilot from p3_kaggle_lane.py: lane A (p3rwA_s101..110: floor
1e-2 and joint 1e-2) and lane B (p3rwB_s101..110: floors 1e-3, 1e-1 and joint 1e-3), both with the plain arm."""
import base64
import hashlib
import json
import pathlib
import py_compile

HERE = pathlib.Path(__file__).parent
PILOT = pathlib.Path(__file__).resolve().parent.parent / "retrieval_weighted_pilot.py"
src = PILOT.read_bytes().replace(b"\r\n", b"\n")
sha = hashlib.sha256(src).hexdigest()
b64 = base64.b64encode(src).decode("ascii")
T = (HERE / "p3_kaggle_lane.py").read_text(encoding="utf-8")
made = []
LANES = {"A": "--u 1e-2 --joint 1e-2", "B": "--u 1e-3 1e-1 --joint 1e-3"}
for lane, seed in [(l, s) for l in ("A", "B") for s in range(101, 111)]:
    tag = f"p3rw{lane}_s{seed}"
    slug = f"p3rw{lane.lower()}-s{seed}-20260923"
    d = HERE / "kernels" / tag
    d.mkdir(parents=True, exist_ok=True)
    code = T.replace("@SEED@", str(seed)).replace("@TAG@", tag).replace("@PILOT_SHA@", sha).replace("@PILOT_B64@", b64).replace("@ARGS@", LANES[lane])
    assert not any(m in code for m in ("@SEED@", "@TAG@", "@PILOT_SHA@", "@PILOT_B64@", "@ARGS@"))
    (d / "lane.py").write_text(code, encoding="utf-8", newline="\n")
    py_compile.compile(str(d / "lane.py"), doraise=True)
    meta = {"id": f"yitzchakshmalo/{slug}", "title": slug, "code_file": "lane.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": ["yitzchakshmalo/jpl-emit-reg-data"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    made.append(tag)
print(len(made), "kernels, pilot sha256", sha[:16])
