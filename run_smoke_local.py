# run_smoke_local.py
import os, json, sys, time
import cv2
import numpy as np
from drop.drop_inference import full_droplet_analysis

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

tests = [
    ("miR-92a", "drop/miR-92a.tif"),
    ("miR-21", "drop/miR-21.tif"),
]

def read_rgb(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    # handle grayscale / BGR / BGRA
    if img.ndim == 2:
        return img  # let pipeline handle ambiguous if needed
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def main():
    for name, path in tests:
        print("\n" + "="*80)
        print(f"[SMOKE] {name}: {path}")
        img = read_rgb(path)
        t0 = time.time()
        res = full_droplet_analysis(img, target="auto", ambiguous_policy="error", max_side=768)
        dt = time.time() - t0
        # Key fields (guard with get)
        keys = ["target_detected","target_source","auto_target_score","n_droplets",
                "log10_concentration","concentration_fM","label","confidence"]
        print(f"[SMOKE] elapsed: {dt:.2f}s")
        for k in keys:
            print(f"  {k}: {res.get(k)}")
        out_json = os.path.join(OUTDIR, f"smoke_local_{name}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            clean = {}
            for k, v in res.items():
                if isinstance(v, (np.integer,)):
                    clean[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean[k] = v.tolist()
                else:
                    clean[k] = v
            json.dump(clean, f, ensure_ascii=False, indent=2)
        print(f"[SMOKE] saved: {out_json}")
    print("\n[SMOKE] ALL PASSED")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[SMOKE] FAILED: {e}", file=sys.stderr)
        sys.exit(1)
