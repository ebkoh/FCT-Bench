import os
import json
import glob
import argparse

import cv2

N_V_MIN = 2
H_COVERAGE_MIN = 0.4853


def extract_features(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    if h == 0 or w == 0:
        return None

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # horizontal segments
    kh = max(int(w * 0.3), 10)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1))
    h_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    h_cnts, _ = cv2.findContours(h_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_segs = []
    for c in h_cnts:
        xc, yc, wc, hc = cv2.boundingRect(c)
        if wc > w * 0.15:
            h_segs.append([yc, yc + hc, xc, xc + wc])

    h_segs.sort(key=lambda s: s[0])
    h_merged = []
    for seg in h_segs:
        if h_merged and seg[0] - h_merged[-1][1] <= 5:
            p = h_merged[-1]
            h_merged[-1] = [p[0], max(p[1], seg[1]), min(p[2], seg[2]), max(p[3], seg[3])]
        else:
            h_merged.append(seg)

    n_h = len(h_merged)

    # vertical segments
    kv = max(int(h * 0.3), 10)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv))
    v_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
    v_cnts, _ = cv2.findContours(v_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    v_segs = []
    for c in v_cnts:
        xc, yc, wc, hc = cv2.boundingRect(c)
        if hc > h * 0.15:
            v_segs.append([xc, xc + wc, yc, yc + hc])

    v_segs.sort(key=lambda s: s[0])
    v_merged = []
    for seg in v_segs:
        if v_merged and seg[0] - v_merged[-1][1] <= 5:
            p = v_merged[-1]
            v_merged[-1] = [p[0], max(p[1], seg[1]), min(p[2], seg[2]), max(p[3], seg[3])]
        else:
            v_merged.append(seg)

    n_v = len(v_merged)

    h_cover = sum(seg[3] - seg[2] for seg in h_merged)
    h_coverage = min(h_cover / (n_h * w), 1.0) if n_h > 0 else 0.0

    return {"n_h": n_h, "n_v": n_v, "h_coverage": round(h_coverage, 4)}


def is_non_table(feats: dict) -> bool:
    if feats["n_v"] < N_V_MIN:
        return True
    if feats["h_coverage"] < H_COVERAGE_MIN:
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Flowchart false-positive filter")
    ap.add_argument("--morphology", default="results_morphology",
                    help="morphology results folder (JSON + crops)")
    ap.add_argument("--out", default="results_morphology_filtered")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total_removed = total_kept = 0

    json_files = sorted(p for p in glob.glob(os.path.join(args.morphology, "*.json"))
                        if not os.path.basename(p).startswith("_"))

    for morph_path in json_files:
        doc = os.path.splitext(os.path.basename(morph_path))[0]
        with open(morph_path, "r", encoding="utf-8") as f:
            morph = json.load(f)

        kept, removed = [], []
        for t in morph["tables"]:
            img_path = os.path.join(args.morphology, t["image_path"])
            feats = extract_features(img_path)
            if feats is None:
                kept.append(t)  # keep if the crop is missing
                continue
            if is_non_table(feats):
                removed.append({**t, "filter_feats": feats})
            else:
                kept.append(t)

        morph_filtered = {
            **morph,
            "total_tables_detected": len(kept),
            "tables": kept,
            "filtered_out": len(removed),
        }
        with open(os.path.join(args.out, f"{doc}.json"), "w", encoding="utf-8") as f:
            json.dump(morph_filtered, f, ensure_ascii=False, indent=2)

        total_removed += len(removed)
        total_kept += len(kept)
        print(f"  {doc:20s}: in={len(morph['tables']):4d}  kept={len(kept):4d}  removed={len(removed):3d}")

    print(f"\n  total: in={total_kept + total_removed}  kept={total_kept}  removed={total_removed}")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
