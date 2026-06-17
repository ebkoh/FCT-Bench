import os
import json
import glob
import argparse

import cv2

FEATURES = ["n_h", "n_v", "n_inter", "grid_ratio", "vh_ratio",
            "h_coverage", "v_coverage", "aspect"]


def extract_features(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # horizontal segments
    kh = max(int(w * 0.3), 10)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1))
    h_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    h_contours, _ = cv2.findContours(h_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_segments = []
    for c in h_contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(c)
        if w_c > w * 0.15:
            h_segments.append((y_c, y_c + h_c, x_c, x_c + w_c))

    h_segments.sort(key=lambda s: s[0])
    h_merged = []
    for seg in h_segments:
        if h_merged and seg[0] - h_merged[-1][1] <= 5:
            prev = h_merged[-1]
            h_merged[-1] = (prev[0], max(prev[1], seg[1]),
                            min(prev[2], seg[2]), max(prev[3], seg[3]))
        else:
            h_merged.append(list(seg))
    n_h = len(h_merged)

    # vertical segments
    kv = max(int(h * 0.3), 10)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv))
    v_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
    v_contours, _ = cv2.findContours(v_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_segments = []
    for c in v_contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(c)
        if h_c > h * 0.15:
            v_segments.append((x_c, x_c + w_c, y_c, y_c + h_c))

    v_segments.sort(key=lambda s: s[0])
    v_merged = []
    for seg in v_segments:
        if v_merged and seg[0] - v_merged[-1][1] <= 5:
            prev = v_merged[-1]
            v_merged[-1] = (prev[0], max(prev[1], seg[1]),
                            min(prev[2], seg[2]), max(prev[3], seg[3]))
        else:
            v_merged.append(list(seg))
    n_v = len(v_merged)

    # intersections
    n_inter = 0
    for hs in h_merged:
        hy0, hy1, hx0, hx1 = hs
        for vs in v_merged:
            vx0, vx1, vy0, vy1 = vs
            if hx0 <= vx1 and hx1 >= vx0 and vy0 <= hy1 and vy1 >= hy0:
                n_inter += 1

    grid_ratio = n_inter / (n_h * n_v) if n_h > 0 and n_v > 0 else 0.0
    vh_ratio = n_v / n_h if n_h > 0 else 0.0
    h_cover_total = sum(seg[3] - seg[2] for seg in h_merged)
    h_coverage = min(h_cover_total / (n_h * w) if n_h > 0 else 0.0, 1.0)
    v_cover_total = sum(seg[3] - seg[2] for seg in v_merged)
    v_coverage = min(v_cover_total / (n_v * h) if n_v > 0 else 0.0, 1.0)

    return {
        "n_h": n_h, "n_v": n_v, "n_inter": n_inter,
        "grid_ratio": round(grid_ratio, 4), "vh_ratio": round(vh_ratio, 4),
        "h_coverage": round(h_coverage, 4), "v_coverage": round(v_coverage, 4),
        "aspect": round(w / h, 4) if h > 0 else 1.0,
        "img_w": w, "img_h": h,
    }


def load_fp_tp_features(gt_dir, morph_dir, documents):
    fp_feats, tp_feats = [], []
    for doc in documents:
        gt_path = os.path.join(gt_dir, f"{doc}_gt.json")
        morph_path = os.path.join(morph_dir, f"{doc}.json")
        if not os.path.exists(gt_path) or not os.path.exists(morph_path):
            continue
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        with open(morph_path, "r", encoding="utf-8") as f:
            morph = json.load(f)

        id_to_img = {t["table_id"]: t["image_path"] for t in morph["tables"]}
        tp_ids = {t["original_table_id"] for t in gt.get("tables", []) if t["verdict"] == "tp"}
        fp_ids = {t["original_table_id"] for t in gt.get("false_positives", [])}

        for tid, img_rel in id_to_img.items():
            img_path = os.path.join(morph_dir, img_rel)
            if not os.path.exists(img_path):
                continue
            feats = extract_features(img_path)
            if feats is None:
                continue
            feats["doc"] = doc
            feats["table_id"] = tid
            if tid in fp_ids:
                fp_feats.append(feats)
            elif tid in tp_ids:
                tp_feats.append(feats)

    return fp_feats, tp_feats


def search_single(fp_feats, tp_feats):
    n_fp = len(fp_feats)
    print()
    print("Single-feature thresholds (zero TP loss)")
    rules = []
    for feat in FEATURES:
        fp_vals = [f[feat] for f in fp_feats]
        tp_vals = [f[feat] for f in tp_feats]
        for thr in sorted(set(fp_vals)):
            for direction in ("<", ">"):
                if direction == "<":
                    fp_caught = sum(1 for v in fp_vals if v < thr)
                    tp_lost = sum(1 for v in tp_vals if v < thr)
                else:
                    fp_caught = sum(1 for v in fp_vals if v > thr)
                    tp_lost = sum(1 for v in tp_vals if v > thr)
                if tp_lost == 0 and fp_caught > 0:
                    rules.append({"rule": f"{feat} {direction} {thr}", "fp_caught": fp_caught})
    rules.sort(key=lambda r: -r["fp_caught"])
    for r in rules[:30]:
        print(f"  {r['rule']:40s}  FP={r['fp_caught']:3d}/{n_fp}  ({r['fp_caught']/n_fp*100:.1f}%)")


def search_or_combos(fp_feats, tp_feats):
    n_fp = len(fp_feats)
    safe = []
    for feat in FEATURES:
        fp_vals = [f[feat] for f in fp_feats]
        tp_vals = [f[feat] for f in tp_feats]
        for thr in sorted(set(fp_vals + tp_vals)):
            for direction in ("<", ">"):
                if direction == "<":
                    fp_mask = [v < thr for v in fp_vals]
                    tp_mask = [v < thr for v in tp_vals]
                else:
                    fp_mask = [v > thr for v in fp_vals]
                    tp_mask = [v > thr for v in tp_vals]
                if sum(tp_mask) == 0 and sum(fp_mask) > 0:
                    safe.append({"feat": feat, "thr": thr, "dir": direction,
                                 "fp_mask": fp_mask, "tp_mask": tp_mask,
                                 "fp_caught": sum(fp_mask)})

    print()
    print(f"Two-feature OR combinations (zero TP loss); {len(safe)} safe single rules")
    combos = []
    for i in range(len(safe)):
        for j in range(i + 1, len(safe)):
            r1, r2 = safe[i], safe[j]
            fp_caught = sum(m1 or m2 for m1, m2 in zip(r1["fp_mask"], r2["fp_mask"]))
            tp_lost = sum(m1 or m2 for m1, m2 in zip(r1["tp_mask"], r2["tp_mask"]))
            if tp_lost == 0 and fp_caught > r1["fp_caught"] and fp_caught > r2["fp_caught"]:
                combos.append({"rule": f"{r1['feat']} {r1['dir']} {r1['thr']} OR "
                                       f"{r2['feat']} {r2['dir']} {r2['thr']}",
                               "fp_caught": fp_caught})
    combos.sort(key=lambda r: -r["fp_caught"])
    for r in combos[:20]:
        print(f"  {r['rule']:60s}  FP={r['fp_caught']:3d}/{n_fp}  ({r['fp_caught']/n_fp*100:.1f}%)")


def print_statistics(fp_feats, tp_feats):
    print()
    print("Feature statistics (FP vs TP)")
    print(f"{'Feature':15s} {'FP_med':>8s} {'FP_q25':>8s} {'FP_q75':>8s} "
          f"{'TP_med':>8s} {'TP_q25':>8s} {'TP_q75':>8s}")
    for feat in FEATURES:
        fp_v = sorted(f[feat] for f in fp_feats)
        tp_v = sorted(f[feat] for f in tp_feats)

        def stats(vs):
            n = len(vs)
            if n == 0:
                return 0, 0, 0
            return vs[n // 2], vs[n // 4], vs[3 * n // 4]

        fm, fq25, fq75 = stats(fp_v)
        tm, tq25, tq75 = stats(tp_v)
        print(f"{feat:15s} {fm:>8.3f} {fq25:>8.3f} {fq75:>8.3f} "
              f"{tm:>8.3f} {tq25:>8.3f} {tq75:>8.3f}")


def main():
    ap = argparse.ArgumentParser(description="Flowchart-filter threshold search")
    ap.add_argument("--gt-dir", default="results/morphology_GT")
    ap.add_argument("--morphology", default="results_morphology")
    args = ap.parse_args()

    documents = sorted(os.path.basename(p)[:-len("_gt.json")]
                       for p in glob.glob(os.path.join(args.gt_dir, "*_gt.json")))

    print("Extracting crop features...")
    fp_feats, tp_feats = load_fp_tp_features(args.gt_dir, args.morphology, documents)
    print(f"FP={len(fp_feats)}, TP={len(tp_feats)}")

    print_statistics(fp_feats, tp_feats)
    search_single(fp_feats, tp_feats)
    search_or_combos(fp_feats, tp_feats)


if __name__ == "__main__":
    main()
