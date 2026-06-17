import os
import json
import csv
import glob
import argparse

import fitz  # PyMuPDF

RENDER_SCALE = 2
IOU_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
PIXEL_COORD_TOOLS = {"OpenCV_Morphology", "OpenCV_Morphology_Filtered"}


def get_page_dimensions(pdf_path):
    doc = fitz.open(pdf_path)
    dims = {}
    for i in range(len(doc)):
        rect = doc[i].rect
        dims[i + 1] = {"pdf_width": rect.width, "pdf_height": rect.height,
                       "pixel_width": int(rect.width * RENDER_SCALE),
                       "pixel_height": int(rect.height * RENDER_SCALE)}
    doc.close()
    return dims


def pdf_bbox_to_pixel(bbox_pdf, page_dim):
    if bbox_pdf is None:
        return None
    x1, y1, x2, y2 = bbox_pdf
    pw, ph = page_dim["pdf_width"], page_dim["pdf_height"]
    pxw, pxh = page_dim["pixel_width"], page_dim["pixel_height"]
    return [(x1 / pw) * pxw, (y1 / ph) * pxh, (x2 / pw) * pxw, (y2 / ph) * pxh]


def calculate_iou(b1, b2):
    x1_i = max(b1[0], b2[0]); y1_i = max(b1[1], b2[1])
    x2_i = min(b1[2], b2[2]); y2_i = min(b1[3], b2[3])
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def load_gt_bboxes(gt_dir, morph_dir, doc_name):
    with open(os.path.join(gt_dir, f"{doc_name}_gt.json"), "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(os.path.join(morph_dir, f"{doc_name}.json"), "r", encoding="utf-8") as f:
        morph = json.load(f)

    morph_tables = {t["table_id"]: t for t in morph["tables"]}
    gt_bboxes = []

    for t in gt.get("tables", []):
        if t["verdict"] == "tp":
            tid = t["original_table_id"]
            if tid in morph_tables:
                mt = morph_tables[tid]
                gt_bboxes.append({"gt_id": t["gt_id"], "page": mt["page"],
                                  "bbox_pixel": mt["bbox"]})

    seen = set()
    fn_count = 0
    for fn in gt.get("false_negatives", []):
        bbox = fn.get("bbox_pixel")
        if bbox is not None:
            key = (fn.get("page", 0), tuple(bbox))
            if key not in seen:
                seen.add(key)
                gt_bboxes.append({"gt_id": fn["gt_id"], "page": fn.get("page", 0),
                                  "bbox_pixel": bbox})
        else:
            fn_count += 1

    return gt_bboxes, fn_count


def load_tool_detections(tool_name, tool_dir, doc_name, page_dims):
    json_path = os.path.join(tool_dir, f"{doc_name}.json")
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    detections = []
    for t in result.get("tables", []):
        page = t.get("page", 0)
        if tool_name in PIXEL_COORD_TOOLS:
            bbox_pixel = t.get("bbox")
        else:
            bbox_pdf = t.get("bbox_pdf")
            if bbox_pdf is None or page not in page_dims:
                continue
            bbox_pixel = pdf_bbox_to_pixel(bbox_pdf, page_dims[page])
        if bbox_pixel:
            detections.append({"table_id": t["table_id"], "page": page,
                               "bbox_pixel": bbox_pixel})
    return detections


def match_detections_to_gt(gt_bboxes, detections, iou_threshold):
    if not gt_bboxes or not detections:
        return {"tp": 0, "fp": len(detections), "fn_detection": len(gt_bboxes)}

    gt_by_page, det_by_page = {}, {}
    for g in gt_bboxes:
        gt_by_page.setdefault(g["page"], []).append(g)
    for d in detections:
        det_by_page.setdefault(d["page"], []).append(d)

    all_pairs = []
    for page in set(list(gt_by_page) + list(det_by_page)):
        for g in gt_by_page.get(page, []):
            for d in det_by_page.get(page, []):
                score = calculate_iou(g["bbox_pixel"], d["bbox_pixel"])
                if score >= iou_threshold:
                    all_pairs.append((score, g["gt_id"], d["table_id"]))

    all_pairs.sort(key=lambda x: -x[0])
    matched_gt, matched_det = set(), set()
    for score, gt_id, det_id in all_pairs:
        if gt_id not in matched_gt and det_id not in matched_det:
            matched_gt.add(gt_id); matched_det.add(det_id)

    return {"tp": len(matched_gt),
            "fp": len(detections) - len(matched_det),
            "fn_detection": len(gt_bboxes) - len(matched_gt)}


def compute_metrics(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def main():
    ap = argparse.ArgumentParser(description="Multi-IoU evaluation")
    ap.add_argument("--gt-dir", default="results/morphology_GT")
    ap.add_argument("--morphology", default="results_morphology")
    ap.add_argument("--morphology-filtered", default="results_morphology_filtered")
    ap.add_argument("--baselines", default="results_baselines")
    ap.add_argument("--pdf-dir", default="data")
    ap.add_argument("--out", default="evaluation_results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    tool_dirs = {
        "OpenCV_Morphology": args.morphology,
        "OpenCV_Morphology_Filtered": args.morphology_filtered,
        "Camelot-py": os.path.join(args.baselines, "Camelot-py"),
        "pdfplumber": os.path.join(args.baselines, "pdfplumber"),
        "PyMuPDF": os.path.join(args.baselines, "PyMuPDF"),
        "Tabula-py": os.path.join(args.baselines, "Tabula-py"),
        "Table_Transformer": os.path.join(args.baselines, "Table_Transformer_Detection"),
        "DocLayout-YOLO": os.path.join(args.baselines, "DocLayout-YOLO"),
    }
    documents = sorted(os.path.basename(p)[:-len("_gt.json")]
                       for p in glob.glob(os.path.join(args.gt_dir, "*_gt.json")))

    # load once per (tool, doc); only the IoU threshold varies
    page_dims_cache = {d: get_page_dimensions(os.path.join(args.pdf_dir, f"{d}.pdf"))
                       for d in documents if os.path.exists(os.path.join(args.pdf_dir, f"{d}.pdf"))}
    cache = {}
    for tool_name, tool_dir in tool_dirs.items():
        for doc_name in documents:
            if doc_name not in page_dims_cache:
                continue
            gt_bboxes, fn_count = load_gt_bboxes(args.gt_dir, args.morphology, doc_name)
            detections = load_tool_detections(tool_name, tool_dir, doc_name, page_dims_cache[doc_name])
            cache[(tool_name, doc_name)] = (gt_bboxes, fn_count, detections)

    summary_rows, doc_rows = [], []
    for iou_thr in IOU_THRESHOLDS:
        print(f"\nIoU >= {iou_thr}")
        for tool_name in tool_dirs:
            tot = {"tp": 0, "fp": 0, "fn": 0}
            for doc_name in documents:
                if (tool_name, doc_name) not in cache:
                    continue
                gt_bboxes, fn_count, detections = cache[(tool_name, doc_name)]
                match = match_detections_to_gt(gt_bboxes, detections, iou_thr)
                tp, fp = match["tp"], match["fp"]
                fn = match["fn_detection"] + fn_count
                tot["tp"] += tp; tot["fp"] += fp; tot["fn"] += fn
                p, r, f1 = compute_metrics(tp, fp, fn)
                doc_rows.append([tool_name, iou_thr, doc_name, len(detections),
                                 len(gt_bboxes) + fn_count, tp, fp, fn, p, r, f1])
            t_p, t_r, t_f1 = compute_metrics(tot["tp"], tot["fp"], tot["fn"])
            print(f"  {tool_name:35s} TP={tot['tp']:5d} FP={tot['fp']:5d} FN={tot['fn']:5d} "
                  f"P={t_p:.4f} R={t_r:.4f} F1={t_f1:.4f}")
            summary_rows.append([tool_name, iou_thr, tot["tp"], tot["fp"], tot["fn"], t_p, t_r, t_f1])

    with open(os.path.join(args.out, "evaluation_multi_iou.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Tool", "IoU_Threshold", "TP", "FP", "FN", "Precision", "Recall", "F1"])
        w.writerows(summary_rows)

    with open(os.path.join(args.out, "evaluation_multi_iou_by_doc.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Tool", "IoU_Threshold", "Document", "Detected", "GT_Tables",
                    "TP", "FP", "FN", "Precision", "Recall", "F1"])
        w.writerows(doc_rows)

    print(f"\n[written] {args.out}/evaluation_multi_iou.csv, evaluation_multi_iou_by_doc.csv")


if __name__ == "__main__":
    main()
