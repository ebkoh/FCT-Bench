import os
import json
import csv
import glob
import argparse

import fitz  # PyMuPDF

RENDER_SCALE = 2
IOU_THRESHOLD = 0.5


def get_page_dimensions(pdf_path):
    doc = fitz.open(pdf_path)
    dims = {}
    for i in range(len(doc)):
        rect = doc[i].rect
        dims[i + 1] = {  # 1-indexed
            "pdf_width": rect.width,
            "pdf_height": rect.height,
            "pixel_width": int(rect.width * RENDER_SCALE),
            "pixel_height": int(rect.height * RENDER_SCALE),
        }
    doc.close()
    return dims


def pdf_bbox_to_pixel(bbox_pdf, page_dim):
    if bbox_pdf is None:
        return None
    x1, y1, x2, y2 = bbox_pdf
    pw, ph = page_dim["pdf_width"], page_dim["pdf_height"]
    pxw, pxh = page_dim["pixel_width"], page_dim["pixel_height"]
    return [(x1 / pw) * pxw, (y1 / ph) * pxh, (x2 / pw) * pxw, (y2 / ph) * pxh]


def calculate_iou(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter
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

    fn_count = len(gt.get("false_negatives", []))
    return gt_bboxes, fn_count


def load_tool_detections(tool_dir, is_morphology, doc_name, page_dims):
    json_path = os.path.join(tool_dir, f"{doc_name}.json")
    if not os.path.exists(json_path):
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    detections = []
    for t in result.get("tables", []):
        page = t.get("page", 0)
        if is_morphology:
            bbox_pixel = t["bbox"]  # already pixel space
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
                iou = calculate_iou(g["bbox_pixel"], d["bbox_pixel"])
                if iou >= iou_threshold:
                    all_pairs.append((iou, g["gt_id"], d["table_id"]))

    all_pairs.sort(key=lambda x: -x[0])
    matched_gt, matched_det = set(), set()
    for iou, gt_id, det_id in all_pairs:
        if gt_id not in matched_gt and det_id not in matched_det:
            matched_gt.add(gt_id)
            matched_det.add(det_id)

    return {"tp": len(matched_gt),
            "fp": len(detections) - len(matched_det),
            "fn_detection": len(gt_bboxes) - len(matched_gt)}


def compute_metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def main():
    ap = argparse.ArgumentParser(description="IoU=0.5 evaluation vs ground truth")
    ap.add_argument("--gt-dir", default="results/morphology_GT")
    ap.add_argument("--morphology", default="results_morphology")
    ap.add_argument("--baselines", default="results_baselines")
    ap.add_argument("--pdf-dir", default="data")
    ap.add_argument("--out", default="evaluation_results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    tool_dirs = {
        "OpenCV_Morphology": args.morphology,
        "Camelot-py": os.path.join(args.baselines, "Camelot-py"),
        "pdfplumber": os.path.join(args.baselines, "pdfplumber"),
        "PyMuPDF": os.path.join(args.baselines, "PyMuPDF"),
        "Tabula-py": os.path.join(args.baselines, "Tabula-py"),
        "Table_Transformer": os.path.join(args.baselines, "Table_Transformer_Detection"),
        "DocLayout-YOLO": os.path.join(args.baselines, "DocLayout-YOLO"),
    }

    # documents = those with a GT file
    documents = sorted(os.path.basename(p)[:-len("_gt.json")]
                       for p in glob.glob(os.path.join(args.gt_dir, "*_gt.json")))

    print(f"Evaluation (IoU >= {IOU_THRESHOLD}) over {len(documents)} documents\n")
    all_results = {}

    for tool_name, tool_dir in tool_dirs.items():
        is_morph = tool_name == "OpenCV_Morphology"
        print(f"[{tool_name}]")
        tool_total = {"tp": 0, "fp": 0, "fn": 0}
        doc_results = []

        for doc_name in documents:
            pdf_path = os.path.join(args.pdf_dir, f"{doc_name}.pdf")
            if not os.path.exists(pdf_path):
                continue
            page_dims = get_page_dimensions(pdf_path)
            gt_bboxes, fn_from_gt = load_gt_bboxes(args.gt_dir, args.morphology, doc_name)
            detections = load_tool_detections(tool_dir, is_morph, doc_name, page_dims)
            match = match_detections_to_gt(gt_bboxes, detections, IOU_THRESHOLD)

            tp = match["tp"]
            fp = match["fp"]
            fn = match["fn_detection"] + fn_from_gt
            precision, recall, f1 = compute_metrics(tp, fp, fn)

            tool_total["tp"] += tp
            tool_total["fp"] += fp
            tool_total["fn"] += fn
            doc_results.append({
                "document": doc_name, "detected": len(detections),
                "gt_tables": len(gt_bboxes) + fn_from_gt,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
            })

        t_p, t_r, t_f1 = compute_metrics(tool_total["tp"], tool_total["fp"], tool_total["fn"])
        print(f"  TOTAL: TP={tool_total['tp']} FP={tool_total['fp']} FN={tool_total['fn']} "
              f"P={t_p:.3f} R={t_r:.3f} F1={t_f1:.3f}")
        all_results[tool_name] = {
            "documents": doc_results,
            "total": {"tp": tool_total["tp"], "fp": tool_total["fp"], "fn": tool_total["fn"],
                      "precision": t_p, "recall": t_r, "f1": t_f1},
        }

    with open(os.path.join(args.out, "evaluation_all.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out, "evaluation_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Tool", "TP", "FP", "FN", "Precision", "Recall", "F1"])
        for tool_name, data in all_results.items():
            t = data["total"]
            w.writerow([tool_name, t["tp"], t["fp"], t["fn"], t["precision"], t["recall"], t["f1"]])

    with open(os.path.join(args.out, "evaluation_by_document.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Tool", "Document", "Detected", "GT_Tables", "TP", "FP", "FN",
                    "Precision", "Recall", "F1"])
        for tool_name, data in all_results.items():
            for doc in data["documents"]:
                w.writerow([tool_name, doc["document"], doc["detected"], doc["gt_tables"],
                            doc["tp"], doc["fp"], doc["fn"],
                            doc["precision"], doc["recall"], doc["f1"]])

    print(f"\n[written] {args.out}/evaluation_all.json, evaluation_summary.csv, evaluation_by_document.csv")


if __name__ == "__main__":
    main()
