import argparse
import json
import sys
from pathlib import Path


# IoU matching

def parse_yolo_file(path):
    if not path.exists():
        return []
    bboxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                xc, yc, w, h = map(float, parts[1:5])
                bboxes.append((xc, yc, w, h))
            except ValueError:
                continue
    return bboxes


def yolo_to_xyxy(bbox):
    xc, yc, w, h = bbox
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = yolo_to_xyxy(a)
    bx1, by1, bx2, by2 = yolo_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(a, b):
    ax1, ay1, ax2, ay2 = yolo_to_xyxy(a)
    bx1, by1, bx2, by2 = yolo_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = (ax2 - ax1) * (ay2 - ay1)
    return inter / aa if aa > 0 else 0.0


def greedy_match(bboxes_a, bboxes_b, iou_threshold=0.5):
    pairs = []
    for i, a in enumerate(bboxes_a):
        for j, b in enumerate(bboxes_b):
            v = iou(a, b)
            if v >= iou_threshold:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)

    matched_a, matched_b = set(), set()
    matches = []
    for v, i, j in pairs:
        if i in matched_a or j in matched_b:
            continue
        matched_a.add(i)
        matched_b.add(j)
        matches.append((i, j, v))

    unmatched_a = [i for i in range(len(bboxes_a)) if i not in matched_a]
    unmatched_b = [j for j in range(len(bboxes_b)) if j not in matched_b]
    return matches, unmatched_a, unmatched_b


# Disagreement classification

def classify_unmatched(unmatched_indices, side_bboxes, other_bboxes,
                       split_threshold=0.6):
    split = 0
    omission = 0
    for idx in unmatched_indices:
        a = side_bboxes[idx]
        covering = [b for b in other_bboxes if overlap_ratio(a, b) >= 0.3]
        if len(covering) >= 2:
            combined = sum(overlap_ratio(a, b) for b in covering)
            if combined >= split_threshold:
                split += 1
                continue
        omission += 1
    return split, omission


# Page paths

def find_original_path(orig_root, doc_id, page):
    if doc_id == "ISMS_P":
        return orig_root / "kisa" / "ISMS_P" / f"page_{page:04d}.txt"
    return orig_root / "kalis" / doc_id / f"page_{page:04d}.txt"


def find_reannotation_path(reann_dir, doc_id, page):
    return reann_dir / f"{doc_id}__page_{page:04d}.txt"


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Compare released annotations with an independent re-annotation."
    )
    parser.add_argument("--original-annotations", type=Path, required=True,
                        help="Root of released annotations (contains kalis/ and kisa/)")
    parser.add_argument("--reannotation", type=Path, required=True,
                        help="Directory with <doc_id>__page_<NNNN>.txt re-annotation files")
    parser.add_argument("--plan", type=Path, required=True,
                        help="blind_subset_plan.json specifying which pages were sampled")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output agreement_report.json path")
    parser.add_argument("--iou-thresholds", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7, 0.9],
                        help="IoU thresholds at which to report Jaccard agreement "
                             "(default: 0.3 0.5 0.7 0.9)")
    parser.add_argument("--primary-threshold", type=float, default=0.5,
                        help="IoU threshold used for classifying disagreements "
                             "(default: 0.5)")
    args = parser.parse_args()
    args.iou_thresholds = sorted(set(args.iou_thresholds + [args.primary_threshold]))

    if not args.plan.exists():
        print(f"ERROR: plan not found: {args.plan}", file=sys.stderr)
        sys.exit(1)

    with open(args.plan, "r", encoding="utf-8") as f:
        plan = json.load(f)

    print(f"Evaluating agreement on {len(plan)} sampled pages")
    print(f"Original:     {args.original_annotations}")
    print(f"Reannotation: {args.reannotation}")
    print()

    threshold_stats = {
        t: {"tp": 0, "fp_original_unmatched": 0, "fn_reann_unmatched": 0}
        for t in args.iou_thresholds
    }

    page_results = []
    total_orig = 0
    total_reann = 0
    boundary_pixel = 0
    split_convention = 0
    genuine_omissions = 0
    missing_reann_files = 0
    empty_pages_both = 0

    iou_distribution = []

    for entry in plan:
        doc_id = entry["doc_id"]
        page = entry["page"]

        orig_path = find_original_path(args.original_annotations, doc_id, page)
        reann_path = find_reannotation_path(args.reannotation, doc_id, page)

        orig_bboxes = parse_yolo_file(orig_path)
        reann_bboxes = parse_yolo_file(reann_path)

        if not reann_path.exists():
            missing_reann_files += 1

        total_orig += len(orig_bboxes)
        total_reann += len(reann_bboxes)

        if not orig_bboxes and not reann_bboxes:
            empty_pages_both += 1

        for t in args.iou_thresholds:
            matches, unmatched_o, unmatched_r = greedy_match(
                orig_bboxes, reann_bboxes, iou_threshold=t)
            threshold_stats[t]["tp"] += len(matches)
            threshold_stats[t]["fp_original_unmatched"] += len(unmatched_o)
            threshold_stats[t]["fn_reann_unmatched"] += len(unmatched_r)

        matches_p, unmatched_o_p, unmatched_r_p = greedy_match(
            orig_bboxes, reann_bboxes, iou_threshold=args.primary_threshold)

        for _, _, v in matches_p:
            iou_distribution.append(v)
            if v < 0.9:
                boundary_pixel += 1

        split_o, omit_o = classify_unmatched(unmatched_o_p, orig_bboxes, reann_bboxes)
        split_r, omit_r = classify_unmatched(unmatched_r_p, reann_bboxes, orig_bboxes)
        split_convention += split_o + split_r
        genuine_omissions += omit_o + omit_r

        page_results.append({
            "doc_id": doc_id,
            "page": page,
            "orig_bboxes": len(orig_bboxes),
            "reann_bboxes": len(reann_bboxes),
            "matches_at_primary": len(matches_p),
            "unmatched_original": len(unmatched_o_p),
            "unmatched_reann": len(unmatched_r_p),
        })

    agreement_per_threshold = {}
    for t, s in threshold_stats.items():
        tp = s["tp"]
        union = tp + s["fp_original_unmatched"] + s["fn_reann_unmatched"]
        jaccard = tp / union if union > 0 else 1.0
        agreement_per_threshold[f"iou_{t}"] = {
            "threshold": t,
            "matched_pairs": tp,
            "unmatched_original": s["fp_original_unmatched"],
            "unmatched_reann": s["fn_reann_unmatched"],
            "jaccard_agreement": round(jaccard, 4),
            "jaccard_agreement_pct": round(jaccard * 100, 2),
        }

    primary_stats = threshold_stats[args.primary_threshold]
    primary_union = primary_stats["tp"] + primary_stats["fp_original_unmatched"] \
                    + primary_stats["fn_reann_unmatched"]
    primary_jaccard = primary_stats["tp"] / primary_union if primary_union > 0 else 1.0

    report = {
        "settings": {
            "sampled_pages": len(plan),
            "primary_iou_threshold": args.primary_threshold,
            "iou_thresholds": args.iou_thresholds,
        },
        "totals": {
            "original_bboxes": total_orig,
            "reannotation_bboxes": total_reann,
            "empty_pages_both_sides": empty_pages_both,
            "missing_reannotation_files": missing_reann_files,
        },
        "primary_threshold_summary": {
            "threshold": args.primary_threshold,
            "jaccard_agreement_pct": round(primary_jaccard * 100, 2),
            "matched_pairs": primary_stats["tp"],
            "disagreements": {
                "boundary_pixel_cases": boundary_pixel,
                "split_convention_cases": split_convention,
                "genuine_omissions": genuine_omissions,
            },
        },
        "agreement_per_threshold": agreement_per_threshold,
        "iou_distribution": {
            "count": len(iou_distribution),
            "mean": round(sum(iou_distribution) / len(iou_distribution), 4)
                    if iou_distribution else None,
            "min": round(min(iou_distribution), 4) if iou_distribution else None,
            "max": round(max(iou_distribution), 4) if iou_distribution else None,
        },
        "per_page": page_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Agreement report (primary threshold: IoU >= {args.primary_threshold})")
    print(f"Sampled pages:               {len(plan)}")
    print(f"Original bboxes:             {total_orig}")
    print(f"Reannotation bboxes:         {total_reann}")
    if missing_reann_files:
        print(f"Missing reannotation files:  {missing_reann_files} (annotator did not complete these)")
    print()
    print(f"Jaccard agreement at IoU >= {args.primary_threshold}:  "
          f"{primary_jaccard * 100:.2f}%")
    print()
    print("Residual disagreements:")
    print(f"  Boundary-pixel cases:      {boundary_pixel}")
    print(f"  Split-convention cases:    {split_convention}")
    print(f"  Genuine omissions:         {genuine_omissions}")
    print()
    print("Agreement at other IoU thresholds:")
    for t in sorted(args.iou_thresholds):
        v = agreement_per_threshold[f"iou_{t}"]
        print(f"  IoU >= {t}:  Jaccard = {v['jaccard_agreement_pct']:.2f}%  "
              f"(matched {v['matched_pairs']})")
    print()
    print(f"Report written to: {args.output}")
    print()
    print("Summary:")
    print(f"At IoU >= {args.primary_threshold}, the Jaccard agreement reached "
          f"{primary_jaccard * 100:.2f}%. The residual disagreements consisted of "
          f"{boundary_pixel} boundary-pixel cases, {split_convention} "
          f"split-convention cases, and {genuine_omissions} genuine omissions.")


if __name__ == "__main__":
    main()
