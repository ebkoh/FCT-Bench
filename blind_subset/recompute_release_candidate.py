import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from compute_agreement import (  
    parse_yolo_file, greedy_match, find_original_path,
    find_reannotation_path, classify_unmatched,
)

PLAN = HERE / "blind_subset_plan.json"
ORIG_ROOT = PROJECT_ROOT / "annotations"
REANN_DIR = HERE / "blind_reannotation" / "labels"
OUT = HERE / "agreement_report_release_candidate.json"

IOU_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
PRIMARY = 0.5

PATCH_PAGES = {("10_pump_station", 69), ("ISMS_P", 197), ("ISMS_P", 237)}


def run(use_release_candidate: bool):
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    thr_stats = {t: {"tp": 0, "uo": 0, "ur": 0} for t in IOU_THRESHOLDS}
    per_page, ious, omission_locs = [], [], []
    total_orig = total_reann = 0
    boundary = split = omissions = empty_both = 0

    for e in plan:
        doc, page = e["doc_id"], e["page"]
        orig = parse_yolo_file(find_original_path(ORIG_ROOT, doc, page))
        reann = parse_yolo_file(find_reannotation_path(REANN_DIR, doc, page))
        if use_release_candidate and (doc, page) in PATCH_PAGES:
            orig = []  # pre-correction state

        total_orig += len(orig)
        total_reann += len(reann)
        if not orig and not reann:
            empty_both += 1

        for t in IOU_THRESHOLDS:
            m, uo, ur = greedy_match(orig, reann, iou_threshold=t)
            thr_stats[t]["tp"] += len(m)
            thr_stats[t]["uo"] += len(uo)
            thr_stats[t]["ur"] += len(ur)

        m, uo, ur = greedy_match(orig, reann, iou_threshold=PRIMARY)
        for _, _, v in m:
            ious.append(v)
            if v < 0.9:
                boundary += 1
        so, oo = classify_unmatched(uo, orig, reann)
        sr, orr = classify_unmatched(ur, reann, orig)
        split += so + sr
        omissions += oo + orr
        if ur:
            omission_locs.append({"doc_id": doc, "page": page, "omitted_in_original": len(ur)})
        per_page.append({"doc_id": doc, "page": page,
                         "orig_bboxes": len(orig), "reann_bboxes": len(reann),
                         "matches_at_primary": len(m),
                         "unmatched_original": len(uo), "unmatched_reann": len(ur)})

    agreement = {}
    for t, s in thr_stats.items():
        union = s["tp"] + s["uo"] + s["ur"]
        j = s["tp"] / union if union else 1.0
        agreement[f"iou_{t}"] = {"threshold": t, "matched_pairs": s["tp"],
                                 "unmatched_original": s["uo"], "unmatched_reann": s["ur"],
                                 "jaccard_agreement": round(j, 4),
                                 "jaccard_agreement_pct": round(j * 100, 2)}

    ps = thr_stats[PRIMARY]
    punion = ps["tp"] + ps["uo"] + ps["ur"]
    pj = ps["tp"] / punion if punion else 1.0
    dist = {}
    if ious:
        dist = {"count": len(ious), "mean": round(statistics.mean(ious), 4),
                "median": round(statistics.median(ious), 4),
                "min": round(min(ious), 4), "max": round(max(ious), 4)}

    return {
        "settings": {"sampled_pages": len(plan), "primary_iou_threshold": PRIMARY,
                     "iou_thresholds": IOU_THRESHOLDS,
                     "gt_basis": "release_candidate" if use_release_candidate else "current_v1.1_patched",
                     "note": ("02_tunnel removed (2-up->1-up reconversion); "
                              "compared against pre-patch GT to avoid circularity")
                     if use_release_candidate else
                     "02_tunnel removed; compared against current patched GT (CIRCULAR - reference only)"},
        "totals": {"original_bboxes": total_orig, "reannotation_bboxes": total_reann,
                   "empty_pages_both_sides": empty_both, "missing_reannotation_files": 0},
        "primary_threshold_summary": {"threshold": PRIMARY,
                                      "jaccard_agreement_pct": round(pj * 100, 2),
                                      "matched_pairs": ps["tp"],
                                      "disagreements": {"boundary_pixel_cases": boundary,
                                                        "split_convention_cases": split,
                                                        "genuine_omissions": omissions}},
        "agreement_per_threshold": agreement,
        "iou_distribution": dist,
        "genuine_omission_locations": omission_locs,
        "per_page": per_page,
    }


def main():
    rc = run(use_release_candidate=True)
    cur = run(use_release_candidate=False)

    OUT.write_text(json.dumps(rc, ensure_ascii=False, indent=2), encoding="utf-8")

    ps = rc["primary_threshold_summary"]
    d = rc["iou_distribution"]
    print("subset vs RELEASE-CANDIDATE GT")
    print(f"  sampled_pages        : {rc['settings']['sampled_pages']}")
    print(f"  Jaccard@0.5          : {ps['jaccard_agreement_pct']} %")
    print(f"  matched pairs        : {ps['matched_pairs']}")
    print(f"  genuine omissions    : {ps['disagreements']['genuine_omissions']}")
    print(f"  boundary/split cases : {ps['disagreements']['boundary_pixel_cases']}"
          f" / {ps['disagreements']['split_convention_cases']}")
    print(f"  original / reann box : {rc['totals']['original_bboxes']}"
          f" / {rc['totals']['reannotation_bboxes']}")
    print(f"  IoU mean / median    : {d.get('mean')} / {d.get('median')}")
    print(f"  IoU min / max        : {d.get('min')} / {d.get('max')}")
    print(f"\n  Jaccard@all IoU      : "
          + ", ".join(f"{k.split('_')[1]}={v['jaccard_agreement_pct']}%"
                      for k, v in rc["agreement_per_threshold"].items()))
    print("\n  genuine omission locations:")
    for o in rc["genuine_omission_locations"]:
        print(f"    - {o['doc_id']} p{o['page']}  ({o['omitted_in_original']} box)")
    print(f"\n  [cross-check] same pages vs CURRENT patched GT (circular): "
          f"{cur['primary_threshold_summary']['jaccard_agreement_pct']}%, "
          f"omissions={cur['primary_threshold_summary']['disagreements']['genuine_omissions']}")
    print(f"\n  written: {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
