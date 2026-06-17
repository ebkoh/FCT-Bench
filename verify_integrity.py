import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Annotation integrity checks")
    ap.add_argument("--annotations", default="annotations",
                    help="root with <subset>/<doc>/page_*.txt")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--out", default="data_integrity_report.txt")
    args = ap.parse_args()

    ann_root = Path(args.annotations)
    ann_files = sorted(ann_root.glob("*/*/page_*.txt"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    man_pages = {r["doc_id"]: r["page_count"] for r in manifest}

    per_doc_files = defaultdict(list)
    for f in ann_files:
        per_doc_files[f.parent.name].append(f)

    total_files = len(ann_files)
    total_boxes = 0
    empty_files = 0
    bad_class, bad_range, nonfinite, bad_wh, dup_boxes = [], [], [], [], []
    classes_seen = set()

    for f in ann_files:
        lines = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            empty_files += 1
        boxes_here = []
        for i, ln in enumerate(lines, 1):
            parts = ln.split()
            total_boxes += 1
            if len(parts) != 5:
                bad_range.append(f"{f}:{i} (ncols={len(parts)})")
                continue
            cls = parts[0]
            classes_seen.add(cls)
            try:
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                nonfinite.append(f"{f}:{i}")
                continue
            if cls != "0":
                bad_class.append(f"{f}:{i} class={cls}")
            xc, yc, w, h = vals
            if any(math.isnan(v) or math.isinf(v) for v in vals):
                nonfinite.append(f"{f}:{i}")
            if any(v < 0.0 or v > 1.0 for v in vals):
                bad_range.append(f"{f}:{i} {vals}")
            if w <= 0 or h <= 0:
                bad_wh.append(f"{f}:{i} w={w} h={h}")
            boxes_here.append((xc, yc, w, h))
        # duplicates within a page (near-identical, tol 1e-3)
        for a in range(len(boxes_here)):
            for b in range(a + 1, len(boxes_here)):
                if all(abs(boxes_here[a][k] - boxes_here[b][k]) < 1e-3 for k in range(4)):
                    dup_boxes.append(f"{f} boxes#{a+1},{b+1}")

    # page-id contiguity and per-doc file count vs manifest
    contiguity_bad, count_mismatch = [], []
    for doc, files in per_doc_files.items():
        nums = sorted(int(f.stem.split("_")[1]) for f in files)
        if nums != list(range(1, len(nums) + 1)):
            contiguity_bad.append(f"{doc}: {nums[:3]}..{nums[-3:]}")
        if doc in man_pages and len(files) != man_pages[doc]:
            count_mismatch.append(f"{doc}: files={len(files)} manifest={man_pages[doc]}")

    man_total = sum(man_pages.values())
    nonempty_files = total_files - empty_files

    log = []
    def p(s=""):
        print(s)
        log.append(s)

    p("Annotation integrity checks")
    p(f"C1  files == sum(manifest pages)   : files={total_files} manifest={man_total}  "
      f"-> {'PASS' if total_files == man_total else 'FAIL'}")
    p(f"C2  per-doc files == manifest pages: {'PASS' if not count_mismatch else 'FAIL ' + str(count_mismatch)}")
    p(f"C3  total boxes (non-empty lines)  : {total_boxes}")
    p(f"C4  all class index == 0           : classes={sorted(classes_seen)} viol={len(bad_class)}  "
      f"-> {'PASS' if not bad_class else 'FAIL'}")
    p(f"C5  coords finite & in [0,1]       : range={len(bad_range)} nonfinite={len(nonfinite)}  "
      f"-> {'PASS' if not bad_range and not nonfinite else 'FAIL'}")
    p(f"C6  width>0 & height>0             : viol={len(bad_wh)}  -> {'PASS' if not bad_wh else 'FAIL'}")
    p(f"C7  page-id contiguity (1..N/doc)  : {'PASS' if not contiguity_bad else 'FAIL ' + str(contiguity_bad)}")
    p(f"C8  empty (table-negative) files   : empty={empty_files} nonempty={nonempty_files} "
      f"(sum={total_files})  -> {'PASS' if empty_files + nonempty_files == total_files else 'FAIL'}")
    p(f"C9  duplicate boxes within a page  : {len(dup_boxes)}  -> {'PASS' if not dup_boxes else 'FAIL'}")
    for label, lst in [("C4", bad_class), ("C5 range", bad_range), ("C5 nonfinite", nonfinite),
                       ("C6", bad_wh), ("C9", dup_boxes)]:
        if lst:
            p(f"  -- {label} violations (first 10): " + "; ".join(lst[:10]))

    Path(args.out).write_text("\n".join(log) + "\n", encoding="utf-8")
    p(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
