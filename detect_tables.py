import os
import json
import time
import argparse

import cv2

import config
from morphology import TableDetector, ImageExtractor, TableFilter


def detect_tables_single_pdf(pdf_path: str, output_dir: str) -> dict:
    det_cfg = config.TABLE_DETECTION
    pdf_name = os.path.basename(pdf_path)
    doc_name = os.path.splitext(pdf_name)[0]

    tables_dir = os.path.join(output_dir, doc_name, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    detector = TableDetector(pdf_path, render_scale=det_cfg.get("render_scale", 2))
    img_extractor = ImageExtractor(pdf_path)
    table_filter = TableFilter(pdf_path)

    total_pages = detector.get_total_pages()

    result = {
        "tool": "OpenCV_Morphology",
        "document": pdf_name,
        "total_pages": total_pages,
        "total_tables_detected": 0,
        "processing_time_sec": 0,
        "tables": [],
    }

    global_table_id = 0
    start_time = time.time()

    for page_idx in range(total_pages):
        detector.load_page(page_idx)
        img_w = detector.img_width
        img_h = detector.img_height

        image_regions = img_extractor.extract_image_regions(
            page_num=page_idx,
            img_width=img_w,
            img_height=img_h,
            min_width=det_cfg.get("min_image_width", 150),
            min_height=det_cfg.get("min_image_height", 150),
        )

        tables = detector.detect_tables(
            image_regions=image_regions,
            min_table_area=det_cfg.get("min_table_area", 1000),
            min_width=det_cfg.get("min_width", 50),
            min_height=det_cfg.get("min_height", 30),
            h_kernel_length=det_cfg.get("h_kernel_length", 40),
            v_kernel_length=det_cfg.get("v_kernel_length", 40),
            padding=det_cfg.get("padding", 5),
            overlap_threshold=det_cfg.get("overlap_threshold", 0.3),
        )

        text_regions = table_filter.extract_text_blocks(
            page_num=page_idx,
            img_width=img_w,
            img_height=img_h,
            x_tolerance=config.PDFPLUMBER.get("x_tolerance", 3),
            y_tolerance=config.PDFPLUMBER.get("y_tolerance", 3),
        )

        flt_cfg = config.TABLE_FILTER
        tables = table_filter.filter_tables(
            tables=tables,
            text_regions=text_regions,
            min_text_overlap_ratio=flt_cfg.get("min_text_overlap_ratio", 0.1),
            require_multi_line=flt_cfg.get("require_multi_line", True),
        )

        for table in tables:
            global_table_id += 1
            bbox = table["bbox"]

            filename = f"page{page_idx + 1:04d}_table{global_table_id:04d}.png"
            save_path = os.path.join(tables_dir, filename)
            crop = detector.img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cv2.imwrite(save_path, crop)

            result["tables"].append({
                "page": page_idx + 1,
                "table_id": global_table_id,
                "bbox": [int(v) for v in bbox],
                "width": int(table["width"]),
                "height": int(table["height"]),
                "area": int(table["area"]),
                "confidence": None,
                "image_path": f"{doc_name}/tables/{filename}",
            })

        if tables:
            print(f"  p.{page_idx + 1:4d}: {len(tables)} tables")

    elapsed = time.time() - start_time
    result["total_tables_detected"] = global_table_id
    result["processing_time_sec"] = round(elapsed, 2)

    json_path = os.path.join(output_dir, f"{doc_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch morphology table detection")
    parser.add_argument("--pdf-dir", default="data", help="folder of input PDFs")
    parser.add_argument("--out", default="results_morphology", help="output folder")
    parser.add_argument("--pdf", default=None, help="process a single PDF file name")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.pdf:
        pdf_files = [args.pdf]
    else:
        pdf_files = sorted(f for f in os.listdir(args.pdf_dir) if f.endswith(".pdf"))

    total_tables = 0
    for i, pdf_name in enumerate(pdf_files, 1):
        pdf_path = os.path.join(args.pdf_dir, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"[skip] not found: {pdf_path}")
            continue
        print(f"[{i}/{len(pdf_files)}] {pdf_name}")
        result = detect_tables_single_pdf(pdf_path, args.out)
        total_tables += result["total_tables_detected"]
        print(f"  -> {result['total_tables_detected']} tables ({result['processing_time_sec']:.1f}s)")

    print(f"\nTotal tables detected: {total_tables}")


if __name__ == "__main__":
    main()
