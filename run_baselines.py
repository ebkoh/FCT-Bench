import os
import json
import time
import argparse
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Render scale used by the DL detectors.
RENDER_SCALE = 2


# Camelot-py
def run_camelot(pdf_path):
    import camelot
    import fitz

    tables_out = []
    with fitz.open(pdf_path) as doc:
        page_heights = {i + 1: doc[i].rect.height for i in range(len(doc))}

    try:
        camelot_tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
    except Exception as e:
        print(f"    [Camelot] Error: {e}")
        return tables_out

    for i, t in enumerate(camelot_tables):
        page = int(t.page)
        x1, y1, x2, y2 = t._bbox
        page_height = page_heights.get(page)
        if page_height:
            bbox_pdf = [x1, page_height - y2, x2, page_height - y1]
        else:
            bbox_pdf = [x1, y1, x2, y2]
        tables_out.append({
            "page": page,
            "table_id": i + 1,
            "bbox_pdf": bbox_pdf,
            "confidence": round(t.accuracy / 100.0, 3) if hasattr(t, 'accuracy') else None,
        })

    return tables_out


# pdfplumber
def run_pdfplumber(pdf_path):
    import pdfplumber

    tables_out = []
    tid = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            found = page.find_tables()
            for t in found:
                tid += 1
                bbox = t.bbox  # (x0, top, x1, bottom) PDF coordinates
                tables_out.append({
                    "page": page_idx + 1,
                    "table_id": tid,
                    "bbox_pdf": list(bbox),
                    "confidence": None,
                })

    return tables_out


# PyMuPDF
def run_pymupdf(pdf_path):
    import fitz

    tables_out = []
    tid = 0

    doc = fitz.open(pdf_path)
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        try:
            tabs = page.find_tables()
            for t in tabs.tables:
                tid += 1
                bbox = t.bbox  # (x0, y0, x1, y1) PDF coordinates
                tables_out.append({
                    "page": page_idx + 1,
                    "table_id": tid,
                    "bbox_pdf": list(bbox),
                    "confidence": None,
                })
        except Exception as e:
            print(f"    [PyMuPDF] Page {page_idx+1} error: {e}")
    doc.close()

    return tables_out


# Tabula-py
def run_tabula(pdf_path):
    import subprocess
    import fitz
    import tabula

    tables_out = []
    tid = 0

    tabula_dir = os.path.dirname(tabula.__file__)
    jar_path = None
    for root, dirs, files in os.walk(tabula_dir):
        for f in files:
            if f.endswith('.jar'):
                jar_path = os.path.join(root, f)
                break
        if jar_path:
            break

    if not jar_path:
        print("    [Tabula] jar not found")
        return tables_out

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    for page_num in range(1, total_pages + 1):
        try:
            cmd = [
                "java", "-jar", jar_path,
                "-f", "JSON",
                "-l",  # lattice mode
                "--pages", str(page_num),
                pdf_path
            ]

            result = subprocess.run(
                cmd, capture_output=True, timeout=30,
                encoding='utf-8', errors='replace'
            )

            if result.returncode != 0 or not result.stdout.strip():
                continue

            raw = json.loads(result.stdout)

            for table_data in raw:
                top = table_data.get("top", 0)
                left = table_data.get("left", 0)
                width = table_data.get("width", 0)
                height = table_data.get("height", 0)

                if width > 0 and height > 0:
                    tid += 1
                    tables_out.append({
                        "page": page_num,
                        "table_id": tid,
                        "bbox_pdf": [left, top, left + width, top + height],
                        "confidence": None,
                    })

        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue

    return tables_out


# Table Transformer
_table_transformer_model = None
_table_transformer_processor = None

def _load_table_transformer():
    global _table_transformer_model, _table_transformer_processor
    if _table_transformer_model is not None:
        return

    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    model_name = "microsoft/table-transformer-detection"
    print(f"  Loading {model_name}...")
    _table_transformer_processor = AutoImageProcessor.from_pretrained(model_name)
    _table_transformer_model = TableTransformerForObjectDetection.from_pretrained(model_name)
    _table_transformer_model.eval()
    print("  Model loaded.")


def run_table_transformer(pdf_path):
    import torch
    from PIL import Image
    import fitz

    _load_table_transformer()

    tables_out = []
    tid = 0
    threshold = 0.7

    doc = fitz.open(pdf_path)
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        inputs = _table_transformer_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = _table_transformer_model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]])  # (height, width)
        results = _table_transformer_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            tid += 1
            x1, y1, x2, y2 = box.tolist()

            # pixel -> PDF coordinates
            page_rect = page.rect
            pdf_x1 = (x1 / pix.width) * page_rect.width
            pdf_y1 = (y1 / pix.height) * page_rect.height
            pdf_x2 = (x2 / pix.width) * page_rect.width
            pdf_y2 = (y2 / pix.height) * page_rect.height

            tables_out.append({
                "page": page_idx + 1,
                "table_id": tid,
                "bbox_pdf": [round(pdf_x1, 2), round(pdf_y1, 2),
                             round(pdf_x2, 2), round(pdf_y2, 2)],
                "confidence": round(score.item(), 3),
            })

    doc.close()
    return tables_out


# DocLayout-YOLO
_doclayout_model = None

def _load_doclayout_yolo():
    global _doclayout_model
    if _doclayout_model is not None:
        return

    from doclayout_yolo import YOLOv10

    # use a local checkpoint if present, otherwise download the pretrained one
    model_path = os.path.join(SCRIPT_DIR, "models",
                              "doclayout_yolo_docstructbench_imgsz1024.pt")
    if os.path.exists(model_path):
        print(f"  Loading DocLayout-YOLO from {model_path}...")
        _doclayout_model = YOLOv10(model_path)
    else:
        print("  Local weight not found; downloading pretrained model...")
        _doclayout_model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench")
    print("  Model loaded.")


def run_doclayout_yolo(pdf_path):
    import fitz
    from PIL import Image

    _load_doclayout_yolo()

    tables_out = []
    tid = 0
    confidence_threshold = 0.25

    # DocStructBench class ids: 0 title, 1 text, 2 abandon, 3 figure,
    # 4 figure_caption, 5 table, 6 table_caption, 7 table_footnote,
    # 8 isolate_formula, 9 formula_caption
    TABLE_CLASS_ID = 5

    doc = fitz.open(pdf_path)
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        results = _doclayout_model.predict(
            img, imgsz=1024,
            conf=confidence_threshold,
            device="cpu"
        )

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id != TABLE_CLASS_ID:
                    continue

                tid += 1
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = box.conf[0].item()

                # pixel -> PDF coordinates
                page_rect = page.rect
                pdf_x1 = (x1 / pix.width) * page_rect.width
                pdf_y1 = (y1 / pix.height) * page_rect.height
                pdf_x2 = (x2 / pix.width) * page_rect.width
                pdf_y2 = (y2 / pix.height) * page_rect.height

                tables_out.append({
                    "page": page_idx + 1,
                    "table_id": tid,
                    "bbox_pdf": [round(pdf_x1, 2), round(pdf_y1, 2),
                                 round(pdf_x2, 2), round(pdf_y2, 2)],
                    "confidence": round(score, 3),
                })

    doc.close()
    return tables_out


TOOLS = {
    "camelot": ("Camelot-py", run_camelot),
    "pdfplumber": ("pdfplumber", run_pdfplumber),
    "pymupdf": ("PyMuPDF", run_pymupdf),
    "tabula": ("Tabula-py", run_tabula),
    "table_transformer": ("Table_Transformer_Detection", run_table_transformer),
    "doclayout_yolo": ("DocLayout-YOLO", run_doclayout_yolo),
}


def run_tool_on_pdf(tool_key, pdf_path, output_dir):
    tool_name, run_fn = TOOLS[tool_key]
    pdf_name = os.path.basename(pdf_path)
    doc_name = os.path.splitext(pdf_name)[0]

    import fitz
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    start = time.time()
    try:
        tables = run_fn(pdf_path)
        error = None
    except Exception as e:
        tables = []
        error = str(e)
        traceback.print_exc()

    elapsed = time.time() - start

    result = {
        "tool": tool_name,
        "document": pdf_name,
        "total_pages": total_pages,
        "total_tables_detected": len(tables),
        "processing_time_sec": round(elapsed, 2),
        "error": error,
        "tables": tables,
    }

    tool_dir = os.path.join(output_dir, tool_name)
    os.makedirs(tool_dir, exist_ok=True)
    json_path = os.path.join(tool_dir, f"{doc_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run baseline table detectors")
    parser.add_argument("--pdf-dir", default="data", help="folder of input PDFs")
    parser.add_argument("--out", default="results_baselines", help="output folder")
    parser.add_argument("--tool", default=None, choices=list(TOOLS.keys()),
                        help="run a single tool")
    parser.add_argument("--pdf", default=None, help="run a single PDF file name")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tool_keys = [args.tool] if args.tool else list(TOOLS.keys())
    if args.pdf:
        pdf_files = [args.pdf]
    else:
        pdf_files = sorted(f for f in os.listdir(args.pdf_dir) if f.endswith(".pdf"))

    for tool_key in tool_keys:
        tool_name = TOOLS[tool_key][0]
        print(f"\n[{tool_name}]")
        total = 0
        for i, pdf_name in enumerate(pdf_files, 1):
            pdf_path = os.path.join(args.pdf_dir, pdf_name)
            if not os.path.exists(pdf_path):
                continue
            print(f"  [{i}/{len(pdf_files)}] {pdf_name}...", end=" ", flush=True)
            result = run_tool_on_pdf(tool_key, pdf_path, args.out)
            status = (f"{result['total_tables_detected']} tables"
                      if not result["error"] else f"ERROR: {result['error'][:50]}")
            print(f"{status} ({result['processing_time_sec']:.1f}s)")
            total += result["total_tables_detected"]
        print(f"  -> {total} tables")


if __name__ == "__main__":
    main()
