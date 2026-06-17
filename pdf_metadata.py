import argparse
import csv
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF


def catalog_has_key(doc, key: str) -> bool:
    try:
        cat_xref = doc.pdf_catalog()
        if not cat_xref:
            return False
        return key in doc.xref_get_keys(cat_xref)
    except Exception:
        return False


def extract_one(pdf_path: Path) -> dict:
    with fitz.open(pdf_path) as doc:
        md = doc.metadata or {}
        first_page_text = doc[0].get_text("text") if doc.page_count > 0 else ""
        return {
            "filename": pdf_path.name,
            "page_count": doc.page_count,
            "creator": md.get("creator") or "",
            "producer": md.get("producer") or "",
            "title": md.get("title") or "",
            "has_markinfo": catalog_has_key(doc, "MarkInfo"),
            "has_structtreeroot": catalog_has_key(doc, "StructTreeRoot"),
            "first_page_has_text": len(first_page_text.strip()) > 0,
        }


def bucket(value: str) -> str:
    if not value:
        return "(NULL)"
    v = value.lower()
    if "hancom" in v or "hwp" in v:
        return "Hancom/HWP"
    if "acrobat distiller" in v:
        return "Acrobat Distiller"
    if "pscript" in v:
        return "PScript"
    if "microsoft" in v or "word" in v or "powerpoint" in v or "excel" in v:
        return "Microsoft Office"
    if "adobe" in v and "pdf library" in v:
        return "Adobe PDF Library"
    if "ghostscript" in v:
        return "Ghostscript"
    if "itext" in v:
        return "iText"
    if "pdfium" in v:
        return "PDFium"
    return value


def main():
    ap = argparse.ArgumentParser(description="PDF metadata / structure summary")
    ap.add_argument("--pdf-dir", default="data", help="folder of PDFs")
    ap.add_argument("--out-dir", default="results", help="output folder")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [extract_one(p) for p in sorted(Path(args.pdf_dir).glob("*.pdf"))]
    if not rows:
        print("no PDFs found")
        return

    fields = ["filename", "page_count", "creator", "producer", "title",
              "has_markinfo", "has_structtreeroot", "first_page_has_text"]
    with open(out_dir / "pdf_metadata.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    creator_counter = Counter(bucket(r["creator"]) for r in rows)
    producer_counter = Counter(bucket(r["producer"]) for r in rows)
    markinfo_false = sum(1 for r in rows if not r["has_markinfo"])
    struct_missing = sum(1 for r in rows if not r["has_structtreeroot"])
    text_yes = sum(1 for r in rows if r["first_page_has_text"])
    format_converted = sum(1 for r in rows
                           if r["first_page_has_text"] and not r["has_structtreeroot"])

    def fmt_dist(counter):
        return ", ".join(f"{k} {v}/{n}" for k, v in counter.most_common())

    lines = [f"# PDF metadata summary ({n} documents)\n",
             "## Per-document metadata\n",
             "| File | Pages | Creator | Producer | Title | MarkInfo | StructTreeRoot | Text layer |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        title = (r["title"] or "").replace("|", "\\|")[:40]
        creator = (r["creator"] or "(NULL)").replace("|", "\\|")[:40]
        producer = (r["producer"] or "(NULL)").replace("|", "\\|")[:40]
        lines.append(f"| {r['filename']} | {r['page_count']} | {creator} | {producer} | "
                     f"{title} | {r['has_markinfo']} | {r['has_structtreeroot']} | "
                     f"{r['first_page_has_text']} |")

    lines += ["\n## Distribution\n",
              f"- Creator: {fmt_dist(creator_counter)}",
              f"- Producer: {fmt_dist(producer_counter)}",
              f"- MarkInfo absent: {markinfo_false}/{n}",
              f"- StructTreeRoot absent: {struct_missing}/{n}",
              f"- First page has text: {text_yes}/{n}",
              f"- Format-converted (text and no structure tree): {format_converted}/{n}"]

    (out_dir / "pdf_metadata_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("Summary")
    print(f"Documents: {n}")
    print(f"Creator dist: {fmt_dist(creator_counter)}")
    print(f"Producer dist: {fmt_dist(producer_counter)}")
    print(f"MarkInfo absent: {markinfo_false}/{n}")
    print(f"StructTreeRoot absent: {struct_missing}/{n}")
    print(f"Format-converted (text+nostruct): {format_converted}/{n}")
    print(f"\n[written] {out_dir}/pdf_metadata.csv, pdf_metadata_summary.md")


if __name__ == "__main__":
    main()
