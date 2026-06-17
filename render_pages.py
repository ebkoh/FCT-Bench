import argparse
from pathlib import Path

import fitz  # PyMuPDF


def render_pdf(pdf_path: Path, out_dir: Path, scale: int = 2, fmt: str = "png") -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    mat = fitz.Matrix(scale, scale)
    doc = fitz.open(str(pdf_path))
    try:
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=mat, alpha=False)
            pix.save(str(out_dir / f"page_{i + 1:04d}.{fmt}"))
        return len(doc)
    finally:
        doc.close()


def main():
    ap = argparse.ArgumentParser(description="Rasterize PDF pages")
    ap.add_argument("--pdf", help="single PDF file")
    ap.add_argument("--pdf-dir", help="folder of PDFs (one subfolder of images per PDF)")
    ap.add_argument("--out-dir", default="pages", help="output folder")
    ap.add_argument("--scale", type=int, default=2, help="render scale factor")
    ap.add_argument("--format", default="png", choices=["png", "jpg"])
    args = ap.parse_args()

    out = Path(args.out_dir)
    if args.pdf:
        n = render_pdf(Path(args.pdf), out, args.scale, args.format)
        print(f"{args.pdf} -> {n} pages")
    elif args.pdf_dir:
        for p in sorted(Path(args.pdf_dir).glob("*.pdf")):
            n = render_pdf(p, out / p.stem, args.scale, args.format)
            print(f"{p.name} -> {n} pages")
    else:
        ap.error("provide --pdf or --pdf-dir")


if __name__ == "__main__":
    main()
