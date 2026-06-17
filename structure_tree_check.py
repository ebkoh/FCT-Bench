import argparse
import csv
import re
from collections import Counter
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

REF_RE = re.compile(r"(\d+)\s+\d+\s+R")
NAMEPAIR_RE = re.compile(r"/([^\s/<>\[\]()]+)\s*/([^\s/<>\[\]()]+)")


def ref_nums(val):
    return [int(m.group(1)) for m in REF_RE.finditer(val or "")]


def get_structtreeroot_xref(doc):
    cat = doc.pdf_catalog()
    if cat <= 0 or "StructTreeRoot" not in doc.xref_get_keys(cat):
        return None
    kind, val = doc.xref_get_key(cat, "StructTreeRoot")
    if kind == "xref":
        nums = ref_nums(val)
        return nums[0] if nums else None
    return None  # inline StructTreeRoot is non-standard; treat as absent


def parse_rolemap(doc, st_xref):
    rm = {}
    if "RoleMap" not in doc.xref_get_keys(st_xref):
        return rm
    kind, val = doc.xref_get_key(st_xref, "RoleMap")
    if kind == "xref":
        nums = ref_nums(val)
        if not nums:
            return rm
        rmx = nums[0]
        for k in doc.xref_get_keys(rmx):
            vk, vv = doc.xref_get_key(rmx, k)
            if vk == "name":
                rm["/" + k] = vv
    elif kind == "dict":
        for role, target in NAMEPAIR_RE.findall(val):
            rm["/" + role] = "/" + target
    return rm


def child_refs(doc, xref):
    if "K" not in doc.xref_get_keys(xref):
        return []
    kind, val = doc.xref_get_key(xref, "K")
    if kind in ("xref", "array", "dict"):
        return ref_nums(val)
    return []  # 'int' (MCID leaf) / 'null'


def traverse(doc, st_xref, rolemap):
    type_counter = Counter()
    table_count = 0
    visited = set()
    stack = list(child_refs(doc, st_xref))
    while stack:
        x = stack.pop()
        if x in visited:
            continue
        visited.add(x)
        try:
            keys = doc.xref_get_keys(x)
        except Exception:
            continue
        if "S" not in keys:
            continue  # not a StructElem (OBJR / MCR / page / content)
        sk, sv = doc.xref_get_key(x, "S")
        if sk != "name":
            continue
        type_counter[sv] += 1
        if rolemap.get(sv, sv) == "/Table":
            table_count += 1
        stack.extend(child_refs(doc, x))
    return type_counter, table_count


def inspect(path):
    rec = {"filename": path.name, "has_structtreeroot": False,
           "table_structelem_count": 0, "has_rolemap_table_mapping": False,
           "method": "recursive_structelem_traversal"}
    types, table_roles, error = Counter(), [], None
    try:
        with fitz.open(str(path)) as doc:
            st = get_structtreeroot_xref(doc)
            if st is None:
                return rec, types, error
            rec["has_structtreeroot"] = True
            rolemap = parse_rolemap(doc, st)
            table_roles = [r for r, t in rolemap.items() if t == "/Table" and r != "/Table"]
            rec["has_rolemap_table_mapping"] = len(table_roles) > 0
            types, tcount = traverse(doc, st, rolemap)
            rec["table_structelem_count"] = tcount
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return rec, types, error


def main():
    ap = argparse.ArgumentParser(description="StructTreeRoot /Table detection")
    ap.add_argument("--pdf-dir", default="data", help="folder of PDFs")
    ap.add_argument("--out", default="table_structure_check.csv", help="output CSV")
    args = ap.parse_args()

    rows = []
    for path in sorted(Path(args.pdf_dir).glob("*.pdf")):
        rec, types, error = inspect(path)
        rows.append(rec)
        err = f"  ERROR: {error}" if error else ""
        print(f"{path.name}{err}")
        print(f"  StructTreeRoot present : {rec['has_structtreeroot']}")
        print(f"  /Table StructElems     : {rec['table_structelem_count']}")
        print(f"  RoleMap custom->/Table : {rec['has_rolemap_table_mapping']}")
        if types:
            top = ", ".join(f"{k}:{v}" for k, v in types.most_common(14))
            print(f"  structure types (top)  : {top}")

    fields = ["filename", "has_structtreeroot", "table_structelem_count",
              "has_rolemap_table_mapping", "method"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
