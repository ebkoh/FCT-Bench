import argparse
import json
import math
import random
import sys
from pathlib import Path


def allocate_pages(docs, target_total):
    total_pages = sum(pc for _, pc in docs)

    raw = [(doc_id, pc, pc / total_pages * target_total) for doc_id, pc in docs]

    floor_alloc = [(doc_id, pc, int(math.floor(frac)), frac - math.floor(frac))
                   for doc_id, pc, frac in raw]

    allocation = [(doc_id, pc, max(1, fa), rem) for doc_id, pc, fa, rem in floor_alloc]

    allocation = [(doc_id, pc, min(alloc, pc), rem) for doc_id, pc, alloc, rem in allocation]

    current_total = sum(alloc for _, _, alloc, _ in allocation)
    diff = target_total - current_total

    if diff > 0:
        candidates = sorted(
            [(doc_id, pc, alloc, rem) for doc_id, pc, alloc, rem in allocation if alloc < pc],
            key=lambda x: -x[3],
        )
        i = 0
        while diff > 0 and i < len(candidates):
            doc_id, pc, alloc, rem = candidates[i]
            if alloc < pc:
                candidates[i] = (doc_id, pc, alloc + 1, rem)
                diff -= 1
            i += 1
            if i >= len(candidates):
                i = 0  # wrap around if needed
                candidates = [(d, p, a, r) for d, p, a, r in candidates if a < p]
                if not candidates:
                    break
        allocation_map = {doc_id: alloc for doc_id, _, alloc, _ in candidates}
        allocation = [(d, p, allocation_map.get(d, a), r) for d, p, a, r in allocation]
    elif diff < 0:
        candidates = sorted(
            [(doc_id, pc, alloc, rem) for doc_id, pc, alloc, rem in allocation if alloc > 1],
            key=lambda x: x[3],
        )
        i = 0
        while diff < 0 and i < len(candidates):
            doc_id, pc, alloc, rem = candidates[i]
            if alloc > 1:
                candidates[i] = (doc_id, pc, alloc - 1, rem)
                diff += 1
            i += 1
        allocation_map = {doc_id: alloc for doc_id, _, alloc, _ in candidates}
        allocation = [(d, p, allocation_map.get(d, a), r) for d, p, a, r in allocation]

    return [(doc_id, alloc) for doc_id, _, alloc, _ in allocation]


def main():
    parser = argparse.ArgumentParser(
        description="Sample pages for independent re-annotation."
    )
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path to manifest.json (must contain doc_id and page_count)")
    parser.add_argument("--target-pages", type=int, default=100,
                        help="Target total number of pages (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for blind_subset_plan.json")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    docs = [(r["doc_id"], r["page_count"]) for r in manifest]
    docs.sort(key=lambda x: x[0])  # deterministic order

    total_pages = sum(pc for _, pc in docs)
    if not docs:
        print("ERROR: manifest contains no documents", file=sys.stderr)
        sys.exit(1)
    if args.target_pages < len(docs):
        print("ERROR: target-pages must be at least the number of documents", file=sys.stderr)
        sys.exit(1)
    if args.target_pages > total_pages:
        print("ERROR: target-pages cannot exceed the total page count", file=sys.stderr)
        sys.exit(1)

    print(f"Manifest: {len(docs)} documents, {total_pages} total pages")
    print(f"Target sample: {args.target_pages} pages "
          f"({args.target_pages / total_pages * 100:.1f}%)")
    print(f"Random seed: {args.seed}")
    print()

    allocation = allocate_pages(docs, args.target_pages)

    print("Per-document allocation:")
    print(f"  {'doc_id':<22} {'pages':>6} {'sample':>6} {'pct':>6}")
    alloc_dict = dict(allocation)
    doc_pages = dict(docs)
    for doc_id in sorted(alloc_dict):
        pc = doc_pages[doc_id]
        alloc = alloc_dict[doc_id]
        pct = alloc / pc * 100
        print(f"  {doc_id:<22} {pc:>6} {alloc:>6} {pct:>5.1f}%")
    total_alloc = sum(alloc_dict.values())
    print(f"  {'TOTAL':<22} {total_pages:>6} {total_alloc:>6}")

    rng = random.Random(args.seed)
    selected_pages = []
    for doc_id in sorted(alloc_dict):
        pc = doc_pages[doc_id]
        alloc = alloc_dict[doc_id]
        pages = sorted(rng.sample(range(1, pc + 1), alloc))
        for p in pages:
            selected_pages.append({"doc_id": doc_id, "page": p})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(selected_pages, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print()
    print(f"Wrote {len(selected_pages)} selected pages to: {args.output}")


if __name__ == "__main__":
    main()
