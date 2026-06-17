import fitz  # PyMuPDF


class ImageExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_image_regions(self, page_num: int,
                              img_width: int, img_height: int,
                              min_width: int = 150, min_height: int = 150,
                              merge_threshold_x: int = 5,
                              merge_threshold_y: int = 5) -> list:
        doc = fitz.open(self.pdf_path)
        page = doc[page_num]

        raw_regions = self._collect_image_positions(doc, page, img_width, img_height)

        doc.close()

        if not raw_regions:
            return []

        merged = self._merge_adjacent_images(
            raw_regions, merge_threshold_x, merge_threshold_y
        )
        merged = self._merge_contained_images(merged)

        # keep only regions large enough (after merge or by original size)
        filtered = [
            r for r in merged
            if (r['width'] >= min_width and r['height'] >= min_height)
            or (r.get('orig_width', 0) >= min_width and r.get('orig_height', 0) >= min_height)
        ]

        return filtered

    def _collect_image_positions(self, doc, page, img_width, img_height) -> list:
        images = page.get_images()
        page_rect = page.rect
        regions = []
        processed_xrefs = set()

        for img in images:
            xref = img[0]
            if xref in processed_xrefs:
                continue
            processed_xrefs.add(xref)

            # original raster size
            try:
                base_image = doc.extract_image(xref)
                orig_width = base_image.get("width", 0)
                orig_height = base_image.get("height", 0)
            except Exception:
                orig_width, orig_height = 0, 0

            img_rects = page.get_image_rects(xref)
            if not img_rects:
                continue

            for rect in img_rects:
                x1 = int((rect.x0 / page_rect.width) * img_width)
                y1 = int((rect.y0 / page_rect.height) * img_height)
                x2 = int((rect.x1 / page_rect.width) * img_width)
                y2 = int((rect.y1 / page_rect.height) * img_height)

                regions.append({
                    'bbox': (x1, y1, x2, y2),
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'area': (x2 - x1) * (y2 - y1),
                    'orig_width': orig_width,
                    'orig_height': orig_height,
                    'xref': xref,
                })

        return regions

    def _merge_adjacent_images(self, regions: list,
                               threshold_x: int, threshold_y: int) -> list:
        if not regions:
            return []

        # group by x-range
        groups = []
        used = set()
        for i, r1 in enumerate(regions):
            if i in used:
                continue
            x1_1, _, x2_1, _ = r1['bbox']
            group = [r1]
            used.add(i)

            for j, r2 in enumerate(regions):
                if j in used:
                    continue
                x1_2, _, x2_2, _ = r2['bbox']
                if abs(x1_1 - x1_2) <= threshold_x and abs(x2_1 - x2_2) <= threshold_x:
                    group.append(r2)
                    used.add(j)

            groups.append(group)

        # merge vertically continuous images within each group
        merged = []
        for group in groups:
            sorted_group = sorted(group, key=lambda r: r['bbox'][1])
            current = None

            for region in sorted_group:
                x1, y1, x2, y2 = region['bbox']
                if current is None:
                    current = {
                        'bbox': [x1, y1, x2, y2],
                        'xrefs': [region['xref']],
                        'orig_width': region.get('orig_width', 0),
                        'orig_height': region.get('orig_height', 0),
                    }
                else:
                    prev_y2 = current['bbox'][3]
                    if abs(prev_y2 - y1) <= threshold_y:
                        current['bbox'][0] = min(current['bbox'][0], x1)
                        current['bbox'][1] = min(current['bbox'][1], y1)
                        current['bbox'][2] = max(current['bbox'][2], x2)
                        current['bbox'][3] = max(current['bbox'][3], y2)
                        current['xrefs'].append(region['xref'])
                        current['orig_width'] = max(current['orig_width'],
                                                    region.get('orig_width', 0))
                        current['orig_height'] = max(current['orig_height'],
                                                     region.get('orig_height', 0))
                    else:
                        merged.append(self._finalize_region(current))
                        current = {
                            'bbox': [x1, y1, x2, y2],
                            'xrefs': [region['xref']],
                            'orig_width': region.get('orig_width', 0),
                            'orig_height': region.get('orig_height', 0),
                        }

            if current is not None:
                merged.append(self._finalize_region(current))

        return merged

    def _merge_contained_images(self, regions: list) -> list:
        if not regions:
            return []

        final = []
        used = set()

        for i, r1 in enumerate(regions):
            if i in used:
                continue

            x1_1, y1_1, x2_1, y2_1 = r1['bbox']
            merged_bbox = [x1_1, y1_1, x2_1, y2_1]
            merged_xrefs = list(r1.get('merged_xrefs', [r1.get('xref', 0)]))
            merged_ow = r1.get('orig_width', 0)
            merged_oh = r1.get('orig_height', 0)

            for j, r2 in enumerate(regions):
                if j == i or j in used:
                    continue
                x1_2, y1_2, x2_2, y2_2 = r2['bbox']

                contains_r2 = (x1_1 <= x1_2 and y1_1 <= y1_2 and
                               x2_1 >= x2_2 and y2_1 >= y2_2)
                contained_by_r2 = (x1_2 <= x1_1 and y1_2 <= y1_1 and
                                   x2_2 >= x2_1 and y2_2 >= y2_1)

                if contains_r2 or contained_by_r2:
                    merged_bbox[0] = min(merged_bbox[0], x1_2)
                    merged_bbox[1] = min(merged_bbox[1], y1_2)
                    merged_bbox[2] = max(merged_bbox[2], x2_2)
                    merged_bbox[3] = max(merged_bbox[3], y2_2)
                    merged_xrefs.extend(r2.get('merged_xrefs', [r2.get('xref', 0)]))
                    merged_ow = max(merged_ow, r2.get('orig_width', 0))
                    merged_oh = max(merged_oh, r2.get('orig_height', 0))
                    used.add(j)

            bx1, by1, bx2, by2 = merged_bbox
            final.append({
                'bbox': tuple(merged_bbox),
                'width': bx2 - bx1,
                'height': by2 - by1,
                'area': (bx2 - bx1) * (by2 - by1),
                'orig_width': merged_ow,
                'orig_height': merged_oh,
                'xref': merged_xrefs[0] if merged_xrefs else 0,
                'merged_xrefs': merged_xrefs,
            })
            used.add(i)

        return final

    @staticmethod
    def _finalize_region(current: dict) -> dict:
        x1, y1, x2, y2 = current['bbox']
        return {
            'bbox': tuple(current['bbox']),
            'width': x2 - x1,
            'height': y2 - y1,
            'area': (x2 - x1) * (y2 - y1),
            'orig_width': current.get('orig_width', 0),
            'orig_height': current.get('orig_height', 0),
            'xref': current['xrefs'][0],
            'merged_xrefs': current['xrefs'],
        }
