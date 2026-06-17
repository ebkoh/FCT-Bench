import pdfplumber


class TableFilter:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_text_blocks(self, page_num: int,
                            img_width: int, img_height: int,
                            x_tolerance: int = 3,
                            y_tolerance: int = 3) -> list:
        text_regions = []

        with pdfplumber.open(self.pdf_path) as pdf:
            page = pdf.pages[page_num]
            page_width = page.width
            page_height = page.height

            words = page.extract_words(
                x_tolerance=x_tolerance,
                y_tolerance=y_tolerance,
                keep_blank_chars=False
            )

            if not words:
                return text_regions

            # group words into lines by top coordinate
            lines = {}
            for word in words:
                top = word['top']
                line_key = None
                for existing_top in lines.keys():
                    if abs(top - existing_top) <= y_tolerance:
                        line_key = existing_top
                        break
                if line_key is None:
                    line_key = top
                    lines[line_key] = []
                lines[line_key].append(word)

            # one block per line
            for line_top, line_words in lines.items():
                if not line_words:
                    continue

                line_words.sort(key=lambda w: w['x0'])
                line_text = ' '.join(w['text'] for w in line_words)

                x0 = min(w['x0'] for w in line_words)
                top = min(w['top'] for w in line_words)
                x1 = max(w['x1'] for w in line_words)
                bottom = max(w['bottom'] for w in line_words)

                # PDF coords -> rendered pixel coords
                x1_pixel = int((x0 / page_width) * img_width)
                y1_pixel = int((top / page_height) * img_height)
                x2_pixel = int((x1 / page_width) * img_width)
                y2_pixel = int((bottom / page_height) * img_height)

                if line_text.strip():
                    text_regions.append({
                        'bbox': (x1_pixel, y1_pixel, x2_pixel, y2_pixel),
                        'text': line_text.strip(),
                        'line_count': 1,
                        'is_multi_line': False
                    })

            text_regions.sort(key=lambda r: r['bbox'][1])

            # merge lines whose vertical gap is <= 50% of line height
            text_regions = self._merge_nearby_lines(text_regions)

        return text_regions

    def filter_tables(self, tables: list, text_regions: list,
                      min_text_overlap_ratio: float = 0.1,
                      require_multi_line: bool = True) -> list:
        filtered = []
        for table in tables:
            if self._has_text_in_table(table['bbox'], text_regions,
                                       min_text_overlap_ratio, require_multi_line):
                filtered.append(table)

        # reassign ids
        for i, table in enumerate(filtered, 1):
            table['id'] = i

        return filtered

    def _has_text_in_table(self, table_bbox: tuple, text_regions: list,
                           min_overlap_ratio: float, require_multi_line: bool) -> bool:
        x1_t, y1_t, x2_t, y2_t = table_bbox
        table_area = (x2_t - x1_t) * (y2_t - y1_t)
        if table_area == 0:
            return False

        total_overlap = 0
        has_multi_line = False
        block_count = 0

        for tr in text_regions:
            x1_text, y1_text, x2_text, y2_text = tr['bbox']
            x1_inter = max(x1_t, x1_text)
            y1_inter = max(y1_t, y1_text)
            x2_inter = min(x2_t, x2_text)
            y2_inter = min(y2_t, y2_text)

            if x2_inter > x1_inter and y2_inter > y1_inter:
                total_overlap += (x2_inter - x1_inter) * (y2_inter - y1_inter)
                block_count += 1
                if tr.get('is_multi_line', False):
                    has_multi_line = True

        overlap_ratio = total_overlap / table_area
        has_enough_text = overlap_ratio >= min_overlap_ratio

        if require_multi_line:
            return has_enough_text and (has_multi_line or block_count >= 2)
        return has_enough_text

    @staticmethod
    def _merge_nearby_lines(regions: list) -> list:
        if not regions:
            return regions

        merged = []
        current = None

        for region in regions:
            if current is None:
                current = region.copy()
                current['texts'] = [region['text']]
            else:
                prev_bottom = current['bbox'][3]
                curr_top = region['bbox'][1]
                vertical_gap = curr_top - prev_bottom
                avg_height = current['bbox'][3] - current['bbox'][1]

                if avg_height > 0 and vertical_gap <= avg_height * 0.5:
                    current['texts'].append(region['text'])
                    current['bbox'] = (
                        min(current['bbox'][0], region['bbox'][0]),
                        current['bbox'][1],
                        max(current['bbox'][2], region['bbox'][2]),
                        region['bbox'][3]
                    )
                    current['line_count'] += 1
                    current['is_multi_line'] = True
                else:
                    current['text'] = '\n'.join(current['texts'])
                    del current['texts']
                    merged.append(current)
                    current = region.copy()
                    current['texts'] = [region['text']]

        if current is not None:
            current['text'] = '\n'.join(current.get('texts', [current['text']]))
            if 'texts' in current:
                del current['texts']
            merged.append(current)

        return merged
