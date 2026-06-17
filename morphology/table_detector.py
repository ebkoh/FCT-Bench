import cv2
import numpy as np
import fitz  # PyMuPDF


class TableDetector:
    def __init__(self, pdf_path: str, render_scale: int = 2):
        self.pdf_path = pdf_path
        self.render_scale = render_scale
        self.img = None
        self.img_height = 0
        self.img_width = 0
        self.current_page = None

    def load_page(self, page_num: int) -> np.ndarray:
        self.current_page = page_num
        doc = fitz.open(self.pdf_path)
        page = doc[page_num]
        scale = self.render_scale
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.img = img
        self.img_height, self.img_width = img.shape[:2]
        doc.close()
        return img

    def get_total_pages(self) -> int:
        doc = fitz.open(self.pdf_path)
        total = len(doc)
        doc.close()
        return total

    def detect_tables(self,
                      image_regions: list = None,
                      min_table_area: int = 1000,
                      min_width: int = 50,
                      min_height: int = 30,
                      h_kernel_length: int = 40,
                      v_kernel_length: int = 40,
                      padding: int = 5,
                      overlap_threshold: float = 0.3) -> list:
        if self.img is None:
            raise RuntimeError("call load_page() first")

        if image_regions is None:
            image_regions = []

        # Binarize
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_length, 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

        # Vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_length))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

        table_lines = cv2.add(h_lines, v_lines)

        # Close gaps
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        table_lines = cv2.dilate(table_lines, kernel_connect, iterations=2)

        tables = self._find_table_candidates(
            table_lines, min_table_area, min_width, min_height,
            padding, image_regions, overlap_threshold
        )

        return tables

    def _find_table_candidates(self, table_lines, min_area, min_width, min_height,
                               padding, image_regions, overlap_threshold) -> list:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            table_lines, connectivity=8
        )

        tables = []

        for i in range(1, num_labels):  # 0 is background
            x, y, w, h, area = stats[i]

            if area < min_area or w < min_width or h < min_height:
                continue

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(self.img.shape[1], x + w + padding)
            y2 = min(self.img.shape[0], y + h + padding)

            table_bbox = (x1, y1, x2, y2)

            if self._is_overlapping_image(table_bbox, image_regions, overlap_threshold):
                continue

            tables.append({
                'id': len(tables) + 1,
                'bbox': table_bbox,
                'width': x2 - x1,
                'height': y2 - y1,
                'area': area,
                'center': (int(centroids[i][0]), int(centroids[i][1]))
            })

        # sort by center-y, reassign ids
        tables.sort(key=lambda t: t['center'][1])
        for i, table in enumerate(tables, 1):
            table['id'] = i

        return tables

    def _is_overlapping_image(self, table_bbox, image_regions,
                              overlap_threshold: float) -> bool:
        x1_t, y1_t, x2_t, y2_t = table_bbox
        table_area = (x2_t - x1_t) * (y2_t - y1_t)

        for img_region in image_regions:
            img_bbox = img_region['bbox']
            x1_i, y1_i, x2_i, y2_i = img_bbox

            # fully inside an image
            if x1_i <= x1_t and y1_i <= y1_t and x2_i >= x2_t and y2_i >= y2_t:
                return True

            # >= 80% of the candidate area inside an image
            x1_inter = max(x1_t, x1_i)
            y1_inter = max(y1_t, y1_i)
            x2_inter = min(x2_t, x2_i)
            y2_inter = min(y2_t, y2_i)

            if x2_inter > x1_inter and y2_inter > y1_inter:
                inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                if table_area > 0 and inter_area / table_area >= 0.8:
                    return True

            # IoU threshold
            iou = self._calculate_iou(table_bbox, img_bbox)
            if iou >= overlap_threshold:
                return True

        return False

    @staticmethod
    def _calculate_iou(bbox1: tuple, bbox2: tuple) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0
