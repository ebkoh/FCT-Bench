# Table detection (OpenCV morphology)
TABLE_DETECTION = {
    "min_table_area": 1000,     # minimum connected-component area (px)
    "min_width": 50,            # minimum candidate width (px)
    "min_height": 30,           # minimum candidate height (px)
    "h_kernel_length": 40,      # horizontal line kernel length
    "v_kernel_length": 40,      # vertical line kernel length
    "padding": 5,               # padding added to each detected box (px)
    "min_image_width": 150,     # minimum raster-image width to exclude (px)
    "min_image_height": 150,    # minimum raster-image height to exclude (px)
    "overlap_threshold": 0.3,   # table/image overlap (IoU) to drop a candidate
    "render_scale": 2,          # PDF -> raster scale factor
}

# Text-based false-positive filter
TABLE_FILTER = {
    "min_text_overlap_ratio": 0.1,   # min text/area overlap to keep a candidate
    "require_multi_line": True,      # require multi-line text inside the box
}

# pdfplumber word extraction (PDF points)
PDFPLUMBER = {
    "x_tolerance": 3,
    "y_tolerance": 3,
}
