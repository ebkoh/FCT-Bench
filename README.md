# FCT-Bench code repository

Code and evaluation scripts accompanying FCT-Bench, a table-detection dataset
for Korean public-sector documents.

The repository includes:

- a morphology-based annotation candidate generator,
- wrappers for six independent baseline tools,
- multi-IoU evaluation,
- PDF metadata and structure-tree inspection,
- annotation integrity checks, and
- blind re-annotation agreement analysis.

The annotations, metadata, evaluation outputs, and other data files are
distributed separately through the FCT-Bench Zenodo data record.

## Repository layout

```text
morphology/                 Morphology-based candidate generator
render_pages.py             PDF page rasterization
detect_tables.py            Candidate generation
run_baselines.py            Six independent baseline tools
evaluate.py                 Evaluation at IoU 0.5
evaluate_multi_iou.py       Evaluation at multiple IoU thresholds
verify_integrity.py         Annotation integrity checks
pdf_metadata.py             PDF metadata extraction
structure_tree_check.py     Structure-tree inspection
flowchart_filter.py         Post-hoc filtering
flowchart_filter_search.py  Filter-threshold search
blind_subset/               Blind-subset sampling and agreement scripts
```

## Installation

```bash
pip install -r requirements.txt
```

Java is required for Tabula, and Ghostscript is required for Camelot.

## Source documents

The source collection consists of 18 HWP guidelines distributed by the Korea
Authority of Land and Infrastructure Safety (KALIS) and one PDF guide
distributed by the Korea Internet & Security Agency (KISA).

The source documents are not redistributed in either this repository or the
Zenodo data record. Official source locations, filenames, access dates, and
other provenance information are provided in the deposited manifest.

The KALIS HWP files were manually exported to PDF with one logical page per
sheet before annotation and evaluation. The HWP-to-PDF conversion is not
automated by the Python scripts in this repository.

## Data record

The following materials are distributed separately through Zenodo:

- YOLO-format table annotations,
- document and page identifiers,
- provenance manifest,
- PDF metadata,
- baseline predictions,
- evaluation outputs,
- annotation-integrity results, and
- blind re-annotation data and results.

Zenodo record:

```text
[ZENODO DOI OR URL]
```

## Basic usage

```bash
python render_pages.py --pdf data/01_bridge.pdf --out-dir pages/01_bridge
python detect_tables.py --pdf-dir data --out results_morphology
python run_baselines.py --pdf-dir data --out results_baselines
python evaluate.py
python evaluate_multi_iou.py
python verify_integrity.py --annotations annotations
```

Command-line options may differ by script. Run the following to inspect the
available arguments:

```bash
python <script_name>.py --help
```

## Blind re-annotation analysis

The blind subset was annotated from scratch without access to detector outputs
or the released annotations.

The sampling and agreement scripts are included in:

```text
blind_subset/
    sample_pages.py
    compute_agreement.py
```

The selected-page list, blind re-annotation labels, and reported agreement
outputs are distributed through the Zenodo data record.

Agreement statistics can be reproduced using the deposited paths, for example:

```bash
python blind_subset/compute_agreement.py \
    --original-annotations annotations \
    --reannotation blind_subset_labels \
    --plan blind_subset_pages.csv \
    --output agreement.json
```

Replace the paths and filenames with those used in the Zenodo release.

## Annotation format

Annotations use single-class YOLO text format:

```text
class_id x_center y_center width height
```

The only class is:

```text
0 table
```

All coordinates are normalized to `[0, 1]`. Pages without tables are represented
by empty label files.

## Licences

- Source code: MIT License
- Author-generated annotations and metadata: `[DATA LICENCE]`
- Third-party source documents: terms stated by KALIS and KISA

The source documents are not redistributed. The MIT License applies only to the
source code in this repository.

## Citation

When using the dataset, cite the Zenodo data record:

```bibtex
@dataset{fct_bench_2026,
  author    = {[DATASET AUTHORS]},
  title     = {Table detection annotations for format-converted Korean public-sector PDF documents},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {[ZENODO DOI]}
}
```

When available, also cite the accompanying Data Descriptor.