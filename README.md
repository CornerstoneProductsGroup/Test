# Home Depot Order Splitter (Anchor-Based)

This Streamlit app splits a Home Depot multi-page order PDF into vendor-specific bundles using **layout-aware, anchor-based extraction** of Model/SKU values with a confidence score and a **Review Queue** for ambiguous pages.

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Inputs
- **PDF**: Home Depot order PDF (text-based or scanned; for scanned PDFs, pdfplumber will still attempt extraction, but OCR is not included by default).
- **SKU Map (.xlsx)**: Excel mapping of **Model or SKU → Vendor**. Use `.xlsx` (OpenXML). Columns (case-insensitive): `model` or `sku`, and `vendor`. You can include both `model` and `sku` columns; the app will index all non-empty keys.

## Outputs
- Vendor ZIPs inside `./output/YYYY-MM-DD/` (one ZIP per vendor).
- `split_report.csv` with per-page detections, chosen vendor, and confidence.
- `errors_low_confidence.csv` listing items that went to the Review Queue.
- Optional overlay PNGs under `./output/YYYY-MM-DD/overlays/`.

## Notes
- This is **Home Depot–only**. Lowe's & Tractor Supply can be added later as separate templates.
- The app supports `.xlsx` via **openpyxl**.
