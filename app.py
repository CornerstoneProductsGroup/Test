import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import io, os, zipfile

from vendor_map import load_vendor_map, normalize_key
import split_core  # local module

st.set_page_config(page_title="Home Depot Order Splitter", layout="wide")

st.title("Home Depot Order Splitter – Anchor-Based")


with st.expander("How it works", expanded=False):
    st.markdown("""- Extracts **Model/SKU** values only **near labels** like *Model #*, *Model Number*, *Store SKU #* (Home Depot template).
- Scores each detection against your **Vendor Map (.xlsx)** with fuzzy matching.
- High-confidence pages are auto-routed; others land in a **Review Queue**.
- Supports `.xlsx` mapping (via **openpyxl**).
    """)

pdf_file = st.file_uploader("Upload Home Depot order PDF", type=["pdf"])
map_file = st.file_uploader("Upload SKU/Model → Vendor map (.xlsx)", type=["xlsx"])
threshold = st.slider("Auto-route confidence threshold", 0.70, 0.99, 0.88, 0.01,
    help="Lower this slightly if many pages are landing in review due to near-misses.")

run = st.button("Process PDF", type="primary", disabled=not (pdf_file and map_file))

if run:
    out_root = Path("output")/datetime.now().strftime("%Y-%m-%d")
    out_root.mkdir(parents=True, exist_ok=True)

    # Save inputs
    pdf_path = out_root/"input.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getvalue())

    map_path = out_root/"vendor_map.xlsx"
    with open(map_path, "wb") as f:
        f.write(map_file.getvalue())

    # Load map
    try:
        vmap = load_vendor_map(str(map_path))
    except Exception as e:
        st.error(f"Failed to load vendor map: {e}")
        st.stop()

    # Split using core
    with st.spinner("Extracting and splitting pages..."):
        report_rows, review_rows, out_pdfs = split_core.split_pdf_to_vendors(str(pdf_path), str(out_root), vmap, threshold=threshold)

    # Reports
    rep_df = pd.DataFrame(report_rows)
    st.subheader("Run report")
    with st.expander("Debug: what did the extractor see?", expanded=False):
        if not rep_df.empty:
            st.write("First 10 rows shown (includes candidate values column).")
            st.dataframe(rep_df.head(10), use_container_width=True)
    st.dataframe(rep_df, use_container_width=True)
    rep_csv = rep_df.to_csv(index=False).encode()
    st.download_button("Download split_report.csv", rep_csv, file_name="split_report.csv" )

    if review_rows:
        st.warning(f"{len(review_rows)} page(s) need review (low confidence or unresolved). See below.")
        err_df = pd.DataFrame(review_rows)
        st.dataframe(err_df, use_container_width=True)
        err_csv = err_df.to_csv(index=False).encode()
        st.download_button("Download errors_low_confidence.csv", err_csv, file_name="errors_low_confidence.csv" )
    else:
        st.success("No review needed. All pages auto-routed above threshold.")

    # Zip vendor outputs
    if out_pdfs:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as z:
            for vend, path in out_pdfs.items():
                z.write(path, arcname=f"{Path(path).name}")
        st.download_button("Download vendor PDFs (ZIP)", zip_buf.getvalue(), file_name="vendor_pdfs.zip")
    else:
        st.info("No vendor PDFs were generated.")
