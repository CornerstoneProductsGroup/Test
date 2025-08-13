
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import io, os, zipfile

from vendor_map import load_vendor_map, normalize_key
import split_core  # local module

st.set_page_config(page_title="Home Depot Order Splitter", layout="wide")
st.title("Home Depot Order Splitter – Anchor-Based")

# Vendors to auto-combine for the Print Pack (one-tap printing)
import re as _re
def _norm_vendor(s: str) -> str:
    s = (s or "").upper()
    return _re.sub(r"[^A-Z0-9]", "", s)

# Synonyms / cleanup (left -> normalized pattern we expect)
PRINT_PACK_SYNONYMS = {
    "POSTPROTECTORHERE": "POSTPROTECTOR",
    "WEEDSHARK": "WEEDSHARK",
}

PRINT_PACK_VENDORS = [
    "Cord Mate", "Cornerstone", "Gate Latch", "Home Selects",
    "Nisus", "Post Protector Here", "Soft Seal", "Weedshark"
]


# ---- Sidebar downloads ----
with st.sidebar:
    st.header("Downloads")

    # Current batch ZIP
    if st.session_state.get("zip_bytes"):
        st.download_button(
            "Download current ZIP",
            st.session_state["zip_bytes"],
            file_name=st.session_state.get("zip_name", "vendor_pdfs.zip"),
            key="dl_zip_sidebar"
        )
    else:
        st.caption("Run a batch to enable current ZIP download.")

    # Print Pack (combined)
    if st.session_state.get("print_pack_bytes"):
        st.download_button(
            "Download print pack (PDF)",
            st.session_state["print_pack_bytes"],
            file_name=st.session_state.get("print_pack_name", "Print Pack.pdf"),
            key="dl_printpack_sidebar"
        )
    else:
        st.caption("Print Pack will appear here after processing.")
    # Show included vendors (if any)
    inc = st.session_state.get("print_pack_included")
    if inc:
        st.caption("Included in Print Pack: " + ", ".join(sorted(set(inc))))

    # Previous Batches
    st.markdown("---")
    st.subheader("Previous Batches")
    output_root = Path("output")
    zip_files = sorted(output_root.rglob("*.zip"))
    if zip_files:
        options = [str(p) for p in zip_files]
        labels = [p.name for p in zip_files]
        sel_idx = st.selectbox(
            "Select a past ZIP",
            range(len(labels)),
            format_func=lambda i: labels[i],
            key="prev_zip_sel_sidebar"
        )
        chosen = zip_files[sel_idx]
        with open(chosen, "rb") as f:
            st.download_button("Download selected ZIP", f.read(), file_name=chosen.name, key="dl_prev_zip_sidebar")
    else:
        st.caption("No previous batches yet.")


# ---- Session state setup ----
for k in ["report_df", "review_df", "zip_bytes", "zip_name", "vendor_counts"]:
    if k not in st.session_state:
        st.session_state[k] = None
st.session_state.setdefault("out_pdfs", {})
st.session_state.setdefault("pdf_path", None)
st.session_state.setdefault("out_root", None)
st.session_state.setdefault("master_name", "Batch")
st.session_state.setdefault("vendor_options", [])
st.session_state.setdefault("auto_assign", {})
st.session_state.setdefault("print_pack_bytes", None)
st.session_state.setdefault("print_pack_name", None)
st.session_state.setdefault("print_pack_disk_path", None)
st.session_state.setdefault("print_pack_included", [])
with st.expander("How it works", expanded=False):
    st.markdown(
        "- Extracts **Model/SKU** values only **under the 'Model Number' label** (Home Depot template).\n"
        "- Scores each detection against your **Vendor Map (.xlsx)** with fuzzy matching.\n"
        "- High-confidence pages are auto-routed; others land in a **Review Queue**.\n"
        "- Supports `.xlsx` mapping (via **openpyxl**)."
    )

pdf_file = st.file_uploader("Upload Home Depot order PDF", type=["pdf"], key="pdf_upl")
map_file = st.file_uploader("Upload SKU/Model → Vendor map (.xlsx)", type=["xlsx"], key="map_upl")
threshold = st.slider(
    "Auto-route confidence threshold", 0.70, 0.99, 0.88, 0.01,
    help="Lower this slightly if many pages are landing in review due to near-misses."
)

col_run, col_clear = st.columns([1,1])
run = col_run.button("Process PDF", type="primary", disabled=not (pdf_file and map_file), key="run_btn")
clear = col_clear.button("Clear Results", key="clear_btn")

if clear:
    for k in ["report_df", "review_df", "zip_bytes", "zip_name", "vendor_counts"]:
    if k not in st.session_state:
        st.session_state[k] = None
st.session_state.setdefault("out_pdfs", {})
        st.session_state[k] = None
    st.experimental_rerun()

def _persist_and_show_outputs():
    # Show report
    if st.session_state["report_df"] is not None:
        st.subheader("Run report")
        st.dataframe(st.session_state["report_df"], use_container_width=True)
        rep_csv = st.session_state["report_df"].to_csv(index=False).encode()
        st.download_button("Download split_report.csv", rep_csv, file_name="split_report.csv", key="dl_report")

    # Show errors/low-confidence
    if st.session_state["review_df"] is not None and not st.session_state["review_df"].empty:
        st.warning(f"{len(st.session_state['review_df'])} page(s) need review (low confidence or unresolved). See below.")
        st.dataframe(st.session_state["review_df"], use_container_width=True)
        err_csv = st.session_state["review_df"].to_csv(index=False).encode()
        st.download_button("Download errors_low_confidence.csv", err_csv, file_name="errors_low_confidence.csv", key="dl_errs")
    elif st.session_state["report_df"] is not None:
        st.success("No review needed. All pages auto-routed above threshold.")

    # Download vendor PDFs
    if st.session_state["zip_bytes"]:
        st.download_button("Download vendor PDFs (ZIP)",
                           st.session_state["zip_bytes"],
                           file_name=st.session_state.get("zip_name","vendor_pdfs.zip"),
                           key="dl_zip")

    # Summary chart (vendor counts)
    if st.session_state["vendor_counts"] is not None:
        vc = st.session_state["vendor_counts"]
        st.subheader("Pages per Vendor")
        # Build a DataFrame like an Excel table
        if hasattr(vc, "to_frame"):
            df_counts = vc.to_frame(name="Pages").reset_index().rename(columns={"index": "Vendor"})
        else:
            import pandas as _pd
            df_counts = _pd.DataFrame({"Vendor": vc.index if hasattr(vc, "index") else [], "Pages": list(vc)})
        # Add Total row
        total_pages = int(df_counts["Pages"].sum()) if not df_counts.empty else 0
        df_total = {"Vendor": "Total", "Pages": total_pages}
        # Show table with total
        st.dataframe(df_counts, use_container_width=True)
        st.write("**Total pages:**", total_pages)

if run:
    # Track paths in session for later corrections
    st.session_state['pdf_path'] = None
    st.session_state['out_root'] = None
    out_root = Path("output")/datetime.now().strftime("%Y-%m-%d")
    out_root.mkdir(parents=True, exist_ok=True)

    # Save inputs
    pdf_path = out_root/"input.pdf"
    st.session_state['out_root'] = str(out_root)
    st.session_state['pdf_path'] = str(pdf_path)
    # Master input name (without extension) for output naming
    master_name = Path(pdf_file.name).stem if getattr(pdf_file, "name", None) else "Batch"
    st.session_state["master_name"] = master_name
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


    # Build vendor options for dropdowns
    vendor_options = sorted(set(vmap.values()))
    st.session_state["vendor_options"] = vendor_options

    with st.spinner("Extracting and splitting pages..."):
        import os as _os
        _os.environ['HD_MASTER_NAME'] = st.session_state.get('master_name','Batch')
        report_rows, review_rows, out_pdfs = split_core.split_pdf_to_vendors(str(pdf_path), str(out_root), vmap, threshold=threshold)

    # Build DataFrames
    rep_df = pd.DataFrame(report_rows)
    err_df = pd.DataFrame(review_rows)
    # Persist out_pdfs so other sections can safely access
    st.session_state["out_pdfs"] = out_pdfs if out_pdfs else {}

    # Build initial auto assignments from the report
    auto_assign = {}
    if not rep_df.empty:
        for _, r in rep_df.iterrows():
            if isinstance(r.get('vendor'), str) and r.get('vendor') and (not isinstance(r.get('flag'), str) or r.get('flag') == ''):
                auto_assign[int(r['page'])] = r['vendor']
    st.session_state['auto_assign'] = auto_assign

    # Compute vendor counts
    if not rep_df.empty:
        vc = rep_df["vendor"].fillna("").replace("", pd.NA).dropna().value_counts()
    else:
        vc = pd.Series(dtype=int)


# Build Print Pack (combined PDF for designated vendors) - initial build
try:
    from pypdf import PdfReader, PdfWriter
    # Normalize targets
    targets = set()
    for t in PRINT_PACK_VENDORS:
        nt = _norm_vendor(t)
        nt = PRINT_PACK_SYNONYMS.get(nt, nt)
        targets.add(nt)
    pack_paths = []
    included = []
    for k, pth in out_pdfs.items():
        nk = _norm_vendor(k)
        for nt in targets:
            if nk == nt or nk.startswith(nt) or nt.startswith(nk) or (nt in nk) or (nk in nt):
                pack_paths.append(pth)
                included.append(k)
                break
    if pack_paths:
        writer = PdfWriter()
        for pth in pack_paths:
            r = PdfReader(pth)
            for pg in r.pages:
                writer.add_page(pg)
        print_pack_path = out_root / f"{st.session_state.get('master_name','Batch')} - Print Pack.pdf"
        with open(print_pack_path, "wb") as f:
            writer.write(f)
        with open(print_pack_path, "rb") as f:
            st.session_state["print_pack_bytes"] = f.read()
        st.session_state["print_pack_name"] = print_pack_path.name
        st.session_state["print_pack_disk_path"] = str(print_pack_path)
        st.session_state["print_pack_included"] = included
    else:
        st.session_state["print_pack_bytes"] = None
        st.session_state["print_pack_name"] = None
        st.session_state["print_pack_disk_path"] = None
        st.session_state["print_pack_included"] = []
except Exception as e:
    st.warning(f"Could not build Print Pack: {e}")

    # Ensure local out_pdfs reference (fallback to session)
    out_pdfs = out_pdfs if "out_pdfs" in locals() and out_pdfs else st.session_state.get("out_pdfs", {})
    # Zip vendor outputs (and keep in session)
    zip_buf = None
    if out_pdfs:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as z:
            for vend, path in out_pdfs.items():
                z.write(path, arcname=f"{Path(path).name}")
            # Add Print Pack if present
            if st.session_state.get("print_pack_disk_path"):
                z.write(st.session_state["print_pack_disk_path"], arcname=Path(st.session_state["print_pack_disk_path"]).name)
        zip_bytes = zip_buf.getvalue()
        zip_name = f"{st.session_state.get('master_name', 'Batch')} - vendor_pdfs.zip"
    else:
        zip_bytes = None
        zip_name = "vendor_pdfs.zip"

    # Persist a copy of the ZIP to disk for history
    if zip_bytes:
        zip_disk_path = out_root / (st.session_state.get("master_name", "Batch") + " - vendor_pdfs.zip")
        with open(zip_disk_path, "wb") as _zf:
            _zf.write(zip_bytes)

    # Persist to session so downloads & chart stay after clicking
    st.session_state["report_df"] = rep_df
    st.session_state["review_df"] = err_df
    st.session_state["vendor_counts"] = vc
    st.session_state["zip_bytes"] = zip_bytes
    st.session_state["zip_name"] = zip_name


# --- Review & Fix ---
st.markdown("---")
st.subheader("Review & Fix (Assign vendors for uncertain pages)")

if st.session_state.get("review_df") is not None and not st.session_state["review_df"].empty:
    vendor_options = st.session_state.get("vendor_options", [])
    if not vendor_options:
        st.info("Upload a vendor map to populate the dropdown options.")
    else:
        st.write("Pick a vendor for each page below, then click **Apply selections**.")
        # Build selection widgets
        selections = {}
        for idx, row in st.session_state["review_df"].iterrows():
            page_no = int(row['page'])
            key_sel = f"sel_vendor_{page_no}"
            default_idx = 0 if vendor_options else None
            sel = st.selectbox(f"Page {page_no} – detected: {row.get('best_value','')}", vendor_options, key=key_sel)
            selections[page_no] = sel

        apply = st.button("Apply selections and update vendor files", type="primary", key="apply_sel_btn")
        if apply:
            # Build combined assignments: auto + overrides
            combined = dict(st.session_state.get("auto_assign", {}))
            combined.update({p: v for p, v in selections.items() if v})

            # Rebuild vendor PDFs from source using combined assignments
            src_pdf = st.session_state.get("pdf_path")
            out_root = st.session_state.get("out_root")
            if not src_pdf or not out_root:
                st.error("Missing source PDF path or output directory in session.")
            else:
                try:
                    from pypdf import PdfReader, PdfWriter
                    rdr = PdfReader(src_pdf)

                    # vendor -> list of zero-based page indices
                    vpages = {}
                    for p, v in combined.items():
                        vpages.setdefault(v, []).append(p-1)

                    # Write each vendor PDF fresh
                    out_pdfs = {}
                    from pathlib import Path as _Path
                    for v, pages in vpages.items():
                        pages = sorted(set([pp for pp in pages if 0 <= pp < len(rdr.pages)]))
                        if not pages:
                            continue
                        w = PdfWriter()
                        for pi in pages:
                            w.add_page(rdr.pages[pi])
                        vend_dir = _Path(out_root)/v
                        vend_dir.mkdir(parents=True, exist_ok=True)
                        out_path = vend_dir / f"{st.session_state.get('master_name', 'Batch')} {v}.pdf"
                        with open(out_path, "wb") as f:
                            w.write(f)
                        out_pdfs[v] = str(out_path)

                    # Update report_df with overrides
                    rep_df = st.session_state["report_df"].copy()
                    for p, v in selections.items():
                        rep_df.loc[rep_df['page'] == p, ['vendor','flag']] = [v, '']

                    # Remove fixed pages from review_df
                    review_df = st.session_state["review_df"].copy()
                    review_df = review_df[~review_df['page'].isin(list(selections.keys()))]

                    # Recompute vendor counts
                    if not rep_df.empty:
                        vc = rep_df["vendor"].fillna("").replace("", pd.NA).dropna().value_counts()
                    else:
                        import pandas as _pd
                        vc = _pd.Series(dtype=int)

                    # Build Print Pack (combined) after review
                    print_pack_path = None
                    try:
                        from pypdf import PdfReader, PdfWriter
                        # Normalize targets
                        targets = set()
                        for t in PRINT_PACK_VENDORS:
                            nt = _norm_vendor(t)
                            nt = PRINT_PACK_SYNONYMS.get(nt, nt)
                            targets.add(nt)
                        pack_paths = []
                        included = []
                        # Match using the out_pdfs dict we just created
                        for k, pth in out_pdfs.items():
                            nk = _norm_vendor(k)
                            for nt in targets:
                                if nk == nt or nk.startswith(nt) or nt.startswith(nk) or (nt in nk) or (nk in nt):
                                    pack_paths.append(pth)
                                    included.append(k)
                                    break
                        if pack_paths:
                            writer = PdfWriter()
                            for pth in pack_paths:
                                r = PdfReader(pth)
                                for pg in r.pages:
                                    writer.add_page(pg)
                            print_pack_path = _Path(out_root) / f"{st.session_state.get('master_name', 'Batch')} - Print Pack.pdf"
                            with open(print_pack_path, "wb") as f:
                                writer.write(f)
                            with open(print_pack_path, "rb") as f:
                                print_pack_bytes = f.read()
                            st.session_state["print_pack_bytes"] = print_pack_bytes
                            st.session_state["print_pack_name"] = print_pack_path.name
                            st.session_state["print_pack_disk_path"] = str(print_pack_path)
                            st.session_state["print_pack_included"] = included
                        else:
                            st.session_state["print_pack_bytes"] = None
                            st.session_state["print_pack_name"] = None
                            st.session_state["print_pack_disk_path"] = None
                            st.session_state["print_pack_included"] = []
                    except Exception as e:
                        st.warning(f"Could not build Print Pack: {e}")
                    
                    # Rebuild ZIP
                    import io, zipfile
                    zip_buf = None
                    if out_pdfs:
                        zip_buf = io.BytesIO()
                        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as z:
                            for vend, path in out_pdfs.items():
                                z.write(path, arcname=_Path(path).name)
                            # Add Print Pack if present
                            if st.session_state.get("print_pack_disk_path"):
                                z.write(st.session_state["print_pack_disk_path"], arcname=_Path(st.session_state["print_pack_disk_path"]).name)
                        zip_bytes = zip_buf.getvalue()
                        zip_name = f"{st.session_state.get('master_name', 'Batch')} - vendor_pdfs.zip"
                    else:
                        zip_bytes = None
                        zip_name = f"{st.session_state.get('master_name', 'Batch')} - vendor_pdfs.zip"

                    # Persist a copy of the ZIP to disk for history
                    if zip_bytes:
                        _zip_disk_path = _Path(out_root) / (st.session_state.get("master_name", "Batch") + " - vendor_pdfs.zip")
                        with open(_zip_disk_path, "wb") as _zf:
                            _zf.write(zip_bytes)

                    # Update out_pdfs in session for downstream consumers
                    st.session_state["out_pdfs"] = out_pdfs if out_pdfs else {}

                    # Persist back
                    st.session_state["report_df"] = rep_df
                    st.session_state["review_df"] = review_df
                    st.session_state["vendor_counts"] = vc
                    st.session_state["zip_bytes"] = zip_bytes
                    st.session_state["zip_name"] = zip_name
                    st.session_state["auto_assign"] = combined

                    st.success("Selections applied. Vendor PDFs and chart updated.")
                except Exception as e:
                    st.error(f"Failed to apply selections: {e}")
else:
    st.caption("No uncertain pages to review.")



# Always show persisted outputs if available
_persist_and_show_outputs()