import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import io, os, zipfile

from vendor_map import load_vendor_map, normalize_key
import split_core  # local module

st.set_page_config(page_title="Home Depot Order Splitter", layout="wide")
st.title("Home Depot Order Splitter – Anchor-Based")

# ---- Sidebar: Vendor Map ----
with st.sidebar:
    st.header("Vendor Map")
    default_map_path = Path("data") / "vendor_map.xlsx"
    st.session_state.setdefault("default_map_path", str(default_map_path))

    if default_map_path.exists():
        st.caption(f"Default map: **{default_map_path.name}**")
        try:
            mtime = default_map_path.stat().st_mtime
            from datetime import datetime as _dt
            st.caption("Last updated: " + _dt.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"))
        except Exception:
            pass
        try:
            with open(default_map_path, "rb") as _f:
                st.download_button("Download default map", _f.read(), file_name=default_map_path.name, key="dl_default_map")
        except Exception:
            st.caption("Default map not readable.")
    else:
        st.warning("No default vendor map found. Upload one below and click 'Set as default'.")

    new_map = st.file_uploader("Upload new vendor map (.xlsx)", type=["xlsx"], key="upl_new_map")
    if new_map is not None:
        if st.button("Set as default", key="btn_set_default"):
            default_map_path.parent.mkdir(parents=True, exist_ok=True)
            with open(default_map_path, "wb") as _out:
                _out.write(new_map.getvalue())
            st.session_state["default_map_path"] = str(default_map_path)
            st.success("Default vendor map updated.")

# ---- Sidebar downloads ----
import re as _re
def _norm_vendor(s: str) -> str:
    s = (s or "").upper()
    return _re.sub(r"[^A-Z0-9]", "", s)

PRINT_PACK_SYNONYMS = {"POSTPROTECTORHERE": "POSTPROTECTOR", "WEEDSHARK": "WEEDSHARK"}
PRINT_PACK_VENDORS = ["Cord Mate", "Cornerstone", "Gate Latch", "Home Selects", "Nisus", "Post Protector Here", "Soft Seal", "Weedshark"]

def _build_print_pack_alpha(out_pdfs: dict, out_root: Path, master_name: str):
    """Build Print Pack from PRINT_PACK_VENDORS with tolerant matching and
    alphabetical vendor order. Returns (path or None, included vendors)."""
    from pypdf import PdfReader, PdfWriter
    # Apply synonyms when normalizing targets
    targets = set()
    for t in PRINT_PACK_VENDORS:
        nt = _norm_vendor(t)
        nt = PRINT_PACK_SYNONYMS.get(nt, nt)
        targets.add(nt)
    matched = {}
    for vendor_name, pth in out_pdfs.items():
        nk = _norm_vendor(vendor_name)
        for nt in targets:
            if nk == nt or nk.startswith(nt) or nt.startswith(nk) or (nt in nk) or (nk in nt):
                if Path(pth).exists():
                    matched[vendor_name] = pth
                break
    if not matched:
        return None, []
    ordered_vendor_names = sorted(matched.keys(), key=lambda s: s.upper())
    writer = PdfWriter()
    for vname in ordered_vendor_names:
        r = PdfReader(matched[vname])
        for pg in r.pages:
            writer.add_page(pg)
    pack_path = Path(out_root) / f"{master_name} - Print Pack.pdf"
    with open(pack_path, "wb") as f:
        writer.write(f)
    return str(pack_path), ordered_vendor_names

with st.sidebar:
    st.header("Downloads")
    # Current batch ZIP
    if st.session_state.get("zip_bytes"):
        st.download_button("Download current ZIP", st.session_state["zip_bytes"], file_name=st.session_state.get("zip_name", "vendor_pdfs.zip"), key="dl_zip_sidebar")
    else:
        st.caption("Run a batch to enable current ZIP download.")
    # Print Pack
    if st.session_state.get("print_pack_bytes"):
        st.download_button("Download print pack (PDF)", st.session_state["print_pack_bytes"], file_name=st.session_state.get("print_pack_name", "Print Pack.pdf"), key="dl_printpack_sidebar")
        inc = st.session_state.get("print_pack_included")
        if inc:
            st.caption("Included in Print Pack: " + ", ".join(sorted(set(inc))))
    else:
        st.caption("Print Pack will appear here after processing.")
    # Previous batches
    st.markdown("---")
    st.subheader("Previous Batches")
    output_root = Path("output")
    zip_files = sorted(output_root.rglob("*.zip"))
    if zip_files:
        labels = [p.name for p in zip_files]
        sel_idx = st.selectbox("Select a past ZIP", range(len(labels)), format_func=lambda i: labels[i], key="prev_zip_sel_sidebar")
        chosen = zip_files[sel_idx]
        with open(chosen, "rb") as f:
            st.download_button("Download selected ZIP", f.read(), file_name=chosen.name, key="dl_prev_zip_sidebar")
    else:
        st.caption("No previous batches yet.")

# ---- Session state setup ----
def _init_state():
    keys = ["report_df", "review_df", "zip_bytes", "zip_name", "vendor_counts"]
    for k in keys:
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
_init_state()

with st.expander("How it works", expanded=False):
    st.markdown(
        "- Extracts **Model** values only under the **'Model Number'** label (Home Depot template).\n"
        "- Matches against your **Vendor Map (.xlsx)**; high-confidence pages auto-route; others go to **Review & Fix**.\n"
        "- Built-in default map; update from the sidebar when needed."
    )

pdf_file = st.file_uploader("Upload Home Depot order PDF", type=["pdf"], key="pdf_upl")
map_file = st.file_uploader("Upload SKU/Model → Vendor map (.xlsx) (optional)", type=["xlsx"], key="map_upl")
threshold = st.slider("Auto-route confidence threshold", 0.70, 0.99, 0.88, 0.01, help="Lower slightly if many pages land in review.")

col_run, col_clear = st.columns([1,1])
has_default_map = (Path("data")/ "vendor_map.xlsx").exists()
run_disabled = not (pdf_file and (map_file or has_default_map))
run = col_run.button("Process PDF", type="primary", disabled=run_disabled, key="run_btn")
clear = col_clear.button("Clear Results", key="clear_btn")

if clear:
    for k in ["report_df", "review_df", "zip_bytes", "zip_name", "vendor_counts"]:
        st.session_state[k] = None
    st.session_state["out_pdfs"] = {}
    st.session_state["print_pack_bytes"] = None
    st.session_state["print_pack_name"] = None
    st.session_state["print_pack_disk_path"] = None
    st.session_state["print_pack_included"] = []
    st.experimental_rerun()

def _persist_and_show_outputs():
    if st.session_state["report_df"] is not None:
        st.subheader("Run report")
        st.dataframe(st.session_state["report_df"], use_container_width=True)
        rep_csv = st.session_state["report_df"].to_csv(index=False).encode()
        st.download_button("Download split_report.csv", rep_csv, file_name="split_report.csv", key="dl_report")

    if st.session_state["review_df"] is not None and not st.session_state["review_df"].empty:
        st.warning(f"{len(st.session_state['review_df'])} page(s) need review. See below.")
        st.dataframe(st.session_state["review_df"], use_container_width=True)
        err_csv = st.session_state["review_df"].to_csv(index=False).encode()
        st.download_button("Download errors_low_confidence.csv", err_csv, file_name="errors_low_confidence.csv", key="dl_errs")
    elif st.session_state["report_df"] is not None:
        st.success("No review needed. All pages auto-routed above threshold.")

    if st.session_state["vendor_counts"] is not None:
        vc = st.session_state["vendor_counts"]
        st.subheader("Pages per Vendor")
        if hasattr(vc, "to_frame"):
            df_counts = vc.to_frame(name="Pages").reset_index().rename(columns={"index": "Vendor"})
        else:
            df_counts = pd.DataFrame({"Vendor": [], "Pages": []})
        total_pages = int(df_counts["Pages"].sum()) if not df_counts.empty else 0
        st.dataframe(df_counts, use_container_width=True)
        st.write("**Total pages:**", total_pages)

    if st.session_state.get("zip_bytes"):
        st.download_button("Download vendor PDFs (ZIP)", st.session_state["zip_bytes"], file_name=st.session_state.get("zip_name","vendor_pdfs.zip"), key="dl_zip_main")

# Processing
if run:
    out_root = Path("output")/datetime.now().strftime("%Y-%m-%d")
    out_root.mkdir(parents=True, exist_ok=True)
    st.session_state["out_root"] = str(out_root)

    # Save PDF
    pdf_path = out_root/"input.pdf"
    st.session_state["master_name"] = Path(pdf_file.name).stem if getattr(pdf_file, "name", None) else "Batch"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getvalue())
    st.session_state["pdf_path"] = str(pdf_path)

    # Decide which map to use
    if map_file is not None:
        map_path = out_root/"vendor_map.xlsx"
        with open(map_path, "wb") as f:
            f.write(map_file.getvalue())
    else:
        _def = Path("data")/ "vendor_map.xlsx"
        if not _def.exists():
            st.error("No vendor map provided and no default found. Upload or set a default map in the sidebar.")
            st.stop()
        map_path = out_root/"vendor_map.xlsx"
        import shutil as _sh
        _sh.copyfile(_def, map_path)

    # Load map
    try:
        vmap = load_vendor_map(str(map_path))
    except Exception as e:
        st.error(f"Failed to load vendor map: {e}")
        st.stop()

    vendor_options = sorted(set(vmap.values()))
    st.session_state["vendor_options"] = vendor_options

    with st.spinner("Extracting and splitting pages..."):
        import os as _os
        _os.environ['HD_MASTER_NAME'] = st.session_state.get('master_name','Batch')
        report_rows, review_rows, out_pdfs = split_core.split_pdf_to_vendors(str(pdf_path), str(out_root), vmap, threshold=threshold)

    # DataFrames & counts
    rep_df = pd.DataFrame(report_rows)
    err_df = pd.DataFrame(review_rows)
    st.session_state["out_pdfs"] = out_pdfs if out_pdfs else {}
    # Fallback: if out_pdfs is empty, rebuild from disk under out_root
    if not st.session_state['out_pdfs']:
        scan_root = Path(st.session_state.get('out_root','output'))
        disk_map = {}
        try:
            for vend_dir in sorted([p for p in scan_root.glob('*') if p.is_dir()]):
                vend = vend_dir.name
                pdfs = sorted(vend_dir.glob('*.pdf'))
                if pdfs:
                    disk_map[vend] = str(pdfs[0])
        except Exception as _e:
            pass
        if disk_map:
            st.session_state['out_pdfs'] = disk_map

    # Fallback: if out_pdfs is empty, rebuild from disk under out_root
    if not st.session_state['out_pdfs']:
        scan_root = Path(st.session_state.get('out_root','output'))
        disk_map = {}
        try:
            for vend_dir in sorted([p for p in scan_root.glob('*') if p.is_dir()]):
                vend = vend_dir.name
                pdfs = sorted(vend_dir.glob('*.pdf'))
                if pdfs:
                    disk_map[vend] = str(pdfs[0])
        except Exception as _e:
            pass
        if disk_map:
            st.session_state['out_pdfs'] = disk_map

    if not rep_df.empty:
        vc = rep_df["vendor"].fillna("").replace("", pd.NA).dropna().value_counts()
    else:
        vc = pd.Series(dtype=int)

    # Persist early so UI shows even if later steps fail
st.session_state['report_df'] = rep_df
st.session_state['review_df'] = err_df
st.session_state['vendor_counts'] = vc

# Build Print Pack (initial)
try:
    pp_path, included = _build_print_pack_alpha(st.session_state["out_pdfs"], out_root, st.session_state.get('master_name','Batch'))
    if pp_path:
        with open(pp_path, 'rb') as f:
            st.session_state['print_pack_bytes'] = f.read()
        st.session_state['print_pack_name'] = Path(pp_path).name
        st.session_state['print_pack_disk_path'] = pp_path
        st.session_state['print_pack_included'] = included
    else:
        st.session_state['print_pack_bytes'] = None
        st.session_state['print_pack_name'] = None
        st.session_state['print_pack_disk_path'] = None
        st.session_state['print_pack_included'] = []
except Exception as e:
    st.warning(f"Could not build Print Pack: {e}")

    # Build ZIP (include Print Pack)
try:
    zip_buf = None
    if st.session_state['out_pdfs']:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as z:
            for vend, path in st.session_state['out_pdfs'].items():
                z.write(path, arcname=f"{Path(path).name}")
            if st.session_state.get('print_pack_disk_path'):
                z.write(st.session_state['print_pack_disk_path'], arcname=Path(st.session_state['print_pack_disk_path']).name)
        zip_bytes = zip_buf.getvalue()
    else:
        zip_bytes = None
    zip_name = f"{st.session_state.get('master_name', 'Batch')} - vendor_pdfs.zip"
    # Persist copy to disk for history if we have bytes
    if zip_bytes:
        zip_disk_path = Path(st.session_state.get('out_root','output')) / zip_name
        with open(zip_disk_path, 'wb') as _zf:
            _zf.write(zip_bytes)
except Exception as e:
    st.warning(f"Could not build ZIP: {e}")

# Persist
st.session_state["report_df"] = rep_df
    st.session_state["review_df"] = err_df
    st.session_state["vendor_counts"] = vc
    st.session_state["zip_bytes"] = zip_bytes
    st.session_state["zip_name"] = zip_namep_namep_name

# Always show persisted outputs
_persist_and_show_outputs()

with st.expander('Debug (counts & keys)', expanded=False):
    st.write('out_pdfs vendors:', list(st.session_state.get('out_pdfs', {}).keys()))
    rep = st.session_state.get('report_df')
    rev = st.session_state.get('review_df')
    st.write('report rows:', 0 if rep is None else len(rep))
    st.write('review rows:', 0 if rev is None else (0 if rev is None else len(rev)))

# --- Review & Fix ---
st.markdown("---")
st.subheader("Review & Fix (Assign vendors for uncertain pages)")

if st.session_state.get("review_df") is not None and not st.session_state["review_df"].empty:
    vendor_options = st.session_state.get("vendor_options", [])
    if not vendor_options:
        st.info("Upload a vendor map to populate the dropdown options.")
    else:
        st.write("Pick a vendor for each page below, then click **Apply selections**.")
        selections = {}
        for idx, row in st.session_state["review_df"].iterrows():
            page_no = int(row['page'])
            key_sel = f"sel_vendor_{page_no}"
            sel = st.selectbox(f"Page {page_no} – detected: {row.get('best_value','')}", vendor_options, key=key_sel)
            selections[page_no] = sel

        apply = st.button("Apply selections and update vendor files", type="primary", key="apply_sel_btn")
        if apply:
            try:
                from pypdf import PdfReader, PdfWriter
                src_pdf = st.session_state.get("pdf_path")
                out_root = st.session_state.get("out_root")
                if not src_pdf or not out_root:
                    st.error("Missing source PDF path or output directory in session.")
                else:
                    rdr = PdfReader(src_pdf)
                    # Combined assignments: previous auto + new overrides
                    combined = dict(st.session_state.get("auto_assign", {}))
                    combined.update({p: v for p, v in selections.items() if v})

                    # Rebuild vendor PDFs
                    vpages = {}
                    for p, v in combined.items():
                        vpages.setdefault(v, []).append(p-1)
                    out_pdfs = {}
                    for v, pages in vpages.items():
                        pages = sorted(set([pp for pp in pages if 0 <= pp < len(rdr.pages)]))
                        if not pages:
                            continue
                        w = PdfWriter()
                        for pi in pages:
                            w.add_page(rdr.pages[pi])
                        vend_dir = Path(out_root)/v
                        vend_dir.mkdir(parents=True, exist_ok=True)
                        out_path = vend_dir / f"{st.session_state.get('master_name', 'Batch')} {v}.pdf"
                        with open(out_path, "wb") as f:
                            w.write(f)
                        out_pdfs[v] = str(out_path)

                    # Update session out_pdfs
                    st.session_state["out_pdfs"] = out_pdfs if out_pdfs else {}
    # Fallback: if out_pdfs is empty, rebuild from disk under out_root
    if not st.session_state['out_pdfs']:
        scan_root = Path(st.session_state.get('out_root','output'))
        disk_map = {}
        try:
            for vend_dir in sorted([p for p in scan_root.glob('*') if p.is_dir()]):
                vend = vend_dir.name
                pdfs = sorted(vend_dir.glob('*.pdf'))
                if pdfs:
                    disk_map[vend] = str(pdfs[0])
        except Exception as _e:
            pass
        if disk_map:
            st.session_state['out_pdfs'] = disk_map

    # Fallback: if out_pdfs is empty, rebuild from disk under out_root
    if not st.session_state['out_pdfs']:
        scan_root = Path(st.session_state.get('out_root','output'))
        disk_map = {}
        try:
            for vend_dir in sorted([p for p in scan_root.glob('*') if p.is_dir()]):
                vend = vend_dir.name
                pdfs = sorted(vend_dir.glob('*.pdf'))
                if pdfs:
                    disk_map[vend] = str(pdfs[0])
        except Exception as _e:
            pass
        if disk_map:
            st.session_state['out_pdfs'] = disk_map


                    # Update report/review
                    rep_df = st.session_state["report_df"].copy()
                    for p, v in selections.items():
                        rep_df.loc[rep_df['page'] == p, ['vendor','flag']] = [v, '']
                    review_df = st.session_state["review_df"].copy()
                    review_df = review_df[~review_df['page'].isin(list(selections.keys()))]

                    # Persist early so UI shows even if later steps fail
                    st.session_state['report_df'] = rep_df
                    st.session_state['review_df'] = review_df

# Recompute counts
                    if not rep_df.empty:
                        vc = rep_df["vendor"].fillna("").replace("", pd.NA).dropna().value_counts()
                    else:
                        vc = pd.Series(dtype=int)

                                        # Rebuild Print Pack after review
                    try:
                        pp_path, included = _build_print_pack_alpha(out_pdfs, out_root, st.session_state.get('master_name','Batch'))
                        if pp_path:
                            with open(pp_path, 'rb') as f:
                                print_pack_bytes = f.read()
                            st.session_state['print_pack_bytes'] = print_pack_bytes
                            st.session_state['print_pack_name'] = Path(pp_path).name
                            st.session_state['print_pack_disk_path'] = pp_path
                            st.session_state['print_pack_included'] = included
                        else:
                            st.session_state['print_pack_bytes'] = None
                            st.session_state['print_pack_name'] = None
                            st.session_state['print_pack_disk_path'] = None
                            st.session_state['print_pack_included'] = []
                    except Exception as e:
                        st.warning(f"Could not build Print Pack: {e}")

                                        # Rebuild ZIP (include Print Pack)
                    try:
                        zip_buf = None
                        if out_pdfs:
                            zip_buf = io.BytesIO()
                            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as z:
                                for vend, path in out_pdfs.items():
                                    z.write(path, arcname=Path(path).name)
                                if st.session_state.get('print_pack_disk_path'):
                                    z.write(st.session_state['print_pack_disk_path'], arcname=Path(st.session_state['print_pack_disk_path']).name)
                            zip_bytes = zip_buf.getvalue()
                        else:
                            zip_bytes = None
                        zip_name = f"{st.session_state.get('master_name','Batch')} - vendor_pdfs.zip"
                        if zip_bytes:
                            zip_disk_path = Path(out_root) / zip_name
                            with open(zip_disk_path, 'wb') as _zf:
                                _zf.write(zip_bytes)
                    except Exception as e:
                        st.warning(f"Could not build ZIP: {e}")

                    # Persist back
# Persist back
                    st.session_state["report_df"] = rep_df
                    st.session_state["review_df"] = review_df
                    st.session_state["vendor_counts"] = vc
                    st.session_state["zip_bytes"] = zip_bytes
                    st.session_state["zip_name"] = zip_namep_namep_name
                    st.success("Selections applied. Vendor PDFs and Print Pack updated.")
            except Exception as e:
                st.error(f"Failed to apply selections: {e}")
else:
    st.caption("No uncertain pages to review.")