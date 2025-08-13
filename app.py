import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import io, os, zipfile

from vendor_map import load_vendor_map, normalize_key
import split_core  # HD
import lowes_core  # Lowe's
import tsc_core  # Tractor Supply

st.set_page_config(page_title="Retail Order Splitter", layout="wide")
st.title("Retail Order Splitter – Anchor-Based (HD & Lowe's)")

def _default_map_for(state_prefix: str) -> Path:
    fname = "vendor_map.xlsx"
    if state_prefix == "hd":
        fname = "vendor_map_hd.xlsx"
    elif state_prefix == "lw":
        fname = "vendor_map_lowes.xlsx"
    elif state_prefix == "tsc":
        fname = "vendor_map_tsc.xlsx"
        fname = "vendor_map_lowes.xlsx"
    return Path("data")/fname

# ---- Utilities ----
import re as _re
def _norm_vendor(s: str) -> str:
    s = (s or "").upper()
    return _re.sub(r"[^A-Z0-9]", "", s)

PRINT_PACK_VENDORS = ["Cord Mate", "Cornerstone", "Gate Latch", "Home Selects", "Nisus", "Post Protector-Here", "Soft Seal", "Weedshark"]

def _build_print_pack_alpha(out_pdfs: dict, out_root: Path, master_name: str):
    """STRICT normalized equality, alphabetical by vendor name"""
    from pypdf import PdfReader, PdfWriter
    targets = { _norm_vendor(t) for t in PRINT_PACK_VENDORS }
    matched = {}
    for vendor_name, pth in out_pdfs.items():
        nk = _norm_vendor(vendor_name)
        if nk in targets and Path(pth).exists():
            matched[vendor_name] = pth
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

# ---- Sidebar: Vendor Map managers ----
with st.sidebar:
    st.header("Vendor Map")
    # Home Depot map
    st.subheader("Home Depot Map")
    def_path_hd = Path("data")/"vendor_map_hd.xlsx"
    if def_path_hd.exists():
        st.caption(f"Default (HD): **{def_path_hd.name}**")
        try:
            from datetime import datetime as _dt
            st.caption("Last updated: " + _dt.fromtimestamp(def_path_hd.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"))
        except Exception: pass
        try:
            with open(def_path_hd, "rb") as _f:
                st.download_button("Download HD default map", _f.read(), file_name=def_path_hd.name, key="dl_default_map_hd")
        except Exception:
            st.caption("HD default map not readable.")
    else:
        st.caption("No HD default map yet.")
    new_map_hd = st.file_uploader("Upload new HD vendor map (.xlsx)", type=["xlsx"], key="upl_new_map_hd")
    if new_map_hd is not None:
        if st.button("Set as HD default", key="btn_set_default_hd"):
            def_path_hd.parent.mkdir(parents=True, exist_ok=True)
            with open(def_path_hd, "wb") as _out:
                _out.write(new_map_hd.getvalue())
            st.success("Home Depot default vendor map updated.")
    st.markdown("---")
    # Lowe's map
    st.subheader("Lowe's Map")
    def_path_lw = Path("data")/"vendor_map_lowes.xlsx"
    if def_path_lw.exists():
        st.caption(f"Default (Lowe's): **{def_path_lw.name}**")
        try:
            from datetime import datetime as _dt
            st.caption("Last updated: " + _dt.fromtimestamp(def_path_lw.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"))
        except Exception: pass
        try:
            with open(def_path_lw, "rb") as _f:
                st.download_button("Download Lowe's default map", _f.read(), file_name=def_path_lw.name, key="dl_default_map_lw")
        except Exception:
            st.caption("Lowe's default map not readable.")
    else:
        st.caption("No Lowe's default map yet.")
    new_map_lw = st.file_uploader("Upload new Lowe's vendor map (.xlsx)", type=["xlsx"], key="upl_new_map_lw")
    if new_map_lw is not None:
        if st.button("Set as Lowe's default", key="btn_set_default_lw"):
            def_path_lw.parent.mkdir(parents=True, exist_ok=True)
            with open(def_path_lw, "wb") as _out:
                _out.write(new_map_lw.getvalue())
            st.success("Lowe's default vendor map updated.")

st.markdown("---")
# Tractor Supply map
# ---- Sidebar: Downloads & history ----
with st.sidebar:
    st.header("Downloads")
    # HD
    with st.expander("Home Depot", expanded=False):
        if st.session_state.get("hd_zip_bytes"):
            st.download_button("Download current ZIP (HD)", st.session_state["hd_zip_bytes"], file_name=st.session_state.get("hd_zip_name","vendor_pdfs.zip"), key="dl_zip_sidebar_hd")
        else:
            st.caption("Run a Home Depot batch to enable ZIP.")
        if st.session_state.get("hd_print_pack_bytes"):
            st.download_button("Download print pack (HD)", st.session_state["hd_print_pack_bytes"], file_name=st.session_state.get("hd_print_pack_name","Print Pack.pdf"), key="dl_printpack_sidebar_hd")
            inc = st.session_state.get("hd_print_pack_included")
            if inc:
                st.caption("Included (HD): " + ", ".join(sorted(set(inc))))
        else:
            st.caption("Print Pack will appear here after HD processing.")
    # Lowe's
    with st.expander("Lowe's", expanded=False):
        if st.session_state.get("lw_zip_bytes"):
            st.download_button("Download current ZIP (Lowe's)", st.session_state["lw_zip_bytes"], file_name=st.session_state.get("lw_zip_name","vendor_pdfs.zip"), key="dl_zip_sidebar_lw")
        else:
            st.caption("Run a Lowe's batch to enable ZIP.")
        if st.session_state.get("lw_print_pack_bytes"):
            st.download_button("Download print pack (Lowe's)", st.session_state["lw_print_pack_bytes"], file_name=st.session_state.get("lw_print_pack_name","Print Pack.pdf"), key="dl_printpack_sidebar_lw")
            inc2 = st.session_state.get("lw_print_pack_included")
            if inc2:
                st.caption("Included (Lowe's): " + ", ".join(sorted(set(inc2))))
        else:
            st.caption("Print Pack will appear here after Lowe's processing.")
    
# Tractor Supply
with st.expander("Tractor Supply", expanded=False):
    if st.session_state.get("tsc_zip_bytes"):
        st.download_button("Download current ZIP (TSC)", st.session_state["tsc_zip_bytes"], file_name=st.session_state.get("tsc_zip_name","vendor_pdfs.zip"), key="dl_zip_sidebar_tsc")
    else:
        st.caption("Run a Tractor Supply batch to enable ZIP.")
    if st.session_state.get("tsc_print_pack_bytes"):
        st.download_button("Download print pack (TSC)", st.session_state["tsc_print_pack_bytes"], file_name=st.session_state.get("tsc_print_pack_name","Print Pack.pdf"), key="dl_printpack_sidebar_tsc")
        inc3 = st.session_state.get("tsc_print_pack_included")
        if inc3:
            st.caption("Included (TSC): " + ", ".join(sorted(set(inc3))))
    else:
        st.caption("Print Pack will appear here after Tractor Supply processing.")

# History
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

# ---- Retailer Runner ----
def run_retailer_panel(label: str, core_module, state_prefix: str, out_subdir: str):
    st.markdown(f"### {label}")
    pdf_key = f"pdf_upl_{state_prefix}"
    map_key = f"map_upl_{state_prefix}"
    run_key = f"run_btn_{state_prefix}"
    clear_key = f"clear_btn_{state_prefix}"

    pdf_file = st.file_uploader(f"Upload {label} order PDF", type=["pdf"], key=pdf_key)
    map_file = st.file_uploader("Upload SKU/Model → Vendor map (.xlsx) (optional)", type=["xlsx"], key=map_key)
    threshold = st.slider("Auto-route confidence threshold", 0.70, 0.99, 0.88, 0.01, key=f"th_{state_prefix}")

    col_run, col_clear = st.columns([1,1])
    has_default_map = _default_map_for(state_prefix).exists()
    run_disabled = not (pdf_file and (map_file or has_default_map))
    run = col_run.button("Process PDF", type="primary", disabled=run_disabled, key=run_key)
    clear = col_clear.button("Clear Results", key=clear_key)

    def _k(k): return f"{state_prefix}_{k}"

    if clear:
        for k in ["report_df","review_df","zip_bytes","zip_name","vendor_counts","out_pdfs",
                  "print_pack_bytes","print_pack_name","print_pack_disk_path","print_pack_included",
                  "pdf_path","out_root","master_name","vendor_options"]:
            st.session_state[_k(k)] = None if k not in ["out_pdfs"] else {}
        st.experimental_rerun()

    for k in ["report_df","review_df","zip_bytes","zip_name","vendor_counts","out_pdfs",
              "print_pack_bytes","print_pack_name","print_pack_disk_path","print_pack_included",
              "pdf_path","out_root","master_name","vendor_options"]:
        st.session_state.setdefault(_k(k), None if k not in ["out_pdfs"] else {})

    def _persist_and_show():
        rep_df = st.session_state.get(_k("report_df"))
        rev_df = st.session_state.get(_k("review_df"))
        vc = st.session_state.get(_k("vendor_counts"))
        if rep_df is not None:
            st.subheader("Run report")
            st.dataframe(rep_df, use_container_width=True)
            st.download_button("Download split_report.csv", rep_df.to_csv(index=False).encode(), file_name="split_report.csv", key=f"dl_rep_{state_prefix}")
        if rev_df is not None and not rev_df.empty:
            st.warning(f"{len(rev_df)} page(s) need review. See below.")
            st.dataframe(rev_df, use_container_width=True)
            st.download_button("Download errors_low_confidence.csv", rev_df.to_csv(index=False).encode(), file_name="errors_low_confidence.csv", key=f"dl_rev_{state_prefix}")
        elif rep_df is not None:
            st.success("No review needed. All pages auto-routed above threshold.")
        if vc is not None:
            st.subheader("Pages per Vendor")
            if hasattr(vc, "to_frame"):
                df_counts = vc.to_frame(name="Pages").reset_index().rename(columns={"index":"Vendor"})
            else:
                df_counts = pd.DataFrame({"Vendor": [], "Pages": []})
            total_pages = int(df_counts["Pages"].sum()) if not df_counts.empty else 0
            st.dataframe(df_counts, use_container_width=True)
            st.write("**Total pages:**", total_pages)
        if st.session_state.get(_k("zip_bytes")):
            st.download_button("Download vendor PDFs (ZIP)", st.session_state[_k("zip_bytes")], file_name=st.session_state.get(_k("zip_name"), "vendor_pdfs.zip"), key=f"dl_zip_{state_prefix}")

    if run:
        out_root = Path("output")/out_subdir/ datetime.now().strftime("%Y-%m-%d")
        out_root.mkdir(parents=True, exist_ok=True)
        st.session_state[_k("out_root")] = str(out_root)

        if getattr(pdf_file, "name", None):
            master_name = Path(pdf_file.name).stem
        else:
            master_name = "Batch"
        st.session_state[_k("master_name")] = master_name
        pdf_path = out_root/"input.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getvalue())
        st.session_state[_k("pdf_path")] = str(pdf_path)

        if map_file is not None:
            map_path = out_root/"vendor_map.xlsx"
            with open(map_path, "wb") as f:
                f.write(map_file.getvalue())
        else:
            _def = _default_map_for(state_prefix)
            if not _def.exists():
                st.error("No vendor map provided and no default found. Upload or set a default map in the sidebar.")
                st.stop()
            map_path = out_root/"vendor_map.xlsx"
            import shutil as _sh
            _sh.copyfile(_def, map_path)

        try:
            vmap = load_vendor_map(str(map_path))
        except Exception as e:
            st.error(f"Failed to load vendor map: {e}")
            st.stop()

        vendor_options = sorted(set(vmap.values()))
        st.session_state[_k("vendor_options")] = vendor_options

        with st.spinner("Extracting and splitting pages..."):
            import os as _os
            _os.environ['HD_MASTER_NAME'] = master_name
            report_rows, review_rows, out_pdfs = core_module.split_pdf_to_vendors(str(pdf_path), str(out_root), vmap, threshold=threshold)

        rep_df = pd.DataFrame(report_rows)
        err_df = pd.DataFrame(review_rows)
        st.session_state[_k("out_pdfs")] = out_pdfs if out_pdfs else {}
        if not rep_df.empty:
            vc = rep_df["vendor"].fillna("").replace("", pd.NA).dropna().value_counts()
        else:
            vc = pd.Series(dtype=int)

        st.session_state[_k("report_df")] = rep_df
        st.session_state[_k("review_df")] = err_df
        st.session_state[_k("vendor_counts")] = vc

        try:
            pp_path, included = _build_print_pack_alpha(st.session_state[_k("out_pdfs")], Path(st.session_state[_k("out_root")]), master_name)
            if pp_path:
                with open(pp_path, "rb") as f:
                    st.session_state[_k("print_pack_bytes")] = f.read()
                st.session_state[_k("print_pack_name")] = Path(pp_path).name
                st.session_state[_k("print_pack_disk_path")] = pp_path
                st.session_state[_k("print_pack_included")] = included
            else:
                st.session_state[_k("print_pack_bytes")] = None
                st.session_state[_k("print_pack_name")] = None
                st.session_state[_k("print_pack_disk_path")] = None
                st.session_state[_k("print_pack_included")] = []
        except Exception as e:
            st.warning(f"Could not build Print Pack: {e}")

        try:
            zip_buf = None
            if st.session_state[_k("out_pdfs")]:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as z:
                    for vend, path in st.session_state[_k("out_pdfs")].items():
                        z.write(path, arcname=f"{Path(path).name}")
                    if st.session_state.get(_k("print_pack_disk_path")):
                        z.write(st.session_state[_k("print_pack_disk_path")], arcname=Path(st.session_state[_k("print_pack_disk_path")]).name)
                zip_bytes = zip_buf.getvalue()
            else:
                zip_bytes = None
            zip_name = f"{master_name} - vendor_pdfs.zip"
            if zip_bytes:
                zip_disk_path = Path(st.session_state[_k("out_root")]) / zip_name
                with open(zip_disk_path, "wb") as _zf:
                    _zf.write(zip_bytes)
        except Exception as e:
            st.warning(f"Could not build ZIP: {e}")

        st.session_state[_k("zip_bytes")] = zip_bytes
        st.session_state[_k("zip_name")] = zip_name

    _persist_and_show()

    st.markdown("---")
    st.subheader("Review & Fix")
    rev_df = st.session_state.get(_k("review_df"))
    if rev_df is not None and not rev_df.empty:
        vendor_options = st.session_state.get(_k("vendor_options"), [])
        if not vendor_options:
            st.info("Upload a vendor map to populate the dropdown options.")
        else:
            st.write("Pick a vendor for each page below, then click **Apply selections**.")
            selections = {}
            for idx, row in rev_df.iterrows():
                page_no = int(row['page'])
                key_sel = f"sel_vendor_{state_prefix}_{page_no}"
                sel = st.selectbox(f"Page {page_no} – detected: {row.get('best_value','')}", vendor_options, key=key_sel)
                selections[page_no] = sel
            apply = st.button("Apply selections and update vendor files", type="primary", key=f"apply_sel_btn_{state_prefix}")
            if apply:
                try:
                    from pypdf import PdfReader, PdfWriter
                    src_pdf = st.session_state.get(_k("pdf_path"))
                    out_root = st.session_state.get(_k("out_root"))
                    if not src_pdf or not out_root:
                        st.error("Missing source PDF path or output directory in session.")
                    else:
                        rdr = PdfReader(src_pdf)
                        rep_df = st.session_state.get(_k("report_df")).copy()
                        vpages = {}
                        if rep_df is not None and not rep_df.empty:
                            df_ok = rep_df[(rep_df['vendor'].astype(str)!='') & ((rep_df['flag'].astype(str)=='') | (rep_df['flag'].isna()))]
                            for _, r in df_ok.iterrows():
                                v = str(r['vendor'])
                                pnum = int(r['page']) - 1
                                vpages.setdefault(v, []).append(pnum)
                        for pnum, vend in selections.items():
                            vpages.setdefault(vend, []).append(int(pnum)-1)
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
                            out_path = vend_dir / f"{st.session_state.get(_k('master_name'), 'Batch')} {v}.pdf"
                            with open(out_path, "wb") as f:
                                w.write(f)
                            out_pdfs[v] = str(out_path)
                        st.session_state[_k("out_pdfs")] = out_pdfs if out_pdfs else {}
                        for pnum, vend in selections.items():
                            rep_df.loc[rep_df['page'] == pnum, ['vendor','flag']] = [vend, '']
                        review_df = rev_df.copy()
                        review_df = review_df[~review_df['page'].isin(list(selections.keys()))]
                        st.session_state[_k("report_df")] = rep_df
                        st.session_state[_k("review_df")] = review_df
                        if not rep_df.empty:
                            vc = rep_df['vendor'].fillna('').replace('', pd.NA).dropna().value_counts()
                        else:
                            vc = pd.Series(dtype=int)
                        st.session_state[_k("vendor_counts")] = vc
                        try:
                            pp_path, included = _build_print_pack_alpha(out_pdfs, Path(out_root), st.session_state.get(_k('master_name'),'Batch'))
                            if pp_path:
                                with open(pp_path, 'rb') as f:
                                    st.session_state[_k('print_pack_bytes')] = f.read()
                                st.session_state[_k('print_pack_name')] = Path(pp_path).name
                                st.session_state[_k('print_pack_disk_path')] = pp_path
                                st.session_state[_k('print_pack_included')] = included
                            else:
                                st.session_state[_k('print_pack_bytes')] = None
                                st.session_state[_k('print_pack_name')] = None
                                st.session_state[_k('print_pack_disk_path')] = None
                                st.session_state[_k('print_pack_included')] = []
                        except Exception as e:
                            st.warning(f"Could not build Print Pack: {e}")
                        try:
                            zip_buf = None
                            if out_pdfs:
                                zip_buf = io.BytesIO()
                                with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as z:
                                    for vend, path in out_pdfs.items():
                                        z.write(path, arcname=Path(path).name)
                                    if st.session_state.get(_k('print_pack_disk_path')):
                                        z.write(st.session_state[_k('print_pack_disk_path')], arcname=Path(st.session_state[_k('print_pack_disk_path')]).name)
                                zip_bytes = zip_buf.getvalue()
                            else:
                                zip_bytes = None
                            zip_name = f"{st.session_state.get(_k('master_name'),'Batch')} - vendor_pdfs.zip"
                            if zip_bytes:
                                zip_disk_path = Path(out_root) / zip_name
                                with open(zip_disk_path, 'wb') as _zf:
                                    _zf.write(zip_bytes)
                        except Exception as e:
                            st.warning(f"Could not build ZIP: {e}")
                        st.session_state[_k('zip_bytes')] = zip_bytes
                        st.session_state[_k('zip_name')] = zip_name
                        st.success("Selections applied. Vendor PDFs and Print Pack updated.")
                except Exception as e:
                    st.error(f"Failed to apply selections: {e}")
    else:
        st.caption("No uncertain pages to review.")

# ---- Tabs ----
tab_hd, tab_lw, tab_tsc = st.tabs(["Home Depot", "Lowe's", "Tractor Supply"])
with tab_hd:
    run_retailer_panel("Home Depot", split_core, "hd", "HD")
with tab_lw:
    run_retailer_panel("Lowe's", lowes_core, "lw", "Lowes")
with tab_tsc:
    run_retailer_panel("Tractor Supply", tsc_core, "tsc", "TSC")
