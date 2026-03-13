from __future__ import annotations

from datetime import date
import html as _html
import re as _re

import pandas as pd
import streamlit as st

from .app_core import read_weekly_workbook
from .shared_core import (
    APP_TITLE,
    load_vendor_map,
    load_store,
    save_store,
    enrich_sales,
    available_month_labels,
    available_year_labels,
    filter_by_period_labels,
    period_from_df,
    compact_selection_label,
    pick_period,
    filter_by_period,
    period_prev_same_length,
    period_yoy,
    ab_labels,
    format_period_range,
    calc_kpis,
    drivers,
    money,
    render_data_management_center,
)
from .ui_styles import apply_global_styles
from . import (
    tab_standard_intelligence,
    tab_month_year_compare,
    tab_multi_compare,
    tab_lookup_center,
)


def render_global_view_mode_toggle():
    """
    Global sidebar selector that controls whether the app shows:
    - existing tab/module content
    - visual analytics versions of those sections
    """
    st.markdown("")
    try:
        selected = st.sidebar.segmented_control(
            "Content View",
            options=["Model View", "Visual Analytics"],
            default=st.session_state.get("global_content_view", "Model View"),
            key="global_content_view",
        )
    except Exception:
        selected = st.sidebar.radio(
            "Content View",
            options=["Model View", "Visual Analytics"],
            index=0 if st.session_state.get("global_content_view", "Model View") == "Model View" else 1,
            key="global_content_view",
        )

    if not selected:
        selected = "Model View"

    st.session_state["global_content_view"] = selected
    return selected


def get_global_view_mode():
    return st.session_state.get("global_content_view", "Model View")


def _safe_grouped_table(df: pd.DataFrame, by: list[str], agg_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    use_by = [c for c in by if c in df.columns]
    if not use_by:
        return pd.DataFrame()

    agg_map = {}
    for c in agg_cols:
        if c in df.columns:
            agg_map[c] = "sum"

    if not agg_map:
        return pd.DataFrame()

    out = (
        df.groupby(use_by, dropna=False, as_index=False)
        .agg(agg_map)
        .copy()
    )
    return out


def _find_date_col(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None

    candidates = [
        "Week End",
        "Week Ending",
        "WeekEnding",
        "Week_End",
        "Date",
        "Period End",
        "PeriodEnd",
        "End Date",
        "Week",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _make_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    date_col = _find_date_col(df)
    if not date_col:
        return pd.DataFrame()

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])

    agg = {}
    if "Sales" in work.columns:
        agg["Sales"] = "sum"
    if "Units" in work.columns:
        agg["Units"] = "sum"

    if not agg:
        return pd.DataFrame()

    out = (
        work.groupby(date_col, as_index=False)
        .agg(agg)
        .sort_values(date_col)
        .rename(columns={date_col: "Period"})
    )
    return out


def _top_table(df: pd.DataFrame, group_col: str, n: int = 10, sort_col: str = "Sales") -> pd.DataFrame:
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame()

    agg = {}
    if "Sales" in df.columns:
        agg["Sales"] = "sum"
    if "Units" in df.columns:
        agg["Units"] = "sum"

    if not agg:
        return pd.DataFrame()

    out = (
        df.groupby(group_col, dropna=False, as_index=False)
        .agg(agg)
        .sort_values(sort_col if sort_col in agg else list(agg.keys())[0], ascending=False)
        .head(n)
        .copy()
    )
    return out


def _render_kpi_strip(dfA: pd.DataFrame, dfB: pd.DataFrame | None = None, a_lbl: str | None = None, b_lbl: str | None = None):
    kA = calc_kpis(dfA) if dfA is not None and not dfA.empty else {"Sales": 0.0, "Units": 0.0, "ASP": 0.0}
    kB = calc_kpis(dfB) if dfB is not None and not dfB.empty else {"Sales": 0.0, "Units": 0.0, "ASP": 0.0}

    sales_delta = kA.get("Sales", 0.0) - kB.get("Sales", 0.0)
    units_delta = kA.get("Units", 0.0) - kB.get("Units", 0.0)
    asp_delta = kA.get("ASP", 0.0) - kB.get("ASP", 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"Sales{f' ({a_lbl})' if a_lbl else ''}",
        money(float(kA.get("Sales", 0.0))),
        None if (dfB is None or dfB.empty or not b_lbl) else money(float(sales_delta)),
    )
    c2.metric(
        f"Units{f' ({a_lbl})' if a_lbl else ''}",
        f"{float(kA.get('Units', 0.0)):,.0f}",
        None if (dfB is None or dfB.empty or not b_lbl) else f"{float(units_delta):,.0f}",
    )
    c3.metric(
        f"ASP{f' ({a_lbl})' if a_lbl else ''}",
        money(float(kA.get("ASP", 0.0))),
        None if (dfB is None or dfB.empty or not b_lbl) else money(float(asp_delta)),
    )


def _render_timeseries_block(dfA: pd.DataFrame, title: str):
    st.markdown(f"#### {title}")
    ts = _make_timeseries(dfA)
    if ts.empty:
        st.info("No time-series column was found for this selection.")
        return

    if "Sales" in ts.columns:
        st.markdown("**Sales Trend**")
        st.line_chart(ts.set_index("Period")[["Sales"]], use_container_width=True)

    if "Units" in ts.columns:
        st.markdown("**Units Trend**")
        st.line_chart(ts.set_index("Period")[["Units"]], use_container_width=True)


def _render_top_tables(dfA: pd.DataFrame):
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Top SKUs")
        sku_tbl = _top_table(dfA, "SKU", n=12, sort_col="Sales")
        if sku_tbl.empty:
            st.info("No SKU data available.")
        else:
            st.dataframe(sku_tbl, use_container_width=True, hide_index=True)

        st.markdown("#### Top Vendors")
        ven_tbl = _top_table(dfA, "Vendor", n=12, sort_col="Sales")
        if ven_tbl.empty:
            st.info("No Vendor data available.")
        else:
            st.dataframe(ven_tbl, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("#### Top Retailers")
        ret_tbl = _top_table(dfA, "Retailer", n=12, sort_col="Sales")
        if ret_tbl.empty:
            st.info("No Retailer data available.")
        else:
            st.dataframe(ret_tbl, use_container_width=True, hide_index=True)

        st.markdown("#### Sales Mix")
        mix_tbl = _safe_grouped_table(dfA, ["Retailer"], ["Sales", "Units"])
        if mix_tbl.empty:
            st.info("No mix table available.")
        else:
            mix_tbl = mix_tbl.sort_values("Sales", ascending=False).head(15)
            st.dataframe(mix_tbl, use_container_width=True, hide_index=True)


def _render_driver_tables(dfA: pd.DataFrame, dfB: pd.DataFrame, driver_level: str):
    st.markdown("#### Change Drivers")
    drv = drivers(dfA, dfB, driver_level)
    if drv is None or drv.empty:
        st.info("No comparison drivers available for this selection.")
        return

    col_name = driver_level if driver_level in drv.columns else drv.columns[0]

    pos = drv[drv["Sales_Δ"] > 0].sort_values("Sales_Δ", ascending=False).head(10).copy() if "Sales_Δ" in drv.columns else pd.DataFrame()
    neg = drv[drv["Sales_Δ"] < 0].sort_values("Sales_Δ", ascending=True).head(10).copy() if "Sales_Δ" in drv.columns else pd.DataFrame()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Top Increases**")
        if pos.empty:
            st.info("No positive drivers.")
        else:
            keep = [c for c in [col_name, "Sales_Δ", "Units_Δ", "ASP_Δ"] if c in pos.columns]
            st.dataframe(pos[keep], use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**Top Decreases**")
        if neg.empty:
            st.info("No negative drivers.")
        else:
            keep = [c for c in [col_name, "Sales_Δ", "Units_Δ", "ASP_Δ"] if c in neg.columns]
            st.dataframe(neg[keep], use_container_width=True, hide_index=True)


def _render_scope_breakdown(dfA: pd.DataFrame):
    st.markdown("#### Breakdown Tables")
    c1, c2, c3 = st.columns(3)

    with c1:
        tbl = _top_table(dfA, "Retailer", n=20)
        st.markdown("**Retailer Breakdown**")
        if tbl.empty:
            st.info("No retailer breakdown available.")
        else:
            st.dataframe(tbl, use_container_width=True, hide_index=True)

    with c2:
        tbl = _top_table(dfA, "Vendor", n=20)
        st.markdown("**Vendor Breakdown**")
        if tbl.empty:
            st.info("No vendor breakdown available.")
        else:
            st.dataframe(tbl, use_container_width=True, hide_index=True)

    with c3:
        tbl = _top_table(dfA, "SKU", n=20)
        st.markdown("**SKU Breakdown**")
        if tbl.empty:
            st.info("No SKU breakdown available.")
        else:
            st.dataframe(tbl, use_container_width=True, hide_index=True)


def render_standard_intelligence_visual(ctx: dict):
    st.subheader("Standard Intelligence • Visual Analytics")
    _render_kpi_strip(ctx["dfA"], ctx["dfB"], ctx.get("a_lbl"), ctx.get("b_lbl"))
    st.divider()
    _render_timeseries_block(ctx["dfA"], "Trend View")
    st.divider()
    _render_top_tables(ctx["dfA"])

    if ctx.get("compare_mode") != "None" and ctx.get("dfB") is not None and not ctx["dfB"].empty:
        st.divider()
        _render_driver_tables(ctx["dfA"], ctx["dfB"], ctx.get("driver_level", "SKU"))


def render_month_year_compare_visual(ctx: dict):
    st.subheader("Month / Year Compare • Visual Analytics")
    _render_kpi_strip(ctx["dfA"], ctx["dfB"], ctx.get("a_lbl"), ctx.get("b_lbl"))
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### Current • {ctx.get('a_lbl', 'Selection')}")
        _render_timeseries_block(ctx["dfA"], "Current Selection Trend")

    with c2:
        st.markdown(f"#### Compare • {ctx.get('b_lbl', 'Selection')}")
        if ctx["dfB"] is None or ctx["dfB"].empty:
            st.info("No compare selection.")
        else:
            _render_timeseries_block(ctx["dfB"], "Compare Selection Trend")

    st.divider()
    _render_driver_tables(ctx["dfA"], ctx["dfB"], ctx.get("driver_level", "SKU"))
    st.divider()
    _render_scope_breakdown(ctx["dfA"])


def render_multi_compare_visual(ctx: dict):
    st.subheader("Multi Month / Year Compare • Visual Analytics")
    st.info("Use the controls inside the existing multi-compare module for the detailed matrix logic. This visual layer gives you a quick dashboard view of the filtered dataset.")
    _render_kpi_strip(ctx["dfA"], None, "Selected Scope", None)
    st.divider()
    _render_timeseries_block(ctx["dfA"], "Filtered Trend")
    st.divider()
    _render_scope_breakdown(ctx["dfA"])


def render_lookup_center_visual(ctx: dict):
    st.subheader("Lookup Center • Visual Analytics")
    _render_kpi_strip(ctx["dfA"], None, ctx.get("a_lbl"), None)
    st.divider()
    _render_timeseries_block(ctx["dfA"], "Lookup Trend")
    st.divider()
    _render_top_tables(ctx["dfA"])
    st.divider()
    _render_scope_breakdown(ctx["dfA"])


def render_current_analysis_view(ctx: dict):
    """
    Existing module-driven experience.
    """
    analysis_view = ctx.get("analysis_view")

    if analysis_view == "Standard Intelligence":
        tab_standard_intelligence.render(ctx)
    elif analysis_view == "Month / Year Compare":
        tab_month_year_compare.render(ctx)
    elif analysis_view == "Multi Month / Year Compare":
        tab_multi_compare.render(ctx)
    elif analysis_view == "Lookup Center":
        tab_lookup_center.render(ctx)


def render_visual_analysis_view(ctx: dict):
    """
    New visual/table-first experience driven by the same filters.
    """
    analysis_view = ctx.get("analysis_view")

    if analysis_view == "Standard Intelligence":
        render_standard_intelligence_visual(ctx)
    elif analysis_view == "Month / Year Compare":
        render_month_year_compare_visual(ctx)
    elif analysis_view == "Multi Month / Year Compare":
        render_multi_compare_visual(ctx)
    elif analysis_view == "Lookup Center":
        render_lookup_center_visual(ctx)


def run_app():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    apply_global_styles()
    st.title(APP_TITLE)

    vm = load_vendor_map()
    store = load_store()

    with st.sidebar:
        st.header("Data")
        up = st.file_uploader("Upload weekly sales workbook (.xlsx)", type=["xlsx"])
        year = st.number_input(
            "Year hint (for filename parsing)",
            min_value=2010,
            max_value=2100,
            value=date.today().year,
            step=1,
        )

        if st.button("Ingest upload", disabled=(up is None)):
            if up is not None:
                raw = read_weekly_workbook(up, int(year))
                _ = enrich_sales(raw, vm)
                merged = pd.concat([store, raw], ignore_index=True)
                save_store(merged)
                st.success(f"Ingested {len(raw):,} rows from {getattr(up, 'name', 'upload.xlsx')}.")
                store = load_store()

        st.divider()
        st.header("Filters")

        scope = st.selectbox("Scope", ["All", "Retailer", "Vendor", "SKU"], index=0)

        df_all = enrich_sales(store, vm)
        scope_pick = None

        if scope == "Retailer":
            scope_pick = st.multiselect(
                "Retailer(s)",
                options=sorted(df_all["Retailer"].dropna().unique()) if "Retailer" in df_all.columns else [],
                default=[],
            )
        elif scope == "Vendor":
            scope_pick = st.multiselect(
                "Vendor(s)",
                options=sorted(df_all["Vendor"].dropna().unique()) if "Vendor" in df_all.columns else [],
                default=[],
            )
        elif scope == "SKU":
            scope_pick = st.multiselect(
                "SKU(s)",
                options=sorted(df_all["SKU"].dropna().unique()) if "SKU" in df_all.columns else [],
                default=[],
            )

        analysis_view = st.radio(
            "Analysis View",
            [
                "Standard Intelligence",
                "Month / Year Compare",
                "Multi Month / Year Compare",
                "Lookup Center",
                "Data Management Center",
            ],
            index=0,
        )

        # NEW GLOBAL TOGGLE
        content_view = render_global_view_mode_toggle()

        multi_granularity = "Month"
        current_labels_sel = []
        compare_labels_sel = []

        if analysis_view == "Data Management Center":
            timeframe = "YTD"
            compare_mode = "None"

        elif analysis_view == "Standard Intelligence":
            timeframe = st.selectbox(
                "Timeframe",
                [
                    "Week (latest)",
                    "Last 4 weeks",
                    "Last 8 weeks",
                    "Last 13 weeks",
                    "Last 26 weeks",
                    "Last 52 weeks",
                    "YTD",
                ],
                index=2,
            )
            compare_mode = st.selectbox(
                "Compare",
                ["None", "Prior period (same length)", "YoY (same dates)"],
                index=1,
            )

        elif analysis_view == "Month / Year Compare":
            multi_granularity = st.selectbox(
                "Compare By",
                ["Month", "Year"],
                index=0,
                key="my_compare_by",
            )
            period_options = (
                available_month_labels(df_all)
                if multi_granularity == "Month"
                else available_year_labels(df_all)
            )
            timeframe = "Custom Months" if multi_granularity == "Month" else "Custom Years"

            current_one = st.selectbox(
                f"Current {multi_granularity}",
                options=period_options,
                index=(len(period_options) - 1 if period_options else 0),
            )
            compare_one = st.selectbox(
                f"Compare {multi_granularity}",
                options=period_options,
                index=(len(period_options) - 2 if len(period_options) > 1 else 0),
            )

            current_labels_sel = [current_one] if current_one else []
            compare_labels_sel = [compare_one] if compare_one else []
            compare_mode = "Custom selection" if compare_labels_sel else "None"

        elif analysis_view == "Lookup Center":
            timeframe = "Last 8 weeks"
            compare_mode = "None"

        else:
            timeframe = "Multi Selection"
            compare_mode = "None"
            multi_granularity = "Year"
            current_labels_sel = []
            compare_labels_sel = []

        min_sales = st.number_input("Min Sales ($) for lists", min_value=0.0, value=0.0, step=100.0)
        min_units = st.number_input("Min Units for lists", min_value=0.0, value=0.0, step=10.0)
        driver_level = st.selectbox("Driver Level", ["SKU", "Vendor", "Retailer"], index=0)
        show_full_history_lifecycle = st.toggle("Lifecycle uses full history", value=True)

    df_scope = df_all.copy()
    if scope == "Retailer" and scope_pick:
        df_scope = df_scope[df_scope["Retailer"].isin(scope_pick)]
    elif scope == "Vendor" and scope_pick:
        df_scope = df_scope[df_scope["Vendor"].isin(scope_pick)]
    elif scope == "SKU" and scope_pick:
        df_scope = df_scope[df_scope["SKU"].isin(scope_pick)]

    if analysis_view == "Data Management Center":
        render_data_management_center(vm, store)
        return

    if analysis_view == "Month / Year Compare":
        dfA = filter_by_period_labels(df_scope, current_labels_sel, multi_granularity)
        dfB = (
            filter_by_period_labels(df_scope, compare_labels_sel, multi_granularity)
            if compare_labels_sel
            else df_scope.iloc[0:0].copy()
        )
        pA = period_from_df(dfA)
        pB = period_from_df(dfB) if not dfB.empty else None

        if pA is None:
            st.info("Choose one or more months/years to begin.")
            return

        a_lbl = compact_selection_label(current_labels_sel, multi_granularity)
        b_lbl = compact_selection_label(compare_labels_sel, multi_granularity) if pB is not None else None

    elif analysis_view == "Multi Month / Year Compare":
        dfA = df_scope.copy()
        dfB = df_scope.iloc[0:0].copy()
        pA = None
        pB = None
        a_lbl = None
        b_lbl = None

    else:
        pA = pick_period(df_scope, timeframe)
        if pA is None:
            st.info("Upload or ingest data to begin.")
            return

        dfA = filter_by_period(df_scope, pA)

        if compare_mode == "None":
            pB = None
            dfB = dfA.iloc[0:0].copy()
        elif compare_mode.startswith("Prior"):
            pB = period_prev_same_length(pA)
            dfB = filter_by_period(df_scope, pB)
        else:
            pB = period_yoy(pA)
            dfB = filter_by_period(df_scope, pB)

        a_lbl, b_lbl = ab_labels(timeframe, compare_mode, pA, pB)

    st.sidebar.markdown("### Period Definition")
    if analysis_view == "Multi Month / Year Compare":
        st.sidebar.markdown(
            "<span style='opacity:0.75'>Multi-select analysis mode</span>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            f"**Current:** {a_lbl}<br><span style='opacity:0.8'>{format_period_range(pA)}</span>",
            unsafe_allow_html=True,
        )
        if compare_mode != "None" and pB is not None:
            st.sidebar.markdown(
                f"**Compare:** {b_lbl}<br><span style='opacity:0.8'>{format_period_range(pB)}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                "<span style='opacity:0.75'>Compare: None</span>",
                unsafe_allow_html=True,
            )
    st.sidebar.divider()

    kA = calc_kpis(dfA)
    kB = calc_kpis(dfB) if pB is not None else {k: 0.0 for k in kA.keys()}

    df_hist_for_new = df_all.copy()
    if scope == "Retailer" and scope_pick:
        df_hist_for_new = df_hist_for_new[df_hist_for_new["Retailer"].isin(scope_pick)]
    elif scope == "Vendor" and scope_pick:
        df_hist_for_new = df_hist_for_new[df_hist_for_new["Vendor"].isin(scope_pick)]
    elif scope == "SKU" and scope_pick:
        df_hist_for_new = df_hist_for_new[df_hist_for_new["SKU"].isin(scope_pick)]

    if analysis_view == "Multi Month / Year Compare":
        st.markdown(
            f"""
            <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin: 4px 0 10px 0;">
                <span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(128,128,128,0.22); background: var(--secondary-background-color); color: var(--text-color); font-weight:700; font-size:12px;">Scope: {scope}</span>
                <span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(128,128,128,0.22); background: var(--secondary-background-color); color: var(--text-color); font-weight:700; font-size:12px;">Mode: Multi-select period analysis</span>
                <span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(128,128,128,0.22); background: var(--secondary-background-color); color: var(--text-color); font-weight:700; font-size:12px;">Content View: {content_view}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        cur_range = format_period_range(pA)
        cmp_range = format_period_range(pB) if pB is not None else ""
        cmp_name = b_lbl or ""
        st.markdown(
            f"""<div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin: 4px 0 10px 0;">
            <span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(128,128,128,0.22); background: var(--secondary-background-color); color: var(--text-color); font-weight:700; font-size:12px;">Scope: {scope}</span>
            <span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(128,128,128,0.22); background: var(--secondary-background-color); color: var(--text-color); font-weight:700; font-size:12px;">Current: {a_lbl} • {cur_range}</span>
            {(f'<span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(128,128,128,0.22); background: var(--secondary-background-color); color: var(--text-color); font-weight:700; font-size:12px;">Compare: {cmp_name} • {cmp_range}</span>' if compare_mode != "None" and pB is not None else '<span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(128,128,128,0.22); background: var(--secondary-background-color); color: var(--text-color); font-weight:700; font-size:12px; opacity:0.75;">Compare: None</span>')}
            <span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(128,128,128,0.22); background: var(--secondary-background-color); color: var(--text-color); font-weight:700; font-size:12px;">Content View: {content_view}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    headline_bits = []
    if analysis_view == "Multi Month / Year Compare":
        headline_bits.append("Use the controls inside this tab to analyze multiple months or years together.")
        headline_bits.append("The matrix table highlights the highest selected period in green and the lowest in red for each row.")
        headline_bits.append("Use Rows By to switch between retailer and vendor views.")
    else:
        sales_delta = kA["Sales"] - kB.get("Sales", 0.0)
        units_delta = kA["Units"] - kB.get("Units", 0.0)
        asp_delta = kA["ASP"] - kB.get("ASP", 0.0)
        drv = drivers(dfA, dfB, driver_level)
        top_pos = drv[drv["Sales_Δ"] > 0].head(1) if drv is not None and not drv.empty and "Sales_Δ" in drv.columns else pd.DataFrame()
        top_neg = drv[drv["Sales_Δ"] < 0].tail(1) if drv is not None and not drv.empty and "Sales_Δ" in drv.columns else pd.DataFrame()

        if compare_mode != "None":
            headline_bits.append(
                f"Sales {'up' if sales_delta >= 0 else 'down'} **{money(abs(sales_delta))}** vs comparison."
            )
            headline_bits.append(
                f"Units: **{kA['Units']:,.0f}** ({'up' if units_delta >= 0 else 'down'} **{abs(units_delta):,.0f}** vs comparison)."
            )
            headline_bits.append(
                f"ASP: **{money(kA['ASP'])}** ({'up' if asp_delta >= 0 else 'down'} **{money(abs(asp_delta))}** vs comparison)."
            )
            if not top_pos.empty:
                headline_bits.append(
                    f"Top driver: **{top_pos.iloc[0][driver_level]}** ({money(float(top_pos.iloc[0]['Sales_Δ']))})."
                )
            if not top_neg.empty:
                headline_bits.append(
                    f"Top drag: **{top_neg.iloc[0][driver_level]}** ({money(float(top_neg.iloc[0]['Sales_Δ']))})."
                )
        else:
            headline_bits.append("Choose a comparison mode to see drivers and deltas.")

    def _md_bold_to_html(s: str) -> str:
        parts = []
        last = 0
        for mm in _re.finditer(r"\*\*(.+?)\*\*", s):
            parts.append(_html.escape(s[last:mm.start()]))
            parts.append(f"<strong>{_html.escape(mm.group(1))}</strong>")
            last = mm.end()
        parts.append(_html.escape(s[last:]))
        return "".join(parts)

    _lis = "".join([f"<li>{_md_bold_to_html(x)}</li>" for x in headline_bits])
    st.markdown(
        f"<div class='intel-card'><div class='intel-header'>INTELLIGENCE SUMMARY</div><div class='intel-body'><ul>{_lis}</ul></div></div>",
        unsafe_allow_html=True,
    )

    ctx = dict(
        dfA=dfA,
        dfB=dfB,
        kA=kA,
        kB=kB,
        a_lbl=a_lbl,
        b_lbl=b_lbl,
        compare_mode=compare_mode,
        min_sales=min_sales,
        min_units=min_units,
        driver_level=driver_level,
        df_scope=df_scope,
        pA=pA,
        pB=pB,
        df_hist_for_new=df_hist_for_new,
        show_full_history_lifecycle=show_full_history_lifecycle,
        analysis_view=analysis_view,
        content_view=content_view,
        scope=scope,
        scope_pick=scope_pick,
    )

    if get_global_view_mode() == "Visual Analytics":
        render_visual_analysis_view(ctx)
    else:
        render_current_analysis_view(ctx)
