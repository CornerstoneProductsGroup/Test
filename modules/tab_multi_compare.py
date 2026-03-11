from __future__ import annotations

import math
import numpy as np
import pandas as pd
import streamlit as st

from .shared_core import (
    money,
    render_df,
    available_month_labels,
    available_year_labels,
    filter_by_period_labels,
)


def _fmt_value(v: float, metric: str) -> str:
    return money(v) if metric == "Sales" else f"{float(v):,.0f}"


def _spark(values) -> str:
    bars = "▁▂▃▄▅▆▇█"
    vals = [float(v) if pd.notna(v) else 0.0 for v in values]
    if not vals:
        return ""
    vmin = min(vals)
    vmax = max(vals)
    if math.isclose(vmax, vmin):
        return bars[3] * len(vals)
    out = []
    for v in vals:
        idx = int(round((v - vmin) / (vmax - vmin) * (len(bars) - 1)))
        idx = max(0, min(idx, len(bars) - 1))
        out.append(bars[idx])
    return "".join(out)


def _style_hi_lo(df_numeric: pd.DataFrame):
    def style_row(row):
        vals = pd.to_numeric(row, errors="coerce")
        valid = vals.dropna()
        styles = [""] * len(row)
        if valid.empty:
            return styles
        hi = valid.max()
        lo = valid.min()
        for i, col in enumerate(row.index):
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(val):
                continue
            if val == hi and hi != lo:
                styles[i] = "background-color: rgba(46,125,50,0.20); font-weight:700;"
            elif val == lo and hi != lo:
                styles[i] = "background-color: rgba(198,40,40,0.20); font-weight:700;"
        return styles

    return df_numeric.style.apply(style_row, axis=1)


def _build_matrix(df: pd.DataFrame, labels: list[str], granularity: str, row_dim: str, metric: str) -> pd.DataFrame:
    pieces = []
    for lbl in labels:
        part = filter_by_period_labels(df, [lbl], granularity)
        grp = part.groupby(row_dim, as_index=False).agg(Value=(metric, "sum"))
        grp = grp.rename(columns={"Value": lbl})
        pieces.append(grp)

    if not pieces:
        return pd.DataFrame(columns=[row_dim])

    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on=row_dim, how="outer")

    out = out.fillna(0.0)

    period_cols = [c for c in labels if c in out.columns]
    out["Total"] = out[period_cols].sum(axis=1)
    out["Average"] = out[period_cols].mean(axis=1) if period_cols else 0.0
    out["Trend"] = out[period_cols].apply(lambda r: _spark(r.tolist()), axis=1) if period_cols else ""
    return out


def _render_matrix_table(df_scope: pd.DataFrame, labels: list[str], granularity: str, row_dim: str, metric: str):
    st.markdown("### Multi-Period Matrix")

    if not labels:
        st.info("Select one or more periods.")
        return

    matrix = _build_matrix(df_scope, labels, granularity, row_dim, metric)

    if matrix.empty:
        st.info("No data available for the selected periods.")
        return

    period_cols = [c for c in labels if c in matrix.columns]
    numeric_cols = period_cols + ["Total", "Average"]

    show = matrix.copy()

    styled = _style_hi_lo(show[period_cols])

    for c in ["Total", "Average"]:
        if c in show.columns:
            pass

    formatted = show.copy()
    for c in period_cols + ["Total", "Average"]:
        if c in formatted.columns:
            formatted[c] = formatted[c].map(lambda v: _fmt_value(v, metric))

    # Rebuild styled df from numeric source but with formatted display strings
    base_numeric = show[[row_dim] + period_cols + ["Total", "Average", "Trend"]].copy()
    base_display = formatted[[row_dim] + period_cols + ["Total", "Average", "Trend"]].copy()

    def style_row(row):
        period_vals = pd.to_numeric(show.loc[row.name, period_cols], errors="coerce")
        valid = period_vals.dropna()
        styles = [""] * len(base_display.columns)
        if not valid.empty:
            hi = valid.max()
            lo = valid.min()
            for idx, col in enumerate(base_display.columns):
                if col in period_cols:
                    val = pd.to_numeric(show.loc[row.name, col], errors="coerce")
                    if pd.isna(val):
                        continue
                    if val == hi and hi != lo:
                        styles[idx] = "background-color: rgba(46,125,50,0.20); font-weight:700;"
                    elif val == lo and hi != lo:
                        styles[idx] = "background-color: rgba(198,40,40,0.20); font-weight:700;"
        return styles

    st.dataframe(
        base_display.style.apply(style_row, axis=1),
        use_container_width=True,
        height=520,
        hide_index=True,
    )


def _render_trend_table(df_scope: pd.DataFrame, labels: list[str], granularity: str, row_dim: str, metric: str):
    st.markdown("### Trend Summary")

    matrix = _build_matrix(df_scope, labels, granularity, row_dim, metric)
    if matrix.empty:
        st.info("No trend data available.")
        return

    period_cols = [c for c in labels if c in matrix.columns]
    if not period_cols:
        st.info("No periods selected.")
        return

    show = matrix[[row_dim] + period_cols + ["Trend"]].copy()
    for c in period_cols:
        show[c] = show[c].map(lambda v: _fmt_value(v, metric))

    render_df(show, height=420)


def _render_kpis(df_scope: pd.DataFrame, labels: list[str], granularity: str, metric: str):
    st.markdown("### Selected Period Summary")

    if not labels:
        st.info("Select one or more periods.")
        return

    df_sel = filter_by_period_labels(df_scope, labels, granularity)
    if df_sel.empty:
        st.info("No data for the selected periods.")
        return

    sales = float(df_sel["Sales"].sum()) if "Sales" in df_sel.columns else 0.0
    units = float(df_sel["Units"].sum()) if "Units" in df_sel.columns else 0.0
    asp = sales / units if units else 0.0
    retailers = int(df_sel["Retailer"].dropna().nunique()) if "Retailer" in df_sel.columns else 0
    vendors = int(df_sel["Vendor"].dropna().nunique()) if "Vendor" in df_sel.columns else 0
    skus = int(df_sel["SKU"].dropna().nunique()) if "SKU" in df_sel.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Total Sales", money(sales))
    with c2:
        st.metric("Total Units", f"{units:,.0f}")
    with c3:
        st.metric("ASP", money(asp))
    with c4:
        st.metric("Retailers", f"{retailers:,}")
    with c5:
        st.metric("Vendors", f"{vendors:,}")
    with c6:
        st.metric("SKUs", f"{skus:,}")


def render(ctx: dict):
    df_scope = ctx["df_scope"].copy()

    st.subheader("Multi Month / Year Compare")
    st.caption("Analyze multiple months or years together in one view.")

    if df_scope.empty:
        st.info("No data available with the current filters.")
        return

    c1, c2, c3, c4 = st.columns([1.0, 2.2, 1.2, 1.0])

    with c1:
        granularity = st.selectbox(
            "Analyze By",
            ["Year", "Month"],
            index=0,
            key="multi_compare_granularity_single",
        )

    options = available_year_labels(df_scope) if granularity == "Year" else available_month_labels(df_scope)

    default_sel = options[-4:] if len(options) >= 4 else options

    with c2:
        labels = st.multiselect(
            f"Select {granularity}s",
            options=options,
            default=default_sel,
            key="multi_compare_labels_single",
        )

    with c3:
        row_dim = st.selectbox(
            "Rows By",
            ["Retailer", "Vendor"],
            index=0,
            key="multi_compare_row_dim_single",
        )

    with c4:
        metric = st.selectbox(
            "Metric",
            ["Sales", "Units"],
            index=0,
            key="multi_compare_metric_single",
        )

    if not labels:
        st.info(f"Select one or more {granularity.lower()}s to continue.")
        return

    # keep selected order from options so trend left→right is chronological based on list ordering
    ordered_labels = [x for x in options if x in labels]

    _render_kpis(df_scope, ordered_labels, granularity, metric)
    _render_matrix_table(df_scope, ordered_labels, granularity, row_dim, metric)
    _render_trend_table(df_scope, ordered_labels, granularity, row_dim, metric)
