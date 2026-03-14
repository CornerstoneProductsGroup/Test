from __future__ import annotations

import io
import math

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from .shared_core import (
    money,
    render_df,
    available_month_labels,
    available_year_labels,
    filter_by_period_labels,
    kpi_card,
)

LINE_ACCENT = "#FF4FC3"
RADAR_FILL = "#4CC9F0"
RADAR_LINE = "#0077B6"
TEXT_LIGHT = "#DCE6F2"
TEXT_BLACK = "#111111"
TEXT_AMBER = "#FFC857"
TEXT_TEAL = "#7FDBFF"
RING_GRAY = "#8A8F98"


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
        return " ".join([bars[3]] * len(vals))

    out = []
    for v in vals:
        idx = int(round((v - vmin) / (vmax - vmin) * (len(bars) - 1)))
        idx = max(0, min(idx, len(bars) - 1))
        out.append(bars[idx])
    return " ".join(out)


def _safe_pct_change(cur: float, prev: float) -> float:
    cur = float(cur)
    prev = float(prev)
    if prev == 0:
        if cur == 0:
            return 0.0
        return np.nan
    return (cur - prev) / prev


def _table_height(n_rows: int, row_px: int = 35, header_px: int = 38, max_px: int = 1200) -> int:
    return min(max_px, header_px + max(1, n_rows) * row_px)


def _append_total_row(df: pd.DataFrame, row_dim: str, numeric_cols: list[str], trend_col: str | None = None) -> pd.DataFrame:
    total_data = {row_dim: "TOTAL"}
    for c in numeric_cols:
        total_data[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).sum()
    if trend_col and trend_col in df.columns:
        total_data[trend_col] = ""
    return pd.concat([df, pd.DataFrame([total_data])], ignore_index=True)


def _build_matrix(df: pd.DataFrame, labels: list[str], granularity: str, row_dim: str, metric: str) -> pd.DataFrame:
    pieces = []
    for lbl in labels:
        part = filter_by_period_labels(df, [lbl], granularity)
        grp = part.groupby(row_dim, as_index=False).agg(Value=(metric, "sum"))
        grp = grp.rename(columns={"Value": lbl})
        pieces.append(grp)

    if not pieces:
        return pd.DataFrame(columns=[row_dim])

    out = pieces[0].copy()
    for p in pieces[1:]:
        out = out.merge(p, on=row_dim, how="outer")

    out = out.fillna(0.0)
    period_cols = [c for c in labels if c in out.columns]

    if period_cols:
        out["Total"] = out[period_cols].sum(axis=1)
        out["Average"] = out[period_cols].mean(axis=1)
        out["Trend"] = out[period_cols].apply(lambda r: _spark(r.tolist()), axis=1)
        out["Latest"] = out[period_cols[-1]]
    else:
        out["Total"] = 0.0
        out["Average"] = 0.0
        out["Trend"] = ""
        out["Latest"] = 0.0

    return out


def _style_matrix(display_df: pd.DataFrame, numeric_df: pd.DataFrame, period_cols: list[str]):
    def style_row(row):
        idx = row.name
        if str(display_df.iloc[idx, 0]) == "TOTAL":
            return ["font-weight:800; border-top:2px solid rgba(128,128,128,0.4);" for _ in row.index]

        vals = pd.to_numeric(numeric_df.loc[idx, period_cols], errors="coerce")
        valid = vals.dropna()
        styles = [""] * len(display_df.columns)

        if valid.empty:
            return styles

        hi = valid.max()
        lo = valid.min()

        for j, col in enumerate(display_df.columns):
            if col not in period_cols:
                continue
            val = pd.to_numeric(numeric_df.loc[idx, col], errors="coerce")
            if pd.isna(val):
                continue
            if val == hi and hi != lo:
                styles[j] = "background-color: rgba(46,125,50,0.20); font-weight:700;"
            elif val == lo and hi != lo:
                styles[j] = "background-color: rgba(198,40,40,0.20); font-weight:700;"
        return styles

    return display_df.style.apply(style_row, axis=1)


def _render_base_metric_cards(df_scope: pd.DataFrame, labels: list[str], granularity: str):
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
        kpi_card("Total Sales", money(sales), "")
    with c2:
        kpi_card("Total Units", f"{units:,.0f}", "")
    with c3:
        kpi_card("ASP", money(asp), "")
    with c4:
        kpi_card("Retailers", f"{retailers:,}", "")
    with c5:
        kpi_card("Vendors", f"{vendors:,}", "")
    with c6:
        kpi_card("SKUs", f"{skus:,}", "")


def _period_entity_summary(
    df_scope: pd.DataFrame,
    labels: list[str],
    granularity: str,
    dim: str,
    metric: str,
) -> pd.DataFrame:
    rows = []
    for lbl in labels:
        part = filter_by_period_labels(df_scope, [lbl], granularity)
        if part.empty or dim not in part.columns or metric not in part.columns:
            continue
        grp = part.groupby(dim, as_index=False).agg(Value=(metric, "sum"))
        total_value = float(grp["Value"].sum()) if not grp.empty else 0.0
        if grp.empty:
            continue
        grp["Share"] = np.where(total_value != 0, grp["Value"] / total_value, 0.0)
        grp["Period"] = lbl
        rows.append(grp[[dim, "Period", "Value", "Share"]])

    if not rows:
        return pd.DataFrame(columns=[dim, "Period", "Value", "Share"])

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={dim: "Entity"})
    return out.sort_values("Value", ascending=False).reset_index(drop=True)


def _growth_entity_summary(
    df_scope: pd.DataFrame,
    labels: list[str],
    granularity: str,
    dim: str,
    metric: str,
) -> pd.DataFrame:
    rows = []
    if len(labels) < 2:
        return pd.DataFrame(columns=[dim, "Period", "Growth", "Pct"])

    matrices = {}
    for lbl in labels:
        part = filter_by_period_labels(df_scope, [lbl], granularity)
        if part.empty or dim not in part.columns or metric not in part.columns:
            matrices[lbl] = pd.DataFrame(columns=[dim, "Value"])
            continue
        grp = part.groupby(dim, as_index=False).agg(Value=(metric, "sum"))
        matrices[lbl] = grp

    for i in range(1, len(labels)):
        prev_lbl = labels[i - 1]
        cur_lbl = labels[i]
        prev_df = matrices[prev_lbl].rename(columns={"Value": "PrevValue"})
        cur_df = matrices[cur_lbl].rename(columns={"Value": "CurValue"})
        m = cur_df.merge(prev_df, on=dim, how="outer").fillna(0.0)
        if m.empty:
            continue
        m["Growth"] = m["CurValue"] - m["PrevValue"]
        m["Pct"] = [
            _safe_pct_change(cur, prev)
            for cur, prev in zip(m["CurValue"].tolist(), m["PrevValue"].tolist())
        ]
        m["Period"] = f"{cur_lbl} vs {prev_lbl}"
        rows.append(m[[dim, "Period", "Growth", "Pct"]])

    if not rows:
        return pd.DataFrame(columns=[dim, "Period", "Growth", "Pct"])

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={dim: "Entity"})
    out = out.sort_values("Growth", ascending=False).reset_index(drop=True)
    return out


def _truncate_text(x: str, max_len: int = 30) -> str:
    x = str(x)
    return x if len(x) <= max_len else x[: max_len - 1] + "…"


def _pack_period_item(df: pd.DataFrame, idx: int, metric: str):
    if len(df) <= idx:
        return None
    row = df.iloc[idx]
    return {
        "name": row["Entity"],
        "value": _fmt_value(float(row["Value"]), metric),
        "detail": f"{row['Period']} • {float(row['Share']):.1%} share",
    }


def _pack_growth_item(df: pd.DataFrame, idx: int, metric: str):
    if len(df) <= idx:
        return None
    row = df.iloc[idx]
    pct_text = "—" if pd.isna(row["Pct"]) else f"{float(row['Pct']):.1%}"
    return {
        "name": row["Entity"],
        "value": _fmt_value(float(row["Growth"]), metric),
        "detail": f"{row['Period']} • {pct_text}",
    }


def _render_item_block(rank_label: str, item: dict | None):
    st.markdown(
        f"<div style='font-size:1.02rem; font-weight:800; line-height:1.1; margin-bottom:4px;'>{rank_label}</div>",
        unsafe_allow_html=True,
    )

    if item is None:
        st.markdown(
            "<div style='font-size:1.06rem; font-weight:700; line-height:1.25; margin-bottom:10px;'>—</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"<div style='font-size:1.20rem; font-weight:800; line-height:1.18; margin-bottom:4px;'>{_truncate_text(item['name'])}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:1.14rem; font-weight:800; line-height:1.15; margin-bottom:4px;'>{item['value']}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:0.99rem; line-height:1.22; opacity:0.92; margin-bottom:10px;'>{item['detail']}</div>",
        unsafe_allow_html=True,
    )


def _render_dual_kpi_box(
    title: str,
    left_label: str,
    right_label: str,
    left_first: dict | None,
    left_second: dict | None,
    right_first: dict | None,
    right_second: dict | None,
):
    with st.container(border=True):
        st.markdown(
            f"<div style='font-size:0.96rem; font-weight:700; margin-top:2px; margin-bottom:12px;'>{title}</div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                f"<div style='font-size:0.95rem; font-weight:800; margin-bottom:8px;'>{left_label}</div>",
                unsafe_allow_html=True,
            )
            _render_item_block("#1", left_first)
            _render_item_block("#2", left_second)

        with c2:
            st.markdown(
                f"<div style='font-size:0.95rem; font-weight:800; margin-bottom:8px;'>{right_label}</div>",
                unsafe_allow_html=True,
            )
            _render_item_block("#1", right_first)
            _render_item_block("#2", right_second)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)


def _render_top2_peak_cards(df_scope: pd.DataFrame, labels: list[str], granularity: str):
    st.markdown("### Biggest by Period")

    retail_sales = _period_entity_summary(df_scope, labels, granularity, "Retailer", "Sales").head(2)
    retail_units = _period_entity_summary(df_scope, labels, granularity, "Retailer", "Units").head(2)

    vendor_sales = _period_entity_summary(df_scope, labels, granularity, "Vendor", "Sales").head(2)
    vendor_units = _period_entity_summary(df_scope, labels, granularity, "Vendor", "Units").head(2)

    sku_sales = _period_entity_summary(df_scope, labels, granularity, "SKU", "Sales").head(2)
    sku_units = _period_entity_summary(df_scope, labels, granularity, "SKU", "Units").head(2)

    c1, c2, c3 = st.columns(3)
    with c1:
        _render_dual_kpi_box(
            "Biggest Retailer",
            "Sales",
            "Units",
            _pack_period_item(retail_sales, 0, "Sales"),
            _pack_period_item(retail_sales, 1, "Sales"),
            _pack_period_item(retail_units, 0, "Units"),
            _pack_period_item(retail_units, 1, "Units"),
        )
    with c2:
        _render_dual_kpi_box(
            "Biggest Vendor",
            "Sales",
            "Units",
            _pack_period_item(vendor_sales, 0, "Sales"),
            _pack_period_item(vendor_sales, 1, "Sales"),
            _pack_period_item(vendor_units, 0, "Units"),
            _pack_period_item(vendor_units, 1, "Units"),
        )
    with c3:
        _render_dual_kpi_box(
            "Biggest SKU",
            "Sales",
            "Units",
            _pack_period_item(sku_sales, 0, "Sales"),
            _pack_period_item(sku_sales, 1, "Sales"),
            _pack_period_item(sku_units, 0, "Units"),
            _pack_period_item(sku_units, 1, "Units"),
        )


def _render_top2_growth_cards(df_scope: pd.DataFrame, labels: list[str], granularity: str):
    st.markdown("### Biggest Growth")

    retail_sales = _growth_entity_summary(df_scope, labels, granularity, "Retailer", "Sales").head(2)
    retail_units = _growth_entity_summary(df_scope, labels, granularity, "Retailer", "Units").head(2)

    vendor_sales = _growth_entity_summary(df_scope, labels, granularity, "Vendor", "Sales").head(2)
    vendor_units = _growth_entity_summary(df_scope, labels, granularity, "Vendor", "Units").head(2)

    sku_sales = _growth_entity_summary(df_scope, labels, granularity, "SKU", "Sales").head(2)
    sku_units = _growth_entity_summary(df_scope, labels, granularity, "SKU", "Units").head(2)

    c1, c2, c3 = st.columns(3)
    with c1:
        _render_dual_kpi_box(
            "Retailer Growth",
            "Sales",
            "Units",
            _pack_growth_item(retail_sales, 0, "Sales"),
            _pack_growth_item(retail_sales, 1, "Sales"),
            _pack_growth_item(retail_units, 0, "Units"),
            _pack_growth_item(retail_units, 1, "Units"),
        )
    with c2:
        _render_dual_kpi_box(
            "Vendor Growth",
            "Sales",
            "Units",
            _pack_growth_item(vendor_sales, 0, "Sales"),
            _pack_growth_item(vendor_sales, 1, "Sales"),
            _pack_growth_item(vendor_units, 0, "Units"),
            _pack_growth_item(vendor_units, 1, "Units"),
        )
    with c3:
        _render_dual_kpi_box(
            "SKU Growth",
            "Sales",
            "Units",
            _pack_growth_item(sku_sales, 0, "Sales"),
            _pack_growth_item(sku_sales, 1, "Sales"),
            _pack_growth_item(sku_units, 0, "Units"),
            _pack_growth_item(sku_units, 1, "Units"),
        )


def _render_multi_period_matrix(
    df_scope: pd.DataFrame,
    labels: list[str],
    granularity: str,
    row_dim: str,
    metric: str,
    sort_by: str,
):
    st.markdown("### Multi-Period Matrix")

    matrix = _build_matrix(df_scope, labels, granularity, row_dim, metric)
    if matrix.empty:
        st.info("No data available for the selected periods.")
        return

    period_cols = [c for c in labels if c in matrix.columns]

    if sort_by == "Latest Selected":
        matrix = matrix.sort_values("Latest", ascending=False)
    elif sort_by == "Total":
        matrix = matrix.sort_values("Total", ascending=False)
    elif sort_by == "Average":
        matrix = matrix.sort_values("Average", ascending=False)
    elif sort_by == "Alphabetical":
        matrix = matrix.sort_values(row_dim, ascending=True)

    show_numeric = matrix[[row_dim] + period_cols + ["Total", "Average", "Trend"]].copy()
    show_numeric = _append_total_row(show_numeric, row_dim, period_cols + ["Total", "Average"], trend_col="Trend")

    show = show_numeric.copy()
    for c in period_cols + ["Total", "Average"]:
        if c in show.columns:
            show[c] = show[c].map(lambda v: _fmt_value(v, metric) if pd.notna(v) else "")

    styled = _style_matrix(show, show_numeric, period_cols)

    st.dataframe(
        styled,
        use_container_width=True,
        height=_table_height(len(show), row_px=36, max_px=1350),
        hide_index=True,
    )


def _render_yoy_growth_table(
    df_scope: pd.DataFrame,
    labels: list[str],
    granularity: str,
    row_dim: str,
    metric: str,
):
    st.markdown("### Year-Over-Year Growth")

    if len(labels) < 2:
        st.info("Select at least two periods to calculate year-over-year growth.")
        return

    matrix = _build_matrix(df_scope, labels, granularity, row_dim, metric)
    if matrix.empty:
        st.info("No growth data available.")
        return

    out = matrix[[row_dim]].copy()
    delta_col_names = []
    delta_numeric = pd.DataFrame(index=matrix.index)

    for i in range(1, len(labels)):
        prev_lbl = labels[i - 1]
        cur_lbl = labels[i]
        delta_col = f"{cur_lbl} vs {prev_lbl}"
        pct_col = f"{cur_lbl} vs {prev_lbl} %"

        delta_vals = matrix[cur_lbl] - matrix[prev_lbl]
        pct_vals = [
            _safe_pct_change(cur, prev)
            for cur, prev in zip(matrix[cur_lbl].tolist(), matrix[prev_lbl].tolist())
        ]

        if metric == "Sales":
            out[delta_col] = delta_vals.map(money)
        else:
            out[delta_col] = delta_vals.map(lambda v: f"{v:,.0f}")

        out[pct_col] = ["—" if pd.isna(v) else f"{v:.1%}" for v in pct_vals]

        delta_col_names.append(delta_col)
        delta_numeric[delta_col] = delta_vals

    total_row = {row_dim: "TOTAL"}
    for col in out.columns[1:]:
        if col in delta_col_names:
            summed = pd.to_numeric(delta_numeric[col], errors="coerce").fillna(0).sum()
            total_row[col] = money(summed) if metric == "Sales" else f"{summed:,.0f}"
        else:
            total_row[col] = ""

    out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)

    def style_row(row):
        styles = [""] * len(out.columns)

        if row[row_dim] == "TOTAL":
            return ["font-weight:800; border-top:2px solid rgba(128,128,128,0.4);" for _ in row.index]

        numeric_vals = pd.to_numeric(delta_numeric.loc[row.name, delta_col_names], errors="coerce")
        valid = numeric_vals.dropna()
        if valid.empty:
            return styles

        hi = valid.max()
        lo = valid.min()

        for j, col in enumerate(out.columns):
            if col not in delta_col_names:
                continue
            val = pd.to_numeric(delta_numeric.loc[row.name, col], errors="coerce")
            if pd.isna(val):
                continue
            if val == hi and hi != lo:
                styles[j] = "background-color: rgba(46,125,50,0.18); font-weight:700;"
            elif val == lo and hi != lo:
                styles[j] = "background-color: rgba(198,40,40,0.18); font-weight:700;"
        return styles

    st.dataframe(
        out.style.apply(style_row, axis=1),
        use_container_width=True,
        height=_table_height(len(out), row_px=36, max_px=1350),
        hide_index=True,
    )


def _render_share_of_total_table(
    df_scope: pd.DataFrame,
    labels: list[str],
    granularity: str,
    row_dim: str,
    metric: str,
):
    st.markdown("### Share of Total")

    matrix = _build_matrix(df_scope, labels, granularity, row_dim, metric)
    if matrix.empty:
        st.info("No share data available.")
        return

    period_cols = [c for c in labels if c in matrix.columns]
    share = matrix[[row_dim]].copy()

    for col in period_cols:
        total = float(matrix[col].sum())
        if total == 0:
            share[col] = "0.0%"
        else:
            share[col] = (matrix[col] / total).map(lambda v: f"{v:.1%}")

    st.dataframe(
        share,
        use_container_width=True,
        height=_table_height(len(share), row_px=36, max_px=1200),
        hide_index=True,
    )


def _render_multi_year_seasonality(
    df_scope: pd.DataFrame,
    labels: list[str],
    granularity: str,
    metric: str,
):
    st.markdown("### Multi-Year Seasonality")

    if granularity != "Year":
        st.info("Multi-Year Seasonality is available when Analyze By is set to Year.")
        return

    if not labels:
        st.info("Select one or more years.")
        return

    df_sel = filter_by_period_labels(df_scope, labels, "Year")
    if df_sel.empty:
        st.info("No seasonality data available.")
        return

    d = df_sel.copy()
    date_col = "WeekEnd" if "WeekEnd" in d.columns else None
    if date_col is None:
        for c in ["Date", "Week", "Week End", "Week_End", "Week Ending", "WeekEnding"]:
            if c in d.columns:
                date_col = c
                break

    if date_col is None:
        st.info("No date column available for seasonality.")
        return

    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d[d[date_col].notna()].copy()
    d["Year"] = d[date_col].dt.year.astype(str)
    d["Quarter"] = d[date_col].dt.quarter.map(lambda q: f"Q{int(q)}")

    q = d.groupby(["Quarter", "Year"], as_index=False).agg(Value=(metric, "sum"))

    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    q["Quarter"] = pd.Categorical(q["Quarter"], categories=quarter_order, ordered=True)

    piv = q.pivot_table(
        index="Quarter",
        columns="Year",
        values="Value",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()

    ordered_years = []
    for lbl in labels:
        y = str(lbl)
        if y in piv.columns:
            ordered_years.append(y)

    show = piv[["Quarter"] + ordered_years].copy()
    for c in ordered_years:
        show[c] = show[c].map(lambda v: _fmt_value(v, metric))

    render_df(show, height=300)


def _render_performance_score(
    df_scope: pd.DataFrame,
    labels: list[str],
    granularity: str,
    row_dim: str,
    metric: str,
):
    st.markdown("### Retailer Performance Score" if row_dim == "Retailer" else "### Performance Score")

    if len(labels) < 2:
        st.info("Select at least two periods to calculate performance score.")
        return

    matrix = _build_matrix(df_scope, labels, granularity, row_dim, metric)
    if matrix.empty:
        st.info("No performance score data available.")
        return

    period_cols = [c for c in labels if c in matrix.columns]
    latest_col = period_cols[-1]
    first_col = period_cols[0]

    total_max = float(matrix["Total"].max()) if not matrix.empty else 0.0

    scores = []
    for _, row in matrix.iterrows():
        first_val = float(row[first_col])
        latest_val = float(row[latest_col])
        total_val = float(row["Total"])
        vals = [float(row[c]) for c in period_cols]

        growth_pct = _safe_pct_change(latest_val, first_val)
        growth_score = 0.0 if pd.isna(growth_pct) else max(0.0, min(100.0, 50.0 + (growth_pct * 100.0)))
        total_score = 0.0 if total_max == 0 else (total_val / total_max) * 100.0

        diffs = np.diff(vals) if len(vals) >= 2 else np.array([0.0])
        pos_moves = float((diffs > 0).sum())
        momentum_score = 0.0 if len(diffs) == 0 else (pos_moves / len(diffs)) * 100.0

        mean_val = np.mean(vals) if len(vals) else 0.0
        std_val = np.std(vals) if len(vals) else 0.0
        cv = 0.0 if mean_val == 0 else std_val / mean_val
        stability_score = max(0.0, 100.0 - (cv * 100.0))

        score = (
            0.40 * growth_score +
            0.30 * total_score +
            0.20 * momentum_score +
            0.10 * stability_score
        )

        if len(vals) >= 2:
            if vals[-1] > vals[-2]:
                momentum_label = "Rising"
            elif vals[-1] < vals[-2]:
                momentum_label = "Falling"
            else:
                momentum_label = "Flat"
        else:
            momentum_label = "Flat"

        scores.append(
            {
                row_dim: row[row_dim],
                "Score": score,
                "Growth": "—" if pd.isna(growth_pct) else f"{growth_pct:.1%}",
                "Total": total_val,
                "Momentum": momentum_label,
                "Trend": row["Trend"],
            }
        )

    out = pd.DataFrame(scores).sort_values("Score", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))
    out["Score"] = out["Score"].map(lambda v: f"{v:.0f}")
    out["Total"] = out["Total"].map(lambda v: _fmt_value(v, metric))

    render_df(out, height=500)


RADAR_MONTH_ORDER = [
    ("Q1", "January", 1),
    ("Q1", "February", 2),
    ("Q1", "March", 3),
    ("Q2", "April", 4),
    ("Q2", "May", 5),
    ("Q2", "June", 6),
    ("Q3", "July", 7),
    ("Q3", "August", 8),
    ("Q3", "September", 9),
    ("Q4", "October", 10),
    ("Q4", "November", 11),
    ("Q4", "December", 12),
]


def _find_date_column(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    for c in ["WeekEnd", "Date", "Week", "Week End", "Week_End", "Week Ending", "WeekEnding"]:
        if c in df.columns:
            return c
    return None


def _prepare_visual_base(df_scope: pd.DataFrame, labels: list[str], granularity: str) -> pd.DataFrame:
    out = []
    for lbl in labels:
        part = filter_by_period_labels(df_scope, [lbl], granularity).copy()
        if part.empty:
            continue
        part["PeriodLabel"] = str(lbl)
        out.append(part)

    if not out:
        return pd.DataFrame()

    df = pd.concat(out, ignore_index=True)

    date_col = _find_date_column(df)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["MonthNum"] = df[date_col].dt.month
        df["QuarterNum"] = df[date_col].dt.quarter
        df["Quarter"] = df["QuarterNum"].map(lambda q: f"Q{int(q)}" if pd.notna(q) else None)
    else:
        if "Quarter" not in df.columns:
            df["Quarter"] = None
        if "MonthNum" not in df.columns:
            df["MonthNum"] = np.nan

    return df


def _period_summary_df(df_vis: pd.DataFrame) -> pd.DataFrame:
    if df_vis.empty:
        return pd.DataFrame(
            columns=[
                "PeriodLabel",
                "Sales",
                "Units",
                "ASP",
                "AvgUnitsPerSKU",
                "AvgSalesPerSKU",
                "SKUs",
                "SortOrder",
            ]
        )

    grp = (
        df_vis.groupby("PeriodLabel", as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Units=("Units", "sum"),
            SKUs=("SKU", pd.Series.nunique),
        )
        .copy()
    )

    grp["SKUs"] = pd.to_numeric(grp["SKUs"], errors="coerce").fillna(0.0)
    grp["ASP"] = np.where(grp["Units"] != 0, grp["Sales"] / grp["Units"], 0.0)
    grp["AvgUnitsPerSKU"] = np.where(grp["SKUs"] != 0, grp["Units"] / grp["SKUs"], 0.0)
    grp["AvgSalesPerSKU"] = np.where(grp["SKUs"] != 0, grp["Sales"] / grp["SKUs"], 0.0)

    order_map = {lbl: i for i, lbl in enumerate(df_vis["PeriodLabel"].drop_duplicates().tolist())}
    grp["SortOrder"] = grp["PeriodLabel"].map(order_map)
    grp = grp.sort_values(["SortOrder", "PeriodLabel"]).reset_index(drop=True)
    return grp


def _render_sales_asp_combo_chart(summary_df: pd.DataFrame):
    if summary_df.empty:
        st.info("No data available.")
        return

    work = summary_df[["PeriodLabel", "Sales", "ASP"]].copy()
    order = work["PeriodLabel"].tolist()

    max_sales = float(work["Sales"].max()) if not work.empty else 0.0
    sales_domain_max = max(max_sales * 1.22, 1.0)

    band_low = sales_domain_max * 0.80
    band_high = sales_domain_max * 0.96

    asp_min = float(work["ASP"].min()) if not work.empty else 0.0
    asp_max = float(work["ASP"].max()) if not work.empty else 0.0
    if math.isclose(asp_min, asp_max):
        work["ASPDisplayY"] = (band_low + band_high) / 2.0
    else:
        work["ASPDisplayY"] = band_low + ((work["ASP"] - asp_min) / (asp_max - asp_min)) * (band_high - band_low)

    work["SalesLabel"] = work["Sales"].map(lambda v: money(float(v)))
    work["ASPLabel"] = work["ASP"].map(lambda v: money(float(v)))
    work["ASPLabelY"] = work["ASPDisplayY"] + (sales_domain_max * 0.04)
    work["SalesLabelY"] = work["Sales"] * 0.90

    bars = (
        alt.Chart(work)
        .mark_bar()
        .encode(
            x=alt.X("PeriodLabel:N", title="Period", sort=order),
            y=alt.Y("Sales:Q", title="Sales", scale=alt.Scale(domain=[0, sales_domain_max])),
            tooltip=[
                alt.Tooltip("PeriodLabel:N", title="Period"),
                alt.Tooltip("Sales:Q", title="Sales", format=",.2f"),
                alt.Tooltip("ASP:Q", title="ASP", format=",.2f"),
            ],
        )
    )

    sales_text = (
        alt.Chart(work)
        .mark_text(color=TEXT_BLACK, fontWeight="bold", fontSize=15)
        .encode(
            x=alt.X("PeriodLabel:N", sort=order),
            y=alt.Y("SalesLabelY:Q", scale=alt.Scale(domain=[0, sales_domain_max])),
            text="SalesLabel:N",
        )
    )

    line = (
        alt.Chart(work)
        .mark_line(
            point=alt.OverlayMarkDef(color=LINE_ACCENT, filled=True, size=80),
            strokeWidth=3,
            color=LINE_ACCENT,
        )
        .encode(
            x=alt.X("PeriodLabel:N", sort=order),
            y=alt.Y("ASPDisplayY:Q", title="Sales", scale=alt.Scale(domain=[0, sales_domain_max])),
            tooltip=[
                alt.Tooltip("PeriodLabel:N", title="Period"),
                alt.Tooltip("ASP:Q", title="ASP", format=",.2f"),
            ],
        )
    )

    asp_text = (
        alt.Chart(work)
        .mark_text(color=LINE_ACCENT, fontWeight="bold", fontSize=13)
        .encode(
            x=alt.X("PeriodLabel:N", sort=order),
            y=alt.Y("ASPLabelY:Q", scale=alt.Scale(domain=[0, sales_domain_max])),
            text="ASPLabel:N",
        )
    )

    chart = (
        alt.layer(bars, sales_text, line, asp_text)
        .properties(
            height=500,
            title="Sales and ASP by Selected Period",
        )
        .configure_title(
            anchor="start",
            fontSize=16,
            offset=12,
            color=TEXT_LIGHT,
        )
    )

    st.altair_chart(chart, use_container_width=True)


def _render_sales_units_combo_chart(summary_df: pd.DataFrame):
    if summary_df.empty:
        st.info("No data available.")
        return

    work = summary_df[["PeriodLabel", "Sales", "Units"]].copy()
    order = work["PeriodLabel"].tolist()

    max_sales = float(work["Sales"].max()) if not work.empty else 0.0
    sales_domain_max = max(max_sales * 1.22, 1.0)

    band_low = sales_domain_max * 0.80
    band_high = sales_domain_max * 0.96

    units_min = float(work["Units"].min()) if not work.empty else 0.0
    units_max = float(work["Units"].max()) if not work.empty else 0.0
    if math.isclose(units_min, units_max):
        work["UnitsDisplayY"] = (band_low + band_high) / 2.0
    else:
        work["UnitsDisplayY"] = band_low + ((work["Units"] - units_min) / (units_max - units_min)) * (band_high - band_low)

    work["SalesLabel"] = work["Sales"].map(lambda v: money(float(v)))
    work["UnitsLabel"] = work["Units"].map(lambda v: f"{float(v):,.0f}")
    work["UnitsLabelY"] = work["UnitsDisplayY"] + (sales_domain_max * 0.04)
    work["SalesLabelY"] = work["Sales"] * 0.90

    bars = (
        alt.Chart(work)
        .mark_bar()
        .encode(
            x=alt.X("PeriodLabel:N", title="Period", sort=order),
            y=alt.Y("Sales:Q", title="Sales", scale=alt.Scale(domain=[0, sales_domain_max])),
            tooltip=[
                alt.Tooltip("PeriodLabel:N", title="Period"),
                alt.Tooltip("Sales:Q", title="Sales", format=",.2f"),
                alt.Tooltip("Units:Q", title="Units", format=",.0f"),
            ],
        )
    )

    sales_text = (
        alt.Chart(work)
        .mark_text(color=TEXT_BLACK, fontWeight="bold", fontSize=15)
        .encode(
            x=alt.X("PeriodLabel:N", sort=order),
            y=alt.Y("SalesLabelY:Q", scale=alt.Scale(domain=[0, sales_domain_max])),
            text="SalesLabel:N",
        )
    )

    line = (
        alt.Chart(work)
        .mark_line(
            point=alt.OverlayMarkDef(color=LINE_ACCENT, filled=True, size=80),
            strokeWidth=3,
            color=LINE_ACCENT,
        )
        .encode(
            x=alt.X("PeriodLabel:N", sort=order),
            y=alt.Y("UnitsDisplayY:Q", title="Sales", scale=alt.Scale(domain=[0, sales_domain_max])),
            tooltip=[
                alt.Tooltip("PeriodLabel:N", title="Period"),
                alt.Tooltip("Units:Q", title="Units", format=",.0f"),
            ],
        )
    )

    units_text = (
        alt.Chart(work)
        .mark_text(color=LINE_ACCENT, fontWeight="bold", fontSize=13)
        .encode(
            x=alt.X("PeriodLabel:N", sort=order),
            y=alt.Y("UnitsLabelY:Q", scale=alt.Scale(domain=[0, sales_domain_max])),
            text="UnitsLabel:N",
        )
    )

    chart = (
        alt.layer(bars, sales_text, line, units_text)
        .properties(
            height=500,
            title="Sales and Units by Selected Period",
        )
        .configure_title(
            anchor="start",
            fontSize=16,
            offset=12,
            color=TEXT_LIGHT,
        )
    )

    st.altair_chart(chart, use_container_width=True)


def _render_avg_sales_units_per_sku_combo_chart(summary_df: pd.DataFrame):
    if summary_df.empty:
        st.info("No data available.")
        return

    work = summary_df[["PeriodLabel", "AvgSalesPerSKU", "AvgUnitsPerSKU"]].copy()
    order = work["PeriodLabel"].tolist()

    max_sales = float(work["AvgSalesPerSKU"].max()) if not work.empty else 0.0
    sales_domain_max = max(max_sales * 1.22, 1.0)

    band_low = sales_domain_max * 0.80
    band_high = sales_domain_max * 0.96

    units_min = float(work["AvgUnitsPerSKU"].min()) if not work.empty else 0.0
    units_max = float(work["AvgUnitsPerSKU"].max()) if not work.empty else 0.0
    if math.isclose(units_min, units_max):
        work["UnitsDisplayY"] = (band_low + band_high) / 2.0
    else:
        work["UnitsDisplayY"] = band_low + (
            (work["AvgUnitsPerSKU"] - units_min) / (units_max - units_min)
        ) * (band_high - band_low)

    work["SalesLabel"] = work["AvgSalesPerSKU"].map(lambda v: money(float(v)))
    work["UnitsLabel"] = work["AvgUnitsPerSKU"].map(lambda v: f"{float(v):,.2f}")
    work["UnitsLabelY"] = work["UnitsDisplayY"] + (sales_domain_max * 0.04)
    work["SalesLabelY"] = work["AvgSalesPerSKU"] * 0.90

    bars = (
        alt.Chart(work)
        .mark_bar()
        .encode(
            x=alt.X("PeriodLabel:N", title="Period", sort=order),
            y=alt.Y(
                "AvgSalesPerSKU:Q",
                title="Average Sales per SKU",
                scale=alt.Scale(domain=[0, sales_domain_max]),
            ),
            tooltip=[
                alt.Tooltip("PeriodLabel:N", title="Period"),
                alt.Tooltip("AvgSalesPerSKU:Q", title="Average Sales per SKU", format=",.2f"),
                alt.Tooltip("AvgUnitsPerSKU:Q", title="Average Units per SKU", format=",.2f"),
            ],
        )
    )

    sales_text = (
        alt.Chart(work)
        .mark_text(color=TEXT_BLACK, fontWeight="bold", fontSize=15)
        .encode(
            x=alt.X("PeriodLabel:N", sort=order),
            y=alt.Y("SalesLabelY:Q", scale=alt.Scale(domain=[0, sales_domain_max])),
            text="SalesLabel:N",
        )
    )

    line = (
        alt.Chart(work)
        .mark_line(
            point=alt.OverlayMarkDef(color=LINE_ACCENT, filled=True, size=80),
            strokeWidth=3,
            color=LINE_ACCENT,
        )
        .encode(
            x=alt.X("PeriodLabel:N", sort=order),
            y=alt.Y(
                "UnitsDisplayY:Q",
                title="Average Sales per SKU",
                scale=alt.Scale(domain=[0, sales_domain_max]),
            ),
            tooltip=[
                alt.Tooltip("PeriodLabel:N", title="Period"),
                alt.Tooltip("AvgUnitsPerSKU:Q", title="Average Units per SKU", format=",.2f"),
            ],
        )
    )

    units_text = (
        alt.Chart(work)
        .mark_text(color=LINE_ACCENT, fontWeight="bold", fontSize=13)
        .encode(
            x=alt.X("PeriodLabel:N", sort=order),
            y=alt.Y("UnitsLabelY:Q", scale=alt.Scale(domain=[0, sales_domain_max])),
            text="UnitsLabel:N",
        )
    )

    chart = (
        alt.layer(bars, sales_text, line, units_text)
        .properties(
            height=500,
            title="Average Sales per SKU and Average Units per SKU",
        )
        .configure_title(
            anchor="start",
            fontSize=16,
            offset=12,
            color=TEXT_LIGHT,
        )
    )

    st.altair_chart(chart, use_container_width=True)


def _quarterly_stacked_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns or "Quarter" not in df.columns:
        return pd.DataFrame(columns=["PeriodLabel", "Quarter", "Value", "ValueLabel", "PeriodOrder", "Start", "LabelY"])

    out = (
        df.dropna(subset=["Quarter"])
        .groupby(["PeriodLabel", "Quarter"], as_index=False)[metric]
        .sum()
        .rename(columns={metric: "Value"})
    )

    q_order = ["Q1", "Q2", "Q3", "Q4"]
    out["Quarter"] = pd.Categorical(out["Quarter"], categories=q_order, ordered=True)
    p_order = {lbl: i for i, lbl in enumerate(df["PeriodLabel"].drop_duplicates().tolist())}
    out["PeriodOrder"] = out["PeriodLabel"].map(p_order)
    out = out.sort_values(["PeriodOrder", "Quarter"]).reset_index(drop=True)

    if metric == "Sales":
        out["ValueLabel"] = out["Value"].map(lambda v: money(float(v)))
    else:
        out["ValueLabel"] = out["Value"].map(lambda v: f"{float(v):,.0f}")

    out["Start"] = out.groupby("PeriodLabel")["Value"].cumsum() - out["Value"]
    out["LabelY"] = out["Start"] + (out["Value"] * 0.18)

    return out


def _render_quarterly_stacked_altair(df: pd.DataFrame, metric: str):
    if df.empty:
        st.info("No quarterly stacked data available.")
        return

    order = df["PeriodLabel"].drop_duplicates().tolist()

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("PeriodLabel:N", title="Period", sort=order),
            y=alt.Y("Value:Q", title=metric, stack="zero"),
            color=alt.Color("Quarter:N", title="", sort=["Q1", "Q2", "Q3", "Q4"]),
            order=alt.Order("Quarter:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("PeriodLabel:N", title="Period"),
                alt.Tooltip("Quarter:N", title="Quarter"),
                alt.Tooltip("Value:Q", title=metric, format=",.2f" if metric == "Sales" else ",.0f"),
            ],
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(size=10, color=TEXT_BLACK, fontWeight="bold")
        .encode(
            x=alt.X("PeriodLabel:N", sort=order),
            y=alt.Y("LabelY:Q", title=metric, stack=None),
            detail="Quarter:N",
            text="ValueLabel:N",
        )
    )

    chart = (bars + text).properties(
        height=440,
        title=f"{metric} by Quarter, stacked within each selected year",
    ).configure_title(
        anchor="start",
        fontSize=15,
        offset=12,
        color=TEXT_LIGHT,
    )

    st.altair_chart(chart, use_container_width=True)


def _make_single_metric_bar_figure(
    summary_df: pd.DataFrame,
    value_col: str,
    title: str,
    label_mode: str = "money",
):
    fig, ax = plt.subplots(figsize=(10.0, 4.8))

    if summary_df.empty or value_col not in summary_df.columns:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        return fig

    labels = summary_df["PeriodLabel"].astype(str).tolist()
    vals = pd.to_numeric(summary_df[value_col], errors="coerce").fillna(0.0).to_numpy()

    x = np.arange(len(labels))
    bars = ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if label_mode == "money":
        txt = [money(float(v)) for v in vals]
    elif label_mode == "int":
        txt = [f"{float(v):,.0f}" for v in vals]
    else:
        txt = [f"{float(v):,.2f}" for v in vals]

    for rect, label in zip(bars, txt):
        height = rect.get_height()
        if height <= 0:
            continue
        ax.annotate(
            label,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    return fig


def _make_sales_asp_combo_figure(summary_df: pd.DataFrame, title: str = "Sales and ASP by Selected Period"):
    fig, ax1 = plt.subplots(figsize=(10.4, 5.2))

    if summary_df.empty:
        ax1.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax1.axis("off")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        return fig

    labels = summary_df["PeriodLabel"].astype(str).tolist()
    sales = pd.to_numeric(summary_df["Sales"], errors="coerce").fillna(0.0).to_numpy()
    asp = pd.to_numeric(summary_df["ASP"], errors="coerce").fillna(0.0).to_numpy()

    x = np.arange(len(labels))
    bars = ax1.bar(x, sales, width=0.55, label="Sales")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Sales")
    ax1.set_title(title, fontsize=14, fontweight="bold")

    max_sales = float(np.max(sales)) if len(sales) else 0.0
    sales_domain_max = max(max_sales * 1.22, 1.0)
    ax1.set_ylim(0, sales_domain_max)

    for rect, val in zip(bars, sales):
        if val <= 0:
            continue
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() * 0.90,
            money(float(val)),
            ha="center",
            va="top",
            fontsize=10,
            color="black",
            fontweight="bold",
        )

    band_low = sales_domain_max * 0.80
    band_high = sales_domain_max * 0.96
    asp_min = float(np.min(asp)) if len(asp) else 0.0
    asp_max = float(np.max(asp)) if len(asp) else 0.0

    if math.isclose(asp_min, asp_max):
        asp_display = np.full(len(asp), (band_low + band_high) / 2.0)
    else:
        asp_display = band_low + ((asp - asp_min) / (asp_max - asp_min)) * (band_high - band_low)

    ax2 = ax1.twinx()
    ax2.set_ylim(0, sales_domain_max)
    ax2.plot(x, asp_display, marker="o", linewidth=2.5, color=LINE_ACCENT, label="ASP")
    ax2.set_yticks([])

    for xi, disp_y, raw_val in zip(x, asp_display, asp):
        ax2.text(
            xi,
            disp_y + (sales_domain_max * 0.04),
            money(float(raw_val)),
            ha="center",
            va="bottom",
            fontsize=9,
            color=LINE_ACCENT,
            fontweight="bold",
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.84, 0.92), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    return fig


def _make_sales_units_combo_figure(summary_df: pd.DataFrame, title: str = "Sales and Units by Selected Period"):
    fig, ax1 = plt.subplots(figsize=(10.4, 5.2))

    if summary_df.empty:
        ax1.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax1.axis("off")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        return fig

    labels = summary_df["PeriodLabel"].astype(str).tolist()
    sales = pd.to_numeric(summary_df["Sales"], errors="coerce").fillna(0.0).to_numpy()
    units = pd.to_numeric(summary_df["Units"], errors="coerce").fillna(0.0).to_numpy()

    x = np.arange(len(labels))
    bars = ax1.bar(x, sales, width=0.55, label="Sales")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Sales")
    ax1.set_title(title, fontsize=14, fontweight="bold")

    max_sales = float(np.max(sales)) if len(sales) else 0.0
    sales_domain_max = max(max_sales * 1.22, 1.0)
    ax1.set_ylim(0, sales_domain_max)

    for rect, val in zip(bars, sales):
        if val <= 0:
            continue
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() * 0.90,
            money(float(val)),
            ha="center",
            va="top",
            fontsize=10,
            color="black",
            fontweight="bold",
        )

    band_low = sales_domain_max * 0.80
    band_high = sales_domain_max * 0.96
    units_min = float(np.min(units)) if len(units) else 0.0
    units_max = float(np.max(units)) if len(units) else 0.0

    if math.isclose(units_min, units_max):
        units_display = np.full(len(units), (band_low + band_high) / 2.0)
    else:
        units_display = band_low + ((units - units_min) / (units_max - units_min)) * (band_high - band_low)

    ax2 = ax1.twinx()
    ax2.set_ylim(0, sales_domain_max)
    ax2.plot(x, units_display, marker="o", linewidth=2.5, color=LINE_ACCENT, label="Units")
    ax2.set_yticks([])

    for xi, disp_y, raw_val in zip(x, units_display, units):
        ax2.text(
            xi,
            disp_y + (sales_domain_max * 0.04),
            f"{float(raw_val):,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=LINE_ACCENT,
            fontweight="bold",
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.84, 0.92), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    return fig


def _make_avg_sales_units_per_sku_combo_figure(
    summary_df: pd.DataFrame,
    title: str = "Average Sales per SKU and Average Units per SKU",
):
    fig, ax1 = plt.subplots(figsize=(10.4, 5.2))

    if summary_df.empty:
        ax1.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax1.axis("off")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        return fig

    labels = summary_df["PeriodLabel"].astype(str).tolist()
    avg_sales = pd.to_numeric(summary_df["AvgSalesPerSKU"], errors="coerce").fillna(0.0).to_numpy()
    avg_units = pd.to_numeric(summary_df["AvgUnitsPerSKU"], errors="coerce").fillna(0.0).to_numpy()

    x = np.arange(len(labels))
    bars = ax1.bar(x, avg_sales, width=0.55, label="Average Sales per SKU")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Average Sales per SKU")
    ax1.set_title(title, fontsize=14, fontweight="bold")

    max_sales = float(np.max(avg_sales)) if len(avg_sales) else 0.0
    sales_domain_max = max(max_sales * 1.22, 1.0)
    ax1.set_ylim(0, sales_domain_max)

    for rect, val in zip(bars, avg_sales):
        if val <= 0:
            continue
        ax1.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() * 0.90,
            money(float(val)),
            ha="center",
            va="top",
            fontsize=10,
            color="black",
            fontweight="bold",
        )

    band_low = sales_domain_max * 0.80
    band_high = sales_domain_max * 0.96
    avg_units_min = float(np.min(avg_units)) if len(avg_units) else 0.0
    avg_units_max = float(np.max(avg_units)) if len(avg_units) else 0.0

    if math.isclose(avg_units_min, avg_units_max):
        avg_units_display = np.full(len(avg_units), (band_low + band_high) / 2.0)
    else:
        avg_units_display = band_low + (
            (avg_units - avg_units_min) / (avg_units_max - avg_units_min)
        ) * (band_high - band_low)

    ax2 = ax1.twinx()
    ax2.set_ylim(0, sales_domain_max)
    ax2.plot(x, avg_units_display, marker="o", linewidth=2.5, color=LINE_ACCENT, label="Average Units per SKU")
    ax2.set_yticks([])

    for xi, disp_y, raw_val in zip(x, avg_units_display, avg_units):
        ax2.text(
            xi,
            disp_y + (sales_domain_max * 0.04),
            f"{float(raw_val):,.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=LINE_ACCENT,
            fontweight="bold",
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.74, 0.92), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    return fig


def _make_quarterly_stacked_figure(df: pd.DataFrame, metric: str):
    title = f"{metric} by Quarter, stacked within each selected year"
    fig, ax = plt.subplots(figsize=(9.8, 5.4))

    if df.empty:
        ax.text(0.5, 0.5, "No quarterly stacked data available", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        return fig

    work = df.copy()
    work["Value"] = pd.to_numeric(work["Value"], errors="coerce").fillna(0.0)

    periods = list(work["PeriodLabel"].drop_duplicates())
    quarters = ["Q1", "Q2", "Q3", "Q4"]

    pivot = (
        work.pivot_table(index="PeriodLabel", columns="Quarter", values="Value", aggfunc="sum", fill_value=0.0)
        .reindex(index=periods, columns=quarters, fill_value=0.0)
    )

    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index))

    for q in quarters:
        vals = pivot[q].to_numpy(dtype=float)
        bars = ax.bar(x, vals, bottom=bottom, label=q)

        for rect, v, b in zip(bars, vals, bottom):
            if v <= 0:
                continue
            label = money(float(v)) if metric == "Sales" else f"{float(v):,.0f}"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                b + (v * 0.18),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )

        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=0)
    ax.set_xlabel("Period")
    ax.set_ylabel(metric)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        title="",
    )

    fig.tight_layout(rect=[0, 0, 0.84, 1])
    return fig


def _top2_per_period(df: pd.DataFrame, dim: str, metric: str = "Sales") -> pd.DataFrame:
    if df.empty or dim not in df.columns or metric not in df.columns:
        return pd.DataFrame(columns=["PeriodLabel", "Entity", "Value", "Rank", "YLabel"])

    grp = (
        df.groupby(["PeriodLabel", dim], as_index=False)[metric]
        .sum()
        .rename(columns={dim: "Entity", metric: "Value"})
    )
    grp["Rank"] = grp.groupby("PeriodLabel")["Value"].rank(method="first", ascending=False)

    out = grp[grp["Rank"] <= 2].copy()
    out["Rank"] = out["Rank"].astype(int)

    p_order = {lbl: i for i, lbl in enumerate(df["PeriodLabel"].drop_duplicates().tolist())}
    out["PeriodOrder"] = out["PeriodLabel"].map(p_order)

    out = out.sort_values(["PeriodOrder", "Rank", "Value"], ascending=[True, True, False]).reset_index(drop=True)
    out["YLabel"] = out["PeriodLabel"].astype(str) + " • #" + out["Rank"].astype(str) + " • " + out["Entity"].astype(str)
    out["ValueLabel"] = out["Value"].map(money)
    return out


def _render_lollipop(df: pd.DataFrame, title: str):
    if df.empty:
        st.info(f"No data available for {title.lower()}.")
        return

    xmax = float(df["Value"].max()) if not df.empty else 0.0
    xmax = xmax * 1.20 if xmax > 0 else 1.0
    df = df.copy()
    df["Zero"] = 0.0

    rules = (
        alt.Chart(df)
        .mark_rule(strokeWidth=2.5)
        .encode(
            y=alt.Y("YLabel:N", sort=None, title=""),
            x=alt.X("Zero:Q", scale=alt.Scale(domain=[0, xmax]), title="Sales"),
            x2="Value:Q",
            color=alt.Color("PeriodLabel:N", title=""),
            tooltip=[
                alt.Tooltip("PeriodLabel:N", title="Period"),
                alt.Tooltip("Entity:N", title="Name"),
                alt.Tooltip("Rank:O", title="Rank"),
                alt.Tooltip("Value:Q", title="Sales", format=",.2f"),
            ],
        )
    )

    dots = (
        alt.Chart(df)
        .mark_circle(size=140)
        .encode(
            y=alt.Y("YLabel:N", sort=None, title=""),
            x=alt.X("Value:Q", scale=alt.Scale(domain=[0, xmax]), title="Sales"),
            color=alt.Color("PeriodLabel:N", title=""),
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(align="left", dx=8, size=11)
        .encode(
            y=alt.Y("YLabel:N", sort=None, title=""),
            x=alt.X("Value:Q", scale=alt.Scale(domain=[0, xmax]), title="Sales"),
            text="ValueLabel:N",
            color=alt.Color("PeriodLabel:N", legend=None),
        )
    )

    st.altair_chart(
        (rules + dots + text).properties(height=max(260, len(df) * 34), title=title),
        use_container_width=True,
    )


def _all_years_radar_month_df(df_hist: pd.DataFrame) -> pd.DataFrame:
    if df_hist is None or df_hist.empty or "Sales" not in df_hist.columns:
        return pd.DataFrame()

    work = df_hist.copy()

    if "MonthNum" not in work.columns:
        date_col = _find_date_column(work)
        if date_col is None:
            return pd.DataFrame()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col]).copy()
        work["MonthNum"] = work[date_col].dt.month

    work = work.dropna(subset=["MonthNum"]).copy()
    work["MonthNum"] = work["MonthNum"].astype(int)

    out = (
        work.groupby("MonthNum", as_index=False)["Sales"]
        .sum()
        .rename(columns={"Sales": "TotalSales"})
    )

    order_df = pd.DataFrame(RADAR_MONTH_ORDER, columns=["Quarter", "Month", "MonthNum"])
    out = order_df.merge(out, on="MonthNum", how="left").fillna({"TotalSales": 0.0})

    max_sales = float(out["TotalSales"].max()) if not out.empty else 0.0
    out["ScaledSales"] = out["TotalSales"] / max_sales if max_sales > 0 else 0.0

    out["ScaledSalesOuter"] = 0.18 + (out["ScaledSales"] * 1.00)
    out["ScaledSalesOuter"] = out["ScaledSalesOuter"].clip(upper=1.18)

    out["StartDeg"] = (out["MonthNum"] - 1) * 30
    out["EndDeg"] = out["MonthNum"] * 30
    out["MidDeg"] = out["StartDeg"] + 15

    out["MonthLabelR"] = 1.30
    out["QuarterLabelR"] = 1.44
    out["ValueLabelR"] = np.where(
        out["ScaledSalesOuter"] > 0.22,
        out["ScaledSalesOuter"] * 0.62,
        0.18,
    )

    out["SalesLabel"] = out["TotalSales"].map(lambda v: money(float(v)))

    return out


def _render_radar_altair(df: pd.DataFrame):
    if df.empty:
        st.info("No radar data available.")
        return

    work = df.copy()
    work["RadarValue"] = work["ScaledSalesOuter"] if "ScaledSalesOuter" in work.columns else work["ScaledSales"]
    work["PointOrder"] = range(len(work))

    radius_scale = alt.Scale(domain=[0, 1.35], rangeMin=0, rangeMax=320)

    rings = pd.DataFrame({"r": [0.30, 0.60, 0.90, 1.20]})

    ring_chart = (
        alt.Chart(rings)
        .mark_arc(fillOpacity=0, stroke=RING_GRAY, strokeOpacity=0.60, strokeWidth=1.1)
        .encode(
            theta=alt.Theta(value=360),
            radius=alt.Radius("r:Q", scale=radius_scale),
        )
    )

    spoke_rows = []
    for _, row in work.iterrows():
        spoke_rows.append(
            {
                "Month": row["Month"],
                "SpokeID": row["Month"],
                "Deg": row["MidDeg"],
                "Radius": 0.0,
                "PointOrder": 0,
            }
        )
        spoke_rows.append(
            {
                "Month": row["Month"],
                "SpokeID": row["Month"],
                "Deg": row["MidDeg"],
                "Radius": 1.20,
                "PointOrder": 1,
            }
        )
    spokes = pd.DataFrame(spoke_rows)

    spoke_chart = (
        alt.Chart(spokes)
        .mark_line(stroke=RING_GRAY, strokeOpacity=0.55, strokeWidth=1.0)
        .encode(
            theta=alt.Theta("Deg:Q", scale=alt.Scale(domain=[0, 360])),
            radius=alt.Radius("Radius:Q", scale=radius_scale),
            detail="SpokeID:N",
            order=alt.Order("PointOrder:Q", sort="ascending"),
        )
    )

    radar_fill = (
        alt.Chart(work)
        .mark_area(
            interpolate="linear-closed",
            color=RADAR_FILL,
            opacity=0.28,
            line=False,
        )
        .encode(
            theta=alt.Theta("MidDeg:Q", scale=alt.Scale(domain=[0, 360])),
            radius=alt.Radius("RadarValue:Q", scale=radius_scale),
            order=alt.Order("PointOrder:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("Quarter:N", title="Quarter"),
                alt.Tooltip("Month:N", title="Month"),
                alt.Tooltip("TotalSales:Q", title="Sales", format=",.2f"),
            ],
        )
    )

    radar_outline = (
        alt.Chart(work)
        .mark_line(
            interpolate="linear-closed",
            stroke=RADAR_LINE,
            strokeWidth=2.5,
            point=alt.OverlayMarkDef(filled=True, size=85, color=LINE_ACCENT),
        )
        .encode(
            theta=alt.Theta("MidDeg:Q", scale=alt.Scale(domain=[0, 360])),
            radius=alt.Radius("RadarValue:Q", scale=radius_scale),
            order=alt.Order("PointOrder:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("Quarter:N", title="Quarter"),
                alt.Tooltip("Month:N", title="Month"),
                alt.Tooltip("TotalSales:Q", title="Sales", format=",.2f"),
            ],
        )
    )

    quarter_labels = pd.DataFrame(
        {
            "Quarter": ["Q1", "Q2", "Q3", "Q4"],
            "Deg": [45, 135, 225, 315],
            "r": [1.30, 1.30, 1.30, 1.30],
        }
    )

    month_names = (
        alt.Chart(work)
        .mark_text(fontSize=14, fontWeight="bold", color=TEXT_TEAL)
        .encode(
            theta=alt.Theta("MidDeg:Q", scale=alt.Scale(domain=[0, 360])),
            radius=alt.Radius("MonthLabelR:Q", scale=radius_scale),
            text="Month:N",
        )
    )

    quarter_names = (
        alt.Chart(quarter_labels)
        .mark_text(fontSize=15, fontWeight="bold", color=TEXT_AMBER)
        .encode(
            theta=alt.Theta("Deg:Q", scale=alt.Scale(domain=[0, 360])),
            radius=alt.Radius("r:Q", scale=radius_scale),
            text="Quarter:N",
        )
    )

    value_labels = (
        alt.Chart(work)
        .mark_text(fontSize=11, color=TEXT_BLACK, fontWeight="bold")
        .encode(
            theta=alt.Theta("MidDeg:Q", scale=alt.Scale(domain=[0, 360])),
            radius=alt.Radius("ValueLabelR:Q", scale=radius_scale),
            text="SalesLabel:N",
        )
    )

    chart = alt.layer(
        ring_chart,
        spoke_chart,
        radar_fill,
        radar_outline,
        value_labels,
        month_names,
        quarter_names,
    ).properties(
        width=820,
        height=820,
        title="All-Years Sales Seasonality Radar (January at top, clockwise)",
    ).configure_title(
        anchor="start",
        fontSize=16,
        offset=12,
        color=TEXT_LIGHT,
    )

    st.altair_chart(chart, use_container_width=True)


def _make_pdf_radar_figure(df: pd.DataFrame):
    if df.empty:
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        ax.text(0.5, 0.5, "No radar data available", ha="center", va="center")
        ax.axis("off")
        fig.suptitle("All-Years Sales Seasonality Radar", fontsize=14, fontweight="bold")
        return fig

    labels = df["Month"].tolist()
    values = df["ScaledSalesOuter"].tolist() if "ScaledSalesOuter" in df.columns else df["ScaledSales"].tolist()

    angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
    angles += angles[:1]
    values += values[:1]

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels, fontsize=11)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.30, 0.60, 0.90, 1.20])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=9)
    ax.set_ylim(0, 1.20)

    ax.plot(angles, values, linewidth=2, color=RADAR_LINE)
    ax.fill(angles, values, alpha=0.35, color=RADAR_FILL)

    quarter_midpoints = [1, 4, 7, 10]
    quarter_names = ["Q1", "Q2", "Q3", "Q4"]
    for idx, qn in zip(quarter_midpoints, quarter_names):
        angle = angles[idx]
        ax.text(angle, 1.30, qn, ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_title("All-Years Sales Seasonality Radar (January at top, clockwise)", pad=24, fontsize=14, fontweight="bold")
    return fig


def _fig_to_rl_image(fig, width_inches: float = 9.6) -> RLImage:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    img = RLImage(buf)
    aspect = img.imageHeight / float(img.imageWidth) if img.imageWidth else 0.6
    img.drawWidth = width_inches * inch
    img.drawHeight = img.drawWidth * aspect
    return img


def _make_pdf_top2_bar_figure(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(10.2, max(4.2, 0.45 * max(len(df), 4))))

    if df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        return fig

    work = df.copy()
    work["Value"] = pd.to_numeric(work["Value"], errors="coerce").fillna(0.0)
    work = work.sort_values(["PeriodOrder", "Rank", "Value"], ascending=[True, True, False]).reset_index(drop=True)

    labels = work["YLabel"].astype(str).tolist()
    values = work["Value"].tolist()
    y = np.arange(len(work))

    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Sales")
    ax.set_title(title, fontsize=14, fontweight="bold")

    xmax = max(values) if values else 0
    pad = xmax * 0.02 if xmax > 0 else 1.0
    for yi, v in zip(y, values):
        ax.text(v + pad, yi, money(float(v)), va="center", fontsize=9)

    fig.tight_layout()
    return fig


def build_visual_analytics_pdf_bytes(
    df_scope: pd.DataFrame,
    df_vis: pd.DataFrame,
    df_hist_all: pd.DataFrame,
    granularity: str,
    labels: list[str],
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(letter),
        leftMargin=18,
        rightMargin=18,
        topMargin=18,
        bottomMargin=18,
    )

    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("Multi Month / Year Compare — Visual Analytics", styles["Title"])
    subtitle = Paragraph(
        f"Analyze By: {granularity} &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; Selected: {', '.join(labels)}",
        styles["Normal"],
    )
    story.extend([title, Spacer(1, 6), subtitle, Spacer(1, 12)])

    summary_df = _period_summary_df(df_vis)

    story.append(Paragraph("Sales and ASP by Selected Period", styles["Heading2"]))
    story.append(_fig_to_rl_image(_make_sales_asp_combo_figure(summary_df, "Sales and ASP by Selected Period"), width_inches=9.8))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Sales and Units by Selected Period", styles["Heading2"]))
    story.append(_fig_to_rl_image(_make_sales_units_combo_figure(summary_df, "Sales and Units by Selected Period"), width_inches=9.8))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Average Sales per SKU and Average Units per SKU", styles["Heading2"]))
    story.append(
        _fig_to_rl_image(
            _make_avg_sales_units_per_sku_combo_figure(
                summary_df,
                "Average Sales per SKU and Average Units per SKU",
            ),
            width_inches=9.8,
        )
    )
    story.append(Spacer(1, 10))

    if granularity == "Year":
        q_sales = _quarterly_stacked_df(df_vis, "Sales")
        q_units = _quarterly_stacked_df(df_vis, "Units")

        story.append(Paragraph("Quarterly Stacked Bars — Sales", styles["Heading2"]))
        story.append(_fig_to_rl_image(_make_quarterly_stacked_figure(q_sales, "Sales"), width_inches=9.8))
        story.append(Spacer(1, 10))

        story.append(Paragraph("Quarterly Stacked Bars — Units", styles["Heading2"]))
        story.append(_fig_to_rl_image(_make_quarterly_stacked_figure(q_units, "Units"), width_inches=9.8))
        story.append(Spacer(1, 10))

    top_retailer = _top2_per_period(df_vis, "Retailer", "Sales")
    top_vendor = _top2_per_period(df_vis, "Vendor", "Sales")
    top_sku = _top2_per_period(df_vis, "SKU", "Sales")

    story.append(Paragraph("Biggest 2 Retailers by Selected Period", styles["Heading2"]))
    story.append(_fig_to_rl_image(_make_pdf_top2_bar_figure(top_retailer, "Biggest 2 Retailers by Selected Period"), width_inches=9.8))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Biggest 2 Vendors by Selected Period", styles["Heading2"]))
    story.append(_fig_to_rl_image(_make_pdf_top2_bar_figure(top_vendor, "Biggest 2 Vendors by Selected Period"), width_inches=9.8))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Biggest 2 SKUs by Selected Period", styles["Heading2"]))
    story.append(_fig_to_rl_image(_make_pdf_top2_bar_figure(top_sku, "Biggest 2 SKUs by Selected Period"), width_inches=9.8))
    story.append(Spacer(1, 10))

    radar_df = _all_years_radar_month_df(df_hist_all)
    story.append(Paragraph("All-Years Seasonality Radar", styles["Heading2"]))
    story.append(_fig_to_rl_image(_make_pdf_radar_figure(radar_df), width_inches=6.8))

    doc.build(story)
    return buf.getvalue()


def render_visual_only(ctx: dict):
    st.subheader("Multi Month / Year Compare • Visual Analytics")
    st.caption("Chart-first view for selected years or months.")

    df_scope = ctx["df_scope"].copy()
    df_hist_all = ctx.get("df_hist_for_new", pd.DataFrame()).copy()

    if df_scope.empty:
        st.info("No data available with the current filters.")
        return

    c1, c2, c3 = st.columns([1.2, 2.4, 1.0])

    with c1:
        granularity = st.selectbox(
            "Analyze By",
            ["Year", "Month"],
            index=0,
            key="multi_compare_visual_granularity",
        )

    options = available_year_labels(df_scope) if granularity == "Year" else available_month_labels(df_scope)
    default_sel = options[-4:] if len(options) >= 4 else options

    with c2:
        labels = st.multiselect(
            f"Select {granularity}s",
            options=options,
            default=default_sel,
            key="multi_compare_visual_labels",
        )

    with c3:
        st.selectbox(
            "Radar Source",
            ["All Filtered History"],
            index=0,
            key="multi_compare_visual_radar_source",
        )

    if not labels:
        st.info(f"Select one or more {granularity.lower()}s to continue.")
        return

    ordered_labels = [x for x in options if x in labels]
    df_vis = _prepare_visual_base(df_scope, ordered_labels, granularity)

    if df_vis.empty:
        st.info("No data available for the selected visual periods.")
        return

    try:
        pdf_bytes = build_visual_analytics_pdf_bytes(
            df_scope=df_scope,
            df_vis=df_vis,
            df_hist_all=df_hist_all,
            granularity=granularity,
            labels=ordered_labels,
        )
        st.download_button(
            "Download Visual Analytics PDF",
            data=pdf_bytes,
            file_name=f"multi_compare_visual_analytics_{granularity.lower()}.pdf",
            mime="application/pdf",
            key="download_multi_compare_visual_pdf",
            use_container_width=False,
        )
    except Exception as e:
        st.warning(f"PDF export unavailable: {e}")

    summary_df = _period_summary_df(df_vis)

    st.markdown("### Sales and ASP by Selected Period")
    _render_sales_asp_combo_chart(summary_df)

    st.markdown("### Sales and Units by Selected Period")
    _render_sales_units_combo_chart(summary_df)

    st.markdown("### Average Sales per SKU and Average Units per SKU")
    _render_avg_sales_units_per_sku_combo_chart(summary_df)

    if granularity == "Year":
        st.markdown("### Quarterly Stacked Bars")
        q_sales, q_units = st.columns(2)
        with q_sales:
            _render_quarterly_stacked_altair(_quarterly_stacked_df(df_vis, "Sales"), "Sales")
        with q_units:
            _render_quarterly_stacked_altair(_quarterly_stacked_df(df_vis, "Units"), "Units")

    st.markdown("### Biggest 2 by Selected Period")
    rv1, rv2 = st.columns(2)
    with rv1:
        _render_lollipop(_top2_per_period(df_vis, "Retailer", "Sales"), "Biggest 2 Retailers by Selected Period")
    with rv2:
        _render_lollipop(_top2_per_period(df_vis, "Vendor", "Sales"), "Biggest 2 Vendors by Selected Period")

    _render_lollipop(_top2_per_period(df_vis, "SKU", "Sales"), "Biggest 2 SKUs by Selected Period")

    st.markdown("### All-Years Seasonality Radar")
    st.caption("Uses all filtered history in the app across all years for the current scope filter.")

    radar_df = _all_years_radar_month_df(df_hist_all)
    _render_radar_altair(radar_df)


def render(ctx: dict):
    df_scope = ctx["df_scope"].copy()

    st.subheader("Multi Month / Year Compare")
    st.caption("Analyze multiple months or years together in one view.")

    if df_scope.empty:
        st.info("No data available with the current filters.")
        return

    c1, c2, c3, c4, c5 = st.columns([1.0, 2.2, 1.2, 1.0, 1.0])

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
            ["Retailer", "Vendor", "SKU"],
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

    with c5:
        sort_by = st.selectbox(
            "Sort By",
            ["Latest Selected", "Total", "Average", "Alphabetical"],
            index=0,
            key="multi_compare_sort_single",
        )

    if not labels:
        st.info(f"Select one or more {granularity.lower()}s to continue.")
        return

    ordered_labels = [x for x in options if x in labels]

    _render_base_metric_cards(df_scope, ordered_labels, granularity)
    _render_top2_peak_cards(df_scope, ordered_labels, granularity)
    _render_top2_growth_cards(df_scope, ordered_labels, granularity)
    _render_multi_period_matrix(df_scope, ordered_labels, granularity, row_dim, metric, sort_by)
    _render_yoy_growth_table(df_scope, ordered_labels, granularity, row_dim, metric)
    _render_share_of_total_table(df_scope, ordered_labels, granularity, row_dim, metric)
    _render_multi_year_seasonality(df_scope, ordered_labels, granularity, metric)
    _render_performance_score(df_scope, ordered_labels, granularity, row_dim, metric)
