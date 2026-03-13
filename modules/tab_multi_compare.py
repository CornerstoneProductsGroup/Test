from __future__ import annotations

import math

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .shared_core import (
    money,
    render_df,
    available_month_labels,
    available_year_labels,
    filter_by_period_labels,
    kpi_card,
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


def _pie_chart_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame(columns=["PeriodLabel", "Value", "Pct"])

    out = (
        df.groupby("PeriodLabel", as_index=False)[metric]
        .sum()
        .rename(columns={metric: "Value"})
    )

    total = float(out["Value"].sum()) if not out.empty else 0.0
    out["Pct"] = np.where(total > 0, out["Value"] / total, 0.0)

    # preserve the visible period order from the multiselect
    order_map = {lbl: i for i, lbl in enumerate(df["PeriodLabel"].drop_duplicates().tolist())}
    out["SortOrder"] = out["PeriodLabel"].map(order_map)
    out = out.sort_values(["SortOrder", "PeriodLabel"]).reset_index(drop=True)
    out["PctLabel"] = out["Pct"].map(lambda v: f"{float(v):.1%}")
    return out


def _render_pie_chart(df: pd.DataFrame, metric: str, title: str):
    if df.empty:
        st.info(f"No {metric.lower()} pie data available.")
        return

    order_list = df["PeriodLabel"].tolist()

    chart = (
        alt.Chart(df)
        .mark_arc(outerRadius=150, innerRadius=35)
        .encode(
            theta=alt.Theta("Value:Q", stack=True),
            color=alt.Color("PeriodLabel:N", title="", sort=order_list),
            order=alt.Order("SortOrder:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("PeriodLabel:N", title="Period"),
                alt.Tooltip("Value:Q", title=metric, format=",.2f" if metric == "Sales" else ",.0f"),
                alt.Tooltip("Pct:Q", title="% of total", format=".1%"),
            ],
        )
        .properties(title=title, height=380)
    )

    pct_text = (
        alt.Chart(df)
        .mark_text(size=13, fontWeight="bold", color="white")
        .encode(
            theta=alt.Theta("Value:Q", stack=True),
            radius=alt.value(92),
            order=alt.Order("SortOrder:Q", sort="ascending"),
            text="PctLabel:N",
        )
    )

    year_text = (
        alt.Chart(df)
        .mark_text(size=11)
        .encode(
            theta=alt.Theta("Value:Q", stack=True),
            radius=alt.value(186),
            order=alt.Order("SortOrder:Q", sort="ascending"),
            text="PeriodLabel:N",
        )
    )

    st.altair_chart(chart + pct_text + year_text, use_container_width=True)

    tbl = df.copy()
    tbl["Value"] = tbl["Value"].map(lambda v: _fmt_value(float(v), metric))
    tbl["Pct"] = tbl["Pct"].map(lambda v: f"{float(v):.1%}")
    st.dataframe(tbl[["PeriodLabel", "Value", "Pct"]], use_container_width=True, hide_index=True)


def _quarterly_stacked_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns or "Quarter" not in df.columns:
        return pd.DataFrame(columns=["PeriodLabel", "Quarter", "Value", "Label"])

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

    out = out.sort_values(["PeriodOrder", "Quarter"]).copy()
    out["Label"] = out["Value"].map(lambda v: _fmt_value(float(v), metric))

    # compute bottom-aligned label positions within each stacked segment
    out["Start"] = out.groupby("PeriodLabel")["Value"].cumsum() - out["Value"]
    out["LabelY"] = out["Start"] + (out["Value"] * 0.18)
    out["ShowLabel"] = np.where(out["Value"] > 0, out["Label"], "")

    return out


def _render_quarterly_stacked(df: pd.DataFrame, metric: str):
    if df.empty:
        st.info("No quarterly stacked data available.")
        return

    work = df.copy()

    bars = (
        alt.Chart(work)
        .mark_bar()
        .encode(
            x=alt.X("PeriodLabel:N", title="Period", sort=work["PeriodLabel"].drop_duplicates().tolist()),
            y=alt.Y("Value:Q", title=metric),
            color=alt.Color("Quarter:N", title="", sort=["Q1", "Q2", "Q3", "Q4"]),
            order=alt.Order("Quarter:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("PeriodLabel:N", title="Period"),
                alt.Tooltip("Quarter:N", title="Quarter"),
                alt.Tooltip("Value:Q", title=metric, format=",.2f" if metric == "Sales" else ",.0f"),
            ],
        )
        .properties(height=440, title=f"{metric} by Quarter, stacked within each selected year")
    )

    text = (
        alt.Chart(work[work["ShowLabel"] != ""])
        .mark_text(size=11, color="black", fontWeight="bold", baseline="top")
        .encode(
            x=alt.X("PeriodLabel:N", sort=work["PeriodLabel"].drop_duplicates().tolist()),
            y=alt.Y("LabelY:Q", title=metric),
            detail="Quarter:N",
            text="ShowLabel:N",
        )
    )

    st.altair_chart(bars + text, use_container_width=True)


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
    out["Label"] = out["Month"]
    out["Rank"] = out["TotalSales"].rank(method="dense", ascending=False).astype(int)
    return out


def _render_radar(df: pd.DataFrame):
    if df.empty:
        st.info("Not enough data to build the radar chart.")
        return

    labels = df["Label"].tolist()
    values = df["ScaledSales"].tolist()

    angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
    angles += angles[:1]
    values += values[:1]

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], labels, fontsize=9)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_ylim(0, 1.0)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.20)
    ax.set_title("All-Years Sales Seasonality Radar (Jan → Dec)", pad=24, fontsize=14)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


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

    st.markdown("### Period Share")
    c_sales, c_units = st.columns(2)
    with c_sales:
        _render_pie_chart(_pie_chart_df(df_vis, "Sales"), "Sales", "Total Sales by Selected Period")
    with c_units:
        _render_pie_chart(_pie_chart_df(df_vis, "Units"), "Units", "Total Units by Selected Period")

    if granularity == "Year":
        st.markdown("### Quarterly Stacked Bars")
        q_sales, q_units = st.columns(2)
        with q_sales:
            _render_quarterly_stacked(_quarterly_stacked_df(df_vis, "Sales"), "Sales")
        with q_units:
            _render_quarterly_stacked(_quarterly_stacked_df(df_vis, "Units"), "Units")

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
    r1, r2 = st.columns([1.6, 1.0])

    with r1:
        _render_radar(radar_df)

    with r2:
        if radar_df.empty:
            st.info("No radar data available.")
        else:
            tbl = radar_df[["Quarter", "Month", "TotalSales", "Rank"]].copy()
            tbl = tbl.sort_values(["Rank", "Month"]).reset_index(drop=True)
            tbl["TotalSales"] = tbl["TotalSales"].map(money)
            st.dataframe(tbl, use_container_width=True, hide_index=True)


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
