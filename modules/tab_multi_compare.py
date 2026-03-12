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


def _period_entity_summary(df_scope: pd.DataFrame, labels: list[str], granularity: str, dim: str) -> pd.DataFrame:
    rows = []
    for lbl in labels:
        part = filter_by_period_labels(df_scope, [lbl], granularity)
        if part.empty or dim not in part.columns:
            continue
        grp = part.groupby(dim, as_index=False).agg(Sales=("Sales", "sum"))
        total_sales = float(grp["Sales"].sum()) if not grp.empty else 0.0
        if grp.empty:
            continue
        grp["Share"] = np.where(total_sales != 0, grp["Sales"] / total_sales, 0.0)
        grp["Period"] = lbl
        rows.append(grp[[dim, "Period", "Sales", "Share"]])

    if not rows:
        return pd.DataFrame(columns=[dim, "Period", "Sales", "Share"])

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={dim: "Entity"})
    return out.sort_values("Sales", ascending=False).reset_index(drop=True)


def _growth_entity_summary(df_scope: pd.DataFrame, labels: list[str], granularity: str, dim: str) -> pd.DataFrame:
    rows = []
    if len(labels) < 2:
        return pd.DataFrame(columns=[dim, "Period", "Growth", "Pct"])

    matrices = {}
    for lbl in labels:
        part = filter_by_period_labels(df_scope, [lbl], granularity)
        if part.empty or dim not in part.columns:
            matrices[lbl] = pd.DataFrame(columns=[dim, "Sales"])
            continue
        grp = part.groupby(dim, as_index=False).agg(Sales=("Sales", "sum"))
        matrices[lbl] = grp

    for i in range(1, len(labels)):
        prev_lbl = labels[i - 1]
        cur_lbl = labels[i]
        prev_df = matrices[prev_lbl].rename(columns={"Sales": "PrevSales"})
        cur_df = matrices[cur_lbl].rename(columns={"Sales": "CurSales"})
        m = cur_df.merge(prev_df, on=dim, how="outer").fillna(0.0)
        if m.empty:
            continue
        m["Growth"] = m["CurSales"] - m["PrevSales"]
        m["Pct"] = [
            _safe_pct_change(cur, prev)
            for cur, prev in zip(m["CurSales"].tolist(), m["PrevSales"].tolist())
        ]
        m["Period"] = f"{cur_lbl} vs {prev_lbl}"
        rows.append(m[[dim, "Period", "Growth", "Pct"]])

    if not rows:
        return pd.DataFrame(columns=[dim, "Period", "Growth", "Pct"])

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={dim: "Entity"})
    out = out.sort_values("Growth", ascending=False).reset_index(drop=True)
    return out


def _truncate_text(x: str, max_len: int = 36) -> str:
    x = str(x)
    return x if len(x) <= max_len else x[: max_len - 1] + "…"


def _render_native_kpi_box(title: str, first: dict | None, second: dict | None):
    with st.container(border=True):
        st.caption(title)

        def _show_item(rank_label: str, item: dict | None):
            st.write(f"**{rank_label}**")
            if item is None:
                st.write("—")
                return

            st.markdown(
                f"<div style='font-size:1.1rem; font-weight:700; line-height:1.25;'>{_truncate_text(item['name'])}</div>",
                unsafe_allow_html=True,
            )
            st.write(f"**{item['value']}**")
            st.caption(item["detail"])

        _show_item("#1", first)
        st.write("")
        _show_item("#2", second)


def _render_top2_peak_cards(df_scope: pd.DataFrame, labels: list[str], granularity: str):
    st.markdown("### Biggest by Period")

    retail = _period_entity_summary(df_scope, labels, granularity, "Retailer").head(2)
    vendor = _period_entity_summary(df_scope, labels, granularity, "Vendor").head(2)
    sku = _period_entity_summary(df_scope, labels, granularity, "SKU").head(2)

    def _pack(df: pd.DataFrame, idx: int):
        if len(df) <= idx:
            return None
        row = df.iloc[idx]
        return {
            "name": row["Entity"],
            "value": money(float(row["Sales"])),
            "detail": f"{row['Period']} • {float(row['Share']):.1%} share",
        }

    c1, c2, c3 = st.columns(3)
    with c1:
        _render_native_kpi_box("Biggest Retailer", _pack(retail, 0), _pack(retail, 1))
    with c2:
        _render_native_kpi_box("Biggest Vendor", _pack(vendor, 0), _pack(vendor, 1))
    with c3:
        _render_native_kpi_box("Biggest SKU", _pack(sku, 0), _pack(sku, 1))


def _render_top2_growth_cards(df_scope: pd.DataFrame, labels: list[str], granularity: str):
    st.markdown("### Biggest Growth")

    retail = _growth_entity_summary(df_scope, labels, granularity, "Retailer").head(2)
    vendor = _growth_entity_summary(df_scope, labels, granularity, "Vendor").head(2)
    sku = _growth_entity_summary(df_scope, labels, granularity, "SKU").head(2)

    def _pack(df: pd.DataFrame, idx: int):
        if len(df) <= idx:
            return None
        row = df.iloc[idx]
        pct_text = "—" if pd.isna(row["Pct"]) else f"{float(row['Pct']):.1%}"
        return {
            "name": row["Entity"],
            "value": money(float(row["Growth"])),
            "detail": f"{row['Period']} • {pct_text}",
        }

    c1, c2, c3 = st.columns(3)
    with c1:
        _render_native_kpi_box("Retailer Growth", _pack(retail, 0), _pack(retail, 1))
    with c2:
        _render_native_kpi_box("Vendor Growth", _pack(vendor, 0), _pack(vendor, 1))
    with c3:
        _render_native_kpi_box("SKU Growth", _pack(sku, 0), _pack(sku, 1))


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
    d["WeekEnd"] = pd.to_datetime(d["WeekEnd"], errors="coerce")
    d = d[d["WeekEnd"].notna()].copy()
    d["Year"] = d["WeekEnd"].dt.year.astype(str)
    d["Quarter"] = d["WeekEnd"].dt.quarter.map(lambda q: f"Q{int(q)}")

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
