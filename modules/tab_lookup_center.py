from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .shared_core import (
    money,
    render_df,
    calc_kpis,
    available_month_labels,
    available_year_labels,
    filter_by_period_labels,
)

TIMEFRAME_OPTIONS = [
    "Last 4 weeks",
    "Last 8 weeks",
    "Last 13 weeks",
    "Last 26 weeks",
    "Last 52 weeks",
    "Last 104 weeks",
    "Last 156 weeks",
    "YTD",
    "All history",
]


def _pick_lookup_period(df: pd.DataFrame, mode: str):
    w = pd.to_datetime(df.get("WeekEnd"), errors="coerce").dropna()
    if w.empty:
        return None, df.iloc[0:0].copy()

    anchor = w.max().normalize()

    if mode == "All history":
        start = w.min().normalize()
        end = anchor
    elif mode == "YTD":
        start = pd.Timestamp(year=anchor.year, month=1, day=1)
        end = anchor
    else:
        weeks = int("".join(ch for ch in mode if ch.isdigit()) or 8)
        start = anchor - pd.Timedelta(days=7 * weeks - 1)
        end = anchor

    out = df[
        (pd.to_datetime(df["WeekEnd"], errors="coerce") >= start)
        & (pd.to_datetime(df["WeekEnd"], errors="coerce") <= end)
    ].copy()

    return (start, end), out


def _filter_period(df: pd.DataFrame, period):
    if period is None:
        return df.iloc[0:0].copy()
    start, end = period
    d = df.copy()
    d["WeekEnd"] = pd.to_datetime(d["WeekEnd"], errors="coerce")
    return d[(d["WeekEnd"] >= start) & (d["WeekEnd"] <= end)].copy()


def _period_prev_same_length(period):
    start, end = period
    days = (end - start).days + 1
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    return prev_start.normalize(), prev_end.normalize()


def _period_yoy(period):
    start, end = period
    return (
        (start - pd.DateOffset(years=1)).normalize(),
        (end - pd.DateOffset(years=1)).normalize(),
    )


def _fmt_num(v, metric: str):
    return money(v) if metric == "Sales" else f"{float(v):,.0f}"


def _render_kpi_card(title: str, value: str, delta: str | None = None):
    delta_html = f"<div class='kpi-delta'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_cards(df_sel: pd.DataFrame):
    k = calc_kpis(df_sel)

    wk = (
        df_sel.groupby("WeekEnd", as_index=False)
        .agg(Sales=("Sales", "sum"), Units=("Units", "sum"))
        .sort_values("WeekEnd")
    )

    best_week = "—"
    if not wk.empty:
        best_row = wk.sort_values("Sales", ascending=False).iloc[0]
        best_week = f"{pd.to_datetime(best_row['WeekEnd']).date()} • {money(best_row['Sales'])}"

    active_retailers = int(df_sel.loc[df_sel["Sales"] > 0, "Retailer"].nunique()) if "Retailer" in df_sel.columns else 0
    active_skus = int(df_sel.loc[df_sel["Sales"] > 0, "SKU"].nunique()) if "SKU" in df_sel.columns else 0
    active_vendors = int(df_sel.loc[df_sel["Sales"] > 0, "Vendor"].nunique()) if "Vendor" in df_sel.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        _render_kpi_card("Total Sales", money(k["Sales"]))
    with c2:
        _render_kpi_card("Total Units", f"{k['Units']:,.0f}")
    with c3:
        _render_kpi_card("ASP", money(k["ASP"]))
    with c4:
        _render_kpi_card("Active Retailers", f"{active_retailers:,}")
    with c5:
        _render_kpi_card("Active SKUs", f"{active_skus:,}")
    with c6:
        _render_kpi_card("Active Vendors", f"{active_vendors:,}")

    c7, c8 = st.columns(2)
    with c7:
        _render_kpi_card("Best Week", best_week)
    with c8:
        if not wk.empty:
            latest_row = wk.sort_values("WeekEnd").iloc[-1]
            latest_week = f"{pd.to_datetime(latest_row['WeekEnd']).date()} • {money(latest_row['Sales'])}"
        else:
            latest_week = "—"
        _render_kpi_card("Latest Week", latest_week)


def _compare_delta_text(cur, prev, money_mode=False):
    delta = float(cur) - float(prev)
    if money_mode:
        return f"{money(delta)} vs compare"
    return f"{delta:,.0f} vs compare"


def _weekly_pivot(df: pd.DataFrame, row_dim: str, metric: str) -> pd.DataFrame:
    d = df.groupby([row_dim, "WeekEnd"], as_index=False).agg(Value=(metric, "sum"))
    d["Week"] = pd.to_datetime(d["WeekEnd"]).dt.date.astype(str)

    piv = d.pivot_table(
        index=row_dim,
        columns="Week",
        values="Value",
        aggfunc="sum",
        fill_value=0.0,
    )

    week_cols = list(piv.columns)
    if week_cols:
        vals = piv[week_cols]
        piv["Average"] = vals.mean(axis=1)
        piv["Current"] = vals.iloc[:, -1]
        piv["Vs Avg"] = piv["Current"] - piv["Average"]
        piv["Active Weeks"] = (vals > 0).sum(axis=1)

    return piv.reset_index()


def _seasonality_tables(df_sel: pd.DataFrame, metric: str):
    d = df_sel.copy()
    d["WeekEnd"] = pd.to_datetime(d["WeekEnd"], errors="coerce")
    d = d[d["WeekEnd"].notna()].copy()

    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), None, None, "Low"

    def _bar(v, vmax, width=10):
        if vmax <= 0 or v <= 0:
            return ""
        n = max(1, int(round((float(v) / float(vmax)) * width)))
        return "▓" * n

    # Month-by-month seasonality in chronological order
    d["MonthPeriod"] = d["WeekEnd"].dt.to_period("M")
    month = d.groupby("MonthPeriod", as_index=False).agg(
        Total=(metric, "sum"),
        ActiveWeeks=("WeekEnd", "nunique"),
    )
    month["AvgWeekly"] = np.where(
        month["ActiveWeeks"] != 0,
        month["Total"] / month["ActiveWeeks"],
        0.0,
    )
    month["MonthStart"] = month["MonthPeriod"].dt.to_timestamp()
    month = month.sort_values("MonthStart", ascending=True).reset_index(drop=True)

    max_month = float(month["AvgWeekly"].max()) if not month.empty else 0.0
    month["Period"] = month["MonthStart"].dt.strftime("%B %Y")
    month["Visual"] = month["AvgWeekly"].map(lambda v: _bar(v, max_month, 10))
    month["Value"] = month["AvgWeekly"].map(lambda v: _fmt_num(v, metric))
    month_show = month[["Period", "Visual", "Value"]].copy()

    peak_row = month.sort_values("AvgWeekly", ascending=False).iloc[0] if max_month > 0 else None
    nz_month = month[month["AvgWeekly"] > 0]
    low_row = nz_month.sort_values("AvgWeekly", ascending=True).iloc[0] if not nz_month.empty else None

    nz = month["AvgWeekly"][month["AvgWeekly"] > 0]
    if len(nz) >= 2:
        ratio = float(nz.max() / nz.mean()) if float(nz.mean()) > 0 else 0.0
        strength = "High" if ratio >= 1.6 else ("Medium" if ratio >= 1.25 else "Low")
    else:
        strength = "Low"

    # Quarter-by-quarter seasonality
    d["Year"] = d["WeekEnd"].dt.year.astype(int)
    d["QuarterNum"] = d["WeekEnd"].dt.quarter.astype(int)
    d["Quarter"] = d["QuarterNum"].map(lambda q: f"Q{int(q)}")

    quarter = d.groupby(["Year", "QuarterNum", "Quarter"], as_index=False).agg(
        Total=(metric, "sum"),
        ActiveWeeks=("WeekEnd", "nunique"),
    )
    quarter["AvgWeekly"] = np.where(
        quarter["ActiveWeeks"] != 0,
        quarter["Total"] / quarter["ActiveWeeks"],
        0.0,
    )

    # Most current year at top, quarters in chronological order within each year
    quarter = quarter.sort_values(["Year", "QuarterNum"], ascending=[False, True]).reset_index(drop=True)

    max_quarter = float(quarter["AvgWeekly"].max()) if not quarter.empty else 0.0
    quarter["Visual"] = quarter["AvgWeekly"].map(lambda v: _bar(v, max_quarter, 10))
    quarter["Value"] = quarter["AvgWeekly"].map(lambda v: _fmt_num(v, metric))
    quarter_show = quarter[["Year", "Quarter", "Visual", "Value"]].copy()
    quarter_show["Year"] = quarter_show["Year"].astype(str)

    peak_txt = (
        f"{peak_row['Period']} • {_fmt_num(peak_row['AvgWeekly'], metric)}"
        if peak_row is not None and peak_row["AvgWeekly"] > 0
        else "—"
    )
    low_txt = (
        f"{low_row['Period']} • {_fmt_num(low_row['AvgWeekly'], metric)}"
        if low_row is not None
        else "—"
    )

    return month_show, quarter_show, peak_txt, low_txt, strength


def _render_seasonality_section(df_sel: pd.DataFrame, metric: str, title: str = "Seasonality"):
    st.markdown(f"### {title}")
    month_show, quarter_show, peak_txt, low_txt, strength = _seasonality_tables(df_sel, metric)

    if month_show.empty:
        st.caption("No seasonality history available.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        _render_kpi_card("Peak Month", peak_txt)
    with c2:
        _render_kpi_card("Low Month", low_txt)
    with c3:
        _render_kpi_card("Seasonality Strength", strength)

    left, right = st.columns(2)
    with left:
        st.markdown("**Month-by-Month Seasonality**")
        render_df(month_show, height=520)
    with right:
        st.markdown("**Quarter-by-Quarter Seasonality**")
        render_df(quarter_show, height=420)


def _render_retailer_breakdown(df_sel: pd.DataFrame, metric: str):
    st.markdown("### Retailer Breakdown")

    if "Retailer" not in df_sel.columns:
        st.caption("Retailer column not available.")
        return

    rb = (
        df_sel.groupby("Retailer", as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Units=("Units", "sum"),
            ActiveWeeks=("WeekEnd", "nunique"),
        )
        .sort_values(metric, ascending=False)
    )

    if rb.empty:
        st.caption("No retailer activity found.")
        return

    rb["ASP"] = np.where(rb["Units"] != 0, rb["Sales"] / rb["Units"], 0.0)
    total_metric = rb[metric].sum()
    rb["Share %"] = np.where(total_metric != 0, rb[metric] / total_metric, 0.0)

    show = rb.copy()
    show["Sales"] = show["Sales"].map(money)
    show["Units"] = show["Units"].map(lambda v: f"{v:,.0f}")
    show["ASP"] = show["ASP"].map(money)
    show["Share %"] = show["Share %"].map(lambda v: f"{v * 100:,.1f}%")
    show["ActiveWeeks"] = show["ActiveWeeks"].map(lambda v: f"{v:,.0f}")
    show = show.rename(columns={"ActiveWeeks": "Active Weeks"})

    render_df(show, height=350)


def _render_vendor_breakdown(df_sel: pd.DataFrame, metric: str):
    st.markdown("### Vendor Breakdown")

    if "Vendor" not in df_sel.columns:
        st.caption("Vendor column not available.")
        return

    rb = (
        df_sel.groupby("Vendor", as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Units=("Units", "sum"),
            ActiveWeeks=("WeekEnd", "nunique"),
        )
        .sort_values(metric, ascending=False)
    )

    if rb.empty:
        st.caption("No vendor activity found.")
        return

    rb["ASP"] = np.where(rb["Units"] != 0, rb["Sales"] / rb["Units"], 0.0)
    total_metric = rb[metric].sum()
    rb["Share %"] = np.where(total_metric != 0, rb[metric] / total_metric, 0.0)

    show = rb.copy()
    show["Sales"] = show["Sales"].map(money)
    show["Units"] = show["Units"].map(lambda v: f"{v:,.0f}")
    show["ASP"] = show["ASP"].map(money)
    show["Share %"] = show["Share %"].map(lambda v: f"{v * 100:,.1f}%")
    show["ActiveWeeks"] = show["ActiveWeeks"].map(lambda v: f"{v:,.0f}")
    show = show.rename(columns={"ActiveWeeks": "Active Weeks"})

    render_df(show, height=350)


def _render_sku_breakdown(df_sel: pd.DataFrame, metric: str):
    st.markdown("### SKU Breakdown")

    if "SKU" not in df_sel.columns:
        st.caption("SKU column not available.")
        return

    rb = (
        df_sel.groupby("SKU", as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Units=("Units", "sum"),
            ActiveWeeks=("WeekEnd", "nunique"),
        )
        .sort_values(metric, ascending=False)
    )

    if rb.empty:
        st.caption("No SKU activity found.")
        return

    rb["ASP"] = np.where(rb["Units"] != 0, rb["Sales"] / rb["Units"], 0.0)
    total_metric = rb[metric].sum()
    rb["Share %"] = np.where(total_metric != 0, rb[metric] / total_metric, 0.0)

    show = rb.copy()
    show["Sales"] = show["Sales"].map(money)
    show["Units"] = show["Units"].map(lambda v: f"{v:,.0f}")
    show["ASP"] = show["ASP"].map(money)
    show["Share %"] = show["Share %"].map(lambda v: f"{v * 100:,.1f}%")
    show["ActiveWeeks"] = show["ActiveWeeks"].map(lambda v: f"{v:,.0f}")
    show = show.rename(columns={"ActiveWeeks": "Active Weeks"})

    render_df(show, height=350)


def _render_weekly_velocity(df_sel: pd.DataFrame, lookup_type: str, metric: str):
    st.markdown("### Weekly Velocity")

    if lookup_type == "SKU":
        row_dim = "Retailer" if "Retailer" in df_sel.columns else None
    elif lookup_type == "Vendor":
        row_dim = "SKU" if "SKU" in df_sel.columns else None
    else:
        row_dim = "Vendor" if "Vendor" in df_sel.columns else None

    if row_dim is None:
        st.caption("Weekly velocity dimension not available.")
        return

    piv = _weekly_pivot(df_sel, row_dim, metric)

    if piv.empty:
        st.caption("No weekly activity available.")
        return

    week_cols = [c for c in piv.columns if c not in [row_dim, "Average", "Current", "Vs Avg", "Active Weeks"]]

    for c in week_cols + [col for col in ["Average", "Current", "Vs Avg"] if col in piv.columns]:
        piv[c] = piv[c].map(lambda v: _fmt_num(v, metric))

    if "Active Weeks" in piv.columns:
        piv["Active Weeks"] = piv["Active Weeks"].map(lambda v: f"{v:,.0f}")

    render_df(piv, height=420)


def _render_compare_section(df_base: pd.DataFrame, metric: str, default_period):
    st.markdown("### Compare")

    compare_mode = st.selectbox(
        "Compare Type",
        [
            "Prior period (same length)",
            "YoY (same dates)",
            "Custom months",
            "Custom years",
        ],
        index=0,
        key="lookup_compare_mode",
    )

    cur_label = ""
    cmp_label = ""

    if compare_mode == "Prior period (same length)":
        cur_period = default_period
        cmp_period = _period_prev_same_length(default_period)
        df_cur = _filter_period(df_base, cur_period)
        df_cmp = _filter_period(df_base, cmp_period)
        cur_label = f"{cur_period[0].date()} → {cur_period[1].date()}"
        cmp_label = f"{cmp_period[0].date()} → {cmp_period[1].date()}"

    elif compare_mode == "YoY (same dates)":
        cur_period = default_period
        cmp_period = _period_yoy(default_period)
        df_cur = _filter_period(df_base, cur_period)
        df_cmp = _filter_period(df_base, cmp_period)
        cur_label = f"{cur_period[0].date()} → {cur_period[1].date()}"
        cmp_label = f"{cmp_period[0].date()} → {cmp_period[1].date()}"

    elif compare_mode == "Custom months":
        month_options = available_month_labels(df_base)
        c1, c2 = st.columns(2)
        with c1:
            cur_months = st.multiselect(
                "Current month(s)",
                options=month_options,
                default=month_options[-1:] if month_options else [],
                key="lookup_compare_cur_months",
            )
        with c2:
            cmp_months = st.multiselect(
                "Compare month(s)",
                options=month_options,
                default=month_options[-2:-1] if len(month_options) > 1 else [],
                key="lookup_compare_cmp_months",
            )

        df_cur = filter_by_period_labels(df_base, cur_months, "Month") if cur_months else df_base.iloc[0:0].copy()
        df_cmp = filter_by_period_labels(df_base, cmp_months, "Month") if cmp_months else df_base.iloc[0:0].copy()
        cur_label = ", ".join(cur_months) if cur_months else "None"
        cmp_label = ", ".join(cmp_months) if cmp_months else "None"

    else:
        year_options = available_year_labels(df_base)
        c1, c2 = st.columns(2)
        with c1:
            cur_years = st.multiselect(
                "Current year(s)",
                options=year_options,
                default=year_options[-1:] if year_options else [],
                key="lookup_compare_cur_years",
            )
        with c2:
            cmp_years = st.multiselect(
                "Compare year(s)",
                options=year_options,
                default=year_options[-2:-1] if len(year_options) > 1 else [],
                key="lookup_compare_cmp_years",
            )

        df_cur = filter_by_period_labels(df_base, cur_years, "Year") if cur_years else df_base.iloc[0:0].copy()
        df_cmp = filter_by_period_labels(df_base, cmp_years, "Year") if cmp_years else df_base.iloc[0:0].copy()
        cur_label = ", ".join(cur_years) if cur_years else "None"
        cmp_label = ", ".join(cmp_years) if cmp_years else "None"

    if df_cur.empty:
        st.info("No data available for the current compare selection.")
        return

    if df_cmp.empty:
        st.info("No data available for the compare selection.")
        return

    k_cur = calc_kpis(df_cur)
    k_cmp = calc_kpis(df_cmp)

    c1, c2, c3 = st.columns(3)
    with c1:
        _render_kpi_card(
            "Sales Δ",
            money(k_cur["Sales"] - k_cmp["Sales"]),
            _compare_delta_text(k_cur["Sales"], k_cmp["Sales"], money_mode=True),
        )
    with c2:
        _render_kpi_card(
            "Units Δ",
            f"{k_cur['Units'] - k_cmp['Units']:,.0f}",
            _compare_delta_text(k_cur["Units"], k_cmp["Units"], money_mode=False),
        )
    with c3:
        _render_kpi_card(
            "ASP Δ",
            money(k_cur["ASP"] - k_cmp["ASP"]),
            _compare_delta_text(k_cur["ASP"], k_cmp["ASP"], money_mode=True),
        )

    st.caption(f"Current: {cur_label}")
    st.caption(f"Compare: {cmp_label}")

    if "Retailer" in df_cur.columns:
        cur_grp = df_cur.groupby("Retailer", as_index=False).agg(
            Current_Sales=("Sales", "sum"),
            Current_Units=("Units", "sum"),
        )
        cmp_grp = df_cmp.groupby("Retailer", as_index=False).agg(
            Compare_Sales=("Sales", "sum"),
            Compare_Units=("Units", "sum"),
        )

        comp = cur_grp.merge(cmp_grp, on="Retailer", how="outer").fillna(0.0)
        comp["Sales Δ"] = comp["Current_Sales"] - comp["Compare_Sales"]
        comp["Units Δ"] = comp["Current_Units"] - comp["Compare_Units"]

        if metric == "Sales":
            comp = comp.sort_values("Sales Δ", ascending=False)
        else:
            comp = comp.sort_values("Units Δ", ascending=False)

        show = comp.copy()
        show["Current_Sales"] = show["Current_Sales"].map(money)
        show["Compare_Sales"] = show["Compare_Sales"].map(money)
        show["Sales Δ"] = show["Sales Δ"].map(money)
        show["Current_Units"] = show["Current_Units"].map(lambda v: f"{v:,.0f}")
        show["Compare_Units"] = show["Compare_Units"].map(lambda v: f"{v:,.0f}")
        show["Units Δ"] = show["Units Δ"].map(lambda v: f"{v:,.0f}")

        st.markdown("#### Compare Breakdown")
        render_df(show, height=360)

    wk_cur = (
        df_cur.groupby("WeekEnd", as_index=False)
        .agg(Current_Value=(metric, "sum"))
        .sort_values("WeekEnd")
    )
    wk_cmp = (
        df_cmp.groupby("WeekEnd", as_index=False)
        .agg(Compare_Value=(metric, "sum"))
        .sort_values("WeekEnd")
    )

    wk_cur["Week"] = pd.to_datetime(wk_cur["WeekEnd"]).dt.date.astype(str)
    wk_cmp["Week"] = pd.to_datetime(wk_cmp["WeekEnd"]).dt.date.astype(str)

    wk_show = wk_cur[["Week", "Current_Value"]].merge(
        wk_cmp[["Week", "Compare_Value"]],
        on="Week",
        how="outer",
    ).fillna(0.0)

    wk_show["Δ"] = wk_show["Current_Value"] - wk_show["Compare_Value"]

    if metric == "Sales":
        wk_show["Current_Value"] = wk_show["Current_Value"].map(money)
        wk_show["Compare_Value"] = wk_show["Compare_Value"].map(money)
        wk_show["Δ"] = wk_show["Δ"].map(money)
    else:
        wk_show["Current_Value"] = wk_show["Current_Value"].map(lambda v: f"{v:,.0f}")
        wk_show["Compare_Value"] = wk_show["Compare_Value"].map(lambda v: f"{v:,.0f}")
        wk_show["Δ"] = wk_show["Δ"].map(lambda v: f"{v:,.0f}")

    st.markdown("#### Weekly Compare")
    render_df(wk_show, height=320)


def render(ctx: dict):
    df_scope = ctx["df_scope"].copy()

    st.header("Lookup Center")
    st.caption("Deep lookup by SKU, Vendor, or Retailer with multi-select and compare mode.")

    if df_scope.empty:
        st.info("No data available with the current sidebar filters.")
        return

    c1, c2, c3, c4, c5 = st.columns([1.1, 2.7, 1.2, 1.0, 1.0])

    with c1:
        lookup_type = st.selectbox(
            "Lookup Type",
            ["SKU", "Vendor", "Retailer"],
            index=0,
            key="lookup_center_type",
        )

    if lookup_type == "SKU":
        options = sorted(df_scope["SKU"].dropna().astype(str).unique().tolist())
    elif lookup_type == "Vendor":
        options = sorted(df_scope["Vendor"].dropna().astype(str).unique().tolist())
    else:
        options = sorted(df_scope["Retailer"].dropna().astype(str).unique().tolist())

    with c2:
        selected_values = st.multiselect(
            f"{lookup_type}(s)",
            options=options,
            default=[],
            key="lookup_center_values",
        )

    with c3:
        select_all = st.checkbox("Select All", value=False, key="lookup_center_select_all")

    if select_all:
        selected_values = options

    with c4:
        timeframe = st.selectbox(
            "Timeframe",
            TIMEFRAME_OPTIONS,
            index=1,
            key="lookup_center_timeframe",
        )

    with c5:
        metric = st.selectbox(
            "Metric",
            ["Sales", "Units"],
            index=0,
            key="lookup_center_metric",
        )

    advanced_compare = st.toggle("Advanced Compare", value=False, key="lookup_center_advanced_compare")

    if not options:
        st.info("No lookup values available with the current filters.")
        return

    if not selected_values:
        st.info(f"Select one or more {lookup_type.lower()} values to continue.")
        return

    if lookup_type == "SKU":
        df_lookup_all = df_scope[df_scope["SKU"].astype(str).isin(selected_values)].copy()
    elif lookup_type == "Vendor":
        df_lookup_all = df_scope[df_scope["Vendor"].astype(str).isin(selected_values)].copy()
    else:
        df_lookup_all = df_scope[df_scope["Retailer"].astype(str).isin(selected_values)].copy()

    period, df_sel = _pick_lookup_period(df_lookup_all, timeframe)

    if period is None or df_sel.empty:
        st.info("No data for that lookup in the selected timeframe.")
        return

    label_preview = ", ".join(selected_values[:5])
    if len(selected_values) > 5:
        label_preview += f" +{len(selected_values) - 5} more"

    st.markdown(
        f"""
        <div style='padding:8px 12px;border:1px solid rgba(128,128,128,0.22);border-radius:12px;
        background:var(--secondary-background-color);display:inline-block;font-weight:700;'>
        {lookup_type}: {label_preview} &nbsp; • &nbsp; {timeframe} &nbsp; • &nbsp;
        {period[0].date()} → {period[1].date()} &nbsp; • &nbsp; Metric: {metric}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown("### Quick Intelligence Summary")
    _render_summary_cards(df_sel)

    if advanced_compare:
        _render_compare_section(df_lookup_all, metric, period)

    if lookup_type == "SKU":
        _render_retailer_breakdown(df_sel, metric)
    elif lookup_type == "Vendor":
        _render_retailer_breakdown(df_sel, metric)
        _render_sku_breakdown(df_sel, metric)
    else:
        _render_vendor_breakdown(df_sel, metric)
        _render_sku_breakdown(df_sel, metric)

    _render_seasonality_section(df_sel, metric, title="Seasonality")
    _render_weekly_velocity(df_sel, lookup_type, metric)
