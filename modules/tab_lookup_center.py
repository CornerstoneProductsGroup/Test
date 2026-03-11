from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from .shared_core import money, render_df, calc_kpis

TIMEFRAME_OPTIONS = [
    "Last 4 weeks",
    "Last 8 weeks",
    "Last 13 weeks",
    "Last 26 weeks",
    "Last 52 weeks",
    "YTD",
    "All history"
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
        (pd.to_datetime(df["WeekEnd"], errors="coerce") >= start) &
        (pd.to_datetime(df["WeekEnd"], errors="coerce") <= end)
    ].copy()

    return (start, end), out


def _fmt_num(v, metric):
    if metric == "Sales":
        return money(v)
    return f"{float(v):,.0f}"


def _render_kpi(title, value):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def _render_summary_cards(df):
    k = calc_kpis(df)

    c1, c2, c3 = st.columns(3)

    with c1:
        _render_kpi("Sales", money(k["Sales"]))

    with c2:
        _render_kpi("Units", f"{k['Units']:,.0f}")

    with c3:
        _render_kpi("ASP", money(k["ASP"]))


def render(ctx: dict):

    df_scope = ctx["df_scope"].copy()

    st.header("Lookup Center")

    if df_scope.empty:
        st.info("No data available.")
        return

    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])

    with col1:
        lookup_type = st.selectbox(
            "Lookup Type",
            ["SKU", "Vendor", "Retailer"]
        )

    if lookup_type == "SKU":
        options = sorted(df_scope["SKU"].dropna().astype(str).unique())

    elif lookup_type == "Vendor":
        options = sorted(df_scope["Vendor"].dropna().astype(str).unique())

    else:
        options = sorted(df_scope["Retailer"].dropna().astype(str).unique())

    with col2:

        multi = st.multiselect(
            f"{lookup_type}s",
            options
        )

        select_all = st.checkbox("Select All")

        if select_all:
            multi = options

    with col3:
        timeframe = st.selectbox(
            "Timeframe",
            TIMEFRAME_OPTIONS,
            index=1
        )

    with col4:
        metric = st.selectbox(
            "Metric",
            ["Sales", "Units"]
        )

    advanced = st.toggle("Advanced Compare")

    if not multi:
        st.info("Select at least one item.")
        return

    df_sel = df_scope[df_scope[lookup_type].isin(multi)]

    period, df_sel = _pick_lookup_period(df_sel, timeframe)

    if period is None or df_sel.empty:
        st.info("No data for this selection.")
        return

    st.markdown("### Quick Intelligence Summary")

    _render_summary_cards(df_sel)

    if advanced:

        st.markdown("### Compare")

        colA, colB = st.columns(2)

        with colA:
            tfA = st.selectbox("Current Period", TIMEFRAME_OPTIONS, index=1)

        with colB:
            tfB = st.selectbox("Compare Period", TIMEFRAME_OPTIONS, index=2)

        _, dfA = _pick_lookup_period(df_sel, tfA)
        _, dfB = _pick_lookup_period(df_sel, tfB)

        kA = calc_kpis(dfA)
        kB = calc_kpis(dfB)

        c1, c2, c3 = st.columns(3)

        with c1:
            _render_kpi("Sales Δ", money(kA["Sales"] - kB["Sales"]))

        with c2:
            _render_kpi("Units Δ", f"{kA['Units'] - kB['Units']:,.0f}")

        with c3:
            _render_kpi("ASP Δ", money(kA["ASP"] - kB["ASP"]))

    st.markdown("### Retailer Breakdown")

    rb = (
        df_sel
        .groupby("Retailer", as_index=False)
        .agg(Sales=("Sales", "sum"), Units=("Units", "sum"))
        .sort_values(metric, ascending=False)
    )

    if not rb.empty:

        rb["Sales"] = rb["Sales"].map(money)
        rb["Units"] = rb["Units"].map(lambda v: f"{v:,.0f}")

        render_df(rb, height=350)

    st.markdown("### Weekly Velocity")

    wk = (
        df_sel
        .groupby(["Retailer", "WeekEnd"], as_index=False)
        .agg(Value=(metric, "sum"))
    )

    wk["Week"] = pd.to_datetime(wk["WeekEnd"]).dt.date.astype(str)

    piv = wk.pivot_table(
        index="Retailer",
        columns="Week",
        values="Value",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    for c in piv.columns[1:]:
        piv[c] = piv[c].map(lambda v: _fmt_num(v, metric))

    render_df(piv, height=420)
