from __future__ import annotations
import numpy as np
import streamlit as st
from .shared_core import money, kpi_card, count_sales_card, selection_total_card, leader_sales_card, biggest_increase_card, render_df

def render(ctx: dict):
    dfA=ctx["dfA"]; dfB=ctx["dfB"]; kA=ctx["kA"]; kB=ctx["kB"]; a_lbl=ctx["a_lbl"]; b_lbl=ctx["b_lbl"];
    def pct_change(cur, prev):
        if prev == 0: return np.nan if cur == 0 else np.inf
        return (cur-prev)/prev
    def kdelta(key: str):
        cur=float(kA.get(key,0.0)); prev=float(kB.get(key,0.0)); d=cur-prev; color = "#2e7d32" if d>0 else ("#c62828" if d<0 else "var(--text-color)")
        return f"<span class='delta-abs' style='color:{color}'>{money(d) if key in ('Sales','ASP') else f'{d:,.0f}'}</span><span class='delta-pct' style='color:{color}'>({pct_change(cur,prev):.1%})</span>"
    c1,c2,c3 = st.columns(3)
    with c1: kpi_card("Total Sales", money(kA["Sales"]), kdelta("Sales"))
    with c2: kpi_card("Total Units", f"{kA['Units']:,.0f}", kdelta("Units"))
    with c3: kpi_card("Avg Selling Price", money(kA["ASP"]), kdelta("ASP"))
    st.subheader("Multi Compare")
    st.info("Each high-level tab now lives in its own module. This multi-compare tab can be iterated without affecting Standard or Month / Year Compare.")
    pivot_dim = st.selectbox("Compare rows by", options=["Retailer","Vendor"], index=0, key="multi_compare_dim_v3")
    comp = dfA.groupby(pivot_dim, as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum"))
    comp["Sales"] = comp["Sales"].map(money); comp["Units"] = comp["Units"].map(lambda v: f"{float(v):,.0f}")
    render_df(comp, height=360)
