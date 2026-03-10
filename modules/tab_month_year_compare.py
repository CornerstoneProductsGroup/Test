from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from .shared_core import money, pct_fmt, rename_ab_columns, render_df, count_sales_card, kpi_card, selection_total_card, top_two_card, biggest_increase_card

def render(ctx: dict):
    st.markdown("""
    <style>
    /* Month / Year Compare only */
    .kpi-card .kpi-title{font-size:13px !important;}
    .kpi-card .kpi-value{font-size:31px !important;}
    .kpi-card .kpi-delta{font-size:15px !important;}
    .kpi-card .kpi-sub{font-size:15px !important;}
    .kpi-card .top-two-item .kpi-big-name{font-size:22px !important;}
    .kpi-card .top-two-item .kpi-value{font-size:30px !important;}
    .kpi-card .top-two-item .kpi-delta{font-size:14px !important;}
    .kpi-card .top-two-item .kpi-sub{font-size:14px !important;}
    </style>
    """, unsafe_allow_html=True)

    dfA=ctx["dfA"]; dfB=ctx["dfB"]; kA=ctx["kA"]; kB=ctx["kB"]; a_lbl=ctx["a_lbl"]; b_lbl=ctx["b_lbl"]; compare_mode=ctx["compare_mode"]; min_sales=ctx["min_sales"];

    def render_shaded_total_table(df: pd.DataFrame, height: int = 760):
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)

    def pct_change(cur, prev):
        if prev == 0: return np.nan if cur == 0 else np.inf
        return (cur-prev)/prev
    def _delta_html(cur: float, prev: float, is_money: bool):
        d = cur - prev; pc = pct_change(cur, prev); color = "#2e7d32" if d > 0 else ("#c62828" if d < 0 else "var(--text-color)")
        arrow = '▲ ' if d>0 else ('▼ ' if d<0 else '')
        abs_s = money(d) if is_money else (f"{d:,.0f}" if abs(d)>=1 else f"{d:,.2f}")
        return f"<span class='delta-abs' style='color:{color}'>{arrow}{abs_s}</span><span class='delta-pct' style='color:{color}'>({pct_fmt(pc)})</span>"
    def kdelta(key: str) -> str:
        cur=float(kA.get(key,0.0)); prev=float(kB.get(key,0.0))
        return _delta_html(cur, prev, is_money=(key in ("Sales","ASP")))
    def _top_by_increase(level: str):
        a = dfA.groupby(level, as_index=False).agg(Sales_A=("Sales","sum"))
        b = dfB.groupby(level, as_index=False).agg(Sales_B=("Sales","sum"))
        m = a.merge(b, on=level, how="outer").fillna(0.0); m["Δ"] = m["Sales_A"] - m["Sales_B"]
        if m.empty: return None
        r = m.sort_values("Δ", ascending=False).iloc[0]
        return str(r[level]), float(r["Sales_A"]), float(r["Sales_B"])
    def _top_decrease(level: str):
        a = dfA.groupby(level, as_index=False).agg(Sales_A=("Sales","sum"))
        b = dfB.groupby(level, as_index=False).agg(Sales_B=("Sales","sum"))
        m = a.merge(b, on=level, how="outer").fillna(0.0); m["Δ"] = m["Sales_A"] - m["Sales_B"]
        if m.empty: return None
        r = m.sort_values("Δ", ascending=True).iloc[0]
        return str(r[level]), float(r["Sales_A"]), float(r["Sales_B"])
    def _top_two_with_compare(df_sel: pd.DataFrame, df_other: pd.DataFrame, level: str):
        if df_sel.empty: return []
        cur = df_sel.groupby(level, as_index=False).agg(Sales=("Sales", "sum"), Units=("Units", "sum"))
        oth = df_other.groupby(level, as_index=False).agg(Other_Sales=("Sales", "sum"), Other_Units=("Units", "sum")) if not df_other.empty else pd.DataFrame(columns=[level, "Other_Sales", "Other_Units"])
        m = cur.merge(oth, on=level, how="left").fillna(0.0)
        total_sales = float(m["Sales"].sum()); total_units = float(m["Units"].sum())
        out=[]
        for _, r in m.sort_values(["Sales", level], ascending=[False, True]).head(2).iterrows():
            sales = float(r["Sales"]); units = float(r["Units"])
            out.append({"name":str(r[level]),"sales":sales,"other_sales":float(r["Other_Sales"]),"share":(sales/total_sales) if total_sales else np.nan,"units":units,"other_units":float(r["Other_Units"]),"unit_share":(units/total_units) if total_units else np.nan})
        return out

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Total Sales", money(kA["Sales"]), kdelta("Sales"))
    with c2: kpi_card("Total Units", f"{kA['Units']:,.0f}", kdelta("Units"))
    with c3: kpi_card("Avg Selling Price", money(kA["ASP"]), kdelta("ASP"))

    cur_sku = dfA.groupby("SKU", as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum"))
    cmp_sku = dfB.groupby("SKU", as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum"))
    cur_only = cur_sku.merge(cmp_sku[["SKU","Sales"]].rename(columns={"Sales":"Compare_Sales"}), on="SKU", how="left").fillna(0.0)
    cur_only = cur_only[(cur_only["Sales"] > 0) & (cur_only["Compare_Sales"] <= 0)].copy()
    cmp_only = cmp_sku.merge(cur_sku[["SKU","Sales"]].rename(columns={"Sales":"Current_Sales"}), on="SKU", how="left").fillna(0.0)
    cmp_only = cmp_only[(cmp_only["Sales"] > 0) & (cmp_only["Current_Sales"] <= 0)].copy()
    new_count = int(len(cur_only)); new_sales = float(cur_only["Sales"].sum())
    lost_count = int(len(cmp_only)); lost_sales = float(cmp_only["Sales"].sum())
    net_count = new_count - lost_count; net_sales = new_sales - lost_sales
    net_pct = (net_sales / lost_sales) if lost_sales != 0 else (np.nan if net_sales == 0 else np.inf)
    n1, n2, n3 = st.columns(3)
    with n1: count_sales_card("New SKUs", new_count, new_sales, color="#2e7d32", signed_sales=True)
    with n2: count_sales_card("Lost SKUs", lost_count, -lost_sales, color="#c62828", signed_sales=True)
    with n3: count_sales_card("Net New vs Lost", net_count, net_sales, color=("#2e7d32" if net_sales > 0 else ("#c62828" if net_sales < 0 else "var(--text-color)")), signed_sales=True, pct=net_pct)

    st.write("")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        selection_total_card(f"{a_lbl} Total", kA, kB); st.write(""); selection_total_card(f"{b_lbl} Total", kB, kA)
    with g2:
        top_two_card(f"Top 2 Retailers ({a_lbl})", _top_two_with_compare(dfA, dfB, "Retailer")); st.write(""); top_two_card(f"Top 2 Retailers ({b_lbl})", _top_two_with_compare(dfB, dfA, "Retailer"))
    with g3:
        top_two_card(f"Top 2 Vendors ({a_lbl})", _top_two_with_compare(dfA, dfB, "Vendor")); st.write(""); top_two_card(f"Top 2 Vendors ({b_lbl})", _top_two_with_compare(dfB, dfA, "Vendor"))
    with g4:
        top_two_card(f"Top 2 SKUs ({a_lbl})", _top_two_with_compare(dfA, dfB, "SKU")); st.write(""); top_two_card(f"Top 2 SKUs ({b_lbl})", _top_two_with_compare(dfB, dfA, "SKU"))

    st.write("")
    i1, i2, i3 = st.columns(3)
    iR = _top_by_increase("Retailer"); iV = _top_by_increase("Vendor"); iS = _top_by_increase("SKU")
    with i1:
        if iR: biggest_increase_card("Retailer w/ Biggest Increase", iR[0], iR[1], iR[2])
    with i2:
        if iV: biggest_increase_card("Vendor w/ Biggest Increase", iV[0], iV[1], iV[2])
    with i3:
        if iS: biggest_increase_card("SKU w/ Biggest Increase", iS[0], iS[1], iS[2])
    d1, d2, d3 = st.columns(3)
    decR = _top_decrease("Retailer"); decV = _top_decrease("Vendor"); decS = _top_decrease("SKU")
    with d1:
        if decR: biggest_increase_card("Retailer w/ Biggest Decrease", decR[0], decR[1], decR[2])
    with d2:
        if decV: biggest_increase_card("Vendor w/ Biggest Decrease", decV[0], decV[1], decV[2])
    with d3:
        if decS: biggest_increase_card("SKU w/ Biggest Decrease", decS[0], decS[1], decS[2])

    st.divider()
    st.subheader("Current Only / Compare Only Activity")
    cur_s = dfA.groupby("SKU", as_index=False).agg(Current_Units=("Units","sum"), Current_Sales=("Sales","sum"))
    cmp_s = dfB.groupby("SKU", as_index=False).agg(Compare_Units=("Units","sum"), Compare_Sales=("Sales","sum"))
    lost = cmp_s.merge(cur_s, on="SKU", how="left").fillna(0.0)
    lost = lost[(lost["Compare_Sales"] > 0) & (lost["Current_Sales"] <= 0)].copy().sort_values("Compare_Sales", ascending=False)
    new_act = cur_s.merge(cmp_s, on="SKU", how="left").fillna(0.0)
    new_act = new_act[(new_act["Current_Sales"] > 0) & (new_act["Compare_Sales"] <= 0)].copy().sort_values("Current_Sales", ascending=False)
    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("**Lost Activity — sold in compare, zero in current**")
        if lost.empty: st.caption("None.")
        else:
            show_lost = lost[["SKU","Compare_Units","Compare_Sales"]].rename(columns={"Compare_Units":"Units","Compare_Sales":"Sales"}).copy()
            show_lost["Units"] = -show_lost["Units"]; show_lost["Sales"] = -show_lost["Sales"]
            total_row = pd.DataFrame([{"SKU":"Total","Units":show_lost["Units"].sum(),"Sales":show_lost["Sales"].sum()}])
            show_lost = pd.concat([show_lost, total_row], ignore_index=True)
            show_lost["Units"] = show_lost["Units"].map(lambda v: f"{v:,.0f}"); show_lost["Sales"] = show_lost["Sales"].map(money)
            render_df(show_lost, height=360)
    with rcol:
        st.markdown("**New Activity — sold in current, zero in compare**")
        if new_act.empty: st.caption("None.")
        else:
            show_new = new_act[["SKU","Current_Units","Current_Sales"]].rename(columns={"Current_Units":"Units","Current_Sales":"Sales"}).copy()
            total_row = pd.DataFrame([{"SKU":"Total","Units":show_new["Units"].sum(),"Sales":show_new["Sales"].sum()}])
            show_new = pd.concat([show_new, total_row], ignore_index=True)
            show_new["Units"] = show_new["Units"].map(lambda v: f"{v:,.0f}"); show_new["Sales"] = show_new["Sales"].map(money)
            render_df(show_new, height=360)

    st.divider(); st.subheader("Comparison Detail")
    pivot_dim = st.selectbox("Compare rows by", options=["Retailer","Vendor"], index=0, key="mod_compare_dim")
    comp_a = dfA.groupby(pivot_dim, as_index=False).agg(Sales_A=("Sales","sum"))
    comp_b = dfB.groupby(pivot_dim, as_index=False).agg(Sales_B=("Sales","sum"))
    comp = comp_a.merge(comp_b, on=pivot_dim, how="outer").fillna(0.0)
    comp["Difference"] = comp["Sales_A"] - comp["Sales_B"]
    comp["% Change"] = np.where(comp["Sales_B"] != 0, comp["Difference"] / comp["Sales_B"], np.nan)
    comp = comp.sort_values("Sales_A", ascending=False)
    total = pd.DataFrame([{pivot_dim:"Total","Sales_A":comp["Sales_A"].sum(),"Sales_B":comp["Sales_B"].sum(),"Difference":comp["Difference"].sum(),"% Change": np.nan if comp["Sales_B"].sum()==0 else comp["Difference"].sum()/comp["Sales_B"].sum()}])
    comp_show = pd.concat([comp, total], ignore_index=True)
    show = rename_ab_columns(comp_show.copy(), a_lbl, b_lbl)
    sales_a_col = f"Sales ({a_lbl})"; sales_b_col = f"Sales ({b_lbl})" if b_lbl else "Sales (Comparison)"
    show[sales_a_col] = show[sales_a_col].map(money); show[sales_b_col] = show[sales_b_col].map(money); show["Difference"] = show["Difference"].map(money); show["% Change"] = show["% Change"].map(pct_fmt)
    render_shaded_total_table(show[[pivot_dim, sales_a_col, sales_b_col, "Difference", "% Change"]], height=900)

    st.divider(); st.subheader("Movers")
    a = dfA.groupby("SKU", as_index=False).agg(Sales_A=("Sales","sum"))
    b = dfB.groupby("SKU", as_index=False).agg(Sales_B=("Sales","sum"))
    m = a.merge(b, on="SKU", how="outer").fillna(0.0)
    m["Difference"] = m["Sales_A"] - m["Sales_B"]; m["% Change"] = np.where(m["Sales_B"] != 0, m["Difference"] / m["Sales_B"], np.nan)
    m = m[(m["Sales_A"] >= min_sales) | (m["Sales_B"] >= min_sales)].copy()
    inc = m[m["Difference"] > 0].sort_values("Difference", ascending=False).head(15).copy()
    dec = m[m["Difference"] < 0].sort_values("Difference", ascending=True).head(15).copy()
    for ddf in (inc, dec):
        ddf.rename(columns={"Sales_A": f"Sales ({a_lbl})", "Sales_B": f"Sales ({b_lbl})"}, inplace=True)
        ddf[f"Sales ({a_lbl})"] = ddf[f"Sales ({a_lbl})"].map(money); ddf[f"Sales ({b_lbl})"] = ddf[f"Sales ({b_lbl})"].map(money); ddf["Difference"] = ddf["Difference"].map(money); ddf["% Change"] = ddf["% Change"].map(pct_fmt)
    x, y = st.columns(2)
    with x:
        st.markdown("**Top Increasing**"); render_df(inc[["SKU", f"Sales ({a_lbl})", f"Sales ({b_lbl})", "Difference", "% Change"]], height=360) if not inc.empty else st.caption("None.")
    with y:
        st.markdown("**Top Declining**"); render_df(dec[["SKU", f"Sales ({a_lbl})", f"Sales ({b_lbl})", "Difference", "% Change"]], height=360) if not dec.empty else st.caption("None.")
