from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from .shared_core import money, pct_fmt, kpi_card, leader_sales_card, biggest_increase_card, rename_ab_columns, render_df, build_momentum, lifecycle_table, opportunity_detector, drivers, first_sale_ever, new_placement

def render(ctx: dict):
    dfA=ctx["dfA"]; dfB=ctx["dfB"]; kA=ctx["kA"]; kB=ctx["kB"]; a_lbl=ctx["a_lbl"]; b_lbl=ctx["b_lbl"]; compare_mode=ctx["compare_mode"]; min_sales=ctx["min_sales"]; min_units=ctx["min_units"]; df_scope=ctx["df_scope"]; pA=ctx["pA"]; df_hist_for_new=ctx["df_hist_for_new"]; show_full_history_lifecycle=ctx["show_full_history_lifecycle"]; driver_level=ctx["driver_level"];
    def pct_change(cur, prev):
        if prev == 0: return np.nan if cur == 0 else np.inf
        return (cur-prev)/prev
    def _delta_html(cur: float, prev: float, is_money: bool):
        d = cur - prev; color = "#2e7d32" if d > 0 else ("#c62828" if d < 0 else "var(--text-color)"); arrow = '▲ ' if d>0 else ('▼ ' if d<0 else '')
        abs_s = money(d) if is_money else (f"{d:,.0f}" if abs(d)>=1 else f"{d:,.2f}")
        return f"<span class='delta-abs' style='color:{color}'>{arrow}{abs_s}</span><span class='delta-pct' style='color:{color}'>({pct_fmt(pct_change(cur,prev))})</span>"
    def kdelta(key: str):
        cur=float(kA.get(key,0.0)); prev=float(kB.get(key,0.0)); return _delta_html(cur, prev, is_money=(key in ("Sales","ASP")))
    def _top_by_current(level: str):
        a = dfA.groupby(level, as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum")); b = dfB.groupby(level, as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum"))
        m = a.merge(b, on=level, how="outer", suffixes=("_A","_B")).fillna(0.0)
        if m.empty: return None
        row = m.sort_values("Sales_A", ascending=False).iloc[0]
        return str(row[level]), float(row["Sales_A"]), float(row["Sales_B"])
    def _top_by_increase(level: str):
        a = dfA.groupby(level, as_index=False).agg(Sales=("Sales","sum")); b = dfB.groupby(level, as_index=False).agg(Sales=("Sales","sum"))
        m = a.merge(b, on=level, how="outer", suffixes=("_A","_B")).fillna(0.0)
        if m.empty: return None
        m["Δ"] = m["Sales_A"] - m["Sales_B"]; row = m.sort_values("Δ", ascending=False).iloc[0]
        return str(row[level]), float(row["Sales_A"]), float(row["Sales_B"])

    first_ever = first_sale_ever(df_hist_for_new, pA); placements = new_placement(df_hist_for_new, pA)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi_card("Total Sales", money(kA["Sales"]), kdelta("Sales"))
    with c2: kpi_card("Total Units", f"{kA['Units']:,.0f}", kdelta("Units"))
    with c3: kpi_card("Avg Selling Price", money(kA["ASP"]), kdelta("ASP"))
    with c4: kpi_card("Active SKUs", f"{kA['Active SKUs']:,}", kdelta("Active SKUs"))
    with c5: kpi_card("First Sales", f"{len(first_ever):,}", "")
    with c6: kpi_card("New Placements", f"{len(placements):,}", "")
    st.write("")
    r1c1,r1c2,r1c3 = st.columns(3)
    tR=_top_by_current("Retailer"); tV=_top_by_current("Vendor"); tS=_top_by_current("SKU")
    with r1c1:
        if tR: leader_sales_card("Top Retailer (Sales)", tR[0], tR[1], tR[2])
    with r1c2:
        if tV: leader_sales_card("Top Vendor (Sales)", tV[0], tV[1], tV[2])
    with r1c3:
        if tS: leader_sales_card("Top SKU (Sales)", tS[0], tS[1], tS[2])
    r2c1,r2c2,r2c3 = st.columns(3)
    iR=_top_by_increase("Retailer"); iV=_top_by_increase("Vendor"); iS=_top_by_increase("SKU")
    with r2c1:
        if iR: biggest_increase_card("Retailer w/ Biggest Increase", iR[0], iR[1], iR[2])
    with r2c2:
        if iV: biggest_increase_card("Vendor w/ Biggest Increase", iV[0], iV[1], iV[2])
    with r2c3:
        if iS: biggest_increase_card("SKU w/ Biggest Increase", iS[0], iS[1], iS[2])

    st.write("")
    st.subheader("Drivers (Contribution to change)")
    if compare_mode == "None":
        st.info("Select a comparison mode to compute drivers.")
    else:
        drv = drivers(dfA, dfB, driver_level)
        drv_show = drv.copy(); drv_show = drv_show[(drv_show["Sales_A"] >= min_sales) | (drv_show["Sales_B"] >= min_sales)]
        pos = drv_show[drv_show["Sales_Δ"] > 0].head(10).copy(); neg = drv_show[drv_show["Sales_Δ"] < 0].sort_values("Sales_Δ").head(10).copy()
        for d in (pos, neg):
            d["Sales_A"] = d["Sales_A"].map(money); d["Sales_B"] = d["Sales_B"].map(money); d["Sales_Δ"] = d["Sales_Δ"].map(lambda v: f"{money(v)}"); d["Contribution_%"] = d["Contribution_%"].map(pct_fmt)
        left,right = st.columns(2)
        pos_disp = rename_ab_columns(pos, a_lbl, b_lbl); neg_disp = rename_ab_columns(neg, a_lbl, b_lbl)
        sales_a_col = f"Sales ({a_lbl})"; sales_b_col = f"Sales ({b_lbl})" if b_lbl else "Sales (Comparison)"
        with left: st.markdown("**Top Positive Contributors**"); render_df(pos_disp[[driver_level, sales_a_col, sales_b_col, "Sales_Δ", "Contribution_%"]], height=320)
        with right: st.markdown("**Top Negative Contributors**"); render_df(neg_disp[[driver_level, sales_a_col, sales_b_col, "Sales_Δ", "Contribution_%"]], height=320)

    st.divider(); st.subheader("New Activity")
    a,b = st.columns(2)
    with a:
        st.markdown("**First Sale Ever (Launches)**")
        if first_ever.empty: st.caption("None in this period.")
        else:
            fe = first_ever.copy(); fe["FirstWeek"] = fe["FirstWeek"].dt.date.astype(str); render_df(fe.rename(columns={"FirstWeek":"First Week"})[["SKU","First Week","FirstRetailer","FirstVendor"]], height=260)
    with b:
        st.markdown("**New Retailer Placements**")
        if placements.empty: st.caption("None in this period.")
        else:
            pl = placements.copy(); pl["FirstWeek"] = pl["FirstWeek"].dt.date.astype(str); render_df(pl.rename(columns={"FirstWeek":"First Week"})[["SKU","Retailer","Vendor","First Week"]], height=260)
    st.divider(); st.subheader("Weekly Detail (Retailer/Vendor x Weeks)")
    d = dfA.copy(); d = d[(d["Sales"] >= min_sales) | (d["Units"] >= min_units)].copy()
    if d.empty: st.caption("No rows match the current thresholds.")
    else:
        pivot_dim = st.selectbox("Pivot rows by", options=["Retailer","Vendor"], index=0, key="std_weekly_pivot_dim")
        wk_sales = d.groupby([pivot_dim,"WeekEnd"], as_index=False).agg(Sales=("Sales","sum")); wk_sales["WeekEnd"] = pd.to_datetime(wk_sales["WeekEnd"], errors="coerce")
        weeks = sorted([pd.to_datetime(x) for x in wk_sales["WeekEnd"].dropna().unique().tolist()]); wk_sales["Week"] = wk_sales["WeekEnd"].dt.date.astype(str)
        piv = wk_sales.pivot_table(index=pivot_dim, columns="Week", values="Sales", aggfunc="sum", fill_value=0.0).reindex(sorted(wk_sales[pivot_dim].dropna().unique().tolist()))
        if len(weeks) >= 2: w_last = str(weeks[-1].date()); w_prev = str(weeks[-2].date()); piv["Δ vs prior week"] = piv.get(w_last, 0.0) - piv.get(w_prev, 0.0)
        else: piv["Δ vs prior week"] = 0.0
        piv_disp = piv.copy()
        for c in piv_disp.columns: piv_disp[c] = piv_disp[c].map(lambda x: ("+" if x > 0 and c=="Δ vs prior week" else "") + money(x))
        render_df(piv_disp.reset_index(), height=320)
    st.subheader("Movers & Trend Leaders")
    if compare_mode == "None": st.info("Select a comparison mode to compute increasing/declining vs the compare period.")
    else:
        a = dfA.groupby("SKU", as_index=False).agg(Sales_A=("Sales","sum"), Units_A=("Units","sum")); b = dfB.groupby("SKU", as_index=False).agg(Sales_B=("Sales","sum"), Units_B=("Units","sum"))
        m = a.merge(b, on="SKU", how="outer").fillna(0.0); m["Sales (Current)"] = m["Sales_A"]; m["Sales (Compare)"] = m["Sales_B"]; m["Sales Δ"] = m["Sales_A"] - m["Sales_B"]; m["Δ %"] = np.where(m["Sales_B"] != 0, m["Sales Δ"] / m["Sales_B"], np.nan)
        m = m[(m["Sales_A"] >= min_sales) | (m["Sales_B"] >= min_sales) | (m["Units_A"] >= min_units) | (m["Units_B"] >= min_units)].copy()
        inc = m[m["Sales Δ"] > 0].sort_values("Sales Δ", ascending=False).head(10); dec = m[m["Sales Δ"] < 0].sort_values("Sales Δ", ascending=True).head(10)
        def _disp(df_in: pd.DataFrame) -> pd.DataFrame:
            if df_in.empty: return df_in
            out = df_in[["SKU","Sales (Current)","Sales (Compare)","Sales Δ","Δ %"]].copy(); out["Sales (Current)"] = out["Sales (Current)"].map(money); out["Sales (Compare)"] = out["Sales (Compare)"].map(money); out["Sales Δ"] = out["Sales Δ"].map(money); out["Δ %"] = out["Δ %"].map(pct_fmt); return out
        inc_disp = _disp(inc); dec_disp = _disp(dec)
        mom = build_momentum(df_scope[df_scope["WeekEnd"] <= pA.end], "SKU", lookback_weeks=8)
        trend_leaders_disp = mom.sort_values("Slope", ascending=False).head(10)[["SKU","Trend","Slope","Weeks Up","Weeks Down","Sales (lookback)"]].copy() if not mom.empty else pd.DataFrame(columns=["SKU","Trend","Slope","Weeks Up","Weeks Down","Sales (lookback)"])
        if not trend_leaders_disp.empty: trend_leaders_disp["Sales (lookback)"] = trend_leaders_disp["Sales (lookback)"].map(money); trend_leaders_disp["Slope"] = trend_leaders_disp["Slope"].map(lambda v: f"{v:,.2f}")
        a,b,c = st.columns(3)
        with a: st.markdown("**Top Increasing**"); render_df(inc_disp, height=320) if not inc_disp.empty else st.caption("None.")
        with b: st.markdown("**Top Declining**"); render_df(dec_disp, height=320) if not dec_disp.empty else st.caption("None.")
        with c: st.markdown("**Trend Leaders (slope over last 8 weeks)**"); render_df(trend_leaders_disp, height=320)

    st.divider(); st.header("Strategic Intelligence")
    st.subheader("1) Contribution Tree (Where did change come from?)")
    if compare_mode == "None": st.info("Select a comparison mode to use the contribution tree.")
    else:
        sales_a_col = f"Sales ({a_lbl})"; sales_b_col = f"Sales ({b_lbl})" if b_lbl else "Sales (Comparison)"
        lvl1 = drivers(dfA, dfB, "Retailer").sort_values("Sales_Δ", ascending=False)
        st.markdown("**Level 1 — Retailers**")
        lvl1_disp = lvl1[["Retailer","Sales_A","Sales_B","Sales_Δ","Contribution_%"]].copy(); lvl1_disp["Sales_A"] = lvl1_disp["Sales_A"].map(money); lvl1_disp["Sales_B"] = lvl1_disp["Sales_B"].map(money); lvl1_disp["Sales_Δ"] = lvl1_disp["Sales_Δ"].map(money); lvl1_disp["Contribution_%"] = lvl1_disp["Contribution_%"].map(pct_fmt); lvl1_disp = rename_ab_columns(lvl1_disp, a_lbl, b_lbl); render_df(lvl1_disp[["Retailer", sales_a_col, sales_b_col, "Sales_Δ", "Contribution_%"]], height=260)
    st.subheader("2) SKU Lifecycle (Launch → Growth → Mature → Decline → Dormant)")
    life_df_src = df_hist_for_new if show_full_history_lifecycle else df_scope
    life = lifecycle_table(life_df_src, pA, lookback_weeks=8)
    if life.empty: st.caption("Not enough data to compute lifecycle.")
    else:
        stage_counts = life["Stage"].value_counts().reset_index(); stage_counts.columns = ["Stage","Count"]
        left,right = st.columns([1,2])
        with left: st.markdown("**Stage Summary**"); render_df(stage_counts, height=220)
        with right:
            st.markdown("**Lifecycle Detail (trend + momentum + recent change)**")
            life_show = life.copy()
            if min_sales > 0: life_show = life_show[life_show["Sales (lookback)"] >= min_sales].copy()
            for c in ["Sales (lookback)","Last Week Sales","WoW Sales Δ"]:
                if c in life_show.columns: life_show[c] = life_show[c].map(lambda v: "" if pd.isna(v) else money(v))
            if "Units (lookback)" in life_show.columns: life_show["Units (lookback)"] = life_show["Units (lookback)"].map(lambda v: f"{v:,.0f}")
            if "Slope" in life_show.columns: life_show["Slope"] = life_show["Slope"].map(lambda v: f"{v:,.2f}")
            cols=[c for c in ["SKU","Stage","Trend","Momentum","Sales (lookback)","Units (lookback)","Last Week Sales","WoW Sales Δ","Slope","Weeks Up","Weeks Down","First Sale","Last Sale"] if c in life_show.columns]
            render_df(life_show[cols].head(60), height=520)
    st.divider(); st.subheader("3) Opportunity Detector (Find expansion + gaps)")
    if compare_mode == "None": st.info("Select a comparison mode to power opportunity signals (needs a comparison).")
    else:
        opp = opportunity_detector(df_hist_for_new, dfA, dfB, pA)
        tabs = st.tabs(list(opp.keys()))
        for t, (name, odf) in zip(tabs, opp.items()):
            with t: render_df(odf, height=420) if not odf.empty else st.caption("No signals found with current filters/thresholds.")
