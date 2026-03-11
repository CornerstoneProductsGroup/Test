
from __future__ import annotations
import calendar
import numpy as np
import pandas as pd
import streamlit as st
from .shared_core import money, render_df, calc_kpis

TIMEFRAME_OPTIONS = [
    "Last 4 weeks", "Last 8 weeks", "Last 13 weeks", "Last 26 weeks", "Last 52 weeks", "YTD", "All history"
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
    out = df[(pd.to_datetime(df["WeekEnd"], errors="coerce") >= start) & (pd.to_datetime(df["WeekEnd"], errors="coerce") <= end)].copy()
    return (start, end), out


def _fmt_num(v, metric: str):
    return money(v) if metric == "Sales" else f"{float(v):,.0f}"


def _weekly_pivot(df: pd.DataFrame, row_dim: str, metric: str) -> pd.DataFrame:
    d = df.groupby([row_dim, "WeekEnd"], as_index=False).agg(Value=(metric, "sum"))
    d["Week"] = pd.to_datetime(d["WeekEnd"]).dt.date.astype(str)
    piv = d.pivot_table(index=row_dim, columns="Week", values="Value", aggfunc="sum", fill_value=0.0)
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

    # Month seasonality: normalize by active week count in each month
    d["MonthNum"] = d["WeekEnd"].dt.month
    d["MonthLabel"] = d["WeekEnd"].dt.month.map(lambda m: calendar.month_abbr[int(m)])
    month = d.groupby("MonthNum", as_index=False).agg(
        Total=(metric, "sum"),
        ActiveWeeks=("WeekEnd", "nunique")
    )
    month["AvgWeekly"] = np.where(month["ActiveWeeks"] != 0, month["Total"] / month["ActiveWeeks"], 0.0)
    all_months = pd.DataFrame({"MonthNum": list(range(1, 13))})
    month = all_months.merge(month, on="MonthNum", how="left").fillna(0.0)
    month["Month"] = month["MonthNum"].map(lambda m: calendar.month_abbr[int(m)])
    max_month = float(month["AvgWeekly"].max()) if not month.empty else 0.0

    def _bar(v, vmax, width=10):
        if vmax <= 0 or v <= 0:
            return ""
        n = max(1, int(round((float(v) / float(vmax)) * width)))
        return "▓" * n

    month["Visual"] = month["AvgWeekly"].map(lambda v: _bar(v, max_month, 10))
    month["Value"] = month["AvgWeekly"].map(lambda v: _fmt_num(v, metric))
    month_show = month[["Month", "Visual", "Value"]].copy()

    peak_row = month.sort_values("AvgWeekly", ascending=False).iloc[0] if max_month > 0 else None
    nz_month = month[month["AvgWeekly"] > 0]
    low_row = nz_month.sort_values("AvgWeekly", ascending=True).iloc[0] if not nz_month.empty else None

    nz = month["AvgWeekly"][month["AvgWeekly"] > 0]
    if len(nz) >= 2:
        ratio = float(nz.max() / nz.mean()) if float(nz.mean()) > 0 else 0.0
        strength = "High" if ratio >= 1.6 else ("Medium" if ratio >= 1.25 else "Low")
    else:
        strength = "Low"

    # Week-of-year seasonality: 4 quarter-like buckets
    d["WeekOfYear"] = d["WeekEnd"].dt.isocalendar().week.astype(int)
    bins = [0, 13, 26, 39, 53]
    labels = ["Week 1–13", "Week 14–26", "Week 27–39", "Week 40–52"]
    d["WeekBucket"] = pd.cut(d["WeekOfYear"], bins=bins, labels=labels, include_lowest=True, right=True)
    woy = d.groupby("WeekBucket", as_index=False, observed=False).agg(
        Total=(metric, "sum"),
        ActiveWeeks=("WeekEnd", "nunique")
    )
    all_woy = pd.DataFrame({"WeekBucket": labels})
    woy = all_woy.merge(woy, on="WeekBucket", how="left").fillna(0.0)
    woy["AvgWeekly"] = np.where(woy["ActiveWeeks"] != 0, woy["Total"] / woy["ActiveWeeks"], 0.0)
    max_woy = float(woy["AvgWeekly"].max()) if not woy.empty else 0.0
    woy["Visual"] = woy["AvgWeekly"].map(lambda v: _bar(v, max_woy, 10))
    woy["Value"] = woy["AvgWeekly"].map(lambda v: _fmt_num(v, metric))
    woy_show = woy.rename(columns={"WeekBucket": "Period"})[["Period", "Visual", "Value"]].copy()

    peak_txt = f"{peak_row['Month']} • {_fmt_num(peak_row['AvgWeekly'], metric)}" if peak_row is not None and peak_row["AvgWeekly"] > 0 else "—"
    low_txt = f"{low_row['Month']} • {_fmt_num(low_row['AvgWeekly'], metric)}" if low_row is not None else "—"
    return month_show, woy_show, peak_txt, low_txt, strength


def _render_seasonality_section(df_sel: pd.DataFrame, metric: str, title: str = "Seasonality"):
    st.markdown(f"### {title}")
    month_show, woy_show, peak_txt, low_txt, strength = _seasonality_tables(df_sel, metric)
    if month_show.empty:
        st.caption("No seasonality history available.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Peak Month", peak_txt)
    with c2:
        st.metric("Low Month", low_txt)
    with c3:
        st.metric("Seasonality Strength", strength)

    left, right = st.columns(2)
    with left:
        st.markdown("**Month Seasonality**")
        render_df(month_show, height=430)
    with right:
        st.markdown("**Week-of-Year Seasonality**")
        render_df(woy_show, height=260)


def _render_single_sku(df_sel: pd.DataFrame, metric: str, lookup_value: str):
    st.markdown("### Quick Intelligence Summary")
    k = calc_kpis(df_sel)
    wk = df_sel.groupby("WeekEnd", as_index=False).agg(Sales=("Sales", "sum"), Units=("Units", "sum"))
    best_week = None
    worst_nonzero = None
    if not wk.empty:
        best_row = wk.sort_values(metric, ascending=False).iloc[0]
        best_week = f"{pd.to_datetime(best_row['WeekEnd']).date()} • {_fmt_num(best_row[metric], metric)}"
        nonzero = wk[wk[metric] > 0]
        if not nonzero.empty:
            low_row = nonzero.sort_values(metric, ascending=True).iloc[0]
            worst_nonzero = f"{pd.to_datetime(low_row['WeekEnd']).date()} • {_fmt_num(low_row[metric], metric)}"
    top_retailer = "—"
    if not df_sel.empty:
        rr = df_sel.groupby("Retailer", as_index=False).agg(Sales=("Sales", "sum"), Units=("Units", "sum"))
        if not rr.empty:
            tr = rr.sort_values(metric, ascending=False).iloc[0]
            top_retailer = f"{tr['Retailer']} • {_fmt_num(tr[metric], metric)}"
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.metric("Total Sales", money(k["Sales"]))
    with c2: st.metric("Total Units", f"{k['Units']:,.0f}")
    with c3: st.metric("ASP", money(k["ASP"]))
    with c4: st.metric("Active Retailers", f"{df_sel.loc[df_sel['Sales'] > 0, 'Retailer'].nunique():,}")
    with c5: st.metric("Top Retailer", top_retailer)
    with c6: st.metric("Best Week", best_week or "—")
    if worst_nonzero:
        st.caption(f"Lowest non-zero week: {worst_nonzero}")

    st.markdown("### Retailer Breakdown")
    rb = df_sel.groupby("Retailer", as_index=False).agg(
        Sales=("Sales", "sum"), Units=("Units", "sum"),
        FirstWeek=("WeekEnd", "min"), LastWeek=("WeekEnd", "max"), ActiveWeeks=("WeekEnd", "nunique")
    )
    if rb.empty:
        st.caption("No activity for this SKU in the selected timeframe.")
    else:
        rb["ASP"] = np.where(rb["Units"] != 0, rb["Sales"] / rb["Units"], 0.0)
        total_metric = rb[metric].sum()
        rb["Share %"] = np.where(total_metric != 0, rb[metric] / total_metric, 0.0)
        rb["First Week"] = pd.to_datetime(rb["FirstWeek"]).dt.date.astype(str)
        rb["Last Week"] = pd.to_datetime(rb["LastWeek"]).dt.date.astype(str)
        show = rb[["Retailer", "Sales", "Units", "ASP", "Share %", "First Week", "Last Week", "ActiveWeeks"]].copy()
        show["Sales"] = show["Sales"].map(money)
        show["Units"] = show["Units"].map(lambda v: f"{v:,.0f}")
        show["ASP"] = show["ASP"].map(money)
        show["Share %"] = show["Share %"].map(lambda v: f"{v*100:,.1f}%")
        show = show.rename(columns={"ActiveWeeks": "Active Weeks"})
        render_df(show, height=340)

    st.markdown("### Weekly Velocity Table")
    piv = _weekly_pivot(df_sel, "Retailer", metric)
    if piv.empty:
        st.caption("No weekly activity available.")
    else:
        week_cols = [c for c in piv.columns if c not in ["Retailer", "Average", "Current", "Vs Avg", "Active Weeks"]]
        for c in week_cols + ["Average", "Current", "Vs Avg"]:
            if c in piv.columns:
                piv[c] = piv[c].map(lambda v: _fmt_num(v, metric))
        if "Active Weeks" in piv.columns:
            piv["Active Weeks"] = piv["Active Weeks"].map(lambda v: f"{v:,.0f}")
        render_df(piv, height=420)

    _render_seasonality_section(df_sel, metric, title="Seasonality")

    st.markdown("### Retailer Contribution Trend")
    ct = df_sel.groupby(["WeekEnd", "Retailer"], as_index=False).agg(Value=(metric, "sum"))
    if ct.empty:
        st.caption("No contribution trend available.")
    else:
        ct["Week"] = pd.to_datetime(ct["WeekEnd"]).dt.date.astype(str)
        totals = ct.groupby("Week", as_index=False).agg(Total=("Value", "sum"))
        ct = ct.merge(totals, on="Week", how="left")
        ct["Share %"] = np.where(ct["Total"] != 0, ct["Value"] / ct["Total"], 0.0)
        piv = ct.pivot_table(index="Retailer", columns="Week", values="Share %", aggfunc="sum", fill_value=0.0).reset_index()
        for c in piv.columns[1:]:
            piv[c] = piv[c].map(lambda v: f"{v*100:,.1f}%")
        render_df(piv, height=300)


def _render_vendor(df_sel: pd.DataFrame, metric: str, lookup_value: str):
    st.markdown("### Vendor Intelligence Summary")
    k = calc_kpis(df_sel)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Sales", money(k["Sales"]))
    with c2: st.metric("Total Units", f"{k['Units']:,.0f}")
    with c3: st.metric("ASP", money(k["ASP"]))
    with c4: st.metric("Active SKUs", f"{df_sel.loc[df_sel['Sales']>0,'SKU'].nunique():,}")
    with c5: st.metric("Retailers", f"{df_sel.loc[df_sel['Sales']>0,'Retailer'].nunique():,}")

    st.markdown("### Top SKUs")
    sku = df_sel.groupby("SKU", as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum"), Weeks=("WeekEnd","nunique")).sort_values(metric, ascending=False)
    if not sku.empty:
        sku["ASP"] = np.where(sku["Units"] != 0, sku["Sales"] / sku["Units"], 0.0)
        sku["Sales"] = sku["Sales"].map(money)
        sku["Units"] = sku["Units"].map(lambda v: f"{v:,.0f}")
        sku["ASP"] = sku["ASP"].map(money)
        render_df(sku.head(50), height=520)
    else:
        st.caption("No SKU activity found.")

    st.markdown("### Retailer Breakdown")
    ret = df_sel.groupby("Retailer", as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum"), SKUs=("SKU","nunique")).sort_values(metric, ascending=False)
    if not ret.empty:
        ret["Share %"] = np.where(ret[metric].sum()!=0, ret[metric]/ret[metric].sum(), 0.0)
        ret["Sales"] = ret["Sales"].map(money)
        ret["Units"] = ret["Units"].map(lambda v: f"{v:,.0f}")
        ret["Share %"] = ret["Share %"].map(lambda v: f"{v*100:,.1f}%")
        render_df(ret, height=320)
    else:
        st.caption("No retailer activity found.")

    st.markdown("### Weekly Velocity Table")
    piv = _weekly_pivot(df_sel, "SKU", metric)
    if not piv.empty:
        for c in piv.columns[1:]:
            if c == "Active Weeks":
                piv[c] = piv[c].map(lambda v: f"{v:,.0f}")
            else:
                piv[c] = piv[c].map(lambda v: _fmt_num(v, metric))
        render_df(piv.head(60), height=420)
    else:
        st.caption("No weekly activity available.")

    _render_seasonality_section(df_sel, metric, title="Seasonality")


def _render_retailer(df_sel: pd.DataFrame, metric: str, lookup_value: str):
    st.markdown("### Retailer Intelligence Summary")
    k = calc_kpis(df_sel)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Sales", money(k["Sales"]))
    with c2: st.metric("Total Units", f"{k['Units']:,.0f}")
    with c3: st.metric("ASP", money(k["ASP"]))
    with c4: st.metric("Active SKUs", f"{df_sel.loc[df_sel['Sales']>0,'SKU'].nunique():,}")
    with c5: st.metric("Vendors", f"{df_sel.loc[df_sel['Sales']>0,'Vendor'].nunique():,}")

    st.markdown("### Vendor Breakdown")
    vend = df_sel.groupby("Vendor", as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum"), SKUs=("SKU","nunique")).sort_values(metric, ascending=False)
    if not vend.empty:
        vend["Share %"] = np.where(vend[metric].sum()!=0, vend[metric]/vend[metric].sum(), 0.0)
        vend["Sales"] = vend["Sales"].map(money)
        vend["Units"] = vend["Units"].map(lambda v: f"{v:,.0f}")
        vend["Share %"] = vend["Share %"].map(lambda v: f"{v*100:,.1f}%")
        render_df(vend, height=320)
    else:
        st.caption("No vendor activity found.")

    st.markdown("### Top SKUs")
    sku = df_sel.groupby("SKU", as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum"), Weeks=("WeekEnd","nunique")).sort_values(metric, ascending=False)
    if not sku.empty:
        sku["ASP"] = np.where(sku["Units"] != 0, sku["Sales"] / sku["Units"], 0.0)
        sku["Sales"] = sku["Sales"].map(money)
        sku["Units"] = sku["Units"].map(lambda v: f"{v:,.0f}")
        sku["ASP"] = sku["ASP"].map(money)
        render_df(sku.head(50), height=520)
    else:
        st.caption("No SKU activity found.")

    st.markdown("### Weekly Velocity Table")
    piv = _weekly_pivot(df_sel, "Vendor", metric)
    if not piv.empty:
        for c in piv.columns[1:]:
            if c == "Active Weeks":
                piv[c] = piv[c].map(lambda v: f"{v:,.0f}")
            else:
                piv[c] = piv[c].map(lambda v: _fmt_num(v, metric))
        render_df(piv, height=420)
    else:
        st.caption("No weekly activity available.")

    _render_seasonality_section(df_sel, metric, title="Seasonality")


def render(ctx: dict):
    df_scope = ctx["df_scope"].copy()
    st.header("Lookup Center")
    st.caption("Deep lookup by SKU, Vendor, or Retailer with timeframe-specific detail tables.")
    if df_scope.empty:
        st.info("No data available with the current sidebar filters.")
        return

    c1, c2, c3, c4 = st.columns([1.0, 2.2, 1.1, 1.0])
    with c1:
        lookup_type = st.selectbox("Lookup type", ["SKU", "Vendor", "Retailer"], index=0, key="lookup_center_type")
    if lookup_type == "SKU":
        options = sorted(df_scope["SKU"].dropna().astype(str).unique().tolist())
    elif lookup_type == "Vendor":
        options = sorted(df_scope["Vendor"].dropna().astype(str).unique().tolist())
    else:
        options = sorted(df_scope["Retailer"].dropna().astype(str).unique().tolist())
    with c2:
        lookup_value = st.selectbox(lookup_type, options=options, index=0 if options else None, key="lookup_center_value")
    with c3:
        timeframe = st.selectbox("Timeframe", TIMEFRAME_OPTIONS, index=1, key="lookup_center_timeframe")
    with c4:
        metric = st.selectbox("Metric", ["Sales", "Units"], index=0, key="lookup_center_metric")

    if not options:
        st.info("No lookup values available with the current filters.")
        return

    if lookup_type == "SKU":
        df_sel = df_scope[df_scope["SKU"] == lookup_value].copy()
    elif lookup_type == "Vendor":
        df_sel = df_scope[df_scope["Vendor"] == lookup_value].copy()
    else:
        df_sel = df_scope[df_scope["Retailer"] == lookup_value].copy()

    period, df_sel = _pick_lookup_period(df_sel, timeframe)
    if period is None or df_sel.empty:
        st.info("No data for that lookup in the selected timeframe.")
        return

    st.markdown(
        f"<div style='padding:8px 12px;border:1px solid rgba(128,128,128,0.22);border-radius:12px;background:var(--secondary-background-color);display:inline-block;font-weight:700;'>"
        f"{lookup_type}: {lookup_value} &nbsp; • &nbsp; {timeframe} &nbsp; • &nbsp; {period[0].date()} → {period[1].date()} &nbsp; • &nbsp; Metric: {metric}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    if lookup_type == "SKU":
        _render_single_sku(df_sel, metric, lookup_value)
    elif lookup_type == "Vendor":
        _render_vendor(df_sel, metric, lookup_value)
    else:
        _render_retailer(df_sel, metric, lookup_value)
