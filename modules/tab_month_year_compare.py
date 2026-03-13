from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .shared_core import (
    money,
    pct_fmt,
    rename_ab_columns,
    render_df,
    count_sales_card,
    kpi_card,
    selection_total_card,
    top_two_card,
    biggest_increase_card,
)


def render(ctx: dict):
    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )

    dfA = ctx["dfA"]
    dfB = ctx["dfB"]
    kA = ctx["kA"]
    kB = ctx["kB"]
    a_lbl = ctx["a_lbl"]
    b_lbl = ctx["b_lbl"]
    compare_mode = ctx["compare_mode"]
    min_sales = ctx["min_sales"]

    view_mode = st.toggle(
        "Visual executive dashboard",
        value=False,
        key="month_year_compare_visual_toggle",
        help="Turn on for a cleaner chart-first executive dashboard. Turn off for the detailed table view.",
    )

    if view_mode:
        render_visual_executive_dashboard(
            dfA=dfA,
            dfB=dfB,
            kA=kA,
            kB=kB,
            a_lbl=a_lbl,
            b_lbl=b_lbl,
            min_sales=min_sales,
        )
        return

    render_standard_view(
        dfA=dfA,
        dfB=dfB,
        kA=kA,
        kB=kB,
        a_lbl=a_lbl,
        b_lbl=b_lbl,
        compare_mode=compare_mode,
        min_sales=min_sales,
    )


def render_visual_executive_dashboard(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    kA: dict,
    kB: dict,
    a_lbl: str,
    b_lbl: str,
    min_sales: float,
):
    PERIOD_DOMAIN = [a_lbl, b_lbl]
    PERIOD_RANGE = ["#1f77b4", "#ff7f0e"]

    Q_DOMAIN = ["Q1", "Q2", "Q3", "Q4"]
    Q_RANGE = ["#dbe7f5", "#a8c4e5", "#6f9fd1", "#2f6fb3"]

    POSITIVE_BAR = "#2e7d32"
    NEGATIVE_BAR = "#c62828"

    def pct_change(cur: float, prev: float):
        if prev == 0:
            return np.nan if cur == 0 else np.inf
        return (cur - prev) / prev

    def delta_html(cur: float, prev: float, is_money: bool):
        d = float(cur) - float(prev)
        pc = pct_change(float(cur), float(prev))
        color = "#2e7d32" if d > 0 else ("#c62828" if d < 0 else "var(--text-color)")
        arrow = "▲ " if d > 0 else ("▼ " if d < 0 else "")
        abs_s = money(d) if is_money else f"{d:,.0f}"
        return (
            f"<span class='delta-abs' style='color:{color}'>{arrow}{abs_s}</span>"
            f"<span class='delta-pct' style='color:{color}'>({pct_fmt(pc)})</span>"
        )

    def prep_compare_metric(
        df_cur: pd.DataFrame,
        df_cmp: pd.DataFrame,
        level: str,
        metric: str = "Sales",
        top_n: int = 10,
    ):
        cur = df_cur.groupby(level, as_index=False).agg(Current=(metric, "sum"))
        cmp = df_cmp.groupby(level, as_index=False).agg(Compare=(metric, "sum"))
        out = cur.merge(cmp, on=level, how="outer").fillna(0.0)
        out["Delta"] = out["Current"] - out["Compare"]
        out["Total"] = out["Current"] + out["Compare"]
        out = out.sort_values(["Total", level], ascending=[False, True]).head(top_n).copy()
        return out

    def infer_quarter_series(df: pd.DataFrame) -> pd.Series | None:
        cols = {str(c).strip().lower(): c for c in df.columns}

        for cand in ["quarter", "qtr", "fiscal quarter"]:
            if cand in cols:
                s = df[cols[cand]].astype(str).str.upper().str.replace(" ", "", regex=False)
                s = s.replace({"1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"})
                return s.where(s.isin(Q_DOMAIN))

        month_num = None

        for cand in ["monthnum", "month_num", "month number", "month"]:
            if cand in cols:
                raw = df[cols[cand]]
                if pd.api.types.is_numeric_dtype(raw):
                    month_num = pd.to_numeric(raw, errors="coerce")
                    break
                month_map = {
                    "jan": 1, "january": 1,
                    "feb": 2, "february": 2,
                    "mar": 3, "march": 3,
                    "apr": 4, "april": 4,
                    "may": 5,
                    "jun": 6, "june": 6,
                    "jul": 7, "july": 7,
                    "aug": 8, "august": 8,
                    "sep": 9, "sept": 9, "september": 9,
                    "oct": 10, "october": 10,
                    "nov": 11, "november": 11,
                    "dec": 12, "december": 12,
                }
                tmp = raw.astype(str).str.strip().str.lower().map(month_map)
                if tmp.notna().any():
                    month_num = pd.to_numeric(tmp, errors="coerce")
                    break

        if month_num is None:
            for cand in [
                "date", "week ending", "week_end", "week end date",
                "week ending date", "invoice date", "ship date"
            ]:
                if cand in cols:
                    dt = pd.to_datetime(df[cols[cand]], errors="coerce")
                    if dt.notna().any():
                        month_num = dt.dt.month
                        break

        if month_num is None:
            return None

        q = (((month_num - 1) // 3) + 1).map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
        return q

    def prep_quarter_stacked(df_cur: pd.DataFrame, df_cmp: pd.DataFrame, metric: str):
        cur_q = infer_quarter_series(df_cur)
        cmp_q = infer_quarter_series(df_cmp)

        if cur_q is None or cmp_q is None:
            return pd.DataFrame()

        cur = df_cur.copy()
        cmp = df_cmp.copy()
        cur["_Quarter_"] = cur_q
        cmp["_Quarter_"] = cmp_q

        cur = cur[cur["_Quarter_"].isin(Q_DOMAIN)]
        cmp = cmp[cmp["_Quarter_"].isin(Q_DOMAIN)]

        a = cur.groupby("_Quarter_", as_index=False)[metric].sum()
        b = cmp.groupby("_Quarter_", as_index=False)[metric].sum()

        a["Period"] = a_lbl
        b["Period"] = b_lbl
        a.rename(columns={metric: "Value", "_Quarter_": "Quarter"}, inplace=True)
        b.rename(columns={metric: "Value", "_Quarter_": "Quarter"}, inplace=True)

        out = pd.concat([a, b], ignore_index=True)
        if out.empty:
            return out

        out["Quarter"] = pd.Categorical(out["Quarter"], categories=Q_DOMAIN, ordered=True)
        out = out.sort_values(["Period", "Quarter"]).copy()
        out["Label"] = out["Value"].map(lambda v: f"{v:,.0f}" if metric == "Units" else money(v))
        return out

    def stacked_total_chart(metric_name: str, df_cur: pd.DataFrame, df_cmp: pd.DataFrame, fallback_cur: float, fallback_cmp: float):
        metric = "Units" if metric_name == "Units" else "Sales"
        stacked = prep_quarter_stacked(df_cur, df_cmp, metric)

        if stacked.empty:
            chart_df = pd.DataFrame(
                [
                    {"Period": a_lbl, "Value": float(fallback_cur)},
                    {"Period": b_lbl, "Value": float(fallback_cmp)},
                ]
            )

            xmax = float(chart_df["Value"].max()) if not chart_df.empty else 0.0
            xmax = xmax * 1.20 if xmax > 0 else 1.0

            color_enc = alt.Color(
                "Period:N",
                scale=alt.Scale(domain=PERIOD_DOMAIN, range=PERIOD_RANGE),
                legend=None,
            )

            bars = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                .encode(
                    y=alt.Y("Period:N", title="", sort=PERIOD_DOMAIN),
                    x=alt.X("Value:Q", title=metric_name, scale=alt.Scale(domain=[0, xmax])),
                    color=color_enc,
                )
                .properties(height=160)
            )

            labels = (
                alt.Chart(chart_df)
                .mark_text(dx=8, align="left", fontSize=14, fontWeight="bold")
                .encode(
                    y=alt.Y("Period:N", sort=PERIOD_DOMAIN),
                    x=alt.X("Value:Q", scale=alt.Scale(domain=[0, xmax])),
                    text=alt.Text("Value:Q", format=",.0f" if metric == "Units" else ",.2f"),
                    color=color_enc,
                )
            )

            return bars + labels

        totals = stacked.groupby("Period", as_index=False)["Value"].sum().rename(columns={"Value": "Total"})
        stacked = stacked.merge(totals, on="Period", how="left")
        stacked["Quarter"] = pd.Categorical(stacked["Quarter"], categories=Q_DOMAIN, ordered=True)
        stacked = stacked.sort_values(["Period", "Quarter"]).copy()

        bars = (
            alt.Chart(stacked)
            .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
            .encode(
                y=alt.Y("Period:N", title="", sort=PERIOD_DOMAIN),
                x=alt.X("Value:Q", title=metric_name),
                color=alt.Color(
                    "Quarter:N",
                    scale=alt.Scale(domain=Q_DOMAIN, range=Q_RANGE),
                    legend=alt.Legend(orient="top", direction="horizontal", title=""),
                ),
                order=alt.Order("Quarter:N", sort="ascending"),
                tooltip=[
                    alt.Tooltip("Period:N", title="Period"),
                    alt.Tooltip("Quarter:N", title="Quarter"),
                    alt.Tooltip("Value:Q", title=metric_name, format=",.0f" if metric == "Units" else ",.2f"),
                ],
            )
            .properties(height=165)
        )

        text_df = stacked.copy()
        text_df["ShowLabel"] = np.where(text_df["Value"] > 0, text_df["Label"], "")

        labels = (
            alt.Chart(text_df[text_df["ShowLabel"] != ""])
            .mark_text(fontSize=13, fontWeight="bold")
            .encode(
                y=alt.Y("Period:N", sort=PERIOD_DOMAIN),
                x=alt.X("Value:Q", stack="center"),
                detail="Quarter:N",
                text="ShowLabel:N",
                color=alt.value("#111111"),
            )
        )

        return bars + labels

    def prep_grouped_share(df: pd.DataFrame, dim_name: str):
        if df.empty:
            return pd.DataFrame(columns=[dim_name, "Series", "Value", "SharePct", "Label", "SortTotal"])

        long_df = df[[dim_name, "Current", "Compare", "Total"]].melt(
            id_vars=[dim_name, "Total"],
            value_vars=["Current", "Compare"],
            var_name="Series",
            value_name="Value",
        )

        long_df["Series"] = long_df["Series"].replace({"Current": a_lbl, "Compare": b_lbl})
        long_df["SharePct"] = np.where(long_df["Total"] > 0, long_df["Value"] / long_df["Total"], 0.0)
        long_df["Label"] = long_df.apply(
            lambda r: f'{money(r["Value"])} • {r["SharePct"]:.0%}' if r["Value"] > 0 else "",
            axis=1,
        )
        long_df["SortTotal"] = long_df["Total"]
        return long_df

    def grouped_lollipop_chart(long_df: pd.DataFrame, dim_name: str, height: int = 470):
        if long_df.empty:
            return None

        xmax = float(long_df["Value"].max()) if not long_df.empty else 0.0
        xmax = xmax * 1.34 if xmax > 0 else 1.0

        color_enc = alt.Color(
            "Series:N",
            scale=alt.Scale(domain=PERIOD_DOMAIN, range=PERIOD_RANGE),
            title="",
            legend=alt.Legend(orient="bottom", direction="horizontal"),
        )

        text_color_enc = alt.Color(
            "Series:N",
            scale=alt.Scale(domain=PERIOD_DOMAIN, range=PERIOD_RANGE),
            legend=None,
        )

        y_enc = alt.Y(
            f"{dim_name}:N",
            sort=alt.SortField(field="SortTotal", order="descending"),
            title="",
            scale=alt.Scale(paddingInner=0.38, paddingOuter=0.18),
        )

        yoff_enc = alt.YOffset(
            "Series:N",
            sort=[a_lbl, b_lbl],
            scale=alt.Scale(paddingInner=0.32),
        )

        rules = (
            alt.Chart(long_df)
            .mark_rule(strokeWidth=2)
            .encode(
                y=y_enc,
                yOffset=yoff_enc,
                x=alt.value(0),
                x2=alt.X2("Value:Q"),
                color=color_enc,
            )
        )

        dots = (
            alt.Chart(long_df)
            .mark_circle(size=120)
            .encode(
                y=y_enc,
                yOffset=yoff_enc,
                x=alt.X("Value:Q", title="Sales", scale=alt.Scale(domain=[0, xmax], nice=True)),
                color=color_enc,
                tooltip=[
                    alt.Tooltip(f"{dim_name}:N", title=dim_name),
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("Value:Q", title="Sales", format=",.2f"),
                    alt.Tooltip("SharePct:Q", title="% of row total", format=".0%"),
                ],
            )
        )

        labels = (
            alt.Chart(long_df[long_df["Label"] != ""])
            .mark_text(align="left", dx=8, baseline="middle", fontSize=13, fontWeight="bold")
            .encode(
                y=y_enc,
                yOffset=yoff_enc,
                x=alt.X("Value:Q", scale=alt.Scale(domain=[0, xmax], nice=True)),
                text="Label:N",
                color=text_color_enc,
            )
        )

        return (rules + dots + labels).properties(height=height)

    def prep_sku_movers():
        sA = dfA.groupby("SKU", as_index=False).agg(Current=("Sales", "sum"))
        sB = dfB.groupby("SKU", as_index=False).agg(Compare=("Sales", "sum"))
        sku = sA.merge(sB, on="SKU", how="outer").fillna(0.0)
        sku["Delta"] = sku["Current"] - sku["Compare"]
        sku = sku[(sku["Current"] >= min_sales) | (sku["Compare"] >= min_sales)].copy()

        inc = sku.sort_values(["Delta", "SKU"], ascending=[False, True]).head(10).copy()
        dec = sku.sort_values(["Delta", "SKU"], ascending=[True, True]).head(10).copy()
        return inc, dec

    def mover_lollipop_chart(df: pd.DataFrame, metric_title: str, positive: bool, height: int = 360):
        if df.empty:
            return None

        df = df.copy()
        df["DeltaLabel"] = df["Delta"].map(money)

        xmax = float(df["Delta"].max()) if positive else float(df["Delta"].abs().max())
        xmax = xmax * 1.34 if xmax > 0 else 1.0

        if positive:
            order = alt.SortField(field="Delta", order="descending")
            x_scale = alt.Scale(domain=[0, xmax], nice=True)
            color = POSITIVE_BAR
            align = "left"
            dx = 8
        else:
            order = alt.SortField(field="Delta", order="ascending")
            x_scale = alt.Scale(domain=[-xmax, 0], nice=True)
            color = NEGATIVE_BAR
            align = "right"
            dx = -8

        rules = (
            alt.Chart(df)
            .mark_rule(strokeWidth=2, color=color)
            .encode(
                y=alt.Y("SKU:N", sort=order, title=""),
                x=alt.value(0),
                x2=alt.X2("Delta:Q"),
            )
        )

        dots = (
            alt.Chart(df)
            .mark_circle(size=120, color=color)
            .encode(
                y=alt.Y("SKU:N", sort=order, title=""),
                x=alt.X("Delta:Q", title=metric_title, scale=x_scale),
                tooltip=[
                    alt.Tooltip("SKU:N", title="SKU"),
                    alt.Tooltip("Current:Q", title=a_lbl, format=",.2f"),
                    alt.Tooltip("Compare:Q", title=b_lbl, format=",.2f"),
                    alt.Tooltip("Delta:Q", title="Change", format=",.2f"),
                ],
            )
        )

        labels = (
            alt.Chart(df)
            .mark_text(align=align, dx=dx, color=color, fontSize=13, fontWeight="bold")
            .encode(
                y=alt.Y("SKU:N", sort=order, title=""),
                x=alt.X("Delta:Q", scale=x_scale),
                text="DeltaLabel:N",
            )
        )

        return (rules + dots + labels).properties(height=height)

    def prep_contrib(df: pd.DataFrame):
        if df.empty:
            return pd.DataFrame(columns=["Retailer", "Current", "Compare", "Delta", "ContribPct", "PctLabel", "DeltaLabel", "BarColor"])

        out = df.copy().sort_values("Delta", ascending=False).copy()
        denom = float(out["Delta"].abs().sum())
        out["ContribPct"] = np.where(denom > 0, out["Delta"].abs() / denom, 0.0)
        out["PctLabel"] = (out["ContribPct"] * 100).round(0).astype(int).astype(str) + "%"
        out["DeltaLabel"] = out["Delta"].map(money)
        out["BarColor"] = np.where(out["Delta"] >= 0, POSITIVE_BAR, NEGATIVE_BAR)
        return out

    def contrib_lollipop_chart(df: pd.DataFrame, height: int = 470):
        if df.empty:
            return None

        xmax = float(df["Delta"].max()) if not df.empty else 0.0
        xmin = float(df["Delta"].min()) if not df.empty else 0.0
        pad = max(abs(xmax), abs(xmin)) * 0.46 if max(abs(xmax), abs(xmin)) > 0 else 1.0

        x_scale = alt.Scale(domain=[xmin - pad, xmax + pad], nice=True)
        order = alt.SortField(field="Delta", order="descending")

        rules = (
            alt.Chart(df)
            .mark_rule(strokeWidth=2)
            .encode(
                y=alt.Y("Retailer:N", sort=order, title=""),
                x=alt.value(0),
                x2=alt.X2("Delta:Q"),
                color=alt.Color("BarColor:N", scale=None, legend=None),
            )
        )

        dots = (
            alt.Chart(df)
            .mark_circle(size=120)
            .encode(
                y=alt.Y("Retailer:N", sort=order, title=""),
                x=alt.X("Delta:Q", title="Sales Delta", scale=x_scale),
                color=alt.Color("BarColor:N", scale=None, legend=None),
                tooltip=[
                    alt.Tooltip("Retailer:N", title="Retailer"),
                    alt.Tooltip("Current:Q", title=a_lbl, format=",.2f"),
                    alt.Tooltip("Compare:Q", title=b_lbl, format=",.2f"),
                    alt.Tooltip("Delta:Q", title="Delta", format=",.2f"),
                    alt.Tooltip("ContribPct:Q", title="% of total change", format=".0%"),
                ],
            )
        )

        pos_df = df[df["Delta"] >= 0].copy()
        neg_df = df[df["Delta"] < 0].copy()

        pos_delta_labels = (
            alt.Chart(pos_df)
            .mark_text(dx=8, align="left", fontSize=13, fontWeight="bold")
            .encode(
                y=alt.Y("Retailer:N", sort=order, title=""),
                x=alt.X("Delta:Q", scale=x_scale),
                text="DeltaLabel:N",
                color=alt.Color("BarColor:N", scale=None, legend=None),
            )
        )

        pos_pct_labels = (
            alt.Chart(pos_df)
            .mark_text(dx=8, dy=14, align="left", fontSize=12, fontWeight="bold")
            .encode(
                y=alt.Y("Retailer:N", sort=order, title=""),
                x=alt.X("Delta:Q", scale=x_scale),
                text="PctLabel:N",
                color=alt.Color("BarColor:N", scale=None, legend=None),
            )
        )

        neg_delta_labels = (
            alt.Chart(neg_df)
            .mark_text(dx=-8, align="right", fontSize=13, fontWeight="bold")
            .encode(
                y=alt.Y("Retailer:N", sort=order, title=""),
                x=alt.X("Delta:Q", scale=x_scale),
                text="DeltaLabel:N",
                color=alt.Color("BarColor:N", scale=None, legend=None),
            )
        )

        neg_pct_labels = (
            alt.Chart(neg_df)
            .mark_text(dx=-8, dy=14, align="right", fontSize=12, fontWeight="bold")
            .encode(
                y=alt.Y("Retailer:N", sort=order, title=""),
                x=alt.X("Delta:Q", scale=x_scale),
                text="PctLabel:N",
                color=alt.Color("BarColor:N", scale=None, legend=None),
            )
        )

        return (rules + dots + pos_delta_labels + pos_pct_labels + neg_delta_labels + neg_pct_labels).properties(height=height)

    st.markdown("### Executive Dashboard")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Total Sales", money(kA["Sales"]), delta_html(kA["Sales"], kB["Sales"], True))
    with c2:
        kpi_card("Total Units", f"{kA['Units']:,.0f}", delta_html(kA["Units"], kB["Units"], False))
    with c3:
        kpi_card("Avg Selling Price", money(kA["ASP"]), delta_html(kA["ASP"], kB["ASP"], True))

    st.write("")

    sales_col, units_col = st.columns(2)

    with sales_col:
        st.markdown(f"#### Sales Totals ({a_lbl} vs {b_lbl})")
        sales_chart = stacked_total_chart(
            metric_name="Sales",
            df_cur=dfA,
            df_cmp=dfB,
            fallback_cur=float(kA["Sales"]),
            fallback_cmp=float(kB["Sales"]),
        )
        st.altair_chart(sales_chart, use_container_width=True)

    with units_col:
        st.markdown(f"#### Units Totals ({a_lbl} vs {b_lbl})")
        units_chart = stacked_total_chart(
            metric_name="Units",
            df_cur=dfA,
            df_cmp=dfB,
            fallback_cur=float(kA["Units"]),
            fallback_cmp=float(kB["Units"]),
        )
        st.altair_chart(units_chart, use_container_width=True)

    st.write("")

    driver_r = prep_compare_metric(dfA, dfB, "Retailer", metric="Sales", top_n=12)
    driver_r = prep_contrib(driver_r)

    if not driver_r.empty:
        st.markdown("#### Retailer Contribution to Change")
        rc_chart = contrib_lollipop_chart(driver_r, height=470)
        st.altair_chart(rc_chart, use_container_width=True)

    retailer = prep_compare_metric(dfA, dfB, "Retailer", metric="Sales", top_n=10)
    vendor = prep_compare_metric(dfA, dfB, "Vendor", metric="Sales", top_n=10)

    left, right = st.columns(2)

    with left:
        st.markdown("#### Top Retailers")
        retailer_long = prep_grouped_share(retailer, "Retailer")
        if retailer_long.empty:
            st.caption("No retailer data available.")
        else:
            retailer_chart = grouped_lollipop_chart(retailer_long, "Retailer", height=470)
            st.altair_chart(retailer_chart, use_container_width=True)

    with right:
        st.markdown("#### Top Vendors")
        vendor_long = prep_grouped_share(vendor, "Vendor")
        if vendor_long.empty:
            st.caption("No vendor data available.")
        else:
            vendor_chart = grouped_lollipop_chart(vendor_long, "Vendor", height=470)
            st.altair_chart(vendor_chart, use_container_width=True)

    st.write("")

    inc, dec = prep_sku_movers()

    left2, right2 = st.columns(2)

    with left2:
        st.markdown("#### Top SKU Increases")
        if inc.empty:
            st.caption("No increasing SKUs found.")
        else:
            inc_chart = mover_lollipop_chart(inc, "Sales Change", positive=True, height=360)
            st.altair_chart(inc_chart, use_container_width=True)

    with right2:
        st.markdown("#### Top SKU Decreases")
        if dec.empty:
            st.caption("No declining SKUs found.")
        else:
            dec_chart = mover_lollipop_chart(dec, "Sales Change", positive=False, height=360)
            st.altair_chart(dec_chart, use_container_width=True)


def render_standard_view(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    kA: dict,
    kB: dict,
    a_lbl: str,
    b_lbl: str,
    compare_mode: str,
    min_sales: float,
):
    def render_shaded_total_table(df: pd.DataFrame, height: int = 760):
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)

    def pct_change(cur, prev):
        if prev == 0:
            return np.nan if cur == 0 else np.inf
        return (cur - prev) / prev

    def _delta_html(cur: float, prev: float, is_money: bool):
        d = cur - prev
        pc = pct_change(cur, prev)
        color = "#2e7d32" if d > 0 else ("#c62828" if d < 0 else "var(--text-color)")
        arrow = "▲ " if d > 0 else ("▼ " if d < 0 else "")
        abs_s = money(d) if is_money else (f"{d:,.0f}" if abs(d) >= 1 else f"{d:,.2f}")
        return (
            f"<span class='delta-abs' style='color:{color}'>{arrow}{abs_s}</span>"
            f"<span class='delta-pct' style='color:{color}'>({pct_fmt(pc)})</span>"
        )

    def kdelta(key: str) -> str:
        cur = float(kA.get(key, 0.0))
        prev = float(kB.get(key, 0.0))
        return _delta_html(cur, prev, is_money=(key in ("Sales", "ASP")))

    def _top_by_increase(level: str):
        a = dfA.groupby(level, as_index=False).agg(Sales_A=("Sales", "sum"))
        b = dfB.groupby(level, as_index=False).agg(Sales_B=("Sales", "sum"))
        m = a.merge(b, on=level, how="outer").fillna(0.0)
        m["Δ"] = m["Sales_A"] - m["Sales_B"]
        if m.empty:
            return None
        r = m.sort_values("Δ", ascending=False).iloc[0]
        return str(r[level]), float(r["Sales_A"]), float(r["Sales_B"])

    def _top_decrease(level: str):
        a = dfA.groupby(level, as_index=False).agg(Sales_A=("Sales", "sum"))
        b = dfB.groupby(level, as_index=False).agg(Sales_B=("Sales", "sum"))
        m = a.merge(b, on=level, how="outer").fillna(0.0)
        m["Δ"] = m["Sales_A"] - m["Sales_B"]
        if m.empty:
            return None
        r = m.sort_values("Δ", ascending=True).iloc[0]
        return str(r[level]), float(r["Sales_A"]), float(r["Sales_B"])

    def _top_two_with_compare(df_sel: pd.DataFrame, df_other: pd.DataFrame, level: str):
        if df_sel.empty:
            return []
        cur = df_sel.groupby(level, as_index=False).agg(Sales=("Sales", "sum"), Units=("Units", "sum"))
        if not df_other.empty:
            oth = df_other.groupby(level, as_index=False).agg(
                Other_Sales=("Sales", "sum"),
                Other_Units=("Units", "sum"),
            )
        else:
            oth = pd.DataFrame(columns=[level, "Other_Sales", "Other_Units"])
        m = cur.merge(oth, on=level, how="left").fillna(0.0)
        total_sales = float(m["Sales"].sum())
        total_units = float(m["Units"].sum())
        out = []
        for _, r in m.sort_values(["Sales", level], ascending=[False, True]).head(2).iterrows():
            sales = float(r["Sales"])
            units = float(r["Units"])
            out.append(
                {
                    "name": str(r[level]),
                    "sales": sales,
                    "other_sales": float(r["Other_Sales"]),
                    "share": (sales / total_sales) if total_sales else np.nan,
                    "units": units,
                    "other_units": float(r["Other_Units"]),
                    "unit_share": (units / total_units) if total_units else np.nan,
                }
            )
        return out

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Total Sales", money(kA["Sales"]), kdelta("Sales"))
    with c2:
        kpi_card("Total Units", f"{kA['Units']:,.0f}", kdelta("Units"))
    with c3:
        kpi_card("Avg Selling Price", money(kA["ASP"]), kdelta("ASP"))

    cur_sku = dfA.groupby("SKU", as_index=False).agg(Sales=("Sales", "sum"), Units=("Units", "sum"))
    cmp_sku = dfB.groupby("SKU", as_index=False).agg(Sales=("Sales", "sum"), Units=("Units", "sum"))

    cur_only = cur_sku.merge(
        cmp_sku[["SKU", "Sales"]].rename(columns={"Sales": "Compare_Sales"}),
        on="SKU",
        how="left",
    ).fillna(0.0)
    cur_only = cur_only[(cur_only["Sales"] > 0) & (cur_only["Compare_Sales"] <= 0)].copy()

    cmp_only = cmp_sku.merge(
        cur_sku[["SKU", "Sales"]].rename(columns={"Sales": "Current_Sales"}),
        on="SKU",
        how="left",
    ).fillna(0.0)
    cmp_only = cmp_only[(cmp_only["Sales"] > 0) & (cmp_only["Current_Sales"] <= 0)].copy()

    new_count = int(len(cur_only))
    new_sales = float(cur_only["Sales"].sum())
    lost_count = int(len(cmp_only))
    lost_sales = float(cmp_only["Sales"].sum())
    net_count = new_count - lost_count
    net_sales = new_sales - lost_sales
    net_pct = (net_sales / lost_sales) if lost_sales != 0 else (np.nan if net_sales == 0 else np.inf)

    n1, n2, n3 = st.columns(3)
    with n1:
        count_sales_card("New SKUs", new_count, new_sales, color="#2e7d32", signed_sales=True)
    with n2:
        count_sales_card("Lost SKUs", lost_count, -lost_sales, color="#c62828", signed_sales=True)
    with n3:
        count_sales_card(
            "Net New vs Lost",
            net_count,
            net_sales,
            color=("#2e7d32" if net_sales > 0 else ("#c62828" if net_sales < 0 else "var(--text-color)")),
            signed_sales=True,
            pct=net_pct,
        )

    st.write("")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        selection_total_card(f"{a_lbl} Total", kA, kB)
        st.write("")
        selection_total_card(f"{b_lbl} Total", kB, kA)
    with g2:
        top_two_card(f"Top 2 Retailers ({a_lbl})", _top_two_with_compare(dfA, dfB, "Retailer"))
        st.write("")
        top_two_card(f"Top 2 Retailers ({b_lbl})", _top_two_with_compare(dfB, dfA, "Retailer"))
    with g3:
        top_two_card(f"Top 2 Vendors ({a_lbl})", _top_two_with_compare(dfA, dfB, "Vendor"))
        st.write("")
        top_two_card(f"Top 2 Vendors ({b_lbl})", _top_two_with_compare(dfB, dfA, "Vendor"))
    with g4:
        top_two_card(f"Top 2 SKUs ({a_lbl})", _top_two_with_compare(dfA, dfB, "SKU"))
        st.write("")
        top_two_card(f"Top 2 SKUs ({b_lbl})", _top_two_with_compare(dfB, dfA, "SKU"))

    st.write("")
    i1, i2, i3 = st.columns(3)
    iR = _top_by_increase("Retailer")
    iV = _top_by_increase("Vendor")
    iS = _top_by_increase("SKU")
    with i1:
        if iR:
            biggest_increase_card("Retailer w/ Biggest Increase", iR[0], iR[1], iR[2])
    with i2:
        if iV:
            biggest_increase_card("Vendor w/ Biggest Increase", iV[0], iV[1], iV[2])
    with i3:
        if iS:
            biggest_increase_card("SKU w/ Biggest Increase", iS[0], iS[1], iS[2])

    d1, d2, d3 = st.columns(3)
    decR = _top_decrease("Retailer")
    decV = _top_decrease("Vendor")
    decS = _top_decrease("SKU")
    with d1:
        if decR:
            biggest_increase_card("Retailer w/ Biggest Decrease", decR[0], decR[1], decR[2])
    with d2:
        if decV:
            biggest_increase_card("Vendor w/ Biggest Decrease", decV[0], decV[1], decV[2])
    with d3:
        if decS:
            biggest_increase_card("SKU w/ Biggest Decrease", decS[0], decS[1], decS[2])

    st.divider()
    st.subheader("Current Only / Compare Only Activity")

    cur_s = dfA.groupby("SKU", as_index=False).agg(Current_Units=("Units", "sum"), Current_Sales=("Sales", "sum"))
    cmp_s = dfB.groupby("SKU", as_index=False).agg(Compare_Units=("Units", "sum"), Compare_Sales=("Sales", "sum"))

    lost = cmp_s.merge(cur_s, on="SKU", how="left").fillna(0.0)
    lost = lost[(lost["Compare_Sales"] > 0) & (lost["Current_Sales"] <= 0)].copy().sort_values(
        "Compare_Sales", ascending=False
    )

    new_act = cur_s.merge(cmp_s, on="SKU", how="left").fillna(0.0)
    new_act = new_act[(new_act["Current_Sales"] > 0) & (new_act["Compare_Sales"] <= 0)].copy().sort_values(
        "Current_Sales", ascending=False
    )

    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("**Lost Activity — sold in compare, zero in current**")
        if lost.empty:
            st.caption("None.")
        else:
            show_lost = lost[["SKU", "Compare_Units", "Compare_Sales"]].rename(
                columns={"Compare_Units": "Units", "Compare_Sales": "Sales"}
            ).copy()
            show_lost["Units"] = -show_lost["Units"]
            show_lost["Sales"] = -show_lost["Sales"]
            total_row = pd.DataFrame(
                [{"SKU": "Total", "Units": show_lost["Units"].sum(), "Sales": show_lost["Sales"].sum()}]
            )
            show_lost = pd.concat([show_lost, total_row], ignore_index=True)
            show_lost["Units"] = show_lost["Units"].map(lambda v: f"{v:,.0f}")
            show_lost["Sales"] = show_lost["Sales"].map(money)
            render_df(show_lost, height=360)

    with rcol:
        st.markdown("**New Activity — sold in current, zero in compare**")
        if new_act.empty:
            st.caption("None.")
        else:
            show_new = new_act[["SKU", "Current_Units", "Current_Sales"]].rename(
                columns={"Current_Units": "Units", "Current_Sales": "Sales"}
            ).copy()
            total_row = pd.DataFrame(
                [{"SKU": "Total", "Units": show_new["Units"].sum(), "Sales": show_new["Sales"].sum()}]
            )
            show_new = pd.concat([show_new, total_row], ignore_index=True)
            show_new["Units"] = show_new["Units"].map(lambda v: f"{v:,.0f}")
            show_new["Sales"] = show_new["Sales"].map(money)
            render_df(show_new, height=360)

    st.divider()
    st.subheader("Comparison Detail")

    pivot_dim = st.selectbox("Compare rows by", options=["Retailer", "Vendor"], index=0, key="mod_compare_dim")
    comp_a = dfA.groupby(pivot_dim, as_index=False).agg(Sales_A=("Sales", "sum"))
    comp_b = dfB.groupby(pivot_dim, as_index=False).agg(Sales_B=("Sales", "sum"))
    comp = comp_a.merge(comp_b, on=pivot_dim, how="outer").fillna(0.0)
    comp["Difference"] = comp["Sales_A"] - comp["Sales_B"]
    comp["% Change"] = np.where(comp["Sales_B"] != 0, comp["Difference"] / comp["Sales_B"], np.nan)
    comp = comp.sort_values("Sales_A", ascending=False)

    total = pd.DataFrame(
        [
            {
                pivot_dim: "Total",
                "Sales_A": comp["Sales_A"].sum(),
                "Sales_B": comp["Sales_B"].sum(),
                "Difference": comp["Difference"].sum(),
                "% Change": np.nan if comp["Sales_B"].sum() == 0 else comp["Difference"].sum() / comp["Sales_B"].sum(),
            }
        ]
    )
    comp_show = pd.concat([comp, total], ignore_index=True)
    show = rename_ab_columns(comp_show.copy(), a_lbl, b_lbl)
    sales_a_col = f"Sales ({a_lbl})"
    sales_b_col = f"Sales ({b_lbl})" if b_lbl else "Sales (Comparison)"
    show[sales_a_col] = show[sales_a_col].map(money)
    show[sales_b_col] = show[sales_b_col].map(money)
    show["Difference"] = show["Difference"].map(money)
    show["% Change"] = show["% Change"].map(pct_fmt)
    render_shaded_total_table(show[[pivot_dim, sales_a_col, sales_b_col, "Difference", "% Change"]], height=900)

    st.divider()
    st.subheader("Movers")

    a = dfA.groupby("SKU", as_index=False).agg(Sales_A=("Sales", "sum"))
    b = dfB.groupby("SKU", as_index=False).agg(Sales_B=("Sales", "sum"))
    m = a.merge(b, on="SKU", how="outer").fillna(0.0)
    m["Difference"] = m["Sales_A"] - m["Sales_B"]
    m["% Change"] = np.where(m["Sales_B"] != 0, m["Difference"] / m["Sales_B"], np.nan)
    m = m[(m["Sales_A"] >= min_sales) | (m["Sales_B"] >= min_sales)].copy()

    inc = m[m["Difference"] > 0].sort_values("Difference", ascending=False).head(15).copy()
    dec = m[m["Difference"] < 0].sort_values("Difference", ascending=True).head(15).copy()

    for ddf in (inc, dec):
        ddf.rename(columns={"Sales_A": f"Sales ({a_lbl})", "Sales_B": f"Sales ({b_lbl})"}, inplace=True)
        ddf[f"Sales ({a_lbl})"] = ddf[f"Sales ({a_lbl})"].map(money)
        ddf[f"Sales ({b_lbl})"] = ddf[f"Sales ({b_lbl})"].map(money)
        ddf["Difference"] = ddf["Difference"].map(money)
        ddf["% Change"] = ddf["% Change"].map(pct_fmt)

    x, y = st.columns(2)
    with x:
        st.markdown("**Top Increasing**")
        if not inc.empty:
            render_df(inc[["SKU", f"Sales ({a_lbl})", f"Sales ({b_lbl})", "Difference", "% Change"]], height=360)
        else:
            st.caption("None.")
    with y:
        st.markdown("**Top Declining**")
        if not dec.empty:
            render_df(dec[["SKU", f"Sales ({a_lbl})", f"Sales ({b_lbl})", "Difference", "% Change"]], height=360)
        else:
            st.caption("None.")
