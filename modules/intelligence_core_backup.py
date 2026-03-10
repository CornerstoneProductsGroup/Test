
from __future__ import annotations

import io
import html
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Reuse ingestion helpers from the current (legacy) app where possible.
from modules.app_core import read_weekly_workbook, parse_date_range_from_filename

APP_TITLE = "Cornerstone Sales Intelligence"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_VENDOR_MAP = DATA_DIR / "vendor_map.xlsx"
DEFAULT_STORE_CSV = DATA_DIR / "sales_store.csv"

# -----------------------------
# Normalization helpers
# -----------------------------
_RETAILER_ALIASES = {
    "home depot": "Depot",
    "the home depot": "Depot",
    "depot": "Depot",
    "lowe's": "Lowes",
    "lowes": "Lowes",
    "tractor supply": "Tractor Supply",
    "tsc": "Tractor Supply",
    "home depot canada": "Home Depot Canada",
    "ace": "Ace",
    "amazon": "Amazon",
    "walmart": "Walmart",
    "zoro": "Zoro",
    "orgill": "Orgill",
}

def norm_retailer(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    k = s.lower()
    return _RETAILER_ALIASES.get(k, s)

def norm_sku(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = s.replace(" ", "")
    return s

# -----------------------------
# Storage
# -----------------------------
BASE_COLUMNS = ["Retailer","Vendor","SKU","Units","Price","Sales","StartDate","EndDate","SourceFile"]

def load_store() -> pd.DataFrame:
    if DEFAULT_STORE_CSV.exists():
        df = pd.read_csv(DEFAULT_STORE_CSV)
    else:
        df = pd.DataFrame(columns=["Retailer","SKU","Units","UnitPrice","StartDate","EndDate","SourceFile"])
    # If it's legacy shape (no Vendor/Price/Sales), keep and enrich later.
    for c in ["StartDate","EndDate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "Retailer" in df.columns:
        df["Retailer"] = df["Retailer"].map(norm_retailer)
    if "SKU" in df.columns:
        df["SKU"] = df["SKU"].map(norm_sku)
    return df

def save_store(df: pd.DataFrame) -> None:
    # Persist in the legacy shape to stay compatible with the existing app if needed.
    keep = df.copy()
    # Ensure these exist
    for c in ["Retailer","SKU","Units","UnitPrice","StartDate","EndDate","SourceFile"]:
        if c not in keep.columns:
            keep[c] = np.nan
    keep = keep[["Retailer","SKU","Units","UnitPrice","StartDate","EndDate","SourceFile"]].copy()
    keep.to_csv(DEFAULT_STORE_CSV, index=False)

def load_vendor_map() -> pd.DataFrame:
    if not DEFAULT_VENDOR_MAP.exists():
        return pd.DataFrame(columns=["Retailer","SKU","Price","Vendor"])
    df = pd.read_excel(DEFAULT_VENDOR_MAP, sheet_name=0, engine="openpyxl")
    # Minimal standardization
    for c in ["Retailer","SKU","Vendor"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "Retailer" in df.columns:
        df["Retailer"] = df["Retailer"].map(norm_retailer)
    if "SKU" in df.columns:
        df["SKU"] = df["SKU"].map(norm_sku)
    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df[["Retailer","SKU","Price","Vendor"]].copy()

def enrich_sales(df_raw: pd.DataFrame, vm: pd.DataFrame) -> pd.DataFrame:
    """Return a fully-enriched fact table with Vendor, Price, Sales, and a weekly key."""
    df = df_raw.copy()
    # Ensure base columns exist
    for c in ["Retailer","SKU","Units","UnitPrice","StartDate","EndDate","SourceFile"]:
        if c not in df.columns:
            df[c] = np.nan
    df["Retailer"] = df["Retailer"].map(norm_retailer)
    df["SKU"] = df["SKU"].map(norm_sku)
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce").fillna(0.0)

    # Merge vendor map (Retailer+SKU)
    m = vm.copy()
    df = df.merge(m, on=["Retailer","SKU"], how="left", suffixes=("","_map"))
    # Use UnitPrice if present, else map Price
    df["UnitPrice"] = pd.to_numeric(df.get("UnitPrice"), errors="coerce")
    df["Price"] = np.where(df["UnitPrice"].notna(), df["UnitPrice"], df["Price"])
    df["Sales"] = df["Units"] * df["Price"].fillna(0.0)

    df["Vendor"] = df["Vendor"].fillna("Unknown").astype(str).str.strip()
    df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
    df["EndDate"] = pd.to_datetime(df["EndDate"], errors="coerce")
    # Weekly key: use EndDate; fallback to StartDate
    df["WeekEnd"] = df["EndDate"].fillna(df["StartDate"])
    df["WeekEnd"] = pd.to_datetime(df["WeekEnd"], errors="coerce")
    df["Year"] = df["WeekEnd"].dt.year
    # Display label like "2026-01-05 / 2026-01-09"
    df["WeekLabel"] = df.apply(
        lambda r: (f"{r['StartDate'].date()} / {r['EndDate'].date()}" if pd.notna(r["StartDate"]) and pd.notna(r["EndDate"]) else (str(r["WeekEnd"].date()) if pd.notna(r["WeekEnd"]) else "")),
        axis=1
    )
    return df

# -----------------------------
# Period selection
# -----------------------------
@dataclass
class Period:
    start: pd.Timestamp
    end: pd.Timestamp

def _safe_max_ts(s: pd.Series) -> Optional[pd.Timestamp]:
    s2 = pd.to_datetime(s, errors="coerce")
    s2 = s2.dropna()
    return None if s2.empty else s2.max()

def pick_period(df: pd.DataFrame, mode: str, n_weeks: int = 8) -> Optional[Period]:
    """Choose the current (A) period based on mode using df's max WeekEnd as anchor."""
    anchor = _safe_max_ts(df.get("WeekEnd", pd.Series(dtype="datetime64[ns]")))
    if anchor is None:
        return None

    if mode.startswith("Last "):
        # "Last 4 weeks", etc.
        weeks = int(re.findall(r"\d+", mode)[0])
        start = anchor - pd.Timedelta(days=(7*weeks - 1))
        return Period(start=start.normalize(), end=anchor.normalize())
    if mode == "Week (latest)":
        start = anchor - pd.Timedelta(days=6)
        return Period(start=start.normalize(), end=anchor.normalize())
    if mode == "This Month":
        start = pd.Timestamp(year=anchor.year, month=anchor.month, day=1)
        return Period(start=start, end=anchor.normalize())
    if mode.startswith("Last ") and "month" in mode.lower():
        months = int(re.findall(r"\d+", mode)[0])
        start = (anchor.normalize() - pd.DateOffset(months=months) + pd.Timedelta(days=1)).normalize()
        return Period(start=start, end=anchor.normalize())
    if mode == "YTD" or mode == "This Year":
        start = pd.Timestamp(year=anchor.year, month=1, day=1)
        return Period(start=start, end=anchor.normalize())
    if mode.startswith("Last ") and "year" in mode.lower():
        years = int(re.findall(r"\d+", mode)[0])
        start = (anchor.normalize() - pd.DateOffset(years=years) + pd.Timedelta(days=1)).normalize()
        return Period(start=start, end=anchor.normalize())
    return None

def period_prev_same_length(p: Period) -> Period:
    length = (p.end - p.start).days + 1
    end = p.start - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=length-1)
    return Period(start=start.normalize(), end=end.normalize())

def period_yoy(p: Period) -> Period:
    # Shift by 1 year (approx) using DateOffset to handle leap years
    start = (p.start - pd.DateOffset(years=1)).normalize()
    end = (p.end - pd.DateOffset(years=1)).normalize()
    return Period(start=start, end=end)

def filter_by_period(df: pd.DataFrame, p: Period) -> pd.DataFrame:
    w = pd.to_datetime(df["WeekEnd"], errors="coerce")
    return df[(w >= p.start) & (w <= p.end)].copy()

def available_month_labels(df: pd.DataFrame) -> List[str]:
    w = pd.to_datetime(df.get("WeekEnd"), errors="coerce")
    vals = sorted(pd.Series(w.dropna().dt.to_period("M").astype(str)).unique().tolist())
    return vals

def available_year_labels(df: pd.DataFrame) -> List[str]:
    w = pd.to_datetime(df.get("WeekEnd"), errors="coerce")
    vals = sorted(pd.Series(w.dropna().dt.year.astype(int).astype(str)).unique().tolist())
    return vals

def filter_by_period_labels(df: pd.DataFrame, labels: List[str], granularity: str) -> pd.DataFrame:
    out = df.copy()
    w = pd.to_datetime(out["WeekEnd"], errors="coerce")
    if not labels:
        return out.iloc[0:0].copy()
    if granularity == "Month":
        keys = w.dt.to_period("M").astype(str)
    else:
        keys = w.dt.year.astype("Int64").astype(str)
    return out[keys.isin(labels)].copy()

def period_from_df(df: pd.DataFrame) -> Optional[Period]:
    w = pd.to_datetime(df.get("WeekEnd"), errors="coerce").dropna()
    if w.empty:
        return None
    return Period(start=w.min().normalize(), end=w.max().normalize())

def compact_selection_label(labels: List[str], granularity: str) -> str:
    if not labels:
        return f"Selected {granularity.lower()}s"
    if len(labels) <= 3:
        return ", ".join(labels)
    return f"{len(labels)} selected {granularity.lower()}s"


def timeframe_short_label(timeframe: str) -> str:
    """Compact label used in column headings."""
    if timeframe.startswith("Last ") and "week" in timeframe.lower():
        m = re.findall(r"\d+", timeframe)
        if m:
            return f"{m[0]}W"
    if timeframe.startswith("Week"):
        return "Wk"
    if timeframe == "YTD" or timeframe == "This Year":
        return "YTD" if timeframe == "YTD" else "Year"
    if timeframe == "This Month":
        return "Month"
    if timeframe == "Custom Months":
        return "Months"
    if timeframe == "Custom Years":
        return "Years"
    if timeframe.startswith("Last ") and "month" in timeframe.lower():
        m = re.findall(r"\d+", timeframe)
        if m:
            return f"{m[0]}M"
    if timeframe.startswith("Last ") and "year" in timeframe.lower():
        m = re.findall(r"\d+", timeframe)
        if m:
            return f"{m[0]}Y"
    return timeframe

def ab_labels(timeframe: str, compare_mode: str, pA: Period, pB: Optional[Period]) -> Tuple[str, Optional[str]]:
    """Human-friendly labels for A/B periods."""
    tf = timeframe_short_label(timeframe)
    a = f"Current {tf}"
    if compare_mode == "None" or pB is None:
        return a, None
    if compare_mode.startswith("Prior"):
        b = f"Prior {tf}"
    else:
        b = f"YoY {tf}"
    return a, b

def rename_ab_columns(df: pd.DataFrame, a_label: str, b_label: Optional[str]) -> pd.DataFrame:
    """Rename Sales_A/B and Units_A/B columns to readable headings."""
    rename = {}
    if "Sales_A" in df.columns:
        rename["Sales_A"] = f"Sales ({a_label})"
    if "Units_A" in df.columns:
        rename["Units_A"] = f"Units ({a_label})"
    if b_label:
        if "Sales_B" in df.columns:
            rename["Sales_B"] = f"Sales ({b_label})"
        if "Units_B" in df.columns:
            rename["Units_B"] = f"Units ({b_label})"
    else:
        # if B exists but no label, keep generic
        if "Sales_B" in df.columns:
            rename["Sales_B"] = "Sales (Comparison)"
        if "Units_B" in df.columns:
            rename["Units_B"] = "Units (Comparison)"
    out = df.rename(columns=rename).copy()
    return out

def format_period_range(p: Period) -> str:
    return f"{p.start.date().isoformat()} → {p.end.date().isoformat()}"

# -----------------------------
# KPI + analytics engines
# -----------------------------
def calc_kpis(df: pd.DataFrame) -> Dict[str, float]:
    sales = float(df["Sales"].sum())
    units = float(df["Units"].sum())
    asp = float(sales / units) if units else 0.0
    active_skus = int(df.loc[df["Sales"] > 0, "SKU"].nunique())
    active_retailers = int(df.loc[df["Sales"] > 0, "Retailer"].nunique())
    active_vendors = int(df.loc[df["Sales"] > 0, "Vendor"].nunique())
    return {
        "Sales": sales,
        "Units": units,
        "ASP": asp,
        "Active SKUs": active_skus,
        "Active Retailers": active_retailers,
        "Active Vendors": active_vendors,
    }

def calc_delta(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, va in a.items():
        vb = b.get(k, 0.0)
        out[k] = va - vb
    return out

def pct_change(cur: float, prev: float) -> float:
    if prev == 0:
        return np.nan if cur == 0 else np.inf
    return (cur - prev) / prev

def drivers(df_a: pd.DataFrame, df_b: pd.DataFrame, level: str) -> pd.DataFrame:
    """Top drivers by contribution (Sales_A - Sales_B)."""
    g = [level]
    a = df_a.groupby(g, as_index=False).agg(Sales_A=("Sales","sum"), Units_A=("Units","sum"))
    b = df_b.groupby(g, as_index=False).agg(Sales_B=("Sales","sum"), Units_B=("Units","sum"))
    m = a.merge(b, on=g, how="outer").fillna(0.0)
    m["Sales_Δ"] = m["Sales_A"] - m["Sales_B"]
    m["Units_Δ"] = m["Units_A"] - m["Units_B"]
    total = float(m["Sales_Δ"].sum())
    denom = float(m["Sales_Δ"].abs().sum())
    m["Contribution_%"] = np.where(denom != 0, (m["Sales_Δ"].abs() / denom), 0.0)
    m = m.sort_values("Sales_Δ", ascending=False)
    return m

def weekly_series(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    s = df.groupby(by + ["WeekEnd"], as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum"))
    s = s.sort_values("WeekEnd")
    return s

def trend_slope(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    x = np.arange(values.size, dtype=float)
    y = values.astype(float)
    if np.all(np.isnan(y)) or np.all(y == 0):
        return 0.0
    # Replace nans with 0 for stability
    y = np.nan_to_num(y, nan=0.0)
    return float(np.polyfit(x, y, 1)[0])

def classify_trend(series: pd.Series, min_weeks: int = 8) -> Tuple[str, float, int, int]:
    y = np.array(series.values, dtype=float)
    slope = trend_slope(y)
    weeks_up = int(np.sum(np.diff(y) > 0))
    weeks_down = int(np.sum(np.diff(y) < 0))
    # threshold scales with median magnitude to avoid tiny-noise labeling
    med = float(np.median(np.abs(y))) if y.size else 0.0
    thr = max(1.0, 0.03 * med)  # 3% of median, min 1
    if slope > thr and weeks_up >= max(2, (min_weeks//2)):
        return ("Increasing", slope, weeks_up, weeks_down)
    if slope < -thr and weeks_down >= max(2, (min_weeks//2)):
        return ("Declining", slope, weeks_up, weeks_down)
    return ("Flat", slope, weeks_up, weeks_down)

def momentum_score(series: pd.Series) -> float:
    y = np.array(series.values, dtype=float)
    if y.size < 3:
        return 0.0
    y = np.nan_to_num(y, nan=0.0)
    slope = trend_slope(y)
    # normalize slope vs median
    med = np.median(np.abs(y)) if np.any(y) else 1.0
    trend = np.clip((slope / max(1.0, med)) * 30.0, -30.0, 30.0)

    # acceleration: last 2 vs prior 4
    recent = np.mean(y[-2:]) if y.size >= 2 else y[-1]
    prior = np.mean(y[-6:-2]) if y.size >= 6 else (np.mean(y[:-2]) if y.size > 2 else y[0])
    accel = 0.0
    if prior != 0:
        accel = np.clip(((recent - prior) / abs(prior)) * 20.0, -20.0, 20.0)
    else:
        accel = 20.0 if recent > 0 else 0.0

    # consistency
    up = np.sum(np.diff(y) > 0)
    denom = max(1, y.size - 1)
    cons = np.clip((up / denom) * 50.0, 0.0, 50.0)

    score = float(np.clip(trend + accel + cons, 0.0, 100.0))
    return score

def momentum_label(score: float) -> str:
    if score >= 80:
        return "Strong Up"
    if score >= 60:
        return "Up"
    if score >= 40:
        return "Neutral"
    if score >= 20:
        return "Down"
    return "Strong Down"

def build_momentum(df_hist: pd.DataFrame, group_level: str, lookback_weeks: int = 8) -> pd.DataFrame:
    """Compute momentum for each group using last N weeks."""
    s = weekly_series(df_hist, [group_level])
    # limit to last N weeks per group
    out_rows = []
    for key, g in s.groupby(group_level):
        g = g.sort_values("WeekEnd")
        g = g.tail(lookback_weeks)
        score = momentum_score(g["Sales"])
        trend, slope, wu, wd = classify_trend(g["Sales"], min_weeks=min(lookback_weeks, len(g)))
        out_rows.append({
            group_level: key,
            "Momentum": score,
            "Momentum Label": momentum_label(score),
            "Trend": trend,
            "Slope": slope,
            "Weeks Up": wu,
            "Weeks Down": wd,
            "Sales (lookback)": float(g["Sales"].sum()),
            "Units (lookback)": float(g["Units"].sum()),
        })
    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["Momentum","Sales (lookback)"], ascending=[False, False])
    return out

# -----------------------------
# Newness / lifecycle
# -----------------------------
def first_sale_ever(df_all: pd.DataFrame, p: Period) -> pd.DataFrame:
    d = df_all[df_all["Sales"] > 0].copy()
    if d.empty:
        return d
    # Ensure we pull Retailer/Vendor from the actual first-sale row (not arbitrary group order)
    d = d.sort_values(["SKU", "WeekEnd"])
    first_week = d.groupby("SKU", as_index=False).agg(FirstWeek=("WeekEnd", "min"))
    first_rows = d.merge(first_week, on=["SKU"], how="inner")
    first_rows = first_rows[first_rows["WeekEnd"] == first_rows["FirstWeek"]].copy()
    # If multiple retailers on same first week, just take the first row after sorting
    first_rows = first_rows.groupby("SKU", as_index=False).first()
    first_rows = first_rows.rename(columns={"Retailer": "FirstRetailer", "Vendor": "FirstVendor"})
    in_period = first_rows[(first_rows["FirstWeek"] >= p.start) & (first_rows["FirstWeek"] <= p.end)].copy()
    return in_period.sort_values("FirstWeek")

def new_placement(df_all: pd.DataFrame, p: Period) -> pd.DataFrame:
    d = df_all[df_all["Sales"] > 0].copy()
    if d.empty:
        return d
    d = d.sort_values(["SKU", "Retailer", "WeekEnd"])
    first_week = d.groupby(["SKU", "Retailer"], as_index=False).agg(FirstWeek=("WeekEnd", "min"))
    first_rows = d.merge(first_week, on=["SKU", "Retailer"], how="inner")
    first_rows = first_rows[first_rows["WeekEnd"] == first_rows["FirstWeek"]].copy()
    first_rows = first_rows.groupby(["SKU", "Retailer"], as_index=False).first()
    first_rows = first_rows.rename(columns={"Vendor": "Vendor"})
    in_period = first_rows[(first_rows["FirstWeek"] >= p.start) & (first_rows["FirstWeek"] <= p.end)].copy()
    # Exclude those that are also first sale ever
    first_sku = d.groupby("SKU", as_index=False).agg(SKUFirst=("WeekEnd", "min"))
    in_period = in_period.merge(first_sku, on="SKU", how="left")
    in_period = in_period[in_period["SKUFirst"] < in_period["FirstWeek"]]
    return in_period.sort_values("FirstWeek")

def reactivated(df_all: pd.DataFrame, p: Period, dormant_weeks: int = 8) -> pd.DataFrame:
    # A SKU with sales in current period, and previously had a gap of >= dormant_weeks with no sales.
    d = df_all.copy()
    d = d.sort_values("WeekEnd")
    # build weekly sales per SKU
    s = weekly_series(d, ["SKU"])
    if s.empty:
        return pd.DataFrame(columns=["SKU","ReactivatedWeek","DormantWeeks"])
    out=[]
    for sku, g in s.groupby("SKU"):
        g = g.sort_values("WeekEnd")
        # mark weeks with sales>0
        g["HasSales"] = g["Sales"] > 0
        # consider weeks within current period
        cur = g[(g["WeekEnd"] >= p.start) & (g["WeekEnd"] <= p.end) & (g["HasSales"])]
        if cur.empty:
            continue
        # find most recent sale before current period
        prev_sales = g[(g["WeekEnd"] < p.start) & (g["HasSales"])]
        if prev_sales.empty:
            continue
        last_prev = prev_sales["WeekEnd"].max()
        first_cur = cur["WeekEnd"].min()
        gap_weeks = int((first_cur - last_prev).days // 7)  # approximate
        if gap_weeks >= dormant_weeks:
            out.append({"SKU": sku, "ReactivatedWeek": first_cur, "DormantWeeks": gap_weeks})
    return pd.DataFrame(out).sort_values("ReactivatedWeek") if out else pd.DataFrame(columns=["SKU","ReactivatedWeek","DormantWeeks"])

def lifecycle_table(df_all: pd.DataFrame, p: Period, lookback_weeks: int = 8) -> pd.DataFrame:
    # Stage per SKU using full history but relative to current period.
    s = weekly_series(df_all, ["SKU"])
    if s.empty:
        return pd.DataFrame(columns=["SKU","Stage","Momentum","Sales (lookback)"])
    # universe and anchor = period end
    anchor = p.end
    lb_start = anchor - pd.Timedelta(days=(7*lookback_weeks - 1))
    out=[]
    first_week = df_all[df_all["Sales"] > 0].groupby("SKU")["WeekEnd"].min()
    last_week = df_all[df_all["Sales"] > 0].groupby("SKU")["WeekEnd"].max()

    # placements
    fr = df_all[df_all["Sales"] > 0].groupby(["SKU","Retailer"])["WeekEnd"].min().reset_index().rename(columns={"WeekEnd":"FirstWeekRetailer"})

    for sku, g in s.groupby("SKU"):
        g = g.sort_values("WeekEnd")
        # window
        gw = g[(g["WeekEnd"] >= lb_start) & (g["WeekEnd"] <= anchor)].copy()
        mom = momentum_score(gw["Sales"]) if not gw.empty else 0.0
        sales_lb = float(gw["Sales"].sum()) if not gw.empty else 0.0
        units_lb = float(gw["Units"].sum()) if (not gw.empty and "Units" in gw.columns) else 0.0

        fw = first_week.get(sku, pd.NaT)
        lw = last_week.get(sku, pd.NaT)

        stage = "Mature"
        trend = "Flat"
        slope = 0.0
        weeks_up = 0
        weeks_down = 0
        last_w_sales = float(gw["Sales"].iloc[-1]) if len(gw) >= 1 else 0.0
        prev_w_sales = float(gw["Sales"].iloc[-2]) if len(gw) >= 2 else 0.0
        wow_sales = last_w_sales - prev_w_sales if len(gw) >= 2 else np.nan
        # Launch: first sale ever in current period
        if pd.notna(fw) and (fw >= p.start) and (fw <= p.end):
            stage = "Launch"
        else:
            # Dormant: no sale in lookback window
            if pd.isna(lw) or lw < lb_start:
                stage = "Dormant"
            else:
                # Growth/Decline based on trend classification in lookback
                if not gw.empty:
                    trend, slope, wu, wd = classify_trend(gw["Sales"], min_weeks=min(lookback_weeks, len(gw)))
                    weeks_up, weeks_down = int(wu), int(wd)
                    if trend == "Increasing":
                        stage = "Growth"
                    elif trend == "Declining":
                        stage = "Decline"
                    else:
                        stage = "Mature"

        out.append({
            "SKU": sku,
            "Stage": stage,
            "Trend": trend,
            "Momentum": mom,
            "Sales (lookback)": sales_lb,
            "Units (lookback)": units_lb,
            "First Sale": fw,
            "Last Sale": lw,
            "Slope": float(slope),
            "Weeks Up": int(weeks_up),
            "Weeks Down": int(weeks_down),
            "Last Week Sales": float(last_w_sales),
            "WoW Sales Δ": float(wow_sales) if not pd.isna(wow_sales) else np.nan,
        })
    out_df = pd.DataFrame(out)
    if not out_df.empty:
        out_df = out_df.sort_values(["Stage","Momentum","Sales (lookback)"], ascending=[True, False, False])
    return out_df

# -----------------------------
# Opportunity detector
# -----------------------------
def opportunity_detector(df_all: pd.DataFrame, df_a: pd.DataFrame, df_b: pd.DataFrame, p: Period) -> Dict[str, pd.DataFrame]:
    retailers = sorted(df_all["Retailer"].dropna().unique().tolist())
    if not retailers:
        retailers = []
    # Momentum per SKU on full history up to period end
    anchor = p.end
    df_hist = df_all[df_all["WeekEnd"] <= anchor].copy()
    mom_sku = build_momentum(df_hist, "SKU", lookback_weeks=8)
    mom_sku = mom_sku.set_index("SKU") if not mom_sku.empty else pd.DataFrame()

    # Current period retailer presence per SKU
    cur = df_a.groupby(["SKU","Retailer"], as_index=False).agg(Sales=("Sales","sum"))
    cur_pos = cur[cur["Sales"] > 0].copy()

    # 1) High momentum + low distribution
    hml = []
    if not cur_pos.empty:
        selling_counts = cur_pos.groupby("SKU")["Retailer"].nunique()
        for sku, cnt in selling_counts.items():
            score = float(mom_sku.loc[sku, "Momentum"]) if (isinstance(mom_sku, pd.DataFrame) and sku in mom_sku.index) else 0.0
            if score >= 80 and cnt <= 1:
                hml.append({"SKU": sku, "Momentum": score, "Retailers Selling": int(cnt)})
    high_mom_low_dist = pd.DataFrame(hml).sort_values(["Momentum","Retailers Selling"], ascending=[False, True]) if hml else pd.DataFrame(columns=["SKU","Momentum","Retailers Selling"])

    # 2) Under-distributed opportunities (missing retailer where vendor is active)
    under = []
    # vendor activity by retailer in current period
    vend_ret = df_a.groupby(["Vendor","Retailer"], as_index=False).agg(Sales=("Sales","sum"))
    vend_ret = vend_ret[vend_ret["Sales"] > 0]
    vend_active = set(zip(vend_ret["Vendor"], vend_ret["Retailer"]))

    sku_vendor = df_all.groupby("SKU", as_index=False).agg(Vendor=("Vendor","first"))
    sku_to_vendor = dict(zip(sku_vendor["SKU"], sku_vendor["Vendor"]))
    if not df_a.empty:
        sku_rets = cur_pos.groupby("SKU")["Retailer"].apply(set).to_dict()
        for sku, sold_set in sku_rets.items():
            score = float(mom_sku.loc[sku, "Momentum"]) if (isinstance(mom_sku, pd.DataFrame) and sku in mom_sku.index) else 0.0
            if score < 60:
                continue
            vendor = sku_to_vendor.get(sku, "Unknown")
            for r in retailers:
                if r in sold_set:
                    continue
                if (vendor, r) in vend_active:
                    under.append({"SKU": sku, "Vendor": vendor, "Missing Retailer": r, "Momentum": score})
    under_df = pd.DataFrame(under).sort_values(["Momentum"], ascending=False) if under else pd.DataFrame(columns=["SKU","Vendor","Missing Retailer","Momentum"])

    # 3) Retailer growth gaps (vendor-level by default)
    a = df_a.groupby(["Vendor","Retailer"], as_index=False).agg(Sales_A=("Sales","sum"))
    b = df_b.groupby(["Vendor","Retailer"], as_index=False).agg(Sales_B=("Sales","sum"))
    m = a.merge(b, on=["Vendor","Retailer"], how="outer").fillna(0.0)
    # compute growth %
    m["Growth_%"] = m.apply(lambda r: pct_change(float(r["Sales_A"]), float(r["Sales_B"])), axis=1)
    # For each vendor, compute range across retailers (max-min)
    gaps=[]
    for vendor, g in m.groupby("Vendor"):
        if g.empty:
            continue
        # require some meaningful base
        if float(g["Sales_A"].sum()) < 500:
            continue
        # ignore inf/nan
        gg = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["Growth_%"])
        if gg.empty:
            continue
        mx = float(gg["Growth_%"].max())
        mn = float(gg["Growth_%"].min())
        if (mx - mn) >= 0.30:  # 30pp gap
            # show top and bottom retailer
            top = gg.loc[gg["Growth_%"].idxmax()]
            bot = gg.loc[gg["Growth_%"].idxmin()]
            gaps.append({
                "Vendor": vendor,
                "Best Retailer": top["Retailer"],
                "Best Growth %": top["Growth_%"],
                "Worst Retailer": bot["Retailer"],
                "Worst Growth %": bot["Growth_%"],
                "Gap (pp)": (mx - mn),
                "Sales A": float(g["Sales_A"].sum()),
            })
    gaps_df = pd.DataFrame(gaps).sort_values("Gap (pp)", ascending=False) if gaps else pd.DataFrame(columns=["Vendor","Best Retailer","Best Growth %","Worst Retailer","Worst Growth %","Gap (pp)","Sales A"])

    return {
        "High Momentum / Low Distribution": high_mom_low_dist,
        "Under-distributed Opportunities": under_df,
        "Retailer Growth Gaps": gaps_df,
    }

# -----------------------------
# UI helpers
# -----------------------------
def money(x: float) -> str:
    return f"${x:,.0f}"

def pct_fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    if x == np.inf:
        return "∞"
    if x == -np.inf:
        return "-∞"
    return f"{x*100:,.1f}%"


def kpi_card(label: str, value: str, delta: Optional[str] = None):
    """Theme-safe KPI card.

    delta supports basic HTML (we generate sign-colored spans).
    """
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(
        f'''
        <div class="kpi-card">
            <div class="kpi-title">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        ''',
        unsafe_allow_html=True,
    )

def biggest_increase_card(label: str, name: str, current_sales: float, previous_sales: float):
    delta = float(current_sales) - float(previous_sales)
    pct = pct_change(float(current_sales), float(previous_sales))
    color = "#2e7d32" if delta > 0 else ("#c62828" if delta < 0 else "var(--text-color)")
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "•")
    pct_html = "" if pd.isna(pct) else f'<div class="kpi-big-pct" style="color:{color}">({pct_fmt(pct)})</div>'
    st.markdown(
        f'''
        <div class="kpi-card">
            <div class="kpi-title">{label}</div>
            <div class="kpi-big-name">{html.escape(str(name))}</div>
            <div class="kpi-big-main" style="color:{color}">{arrow} {money(delta)}</div>
            <div class="kpi-big-total">Total: {money(float(current_sales))}</div>
            {pct_html}
        </div>
        ''',
        unsafe_allow_html=True,
    )

def leader_sales_card(label: str, name: str, current_sales: float, previous_sales: float):
    delta = float(current_sales) - float(previous_sales)
    pct = pct_change(float(current_sales), float(previous_sales))
    color = "#2e7d32" if delta > 0 else ("#c62828" if delta < 0 else "var(--text-color)")
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "•")
    pct_html = "" if pd.isna(pct) else f'<span class="delta-pct" style="color:{color}">({pct_fmt(pct)})</span>'
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{label}</div>
            <div class="kpi-big-name">{html.escape(str(name))}</div>
            <div class="kpi-value">{money(float(current_sales))}</div>
            <div class="kpi-delta" style="color:{color}"><span class="delta-abs">{arrow} {money(delta)}</span>{pct_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_df(df: pd.DataFrame, height: int = 320):
    st.dataframe(df, use_container_width=True, height=height, hide_index=True)

# -----------------------------
# Main app
# -----------------------------
def run_app():
    st.set_page_config(page_title=APP_TITLE, layout="wide")


    # Theme-aware global styles
    st.markdown(
        """
        <style>
        /* Streamlit theme variables: --text-color, --background-color, --secondary-background-color */
        .kpi-card{
            border:1px solid rgba(128,128,128,0.22);
            border-radius:14px;
            padding:14px 14px;
            background: var(--secondary-background-color);
        }
        .kpi-title{
            font-size:12px;
            font-weight:600;
            letter-spacing:0.02em;
            color: var(--text-color);
            opacity: 0.70;
        }
        .kpi-value{
            font-size:28px;
            font-weight:800;
            line-height:1.15;
            color: var(--text-color);
        }
        .kpi-delta{
            font-size:13px;
            margin-top:6px;
            color: var(--text-color);
            opacity: 0.80;
        }
        .kpi-delta .delta-abs{ font-weight:800; }
        .kpi-delta .delta-pct{ font-weight:700; opacity:0.88; margin-left:6px; }
        .kpi-delta .delta-note{ opacity:0.75; margin-left:6px; }
        .kpi-big-main{
            font-size:30px;
            font-weight:800;
            line-height:1.05;
            margin-top:4px;
        }
        .kpi-big-name{
            font-size:22px;
            font-weight:700;
            line-height:1.15;
            margin-top:6px;
            color: var(--text-color);
        }
        .kpi-big-total{
            font-size:13px;
            opacity:0.78;
            margin-top:6px;
            color: var(--text-color);
        }
        .kpi-big-pct{
            font-size:13px;
            font-weight:700;
            margin-top:4px;
        }
        .intel-card{
            border:1px solid rgba(128,128,128,0.22);
            border-radius:16px;
            padding:14px 16px;
            background: var(--secondary-background-color);
            margin-bottom:14px;
        }
        .intel-header{
            font-size:12px;
            font-weight:800;
            letter-spacing:0.06em;
            color: var(--text-color);
            opacity:0.70;
        }
        .intel-body{
            margin-top:8px;
            color: var(--text-color);
            font-size:15px;
            line-height:1.45;
        }
        .intel-body ul{
            margin: 0;
            padding-left: 18px;
        }
        .intel-body li{
            margin: 6px 0;
        }
        /* Compact HTML tables used below Weekly Detail / Movers */
        .report-table{
            width:100% !important;
            table-layout:auto;
            border-collapse: collapse;
            font-size:14px !important;
            line-height:1.3;
        }
        .report-table th, .report-table td{
            padding:6px 8px;
            border-bottom:1px solid rgba(128,128,128,0.18);
            text-align:left;
            white-space:nowrap;
        }
        .report-table th{
            font-size:13px !important;
            font-weight:700;
            color:var(--text-color);
            opacity:0.82;
        }
        .report-table td{
            color:var(--text-color);
        }
        /* Keep dataframe text aligned with the rest of the app */
        div[data-testid="stDataFrame"] * {
            font-size:14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(APP_TITLE)

    vm = load_vendor_map()
    store = load_store()

    with st.sidebar:
        st.header("Data")
        up = st.file_uploader("Upload weekly sales workbook (.xlsx)", type=["xlsx"])
        year = st.number_input("Year hint (for filename parsing)", min_value=2010, max_value=2100, value=date.today().year, step=1)
        if st.button("Ingest upload", disabled=(up is None)):
            if up is not None:
                raw = read_weekly_workbook(up, int(year))
                # enrich now so we can persist UnitPrice (legacy) but also show computed
                new = enrich_sales(raw, vm)
                # Persist in legacy shape
                merged = pd.concat([store, raw], ignore_index=True)
                save_store(merged)
                st.success(f"Ingested {len(raw):,} rows from {getattr(up,'name','upload.xlsx')}.")
                store = load_store()

        st.divider()
        st.header("Filters")

        scope = st.selectbox("Scope", ["All", "Retailer", "Vendor", "SKU"], index=0)
        # We build an enriched df first (so Vendor exists)
        df_all = enrich_sales(store, vm)
        # Scope pickers
        scope_pick = None
        if scope == "Retailer":
            scope_pick = st.multiselect("Retailer(s)", options=sorted(df_all["Retailer"].dropna().unique()), default=[])
        elif scope == "Vendor":
            scope_pick = st.multiselect("Vendor(s)", options=sorted(df_all["Vendor"].dropna().unique()), default=[])
        elif scope == "SKU":
            scope_pick = st.multiselect("SKU(s)", options=sorted(df_all["SKU"].dropna().unique()), default=[])

        analysis_view = st.radio("Analysis View", ["Standard Intelligence", "Multi Month / Year Compare"], index=0)
        multi_granularity = "Month"
        current_labels_sel: List[str] = []
        compare_labels_sel: List[str] = []
        if analysis_view == "Standard Intelligence":
            timeframe_options = ["Week (latest)", "Last 4 weeks", "Last 8 weeks", "Last 13 weeks", "Last 26 weeks", "Last 52 weeks", "YTD"]
            timeframe_index = 2
            timeframe = st.selectbox("Timeframe", timeframe_options, index=timeframe_index)
            compare_mode = st.selectbox("Compare", ["None", "Prior period (same length)", "YoY (same dates)"], index=1)
        else:
            multi_granularity = st.selectbox("Compare By", ["Month", "Year"], index=0)
            if multi_granularity == "Month":
                timeframe = "Custom Months"
                period_options = available_month_labels(df_all)
                default_current = period_options[-1:]
                default_compare = period_options[-2:-1]
            else:
                timeframe = "Custom Years"
                period_options = available_year_labels(df_all)
                default_current = period_options[-1:]
                default_compare = period_options[-2:-1]
            current_labels_sel = st.multiselect(f"Current {multi_granularity}s", options=period_options, default=default_current)
            compare_labels_sel = st.multiselect(f"Compare {multi_granularity}s", options=period_options, default=default_compare)
            compare_mode = "Custom selection" if compare_labels_sel else "None"

        min_sales = st.number_input("Min Sales ($) for lists", min_value=0.0, value=0.0, step=100.0)
        min_units = st.number_input("Min Units for lists", min_value=0.0, value=0.0, step=10.0)

        driver_level = st.selectbox("Driver Level", ["SKU", "Vendor", "Retailer"], index=0)
        show_full_history_lifecycle = st.toggle("Lifecycle uses full history", value=True)

    # Apply scope filter
    df_scope = df_all.copy()
    if scope == "Retailer" and scope_pick:
        df_scope = df_scope[df_scope["Retailer"].isin(scope_pick)]
    elif scope == "Vendor" and scope_pick:
        df_scope = df_scope[df_scope["Vendor"].isin(scope_pick)]
    elif scope == "SKU" and scope_pick:
        df_scope = df_scope[df_scope["SKU"].isin(scope_pick)]

    # Choose current / comparison periods
    custom_labels = None
    if analysis_view == "Multi Month / Year Compare":
        dfA = filter_by_period_labels(df_scope, current_labels_sel, multi_granularity)
        dfB = filter_by_period_labels(df_scope, compare_labels_sel, multi_granularity) if compare_labels_sel else df_scope.iloc[0:0].copy()
        pA = period_from_df(dfA)
        pB = period_from_df(dfB) if not dfB.empty else None
        if pA is None:
            st.info("Choose one or more months/years to begin.")
            return
        a_lbl = compact_selection_label(current_labels_sel, multi_granularity)
        b_lbl = compact_selection_label(compare_labels_sel, multi_granularity) if pB is not None else None
        custom_labels = (a_lbl, b_lbl)
    else:
        pA = pick_period(df_scope, timeframe)
        if pA is None:
            st.info("Upload or ingest data to begin.")
            return
        dfA = filter_by_period(df_scope, pA)
        if compare_mode == "None":
            pB = None
            dfB = dfA.iloc[0:0].copy()
        elif compare_mode.startswith("Prior"):
            pB = period_prev_same_length(pA)
            dfB = filter_by_period(df_scope, pB)
        else:
            pB = period_yoy(pA)
            dfB = filter_by_period(df_scope, pB)
        a_lbl, b_lbl = ab_labels(timeframe, compare_mode, pA, pB)

    # Show period definitions in the sidebar for clarity
    st.sidebar.markdown("### Period Definition")
    st.sidebar.markdown(f"**Current:** {a_lbl}<br><span style='opacity:0.8'>{format_period_range(pA)}</span>", unsafe_allow_html=True)
    if compare_mode != "None" and pB is not None:
        st.sidebar.markdown(f"**Compare:** {b_lbl}<br><span style='opacity:0.8'>{format_period_range(pB)}</span>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<span style='opacity:0.75'>Compare: None</span>", unsafe_allow_html=True)
    st.sidebar.divider()
# KPI compute
    kA = calc_kpis(dfA)
    kB = calc_kpis(dfB) if pB is not None else {k:0.0 for k in kA.keys()}

    # Newness metrics must use history; use df_all unless scope filtered is desired
    df_hist_for_new = df_all.copy()
    if scope == "Retailer" and scope_pick:
        df_hist_for_new = df_hist_for_new[df_hist_for_new["Retailer"].isin(scope_pick)]
    elif scope == "Vendor" and scope_pick:
        df_hist_for_new = df_hist_for_new[df_hist_for_new["Vendor"].isin(scope_pick)]
    elif scope == "SKU" and scope_pick:
        df_hist_for_new = df_hist_for_new[df_hist_for_new["SKU"].isin(scope_pick)]

    first_ever = first_sale_ever(df_hist_for_new, pA)
    placements = new_placement(df_hist_for_new, pA)

    # Layout
    cur_range = format_period_range(pA)
    cmp_range = format_period_range(pB) if pB is not None else ""
    cmp_name = b_lbl or ""

    st.markdown(
        f"""
        <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin: 4px 0 10px 0;">
            <span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(128,128,128,0.22); background: var(--secondary-background-color); color: var(--text-color); font-weight:700; font-size:12px;">
                Scope: {scope}
            </span>
            <span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(128,128,128,0.22); background: var(--secondary-background-color); color: var(--text-color); font-weight:700; font-size:12px;">
                Current: {a_lbl} • {cur_range}
            </span>
            {(
                f'<span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(128,128,128,0.22); background: var(--secondary-background-color); color: var(--text-color); font-weight:700; font-size:12px;">Compare: {cmp_name} • {cmp_range}</span>'
                if compare_mode != "None" and pB is not None else
                '<span style="padding:6px 10px; border-radius:999px; border:1px solid rgba(128,128,128,0.22); background: var(--secondary-background-color); color: var(--text-color); font-weight:700; font-size:12px; opacity:0.75;">Compare: None</span>'
            )}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # KPI add-ons: Top entities (current) + biggest increases
    if compare_mode != "None":
        def _top_by_current(level: str):
            a = dfA.groupby(level, as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum"))
            b = dfB.groupby(level, as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum"))
            m = a.merge(b, on=level, how="outer", suffixes=("_A","_B")).fillna(0.0)
            if m.empty:
                return None
            m = m.sort_values("Sales_A", ascending=False)
            row = m.iloc[0]
            return str(row[level]), float(row["Sales_A"]), float(row["Sales_B"])

        def _top_by_increase(level: str):
            a = dfA.groupby(level, as_index=False).agg(Sales=("Sales","sum"))
            b = dfB.groupby(level, as_index=False).agg(Sales=("Sales","sum"))
            m = a.merge(b, on=level, how="outer", suffixes=("_A","_B")).fillna(0.0)
            if m.empty:
                return None
            m["Δ"] = m["Sales_A"] - m["Sales_B"]
            m = m.sort_values("Δ", ascending=False)
            row = m.iloc[0]
            return str(row[level]), float(row["Sales_A"]), float(row["Sales_B"])

        # helpers for KPI deltas / leader cards
        def _delta_html(cur: float, prev: float, is_money: bool, note: str = "") -> str:
            if compare_mode == "None":
                return ""
            d = cur - prev
            pc = pct_change(cur, prev)
            green = "#2e7d32"
            red = "#c62828"
            color = green if d > 0 else (red if d < 0 else "var(--text-color)")
            arrow = '▲ ' if d>0 else ('▼ ' if d<0 else '')
            if is_money:
                abs_s = money(d)
                if d > 0:
                    abs_s = "▲ " + abs_s
            else:
                abs_s = f"{d:,.0f}" if abs(d) >= 1 else f"{d:,.2f}"
                if d > 0:
                    abs_s = "▲ " + abs_s
            pct_s = pct_fmt(pc)
            note_html = f"<span class='delta-note'>{note}</span>" if note else ""
            return (
                f"<span class='delta-abs' style='color:{color}'>{arrow}{abs_s}</span>"
                f"<span class='delta-pct' style='color:{color}'>({pct_s})</span>"
                f"{note_html}"
            )

# 0) Intelligence summary
    sales_delta = kA["Sales"] - kB.get("Sales", 0.0)
    units_delta = kA["Units"] - kB.get("Units", 0.0)
    aspA = kA["ASP"]
    aspB = kB.get("ASP", 0.0)
    asp_delta = aspA - aspB

    # Driver headline (top contributor by Sales_Δ)
    drv = drivers(dfA, dfB, driver_level)
    top_pos = drv[drv["Sales_Δ"] > 0].head(1)
    top_neg = drv[drv["Sales_Δ"] < 0].tail(1)  # bottom
    headline_bits = []
    if compare_mode != "None":
        headline_bits.append(f"Sales {('up' if sales_delta >= 0 else 'down')} **{money(abs(sales_delta))}** vs comparison.")
        headline_bits.append(f"Units {('up' if units_delta >= 0 else 'down')} **{abs(units_delta):,.0f}**.")
        if not np.isnan(asp_delta):
            headline_bits.append(f"ASP {('up' if asp_delta >= 0 else 'down')} **{money(abs(asp_delta))}**.")
        if not top_pos.empty:
            headline_bits.append(f"Top driver: **{top_pos.iloc[0][driver_level]}** ({money(float(top_pos.iloc[0]['Sales_Δ']))}).")
        if not top_neg.empty:
            headline_bits.append(f"Top drag: **{top_neg.iloc[0][driver_level]}** ({money(float(top_neg.iloc[0]['Sales_Δ']))}).")
    else:
        headline_bits.append("Choose a comparison mode to see drivers and deltas.")

        # Render as a theme-aware card with readable text in light/dark mode
    _items = [b for b in headline_bits if str(b).strip()]
    def _md_bold_to_html(s: str) -> str:
        import re as _re, html as _html
        parts = []
        last = 0
        for mm in _re.finditer(r"\*\*(.+?)\*\*", s):
            parts.append(_html.escape(s[last:mm.start()]))
            parts.append(f"<strong>{_html.escape(mm.group(1))}</strong>")
            last = mm.end()
        parts.append(_html.escape(s[last:]))
        return "".join(parts)

    if not _items:
        _items = ["Choose a comparison mode to see drivers and deltas."]

    _lis = "".join([f"<li>{_md_bold_to_html(x)}</li>" for x in _items])

    st.markdown(
        f"""
        <div class="intel-card">
            <div class="intel-header">INTELLIGENCE SUMMARY</div>
            <div class="intel-body">
                <ul>
                    {_lis}
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1) KPI row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    def kdelta(key: str) -> str:
        if compare_mode == "None":
            return ""
        cur = float(kA.get(key, 0.0))
        prev = float(kB.get(key, 0.0))
        if key in ("Sales", "ASP"):
            return _delta_html(cur, prev, is_money=True)
        if key in ("Units", "Active SKUs"):
            return _delta_html(cur, prev, is_money=False)
        # fallback percent-only
        d = cur - prev
        green = "#2e7d32"
        red = "#c62828"
        color = green if d > 0 else (red if d < 0 else "var(--text-color)")
        arrow = '▲ ' if d>0 else ('▼ ' if d<0 else '')
        return (
            f"<span class='delta-abs' style='color:{color}'>{arrow}{pct_fmt(pct_change(cur, prev))}</span>"
            f"<span class='delta-note'>vs comp</span>"
        )

    

    with c1: kpi_card("Total Sales", money(kA["Sales"]), kdelta("Sales"))
    with c2: kpi_card("Total Units", f"{kA['Units']:,.0f}", kdelta("Units"))
    with c3: kpi_card("Avg Selling Price", money(kA["ASP"]), kdelta("ASP"))
    with c4: kpi_card("Active SKUs", f"{kA['Active SKUs']:,}", kdelta("Active SKUs"))
    with c5: kpi_card("First Sales", f"{len(first_ever):,}" , "")
    with c6: kpi_card("New Placements", f"{len(placements):,}", "")

    st.write("")


    # 1B) Leader KPI rows (based on current period + comparison)
    r1c1, r1c2, r1c3 = st.columns(3)
    tR = _top_by_current("Retailer")
    tV = _top_by_current("Vendor")
    tS = _top_by_current("SKU")

    with r1c1:
        if tR:
            leader_sales_card("Top Retailer (Sales)", tR[0], tR[1], tR[2])
    with r1c2:
        if tV:
            leader_sales_card("Top Vendor (Sales)", tV[0], tV[1], tV[2])
    with r1c3:
        if tS:
            leader_sales_card("Top SKU (Sales)", tS[0], tS[1], tS[2])

    r2c1, r2c2, r2c3 = st.columns(3)
    iR = _top_by_increase("Retailer")
    iV = _top_by_increase("Vendor")
    iS = _top_by_increase("SKU")

    with r2c1:
        if iR:
            biggest_increase_card("Retailer w/ Biggest Increase", iR[0], iR[1], iR[2])
    with r2c2:
        if iV:
            biggest_increase_card("Vendor w/ Biggest Increase", iV[0], iV[1], iV[2])
    with r2c3:
        if iS:
            biggest_increase_card("SKU w/ Biggest Increase", iS[0], iS[1], iS[2])

    st.write("")
    # 2) Drivers (two tables)
    st.subheader("Drivers (Contribution to change)")
    if compare_mode == "None":
        st.info("Select a comparison mode to compute drivers.")
    else:
        drv_show = drv.copy()
        drv_show = drv_show[(drv_show["Sales_A"] >= min_sales) | (drv_show["Sales_B"] >= min_sales)]
        pos = drv_show[drv_show["Sales_Δ"] > 0].head(10).copy()
        neg = drv_show[drv_show["Sales_Δ"] < 0].sort_values("Sales_Δ").head(10).copy()

        for d in (pos, neg):
            d["Sales_A"] = d["Sales_A"].map(money)
            d["Sales_B"] = d["Sales_B"].map(money)
            d["Sales_Δ"] = d["Sales_Δ"].map(lambda v: f"{money(v)}")
            d["Contribution_%"] = d["Contribution_%"].map(pct_fmt)

        left,right = st.columns(2)

        # Rename A/B columns for clearer display
        pos_disp = rename_ab_columns(pos, a_lbl, b_lbl)
        neg_disp = rename_ab_columns(neg, a_lbl, b_lbl)
        sales_a_col = f"Sales ({a_lbl})"
        sales_b_col = f"Sales ({b_lbl})" if b_lbl else "Sales (Comparison)"

        with left:
            st.markdown("**Top Positive Contributors**")
            render_df(pos_disp[[driver_level, sales_a_col, sales_b_col, "Sales_Δ", "Contribution_%"]], height=320)
        with right:
            st.markdown("**Top Negative Contributors**")
            render_df(neg_disp[[driver_level, sales_a_col, sales_b_col, "Sales_Δ", "Contribution_%"]], height=320)
    st.divider()

    # 3) Movers + Momentum
    st.subheader("New Activity")
    a,b = st.columns(2)
    with a:
        st.markdown("**First Sale Ever (Launches)**")
        if first_ever.empty:
            st.caption("None in this period.")
        else:
            fe = first_ever.copy()
            fe["FirstWeek"] = fe["FirstWeek"].dt.date.astype(str)
            render_df(fe.rename(columns={"FirstWeek":"First Week"})[["SKU","First Week","FirstRetailer","FirstVendor"]], height=260)
    with b:
        st.markdown("**New Retailer Placements**")
        if placements.empty:
            st.caption("None in this period.")
        else:
            pl = placements.copy()
            pl["FirstWeek"] = pl["FirstWeek"].dt.date.astype(str)
            render_df(pl.rename(columns={"FirstWeek":"First Week"})[["SKU","Retailer","Vendor","First Week"]], height=260)

    # New item tracker
    st.markdown("**New Item Tracker (track launches + placements over the next N weeks)**")
    track_weeks = st.slider("Tracking window (weeks)", min_value=4, max_value=13, value=8, step=1)

    def _track_rows() -> pd.DataFrame:
        rows = []
        horizon_end_days = 7 * track_weeks - 1
        # Launches
        if not first_ever.empty:
            for _, r in first_ever.iterrows():
                sku = r["SKU"]
                fw = pd.to_datetime(r["FirstWeek"], errors="coerce")
                if pd.isna(fw):
                    continue
                end = fw + pd.Timedelta(days=horizon_end_days)
                d = df_hist_for_new[(df_hist_for_new["SKU"]==sku) & (df_hist_for_new["WeekEnd"]>=fw) & (df_hist_for_new["WeekEnd"]<=end)].copy()
                if d.empty:
                    continue
                wk = d.groupby("WeekEnd", as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum")).sort_values("WeekEnd")
                total_sales = float(wk["Sales"].sum())
                total_units = float(wk["Units"].sum())
                last_sales = float(wk["Sales"].iloc[-1])
                prev_sales = float(wk["Sales"].iloc[-2]) if len(wk)>=2 else 0.0
                rows.append({
                    "Type": "Launch",
                    "SKU": sku,
                    "Start Week": fw.date().isoformat(),
                    "Weeks Tracked": int(len(wk)),
                    "Total Sales": total_sales,
                    "Total Units": total_units,
                    "Last Week Sales": last_sales,
                    "WoW Sales Δ": (last_sales - prev_sales) if len(wk)>=2 else np.nan,
                })
        # Placements (track within placement retailer)
        if not placements.empty:
            for _, r in placements.iterrows():
                sku = r["SKU"]
                ret = r["Retailer"]
                fw = pd.to_datetime(r["FirstWeek"], errors="coerce")
                if pd.isna(fw):
                    continue
                end = fw + pd.Timedelta(days=horizon_end_days)
                d = df_hist_for_new[(df_hist_for_new["SKU"]==sku) & (df_hist_for_new["Retailer"]==ret) & (df_hist_for_new["WeekEnd"]>=fw) & (df_hist_for_new["WeekEnd"]<=end)].copy()
                if d.empty:
                    continue
                wk = d.groupby("WeekEnd", as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum")).sort_values("WeekEnd")
                total_sales = float(wk["Sales"].sum())
                total_units = float(wk["Units"].sum())
                last_sales = float(wk["Sales"].iloc[-1])
                prev_sales = float(wk["Sales"].iloc[-2]) if len(wk)>=2 else 0.0
                rows.append({
                    "Type": "Placement",
                    "SKU": sku,
                    "Retailer": ret,
                    "Start Week": fw.date().isoformat(),
                    "Weeks Tracked": int(len(wk)),
                    "Total Sales": total_sales,
                    "Total Units": total_units,
                    "Last Week Sales": last_sales,
                    "WoW Sales Δ": (last_sales - prev_sales) if len(wk)>=2 else np.nan,
                })
        if not rows:
            return pd.DataFrame(columns=["Type","SKU","Retailer","Start Week","Weeks Tracked","Total Sales","Total Units","Last Week Sales","WoW Sales Δ"])
        return pd.DataFrame(rows)

    tracker = _track_rows()
    if tracker.empty:
        st.caption("No new items to track in the selected period.")
    else:
        show = tracker.copy()
        for c in ["Total Sales","Last Week Sales","WoW Sales Δ"]:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else money(float(v)))
        show["Total Units"] = show["Total Units"].map(lambda v: f"{float(v):,.0f}")
        render_df(show.sort_values(["Type","Start Week","SKU"], ascending=[True, False, True]), height=320)

    st.divider()

    # 5) Weekly detail (pivoted by Retailer)
    

    st.subheader("Weekly Detail (Retailer/Vendor x Weeks)")
    d = dfA.copy()
    d = d[(d["Sales"] >= min_sales) | (d["Units"] >= min_units)].copy()
    if d.empty:
        st.caption("No rows match the current thresholds.")
    else:
        pivot_dim = st.selectbox("Pivot rows by", options=["Retailer","Vendor"], index=0, key="weekly_pivot_dim")
        wk_sales = d.groupby([pivot_dim,"WeekEnd"], as_index=False).agg(Sales=("Sales","sum"))
        wk_sales["WeekEnd"] = pd.to_datetime(wk_sales["WeekEnd"], errors="coerce")
        # Use the actual weeks in the selected period, sorted
        weeks = sorted([pd.to_datetime(x) for x in wk_sales["WeekEnd"].dropna().unique().tolist()])
        wk_sales["Week"] = wk_sales["WeekEnd"].dt.date.astype(str)
        piv = wk_sales.pivot_table(index=pivot_dim, columns="Week", values="Sales", aggfunc="sum", fill_value=0.0)
        piv = piv.reindex(sorted(piv.index.tolist()))

        # Difference column (last two weeks in view)
        if len(weeks) >= 2:
            w_last = str(weeks[-1].date())
            w_prev = str(weeks[-2].date())
            piv["Δ vs prior week"] = piv.get(w_last, 0.0) - piv.get(w_prev, 0.0)
        else:
            piv["Δ vs prior week"] = 0.0

        # Pretty currency formatting to match the rest of the app tables
        piv_disp = piv.copy()
        for c in piv_disp.columns:
            if c == "Δ vs prior week":
                piv_disp[c] = piv_disp[c].map(lambda x: ("+" if x > 0 else "") + money(x))
            else:
                piv_disp[c] = piv_disp[c].map(money)
        piv_disp = piv_disp.reset_index()
        render_df(piv_disp, height=320)


    st.subheader("Movers & Trend Leaders")

    # Top Increasing / Declining (Avg weekly sales vs compare period)
    if compare_mode == "None":
        st.info("Select a comparison mode to compute increasing/declining vs the compare period.")
    else:
        # count weeks in each period (use unique WeekEnd dates)
        nA = max(1, dfA["WeekEnd"].nunique())
        nB = max(1, dfB["WeekEnd"].nunique())

        a = dfA.groupby("SKU", as_index=False).agg(Sales_A=("Sales","sum"), Units_A=("Units","sum"))
        b = dfB.groupby("SKU", as_index=False).agg(Sales_B=("Sales","sum"), Units_B=("Units","sum"))
        m = a.merge(b, on="SKU", how="outer").fillna(0.0)

        m["Sales (Current)"] = m["Sales_A"]
        m["Sales (Compare)"] = m["Sales_B"]
        m["Sales Δ"] = m["Sales_A"] - m["Sales_B"]
        m["Δ %"] = np.where(m["Sales_B"] != 0, m["Sales Δ"] / m["Sales_B"], np.nan)

        # thresholds based on volume in either period
        m = m[(m["Sales_A"] >= min_sales) | (m["Sales_B"] >= min_sales) | (m["Units_A"] >= min_units) | (m["Units_B"] >= min_units)].copy()

        inc = m[m["Sales Δ"] > 0].sort_values("Sales Δ", ascending=False).head(10)
        dec = m[m["Sales Δ"] < 0].sort_values("Sales Δ", ascending=True).head(10)

        def _fmt_diff(val: float) -> str:
            green = "#2e7d32"
            red = "#c62828"
            color = green if val > 0 else (red if val < 0 else "var(--text-color)")
            s = money(val)
            if val > 0:
                s = "+" + s
            return f"<span style='color:{color}'>{s}</span>"

        def _fmt_pct(val: float) -> str:
            if pd.isna(val) or val == np.inf or val == -np.inf:
                return ""
            green = "#2e7d32"
            red = "#c62828"
            color = green if val > 0 else (red if val < 0 else "var(--text-color)")
            s = pct_fmt(val)
            return f"<span style='color:{color}'>({s})</span>"

        def _disp(df_in: pd.DataFrame) -> pd.DataFrame:
            if df_in.empty:
                return df_in
            out = df_in[["SKU","Sales (Current)","Sales (Compare)","Sales Δ","Δ %"]].copy()
            out["Sales (Current)"] = out["Sales (Current)"].map(money)
            out["Sales (Compare)"] = out["Sales (Compare)"].map(money)
            out["Sales Δ"] = out["Sales Δ"].map(_fmt_diff)
            out["Δ %"] = out["Δ %"].map(_fmt_pct)
            return out

        inc_disp = _disp(inc)
        dec_disp = _disp(dec)

        # Trend leaders (slope-based)
        mom = build_momentum(df_scope[df_scope["WeekEnd"] <= pA.end], "SKU", lookback_weeks=8)
        if not mom.empty:
            mom = mom[(mom["Sales (lookback)"] >= min_sales) | (mom["Units (lookback)"] >= min_units)].copy()
            trend_leaders = mom.sort_values("Slope", ascending=False).head(10).copy()
            trend_leaders_disp = trend_leaders[["SKU","Trend","Slope","Weeks Up","Weeks Down","Sales (lookback)"]].copy()
            trend_leaders_disp["Sales (lookback)"] = trend_leaders_disp["Sales (lookback)"].map(money)
            trend_leaders_disp["Slope"] = trend_leaders_disp["Slope"].map(lambda v: f"{v:,.2f}")
        else:
            trend_leaders_disp = pd.DataFrame(columns=["SKU","Trend","Slope","Weeks Up","Weeks Down","Sales (lookback)"])

        a,b,c = st.columns(3)
        with a:
            st.markdown("**Top Increasing**")
            if not inc_disp.empty:
                st.markdown(inc_disp.to_html(escape=False, index=False, classes='report-table'), unsafe_allow_html=True)
            else:
                st.caption("None.")
        with b:
            st.markdown("**Top Declining**")
            if not dec_disp.empty:
                st.markdown(dec_disp.to_html(escape=False, index=False, classes='report-table'), unsafe_allow_html=True)
            else:
                st.caption("None.")
        with c:
            st.markdown("**Trend Leaders (slope over last 8 weeks)**")
            render_df(trend_leaders_disp, height=320)

    st.divider()
    st.header("Strategic Intelligence")

    # A) Contribution Tree
    
    # A) Contribution Tree
    st.subheader("1) Contribution Tree (Where did change come from?)")
    if compare_mode == "None":
        st.info("Select a comparison mode to use the contribution tree.")
    else:
        sales_a_col = f"Sales ({a_lbl})"
        sales_b_col = f"Sales ({b_lbl})" if b_lbl else "Sales (Comparison)"

        # Level 1: Retailers
        lvl1 = drivers(dfA, dfB, "Retailer").sort_values("Sales_Δ", ascending=False)
        st.markdown("**Level 1 — Retailers**")
        lvl1_disp = lvl1[["Retailer","Sales_A","Sales_B","Sales_Δ","Contribution_%"]].copy()
        lvl1_disp["Sales_A"] = lvl1_disp["Sales_A"].map(money)
        lvl1_disp["Sales_B"] = lvl1_disp["Sales_B"].map(money)
        lvl1_disp["Sales_Δ"] = lvl1_disp["Sales_Δ"].map(money)
        lvl1_disp["Contribution_%"] = lvl1_disp["Contribution_%"].map(pct_fmt)
        lvl1_disp = rename_ab_columns(lvl1_disp, a_lbl, b_lbl)
        render_df(lvl1_disp[["Retailer", sales_a_col, sales_b_col, "Sales_Δ", "Contribution_%"]], height=260)

        pick_r = st.selectbox("Drill into Retailer", options=["(none)"] + sorted(lvl1["Retailer"].astype(str).tolist()), index=0)
        if pick_r != "(none)":
            dfA_r = dfA[dfA["Retailer"] == pick_r].copy()
            dfB_r = dfB[dfB["Retailer"] == pick_r].copy()

            lvl2 = drivers(dfA_r, dfB_r, "Vendor").sort_values("Sales_Δ", ascending=False)
            st.markdown(f"**Level 2 — Vendors inside {pick_r}**")
            lvl2_disp = lvl2[["Vendor","Sales_A","Sales_B","Sales_Δ","Contribution_%"]].copy()
            lvl2_disp["Sales_A"] = lvl2_disp["Sales_A"].map(money)
            lvl2_disp["Sales_B"] = lvl2_disp["Sales_B"].map(money)
            lvl2_disp["Sales_Δ"] = lvl2_disp["Sales_Δ"].map(money)
            lvl2_disp["Contribution_%"] = lvl2_disp["Contribution_%"].map(pct_fmt)
            lvl2_disp = rename_ab_columns(lvl2_disp, a_lbl, b_lbl)
            render_df(lvl2_disp[["Vendor", sales_a_col, sales_b_col, "Sales_Δ", "Contribution_%"]], height=240)

            pick_v = st.selectbox("Drill into Vendor", options=["(none)"] + sorted(lvl2["Vendor"].astype(str).tolist()), index=0)
            if pick_v != "(none)":
                dfA_v = dfA_r[dfA_r["Vendor"] == pick_v].copy()
                dfB_v = dfB_r[dfB_r["Vendor"] == pick_v].copy()

                lvl3 = drivers(dfA_v, dfB_v, "SKU").sort_values("Sales_Δ", ascending=False)
                st.markdown(f"**Level 3 — SKUs inside {pick_r} → {pick_v}**")
                lvl3_disp = lvl3[["SKU","Sales_A","Sales_B","Sales_Δ","Contribution_%"]].copy()
                lvl3_disp["Sales_A"] = lvl3_disp["Sales_A"].map(money)
                lvl3_disp["Sales_B"] = lvl3_disp["Sales_B"].map(money)
                lvl3_disp["Sales_Δ"] = lvl3_disp["Sales_Δ"].map(money)
                lvl3_disp["Contribution_%"] = lvl3_disp["Contribution_%"].map(pct_fmt)
                lvl3_disp = rename_ab_columns(lvl3_disp, a_lbl, b_lbl)
                render_df(lvl3_disp[["SKU", sales_a_col, sales_b_col, "Sales_Δ", "Contribution_%"]], height=240)
    st.subheader("2) SKU Lifecycle (Launch → Growth → Mature → Decline → Dormant)")
    life_df_src = df_hist_for_new if show_full_history_lifecycle else df_scope
    life = lifecycle_table(life_df_src, pA, lookback_weeks=8)
    if life.empty:
        st.caption("Not enough data to compute lifecycle.")
    else:
        # small stage summary
        stage_counts = life["Stage"].value_counts().reset_index()
        stage_counts.columns = ["Stage","Count"]
        left,right = st.columns([1,2])
        with left:
            st.markdown("**Stage Summary**")
            render_df(stage_counts, height=220)
        with right:
            st.markdown("**Lifecycle Detail (trend + momentum + recent change)**")
            life_show = life.copy()
            if min_sales > 0:
                life_show = life_show[life_show["Sales (lookback)"] >= min_sales].copy()

            # Pretty formatting
            life_show["Sales (lookback)"] = life_show["Sales (lookback)"].map(money)
            life_show["Units (lookback)"] = life_show["Units (lookback)"].map(lambda v: f"{v:,.0f}")
            life_show["Last Week Sales"] = life_show["Last Week Sales"].map(money)
            life_show["WoW Sales Δ"] = life_show["WoW Sales Δ"].map(lambda v: "" if pd.isna(v) else money(v))
            life_show["Slope"] = life_show["Slope"].map(lambda v: f"{v:,.2f}") if "Slope" in life_show.columns else life_show.get("Slope")
            for dc in ["First Sale","Last Sale"]:
                if dc in life_show.columns:
                    life_show[dc] = pd.to_datetime(life_show[dc], errors="coerce").dt.date.astype(str)

            cols = [
                "SKU","Stage","Trend","Momentum",
                "Sales (lookback)","Units (lookback)",
                "Last Week Sales","WoW Sales Δ",
                "Slope","Weeks Up","Weeks Down","First Sale","Last Sale",
            ]
            cols = [c for c in cols if c in life_show.columns]
            render_df(life_show[cols].head(60), height=520)

    st.divider()

    # C) Opportunity Detector
    st.subheader("3) Opportunity Detector (Find expansion + gaps)")
    if compare_mode == "None":
        st.info("Select a comparison mode to power opportunity signals (needs a comparison).")
    else:
        opp = opportunity_detector(df_hist_for_new, dfA, dfB, pA)
        tabs = st.tabs(list(opp.keys()))
        for t, (name, odf) in zip(tabs, opp.items()):
            with t:
                if odf.empty:
                    st.caption("No signals found with current filters/thresholds.")
                else:
                    render_df(odf, height=420)

    # Footer
    st.caption("Tip: Use Scope + Driver Level together. Example: Scope=Retailer (Depot), Driver Level=Vendor to see which vendors drove Depot’s change.")
