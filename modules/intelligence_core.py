
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


def _table_height(df, row_px: int = 34, header_px: int = 40, min_px: int = 140, max_px: int = 900):
    """Safe dataframe height helper used across table renders."""
    try:
        n = 0 if df is None else len(df)
    except Exception:
        n = 0
    h = header_px + max(1, int(n)) * row_px
    return max(min_px, min(int(h), int(max_px)))

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
    m["Contribution_%"] = np.where(denom != 0, (m["Sales_Δ"] / denom), 0.0)
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

def lifecycle_table(df_all: pd.DataFrame, p: Period, lookback_weeks: int = 8, scope: str = "SKU (All Retailers)") -> pd.DataFrame:
    """Lifecycle stage for SKUs overall or retailer-specific."""
    by = ["SKU"] if scope == "SKU (All Retailers)" else ["Retailer", "SKU"]
    s = weekly_series(df_all, by)
    base_cols = (["Retailer", "SKU"] if len(by) == 2 else ["SKU"]) + [
        "Stage", "Trend", "Sales (lookback)", "Units (lookback)",
        "Last Week Sales", "WoW Sales Δ", "Weeks Up", "Weeks Down", "Weeks With Sales"
    ]
    if s.empty:
        return pd.DataFrame(columns=base_cols)

    anchor = p.end
    lb_start = anchor - pd.Timedelta(days=(7 * lookback_weeks - 1))
    full_hist = df_all[df_all["Sales"] > 0].copy()
    first_week = full_hist.groupby(by)["WeekEnd"].min() if not full_hist.empty else pd.Series(dtype="datetime64[ns]")
    last_week = full_hist.groupby(by)["WeekEnd"].max() if not full_hist.empty else pd.Series(dtype="datetime64[ns]")

    out = []
    for key, g in s.groupby(by):
        if len(by) == 1:
            sku = key[0] if isinstance(key, tuple) else key
            entity = {"SKU": sku}
            lookup_key = sku
        else:
            retailer, sku = key
            entity = {"Retailer": retailer, "SKU": sku}
            lookup_key = key

        g = g.sort_values("WeekEnd")
        gw = g[(g["WeekEnd"] >= lb_start) & (g["WeekEnd"] <= anchor)].copy()
        sales_lb = float(gw["Sales"].sum()) if not gw.empty else 0.0
        units_lb = float(gw["Units"].sum()) if (not gw.empty and "Units" in gw.columns) else 0.0
        last_w_sales = float(gw["Sales"].iloc[-1]) if len(gw) >= 1 else 0.0
        prev_w_sales = float(gw["Sales"].iloc[-2]) if len(gw) >= 2 else 0.0
        wow_sales = last_w_sales - prev_w_sales if len(gw) >= 2 else np.nan
        weeks_with_sales = int((pd.to_numeric(gw.get("Sales", 0.0), errors="coerce").fillna(0.0) > 0).sum()) if not gw.empty else 0

        fw = first_week.get(lookup_key, pd.NaT)
        lw = last_week.get(lookup_key, pd.NaT)

        trend_symbol = "→"
        weeks_up = 0
        weeks_down = 0
        stage = "Mature"

        if pd.notna(fw) and (fw >= p.start) and (fw <= p.end):
            stage = "Launch"
            trend_symbol = "↗"
        elif pd.isna(lw):
            stage = "Inactive 12+ Weeks"
            trend_symbol = "↓"
        else:
            weeks_since_sale = int((anchor.normalize() - pd.to_datetime(lw).normalize()).days // 7)
            if weeks_since_sale >= 12:
                stage = "Inactive 12+ Weeks"
                trend_symbol = "↓"
            elif weeks_with_sales == 0:
                stage = "Dormant"
                trend_symbol = "→"
            else:
                trend_label, slope, wu, wd = classify_trend(gw["Sales"], min_weeks=min(lookback_weeks, max(2, len(gw))))
                weeks_up, weeks_down = int(wu), int(wd)
                if trend_label == "Increasing":
                    stage = "Growth"
                    trend_symbol = "↗"
                elif trend_label == "Declining":
                    stage = "Decline"
                    trend_symbol = "↘"
                else:
                    stage = "Mature" if weeks_with_sales >= max(2, lookback_weeks // 2) else "Dormant"
                    trend_symbol = "→"

        out.append({
            **entity,
            "Stage": stage,
            "Trend": trend_symbol,
            "Sales (lookback)": sales_lb,
            "Units (lookback)": units_lb,
            "Last Week Sales": last_w_sales,
            "WoW Sales Δ": float(wow_sales) if not pd.isna(wow_sales) else np.nan,
            "Weeks Up": int(weeks_up),
            "Weeks Down": int(weeks_down),
            "Weeks With Sales": int(weeks_with_sales),
        })
    out_df = pd.DataFrame(out)
    if not out_df.empty:
        stage_order = {"Launch": 0, "Growth": 1, "Mature": 2, "Decline": 3, "Dormant": 4, "Inactive 12+ Weeks": 5}
        out_df["__stage_order"] = out_df["Stage"].map(stage_order).fillna(99)
        out_df = out_df.sort_values(["__stage_order", "Sales (lookback)"], ascending=[True, False]).drop(columns=["__stage_order"])
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

def fmt_currency(x: float) -> str:
    return money(float(x))

def fmt_int(x: float) -> str:
    return f"{float(x):,.0f}"

def pct_fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    if x == np.inf:
        return "∞"
    if x == -np.inf:
        return "-∞"
    return f"{x*100:,.1f}%"


def month_year_display(period_str: str) -> str:
    try:
        return pd.Period(str(period_str), freq="M").strftime("%B %Y")
    except Exception:
        return str(period_str)


def _display_to_period(label: str) -> str:
    try:
        return pd.to_datetime(str(label), format="%B %Y").to_period("M").strftime("%Y-%m")
    except Exception:
        return str(label)


def signed_money_html(v: float) -> str:
    color = "#2e7d32" if v > 0 else ("#c62828" if v < 0 else "var(--text-color)")
    sign = "+" if v > 0 else ""
    return f"<span style='color:{color}; font-weight:700'>{sign}{money(v)}</span>"


def signed_int_html(v: float) -> str:
    color = "#2e7d32" if v > 0 else ("#c62828" if v < 0 else "var(--text-color)")
    sign = "+" if v > 0 else ""
    return f"<span style='color:{color}; font-weight:700'>{sign}{float(v):,.0f}</span>"


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

def leader_sales_card(label: str, name: str, current_sales: float, previous_sales: float,
                      current_units: Optional[float] = None, previous_units: Optional[float] = None):
    sales_delta = float(current_sales) - float(previous_sales)
    sales_pct = pct_change(float(current_sales), float(previous_sales))
    sales_color = "#2e7d32" if sales_delta > 0 else ("#c62828" if sales_delta < 0 else "var(--text-color)")
    sales_arrow = "▲" if sales_delta > 0 else ("▼" if sales_delta < 0 else "•")
    sales_pct_html = "" if pd.isna(sales_pct) else f'<span class="delta-pct" style="color:{sales_color}">({pct_fmt(sales_pct)})</span>'

    if current_units is None:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">{html.escape(str(label))}</div>
                <div class="kpi-big-name">{html.escape(str(name))}</div>
                <div class="kpi-value">{money(float(current_sales))}</div>
                <div class="kpi-delta" style="color:{sales_color}">
                    <span class="delta-abs">{sales_arrow} {money(sales_delta)}</span>{sales_pct_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    cur_u = float(current_units)
    prev_u = float(previous_units or 0.0)
    units_delta = cur_u - prev_u
    units_pct = pct_change(cur_u, prev_u)
    units_color = "#2e7d32" if units_delta > 0 else ("#c62828" if units_delta < 0 else "var(--text-color)")
    units_arrow = "▲" if units_delta > 0 else ("▼" if units_delta < 0 else "•")
    units_pct_html = "" if pd.isna(units_pct) else f'<span class="delta-pct" style="color:{units_color}">({pct_fmt(units_pct)})</span>'

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{html.escape(str(label))}</div>
            <div class="kpi-big-name">{html.escape(str(name))}</div>
            <div style="display:flex; gap:18px; align-items:flex-start;">
                <div style="flex:1 1 0; min-width:0;">
                    <div style="font-size:11px; opacity:0.75; margin-bottom:2px;">Sales</div>
                    <div class="kpi-value">{money(float(current_sales))}</div>
                    <div class="kpi-delta" style="color:{sales_color}">
                        <span class="delta-abs">{sales_arrow} {money(sales_delta)}</span>{sales_pct_html}
                    </div>
                </div>
                <div style="flex:1 1 0; min-width:0;">
                    <div style="font-size:11px; opacity:0.75; margin-bottom:2px;">Units</div>
                    <div class="kpi-value">{cur_u:,.0f}</div>
                    <div class="kpi-delta" style="color:{units_color}">
                        <span class="delta-abs">{units_arrow} {units_delta:,.0f}</span>{units_pct_html}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def selection_total_card(label: str, cur_kpi: Dict[str, float], cmp_kpi: Dict[str, float]):
    sales = float(cur_kpi.get("Sales", 0.0))
    prev_sales = float(cmp_kpi.get("Sales", 0.0))
    sales_delta = sales - prev_sales
    sales_pct = pct_change(sales, prev_sales)
    sales_color = "#2e7d32" if sales_delta > 0 else ("#c62828" if sales_delta < 0 else "var(--text-color)")
    sales_arrow = "▲" if sales_delta > 0 else ("▼" if sales_delta < 0 else "•")

    units = float(cur_kpi.get("Units", 0.0))
    prev_units = float(cmp_kpi.get("Units", 0.0))
    units_delta = units - prev_units
    units_pct = pct_change(units, prev_units)
    units_color = "#2e7d32" if units_delta > 0 else ("#c62828" if units_delta < 0 else "var(--text-color)")
    units_arrow = "▲" if units_delta > 0 else ("▼" if units_delta < 0 else "•")

    asp = float(cur_kpi.get("ASP", 0.0))
    sales_pct_html = f" ({pct_fmt(sales_pct)})" if not pd.isna(sales_pct) else ""
    units_pct_html = f" ({pct_fmt(units_pct)})" if not pd.isna(units_pct) else ""

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{label}</div>
            <div class="kpi-sub" style="font-size:12px;opacity:0.75;margin-bottom:2px;">Total Sales</div>
            <div class="kpi-value">{money(sales)}</div>
            <div class="kpi-delta" style="color:{sales_color};margin-bottom:8px;">{sales_arrow} {money(sales_delta)}{sales_pct_html}</div>
            <div class="kpi-sub" style="font-size:12px;opacity:0.75;margin-bottom:2px;">Total Units</div>
            <div class="kpi-sub" style="font-size:16px;font-weight:700;color:var(--text-color);">{units:,.0f}</div>
            <div class="kpi-delta" style="color:{units_color};margin-bottom:8px;">{units_arrow} {units_delta:,.0f}{units_pct_html} &nbsp; • &nbsp; ASP {money(asp)}</div>
            <div class="kpi-sub" style="margin-top:6px;">Active SKUs: {int(cur_kpi.get('Active SKUs', 0)):,} &nbsp; • &nbsp; Retailers: {int(cur_kpi.get('Active Retailers', 0)):,} &nbsp; • &nbsp; Vendors: {int(cur_kpi.get('Active Vendors', 0)):,}</div>
            <div class="kpi-sub" style="margin-top:6px;">Avg Units / SKU {((units / cur_kpi.get('Active SKUs', 0)) if cur_kpi.get('Active SKUs', 0) else 0):,.1f} &nbsp; • &nbsp; Avg Sales / SKU {money((sales / cur_kpi.get('Active SKUs', 0)) if cur_kpi.get('Active SKUs', 0) else 0)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def top_two_card(label: str, entries: List[dict]):
    rows = []
    for item in entries[:2]:
        name = html.escape(str(item.get("name", "")))
        sales = float(item.get("sales", 0.0))
        other_sales = float(item.get("other_sales", 0.0))
        share = float(item.get("share", np.nan))
        delta = sales - other_sales
        pct = pct_change(sales, other_sales)
        color = "#2e7d32" if delta > 0 else ("#c62828" if delta < 0 else "var(--text-color)")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "•")
        pct_html = f" ({pct_fmt(pct)})" if not pd.isna(pct) else ""
        share_html = f"{share*100:,.1f}% of total" if not pd.isna(share) else ""
        rows.append(
            f"<div class='top-two-item'>"
            f"<div class='kpi-big-name'>{name}</div>"
            f"<div class='kpi-sub' style='font-size:16px;font-weight:700;color:var(--text-color)'>{money(sales)}</div>"
            f"<div class='kpi-delta' style='color:{color}'>{arrow} {money(delta)}{pct_html}</div>"
            f"<div class='kpi-sub'>{share_html}</div>"
            f"</div>"
        )
    if not rows:
        rows = ["<div class='kpi-sub'>None</div>"]
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{label}</div>
            {''.join(rows)}
        </div>
        """,
        unsafe_allow_html=True,
    )

def count_sales_card(label: str, count_value: int, sales_value: float, color: str = "var(--text-color)", signed_sales: bool = False, pct: float = np.nan):
    sales_txt = money(abs(float(sales_value)))
    if signed_sales and sales_value > 0:
        sales_txt = "+" + sales_txt
    elif signed_sales and sales_value < 0:
        sales_txt = "-" + sales_txt
    pct_html = "" if pd.isna(pct) else f'<div class="kpi-delta" style="color:{color}">({pct_fmt(pct)})</div>'
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{label}</div>
            <div class="kpi-value" style="color:{color}">{count_value:,}</div>
            <div class="kpi-sub" style="color:{color}">Sales: {sales_txt}</div>
            {pct_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_df(df: pd.DataFrame, height: int = 320):
    st.dataframe(df, use_container_width=True, height=height, hide_index=True)


@st.cache_data(show_spinner=False)
def build_weekly_detail_table_cached(
    d_in: pd.DataFrame,
    pivot_dim: str,
    metric_col: str,
    weeks_to_show: int,
    avg_basis: str,
) -> pd.DataFrame:
    """Build the weekly detail table with independent display + average windows.

    Cached so the Standard Intelligence tab feels much faster when changing other controls.
    """
    d = d_in.copy()
    if d.empty or pivot_dim not in d.columns or metric_col not in d.columns:
        return pd.DataFrame(columns=[pivot_dim, "Δ vs prior week", "Average Value", "Current vs Avg"])

    d["WeekEnd"] = pd.to_datetime(d["WeekEnd"], errors="coerce")
    d = d[d["WeekEnd"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=[pivot_dim, "Δ vs prior week", "Average Value", "Current vs Avg"])

    all_weeks = sorted(pd.to_datetime(d["WeekEnd"].dropna().unique()).tolist())
    if not all_weeks:
        return pd.DataFrame(columns=[pivot_dim, "Δ vs prior week", "Average Value", "Current vs Avg"])

    weeks_to_show = int(max(1, weeks_to_show))
    display_weeks = all_weeks[-weeks_to_show:]

    wk_vals = d.groupby([pivot_dim, "WeekEnd"], as_index=False).agg(Value=(metric_col, "sum"))
    wk_vals = wk_vals[wk_vals["WeekEnd"].isin(display_weeks)].copy()
    wk_vals["Week"] = wk_vals["WeekEnd"].dt.strftime("%Y-%m-%d")
    piv = wk_vals.pivot_table(index=pivot_dim, columns="Week", values="Value", aggfunc="sum", fill_value=0.0)
    piv = piv.reindex(sorted(piv.index.tolist()))

    avg_source = d.groupby([pivot_dim, "WeekEnd"], as_index=False).agg(Value=(metric_col, "sum"))
    avg_source["WeekEnd"] = pd.to_datetime(avg_source["WeekEnd"], errors="coerce")

    avg_weeks = []
    if isinstance(avg_basis, str) and avg_basis.endswith("weeks"):
        try:
            avg_n = int(str(avg_basis).split()[0])
        except Exception:
            avg_n = 8
        avg_weeks = all_weeks[-avg_n:]
    else:
        avg_period = _display_to_period(avg_basis)
        avg_source = avg_source[avg_source["WeekEnd"].dt.to_period("M").astype(str) == str(avg_period)].copy()
        avg_weeks = sorted(pd.to_datetime(avg_source["WeekEnd"].dropna().unique()).tolist())

    if avg_weeks:
        avg_source = avg_source[avg_source["WeekEnd"].isin(avg_weeks)].copy()
        avg_tbl = avg_source.groupby(pivot_dim, as_index=False).agg(**{"Average Value": ("Value", "mean")})
    else:
        avg_tbl = pd.DataFrame(columns=[pivot_dim, "Average Value"])

    out = piv.reset_index().merge(avg_tbl, on=pivot_dim, how="left")
    out["Average Value"] = pd.to_numeric(out["Average Value"], errors="coerce").fillna(0.0)

    last_key = display_weeks[-1].strftime("%Y-%m-%d") if display_weeks else None
    prev_key = display_weeks[-2].strftime("%Y-%m-%d") if len(display_weeks) >= 2 else None
    last_vals = pd.to_numeric(out.get(last_key, 0.0), errors="coerce").fillna(0.0) if last_key else 0.0
    prev_vals = pd.to_numeric(out.get(prev_key, 0.0), errors="coerce").fillna(0.0) if prev_key else 0.0
    out["Δ vs prior week"] = last_vals - prev_vals
    out["Current vs Avg"] = last_vals - out["Average Value"]

    ordered_cols = [pivot_dim] + [w.strftime("%Y-%m-%d") for w in display_weeks] + ["Δ vs prior week", "Average Value", "Current vs Avg"]
    out = out[[c for c in ordered_cols if c in out.columns]].copy()
    return out


def render_data_management_center(vm: pd.DataFrame, store: pd.DataFrame):
    st.subheader("Data Management Center")
    st.caption("Manage CSV backup/restore, ingest new workbooks, and review coverage by year.")
    base_data_dir = DATA_DIR
    base_data_dir.mkdir(parents=True, exist_ok=True)

    # CSV backup / restore
    backup_df = load_store().copy()
    csv_bytes = backup_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Backup CSV",
        data=csv_bytes,
        file_name="cornerstone_sales_backup.csv",
        mime="text/csv",
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        restore = st.file_uploader("Restore Backup CSV", type=["csv"], key="restore_backup_csv")
        if restore is not None and st.button("Restore Backup CSV", key="restore_backup_csv_btn", use_container_width=True):
            try:
                restored = pd.read_csv(restore)
                required = ["Retailer","SKU","Units","UnitPrice","StartDate","EndDate","SourceFile"]
                for c in required:
                    if c not in restored.columns:
                        restored[c] = np.nan
                restored = restored[required].copy()
                save_store(restored)
                st.success(f"Backup restored from CSV with {len(restored):,} rows.")
            except Exception as e:
                st.error(f"Restore failed: {e}")
    with c2:
        year_ingest = st.number_input("Year hint for new workbook(s)", min_value=2010, max_value=2100, value=date.today().year, step=1, key="dmc_year")
        replace_year = st.toggle("Replace existing rows for that year before ingest", value=False, key="dmc_replace_year")
        uploads = st.file_uploader("Upload weekly workbook(s)", type=["xlsx"], accept_multiple_files=True, key="dmc_uploads")
        if uploads and st.button("Ingest Uploaded Workbook(s)", key="dmc_ingest_btn", use_container_width=True):
            try:
                store_cur = load_store()
                if replace_year:
                    sdates = pd.to_datetime(store_cur.get("EndDate"), errors="coerce").fillna(pd.to_datetime(store_cur.get("StartDate"), errors="coerce"))
                    keep_mask = sdates.dt.year != int(year_ingest)
                    store_cur = store_cur.loc[keep_mask].copy()
                added_rows = 0
                for up in uploads:
                    raw = read_weekly_workbook(up, int(year_ingest))
                    store_cur = pd.concat([store_cur, raw], ignore_index=True)
                    added_rows += len(raw)
                save_store(store_cur)
                st.success(f"Ingested {added_rows:,} rows from {len(uploads)} workbook(s).")
            except Exception as e:
                st.error(f"Ingest failed: {e}")

    df_all = enrich_sales(load_store(), load_vendor_map())
    if df_all.empty:
        st.info("No sales data loaded yet.")
        return

    cov = (
        df_all.groupby("Year", as_index=False)
        .agg(
            Weeks=("WeekEnd", lambda s: pd.to_datetime(s, errors="coerce").dt.normalize().nunique()),
            Rows=("SKU", "size"),
            Retailers=("Retailer", "nunique"),
            Vendors=("Vendor", "nunique"),
            SKUs=("SKU", "nunique"),
            Units=("Units", "sum"),
            Sales=("Sales", "sum"),
        )
        .sort_values("Year")
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Years Loaded", f"{cov['Year'].nunique():,}")
    k2.metric("Total Weeks", f"{int(cov['Weeks'].sum()):,}")
    k3.metric("Total Rows", f"{int(cov['Rows'].sum()):,}")
    k4.metric("Total Sales", money(float(cov["Sales"].sum())))

    st.markdown("### Coverage by Year")
    cov_disp = cov.copy()
    cov_disp["Units"] = cov_disp["Units"].map(lambda x: f"{x:,.0f}")
    cov_disp["Sales"] = cov_disp["Sales"].map(money)
    st.dataframe(cov_disp, use_container_width=True, hide_index=True)

    year_options = cov["Year"].dropna().astype(int).tolist()
    if year_options:
        sel_year = st.selectbox("Show uploaded weeks for year", options=year_options, index=len(year_options)-1)
        yw = df_all[df_all["Year"] == int(sel_year)].copy()
        yw["WeekEnd"] = pd.to_datetime(yw["WeekEnd"], errors="coerce").dt.date.astype(str)
        weeks = (
            yw.groupby("WeekEnd", as_index=False)
              .agg(Rows=("SKU", "size"), Retailers=("Retailer", "nunique"), Vendors=("Vendor", "nunique"), SKUs=("SKU", "nunique"), Units=("Units", "sum"), Sales=("Sales", "sum"))
              .sort_values("WeekEnd")
        )
        weeks["Units"] = weeks["Units"].map(lambda x: f"{x:,.0f}")
        weeks["Sales"] = weeks["Sales"].map(money)
        st.markdown(f"### Uploaded Weeks in {sel_year}")
        st.dataframe(weeks, use_container_width=True, hide_index=True)

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

        analysis_view = st.radio("Analysis View", ["Standard Intelligence", "Month / Year Compare", "Multi Month / Year Compare", "Data Management Center"], index=0)
        multi_granularity = "Month"
        current_labels_sel: List[str] = []
        compare_labels_sel: List[str] = []
        if analysis_view == "Data Management Center":
            timeframe = "YTD"
            compare_mode = "None"
        elif analysis_view == "Standard Intelligence":
            timeframe_options = ["Week (latest)", "Last 4 weeks", "Last 8 weeks", "Last 13 weeks", "Last 26 weeks", "Last 52 weeks", "YTD"]
            timeframe_index = 2
            timeframe = st.selectbox("Timeframe", timeframe_options, index=timeframe_index)
            compare_mode = st.selectbox("Compare", ["None", "Prior period (same length)", "YoY (same dates)"], index=1)
        elif analysis_view == "Month / Year Compare":
            multi_granularity = st.selectbox("Compare By", ["Month", "Year"], index=0, key="my_compare_by")
            if multi_granularity == "Month":
                timeframe = "Custom Months"
                period_options = available_month_labels(df_all)
            else:
                timeframe = "Custom Years"
                period_options = available_year_labels(df_all)
            default_current = period_options[-1] if period_options else None
            default_compare = period_options[-2] if len(period_options) > 1 else default_current
            current_one = st.selectbox(f"Current {multi_granularity}", options=period_options, index=(len(period_options)-1 if period_options else 0))
            compare_one = st.selectbox(f"Compare {multi_granularity}", options=period_options, index=(len(period_options)-2 if len(period_options)>1 else 0))
            current_labels_sel = [current_one] if current_one else []
            compare_labels_sel = [compare_one] if compare_one else []
            compare_mode = "Custom selection" if compare_labels_sel else "None"
        else:
            multi_granularity = st.selectbox("Compare By", ["Month", "Year"], index=0, key="multi_compare_by")
            if multi_granularity == "Month":
                timeframe = "Custom Months"
                period_options = available_month_labels(df_all)
                default_current = period_options[-1:] if period_options else []
                default_compare = period_options[-2:-1] if len(period_options) > 1 else default_current
            else:
                timeframe = "Custom Years"
                period_options = available_year_labels(df_all)
                default_current = period_options[-1:] if period_options else []
                default_compare = period_options[-2:-1] if len(period_options) > 1 else default_current
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

    if analysis_view == "Data Management Center":
        render_data_management_center(vm, store)
        return

    # Choose current / comparison periods
    custom_labels = None
    if analysis_view in ("Month / Year Compare", "Multi Month / Year Compare"):
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
            m = m.sort_values(["Sales_A", level], ascending=[False, True])
            row = m.iloc[0]
            return str(row[level]), float(row["Sales_A"]), float(row["Sales_B"]), float(row.get("Units_A", 0.0)), float(row.get("Units_B", 0.0))

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

        def _top_two_in_selection(df_in: pd.DataFrame, level: str):
            if df_in.empty:
                return []
            g = df_in.groupby(level, as_index=False).agg(Sales=("Sales","sum")).sort_values("Sales", ascending=False)
            return [(str(r[level]), float(r["Sales"])) for _, r in g.head(2).iterrows()]

        def _top_decrease(level: str):
            a = dfA.groupby(level, as_index=False).agg(Sales=("Sales","sum"))
            b = dfB.groupby(level, as_index=False).agg(Sales=("Sales","sum"))
            m = a.merge(b, on=level, how="outer", suffixes=("_A","_B")).fillna(0.0)
            if m.empty:
                return None
            m["Δ"] = m["Sales_A"] - m["Sales_B"]
            m = m.sort_values("Δ", ascending=True)
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

        def kdelta(key: str) -> str:
            if compare_mode == "None":
                return ""
            cur = float(kA.get(key, 0.0))
            prev = float(kB.get(key, 0.0))
            if key in ("Sales", "ASP"):
                return _delta_html(cur, prev, is_money=True)
            if key in ("Units", "Active SKUs"):
                return _delta_html(cur, prev, is_money=False)
            d = cur - prev
            green = "#2e7d32"
            red = "#c62828"
            color = green if d > 0 else (red if d < 0 else "var(--text-color)")
            arrow = '▲ ' if d > 0 else ('▼ ' if d < 0 else '')
            return f"<span class='delta-abs' style='color:{color}'>{arrow}{pct_fmt(pct_change(cur, prev))}</span>"

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


    # 1) KPI rows
    def _new_lost_stats(df_cur: pd.DataFrame, df_cmp: pd.DataFrame):
        cur_s = df_cur.groupby("SKU", as_index=False).agg(Sales_A=("Sales","sum"))
        cmp_s = df_cmp.groupby("SKU", as_index=False).agg(Sales_B=("Sales","sum"))
        m = cur_s.merge(cmp_s, on="SKU", how="outer").fillna(0.0)
        new_df = m[(m["Sales_A"] > 0) & (m["Sales_B"] <= 0)].copy()
        lost_df = m[(m["Sales_B"] > 0) & (m["Sales_A"] <= 0)].copy()
        return new_df, lost_df
    
    def _top_entity_in_selection(df_sel: pd.DataFrame, level: str):
        if df_sel.empty:
            return None
        g = df_sel.groupby(level, as_index=False).agg(Sales=("Sales","sum"))
        if g.empty:
            return None
        g = g.sort_values(["Sales", level], ascending=[False, True])
        r = g.iloc[0]
        return str(r[level]), float(r["Sales"])

    def _top_two_with_compare(df_sel: pd.DataFrame, df_other: pd.DataFrame, level: str):
        if df_sel.empty:
            return []
        cur = df_sel.groupby(level, as_index=False).agg(Sales=("Sales", "sum"))
        oth = df_other.groupby(level, as_index=False).agg(Other_Sales=("Sales", "sum")) if not df_other.empty else pd.DataFrame(columns=[level, "Other_Sales"])
        m = cur.merge(oth, on=level, how="left").fillna(0.0)
        if m.empty:
            return []
        total_sales = float(m["Sales"].sum())
        m = m.sort_values(["Sales", level], ascending=[False, True]).head(2)
        out = []
        for _, r in m.iterrows():
            sales = float(r["Sales"])
            out.append({
                "name": str(r[level]),
                "sales": sales,
                "other_sales": float(r["Other_Sales"]),
                "share": (sales / total_sales) if total_sales else np.nan,
            })
        return out
    
    def _top_decrease(level: str):
        a = dfA.groupby(level, as_index=False).agg(Sales_A=("Sales","sum"))
        b = dfB.groupby(level, as_index=False).agg(Sales_B=("Sales","sum"))
        m = a.merge(b, on=level, how="outer").fillna(0.0)
        m["Δ"] = m["Sales_A"] - m["Sales_B"]
        if m.empty:
            return None
        m = m.sort_values("Δ", ascending=True)
        r = m.iloc[0]
        return str(r[level]), float(r["Sales_A"]), float(r["Sales_B"])
    
    if analysis_view == "Month / Year Compare":
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
            if iR: biggest_increase_card("Retailer w/ Biggest Increase", iR[0], iR[1], iR[2])
        with i2:
            if iV: biggest_increase_card("Vendor w/ Biggest Increase", iV[0], iV[1], iV[2])
        with i3:
            if iS: biggest_increase_card("SKU w/ Biggest Increase", iS[0], iS[1], iS[2])

        d1, d2, d3 = st.columns(3)
        decR = _top_decrease("Retailer")
        decV = _top_decrease("Vendor")
        decS = _top_decrease("SKU")
        with d1:
            if decR: biggest_increase_card("Retailer w/ Biggest Decrease", decR[0], decR[1], decR[2])
        with d2:
            if decV: biggest_increase_card("Vendor w/ Biggest Decrease", decV[0], decV[1], decV[2])
        with d3:
            if decS: biggest_increase_card("SKU w/ Biggest Decrease", decS[0], decS[1], decS[2])

    elif analysis_view == "Multi Month / Year Compare":
        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("Total Sales", money(kA["Sales"]), kdelta("Sales"))
        with c2: kpi_card("Total Units", f"{kA['Units']:,.0f}", kdelta("Units"))
        with c3: kpi_card("Avg Selling Price", money(kA["ASP"]), kdelta("ASP"))

        cur_sku = dfA.groupby("SKU", as_index=False).agg(Sales=("Sales","sum"))
        cmp_sku = dfB.groupby("SKU", as_index=False).agg(Sales=("Sales","sum"))
        new_df = cur_sku.merge(cmp_sku.rename(columns={"Sales":"Compare_Sales"}), on="SKU", how="left").fillna(0.0)
        new_df = new_df[(new_df["Sales"] > 0) & (new_df["Compare_Sales"] <= 0)].copy()
        lost_df = cmp_sku.merge(cur_sku.rename(columns={"Sales":"Current_Sales"}), on="SKU", how="left").fillna(0.0)
        lost_df = lost_df[(lost_df["Sales"] > 0) & (lost_df["Current_Sales"] <= 0)].copy()
        new_sales = float(new_df["Sales"].sum())
        lost_sales = float(lost_df["Sales"].sum())
        net_count = int(len(new_df)) - int(len(lost_df))
        net_sales = new_sales - lost_sales
        net_pct = (net_sales / lost_sales) if lost_sales != 0 else (np.nan if net_sales == 0 else np.inf)

        n1, n2, n3 = st.columns(3)
        with n1: count_sales_card("New SKUs", int(len(new_df)), new_sales, color="#2e7d32", signed_sales=True)
        with n2: count_sales_card("Lost SKUs", int(len(lost_df)), -lost_sales, color="#c62828", signed_sales=True)
        with n3: count_sales_card("Net New vs Lost", net_count, net_sales, color=("#2e7d32" if net_sales > 0 else ("#c62828" if net_sales < 0 else "var(--text-color)")), signed_sales=True, pct=net_pct)

        st.write("")
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        trA = _top_entity_in_selection(dfA, "Retailer")
        trB = _top_entity_in_selection(dfB, "Retailer")
        tvA = _top_entity_in_selection(dfA, "Vendor")
        tvB = _top_entity_in_selection(dfB, "Vendor")
        with s1: selection_total_card(f"{a_lbl} Total", kA, kB)
        with s2: selection_total_card(f"{b_lbl} Total", kB, kA)
        with s3:
            if trA: leader_sales_card(f"Top Retailer ({a_lbl})", trA[0], trA[1], 0.0)
        with s4:
            if trB: leader_sales_card(f"Top Retailer ({b_lbl})", trB[0], trB[1], 0.0)
        with s5:
            if tvA: leader_sales_card(f"Top Vendor ({a_lbl})", tvA[0], tvA[1], 0.0)
        with s6:
            if tvB: leader_sales_card(f"Top Vendor ({b_lbl})", tvB[0], tvB[1], 0.0)

        st.write("")
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        tsA = _top_entity_in_selection(dfA, "SKU")
        tsB = _top_entity_in_selection(dfB, "SKU")
        iR = _top_by_increase("Retailer")
        iV = _top_by_increase("Vendor")
        iS = _top_by_increase("SKU")
        with t1:
            if tsA: leader_sales_card(f"Top SKU ({a_lbl})", tsA[0], tsA[1], 0.0)
        with t2:
            if tsB: leader_sales_card(f"Top SKU ({b_lbl})", tsB[0], tsB[1], 0.0)
        with t3:
            if iR: biggest_increase_card("Retailer w/ Biggest Increase", iR[0], iR[1], iR[2])
        with t4:
            if iV: biggest_increase_card("Vendor w/ Biggest Increase", iV[0], iV[1], iV[2])
        with t5:
            if iS: biggest_increase_card("SKU w/ Biggest Increase", iS[0], iS[1], iS[2])
        with t6:
            st.empty()

        st.write("")
        d1, d2, d3 = st.columns(3)
        decR = _top_decrease("Retailer")
        decV = _top_decrease("Vendor")
        decS = _top_decrease("SKU")
        with d1:
            if decR: biggest_increase_card("Retailer w/ Biggest Decrease", decR[0], decR[1], decR[2])
        with d2:
            if decV: biggest_increase_card("Vendor w/ Biggest Decrease", decV[0], decV[1], decV[2])
        with d3:
            if decS: biggest_increase_card("SKU w/ Biggest Decrease", decS[0], decS[1], decS[2])
    else:
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: kpi_card("Total Sales", money(kA["Sales"]), kdelta("Sales"))
        with c2: kpi_card("Total Units", f"{kA['Units']:,.0f}", kdelta("Units"))
        with c3: kpi_card("Avg Selling Price", money(kA["ASP"]), kdelta("ASP"))
        with c4: kpi_card("Active SKUs", f"{kA['Active SKUs']:,}", kdelta("Active SKUs"))
        with c5: kpi_card("First Sales", f"{len(first_ever):,}" , "")
        with c6: kpi_card("New Placements", f"{len(placements):,}", "")
    
        st.write("")
        r1c1, r1c2, r1c3 = st.columns(3)
        tR = _top_by_current("Retailer")
        tV = _top_by_current("Vendor")
        tS = _top_by_current("SKU")
        with r1c1:
            if tR: leader_sales_card("Top Retailer (Sales)", tR[0], tR[1], tR[2], tR[3], tR[4])
        with r1c2:
            if tV: leader_sales_card("Top Vendor (Sales)", tV[0], tV[1], tV[2], tV[3], tV[4])
        with r1c3:
            if tS: leader_sales_card("Top SKU (Sales)", tS[0], tS[1], tS[2], tS[3], tS[4])
    
        r2c1, r2c2, r2c3 = st.columns(3)
        iR = _top_by_increase("Retailer")
        iV = _top_by_increase("Vendor")
        iS = _top_by_increase("SKU")
        with r2c1:
            if iR: biggest_increase_card("Retailer w/ Biggest Increase", iR[0], iR[1], iR[2])
        with r2c2:
            if iV: biggest_increase_card("Vendor w/ Biggest Increase", iV[0], iV[1], iV[2])
        with r2c3:
            if iS: biggest_increase_card("SKU w/ Biggest Increase", iS[0], iS[1], iS[2])
    
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
    
    if analysis_view == "Month / Year Compare":
        st.subheader("Current Only / Compare Only Activity")
        fe_cur = first_sale_ever(df_hist_for_new, pA)
        fe_cmp = first_sale_ever(df_hist_for_new, pB) if pB is not None else pd.DataFrame()
        pl_cur = new_placement(df_hist_for_new, pA)
        pl_cmp = new_placement(df_hist_for_new, pB) if pB is not None else pd.DataFrame()
        cur_s = dfA.groupby("SKU", as_index=False).agg(Current_Sales=("Sales","sum"))
        cmp_s = dfB.groupby("SKU", as_index=False).agg(Compare_Sales=("Sales","sum"))
        lost = cmp_s.merge(cur_s, on="SKU", how="left").fillna(0.0)
        lost = lost[(lost["Compare_Sales"] > 0) & (lost["Current_Sales"] <= 0)].copy().sort_values("Compare_Sales", ascending=False)
    
        a1, a2 = st.columns(2)
        with a1:
            st.markdown(f"**First Ever Sales — {a_lbl}**")
            if fe_cur.empty: st.caption("None.")
            else: render_df(fe_cur[["SKU","FirstWeek","FirstRetailer","FirstVendor"]].rename(columns={"FirstWeek":"First Week","FirstRetailer":"Retailer","FirstVendor":"Vendor"}), height=240)
            st.markdown(f"**New Placements — {a_lbl}**")
            if pl_cur.empty: st.caption("None.")
            else: render_df(pl_cur[["SKU","Retailer","FirstWeek","Vendor"]].rename(columns={"FirstWeek":"First Week"}), height=240)
        with a2:
            st.markdown(f"**First Ever Sales — {b_lbl}**")
            if fe_cmp.empty: st.caption("None.")
            else: render_df(fe_cmp[["SKU","FirstWeek","FirstRetailer","FirstVendor"]].rename(columns={"FirstWeek":"First Week","FirstRetailer":"Retailer","FirstVendor":"Vendor"}), height=240)
            st.markdown(f"**New Placements — {b_lbl}**")
            if pl_cmp.empty: st.caption("None.")
            else: render_df(pl_cmp[["SKU","Retailer","FirstWeek","Vendor"]].rename(columns={"FirstWeek":"First Week"}), height=240)
    
        st.markdown("**Lost Activity — sold in compare, zero in current**")
        if lost.empty: st.caption("None.")
        else:
            show_lost = lost[["SKU","Compare_Sales"]].rename(columns={"Compare_Sales":"Compare Sales"}).copy()
            show_lost["Compare Sales"] = show_lost["Compare Sales"].map(money)
            render_df(show_lost, height=280)
    
        st.divider()
        st.subheader("Comparison Detail")
        pivot_dim = st.selectbox("Compare rows by", options=["Retailer","Vendor"], index=0, key="multi_compare_dim")
        comp_a = dfA.groupby(pivot_dim, as_index=False).agg(Sales_A=("Sales","sum"))
        comp_b = dfB.groupby(pivot_dim, as_index=False).agg(Sales_B=("Sales","sum"))
        comp = comp_a.merge(comp_b, on=pivot_dim, how="outer").fillna(0.0)
        comp["Difference"] = comp["Sales_A"] - comp["Sales_B"]
        comp["% Change"] = np.where(comp["Sales_B"] != 0, comp["Difference"] / comp["Sales_B"], np.nan)
        comp = comp.sort_values("Sales_A", ascending=False)
        show = rename_ab_columns(comp.copy(), a_lbl, b_lbl)
        sales_a_col = f"Sales ({a_lbl})"
        sales_b_col = f"Sales ({b_lbl})" if b_lbl else "Sales (Comparison)"
        show[sales_a_col] = show[sales_a_col].map(money)
        show[sales_b_col] = show[sales_b_col].map(money)
        show["Difference"] = show["Difference"].map(money)
        show["% Change"] = show["% Change"].map(pct_fmt)
        render_df(show[[pivot_dim, sales_a_col, sales_b_col, "Difference", "% Change"]], height=360)
    
        st.divider()
        st.subheader("Movers")
        a = dfA.groupby("SKU", as_index=False).agg(Sales_A=("Sales","sum"))
        b = dfB.groupby("SKU", as_index=False).agg(Sales_B=("Sales","sum"))
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
            if inc.empty: st.caption("None.")
            else: render_df(inc[["SKU", f"Sales ({a_lbl})", f"Sales ({b_lbl})", "Difference", "% Change"]], height=360)
        with y:
            st.markdown("**Top Declining**")
            if dec.empty: st.caption("None.")
            else: render_df(dec[["SKU", f"Sales ({a_lbl})", f"Sales ({b_lbl})", "Difference", "% Change"]], height=360)
    elif analysis_view == "Multi Month / Year Compare":
        st.subheader("Multi Compare")
        st.info("Multi Month / Year Compare is ready for selecting multiple months or years. The detailed KPI and table layout for 3+ period analysis is being kept separate so the 2-period compare stays stable.")
        pivot_dim = st.selectbox("Compare rows by", options=["Retailer","Vendor"], index=0, key="multi_compare_dim_v2")
        comp = dfA.groupby(pivot_dim, as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum"))
        comp["Sales"] = comp["Sales"].map(money)
        comp["Units"] = comp["Units"].map(lambda v: f"{float(v):,.0f}")
        render_df(comp, height=360)
    else:
        st.divider()
        st.subheader("Weekly Detail (Retailer/Vendor x Weeks)")
        weekly_hist = df_scope[df_scope["WeekEnd"] <= pA.end].copy()
        with st.expander("Advanced settings — Weekly Detail", expanded=False):
            wd1, wd2, wd3, wd4 = st.columns(4)
            with wd1:
                pivot_dim = st.selectbox("Pivot rows by", options=["Retailer","Vendor"], index=0, key="weekly_pivot_dim")
            with wd2:
                weekly_metric = st.selectbox("Metric", options=["Sales","Units"], index=0, key="weekly_metric")
            with wd3:
                weekly_weeks_to_show = st.selectbox("Weeks displayed", options=[4,8,12,26,52], index=2, key="weekly_weeks_to_show")
            month_periods = available_month_labels(weekly_hist)
            avg_options = ["4 weeks","8 weeks","12 weeks","26 weeks","52 weeks"] + [month_year_display(p) for p in month_periods]
            default_avg_index = avg_options.index("8 weeks") if "8 weeks" in avg_options else 0
            with wd4:
                weekly_avg_basis = st.selectbox("Average basis", options=avg_options, index=default_avg_index, key="weekly_avg_basis")
        d = weekly_hist.copy(); d = d[(d["Sales"] >= min_sales) | (d["Units"] >= min_units)].copy()
        if d.empty: st.caption("No rows match the current thresholds.")
        else:
            metric_col = "Sales" if weekly_metric == "Sales" else "Units"
            piv = build_weekly_detail_table_cached(
                d[[c for c in [pivot_dim, "WeekEnd", "Sales", "Units"] if c in d.columns]].copy(),
                pivot_dim=pivot_dim,
                metric_col=metric_col,
                weeks_to_show=int(weekly_weeks_to_show),
                avg_basis=weekly_avg_basis,
            )
            week_cols = [c for c in piv.columns if c not in [pivot_dim, "Δ vs prior week", "Average Value", "Current vs Avg"]]

            fmt_value = (lambda v: money(float(v))) if metric_col == "Sales" else (lambda v: f"{float(v):,.0f}")
            piv_disp = piv.copy()
            for c in week_cols + ["Average Value", "Δ vs prior week", "Current vs Avg"]:
                if c in piv_disp.columns:
                    piv_disp[c] = pd.to_numeric(piv_disp[c], errors="coerce").fillna(0.0)
                    piv_disp[c] = piv_disp[c].map(fmt_value)

            def _posneg_str(v):
                try:
                    x = float(str(v).replace('$','').replace(',',''))
                except Exception:
                    return ""
                if x > 0:
                    return "color:#2e7d32; font-weight:700;"
                if x < 0:
                    return "color:#c62828; font-weight:700;"
                return ""

            sty = piv_disp.style
            for c in ["Δ vs prior week", "Current vs Avg"]:
                if c in piv_disp.columns:
                    sty = sty.applymap(_posneg_str, subset=[c])
            st.dataframe(sty, use_container_width=True, hide_index=True, height=_table_height(piv_disp, max_px=650))

        # original standard view sections
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
        st.markdown("**New Item Tracker (track launches + placements over the next N weeks)**")
        track_weeks = st.slider("Tracking window (weeks)", min_value=4, max_value=13, value=8, step=1)
        def _track_rows() -> pd.DataFrame:
            rows = []
            horizon_end_days = 7 * track_weeks - 1
            if not first_ever.empty:
                for _, r in first_ever.iterrows():
                    sku = r["SKU"]
                    fw = pd.to_datetime(r["FirstWeek"], errors="coerce")
                    if pd.isna(fw): continue
                    end = fw + pd.Timedelta(days=horizon_end_days)
                    d = df_hist_for_new[(df_hist_for_new["SKU"]==sku) & (df_hist_for_new["WeekEnd"]>=fw) & (df_hist_for_new["WeekEnd"]<=end)].copy()
                    if d.empty: continue
                    wk = d.groupby("WeekEnd", as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum")).sort_values("WeekEnd")
                    total_sales = float(wk["Sales"].sum()); total_units = float(wk["Units"].sum())
                    last_sales = float(wk["Sales"].iloc[-1]); prev_sales = float(wk["Sales"].iloc[-2]) if len(wk)>=2 else 0.0
                    rows.append({"Type":"Launch","SKU":sku,"Start Week":fw.date().isoformat(),"Weeks Tracked":int(len(wk)),"Total Sales":total_sales,"Total Units":total_units,"Last Week Sales":last_sales,"WoW Sales Δ":(last_sales - prev_sales) if len(wk)>=2 else np.nan})
            if not placements.empty:
                for _, r in placements.iterrows():
                    sku = r["SKU"]; ret = r["Retailer"]; fw = pd.to_datetime(r["FirstWeek"], errors="coerce")
                    if pd.isna(fw): continue
                    end = fw + pd.Timedelta(days=horizon_end_days)
                    d = df_hist_for_new[(df_hist_for_new["SKU"]==sku) & (df_hist_for_new["Retailer"]==ret) & (df_hist_for_new["WeekEnd"]>=fw) & (df_hist_for_new["WeekEnd"]<=end)].copy()
                    if d.empty: continue
                    wk = d.groupby("WeekEnd", as_index=False).agg(Sales=("Sales","sum"), Units=("Units","sum")).sort_values("WeekEnd")
                    total_sales = float(wk["Sales"].sum()); total_units = float(wk["Units"].sum())
                    last_sales = float(wk["Sales"].iloc[-1]); prev_sales = float(wk["Sales"].iloc[-2]) if len(wk)>=2 else 0.0
                    rows.append({"Type":"Placement","SKU":sku,"Retailer":ret,"Start Week":fw.date().isoformat(),"Weeks Tracked":int(len(wk)),"Total Sales":total_sales,"Total Units":total_units,"Last Week Sales":last_sales,"WoW Sales Δ":(last_sales - prev_sales) if len(wk)>=2 else np.nan})
            if not rows: return pd.DataFrame(columns=["Type","SKU","Retailer","Start Week","Weeks Tracked","Total Sales","Total Units","Last Week Sales","WoW Sales Δ"])
            return pd.DataFrame(rows)
        tracker = _track_rows()
        if tracker.empty: st.caption("No new items to track in the selected period.")
        else:
            show = tracker.copy()
            for c in ["Total Sales","Last Week Sales","WoW Sales Δ"]: show[c] = show[c].map(lambda v: "" if pd.isna(v) else money(float(v)))
            show["Total Units"] = show["Total Units"].map(lambda v: f"{float(v):,.0f}")
            render_df(show.sort_values(["Type","Start Week","SKU"], ascending=[True, False, True]), height=320)
        st.subheader("Movers & Trend Leaders")
        if compare_mode == "None": st.info("Select a comparison mode to compute increasing/declining vs the compare period.")
        else:
            a = dfA.groupby("SKU", as_index=False).agg(Sales_A=("Sales","sum"), Units_A=("Units","sum"))
            b = dfB.groupby("SKU", as_index=False).agg(Sales_B=("Sales","sum"), Units_B=("Units","sum"))
            m = a.merge(b, on="SKU", how="outer").fillna(0.0)
            m["Sales (Current)"] = m["Sales_A"]; m["Sales (Compare)"] = m["Sales_B"]; m["Sales Δ"] = m["Sales_A"] - m["Sales_B"]; m["Δ %"] = np.where(m["Sales_B"] != 0, m["Sales Δ"] / m["Sales_B"], np.nan)
            m = m[(m["Sales_A"] >= min_sales) | (m["Sales_B"] >= min_sales) | (m["Units_A"] >= min_units) | (m["Units_B"] >= min_units)].copy()
            inc = m[m["Sales Δ"] > 0].sort_values("Sales Δ", ascending=False).head(10); dec = m[m["Sales Δ"] < 0].sort_values("Sales Δ", ascending=True).head(10)
            def _fmt_diff(val: float) -> str:
                color = "#2e7d32" if val > 0 else ("#c62828" if val < 0 else "var(--text-color)"); s = money(val); s = "+" + s if val > 0 else s; return f"<span style='color:{color}'>{s}</span>"
            def _fmt_pct(val: float) -> str:
                if pd.isna(val) or val == np.inf or val == -np.inf: return ""
                color = "#2e7d32" if val > 0 else ("#c62828" if val < 0 else "var(--text-color)"); return f"<span style='color:{color}'>({pct_fmt(val)})</span>"
            def _disp(df_in: pd.DataFrame) -> pd.DataFrame:
                if df_in.empty: return df_in
                out = df_in[["SKU","Sales (Current)","Sales (Compare)","Sales Δ","Δ %"]].copy(); out["Sales (Current)"] = out["Sales (Current)"].map(money); out["Sales (Compare)"] = out["Sales (Compare)"].map(money); out["Sales Δ"] = out["Sales Δ"].map(_fmt_diff); out["Δ %"] = out["Δ %"].map(_fmt_pct); return out
            inc_disp = _disp(inc); dec_disp = _disp(dec)
            mom = build_momentum(df_scope[df_scope["WeekEnd"] <= pA.end], "SKU", lookback_weeks=8)
            if not mom.empty:
                mom = mom[(mom["Sales (lookback)"] >= min_sales) | (mom["Units (lookback)"] >= min_units)].copy(); trend_leaders = mom.sort_values("Slope", ascending=False).head(10).copy(); trend_leaders_disp = trend_leaders[["SKU","Trend","Slope","Weeks Up","Weeks Down","Sales (lookback)"]].copy(); trend_leaders_disp["Sales (lookback)"] = trend_leaders_disp["Sales (lookback)"].map(money); trend_leaders_disp["Slope"] = trend_leaders_disp["Slope"].map(lambda v: f"{v:,.2f}")
            else: trend_leaders_disp = pd.DataFrame(columns=["SKU","Trend","Slope","Weeks Up","Weeks Down","Sales (lookback)"])
            a,b,c = st.columns(3)
            with a:
                st.markdown("**Top Increasing**"); st.markdown(inc_disp.to_html(escape=False, index=False, classes='report-table'), unsafe_allow_html=True) if not inc_disp.empty else st.caption("None.")
            with b:
                st.markdown("**Top Declining**"); st.markdown(dec_disp.to_html(escape=False, index=False, classes='report-table'), unsafe_allow_html=True) if not dec_disp.empty else st.caption("None.")
            with c:
                st.markdown("**Trend Leaders (slope over last 8 weeks)**"); render_df(trend_leaders_disp, height=320)
    
        st.divider()
        st.header("Strategic Intelligence")
    
        # A) Contribution Tree
        st.subheader("1) Contribution Tree (Where did change come from?)")
        if compare_mode == "None":
            st.info("Select a comparison mode to use the contribution tree.")
        else:
            sales_a_col = f"Sales ({a_lbl})"
            sales_b_col = f"Sales ({b_lbl})" if b_lbl else "Sales (Comparison)"
            level1_mode = st.selectbox("Level 1 view", options=["Retailer","Vendor"], index=0, key="strategic_lvl1_mode")

            if level1_mode == "Retailer":
                lvl1 = drivers(dfA, dfB, "Retailer").sort_values("Sales_Δ", ascending=False)
                st.markdown("**Level 1 — Retailers**")
                lvl1_disp = lvl1[["Retailer","Sales_A","Sales_B","Sales_Δ","Contribution_%"]].copy()
                lvl1_disp["Sales_A"] = lvl1_disp["Sales_A"].map(money)
                lvl1_disp["Sales_B"] = lvl1_disp["Sales_B"].map(money)
                lvl1_disp["Sales_Δ"] = lvl1_disp["Sales_Δ"].map(money)
                lvl1_disp["Contribution_%"] = lvl1_disp["Contribution_%"].map(pct_fmt)
                lvl1_disp = rename_ab_columns(lvl1_disp, a_lbl, b_lbl)
                render_df(lvl1_disp[["Retailer", sales_a_col, sales_b_col, "Sales_Δ", "Contribution_%"]], height=260)

                pick_r = st.selectbox("Drill into Retailer", options=["(none)"] + sorted(lvl1["Retailer"].astype(str).tolist()), index=0, key="strategic_pick_retailer")
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

                    pick_v = st.selectbox("Drill into Vendor", options=["(none)"] + sorted(lvl2["Vendor"].astype(str).tolist()), index=0, key="strategic_pick_vendor")
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
            else:
                lvl1 = drivers(dfA, dfB, "Vendor").sort_values("Sales_Δ", ascending=False)
                st.markdown("**Level 1 — Vendors**")
                lvl1_disp = lvl1[["Vendor","Sales_A","Sales_B","Sales_Δ","Contribution_%"]].copy()
                lvl1_disp["Sales_A"] = lvl1_disp["Sales_A"].map(money)
                lvl1_disp["Sales_B"] = lvl1_disp["Sales_B"].map(money)
                lvl1_disp["Sales_Δ"] = lvl1_disp["Sales_Δ"].map(money)
                lvl1_disp["Contribution_%"] = lvl1_disp["Contribution_%"].map(pct_fmt)
                lvl1_disp = rename_ab_columns(lvl1_disp, a_lbl, b_lbl)
                render_df(lvl1_disp[["Vendor", sales_a_col, sales_b_col, "Sales_Δ", "Contribution_%"]], height=260)

                pick_v = st.selectbox("Drill into Vendor", options=["(none)"] + sorted(lvl1["Vendor"].astype(str).tolist()), index=0, key="strategic_pick_vendor_lvl1")
                if pick_v != "(none)":
                    dfA_v = dfA[dfA["Vendor"] == pick_v].copy()
                    dfB_v = dfB[dfB["Vendor"] == pick_v].copy()

                    lvl2 = drivers(dfA_v, dfB_v, "Retailer").sort_values("Sales_Δ", ascending=False)
                    st.markdown(f"**Level 2 — Retailers inside {pick_v}**")
                    lvl2_disp = lvl2[["Retailer","Sales_A","Sales_B","Sales_Δ","Contribution_%"]].copy()
                    lvl2_disp["Sales_A"] = lvl2_disp["Sales_A"].map(money)
                    lvl2_disp["Sales_B"] = lvl2_disp["Sales_B"].map(money)
                    lvl2_disp["Sales_Δ"] = lvl2_disp["Sales_Δ"].map(money)
                    lvl2_disp["Contribution_%"] = lvl2_disp["Contribution_%"].map(pct_fmt)
                    lvl2_disp = rename_ab_columns(lvl2_disp, a_lbl, b_lbl)
                    render_df(lvl2_disp[["Retailer", sales_a_col, sales_b_col, "Sales_Δ", "Contribution_%"]], height=240)

                    pick_r = st.selectbox("Drill into Retailer", options=["(none)"] + sorted(lvl2["Retailer"].astype(str).tolist()), index=0, key="strategic_pick_retailer_lvl2")
                    if pick_r != "(none)":
                        dfA_r = dfA_v[dfA_v["Retailer"] == pick_r].copy()
                        dfB_r = dfB_v[dfB_v["Retailer"] == pick_r].copy()
                        lvl3 = drivers(dfA_r, dfB_r, "SKU").sort_values("Sales_Δ", ascending=False)
                        st.markdown(f"**Level 3 — SKUs inside {pick_v} → {pick_r}**")
                        lvl3_disp = lvl3[["SKU","Sales_A","Sales_B","Sales_Δ","Contribution_%"]].copy()
                        lvl3_disp["Sales_A"] = lvl3_disp["Sales_A"].map(money)
                        lvl3_disp["Sales_B"] = lvl3_disp["Sales_B"].map(money)
                        lvl3_disp["Sales_Δ"] = lvl3_disp["Sales_Δ"].map(money)
                        lvl3_disp["Contribution_%"] = lvl3_disp["Contribution_%"].map(pct_fmt)
                        lvl3_disp = rename_ab_columns(lvl3_disp, a_lbl, b_lbl)
                        render_df(lvl3_disp[["SKU", sales_a_col, sales_b_col, "Sales_Δ", "Contribution_%"]], height=240)
        st.subheader("2) SKU Lifecycle (Launch → Growth → Mature → Decline → Dormant)")
        life_df_src = df_hist_for_new if show_full_history_lifecycle else df_scope
        lc1, lc2, lc3 = st.columns([1.25, 1, 1])
        with lc1:
            lifecycle_scope = st.selectbox("Lifecycle scope", options=["SKU (All Retailers)", "SKU by Retailer"], index=0, key="lifecycle_scope")
        with lc2:
            lifecycle_lookback = st.selectbox("Lookback weeks", options=[4, 6, 8, 12, 26, 52], index=2, key="lifecycle_lookback")
        stage_options = ["Launch", "Growth", "Mature", "Decline", "Dormant", "Inactive 12+ Weeks"]
        with lc3:
            lifecycle_stage_filter = st.multiselect("Stages", options=stage_options, default=stage_options, key="lifecycle_stage_filter")

        life = lifecycle_table(life_df_src, pA, lookback_weeks=int(lifecycle_lookback), scope=lifecycle_scope)
        if life.empty:
            st.caption("Not enough data to compute lifecycle.")
        else:
            life_show = life.copy()
            if lifecycle_stage_filter:
                life_show = life_show[life_show["Stage"].isin(lifecycle_stage_filter)].copy()
            if min_sales > 0:
                life_show = life_show[life_show["Sales (lookback)"] >= min_sales].copy()

            stage_counts = life_show["Stage"].value_counts().reindex(stage_options, fill_value=0).reset_index()
            stage_counts.columns = ["Stage", "Count"]
            st.markdown("**Stage Summary**")
            render_df(stage_counts, height=220)

            st.markdown("**Lifecycle Detail**")
            if life_show.empty:
                st.caption("No lifecycle rows match the current stage/threshold filters.")
            else:
                life_show["Sales (lookback)"] = life_show["Sales (lookback)"].map(money)
                life_show["Units (lookback)"] = life_show["Units (lookback)"].map(lambda v: f"{v:,.0f}")
                life_show["Last Week Sales"] = life_show["Last Week Sales"].map(money)
                life_show["WoW Sales Δ"] = life_show["WoW Sales Δ"].map(lambda v: "" if pd.isna(v) else money(v))
                cols = ["Retailer", "SKU", "Stage", "Trend", "Sales (lookback)", "Units (lookback)", "Last Week Sales", "WoW Sales Δ", "Weeks Up", "Weeks Down", "Weeks With Sales"]
                cols = [c for c in cols if c in life_show.columns]
                render_df(life_show[cols].head(200), height=520)

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
