from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_WEEKLIKE_COLUMNS = [
    "Week",
    "Week Ending",
    "Week Ending Date",
    "Week End",
    "Week End Date",
    "week",
    "week ending",
    "week ending date",
    "week end",
    "week end date",
    "week_ending",
    "week_end",
    "Date",
    "date",
    "Reporting Week",
    "reporting week",
    "Week Label",
    "week label",
]


def _parse_month_from_text_series(s: pd.Series) -> pd.Series:
    txt = s.astype(str).str.strip().str.lower()

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

    out = pd.Series(np.nan, index=s.index, dtype="float64")

    for key, val in month_map.items():
        mask = txt.str.contains(rf"\b{re.escape(key)}\b", regex=True, na=False)
        out = out.where(~mask, val)

    if out.notna().any():
        return out

    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().any():
        return dt.dt.month.astype("float64")

    # catches strings like 1/5, 01-09, 1/5/2026, etc.
    mmdd = txt.str.extract(r"(?P<m>\d{1,2})[/-](?P<d>\d{1,2})(?:[/-](?P<y>\d{2,4}))?")
    if not mmdd.empty:
        m = pd.to_numeric(mmdd["m"], errors="coerce")
        if m.notna().any():
            return m.astype("float64")

    # catches ranges like 1-5 / 1-9 or 1/5 - 1/9
    range_pat = txt.str.extract(r"(?P<m1>\d{1,2})\s*[/-]\s*\d{1,2}\s*(?:to|-|/)\s*(?P<m2>\d{1,2})\s*[/-]\s*\d{1,2}")
    if not range_pat.empty:
        m1 = pd.to_numeric(range_pat["m1"], errors="coerce")
        if m1.notna().any():
            return m1.astype("float64")

    return out


def _parse_year_from_text_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().any():
        return dt.dt.year.astype("float64")

    txt = s.astype(str).str.strip()
    year_part = txt.str.extract(r"(?P<y>(?:19|20)\d{2})")
    y = pd.to_numeric(year_part["y"], errors="coerce")
    return y.astype("float64")


def _find_best_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        hit = lower_map.get(str(cand).strip().lower())
        if hit is not None:
            return hit
    return None


def add_quarter_columns(
    df: pd.DataFrame,
    week_column: str | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Adds:
      - MonthNum
      - QuarterNum
      - Quarter
      - Year
      - YearQuarter

    This is meant to run upstream on the normalized weekly master dataframe
    before filters and tabs are built.
    """
    out = df if inplace else df.copy()

    if week_column is None:
        week_column = _find_best_column(out, DEFAULT_WEEKLIKE_COLUMNS)

    month_num = pd.Series(np.nan, index=out.index, dtype="float64")
    year_num = pd.Series(np.nan, index=out.index, dtype="float64")

    # 1) explicit quarter already exists
    if "Quarter" in out.columns:
        qtxt = out["Quarter"].astype(str).str.upper().str.replace(" ", "", regex=False)
        qnum = qtxt.replace({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "1": 1, "2": 2, "3": 3, "4": 4})
        qnum = pd.to_numeric(qnum, errors="coerce")
        if qnum.notna().any():
            out["QuarterNum"] = qnum
            out["Quarter"] = qnum.map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
        else:
            out["QuarterNum"] = np.nan
            out["Quarter"] = np.nan
    else:
        out["QuarterNum"] = np.nan
        out["Quarter"] = np.nan

    # 2) explicit month columns if present
    if "MonthNum" in out.columns:
        month_num = pd.to_numeric(out["MonthNum"], errors="coerce")
    else:
        month_col = _find_best_column(out, ["Month", "Month Name", "month", "month name", "MonthNum", "monthnum"])
        if month_col is not None:
            raw = out[month_col]
            if pd.api.types.is_numeric_dtype(raw):
                month_num = pd.to_numeric(raw, errors="coerce")
            else:
                month_num = _parse_month_from_text_series(raw)

    # 3) week/date column fallback
    if month_num.notna().sum() == 0 and week_column is not None and week_column in out.columns:
        month_num = _parse_month_from_text_series(out[week_column])

    # 4) year
    if "Year" in out.columns:
        year_num = pd.to_numeric(out["Year"], errors="coerce")
    elif week_column is not None and week_column in out.columns:
        year_num = _parse_year_from_text_series(out[week_column])

    # 5) derive quarter from month if needed
    q_from_month = (((month_num - 1) // 3) + 1).where(month_num.notna())
    q_from_month = pd.to_numeric(q_from_month, errors="coerce")

    out["MonthNum"] = month_num
    out["QuarterNum"] = out["QuarterNum"].where(out["QuarterNum"].notna(), q_from_month)
    out["Quarter"] = out["QuarterNum"].map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
    out["Year"] = year_num

    out["YearQuarter"] = np.where(
        out["Year"].notna() & out["Quarter"].notna(),
        out["Year"].astype("Int64").astype(str) + " " + out["Quarter"].astype(str),
        np.nan,
    )

    return out
