import pandas as pd
from pathlib import Path

def load_vendor_map(xlsx_path: str):
    """Load an .xlsx mapping of Model/SKU -> Vendor.
    Accepts columns: model, sku, vendor (any case). Extra columns ignored.
    Returns a dict with normalized keys mapped to vendor string.
    """
    p = Path(xlsx_path)
    if not p.exists():
        raise FileNotFoundError(f"Map not found: {xlsx_path}")
    # engine='openpyxl' ensures .xlsx is handled
    df = pd.read_excel(p, engine='openpyxl')
    cols = {c.lower(): c for c in df.columns}
    if 'vendor' not in cols:
        raise ValueError("The XLSX must include a 'vendor' column.")
    vendor_col = cols['vendor']
    keys = []
    for candidate in ['model', 'sku']:
        if candidate in cols:
            keys.append(cols[candidate])
    if not keys:
        # try to infer other likely names
        for c in df.columns:
            lc = c.lower().strip()
            if lc in ('model #','model number','store sku #','store sku','internet #','internet number'):
                keys.append(c)
        if not keys:
            raise ValueError("Provide at least one key column: model or sku.")
    mapping = {}
    for _, row in df.iterrows():
        vendor = str(row[vendor_col]).strip() if pd.notna(row[vendor_col]) else ''
        if not vendor:
            continue
        for kcol in keys:
            if kcol in row and pd.notna(row[kcol]):
                raw = str(row[kcol]).strip()
                if raw:
                    norm = normalize_key(raw)
                    mapping[norm] = vendor
    return mapping

def normalize_key(s: str) -> str:
    # Keep letters/numbers/-/_/./slash; remove spaces, make upper
    import re
    s = s.strip().upper()
    s = s.replace(' ', '')
    s = re.sub(r"[^A-Z0-9\-_/\.]+", "", s)
    return s
