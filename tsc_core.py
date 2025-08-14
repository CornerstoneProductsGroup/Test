
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pdfplumber, re, os
from rapidfuzz import fuzz, process
from pathlib import Path
from vendor_map import normalize_key
from pypdf import PdfReader, PdfWriter

# Require at least 2 chars, allow -, /, _, .
SKU_PAT = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/_.]{1,29}")

@dataclass
class Candidate:
    value: str
    kind: str
    anchor_text: str
    anchor_dist: float
    bbox: Tuple[float, float, float, float]

def _page_words(page):
    return page.extract_words(extra_attrs=['size']) or []

def _seq_match(words_lower, i, seq_tokens):
    w0 = words_lower[i][2]
    baseline = (w0.get('top',0)+w0.get('bottom',0))/2.0
    cur_x = w0.get('x1',0)
    last = w0
    for need in seq_tokens[1:]:
        nxt = None
        for _, txt2, w2 in words_lower:
            if abs(((w2.get('top',0)+w2.get('bottom',0))/2.0)-baseline) <= 4.0 and w2.get('x0',0) >= cur_x - 2 and (w2.get('x0',0)-cur_x) <= 160:
                if txt2 == need:
                    nxt = w2
                    break
        if nxt is None:
            return None
        cur_x = nxt.get('x1',0)
        last = nxt
    return last

def _same_baseline(w1, w2, tol=4.0):
    return abs(((w1.get('top',0)+w1.get('bottom',0))/2.0) - ((w2.get('top',0)+w2.get('bottom',0))/2.0)) <= tol

def _find_vendor_pn_column_band(words, x_pad_left=14, x_pad_right=44, below_px=90):
    \"\"\"Find the 'Vendor PN' header and return a vertical band below that column.\"\"\"
    lower = [(i, (w.get('text','') or '').strip().lower(), w) for i,w in enumerate(words)]
    for i, txt, w0 in lower:
        if txt in ('vendor', 'vendorpn', 'vendor pn'):
            if txt in ('vendorpn', 'vendor pn'):
                x0 = w0.get('x0',0) - x_pad_left
                x1 = w0.get('x1',0) + x_pad_right
                anchor_bottom = w0.get('bottom',0)
                band_y1 = anchor_bottom + below_px
                return (x0, x1, anchor_bottom, band_y1, "Vendor PN")
            nxt = None
            for _, t2, w2 in lower:
                if _same_baseline(w0, w2) and w2.get('x0',0) >= w0.get('x1',0) - 2 and (w2.get('x0',0) - w0.get('x1',0)) <= 160:
                    if t2 == 'pn':
                        nxt = w2
                        break
            if nxt is not None:
                x0 = min(w0.get('x0',0), nxt.get('x0',0)) - x_pad_left
                x1 = max(w0.get('x1',0), nxt.get('x1',0)) + x_pad_right
                anchor_bottom = max(w0.get('bottom',0), nxt.get('bottom',0))
                band_y1 = anchor_bottom + below_px
                return (x0, x1, anchor_bottom, band_y1, "Vendor PN")
    return None

def _find_vendor_itemnum_band(words, x_pad_left=18, x_pad_right=48, below_px=100):
    \"\"\"Looks for 'Vendors (Sellers) Item Number:' (and variants) then captures a tight band below it.\"\"\"
    lower = [(i, (w.get('text','') or '').strip().lower(), w) for i, w in enumerate(words)]
    variants = [
        ["vendors", "(sellers)", "item", "number", ":"],
        ["vendors", "sellers", "item", "number", ":"],
        ["vendors", "(sellers)", "item", "number"],
        ["vendors", "sellers", "item", "number"],
        ["vendors", "(sellers)", "item", "no", ":"],
        ["vendors", "sellers", "item", "no", ":"],
        ["vendors", "(sellers)", "item", "numner", ":"],
        ["vendors", "sellers", "item", "numner", ":"],
        ["vendors(sellers)itemnumber"],
        ["vendors(sellers)itemnumner"],
    ]
    for i, t, w0 in lower:
        if t == "vendors":
            for seq in variants:
                last = _seq_match(lower, i, seq)
                if last is not None:
                    x0 = min(w0.get('x0',0), last.get('x0',0)) - x_pad_left
                    x1 = max(w0.get('x1',0), last.get('x1',0)) + x_pad_right
                    anchor_bottom = max(w0.get('bottom',0), last.get('bottom',0))
                    band_y1 = anchor_bottom + below_px
                    return (x0, x1, anchor_bottom, band_y1, "Vendors (Sellers) Item Number")
    return None

_TEXT_FALLBACK_PAT = re.compile(
    r"Vendors?\\s*\\(Sellers?\\)\\s*Item\\s+Num(?:be|ne)r\\s*:?\\s*([A-Za-z0-9][A-Za-z0-9\\-/_.]{1,29})",
    re.IGNORECASE
)

def _regex_text_model(page) -> Optional[str]:
    try:
        t = page.extract_text() or ""
    except Exception:
        t = ""
    m = _TEXT_FALLBACK_PAT.search(t.replace('\\n', ' '))
    if m:
        return m.group(1)
    return None

def _extract_sku_below(words) -> Optional[Candidate]:
    band = _find_vendor_itemnum_band(words)
    label_used = None
    if not band:
        band = _find_vendor_pn_column_band(words)
        label_used = "Vendor PN"
    else:
        label_used = "Vendors (Sellers) Item Number"

    if band:
        bx0, bx1, ay, by, _lab = band
        below = [w for w in words if ay - 0.5 <= w.get('top',0) <= by and bx0 <= w.get('x0',0) and w.get('x1',0) <= bx1]
        below.sort(key=lambda w: (w.get('top',0.0), w.get('x0',0.0)))
        if not below:
            return None
        first_top = below[0].get('top', 0.0)
        same_line = [w for w in below if abs(w.get('top',0.0) - first_top) <= 2.5]
        for w in same_line:
            txt = (w.get('text') or '').strip()
            if not txt:
                continue
            low = txt.lower()
            if low in ('vendors', '(sellers)', 'sellers', 'item', 'number', 'numner', 'no', ':', 'vendor', 'pn', 'vendor pn'):
                continue
            # need at least one digit to avoid picking Description words like 'Niban'/'Walnut'
            if len(txt) < 2 or not any(ch.isdigit() for ch in txt):
                continue
            m = SKU_PAT.fullmatch(txt) or SKU_PAT.search(txt)
            if m:
                val = m.group(0)
                return Candidate(value=val, kind='sku', anchor_text=label_used, anchor_dist=0.0,
                                 bbox=(w.get('x0',0.0), w.get('top',0.0), w.get('x1',0.0), w.get('bottom',0.0)))
    return None

def extract_candidates_from_page(page) -> List[Candidate]:
    words = _page_words(page)
    c = _extract_sku_below(words)
    if c:
        return [c]
    val = _regex_text_model(page)
    if val:
        return [Candidate(value=val, kind='sku', anchor_text='Vendors (Sellers) Item Number (text)', anchor_dist=0.0, bbox=(0,0,0,0))]
    return []

def score_candidate(val: str, vendor_map: Dict[str,str], idx_keys) -> float:
    pattern_score = 0.9
    dict_score = 0.0
    nval = normalize_key(val)
    if nval in vendor_map:
        dict_score = 1.0
    else:
        best = process.extractOne(nval, idx_keys, scorer=fuzz.token_set_ratio)
        if best and best[1] >= 92:
            dict_score = 0.85
    return 0.55*dict_score + 0.45*pattern_score

def choose_best_vendor(cands: List[Candidate], vendor_map: Dict[str,str], threshold=0.88):
    if not cands:
        return None, None, 0.0, None
    idx_keys = list(vendor_map.keys())
    best = None
    best_score = -1.0
    best_vendor = None
    for c in cands:
        s = score_candidate(c.value, vendor_map, idx_keys)
        if s > best_score:
            best_score = s
            nkey = normalize_key(c.value)
            v = vendor_map.get(nkey)
            if not v:
                match = process.extractOne(nkey, idx_keys, scorer=fuzz.token_set_ratio)
                if match and match[1] >= 92:
                    v = vendor_map.get(match[0])
            best = c
            best_vendor = v
    if best_score >= threshold and best_vendor:
        return best_vendor, best, best_score, None
    else:
        return None, best, best_score, 'LOW_CONFIDENCE'

def split_pdf_to_vendors(pdf_path: str, out_dir: str, vendor_map: Dict[str,str], threshold=0.88):
    os.makedirs(out_dir, exist_ok=True)
    report_rows = []
    review_rows = []
    page_exports = {}

    rdr = PdfReader(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            cands = extract_candidates_from_page(page)
            vendor, best, score, flag = choose_best_vendor(cands, vendor_map, threshold=threshold)
            report_rows.append({
                'page': i+1,
                'vendor': vendor or '',
                'score': round(score, 3),
                'best_value': best.value if best else '',
                'best_kind': best.kind if best else '',
                'anchor': best.anchor_text if best else '',
                'flag': flag or ''
            })
            if vendor and not flag:
                page_exports.setdefault(vendor, []).append(i)
            else:
                review_rows.append({
                    'page': i+1,
                    'score': round(score,3),
                    'best_value': best.value if best else '',
                    'best_kind': best.kind if best else '',
                    'anchor': best.anchor_text if best else '',
                })

    out_pdfs = {}
    master_name = os.environ.get('HD_MASTER_NAME', 'Batch')
    for vendor, pages in page_exports.items():
        w = PdfWriter()
        for p in pages:
            w.add_page(rdr.pages[p])
        vend_dir = Path(out_dir)/vendor
        vend_dir.mkdir(parents=True, exist_ok=True)
        out_path = vend_dir / f"{master_name} {vendor}.pdf"
        with open(out_path, 'wb') as f:
            w.write(f)
        out_pdfs[vendor] = str(out_path)

    return report_rows, review_rows, out_pdfs
